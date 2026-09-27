#!/usr/bin/env python
"""Generate the bimanual I2RT YAM assets used by ``YamBimanualPoseFabric``.

Two YAM arms (MuJoCo Menagerie ``i2rt_yam/yam.xml``) on one table, as one 12-joint fabric so
that self-collision *and* arm-arm collision are body-sphere pairs of the same fabric.

Produces, inside ``src/fabrics_sim``:

* ``models/robots/urdf/yam_bimanual/yam_bimanual.urdf``  fabric-ready URDF (see below)
* ``fabric_params/yam_bimanual_pose_params.yaml``        fabric parameters incl. spheres
* ``worlds/yam_bimanual_table.yaml``                     table + the two static arm bases

Frames and conventions (must match the Newton scene in ``tasks/lego_disassembly/task.py``):

* Root link ``world`` is the simulator's world frame. Arm ``L`` is mounted at (0, +0.22, 0.40),
  arm ``R`` at (0, -0.22, 0.40), both facing +x; the table top is the z = 0.40 plane.
* Joint order: ``L_joint1..6`` then ``R_joint1..6`` (the order the task's planner uses).
* MJCF bodies become links ``<side>_<body>``; each MJCF body offset becomes the joint origin.
* The two finger slide joints become *fixed* at the open width ``FINGER_OPEN`` (fabric kinematics
  support revolute + fixed only). The gripper is commanded in the simulator.
* ``<side>_tcp``: fixed frame in ``<side>_link_6`` at ``TCP`` (centre between the fingertip pads,
  same as the task's TCP), with ``link_6``'s orientation. Six axis-point frames at +-0.25 m give the
  pose attractor its orientation handle.
* Collision spheres: fitted along the MJCF collision capsules/boxes (the geometry Newton collides),
  radius = capsule radius + ``SPHERE_MARGIN``, spacing <= radius. Frames ``<link>_sphere<i>``.

Collision pairs:

* self, per arm: wrist/gripper/fingers vs link_1/link_2, link_5 vs link_1/link_2, link_4 vs link_1.
* cross (the point of the bimanual fabric): every link of L vs every link of R, except
  finger vs finger. In the LEGO hand-off the two sets of fingertips grip the two bricks 5 mm apart
  on purpose, so finger-finger repulsion would forbid the task.
* world (table + bases): every sphere except the fingers (``finger_world_collision``), which have to
  reach bricks lying on the table. Masked per sphere via ``world_collision_mask``.

Usage (env_fabrics active):
    python tools/make_yam_fabric_assets.py --mjcf <repo>/src/asd/assets/i2rt_yam/yam.xml
"""
from __future__ import annotations

import argparse
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "src", "fabrics_sim")
DEFAULT_MJCF = os.path.normpath(os.path.join(HERE, "..", "..", "..", "src", "asd", "assets", "i2rt_yam", "yam.xml"))

ROBOT = "yam_bimanual"
BASES = {"L": (0.0, 0.22, 0.40), "R": (0.0, -0.22, 0.40)}
TABLE_Z = 0.40
TCP = (0.0, -0.044, 0.1435)   # in link_6; tips of the pads reach z = 0.1465
FINGER_OPEN = 0.030           # slide-joint travel of the left finger (the right one mirrors it)
AXIS_POINT_OFFSET = 0.25
SPHERE_MARGIN = 0.004
MIN_RADIUS = 0.008  # the 3 mm finger ribs are covered by fewer, fatter spheres
DEDUPE = 1.5       # drop a sphere whose centre lies within this many radii of one kept from an earlier geom
SPACING = 1.3       # centre spacing along a capsule, in radii (gap sag < 0.2 r)
HOME = [0.0, 1.047, 1.047, 0.0, 0.0, 0.0]  # Menagerie "home" keyframe

ARM_LINKS = ["link_1", "link_2", "link_3", "link_4", "link_5", "link_6"]
FINGER_LINKS = ["link_left_finger", "lf_rot", "lf_down", "link_right_finger", "rf_rot", "rf_down"]
SELF_PAIRS = [(a, b) for a in ["link_6"] + FINGER_LINKS for b in ["link_1", "link_2"]] + \
             [("link_5", "link_1"), ("link_5", "link_2"), ("link_4", "link_1")]

INERTIAL_XML = ('<inertial><mass value="0.01"/>'
                '<inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/></inertial>')


def quat_wxyz_to_matrix(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_to_rpy(R):
    """URDF rpy (R = Rz(y) Ry(p) Rx(r))."""
    p = math.asin(-max(-1.0, min(1.0, R[2, 0])))
    if abs(math.cos(p)) > 1e-9:
        r = math.atan2(R[2, 1], R[2, 2]); y = math.atan2(R[1, 0], R[0, 0])
    else:
        r = math.atan2(-R[1, 2], R[1, 1]); y = 0.0
    return r, p, y


def fvec(s, n=3, default=None):
    if s is None:
        return list(default)
    v = [float(x) for x in s.split()]
    return v + [0.0] * (n - len(v))


def sub(parent, tag, **attrib):
    e = ET.SubElement(parent, tag)
    for k, v in attrib.items():
        e.set(k, v)
    return e


def fmt(v):
    return " ".join(f"{x:.6f}" for x in v)


def add_link(robot, name):
    link = sub(robot, "link", name=name)
    link.append(ET.fromstring(INERTIAL_XML))
    return link


def add_joint(robot, name, jtype, parent, child, xyz, R, axis=None, limit=None):
    j = sub(robot, "joint", name=name, type=jtype)
    sub(j, "origin", xyz=fmt(xyz), rpy=fmt(matrix_to_rpy(R)))
    sub(j, "parent", link=parent)
    sub(j, "child", link=child)
    if axis is not None:
        sub(j, "axis", xyz=fmt(axis))
    if limit is not None:
        sub(j, "limit", lower=f"{limit[0]:.6f}", upper=f"{limit[1]:.6f}", effort="30", velocity="10")
    return j


def geom_spheres(g):
    """Sphere centres (in the body frame) and radius covering one MJCF collision geom."""
    t = g.get("type", "capsule")
    size = fvec(g.get("size"), 3)
    pos = np.array(fvec(g.get("pos"), 3, (0, 0, 0)))
    R = quat_wxyz_to_matrix(fvec(g.get("quat"), 4, (1, 0, 0, 0)))
    if t == "capsule":
        r, h = size[0], size[1]
        axis = R[:, 2]
    elif t == "box":  # finger pads: a line of spheres along the long axis, radius = the larger half-thickness
        he = np.array(size)
        k = int(np.argmax(he))
        r, h, axis = float(np.sort(he)[1]), he[k], R[:, k]
    else:
        return [], 0.0
    r = max(r, MIN_RADIUS)
    n = max(1, int(math.ceil(2 * h / (SPACING * r))) + 1)
    return [pos + axis * s for s in np.linspace(-h, h, n)], r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", default=DEFAULT_MJCF, help="Menagerie i2rt_yam/yam.xml")
    ap.add_argument("--finger-world-collision", action="store_true",
                    help="let the finger spheres collide with the table too (off: they must reach bricks on it)")
    args = ap.parse_args()

    mj = ET.parse(args.mjcf).getroot()
    arm = mj.find("worldbody/body[@name='arm']")

    robot = ET.Element("robot", name=ROBOT)
    add_link(robot, "world")
    sphere_frames, sphere_radii, world_mask = [], [], []
    link_of_sphere = []

    def walk(body, side, parent_link):
        """Emit body's link + its joint to parent_link, spheres, then recurse."""
        name = body.get("name")
        link = f"{side}_{name}"
        add_link(robot, link)
        xyz = np.array(fvec(body.get("pos"), 3, (0, 0, 0)))
        R = quat_wxyz_to_matrix(fvec(body.get("quat"), 4, (1, 0, 0, 0)))
        j = body.find("joint")
        jtype = None if j is None else j.get("type", "slide" if j.get("class") == "finger" else "hinge")
        if jtype == "hinge":
            lo, hi = fvec(j.get("range"), 2)
            add_joint(robot, f"{side}_{j.get('name')}", "revolute", parent_link, link, xyz, R,
                      axis=fvec(j.get("axis"), 3, (0, 0, 1)), limit=(lo, hi))
        elif jtype == "slide":
            q = FINGER_OPEN if j.get("name") == "left_finger" else -FINGER_OPEN
            axis = np.array(fvec(j.get("axis"), 3, (0, 0, 1)))
            add_joint(robot, f"{side}_{j.get('name')}_fixed", "fixed", parent_link, link, xyz + R @ (axis * q), R)
        else:
            add_joint(robot, f"{side}_{name}_fixed", "fixed", parent_link, link, xyz, R)
        k = 0
        body_spheres = []  # (centre, r), deduplicated: overlapping geoms (finger ribs, pads) share spheres
        for g in body.findall("geom"):
            if g.get("class") != "collision":  # visual meshes and the 0.6 mm pad marker spheres
                continue
            centres, r = geom_spheres(g)
            prev = list(body_spheres)  # dedupe against earlier geoms only, never within one capsule
            body_spheres += [(c, r) for c in centres if all(np.linalg.norm(c - c2) > DEDUPE * min(r, r2) for c2, r2 in prev)]
        for c, r in body_spheres:
            if True:
                fname = f"{link}_sphere{k}"
                add_link(robot, fname)
                add_joint(robot, fname + "_joint", "fixed", link, fname, c, np.eye(3))
                sphere_frames.append(fname); sphere_radii.append(round(r + SPHERE_MARGIN, 4))
                is_finger = name in FINGER_LINKS
                world_mask.append(1 if (args.finger_world_collision or not is_finger) and name != "link_1" else 0)
                link_of_sphere.append((side, name))
                k += 1
        if name == "link_6":
            add_link(robot, f"{side}_tcp")
            add_joint(robot, f"{side}_tcp_joint", "fixed", link, f"{side}_tcp", TCP, np.eye(3))
            a = AXIS_POINT_OFFSET
            for sfx, v in (("x", (a, 0, 0)), ("x_neg", (-a, 0, 0)), ("y", (0, a, 0)), ("y_neg", (0, -a, 0)),
                           ("z", (0, 0, a)), ("z_neg", (0, 0, -a))):
                add_link(robot, f"{side}_tcp_{sfx}")
                add_joint(robot, f"{side}_tcp_{sfx}_joint", "fixed", f"{side}_tcp", f"{side}_tcp_{sfx}", v, np.eye(3))
        for child in body.findall("body"):
            walk(child, side, link)

    for side in ("L", "R"):
        base = f"{side}_base"
        add_link(robot, base)
        add_joint(robot, f"{side}_base_fixed", "fixed", "world", base, BASES[side], np.eye(3))
        for child in arm.findall("body"):
            walk(child, side, base)

    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    out_dir = os.path.join(PKG, "models", "robots", "urdf", ROBOT)
    os.makedirs(out_dir, exist_ok=True)
    out_urdf = os.path.join(out_dir, f"{ROBOT}.urdf")
    tree.write(out_urdf, xml_declaration=True, encoding="utf-8")
    print("wrote", out_urdf, f"({len(sphere_frames)} spheres)")

    # collision pairs as explicit sphere pairs (one-directional, like the Kuka/Franka convention)
    pairs = []
    idx = {}
    for i, (side, name) in enumerate(link_of_sphere):
        idx.setdefault((side, name), []).append(i)
    for side in ("L", "R"):
        for a, b in SELF_PAIRS:
            for i in idx.get((side, a), []):
                for j in idx.get((side, b), []):
                    pairs.append([sphere_frames[i], sphere_frames[j]])
    for (sa, na), ia in idx.items():
        if sa != "L":
            continue
        for (sb, nb), ib in idx.items():
            if sb != "R" or (na in FINGER_LINKS and nb in FINGER_LINKS):
                continue
            for i in ia:
                for j in ib:
                    pairs.append([sphere_frames[i], sphere_frames[j]])
    print(f"{len(pairs)} collision pairs")

    # DM4340 (joints 1-3) / DM4310 (4-6): accel / jerk limits for smooth motion of a 1 kg-class arm
    acc = [12.0, 12.0, 12.0, 20.0, 20.0, 20.0]
    jerk = [6000.0, 6000.0, 6000.0, 10000.0, 10000.0, 10000.0]
    params = {
        "fabric_params": {
            "cspace_attractor": {
                "min_isotropic_mass": 1., "max_isotropic_mass": 1., "mass_sharpness": 20.0,
                "mass_switch_offset": 1., "conical_sharpness": 20., "conical_gain": 1.},
            "joint_limit_repulsion": {
                "velocity_gate": True, "breakaway_distance": 0.0, "breakaway_velocity": 0.,
                "soft_relu_gain": 8., "metric_scalar": 0.02, "metric_exploder_offset": 0.1,
                "max_metric": 100., "damping_gain": 8.},
            "ee_attractor": {
                "min_isotropic_mass": 1., "max_isotropic_mass": 1., "mass_sharpness": 10.,
                "mass_switch_offset": 0.5, "conical_sharpness": 40., "conical_gain": 50.,
                "damping": 50., "damping_sharpness": 10., "damping_radius": .2},
            "body_repulsion": {
                "max_depth": 2., "min_depth": 0.005, "engage_depth": 0.1, "breakaway_depth": 0.0,
                "breakaway_velocity": 0., "metric_scalar": 1., "velocity_gate": True,
                "velocity_gate_sharpness": 1000., "velocity_gate_offset": 0.002,
                "rescaled_min_dist": .01, "forcing_metric_scalar": .01, "geom_metric_scalar": .01,
                "constant_accel": 5., "constant_accel_geom": 15., "damping_gain": 0.,
                "collision_sphere_frames": sphere_frames,
                "collision_sphere_radii": sphere_radii,
                "world_collision_mask": world_mask,
                "collision_sphere_pairs": pairs,
                "collision_link_prefix_pairs": []},
            "cspace_damping": {"gain": 10.},
            "speed_control": {"active": False, "energy_target": 10., "damping": 100.},
            "joint_limits": {"active": True, "acceleration": acc + acc, "jerk": jerk + jerk},
        }
    }
    out_params = os.path.join(PKG, "fabric_params", f"{ROBOT}_pose_params.yaml")
    with open(out_params, "w") as f:
        f.write("# Generated by tools/make_yam_fabric_assets.py -- edit gains here, regenerate spheres there.\n")
        yaml.safe_dump(params, f, sort_keys=False, default_flow_style=None, width=110)
    print("wrote", out_params)

    # world: table top at z = 0.40 and the two static arm bases (MJCF base capsule r 0.033, 0-0.06 m)
    world = {"table": {"env_index": "all", "type": "box", "scaling": "0.8 1.0 0.04",
                       "transform": f"0.25 0. {TABLE_Z - 0.02:.3f} 0. 0. 0. 1."}}
    for side, (x, y, z) in BASES.items():
        world[f"base_{side}"] = {"env_index": "all", "type": "box", "scaling": "0.07 0.07 0.05",
                                 "transform": f"{x:.3f} {y:.3f} {z + 0.025:.3f} 0. 0. 0. 1."}
    out_world = os.path.join(PKG, "worlds", f"{ROBOT}_table.yaml")
    with open(out_world, "w") as f:
        f.write("# Generated by tools/make_yam_fabric_assets.py.\n"
                "# World frame = simulator world frame; table top is the z = 0.40 plane.\n")
        yaml.safe_dump(world, f, sort_keys=False)
    print("wrote", out_world)


if __name__ == "__main__":
    main()
