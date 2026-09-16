#!/usr/bin/env python
"""Generate the Franka Panda assets used by ``FrankaPoseFabric``.

Produces, inside ``src/fabrics_sim``:

* ``models/robots/urdf/franka/franka.urdf``   fabric-ready URDF (see below)
* ``models/robots/urdf/franka/meshes/*.stl``   collision meshes copied from the source description
* ``fabric_params/franka_pose_params.yaml``    fabric parameters incl. fitted collision spheres
* ``worlds/franka_table.yaml``                 table + static base block for world collision

URDF transformations relative to the stock ``panda_arm_hand.urdf``:

* ``package://franka_description/meshes/...`` -> ``meshes/...`` (STL for visual and collision;
  the fabric never renders, and STL avoids a COLLADA dependency).
* The two prismatic finger joints become *fixed* joints at the open position. The fabric
  kinematics kernels support revolute + fixed joints only; the gripper is commanded outside
  the fabric. Finger spheres therefore sit at the open width (conservative).
* ``panda_tcp``: fixed frame 0.1034 m along +z of ``panda_hand`` (between the fingertips; the
  same offset IsaacLab uses for its ``ee_frame``). This is the frame the pose attractor drives.
  Six axis-point frames ``panda_tcp_{x,x_neg,y,y_neg,z,z_neg}`` at +-0.25 m give the
  7-point attractor its orientation handle (same construction as the Kuka-Allegro fabric).
* ``<link>_sphere<i>`` fixed frames at the centres of collision spheres fitted to each link's
  collision mesh (k-means over surface samples; radius = max sample distance + margin, i.e. an
  over-approximation of the surface).
* Every link gets an ``<inertial>`` block (the URDF parser used by the kinematics dereferences it).

Usage (env_fabrics active):
    python tools/make_franka_fabric_assets.py --source <franka_description dir> [--seed 0]
"""
from __future__ import annotations

import argparse
import os
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "src", "fabrics_sim")

DEFAULT_SOURCE = os.path.expanduser(
    "~/Volume_E/Active/others/summer_work/agentic-skill-discovery/external_installations/"
    "env_isaaclab/lib/python3.12/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf/"
    "data/urdf/robots/franka_description")

TCP_OFFSET = 0.1034        # panda_hand -> tool centre point along hand z
AXIS_POINT_OFFSET = 0.25   # attractor axis points
FINGER_OPEN = 0.04         # metres, prismatic joint upper limit
SPHERE_MARGIN = 0.005
# Radius = this percentile of sample distances within a cluster (+ margin). 1.0 would be a
# strict over-approximation but is dominated by the longitudinal extent of the patch; 0.95
# hugs the surface with more spheres.
RADIUS_PERCENTILE = 0.95
# Height of the static box standing in for panda_link0. The real base is 0.14 m tall but its
# upper part is a narrow neck under link1's spheres; a full-height box penetrates them.
BASE_BOX_HEIGHT = 0.06

# Number of spheres per link. link0 is the static base; it is represented as a static
# world box instead (see worlds/franka_table.yaml) so it never reports a spurious
# self-collision against the table it sits on.
SPHERES_PER_LINK = {
    "panda_link1": 10, "panda_link2": 10, "panda_link3": 8, "panda_link4": 8,
    "panda_link5": 12, "panda_link6": 5, "panda_link7": 4, "panda_hand": 6,
    "panda_leftfinger": 2, "panda_rightfinger": 2,
}

# One-directional prefix pairs, mirroring the Kuka-Allegro convention: spheres whose frame
# name contains the first prefix are repulsed from spheres containing the second.
SELF_COLLISION_PREFIX_PAIRS = [
    ["panda_hand", "panda_link1"], ["panda_hand", "panda_link2"],
    ["panda_hand", "panda_link3"], ["panda_hand", "panda_link4"],
    ["panda_leftfinger", "panda_link1"], ["panda_leftfinger", "panda_link2"],
    ["panda_rightfinger", "panda_link1"], ["panda_rightfinger", "panda_link2"],
    ["panda_link7", "panda_link1"], ["panda_link7", "panda_link2"],
    ["panda_link6", "panda_link1"], ["panda_link6", "panda_link2"],
    ["panda_link5", "panda_link1"], ["panda_link5", "panda_link2"],
    ["panda_link4", "panda_link1"],
]

INERTIAL_XML = ('<inertial><mass value="0.01"/>'
                '<inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/></inertial>')


def rpy_to_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def kmeans(points, k, seed, iters=60):
    rng = np.random.default_rng(seed)
    # k-means++ init
    centers = [points[rng.integers(len(points))]]
    for _ in range(1, k):
        d2 = np.min(((points[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(points[rng.choice(len(points), p=d2 / d2.sum())])
    centers = np.array(centers)
    for _ in range(iters):
        assign = np.argmin(((points[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        new = np.array([points[assign == j].mean(0) if np.any(assign == j) else centers[j]
                        for j in range(k)])
        if np.allclose(new, centers, atol=1e-6):
            break
        centers = new
    assign = np.argmin(((points[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    radii = np.array([np.quantile(np.linalg.norm(points[assign == j] - centers[j], axis=1),
                                  RADIUS_PERCENTILE) for j in range(k)])
    return centers, radii


def fit_link_spheres(mesh_path, origin_xyz, origin_rpy, k, seed, n_samples=4000):
    m = trimesh.load(mesh_path, force="mesh")
    pts, _ = trimesh.sample.sample_surface(m, n_samples, seed=seed)
    R = rpy_to_matrix(*origin_rpy)
    pts = pts @ R.T + np.asarray(origin_xyz)
    centers, radii = kmeans(np.asarray(pts), k, seed)
    return centers, radii + SPHERE_MARGIN


def sub(parent, tag, **attrib):
    e = ET.SubElement(parent, tag)
    for k, v in attrib.items():
        e.set(k, v)
    return e


def add_fixed_frame(robot, parent_link, name, xyz, rpy="0 0 0"):
    link = sub(robot, "link", name=name)
    link.append(ET.fromstring(INERTIAL_XML))
    j = sub(robot, "joint", name=name + "_joint", type="fixed")
    sub(j, "origin", rpy=rpy, xyz=xyz)
    sub(j, "parent", link=parent_link)
    sub(j, "child", link=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="franka_description directory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src_urdf = os.path.join(args.source, "robots", "panda_arm_hand.urdf")
    src_meshes = os.path.join(args.source, "meshes", "collision")
    out_dir = os.path.join(PKG, "models", "robots", "urdf", "franka")
    out_meshes = os.path.join(out_dir, "meshes")
    os.makedirs(out_meshes, exist_ok=True)

    tree = ET.parse(src_urdf)
    robot = tree.getroot()
    robot.set("name", "franka")

    # 1) meshes -> local STL copies, referenced relatively (visual and collision).
    for mesh in robot.iter("mesh"):
        fn = mesh.get("filename")
        base = os.path.splitext(os.path.basename(fn))[0]
        shutil.copy(os.path.join(src_meshes, base + ".stl"), os.path.join(out_meshes, base + ".stl"))
        mesh.set("filename", f"meshes/{base}.stl")

    # 2) inertial on every link.
    for link in robot.findall("link"):
        if link.find("inertial") is None:
            link.append(ET.fromstring(INERTIAL_XML))

    # 3) finger joints -> fixed at open width.
    for j in robot.findall("joint"):
        if j.get("type") == "prismatic":
            axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
            org = j.find("origin")
            xyz = np.array([float(v) for v in org.get("xyz").split()]) + FINGER_OPEN * axis
            org.set("xyz", " ".join(f"{v:.6f}" for v in xyz))
            j.set("type", "fixed")
            j.remove(j.find("limit"))
            j.remove(j.find("axis"))

    # Collision origins per link (needed to place fitted spheres in the link frame).
    coll_origin = {}
    for link in robot.findall("link"):
        c = link.find("collision")
        if c is None:
            continue
        o = c.find("origin")
        xyz = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()]
        mesh = c.find("geometry/mesh")
        coll_origin[link.get("name")] = (xyz, rpy, os.path.basename(mesh.get("filename")))

    # 4) TCP + axis points.
    add_fixed_frame(robot, "panda_hand", "panda_tcp", f"0 0 {TCP_OFFSET}")
    a = AXIS_POINT_OFFSET
    for name, xyz in [("panda_tcp_x", f"{a} 0 0"), ("panda_tcp_x_neg", f"-{a} 0 0"),
                      ("panda_tcp_y", f"0 {a} 0"), ("panda_tcp_y_neg", f"0 -{a} 0"),
                      ("panda_tcp_z", f"0 0 {a}"), ("panda_tcp_z_neg", f"0 0 -{a}")]:
        add_fixed_frame(robot, "panda_tcp", name, xyz)

    # 5) collision spheres.
    sphere_frames, sphere_radii = [], []
    for link_name, k in SPHERES_PER_LINK.items():
        xyz, rpy, mesh_file = coll_origin[link_name]
        centers, radii = fit_link_spheres(os.path.join(out_meshes, mesh_file), xyz, rpy, k, args.seed)
        for i, (c, r) in enumerate(zip(centers, radii)):
            fname = f"{link_name}_sphere{i}"
            add_fixed_frame(robot, link_name, fname, " ".join(f"{v:.5f}" for v in c))
            sphere_frames.append(fname)
            sphere_radii.append(round(float(r), 4))
        print(f"{link_name:18s} {k} spheres, radii {np.round(radii, 3)}")

    ET.indent(tree, space="  ")
    out_urdf = os.path.join(out_dir, "franka.urdf")
    tree.write(out_urdf, xml_declaration=True, encoding="utf-8")
    print("wrote", out_urdf)

    # 6) fabric params.
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
                "max_depth": 2., "min_depth": 0.01, "engage_depth": 0.5, "breakaway_depth": 0.0,
                "breakaway_velocity": 0., "metric_scalar": 1., "velocity_gate": True,
                "velocity_gate_sharpness": 1000., "velocity_gate_offset": 0.002,
                "rescaled_min_dist": .01, "forcing_metric_scalar": .01, "geom_metric_scalar": .01,
                # 5/15 (Kuka defaults 1/5): kept >=0.7 cm sphere clearance on the franka_cylinders
                # course where 1/5 grazed to -0.1 cm; free-space convergence unchanged.
                "constant_accel": 5., "constant_accel_geom": 15., "damping_gain": 0.,
                "collision_sphere_frames": sphere_frames,
                "collision_sphere_radii": sphere_radii,
                "collision_sphere_pairs": [],
                "collision_link_prefix_pairs": SELF_COLLISION_PREFIX_PAIRS},
            "cspace_damping": {"gain": 10.},
            "speed_control": {"active": False, "energy_target": 10., "damping": 100.},
            # Franka spec-sheet limits (same values as prod/config/franka_fabric.yaml, arm only).
            "joint_limits": {
                "active": True,
                "acceleration": [15., 7.5, 10., 12.5, 15., 20., 20.],
                "jerk": [7500., 3750., 5000., 6250., 7500., 10000., 10000.]},
        }
    }
    out_params = os.path.join(PKG, "fabric_params", "franka_pose_params.yaml")
    with open(out_params, "w") as f:
        f.write("# Generated by tools/make_franka_fabric_assets.py -- edit gains here, regenerate "
                "spheres there.\n")
        yaml.safe_dump(params, f, sort_keys=False, default_flow_style=None, width=100)
    print("wrote", out_params)

    # 7) world: table (top at z=0, robot base plane) and a static box standing in for link0.
    m0 = trimesh.load(os.path.join(out_meshes, "link0.stl"), force="mesh")
    lo, hi = m0.bounds
    size = np.array([hi[0] - lo[0], hi[1] - lo[1], BASE_BOX_HEIGHT])
    center = np.array([(hi[0] + lo[0]) / 2, (hi[1] + lo[1]) / 2, BASE_BOX_HEIGHT / 2])
    world = {
        "table": {"env_index": "all", "type": "box",
                  "scaling": "1.6 1.6 0.04", "transform": "0.4 0. -0.02 0. 0. 0. 1."},
        "robot_base": {"env_index": "all", "type": "box",
                       "scaling": " ".join(f"{v:.3f}" for v in size),
                       "transform": " ".join(f"{v:.3f}" for v in center) + " 0. 0. 0. 1."},
    }
    out_world = os.path.join(PKG, "worlds", "franka_table.yaml")
    with open(out_world, "w") as f:
        f.write("# Generated by tools/make_franka_fabric_assets.py.\n"
                "# Robot base frame is the world origin; the table top is the z=0 plane.\n"
                "# robot_base is a static box around panda_link0 so the hand avoids the base.\n")
        yaml.safe_dump(world, f, sort_keys=False)
    print("wrote", out_world)


if __name__ == "__main__":
    main()
