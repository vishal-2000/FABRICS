"""YamBimanualPoseFabric: kinematics, home clearance, reaching, and arm-arm collision avoidance.

FK reference values are Newton's (MJCF i2rt_yam, arms at y = +-0.22, z = 0.40, TCP (0, -0.044, 0.1435)
in link_6), from the agentic-skill-discovery LEGO scene.
"""
import math

import numpy as np
import torch
import warp as wp

from fabrics_sim.fabrics.yam_bimanual_pose_fabric import YamBimanualPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

DEVICE = "cuda:0"
# (q L1..6 R1..6, L tcp (x y z qx qy qz qw), R tcp)
FK_REF = [
    ([0.0, 1.047, 1.047, 0.0, 0.0, 0.0] * 2,
     [0.38575, 0.22, 0.74860, 0.5, 0.5, 0.5, 0.5], [0.38575, -0.22, 0.74860, 0.5, 0.5, 0.5, 0.5]),
    ([0.99506, 0.98882, 0.15018, -1.51888, 0.98417, 1.72895, 0.82304, 2.67374, 1.99249, 1.36682, 0.99229, -2.08293],
     [0.07211, 0.10177, 0.42982, 0.67021, 0.49625, -0.04529, 0.55], [0.5183, 0.08787, 0.62047, 0.10463, 0.56211, -0.07839, 0.81666]),
    ([2.24548, 0.1231, 2.67433, -1.01896, 1.14096, 0.17367, -0.91793, 1.54923, 0.1038, -1.18035, 0.53603, 0.61655],
     [0.39283, -0.01596, 0.77659, 0.51906, 0.14955, 0.77119, -0.33685], [-0.12966, -0.16477, 0.44486, 0.83066, 0.14841, -0.47207, -0.25519]),
]


_CACHE = {}


def _setup(dt=1. / 200.):
    """One fabric + world per process: a second WorldMeshesModel/fabric in the same process faults in warp 1.8."""
    if not _CACHE:
        wp.init()
        world = WorldMeshesModel(batch_size=1, max_objects_per_env=20, device=DEVICE, world_filename='yam_bimanual_table')
        ids, ind = world.get_object_ids()
        fabric = YamBimanualPoseFabric(1, DEVICE, dt, graph_capturable=False)
        _CACHE.update(world=world, out=(fabric, DisplacementIntegrator(fabric), ids, ind))
    return _CACHE["out"]


def _qang(a, b):
    """Rotation angle between quaternions in degrees (atan2 form: no acos round-off near zero)."""
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    return 2 * math.degrees(math.atan2(min(np.linalg.norm(a - b), np.linalg.norm(a + b)), max(np.linalg.norm(a + b), np.linalg.norm(a - b))))


def test_fk_matches_newton():
    fabric, _, _, _ = _setup()
    for q, pl, pr in FK_REF:
        qt = torch.tensor([q], device=DEVICE)
        for h, ref in (("L", pl), ("R", pr)):
            e = fabric.get_ee_pose(qt, h)[0].cpu().numpy()
            assert np.linalg.norm(e[:3] - np.array(ref[:3])) < 2e-4, (h, e, ref)
            assert _qang(e[3:], np.array(ref[3:])) < 0.05, (h, e, ref)


def test_home_clearance():
    fabric, _, ids, ind = _setup()
    q = fabric.default_config.clone()
    tg = {h: fabric.get_ee_pose(q, h) for h in fabric.HANDS}
    fabric.set_features(tg, q, torch.zeros_like(q), ids, ind)
    sd = fabric.base_fabric_repulsion.signed_distance[0]
    assert float(sd.min()) > 0.0, float(sd.min())
    assert not bool(fabric.collision_status.max().item())


def _run(fabric, integ, ids, ind, q, targets, steps):
    qd, qdd = torch.zeros_like(q), torch.zeros_like(q)
    min_sd = 1e9
    for _ in range(steps):
        fabric.set_features(targets, q.detach(), qd.detach(), ids, ind)
        q, qd, qdd = integ.step(q.detach(), qd.detach(), qdd.detach(), fabric.allocated_data['timestep'])
        min_sd = min(min_sd, float(fabric.base_fabric_repulsion.signed_distance.min()))
    return q, min_sd


def test_reach_both_hands():
    fabric, integ, ids, ind = _setup()
    q = fabric.default_config.clone()
    tg = {}
    for h, dy in (("L", -0.08), ("R", 0.08)):  # 8 cm inward and 15 cm down, same orientation
        p = fabric.get_ee_pose(q, h)
        p[:, 1] += dy; p[:, 2] -= 0.15
        tg[h] = p
    q, min_sd = _run(fabric, integ, ids, ind, q, tg, 800)
    for h in fabric.HANDS:
        e = fabric.get_ee_pose(q, h)[0].cpu().numpy(); t = tg[h][0].cpu().numpy()
        assert np.linalg.norm(e[:3] - t[:3]) < 0.002, (h, e, t)
        assert _qang(e[3:], t[3:]) < 1.0
    assert min_sd > 0.0


def test_arms_do_not_collide_on_conflicting_targets():
    """Both TCPs are sent to the same point between the arms: the arm-arm sphere pairs must keep them apart."""
    fabric, integ, ids, ind = _setup()
    q = fabric.default_config.clone()
    tg = {h: fabric.get_ee_pose(q, h) for h in fabric.HANDS}
    for h in fabric.HANDS:
        tg[h][:, 1] = 0.0
        tg[h][:, 2] = 0.60
    q, min_sd = _run(fabric, integ, ids, ind, q, tg, 800)
    cm = fabric.collision_matrix.bool()
    x = fabric.get_taskmap("body_points")(q, None)[0][0].reshape(-1, 3)
    r = torch.tensor(fabric.get_sphere_radii(), device=DEVICE)
    d = torch.cdist(x, x) - r[:, None] - r[None, :]
    pair_min = float(torch.where(cm | cm.T, d, torch.full_like(d, 9.)).min())
    gap = float(np.linalg.norm(fabric.get_ee_pose(q, "L")[0, :3].cpu().numpy() - fabric.get_ee_pose(q, "R")[0, :3].cpu().numpy()))
    print(f"min pair distance {1000 * pair_min:.1f} mm, TCP gap {1000 * gap:.1f} mm, min signed {1000 * min_sd:.1f} mm")
    assert pair_min > 0.0
    assert min_sd > 0.0
    assert gap > 0.05


if __name__ == "__main__":
    test_fk_matches_newton(); print("FK OK")
    test_home_clearance(); print("HOME CLEARANCE OK")
    test_reach_both_hands(); print("REACH OK")
    test_arms_do_not_collide_on_conflicting_targets(); print("ARM-ARM AVOIDANCE OK")
