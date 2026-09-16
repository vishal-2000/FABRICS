"""At the home configuration no Franka body sphere may penetrate the world or another sphere.

Prints the signed distance of every sphere so a bad sphere fit or world object is obvious.
"""
import torch
import warp as wp

from fabrics_sim.fabrics.franka_pose_fabric import FrankaPoseFabric
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel


def test_home_clearance(verbose=True):
    wp.init()
    device = 'cuda:0'
    world = WorldMeshesModel(batch_size=1, max_objects_per_env=20, device=device,
                             world_filename='franka_table')
    object_ids, object_indicator = world.get_object_ids()
    fabric = FrankaPoseFabric(1, device, 1./60., graph_capturable=False)
    q = fabric.default_config.clone()
    qd = torch.zeros_like(q)
    pose = fabric.get_ee_pose(q, "quaternion")
    fabric.set_features(pose, "quaternion", q, qd, object_ids, object_indicator)

    frames = fabric.fabric_params['body_repulsion']['collision_sphere_frames']
    radii = fabric.get_sphere_radii()
    sd = fabric.base_fabric_repulsion.signed_distance[0].cpu().numpy()
    pos = fabric.get_taskmap("body_points")(q, None)[0][0].reshape(-1, 3).cpu().numpy()
    if verbose:
        order = sd.argsort()
        print(f"{'sphere':26s} {'r':>6s} {'z':>7s} {'signed_dist':>12s}")
        for i in order[:12]:
            print(f"{frames[i]:26s} {radii[i]:6.3f} {pos[i,2]:7.3f} {sd[i]:12.4f}")
    bad = [(frames[i], float(sd[i])) for i in range(len(frames)) if sd[i] < 0]
    assert not bad, f"spheres penetrating at home: {bad}"
    assert not bool(fabric.collision_status.max().item())


if __name__ == "__main__":
    test_home_clearance()
    print("HOME CLEARANCE OK")
