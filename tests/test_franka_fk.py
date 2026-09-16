"""Cross-check the fabric's TCP forward kinematics against urdfpy on the same URDF."""
import numpy as np
import torch
import warp as wp
from urdfpy import URDF

from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap
from fabrics_sim.utils.path_utils import get_robot_urdf_path


def test_tcp_fk_matches_urdfpy(batch=8, seed=0, tol=1e-4):
    wp.init()
    device = 'cuda:0'
    urdf_path = get_robot_urdf_path("franka", "franka")
    robot = URDF.load(urdf_path)
    names = [j.name for j in robot.joints if j.joint_type == 'revolute']
    assert names == [f"panda_joint{i}" for i in range(1, 8)], names

    rng = np.random.default_rng(seed)
    lower = np.array([j.limit.lower for j in robot.joints if j.joint_type == 'revolute'])
    upper = np.array([j.limit.upper for j in robot.joints if j.joint_type == 'revolute'])
    qs = lower + (upper - lower) * rng.random((batch, 7))
    qs[0] = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]   # IsaacLab default

    frames = ["panda_tcp", "panda_tcp_x", "panda_tcp_y", "panda_tcp_z", "panda_hand", "panda_link4"]
    taskmap = RobotFrameOriginsTaskMap(urdf_path, frames, batch, device)
    x, _ = taskmap(torch.tensor(qs, dtype=torch.float32, device=device), None)
    x = x.reshape(batch, len(frames), 3).cpu().numpy()

    link_map = {l.name: l for l in robot.links}
    worst = 0.0
    for b in range(batch):
        fk = robot.link_fk(cfg=dict(zip(names, qs[b])))
        for fi, f in enumerate(frames):
            ref = fk[link_map[f]][:3, 3]
            err = np.linalg.norm(ref - x[b, fi])
            worst = max(worst, err)
    print(f"default-config TCP (fabric): {x[0, 0].round(4)}  worst |err| over {batch}x{len(frames)} frames: {worst:.2e}")
    assert worst < tol, worst


if __name__ == "__main__":
    test_tcp_fk_matches_urdfpy()
    print("FK OK")
