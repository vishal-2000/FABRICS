"""Franka Panda TCP pose fabric: headless convergence + collision check.

Runs a batch of Franka fabrics toward a sequence of random reachable TCP targets above a
table, then toward a deliberately unreachable target *below* the table, and reports:

* position / orientation error at the end of each target hold,
* joint-limit margins and collision flags throughout,
* that the below-table target is refused (hand stays above the table, no collision).

Targets are exactly vertical (gripper pointing down) with random yaw, positions sampled in a box
over the table. Expected: 100% of holds within 1 cm / 5 deg (typically < 1 mm / < 0.5 deg).

Example:
    python examples/franka_pose_fabric_example.py --cuda_graph
    python examples/franka_pose_fabric_example.py --zrange 0.03 0.30 --diag
"""
import argparse
import time

import numpy as np
import torch

from fabrics_sim.fabrics.franka_pose_fabric import FrankaPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp, capture_fabric
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

torch.set_printoptions(precision=4, sci_mode=False)

parser = argparse.ArgumentParser(description='Franka TCP pose fabric example.')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--cuda_graph', action='store_true')
parser.add_argument('--num_targets', type=int, default=6, help='random reachable targets')
parser.add_argument('--hold_s', type=float, default=6.0, help='seconds per target')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--zrange', type=float, nargs=2, default=[0.03, 0.50], help='TCP target z range (m)')
parser.add_argument('--forcing_cspace_gain', type=float, default=None,
                    help='add a forcing home-posture attractor with this conical gain')
parser.add_argument('--home_q', type=float, nargs=7, default=None,
                    help='override the home configuration used by the posture attractor')
parser.add_argument('--ee_damping', type=float, default=None, help='override ee_attractor.damping')
parser.add_argument('--ee_gain', type=float, default=None, help='override ee_attractor.conical_gain')
parser.add_argument('--stage', action='store_true',
                    help='first command the TCP pose of the reach posture (staging move), then targets')
parser.add_argument('--start_orientation', action='store_true',
                    help='use the start pose TCP orientation (45deg tilted for the IsaacLab init) as the '
                         'base orientation instead of an exactly vertical gripper')
parser.add_argument('--param', action='append', default=[], metavar='SECTION.KEY=VALUE',
                    help='override any scalar fabric parameter, e.g. --param body_repulsion.constant_accel=1')
parser.add_argument('--cspace_mass', type=float, default=None,
                    help='override isotropic mass of the geometric home-posture attractor')
parser.add_argument('--diag', action='store_true', help='print joint/sphere diagnostics for the worst element of each hold')
args = parser.parse_args()

batch_size = args.batch_size
device = 'cuda:0'
initialize_warp("0")
torch.manual_seed(args.seed)

# World: table top at z=0 + static box for the robot base.
world_model = WorldMeshesModel(batch_size=batch_size, max_objects_per_env=20,
                               device=device, world_filename='franka_table')
object_ids, object_indicator = world_model.get_object_ids()

control_rate = 60.
timestep = 1. / control_rate

import copy, yaml
from fabrics_sim.utils.path_utils import get_params_path
import os
with open(os.path.join(get_params_path(), "franka_pose_params.yaml")) as f:
    params = yaml.safe_load(f)['fabric_params']
if args.cspace_mass is not None:
    params['cspace_attractor']['min_isotropic_mass'] = args.cspace_mass
    params['cspace_attractor']['max_isotropic_mass'] = args.cspace_mass
if args.forcing_cspace_gain is not None:
    params['forcing_cspace_attractor'] = {
        'min_isotropic_mass': 1., 'max_isotropic_mass': 1., 'mass_sharpness': 20.0,
        'mass_switch_offset': 1., 'conical_sharpness': 20., 'conical_gain': args.forcing_cspace_gain,
        'damping': 0., 'damping_sharpness': 10., 'damping_radius': 0.2}
for kv in args.param:
    k, v = kv.split('=', 1)
    sec, key = k.split('.', 1)
    params[sec][key] = type(params[sec][key])(float(v)) if isinstance(params[sec][key], (int, float)) else v
if args.ee_damping is not None:
    params['ee_attractor']['damping'] = args.ee_damping
if args.ee_gain is not None:
    params['ee_attractor']['conical_gain'] = args.ee_gain
fabric = FrankaPoseFabric(batch_size, device, timestep, graph_capturable=args.cuda_graph,
                          fabric_params=params)
if args.home_q is not None:
    fabric.default_config = torch.tensor(args.home_q, device=device).unsqueeze(0).repeat(batch_size, 1)
integrator = DisplacementIntegrator(fabric)
num_joints = fabric.num_joints

# Start where the IsaacLab Franka starts, regardless of the posture attractor's home.
q = torch.tensor(FrankaPoseFabric.INIT_CONFIG, device=device).unsqueeze(0).repeat(batch_size, 1).contiguous()
qd = torch.zeros(batch_size, num_joints, device=device)
qdd = torch.zeros(batch_size, num_joints, device=device)

# The start pose gives the "gripper pointing down" orientation we perturb with yaw.
home_pose = fabric.get_ee_pose(q, "quaternion")
print('home TCP pose (x y z qx qy qz qw):', home_pose[0].cpu().numpy().round(4))
home_quat = home_pose[:, 3:7]
if not args.start_orientation:
    # Hand z pointing straight down, hand x along world x: 180 degrees about world x.
    home_quat = torch.tensor([1., 0., 0., 0.], device=device).expand(batch_size, 4).contiguous()


def quat_mul(a, b):  # (x,y,z,w) Hamilton product a*b
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw*bx + ax*bw + ay*bz - az*by,
                        aw*by - ax*bz + ay*bw + az*bx,
                        aw*bz + ax*by - ay*bx + az*bw,
                        aw*bw - ax*bx - ay*by - az*bz], dim=-1)


def yaw_quat(yaw):  # rotation about world z
    return torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw),
                        torch.sin(yaw/2), torch.cos(yaw/2)], dim=-1)


def sample_targets():
    """Random TCP targets in a box above the table, gripper down, random yaw."""
    lo = torch.tensor([0.30, -0.35, args.zrange[0]], device=device)
    hi = torch.tensor([0.65, 0.35, args.zrange[1]], device=device)
    pos = lo + (hi - lo) * torch.rand(batch_size, 3, device=device)
    yaw = (torch.rand(batch_size, device=device) - 0.5) * np.pi
    quat = quat_mul(yaw_quat(yaw), home_quat)
    return torch.cat([pos, quat], dim=-1)


def quat_angle(a, b):
    dot = (a * b).sum(-1).abs().clamp(max=1.0)
    return 2. * torch.acos(dot)


target = sample_targets().contiguous()

g = None
if args.cuda_graph:
    inputs = [target, "quaternion", q.detach(), qd.detach(), object_ids, object_indicator]
    g, q_new, qd_new, qdd_new = capture_fabric(fabric, q, qd, qdd, timestep, integrator, inputs, device)


def step():
    global q, qd, qdd
    if args.cuda_graph:
        g.replay()
        q.copy_(q_new); qd.copy_(qd_new); qdd.copy_(qdd_new)
    else:
        fabric.set_features(target, "quaternion", q.detach(), qd.detach(), object_ids, object_indicator)
        q, qd, qdd = integrator.step(q.detach(), qd.detach(), qdd.detach(), timestep)


def run_hold(label, seconds):
    """Hold the current target for `seconds`; return summary dict."""
    n = int(control_rate * seconds)
    min_signed = float('inf'); any_collision = False
    min_upper = float('inf'); min_lower = float('inf')
    t0 = time.time()
    for i in range(n):
        step()
        min_signed = min(min_signed, fabric.base_fabric_repulsion.signed_distance.min().item())
        any_collision |= bool(fabric.collision_status.max().item())
        min_upper = min(min_upper, fabric.get_taskmap_position("upper_joint_limit").min().item())
        min_lower = min(min_lower, fabric.get_taskmap_position("lower_joint_limit").min().item())
        if args.verbose and i % 30 == 0:
            pose = fabric.get_ee_pose(q, "quaternion")
            print(f'  t={i*timestep:5.2f}s pos_err={torch.norm(pose[:, :3]-target[:, :3], dim=1).max():.4f}')
    wall = time.time() - t0
    pose = fabric.get_ee_pose(q, "quaternion")
    pos_err = torch.norm(pose[:, :3] - target[:, :3], dim=1)
    ang_err = quat_angle(pose[:, 3:7], target[:, 3:7]) * 180. / np.pi
    print(f'[{label}] {seconds:.1f}s sim in {wall:.2f}s wall | pos_err cm: '
          f'{(pos_err*100).cpu().numpy().round(2)} | ang_err deg: {ang_err.cpu().numpy().round(1)} | '
          f'min_signed_dist {min_signed:.3f} collision {any_collision} | '
          f'joint-limit margin up {min_upper:.3f} low {min_lower:.3f} | '
          f'|qd|max {qd.abs().max().item():.2f}')
    if args.diag:
        worst = int(torch.argmax(pos_err).item())
        frames = fabric.fabric_params['body_repulsion']['collision_sphere_frames']
        sd = fabric.base_fabric_repulsion.signed_distance[worst].cpu().numpy()
        up = fabric.get_taskmap_position("upper_joint_limit")[worst].cpu().numpy()
        lo = fabric.get_taskmap_position("lower_joint_limit")[worst].cpu().numpy()
        print(f'    worst elem {worst}: pos_err {pos_err[worst]*100:.2f}cm ang {ang_err[worst]:.1f}deg')
        print(f'    q        {q[worst].cpu().numpy().round(3)}')
        print(f'    to upper {up.round(3)}')
        print(f'    to lower {lo.round(3)}')
        print(f'    target   {target[worst].cpu().numpy().round(3)}')
        print(f'    tcp      {pose[worst].cpu().numpy().round(3)}')
        order = sd.argsort()[:4]
        print('    closest spheres: ' + ', '.join(f'{frames[i]}={sd[i]:.3f}' for i in order))
    return dict(pos_err=pos_err, ang_err=ang_err, collision=any_collision, min_signed=min_signed,
                tcp_z=pose[:, 2])


# --- 0) optional staging move: TCP pose of the posture attractor's home ----------------------
if args.stage:
    stage_pose = fabric.get_ee_pose(fabric.default_config, "quaternion")
    target.copy_(stage_pose)
    run_hold('staging move to reach posture', args.hold_s)

# --- 1) reachable targets ------------------------------------------------------------------
results = []
for k in range(args.num_targets):
    target.copy_(sample_targets())
    results.append(run_hold(f'target {k+1}/{args.num_targets}', args.hold_s))

pos = torch.cat([r['pos_err'] for r in results]) * 100.
ang = torch.cat([r['ang_err'] for r in results])
ok = ((pos < 1.0) & (ang < 5.0)).float().mean().item()
print(f'\nREACHABLE: {len(results)*batch_size} target holds | pos_err cm median {pos.median():.2f} '
      f'max {pos.max():.2f} | ang_err deg median {ang.median():.1f} max {ang.max():.1f} | '
      f'success(<1cm,<5deg) {ok*100:.0f}% | any collision {any(r["collision"] for r in results)}')

# --- 2) unreachable: target below the table -------------------------------------------------
below = sample_targets()
below[:, 0] = 0.45; below[:, 1] = 0.0; below[:, 2] = -0.15
target.copy_(below)
r = run_hold('below-table target', args.hold_s)
print(f'BELOW-TABLE: tcp_z after hold {r["tcp_z"].cpu().numpy().round(3)} (target z=-0.15) | '
      f'collision {r["collision"]} | min_signed_dist {r["min_signed"]:.3f}')

# --- 3) back home -----------------------------------------------------------------------------
target.copy_(home_pose)
r = run_hold('return home', args.hold_s)
print('Done')

