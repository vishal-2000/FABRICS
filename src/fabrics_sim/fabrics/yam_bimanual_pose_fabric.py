"""Bimanual I2RT YAM pose fabric.

Two 6-DoF YAM arms as one 12-joint fabric, with a 6-DoF pose action space per hand on the tool
centre points ``L_tcp`` / ``R_tcp`` (between the fingertip pads). Because both arms live in one
fabric, arm-arm collision avoidance is the same body-sphere pair repulsion as self-collision.
Joint limits, acceleration/jerk limits and table/base collision (world meshes) are handled as in
``FrankaPoseFabric``.

Everything is in the simulator's world frame (see ``tools/make_yam_fabric_assets.py``). Joint
order: L_joint1..6, R_joint1..6. The grippers are not part of the fabric; the finger joints are
fixed open in the URDF and the finger spheres collide only with the other arm's links (not with
the world, not with the other arm's fingers).
"""
import torch

from fabrics_sim.fabric_terms.attractor import Attractor
from fabrics_sim.fabric_terms.joint_limit_repulsion import JointLimitRepulsion
from fabrics_sim.fabric_terms.body_sphere_3d_repulsion import BodySphereRepulsion
from fabrics_sim.fabric_terms.body_sphere_3d_repulsion import BaseFabricRepulsion
from fabrics_sim.fabrics.fabric import BaseFabric
from fabrics_sim.taskmaps.identity import IdentityMap
from fabrics_sim.taskmaps.upper_joint_limit import UpperJointLimitMap
from fabrics_sim.taskmaps.lower_joint_limit import LowerJointLimitMap
from fabrics_sim.energy.euclidean_energy import EuclideanEnergy
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap
from fabrics_sim.utils.path_utils import get_robot_urdf_path
from fabrics_sim.utils.rotation_utils import quaternion_to_matrix, matrix_to_quaternion


class YamBimanualPoseFabric(BaseFabric):
    """Fabric for two YAM arms with one TCP pose target per hand."""

    ROBOT_DIR_NAME = "yam_bimanual"
    ROBOT_NAME = "yam_bimanual"
    HANDS = ("L", "R")
    # Must match AXIS_POINT_OFFSET in tools/make_yam_fabric_assets.py.
    AXIS_POINT_OFFSET = 0.25
    # Menagerie "home" keyframe for both arms.
    INIT_CONFIG = [0.0, 1.047, 1.047, 0.0, 0.0, 0.0] * 2
    DEFAULT_CONFIG = INIT_CONFIG

    def __init__(self, batch_size, device, timestep, graph_capturable=True,
                 fabric_params_filename="yam_bimanual_pose_params.yaml", fabric_params=None):
        super().__init__(device, batch_size, timestep, fabric_params_filename,
                         fabric_params=fabric_params, graph_capturable=graph_capturable)

        self.urdf_path = get_robot_urdf_path(self.ROBOT_DIR_NAME, self.ROBOT_NAME)
        self.load_robot(self.ROBOT_DIR_NAME, self.ROBOT_NAME, batch_size)
        assert self._num_joints == 12, f"expected 12 revolute joints, found {self._num_joints}"

        default_config = torch.tensor(self.DEFAULT_CONFIG, device=self.device)
        self.default_config = default_config.unsqueeze(0).repeat(self.batch_size, 1)

        self.construct_fabric()

        a = self.AXIS_POINT_OFFSET
        self._axis_points_local = torch.tensor(
            [[0, 0, 0, 1], [a, 0, 0, 1], [-a, 0, 0, 1], [0, a, 0, 1], [0, -a, 0, 1],
             [0, 0, a, 1], [0, 0, -a, 1]], device=device, dtype=torch.float32)
        self._T = torch.zeros(batch_size, 4, 4, device=device)
        self._T[:, 3, 3] = 1.
        self._native_ee_target = {h: None for h in self.HANDS}

    # ------------------------------------------------------------------ terms
    def add_joint_limit_repulsion(self):
        joints = [j for j in self.urdfpy_robot.joints if j.joint_type == 'revolute']
        upper = [j.limit.upper for j in joints]
        lower = [j.limit.lower for j in joints]
        self.add_taskmap("upper_joint_limit", UpperJointLimitMap(upper, self.batch_size, self.device),
                         graph_capturable=self.graph_capturable)
        self.add_fabric("upper_joint_limit", "joint_limit_repulsion",
                        JointLimitRepulsion(True, self.fabric_params['joint_limit_repulsion'],
                                            self.device, graph_capturable=self.graph_capturable))
        self.add_taskmap("lower_joint_limit", LowerJointLimitMap(lower, self.batch_size, self.device),
                         graph_capturable=self.graph_capturable)
        self.add_fabric("lower_joint_limit", "joint_limit_repulsion",
                        JointLimitRepulsion(True, self.fabric_params['joint_limit_repulsion'],
                                            self.device, graph_capturable=self.graph_capturable))

    def add_cspace_attractor(self):
        """Geometric (energy-neutral) posture attractor toward the home configuration.

        A 6-DoF arm has no null space for a full pose target, so this term only picks the IK branch
        and shapes the path; set the home to the target's IK solution for a stage to keep them
        consistent.
        """
        self.add_taskmap("identity", IdentityMap(self.device), graph_capturable=self.graph_capturable)
        self.add_fabric("identity", "cspace_attractor",
                        Attractor(False, self.fabric_params['cspace_attractor'],
                                  self.device, graph_capturable=self.graph_capturable))

    def add_ee_points_attractor(self):
        """Per hand: forcing attractor on 7 points rigidly attached to the TCP."""
        for h in self.HANDS:
            ee = f"{h}_tcp"
            frames = [ee, f"{ee}_x", f"{ee}_x_neg", f"{ee}_y", f"{ee}_y_neg", f"{ee}_z", f"{ee}_z_neg"]
            self.add_taskmap(f"ee_{h}", RobotFrameOriginsTaskMap(self.urdf_path, frames, self.batch_size, self.device),
                             graph_capturable=self.graph_capturable)
            self.add_fabric(f"ee_{h}", "ee_attractor",
                            Attractor(True, self.fabric_params['ee_attractor'],
                                      self.device, graph_capturable=self.graph_capturable))

    def add_body_repulsion(self):
        params = self.fabric_params['body_repulsion']
        frames = params['collision_sphere_frames']
        self.collision_sphere_radii = params['collision_sphere_radii']
        assert len(frames) == len(self.collision_sphere_radii)
        index = {f: i for i, f in enumerate(frames)}
        collision_matrix = torch.zeros(len(frames), len(frames), dtype=int, device=self.device)
        for s1, s2 in params['collision_sphere_pairs']:
            collision_matrix[index[s1], index[s2]] = 1
        self.collision_matrix = collision_matrix

        self.add_taskmap("body_points", RobotFrameOriginsTaskMap(self.urdf_path, frames, self.batch_size, self.device),
                         graph_capturable=self.graph_capturable)
        sphere_radius = torch.tensor(self.collision_sphere_radii, device=self.device).repeat(self.batch_size, 1)
        self.add_fabric("body_points", "repulsion",
                        BodySphereRepulsion(True, params, self.batch_size, sphere_radius, collision_matrix,
                                            self.device, graph_capturable=self.graph_capturable))
        self.add_fabric("body_points", "geom_repulsion",
                        BodySphereRepulsion(False, params, self.batch_size, sphere_radius, collision_matrix,
                                            self.device, graph_capturable=self.graph_capturable))
        self.base_fabric_repulsion = BaseFabricRepulsion(params, self.batch_size, sphere_radius,
                                                         collision_matrix, self.device)

    def add_cspace_energy(self):
        self.add_energy("identity", "euclidean", EuclideanEnergy(self.batch_size, self._num_joints, self.device))

    def construct_fabric(self):
        self.add_joint_limit_repulsion()
        self.add_cspace_attractor()
        self.add_ee_points_attractor()
        self.add_body_repulsion()
        self.add_cspace_energy()

    # ---------------------------------------------------------------- targets
    def _pose_to_points(self, pose):
        """b x 7 (x, y, z, qx, qy, qz, qw) -> the 7 attractor points, b x 21."""
        b = self.batch_size
        wxyz = torch.cat([pose[:, 6:7], pose[:, 3:6]], dim=1)
        T = self._T
        T[:, :3, :3] = quaternion_to_matrix(wxyz)
        T[:, :3, 3] = pose[:, :3]
        return torch.einsum('bij,pj->bpi', T, self._axis_points_local)[:, :, :3].reshape(b, 21)

    def get_ee_pose(self, cspace_position, hand):
        """TCP pose of ``hand`` for joint positions (b x 12) as b x 7 (x, y, z, qx, qy, qz, qw)."""
        pts, _ = self.get_taskmap(f"ee_{hand}")(cspace_position, None)
        origin = pts[:, :3]
        x_axis = torch.nn.functional.normalize(pts[:, 3:6] - origin, dim=1)
        y_axis = torch.nn.functional.normalize(pts[:, 9:12] - origin, dim=1)
        z_axis = torch.nn.functional.normalize(pts[:, 15:18] - origin, dim=1)
        wxyz = matrix_to_quaternion(torch.stack([x_axis, y_axis, z_axis], dim=2))
        return torch.cat([origin, wxyz[:, 1:4], wxyz[:, 0:1]], dim=-1)

    def get_sphere_radii(self):
        return self.collision_sphere_radii

    @property
    def collision_status(self):
        return self.base_fabric_repulsion.collision_status

    def set_features(self, ee_pose_targets, batched_cspace_position, batched_cspace_velocity,
                     object_ids, object_indicator, cspace_target=None):
        """Feed one control step's inputs to the fabric terms.

        :param ee_pose_targets: dict hand -> b x 7 (x, y, z, qx, qy, qz, qw) TCP target
        :param batched_cspace_position: b x 12 joint position
        :param batched_cspace_velocity: b x 12 joint velocity
        :param object_ids, object_indicator: from WorldMeshesModel.get_object_ids()
        :param cspace_target: optional b x 12 home configuration for the posture attractor
        """
        self.fabrics_features["identity"]["cspace_attractor"] = self.default_config if cspace_target is None else cspace_target
        for h in self.HANDS:
            native = self._pose_to_points(ee_pose_targets[h])
            if self._native_ee_target[h] is None:
                self._native_ee_target[h] = torch.clone(native)
            else:
                self._native_ee_target[h].copy_(native)
            self.fabrics_features[f"ee_{h}"]["ee_attractor"] = self._native_ee_target[h]
            self.get_fabric_term(f"ee_{h}", "ee_attractor").damping_position = self._native_ee_target[h]

        body_pos, jac = self.get_taskmap("body_points")(batched_cspace_position, None)
        body_vel = torch.bmm(jac, batched_cspace_velocity.unsqueeze(2)).squeeze(2)
        self.base_fabric_repulsion.calculate_response(body_pos, body_vel, object_ids, object_indicator)
        self.fabrics_features["body_points"]["repulsion"] = self.base_fabric_repulsion
        self.fabrics_features["body_points"]["geom_repulsion"] = self.base_fabric_repulsion
