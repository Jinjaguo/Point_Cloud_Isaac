import numpy as np
import os
import math
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torch
import matplotlib.pyplot as plt
from abc import abstractmethod

# pytorch3d uses real part first for quaternions
# isaac gym uses real part last for quaternions
from pytorch_kinematics.transforms.rotation_conversions import quaternion_to_matrix
import pathlib
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf

# import pytorch3d.transforms as torch3d_tf

ROOT = pathlib.Path(__file__).resolve().parents[1]
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation as R

import random
import pandas as pd


def quat_change_convention(q, current='xyzw'):
    if current == 'xyzw':
        return torch.stack(
            (q[:, 3], q[:, 0], q[:, 1], q[:, 2]), dim=-1)

    if current == 'wxyz':
        return torch.stack((
            q[:, 1], q[:, 2], q[:, 3], q[:, 0]), dim=-1)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


class AllegroEnv:
    """
    base class for Allegro hand environment
    NOTE: in isaac gym, the orientation of object is represented as quaternion of XYZ W instead of WXYZ
    but pytorch3d and pytorch kinematics uses WXYZ convention
    and scipy transform uses XYZW convention
    """

    def __init__(self, num_envs,
                 hand_p,
                 hand_r,
                 camera_pos,
                 camera_target,
                 steps_per_action=60,
                 control_mode='cartesian_impedance',
                 viewer=False,
                 device='cuda:0',
                 use_cartesian_controller=True,
                 friction_coefficient=1.0,
                 contact_controller=False,
                 video_save_path=None,
                 joint_stiffness=6.0,
                 fingers=['index', 'thumb'],  # order matters, please follow index, middle, ring, thumb
                 gravity=True,
                 gradual_control=False,
                 randomize_obj_start=False,
                 arm_type='None',  # choose between none, 'robot', 'floating_3d'
                 ):
        if arm_type == 'robot':
            urdf = 'xela_models/victor_allegro.urdf'
        elif arm_type == 'None':
            urdf = 'xela_models/allegro_hand_right.urdf'
        elif arm_type == 'floating_3d':
            urdf = 'xela_models/allegro_hand_right_floating_3d.urdf'
        elif arm_type == 'floating_6d':
            urdf = 'xela_models/allegro_hand_right_floating_6d.urdf'
        else:
            raise ValueError('Invalid arm type')
        self.gym = gymapi.acquire_gym()
        self.steps_per_action = steps_per_action
        self.control_mode = control_mode
        self.num_envs = num_envs
        self.device = device
        self.assets = []
        self.use_cartesian_controller = use_cartesian_controller
        self.joint_stiffness = joint_stiffness
        # self.joint_stiffness = 50
        self.fingers = fingers
        self.num_fingers = len(fingers)
        self.friction_coefficient = friction_coefficient
        # the indexing of the joints are ranked alphabetically, rather than the order in the urdf file
        self.asset_root = f'{ROOT}/assets'
        # urdf_fpath = f'{self.asset_root}/{urdf}'

        ###
        self.camera_handles = []  # Here we will store the camera handles for each environment
        ###
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.enable_tensors = True
        self.camera_props.width = 512
        self.camera_props.height = 512
        self.camera_props.near_plane = 0.1
        self.camera_props.far_plane = 10.0

        self.robot_p = hand_p
        self.robot_r = hand_r

        self.randomize_obj_start = randomize_obj_start

        assert control_mode in ['cartesian_impedance', 'joint_torque', 'joint_impedance', 'joint_torque_position']
        self.contact_controller = contact_controller
        if contact_controller and control_mode != 'joint_impedance':
            raise ValueError('Contact controller only works with joint impedance control')
        self.gradual_control = gradual_control
        self.arm_type = arm_type
        if arm_type == 'None':
            self.robot_dof = 4 * self.num_fingers
            self.arm_dof = 0
        elif arm_type == 'robot':
            self.robot_dof = 7 + 4 * self.num_fingers
            self.arm_dof = 7
        elif arm_type == 'floating_3d':
            self.robot_dof = 3 + 4 * self.num_fingers
            self.arm_dof = 3
        elif arm_type == 'floating_6d':
            self.robot_dof = 6 + 4 * self.num_fingers
            self.arm_dof = 6

        # TODO have a config file?
        sim_params = gymapi.SimParams()
        sim_params.dt = 1. / 60
        sim_params.substeps = 1
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 8
        if device == 'cpu':
            sim_params.physx.use_gpu = False
        else:
            sim_params.physx.use_gpu = True
            sim_params.use_gpu_pipeline = True
        if gravity:
            sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        else:
            sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        # self.sim = self.gym.create_sim(int(self.device[-1]), 0, gymapi.SIM_PHYSX, sim_params)

        self.sim = self.gym.create_sim(0,
                                       0, gymapi.SIM_PHYSX, sim_params)

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        # set up the camera
        self.viewer = None
        if viewer:
            viewer_props = gymapi.CameraProperties()
            viewer_props.use_collision_geometry = True
            self.viewer = self.gym.create_viewer(self.sim, viewer_props)
            cam_pos = gymapi.Vec3(camera_pos[0], camera_pos[1], camera_pos[2])
            cam_target = gymapi.Vec3(camera_target[0], camera_target[1], camera_target[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            print(cam_pos, cam_target)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = not gravity
        asset_options.thickness = 0.001
        asset_options.armature = 0.001
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.replace_cylinder_with_capsule = True
        self.asset_options = asset_options

        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_com = False
        asset_options.override_inertia = False
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 10000

        # asset_options.disable_gravity = True
        allegro_asset = self.gym.load_asset(self.sim, self.asset_root, urdf, asset_options)
        # asset_options.disable_gravity = not gravity
        # Get joint limits
        allegro_dof_props = self.gym.get_asset_dof_properties(allegro_asset)

        allegro_lower_limits = allegro_dof_props['lower']
        allegro_upper_limits = allegro_dof_props['upper']
        allegro_ranges = allegro_upper_limits - allegro_lower_limits
        allegro_mids = 0.5 * (allegro_upper_limits + allegro_lower_limits)
        num_dofs = len(allegro_dof_props)
        # set to effort mode
        if (
                control_mode == 'joint_impedance' and not self.use_cartesian_controller) or control_mode == 'joint_torque_position':
            allegro_dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
            allegro_dof_props['stiffness'][:self.arm_dof] = 500 * self.joint_stiffness
            allegro_dof_props['damping'][:self.arm_dof] = 200.0
            allegro_dof_props['stiffness'][self.arm_dof:] = self.joint_stiffness
            allegro_dof_props['damping'][self.arm_dof:] = 1.0
            # else:
            #     allegro_dof_props['stiffness'][:] = self.joint_stiffness  # zero passive stiffness
            #     allegro_dof_props['damping'][:] = 1.0
            self.kp = self.joint_stiffness * torch.eye(num_dofs)
            # self.kp_inv = torch.linalg.inv(self.kp).unsqueeze(0).to(self.device)
        else:
            allegro_dof_props['driveMode'].fill(gymapi.DOF_MODE_EFFORT)
            allegro_dof_props['stiffness'][:] = 0.0  # zero passive stiffness
            allegro_dof_props['damping'][:] = 0.0  # zero passive damping

        # set up the env grid
        spacing = 1.5
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(math.sqrt(num_envs))
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self.robot_p)
        # NOTE: for isaac gym quat, angle goes last, but for pytorch kinematics, angle goes first
        pose.r = gymapi.Quat(*self.robot_r)
        self.world_trans = tf.Transform3d(pos=torch.tensor(self.robot_p, device=self.device),
                                          rot=torch.tensor(
                                              [self.robot_r[3], self.robot_r[0], self.robot_r[1], self.robot_r[2]],
                                              device=self.device), device=self.device)

        self.assets.append(
            {'name': 'allegro',
             'asset': allegro_asset,
             'pose': pose,
             'dof_props': allegro_dof_props
             }
        )
        finger_to_ee_name = {
            'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'thumb': 'allegro_hand_oya_finger_3_aftc_base_link'
        }
        # NOTE: very important, the index is not the same as that in our algorithm. For isaac gym, it orders alphabetically.
        self.finger_to_joint_index = {
            'index': [0, 1, 2, 3],
            'middle': [8, 9, 10, 11],
            'ring': [4, 5, 6, 7],
            'thumb': [12, 13, 14, 15]
        }
        for finger in self.finger_to_joint_index.keys():
            self.finger_to_joint_index[finger] = (np.array(self.finger_to_joint_index[finger]) + self.arm_dof).tolist()
        if self.arm_type == 'robot':
            self.arm_index = [0, 1, 2, 3, 4, 5, 6]
        elif self.arm_type == 'floating_3d':
            self.arm_index = [0, 1, 2]
        elif self.arm_type == 'floating_6d':
            self.arm_index = [0, 1, 2, 3, 4, 5]
        elif self.arm_type == 'None':
            self.arm_index = []

            # self.ee_names = [finger_to_ee_name[f] for f in fingers]
        self.finger_ee_index = {finger: self.gym.find_asset_rigid_body_index(allegro_asset, finger_to_ee_name[finger])
                                for finger in self.fingers}
        self.num_dofs = num_dofs

        self._rb_states, self.rb_states = None, None
        self._actor_rb_states, self.actor_rb_states = None, None
        self._dof_states, self.dof_states = None, None
        self._q, self._qd = None, None
        self._ft_data, self.ft_data = None, None
        self._jacobian, self.jacobian = None, None
        self.J_ee = None
        self._massmatrix, self.M = None, None
        self.default_dof_pos = None

        self.save_image_fpath = None
        self.frame_fpath = video_save_path
        self.frame_id = 0

    def _create_env(self, assets):
        self.envs = []
        self.handles = {}
        for asset in assets:
            self.handles[asset['name']] = []

        spacing = 1.5
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(math.sqrt(self.num_envs))

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            for asset in assets:
                if asset['name'] == 'allegro':
                    allegro_asset = asset
                    assert allegro_asset['name'] == 'allegro'
                    handle = self.gym.create_actor(env, allegro_asset['asset'], allegro_asset['pose'],
                                                   allegro_asset['name'], i,
                                                   0, 0)
                    self.gym.set_actor_dof_properties(env, handle, allegro_asset['dof_props'])
                    self.handles['allegro'].append(handle)
                    allegro_shape_props = self.gym.get_asset_rigid_shape_properties(allegro_asset['asset'])
                    for j in range(len(allegro_shape_props)):
                        allegro_shape_props[j].friction = self.friction_coefficient
                    self.gym.set_actor_rigid_shape_properties(self.envs[i], self.handles['allegro'][0],
                                                              allegro_shape_props)
                elif asset['name'] == 'valve':
                    handle = self.gym.create_actor(env, asset['asset'], asset['pose'], asset['name'], i, 0, 1)
                    free_dofs = [0]
                    dof_props = self.gym.get_actor_dof_properties(env, handle)
                    dof_props['driveMode'][free_dofs] = gymapi.DOF_MODE_NONE
                    dof_props['stiffness'][free_dofs] = 0
                    dof_props['damping'][free_dofs] = 0.5
                    self.gym.set_actor_dof_properties(env, handle, dof_props)
                    self.handles[asset['name']].append(handle)
                elif asset['name'] == 'screwdriver':
                    handle = self.gym.create_actor(env, asset['asset'], asset['pose'], asset['name'], i, 0, 2)
                    # free_dofs = [0, 1, 2, 3, 4, 5, 6]
                    free_dofs = [0, 1, 2, 3]
                    dof_props = self.gym.get_actor_dof_properties(env, handle)
                    dof_props['driveMode'][free_dofs] = gymapi.DOF_MODE_NONE
                    dof_props['stiffness'][free_dofs] = 0
                    # dof_props['damping'][free_dofs] = 0
                    dof_props['damping'][[0, 1]] = 2.5
                    dof_props['damping'][2] = .1
                    self.gym.set_actor_dof_properties(env, handle, dof_props)
                    self.handles[asset['name']].append(handle)
                elif asset['name'] == 'peg' or asset['name'] == 'card':
                    handle = self.gym.create_actor(env, asset['asset'], asset['pose'], asset['name'], i, 0, 3)
                    free_dofs = [0, 1, 2, 3, 4, 5]
                    dof_props = self.gym.get_actor_dof_properties(env, handle)
                    dof_props['driveMode'][free_dofs] = gymapi.DOF_MODE_NONE
                    dof_props['stiffness'][free_dofs] = 0
                    # dof_props['damping'][free_dofs] = 0.3
                    dof_props['damping'][free_dofs] = 0.001
                    self.gym.set_actor_dof_properties(env, handle, dof_props)
                    self.handles[asset['name']].append(handle)
                elif asset['name'] == 'table' or asset['name'] == 'wall':
                    handle = self.gym.create_actor(env, asset['asset'], asset['pose'], asset['name'], i, 0, 4)
                    dof_props = self.gym.get_actor_dof_properties(env, handle)
                    self.gym.set_actor_dof_properties(env, handle, dof_props)
                    self.handles[asset['name']].append(handle)

    def prepare_tensors(self):
        # prepare tensors for GPU usage -- must use tensor API from here on out
        self.gym.prepare_sim(self.sim)

        # state tensor
        self._rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(self._rb_states)
        self._actor_rb_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.actor_rb_states = gymtorch.wrap_tensor(self._actor_rb_states).view(self.num_envs, -1, 13)

        # DOF state tensor
        self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(self._dof_states).view(self.num_envs, -1, 2)
        self._q = self.dof_states[..., 0]
        self._qd = self.dof_states[..., 1]

        # ft sensor
        # self._ft_data = self.gym.acquire_force_sensor_tensor(self.sim)
        # self.ft_data = gymtorch.wrap_tensor(self._ft_data)

        # jacobian
        self._jacobian = self.gym.acquire_jacobian_tensor(self.sim, 'allegro')
        self.jacobian = gymtorch.wrap_tensor(self._jacobian)
        # self._jacobian_thumb = self.jacobian
        # self.J_ee = self.jacobian[:, self.ee_index - 1, :, :7]
        # self.J_ee = torch.cat((self.jacobian[:, self.ee_index - 1, :, :7], self.jacobian[:, self.ee_index - 1, :, 14:18]), dim=-1)

        self._massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, 'allegro')
        self.M = gymtorch.wrap_tensor(self._massmatrix)
        # self.M = self.M[:, :7, :7]

    def reset(self, dof_pos=None):
        num_actors = self.actor_rb_states.shape[1]
        global_indices = torch.arange(self.num_envs * num_actors,
                                      dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        default_dof_pos = torch.zeros_like(self.default_dof_pos)
        default_dof_pos[:, :self.arm_dof] = self.default_dof_pos.clone()[:, :self.arm_dof]
        dof_pos_to_set = dof_pos if dof_pos is not None else self.default_dof_pos
        for i, finger in enumerate(['index', 'middle', 'ring', 'thumb']):
            idx = [self.arm_dof + i * 4, self.arm_dof + i * 4 + 1, self.arm_dof + i * 4 + 2, self.arm_dof + i * 4 + 3]
            if finger not in self.fingers:
                default_dof_pos[:, self.finger_to_joint_index[finger]] = self.default_dof_pos[:, idx]
            else:
                default_dof_pos[:, self.finger_to_joint_index[finger]] = dof_pos_to_set[:, idx]
        default_dof_pos[:, (16 + self.arm_dof):] = dof_pos_to_set[:, (16 + self.arm_dof):]
        self.dof_states[:, :, 0] = default_dof_pos
        self.dof_states[:, :, 1] *= 0

        if self.randomize_obj_start:
            self.dof_states[:, 16:16 + 2, 0] = default_dof_pos[:, 16:16 + 2] + 0.05 * torch.randn_like(
                self.default_dof_pos[:, :16:16 + 2])
            self.dof_states[:, 18, 0] = default_dof_pos[:, 18] + np.pi * 2 * (
                    torch.rand_like(self.default_dof_pos[:, 18]) - 0.5)

        robot_ids = global_indices[:, self.handles['allegro'][0]].contiguous()
        # obj_ids = global_indices[:, self.handles['valve'][0]].contiguous()
        # update

        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))

        # rval = self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                      gymtorch.unwrap_tensor(self.dof_states),
        #                                      gymtorch.unwrap_tensor(robot_ids),
        #                                      self.num_envs
        #                                      )
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(default_dof_pos),
                                                        gymtorch.unwrap_tensor(robot_ids),
                                                        self.num_envs
                                                        )
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(torch.zeros_like(self.default_dof_pos)),
                                                        gymtorch.unwrap_tensor(robot_ids),
                                                        self.num_envs
                                                        )
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)
        # to resolve contact
        for _ in range(64):
            self._step_sim()
        self._refresh_tensors()

    def set_pose(self, pose, semantic_order=True, zero_velocity=True):
        # semantic order: index, middle, ring, thumb. If the input is in this order, we have to swap the order
        # to match that in sim
        if len(pose.shape) == 1:
            assert zero_velocity
            pose = pose.unsqueeze(0)
            pose = pose.unsqueeze(-1)
        if len(pose.shape) == 2:
            pose = pose.unsqueeze(-1)
        if zero_velocity:
            tmp = torch.zeros_like(pose)
            pose = torch.cat((pose, tmp), dim=-1)
        if semantic_order:
            tmp = self.dof_states.clone()
            # print('tmp', tmp.shape)
            # print('pose', pose.shape)
            # print(self.fingers)
            # swap the order to match that in sim
            for i, finger in enumerate(self.fingers):
                idx = [i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3]
                tmp[..., self.finger_to_joint_index[finger], :] = pose[..., idx, :]
            tmp[..., 16:, :] = pose[..., 4 * len(self.fingers):, :]
        else:
            tmp = pose
        assert pose.shape[-1] == 2
        self.dof_states[:, :, 0] = tmp[..., 0]
        self.dof_states[:, :, 1] = tmp[:, :, 1]
        num_actors = self.actor_rb_states.shape[1]
        global_indices = torch.arange(self.num_envs * num_actors,
                                      dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        robot_ids = global_indices[:, self.handles['allegro'][0]].contiguous()

        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.default_dof_pos),
                                                        gymtorch.unwrap_tensor(robot_ids),
                                                        self.num_envs
                                                        )
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(torch.zeros_like(self.default_dof_pos)),
                                                        gymtorch.unwrap_tensor(robot_ids),
                                                        self.num_envs
                                                        )
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

            self.gym.write_viewer_image_to_file(self.viewer, f'{self.frame_fpath}/frame_{self.frame_id:06d}.png')
            self.frame_id += 1

        # self._step_sim()
        self._refresh_tensors()

    def get_sim(self):
        return self.sim, self.gym, self.viewer

    def _step_sim(self):
        # simulation step
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def _refresh_tensors(self):
        # refresh tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

    def single_step(self, actions):
        des_q = None
        torques = None
        if self.control_mode == 'cartesian_impedance':
            target_position = actions[:, :3]
            target_orientation = actions[:, 3:]
            # torques = self._cartesian_impedance_controller(target_position,
            #                                                target_orientation,
            #                                                self._q[:, :7].squeeze(-1).clone())
        elif self.control_mode == 'joint_torque':
            torques = actions

        elif self.control_mode == 'joint_impedance':
            if self.use_cartesian_controller:
                m = self.chain.forward_kinematics(actions)
                target_position, target_orientation = m[:, :3, 3], pk.matrix_to_quaternion(m[:, :3, :3])
                target_orientation = quat_change_convention(target_orientation, 'wxyz')
                # torques = self._cartesian_impedance_controller(target_position, target_orientation, actions)
            else:
                des_q = self.default_dof_pos[:, :(self.arm_dof + 16)].clone().float()
                tmp = des_q.clone()[:, (4 + self.arm_dof):(8 + self.arm_dof)]
                des_q[:, (4 + self.arm_dof):(8 + self.arm_dof)] = des_q[:, (8 + self.arm_dof):(12 + self.arm_dof)]
                des_q[:, (8 + self.arm_dof):(12 + self.arm_dof)] = tmp
                for i, finger in enumerate(self.fingers):
                    des_q[:, self.finger_to_joint_index[finger]] = actions[:,
                                                                   self.arm_dof + i * 4: self.arm_dof + (i + 1) * 4]
                # add palm movement
                des_q[:, :self.arm_dof] = actions[:, :self.arm_dof]
                des_q = torch.cat((des_q, self._q[:, (16 + self.arm_dof):]), dim=-1)
                if self.contact_controller:
                    # TODO hardcoded for now
                    sin_valve = torch.sin(self._q[:, -1] + np.pi / 2)
                    cos_valve = torch.cos(self._q[:, -1] + np.pi / 2)
                    thumb_normal_vector = -torch.stack(
                        (sin_valve, torch.zeros_like(sin_valve), cos_valve), dim=-1)
                    index_normal_vector = thumb_normal_vector

                    normal_vectors = torch.stack((thumb_normal_vector, index_normal_vector), dim=1)

                    new_torque = self._contact_controller(des_q - self._q, normal_vectors)

                    print('old des q', des_q)

                    des_q[:, :16] = self._q[:, :16] + new_torque / self.joint_stiffness
                    print('new des q', des_q)

        # elif self.control_mode == 'joint_torque_position':
        #     "it takes in the desired torque and transform it into position control"
        #     zero_joint_q = torch.zeros((1, 8)).float().to(self.device)
        #     delta_q = self.kp_inv @ torch.cat((actions[:, :4], zero_joint_q, actions[:, 4:8]), dim=-1).unsqueeze(-1)
        #     delta_q = delta_q.squeeze(-1)
        #     current_q = self._q[:, :16]
        #     des_q = current_q + delta_q
        #     des_q = torch.cat((des_q, self._q[:, 16:]), dim=-1)

        if torques is not None:
            torques = torch.zeros((self.num_envs, 16)).float().to(self.device)
            for i, finger in enumerate(self.fingers):
                torques[:, self.finger_to_joint_index[finger]] = actions[:, i * 4:(i + 1) * 4]
        # apply action
        if torques is not None:
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
        else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(des_q))

        self._step_sim()
        self._refresh_tensors()

        # update viewer
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

    def step(self, actions, ignore_img=False):
        if self.gradual_control:
            state = self.get_state()
            robot_q = state['q'][:, :self.robot_dof]
        for i in range(self.steps_per_action):
            if self.gradual_control:
                if i < self.steps_per_action * 0.75:
                    temp_action = (i + 1) / (self.steps_per_action * 0.75) * (actions - robot_q) + robot_q
                else:
                    temp_action = actions
                self.single_step(temp_action)
            else:
                self.single_step(actions)
            if self.frame_fpath is not None and i % 20 == 0:
                if not ignore_img:
                    self.gym.write_viewer_image_to_file(self.viewer,
                                                        f'{self.frame_fpath}/frame_{self.frame_id:06d}.png')
                    self.frame_id += 1
        # contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        # breakpoint()

        # J = self.chain.jacobian(self._q[:, :7].reshape(-1, 7))
        # print(self._q)
        # print(actions - self.get_state())
        return self.get_state()

    def get_state(self):
        arm_q = {'arm_q': self._q[:, self.arm_index]}
        finger_q = {finger + '_q': self._q[:, self.finger_to_joint_index[finger]] for finger in self.fingers}
        finger_ee_pos = {finger + '_pos': self.rb_states[self.finger_ee_index[finger], :3] for finger in self.fingers}
        if self.arm_type != 'None':
            results = {**finger_q, **finger_ee_pos, **arm_q}
        else:
            results = {**finger_q, **finger_ee_pos}
        return results

    def _get_analytic_J(self):
        J_geometric = self.J_ee
        N = J_geometric.shape[0]
        ee_ori_q = self.rb_states[self.ee_indices, 3:7]
        ee_R_mat = quaternion_to_matrix(quat_change_convention(ee_ori_q, current='xyzw')).transpose(1, 2)
        I = torch.eye(3).to(ee_R_mat).reshape(-1, 3, 3).repeat(N, 1, 1)
        T1 = torch.cat((I, torch.zeros_like(I)), dim=2)
        T2 = torch.cat((torch.zeros_like(I), ee_R_mat), dim=2)
        T = torch.cat((T1, T2), dim=1)
        return T @ J_geometric

    def _get_transform_matrix(self, pos, quat):
        mat = quaternion_to_matrix(quat_change_convention(quat, current='xyzw'))
        mat = torch.cat((mat, pos.reshape(-1, 3, 1)), dim=2)
        ones = torch.tensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 1, 4).repeat(mat.shape[0], 1, 1)
        mat = torch.cat((mat, ones.to(device=self.device)), dim=1)
        return mat

    def _contact_controller(self, delta_q, normal_vectors):
        """

        :param delta_q:
        :param normal_vectors: (B x num_contacts x 3)
        :return:
        """
        # TODO: deprecated, needs to be updated with multiple fingers
        B, num_fingers, _ = normal_vectors.shape
        if B > 1:
            raise NotImplementedError('Batch size > 1 not supported yet')
        R = get_rotation_from_normal(normal_vectors.reshape(-1, 3)).reshape(B, num_fingers, 3, 3)
        torque = self.joint_stiffness * delta_q[:, :16]

        k = 4
        friction_polytope = self.get_friction_polytope(k=k, mu=0.3).float()

        # Solve a QP with primal variables of both force and end-effector torque
        # x is [tau, f] f in world frame

        # Torque limits
        upper_torque = 10 * np.ones(16)
        lower_torque = -10 * np.ones(16)
        A_torque_limits = np.zeros((16, 16 + 3 * num_fingers))
        A_torque_limits[:, :16] = np.eye(16)

        # Friction cone constraints
        upper_force = np.zeros(num_fingers * k)
        lower_force = -np.inf * np.ones(num_fingers * k)
        A_friction = np.zeros((num_fingers * k, 16 + 3 * num_fingers))
        A_friction[:k, 16:16 + 3] = (friction_polytope @ R[:, 0]).squeeze(-1).cpu().numpy()
        A_friction[k:2 * k, 16 + 3:16 + 6] = (friction_polytope @ R[:, 1]).squeeze(-1).cpu().numpy()

        # Relationship between force and torque
        upper_force_torque = np.zeros(num_fingers * 3)
        lower_force_torque = np.zeros(num_fingers * 3)
        A_force_torque = np.zeros((3 * num_fingers, 16 + 3 * num_fingers))
        A_force_torque[:, :16] = np.concatenate(
            (np.linalg.pinv(self.jacobian[0, self.thumb_index - 1, :3, :16].cpu().numpy().T),
             np.linalg.pinv(self.jacobian[0, self.index_index - 1, :3, :16].cpu().numpy()).T), axis=0)
        A_force_torque[:, 16:] = -np.eye(3 * num_fingers)

        # combinr
        u_total = np.concatenate((upper_torque, upper_force, upper_force_torque))
        l_total = np.concatenate((lower_torque, lower_force, lower_force_torque))
        A_total = np.concatenate((A_torque_limits, A_friction, A_force_torque), axis=0)

        A = sparse.csc_matrix(A_total)
        # minimize difference to nominal torque
        P = np.zeros((16 + 3 * num_fingers, 16 + 3 * num_fingers))
        P[:16, :16] = np.eye(16)
        P = sparse.csc_matrix(P)
        q = -np.concatenate((torque.cpu().numpy().reshape(-1), np.zeros(num_fingers * 3)))
        prob = osqp.OSQP()
        prob.setup(P=P, q=q, u=u_total, l=l_total, A=A, verbose=True)
        res = prob.solve()

        new_force_torque = res.x

        new_torque = new_force_torque[:16]
        new_force = new_force_torque[16:].reshape(num_fingers, 3)
        tmp = self.jacobian[0, self.thumb_index - 1, :3, :16].cpu().numpy().T @ new_force[0]
        tmp += self.jacobian[0, self.index_index - 1, :3, :16].cpu().numpy().T @ new_force[1]
        new_torque = torch.from_numpy(new_torque).reshape(-1, 16).to(torque)
        return new_torque

    def get_friction_polytope(self, k, mu):
        """
        :param k: the number of faces of the friction cone
        :return: a list of normal vectors of the faces of the friction cone
        """

        normal_vectors = []
        for i in range(k):
            theta = 2 * np.pi * i / k
            normal_vector = torch.tensor([np.cos(theta), np.sin(theta), -mu]).to(device=self.device)
            normal_vectors.append(normal_vector)
        normal_vectors = torch.stack(normal_vectors, dim=0)
        return normal_vectors

    def initialize_cameras(self):
        for env in self.envs:
            camera_handle = self.gym.create_camera_sensor(env, self.camera_props)
            self.camera_handles.append(camera_handle)
            self.gym.set_light_parameters(self.sim, 1, gymapi.Vec3(1, 1, 1), gymapi.Vec3(.3, .3, .3),
                                          gymapi.Vec3(0, 0, 0))
        print(f"Initialized {len(self.camera_handles)} cameras.")
        print(f"Camera properties: Resolution {self.camera_props.width}x{self.camera_props.height}, "
              f"Near plane: {self.camera_props.near_plane}, Far plane: {self.camera_props.far_plane}")

    def get_new_folder(self, base_path):
        """ get a new folder name with the format """
        os.makedirs(base_path, exist_ok=True)
        existing_folders = [f for f in os.listdir(base_path) if f.startswith("run_")]

        if existing_folders:
            existing_indices = [int(f.split("_")[-1]) for f in existing_folders if f.split("_")[-1].isdigit()]
            new_index = max(existing_indices, default=0) + 1
        else:
            new_index = 1

        new_folder = os.path.join(base_path, f"run_{new_index}")
        os.makedirs(new_folder, exist_ok=True)
        return new_folder

    def get_depth_image(self, env_index=0):
        """
        get the depth image of the specified environment
        Args:
            env_index (int): index of the environment to get the depth image from.
        Returns:
            depth_image (numpy.ndarray): data of the depth image.
        """
        self.camera_props.enable_tensors = True

        if env_index >= len(self.envs) or env_index >= len(self.camera_handles):
            raise ValueError(f"Invalid env_index {env_index}. Must be < {len(self.envs)}.")

        camera_handle = self.camera_handles[env_index]

        self.gym.set_camera_location(camera_handle, self.envs[env_index], gymapi.Vec3(-0.3, 0.4, 1.48),
                                     gymapi.Vec3(0.0, 0.0, 1.405))

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # import datetime
        # timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        # filename = f"color_pic_{timestamp}.png"
        # self.gym.write_camera_image_to_file(self.sim, self.envs[env_index], camera_handle, gymapi.IMAGE_COLOR, f"{save_dir}/{filename}")

        # get depth image
        camera_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim,
            self.envs[env_index],
            camera_handle,
            gymapi.IMAGE_DEPTH)
        raw_depth_tensor = gymtorch.wrap_tensor(camera_tensor)
        depth_tensor_for_pointcloud = raw_depth_tensor.clone()
        # print(f"Depth tensor shape: {depth_tensor_for_pointcloud.shape}")

        seg_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim,
            self.envs[env_index],
            camera_handle,
            gymapi.IMAGE_SEGMENTATION
        )
        seg_tensor = gymtorch.wrap_tensor(seg_tensor)

        return depth_tensor_for_pointcloud, seg_tensor

    def depth_image_to_point_cloud_GPU(
            self,
            env_index,
            camera_tensor,
            segmentation_tensor,
            device,
            depth_bar=1.1,
            screwdriver_id=1
    ):
        width = self.camera_props.width
        height = self.camera_props.height

        depth_buffer = camera_tensor.to(device)
        seg_tensor = segmentation_tensor.to(device)

        camera_handle = self.camera_handles[env_index]
        vinv = torch.tensor(
            self.gym.get_camera_view_matrix(self.sim, self.envs[env_index], camera_handle),
            device=self.device
        )
        vinv = torch.inverse(vinv)
        proj = torch.tensor(
            self.gym.get_camera_proj_matrix(self.sim, self.envs[env_index], camera_handle),
            device=self.device
        )

        fu = 2.0 / proj[0, 0]
        fv = 2.0 / proj[1, 1]

        centerU = width / 2
        centerV = height / 2

        camera_u = torch.arange(0, width, device=self.device)
        camera_v = torch.arange(0, height, device=self.device)
        u, v = torch.meshgrid(camera_v, camera_u, indexing='ij')

        Z = depth_buffer.view(-1)
        X = -(u - centerU).view(-1) / width * Z * fu
        Y = (v - centerV).view(-1) / height * Z * fv

        ones = torch.ones_like(X)
        position = torch.stack([X, Y, Z, ones], dim=-1)

        position = (position @ vinv.T)
        points_all = position[:, :3]

        valid_depth = (Z > -depth_bar) & (Z < 0)
        seg_tensor_1d = seg_tensor.view(-1)
        mask_screwdriver = (seg_tensor_1d == 2)

        combined_mask = valid_depth & mask_screwdriver

        points_screwdriver = points_all[combined_mask]

        return points_screwdriver

    def visualize_point_cloud_as_spheres(self, env, points, prefix="cloud_sphere"):
        """
        在 Isaac Gym 中，把 points (N,3) 中的每个点都可视化为一个 sphere actor。
        - gym: gymapi.Gym 实例
        - sim: 仿真对象
        - env_handle: 指定要放在哪个 environment
        - sphere_asset: 小球的 asset
        - points: (N,3) 的点云
        - prefix: 给 actor 命名时的前缀
        """
        radius = 0.02
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        sphere_asset = self.gym.create_sphere(self.sim, radius, asset_options)

        max_num = 500
        N = points.shape[0]
        if N <= max_num:
            return points
        idx = np.random.choice(N, max_num, replace=False)
        sample_points = points[idx]
        group_id = 0
        for i, pt in enumerate(sample_points):
            x, y, z = pt
            # 每个小球 actor 的初始变换
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(x, y, z)

            actor_name = f"{prefix}_{i}"
            self.gym.create_actor(env, sphere_asset, pose, actor_name, group_id, 1)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def save_point_clouds(self, points, save_dir):
        import open3d as o3d
        import os
        import datetime
        pcd = o3d.geometry.PointCloud()
        points_np = points.detach().cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)

        timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        filename = f"pointcloud_{timestamp}.ply"
        save_path = os.path.join(save_dir, filename)
        o3d.io.write_point_cloud(save_path, pcd)
        # print(f"Saved point cloud to: {save_path}")

        return pcd

    def get_pose_of_screwdriver(self, env_index=0):
        body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        body_states_tensor = gymtorch.wrap_tensor(body_states)
        screwdriver_index = self.gym.find_actor_rigid_body_index(self.envs[env_index], self.handles['screwdriver'][0],
                                                                 "screwdriver", gymapi.DOMAIN_SIM)
        screwdriver_pos = body_states_tensor[screwdriver_index, :3]
        print(f"Screwdriver position: {screwdriver_pos}")
        return screwdriver_pos

    def save_to_csv(self, poses, oris, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        pose = np.array(poses)  # (num_samples, 3)
        ori = np.array(oris)  # (num_samples, 3)

        data = np.hstack((pose, ori))
        df = pd.DataFrame(data, columns=["x", "y", "z", "roll", "pitch", "yaw"])

        save_path = os.path.join(save_dir, "pose_ori.csv")
        df.to_csv(save_path, index=False)
        print(f"Saved data to: {save_path}")

    def set_screwdriver_pose(self, T_icp, env_idx=0, q_guess=None):
        """
        根据外部观测(ICP)的 4x4 变换矩阵 T_icp，使用逆变换变为欧拉角度，
        并返回与原先 [roll, pitch, yaw, screwdriver_angle] 风格一致的 new_pose。
        """
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        import scipy.optimize as opt
        import torch

        rot_matrix = T_icp[:3, :3]
        rot_matrix_copy = np.copy(rot_matrix)
        r = R.from_matrix(rot_matrix_copy)
        euler_angles = r.as_euler('xyz', degrees=False)
        '''
        # 1) 关节上下限 (根据 URDF limit)
        q_guess = np.array([0.00160723,  0.01335732, -0.07890692])  # 初始值

        # 2) 关节上下限 (根据 URDF limit)
        bounds = [(-1.57, 1.57),  # q1 -> joint_1, axis x
                  (-1.57, 1.57),  # q2 -> joint_2, axis y
                  (-3.14, 3.14)]  # q3 -> joint_3, axis z


        # === 需要先定义自己的 FK 和 cost_func ===#
        def fk_screwdriver(q):
            """
            简化演示版：给定4个角度(弧度)，返回 base->cap 的 4x4 变换矩阵
            具体实现要和你的 URDF 关节顺序对应
            """
            from math import sin, cos
            Rx = np.array([[1, 0, 0, 0],
                           [0, cos(q[0]), -sin(q[0]), 0],
                           [0, sin(q[0]), cos(q[0]), 0],
                           [0, 0, 0, 1]], dtype=np.float32)

            Ry = np.array([[cos(q[1]), 0, sin(q[1]), 0],
                           [0, 1, 0, 0],
                           [-sin(q[1]), 0, cos(q[1]), 0],
                           [0, 0, 0, 1]], dtype=np.float32)

            Rz = np.array([[cos(q[2]), -sin(q[2]), 0, 0],
                           [sin(q[2]), cos(q[2]), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

            # 固定关节 stick->body (xyz=0,0,0.1)
            T_sb = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0.1],
                             [0, 0, 0, 1]], dtype=np.float32)

            T_bc = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0.1],
                             [0, 0, 0, 1]], dtype=np.float32)
            Rz4 = np.array([[cos(q[3]), -sin(q[3]), 0, 0],
                            [sin(q[3]), cos(q[3]), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)
            # T_bc = T_bc @ Rz4

            # body_cap_joint: 先平移(0,0,0.1)，再绕z(q[3])

            T_base_cap = Rx @ Ry @ Rz
            return T_base_cap

        def pose_error(q, T_target):
            """
            计算(q1,q2,q3,q4)与目标T_target的位姿误差，用于数值优化
            """
            from scipy.spatial.transform import Rotation as R
            T = fk_screwdriver(q)
            # 平移误差
            trans_err = T[:3, 3] - T_target[:3, 3]
            # 旋转误差(用旋转向量差)
            R_current = R.from_matrix(T[:3, :3])
            R_target = R.from_matrix(T_target[:3, :3].copy())
            rot_err = R_current.as_rotvec() - R_target.as_rotvec()
            return np.concatenate([trans_err, rot_err])

        def cost_func(q):
            err = pose_error(q, T_icp)
            error1 = (err**2).sum()
            error2 = 50 * (q[:3]**2).sum()
            return error1 + error2

        # === 数值优化 ===#
        res = opt.minimize(cost_func, q_guess, method='SLSQP', bounds=bounds)
        q_sol = res.x  # 得到 [q1, q2, q3]
        '''

        # 1) interpret q1=roll, q2=pitch, q3=yaw
        roll = euler_angles[0]
        pitch = euler_angles[1]
        yaw = euler_angles[2]

        # 2) 做成 torch 的 (1,3) 和 (1,1)
        screwdriver_ori_euler_np = np.array([roll, pitch, yaw])
        # screwdriver_angle_np = np.array([cap_angle])
        screwdriver_ori_euler = torch.tensor(screwdriver_ori_euler_np, device=self.device,
                                             dtype=torch.float).reshape(1, 3)
        # screwdriver_angle = torch.tensor(screwdriver_angle_np, device=self.device, dtype=torch.float).reshape(1, 1)

        # 3) 最终拼成 shape=(1,4)，比如 [roll, pitch, yaw, capAngle]
        # new_pose = torch.cat([screwdriver_ori_euler, screwdriver_angle], dim=-1)  # (1,4)
        new_pose = torch.cat([screwdriver_ori_euler], dim=-1)
        print('-------------- observation pose -------------------')
        # print(f'observation screwdriver pose: rotation={screwdriver_ori_euler_np}, yaw={screwdriver_angle_np}')
        print(f'observation screwdriver pose: rotation={screwdriver_ori_euler_np}')

        return new_pose

    def update_pose_pcd(self):

        base_points_path = "./pointclouds"
        # base_pics_path = "./pics"
        # pics_path = self.get_new_folder(base_pics_path)
        # points_path = self.get_new_folder(base_points_path)

        depth_tensor, mask_tensor = self.get_depth_image()

        if depth_tensor is not None:
            # print('successfully get the depth image')
            points = self.depth_image_to_point_cloud_GPU(0, depth_tensor, mask_tensor, device='cuda:0')
            pcd = self.save_point_clouds(points, base_points_path)

            # segmentation
            from segmentation_pc import process_one_pcd
            screwdriver_pcd_np = process_one_pcd(pcd)
            point_cloud = screwdriver_pcd_np
            # self.visualize_point_cloud_as_spheres(self.envs[0], point_cloud)

            # registration
            from PointsRegistration import points_registration
            import open3d as o3d
            reg = points_registration()
            # point_cloud = reg.add_noise_to_ply(screwdriver_pcd_np)

            sample_points = o3d.io.read_point_cloud('screwdriver_pcd.ply')
            pc = np.asarray(sample_points.points)

            T_icp = reg.get_pose_estimation(point_cloud, pc)

            T_delta = np.array(
                [[-0.42959849999999999426, 0.45388388000000001732, 0.78336249000000002241, 0.95050291999999991788],
                 [-0.69028433500000008216, -0.72216643499999999545, 0.04426025499999999835, -0.44512141999999998987],
                 [0.58580586999999995079, -0.52198107500000001657, 0.61994334500000003452, 0.66040783499999999862],
                 [0, 0, 0, 1]])
            T_icp = T_icp @ T_delta

            # print(T_icp)
            new_pose = self.set_screwdriver_pose(T_icp, env_idx=0)
            new_pose = new_pose.unsqueeze(0)

            print('--------------using observation point cloud as input--------------------')

        return new_pose


class AllegroValveTurningEnv(AllegroEnv):
    """In this environment, we assume we only have access to two fingers, and we want to turn a cuboid valve"""

    def __init__(self, num_envs,
                 steps_per_action=60,
                 control_mode='cartesian_impedance',
                 viewer=False,
                 device='cuda:0',
                 use_cartesian_controller=True,
                 friction_coefficient=1.0,
                 contact_controller=False,
                 valve='cylinder',
                 video_save_path=None,
                 joint_stiffness=6.0,
                 random_robot_pose=False,
                 fingers=['index', 'thumb'],  # order matters, please follow index, middle, ring, thumb,
                 gravity=False
                 ):
        self.random_robot_pose = random_robot_pose
        cam_pos = [0.8, 0.2, 1.48]
        cam_target = [0.85, 0.60, 1.405]
        p = np.array([0.936, 0.6, 1.381]).astype(np.float32)
        if random_robot_pose:
            self.random_bias = np.random.uniform(-0.02, 0.02, size=3).astype(np.float32)
            p += self.random_bias
        r = [-0.0174524, 0, 0.9998477, 0]
        print("robot pose", p, r)
        # previous set up
        # p = [0.89, 0.45, 1.375]
        # r = [0.2425619, 0.2423688, 0.6639723, 0.6645012]
        # cam_pos = gymapi.Vec3(0.8, 0.75, 1.6)
        super(AllegroValveTurningEnv, self).__init__(num_envs, hand_p=p, hand_r=r, camera_pos=cam_pos,
                                                     camera_target=cam_target, steps_per_action=steps_per_action,
                                                     control_mode=control_mode, viewer=viewer, device=device,
                                                     use_cartesian_controller=use_cartesian_controller,
                                                     friction_coefficient=friction_coefficient,
                                                     contact_controller=contact_controller,
                                                     video_save_path=video_save_path, joint_stiffness=joint_stiffness,
                                                     fingers=fingers, gravity=gravity)

        # load valve
        valve_pose = gymapi.Transform()
        valve_pose.p = gymapi.Vec3(0.85, 0.75, 1.405)
        self.valve_pose = np.array([0.85, 0.75, 1.405])

        if valve == 'cylinder':
            valve_urdf = 'valve/valve_cylinder.urdf'
        elif valve == 'cuboid':
            valve_urdf = 'valve/valve_cuboid.urdf'
        elif valve == 'screwdriver':
            valve_urdf = 'screwdriver/screwdriver.urdf'
        valve_asset = self.gym.load_asset(self.sim, self.asset_root, valve_urdf, self.asset_options)
        valve_shape_props = self.gym.get_asset_rigid_shape_properties(valve_asset)
        for i in range(len(valve_shape_props)):
            valve_shape_props[i].friction = friction_coefficient
        self.assets.append(

            {'name': 'valve',
             'asset': valve_asset,
             'pose': valve_pose,
             }
        )

        self._create_env(self.assets)
        self.gym.set_actor_rigid_shape_properties(self.envs[0], self.handles['valve'][0], valve_shape_props)
        self.prepare_tensors()
        # self.default_dof_pos = torch.zeros((1,16)).float().to(device=self.device)

        # previous initial pose
        # self.default_dof_pos = torch.cat((torch.tensor([[-0.2, 0.85, 0.9, 0.1]]).float().to(device=self.device),
        #                                 torch.zeros((1, 8)).float().to(device=self.device),
        #                                 torch.tensor([[.5, 0.7, 0.7, 0.4]]).float().to(device=self.device)),
        #                                 dim=1).to(self.device)
        # NOTE: it's in the order of index, ring, middle ,thumb
        self.default_dof_pos = torch.cat((torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=self.device),
                                          torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=self.device),
                                          torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=self.device),
                                          torch.tensor([[1.3, 0.1, -0.1, 1.0]]).float().to(device=self.device)),
                                         dim=1).to(self.device)
        # add the valve angle to it
        self.default_dof_pos = torch.cat((self.default_dof_pos, torch.zeros((1, 1)).float().to(device=self.device)),
                                         dim=1).to(self.device)
        self.default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.reset()

    def get_state(self):
        results = super(AllegroValveTurningEnv, self).get_state()
        results['valve'] = self._q[:, -1:]
        q = []
        for finger in self.fingers:
            q.append(results[f'{finger}_q'])
        q.append(results['valve'])
        q = torch.cat(q, dim=1)
        results['q'] = q
        return results

    def get_valve_inertia(self):
        valve_handle = self.handles['valve'][0]
        properties = self.gym.get_actor_rigid_body_properties(self.envs[0], valve_handle)
        inertia = properties[1].inertia  # Mat33 object
        return inertia


class AllegroScrewdriverTurningEnv(AllegroEnv):
    def __init__(self, num_envs,
                 steps_per_action=60,
                 control_mode='cartesian_impedance',
                 viewer=False,
                 device='cuda:0',
                 use_cartesian_controller=True,
                 friction_coefficient=1.0,
                 contact_controller=False,
                 video_save_path=None,
                 joint_stiffness=6.0,
                 fingers=['index', 'thumb'],  # order matters, please follow index, middle, ring, thumb
                 table_pose=None,
                 gradual_control=False,
                 gravity=False,
                 randomize_obj_start=False,
                 arm_type='None',
                 ):
        cam_pos = [-0.3, 0.4, 1.48]
        cam_target = [0.0, 0.0, 1.405]

        if arm_type == 'robot':
            p = [-0.8, 0, 0]
            r = [0, 0, 0, 1]
        elif arm_type == 'None' or arm_type == "floating_3d" or arm_type == "floating_6d":
            p = [0, -0.095, 1.33]
            r = [0.2418448, 0.2418448, 0.664463, 0.664463]
        super(AllegroScrewdriverTurningEnv, self).__init__(num_envs, hand_p=p, hand_r=r, camera_pos=cam_pos,
                                                           camera_target=cam_target, steps_per_action=steps_per_action,
                                                           control_mode=control_mode, viewer=viewer, device=device,
                                                           use_cartesian_controller=use_cartesian_controller,
                                                           friction_coefficient=friction_coefficient,
                                                           contact_controller=contact_controller,
                                                           video_save_path=video_save_path,
                                                           joint_stiffness=joint_stiffness, fingers=fingers,
                                                           gradual_control=gradual_control,
                                                           gravity=gravity,
                                                           randomize_obj_start=randomize_obj_start,
                                                           arm_type=arm_type)
        table_pose_tf = gymapi.Transform()
        if table_pose is None:
            table_pose_tf.p = gymapi.Vec3(0, 0, 1.205)
            self.table_pose = np.array([0, 0, 1.205])
        else:
            table_pose_tf.p = gymapi.Vec3(table_pose[0], table_pose[1], table_pose[2])
            self.table_pose = np.array([table_pose[0], table_pose[1], table_pose[2]])

        screwdriver_urdf = 'screwdriver/screwdriver.urdf'
        self.asset_options.replace_cylinder_with_capsule = False
        screwdriver_asset = self.gym.load_asset(self.sim, self.asset_root, screwdriver_urdf, self.asset_options)
        screwdriver_shape_props = self.gym.get_asset_rigid_shape_properties(screwdriver_asset)

        for i in range(len(screwdriver_shape_props)):
            screwdriver_shape_props[i].friction = friction_coefficient
        self.assets.append(
            {'name': 'screwdriver',
             'asset': screwdriver_asset,
             'pose': table_pose_tf,
             }
        )

        self._create_env(self.assets)
        self.gym.set_actor_rigid_shape_properties(self.envs[0], self.handles['screwdriver'][0], screwdriver_shape_props)
        self.prepare_tensors()

        self.default_dof_pos = torch.cat((torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float().to(device=self.device),
                                          torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float().to(device=self.device),
                                          torch.tensor([[0., 0.5, 0.65, 0.65]]).float().to(device=self.device),
                                          torch.tensor([[1.2, 0.3, 0.3, 1.2]]).float().to(device=self.device)),
                                         dim=1).to(self.device)
        if self.arm_type != 'None':
            if self.arm_type == 'robot':
                self.arm_default_dof = torch.tensor([[-0.4627, 0.5445, 0.3865, -1.6972, -1.1118, -1.4570, 0.1162]]).to(
                    device=self.device)
            elif self.arm_type == 'floating_3d':
                self.arm_default_dof = torch.tensor([[0.0, 0.0, 0.0]]).to(device=self.device)
            elif self.arm_type == 'floating_6d':
                self.arm_default_dof = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).to(device=self.device)
            self.default_dof_pos = torch.cat((self.arm_default_dof, self.default_dof_pos), dim=1)

        # add the screwdriver angle to it
        self.default_dof_pos = torch.cat((self.default_dof_pos, torch.zeros((1, 4)).float().to(device=self.device)),
                                         dim=1).to(self.device)
        self.default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.reset()
        self.initialize_cameras()

        '''
        print(f"Camera position: {cam_pos}")
        print(f"Camera target: {cam_target}")
        object_pos = np.array([table_pose_tf.p.x, table_pose_tf.p.y, table_pose_tf.p.z])
        camera_pos = np.array(cam_pos)
        distance = np.linalg.norm(object_pos - camera_pos)
        print(f"Distance from camera to object: {distance}")
        screwdriver_shape_props = self.gym.get_asset_rigid_shape_properties(screwdriver_asset)
        for i, shape in enumerate(screwdriver_shape_props):
            print(f"Shape {i} properties:")
            print(f"Friction: {shape.friction}")
            print(f"Restitution: {shape.restitution}")
            print(f"Thickness: {shape.thickness}")
            print(f"Collision type: {shape.collision_type}")
        '''

    def get_state(self):
        results = super(AllegroScrewdriverTurningEnv, self).get_state()

        screwdriver_ori_euler = self._q[:, -4:-1]
        screwdriver_ori_axis_angle = R.from_euler('xyz', screwdriver_ori_euler.cpu().numpy()).as_rotvec()
        screwdriver_ori_axis_angle = torch.tensor(screwdriver_ori_axis_angle).to(device=self.device).float()

        # results['screwdriver_ori_euler'] = screwdriver_ori_euler
        # results['screwdriver_ori_axis_angle'] = screwdriver_ori_axis_angle
        # results['screwdriver_ori'] = screwdriver_ori_euler  # keeps using the euler angle since the pytorch volumetric might have to use it.
        # results['screwdriver_ori'] = screwdriver_ori_axis_angle  # keeps using the euler angle since the pytorch volumetric might have to use it.
        # results['screwdriver_angle'] = self._q[:, -1:]
        # gt_quat = R.from_euler('XYZ', screwdriver_ori_euler).as_quat()
        # temp_euler = torch.stack((screwdriver_ori_euler[:, 2], screwdriver_ori_euler[:, 1], screwdriver_ori_euler[:, 0]), dim=1).double()
        # change the order of the euler angle since the pytorch3d only supports fixed axis euler angle
        # temp1 = torch3d_tf.matrix_to_quaternion(torch3d_tf.euler_angles_to_matrix(screwdriver_ori_euler, 'XYZ'))
        # temp1 = torch.cat((temp1[..., 1:], temp1[...,:1]), dim=-1)
        # print(self.rb_states[-3, 3:7] - temp1)

        results['screwdriver_ori_euler'] = screwdriver_ori_euler
        results['screwdriver_ori_axis_angle'] = screwdriver_ori_axis_angle
        # results['screwdriver_ori'] = screwdriver_ori_euler  # keeps using the euler angle since the pytorch volumetric might have to use it.
        results[
            'screwdriver_ori'] = screwdriver_ori_axis_angle  # keeps using the euler angle since the pytorch volumetric might have to use it.
        results['screwdriver_angle'] = self._q[:, -1:]

        q = []
        if self.arm_type != 'None':
            q.append(results['arm_q'])
        for finger in self.fingers:
            q.append(results[f'{finger}_q'])
        q.append(results['screwdriver_ori'])
        q.append(results['screwdriver_angle'])
        q = torch.cat(q, dim=1)
        results['q'] = q

        return results


class AllegroScrewdriverEnv(AllegroEnv):
    "6D screwdriver environment"

    def __init__(self, num_envs,
                 steps_per_action=60,
                 control_mode='cartesian_impedance',
                 viewer=False,
                 device='cuda:0',
                 use_cartesian_controller=True,
                 friction_coefficient=1.0,
                 contact_controller=False,
                 video_save_path=None,
                 joint_stiffness=6.0,
                 fingers=['index', 'thumb'],  # order matters, please follow index, middle, ring, thumb
                 gravity=False,
                 gradual_control=False,
                 ):
        cam_pos = [-0.3, 0.4, 0.38]
        cam_target = [0.0, 0.0, 0.305]
        # p = [0, -0.1, 1.33]
        # r = [0.2418448, 0.2418448, 0.664463, 0.664463]
        p = [0.01, -0.028, 0.31]
        # r = [0.5, 0.5, 0.5, 0.5]
        r = [-0.5, 0.5, 0.5, 0.5]
        super(AllegroScrewdriverEnv, self).__init__(num_envs, hand_p=p, hand_r=r, camera_pos=cam_pos,
                                                    camera_target=cam_target, steps_per_action=steps_per_action,
                                                    control_mode=control_mode, viewer=viewer, device=device,
                                                    use_cartesian_controller=use_cartesian_controller,
                                                    friction_coefficient=friction_coefficient,
                                                    contact_controller=contact_controller,
                                                    video_save_path=video_save_path, joint_stiffness=joint_stiffness,
                                                    fingers=fingers, gravity=gravity, gradual_control=gradual_control)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0, 0, 0.205)
        self.table_pose = np.array([0, 0, 0.205])

        screwdriver_urdf = 'screwdriver/screwdriver_6d.urdf'
        self.asset_options.replace_cylinder_with_capsule = False
        screwdriver_asset = self.gym.load_asset(self.sim, self.asset_root, screwdriver_urdf, self.asset_options)
        screwdriver_shape_props = self.gym.get_asset_rigid_shape_properties(screwdriver_asset)
        for i in range(len(screwdriver_shape_props)):
            screwdriver_shape_props[i].friction = friction_coefficient
        self.assets.append(
            {'name': 'screwdriver',
             'asset': screwdriver_asset,
             'pose': table_pose,
             }
        )

        self._create_env(self.assets)
        self.gym.set_actor_rigid_shape_properties(self.envs[0], self.handles['screwdriver'][0], screwdriver_shape_props)
        self.prepare_tensors()

        self.default_dof_pos = torch.cat((torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=self.device),
                                          torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=self.device),
                                          torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=self.device),
                                          torch.tensor([[1.3, 0.3, 0.2, 1.1]]).float().to(device=self.device)),
                                         dim=1).to(self.device)
        # add the screwdriver angle to it
        screwdriver_default_pos = torch.tensor([0, 0, 0, 0, -1.57, 0, 0]).float().to(device=self.device)
        self.default_dof_pos = torch.cat((self.default_dof_pos, screwdriver_default_pos.unsqueeze(0)),
                                         dim=1).to(self.device)
        self.default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.reset()

    def get_state(self):
        results = super(AllegroScrewdriverEnv, self).get_state()
        screwdriver_ori_euler = self._q[:, -4:-1]
        screwdriver_position = self._q[:, -7:-4]
        results['screwdriver_ori_euler'] = screwdriver_ori_euler
        results['screwdriver_ori'] = screwdriver_ori_euler
        results['screwdriver_position'] = screwdriver_position
        results['screwdriver_angle'] = self._q[:, -1:]
        # gt_euler = R.from_quat(self.rb_states[-4, 3:7].cpu()).as_euler('XYZ')
        # print(gt_euler, screwdriver_ori_euler)
        q = []
        for finger in self.fingers:
            q.append(results[f'{finger}_q'])
        q.append(results['screwdriver_position'])
        q.append(results['screwdriver_ori'])
        q.append(results['screwdriver_angle'])
        q = torch.cat(q, dim=1)
        results['q'] = q
        return results


class AllegroPegInsertionEnv(AllegroEnv):
    def __init__(self, num_envs,
                 steps_per_action=60,
                 control_mode='cartesian_impedance',
                 viewer=False,
                 device='cuda:0',
                 use_cartesian_controller=True,
                 friction_coefficient=1.0,
                 contact_controller=False,
                 video_save_path=None,
                 joint_stiffness=6.0,
                 fingers=['index', 'thumb'],  # order matters, please follow index, middle, ring, thumb
                 gradual_control=False,
                 ):
        cam_pos = [-0.4, 0.4, 0.48]
        cam_target = [0.0, 0.0, 0.305]
        # p = [0.01, 0.1, 0.34]
        # r = [0.6532815, -0.2705981, -0.6532815, 0.2705981 ]
        p = [0.11, -0.023, 0.30]
        r = [0, 0.4226183, 0.9063078, 0]
        super(AllegroPegInsertionEnv, self).__init__(num_envs, hand_p=p, hand_r=r, camera_pos=cam_pos,
                                                     camera_target=cam_target, steps_per_action=steps_per_action,
                                                     control_mode=control_mode, viewer=viewer, device=device,
                                                     use_cartesian_controller=use_cartesian_controller,
                                                     friction_coefficient=friction_coefficient,
                                                     contact_controller=contact_controller,
                                                     video_save_path=video_save_path, joint_stiffness=joint_stiffness,
                                                     fingers=fingers, gradual_control=gradual_control)
        peg_pose = gymapi.Transform()
        peg_pose.p = gymapi.Vec3(0, 0, 0.205)
        self.peg_pose = np.array([0, 0, 0.205])

        peg_urdf = 'peg_insertion/peg.urdf'
        self.asset_options.replace_cylinder_with_capsule = True
        peg_asset = self.gym.load_asset(self.sim, self.asset_root, peg_urdf, self.asset_options)
        peg_shape_props = self.gym.get_asset_rigid_shape_properties(peg_asset)
        for i in range(len(peg_shape_props)):
            peg_shape_props[i].friction = friction_coefficient
        self.assets.append(
            {'name': 'peg',
             'asset': peg_asset,
             'pose': peg_pose,
             }
        )

        # create table
        table_dims = gymapi.Vec3(0.4, 1.0, 0.05)
        table_pose = gymapi.Transform()
        self.table_pose = np.array([0, 0, 0.105])
        table_pose.p = gymapi.Vec3(*self.table_pose)
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, self.asset_options)

        table_shape_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        for i in range(len(table_shape_props)):
            table_shape_props[i].friction = 100
        self.assets.append(
            {'name': 'table',
             'asset': table_asset,
             'pose': table_pose,
             }
        )

        # create wall
        self.wall_dims = np.array([0.1, 0.5, 0.12])
        wall_dims = gymapi.Vec3(*self.wall_dims)
        wall_pose = gymapi.Transform()
        self.wall_pose = self.table_pose + np.array([0, -0.25, 0.085])
        wall_pose.p = gymapi.Vec3(self.wall_pose[0], self.wall_pose[1], self.wall_pose[2])
        wall_asset = self.gym.create_box(self.sim, wall_dims.x, wall_dims.y, wall_dims.z, self.asset_options)

        # wall_shape_props = self.gym.get_asset_rigid_shape_properties(wall_asset)
        # for i in range(len(wall_shape_props)):
        #     wall_shape_props[i].friction = friction_coefficient
        self.assets.append(
            {'name': 'wall',
             'asset': wall_asset,
             'pose': wall_pose,
             }
        )

        self._create_env(self.assets)
        self.gym.set_actor_rigid_shape_properties(self.envs[0], self.handles['peg'][0], peg_shape_props)
        # self.gym.set_actor_rigid_shape_properties(self.envs[0], self.handles['table'][0], table_shape_props)
        # self.gym.set_actor_rigid_shape_properties(self.envs[0], self.handles['wall'][0], wall_shape_props)

        self.prepare_tensors()

        # self.default_dof_pos = torch.cat((torch.tensor([[0, 0.7, 0.8, 0.6]]).float().to(device=self.device),
        #                             torch.tensor([[0, 0.7, 0.8, 0.6]]).float().to(device=self.device),
        #                             torch.tensor([[0, 0.3, 0.3, 0.6]]).float().to(device=self.device),
        #                             torch.tensor([[1.2, 0.3, 0.0, 1.1]]).float().to(device=self.device)),
        #                             dim=1).to(self.device)

        # 3 cm peg
        # self.default_dof_pos = torch.cat((torch.tensor([[0, 0.4, 0.9, 0.9]]).float().to(device=self.device),
        #                             torch.tensor([[0, 0.7, 0.7, 0.6]]).float().to(device=self.device),
        #                             torch.tensor([[0, 0.3, 0.3, 0.6]]).float().to(device=self.device),
        #                             torch.tensor([[1.2, 0.3, -0.1, 1.1]]).float().to(device=self.device)),
        #                             dim=1).to(self.device)

        self.default_dof_pos = torch.cat((torch.tensor([[0, 0.7, 0.8, 0.8]]).float().to(device=self.device),
                                          torch.tensor([[0, 0.8, 0.7, 0.6]]).float().to(device=self.device),
                                          torch.tensor([[0, 0.3, 0.3, 0.6]]).float().to(device=self.device),
                                          torch.tensor([[1.2, 0.3, 0.0, 1.1]]).float().to(device=self.device)),
                                         dim=1).to(self.device)

        # add the screwdriver angle to it
        self.default_dof_pos = torch.cat(
            (self.default_dof_pos, torch.tensor([[0, 0, 0.08, 0.67, 0, 0]]).float().to(device=self.device)),
            dim=1).to(self.device)
        self.default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.reset()

    def get_state(self):
        results = super(AllegroPegInsertionEnv, self).get_state()
        peg_ori_euler = self._q[:, -3:]
        peg_position = self._q[:, -6:-3]
        results['peg_ori'] = peg_ori_euler
        results['peg_position'] = peg_position
        # gt_euler = R.from_quat(self.rb_states[-4, 3:7].cpu()).as_euler('XYZ')
        # print(gt_euler, peg_ori_euler)
        q = []
        for finger in self.fingers:
            q.append(results[f'{finger}_q'])
        q.append(results['peg_position'])
        q.append(results['peg_ori'])
        q = torch.cat(q, dim=1)
        results['q'] = q
        return results

    def set_table_pose(self, handles, object_state):
        self._refresh_tensors()
        # index = self.gym.find_actor_rigid_body_handle(self.envs[0], handles, 'box')
        # assert index != -1, 'The object handle is not found'
        # new_rb_states = self.rb_states.clone()
        # new_rb_states[index, :7] = object_state[0]
        # self.gym.set_rigid_body_state_tensor(self.sim, gymtorch.unwrap_tensor(new_rb_states))
        new_rb_states = self.actor_rb_states.clone()
        new_rb_states[:, handles, :7] = object_state
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(new_rb_states))
        self._step_sim()
        self._refresh_tensors()

    def reset(self):
        desired_table_pose = torch.tensor([0, 0, 0.105, 0, 0, 0, 1]).float().to(self.device)
        self.set_table_pose(self.handles['table'][0], desired_table_pose)
        super(AllegroPegInsertionEnv, self).reset()
        self._refresh_tensors()


class AllegroScrewdriverTruningRLEnv(AllegroScrewdriverTurningEnv):
    def __init__(self, num_envs,
                 steps_per_action=60,
                 control_mode='cartesian_impedance',
                 viewer=False,
                 device='cuda:0',
                 use_cartesian_controller=True,
                 friction_coefficient=1.0,
                 contact_controller=False,
                 video_save_path=None,
                 joint_stiffness=6.0,
                 fingers=['index', 'thumb'],  # order matters, please follow index, middle, ring, thumb
                 table_pose=None,
                 gradual_control=False,
                 gravity=False,
                 randomize_obj_start=False,
                 goal=None,
                 nominal_pos=False
                 ):
        super(AllegroScrewdriverTruningRLEnv, self).__init__(num_envs,
                                                             steps_per_action=steps_per_action,
                                                             control_mode=control_mode, viewer=viewer, device=device,
                                                             use_cartesian_controller=use_cartesian_controller,
                                                             friction_coefficient=friction_coefficient,
                                                             contact_controller=contact_controller,
                                                             video_save_path=video_save_path,
                                                             joint_stiffness=joint_stiffness, fingers=fingers,
                                                             table_pose=table_pose, gradual_control=gradual_control,
                                                             gravity=gravity,
                                                             randomize_obj_start=randomize_obj_start)
        self.goal = goal
        self.goal_mat = R.from_euler('xyz', self.goal.numpy()).as_matrix()
        # self.goal_mat = tf.euler_angles_to_matrix(self.goal, 'xyz')
        if nominal_pos:
            nominal_pos_list = []
            for i, finger in enumerate(['index', 'middle', 'ring', 'thumb']):
                if finger in self.fingers:
                    nominal_pos_list.append(self.default_dof_pos[0, i * 4: (i + 1) * 4])
            self.nominal_pos = torch.cat(nominal_pos_list, dim=0)
        else:
            self.nominal_pos = None

    def reward(self, state_dict, action):
        assert len(action.shape) == 2
        reward = 0
        if self.nominal_pos is not None:
            nominal_cost = -torch.norm(action[0] - self.nominal_pos).item()
            reward += 0.05 * nominal_cost
        # goal cost
        screwdriver_state = state_dict['screwdriver_ori']
        screwdriver_mat = R.from_euler('xyz', screwdriver_state).as_matrix()
        distance2goal = tf.so3_relative_angle(torch.tensor(screwdriver_mat), \
                                              torch.tensor(self.goal_mat).unsqueeze(0),
                                              cos_angle=False).detach().cpu().abs()
        reward += -torch.pow(distance2goal, 2).item()

        # upright cost
        upright_cost = (screwdriver_state[0, 0] ** 2 + screwdriver_state[0, 1] ** 2).item()
        reward += -10 * upright_cost
        return reward

    def check_done(self, state_dict):
        screwdriver_state = state_dict['screwdriver_ori']
        screwdriver_mat = R.from_euler('xyz', screwdriver_state).as_matrix()
        distance2goal = tf.so3_relative_angle(torch.tensor(screwdriver_mat), \
                                              torch.tensor(self.goal_mat).unsqueeze(0),
                                              cos_angle=False).detach().cpu().abs()
        distance2goal = distance2goal.item()
        if distance2goal < 0.01:
            return True
        else:
            return False

    def get_rl_state(self, state_dict):
        state_list = []
        for finger in self.fingers:
            state_list.append(state_dict[f'{finger}_q'][0])
        state_list.append(state_dict['screwdriver_ori'][0])
        state = torch.cat(state_list, dim=-1)
        return state

    def step(self, action):
        state_dict = super(AllegroScrewdriverTruningRLEnv, self).step(action)
        state = self.get_rl_state(state_dict)
        reward = self.reward(state_dict, action)
        done = self.check_done(state_dict)
        info = state_dict
        return state, reward, done, info

    def reset(self):
        super(AllegroScrewdriverTruningRLEnv, self).reset()
        state_dict = self.get_state()
        state = self.get_rl_state(state_dict)
        return state


def get_rotation_from_normal(normal_vector):
    """
    :param normal_vector: (batch_size, 3)
    :return: (batch_size, 3, 3) rotation matrix with normal vector as the z-axis
    """
    z_axis = normal_vector / torch.norm(normal_vector, dim=1, keepdim=True)
    y_axis = torch.randn_like(z_axis)
    y_axis = y_axis - torch.sum(y_axis * z_axis, dim=1).unsqueeze(-1) * z_axis
    y_axis = y_axis / torch.norm(y_axis, dim=1, keepdim=True)
    x_axis = torch.cross(y_axis, z_axis)
    x_axis = x_axis / torch.norm(x_axis, dim=1, keepdim=True)
    R = torch.stack((x_axis, y_axis, z_axis), dim=2)
    return R


class AllegroCardSlidingEnv(AllegroEnv):
    def __init__(self, num_envs,
                 steps_per_action=60,
                 control_mode='cartesian_impedance',
                 viewer=False,
                 device='cuda:0',
                 use_cartesian_controller=True,
                 friction_coefficient=1.0,
                 contact_controller=False,
                 video_save_path=None,
                 joint_stiffness=6.0,
                 fingers=['index', 'thumb'],  # order matters, please follow index, middle, ring, thumb
                 gradual_control=False,
                 gravity=False,
                 randomize_obj_start=False
                 ):
        cam_pos = [-0.4, 0.4, 0.48]
        cam_target = [0.0, 0.0, 0.305]
        # p = [0.03, -0.14, 0.20]
        p = [0.03, -0.14, 0.21]
        # r = [-0.3535534, 0.3535534, 0.6123724, 0.6123724 ]
        r = [-0.4545195, 0.4545195, 0.5416752, 0.5416752]
        super(AllegroCardSlidingEnv, self).__init__(num_envs, hand_p=p, hand_r=r, camera_pos=cam_pos,
                                                    camera_target=cam_target,
                                                    steps_per_action=steps_per_action,
                                                    control_mode=control_mode, viewer=viewer, device=device,
                                                    use_cartesian_controller=use_cartesian_controller,
                                                    friction_coefficient=friction_coefficient,
                                                    contact_controller=contact_controller,
                                                    video_save_path=video_save_path,
                                                    joint_stiffness=joint_stiffness, fingers=fingers,
                                                    gradual_control=gradual_control,
                                                    gravity=gravity,
                                                    randomize_obj_start=randomize_obj_start)
        self.card_pose = np.array([0, 0, 0.132], dtype=np.float32)
        card_pose = gymapi.Transform()
        card_pose.p = gymapi.Vec3(*self.card_pose)
        # card_pose.p = gymapi.Vec3(self.card_pose[0], self.card_pose[1]. self.card_pose[2])

        card_urdf = 'card/card.urdf'
        self.asset_options.replace_cylinder_with_capsule = True
        card_asset = self.gym.load_asset(self.sim, self.asset_root, card_urdf, self.asset_options)
        card_shape_props = self.gym.get_asset_rigid_shape_properties(card_asset)
        for i in range(len(card_shape_props)):
            card_shape_props[i].friction = friction_coefficient
            # card_shape_props[i].friction = 0.1
        self.assets.append(
            {'name': 'card',
             'asset': card_asset,
             'pose': card_pose,
             }
        )

        # create table
        table_dims = gymapi.Vec3(0.4, 0.4, 0.05)
        table_pose = gymapi.Transform()
        self.table_pose = np.array([0, 0.1, 0.1])
        table_pose.p = gymapi.Vec3(*self.table_pose)
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, self.asset_options)

        table_shape_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        for i in range(len(table_shape_props)):
            table_shape_props[i].friction = 0.000001
        self.gym.set_asset_rigid_shape_properties(table_asset, table_shape_props)
        self.assets.append(
            {'name': 'table',
             'asset': table_asset,
             'pose': table_pose,
             }
        )

        self._create_env(self.assets)
        self.gym.set_actor_rigid_shape_properties(self.envs[0], self.handles['card'][0], card_shape_props)
        # self.gym.set_actor_rigid_shape_properties(self.envs[0], self.handles['table'][0], table_shape_props)
        # self.gym.set_actor_rigid_shape_properties(self.envs[0], self.handles['wall'][0], wall_shape_props)

        self.prepare_tensors()

        # self.default_dof_pos = torch.cat((torch.tensor([[0, 0.3, 0.4, 0.6]]).float().to(device=self.device),
        #                             torch.tensor([[0, 0.3, 0.4, 0.6]]).float().to(device=self.device),
        #                             torch.tensor([[0, 0.3, 0.3, 0.6]]).float().to(device=self.device),
        #                             torch.tensor([[1.2, 0.3, 0.0, 0.8]]).float().to(device=self.device)),
        #                             dim=1).to(self.device)
        self.default_dof_pos = torch.cat((torch.tensor([[0, 0.35, 0.5, 0.225]]).float().to(device=self.device),
                                          torch.tensor([[0, 0.35, 0.5, 0.225]]).float().to(device=self.device),
                                          torch.tensor([[0, 0.2, 0.3, 0.2]]).float().to(device=self.device),
                                          torch.tensor([[1.2, 0.3, 0.0, 0.8]]).float().to(device=self.device)),
                                         dim=1).to(self.device)
        # add the screwdriver angle to it
        self.default_dof_pos = torch.cat(
            (self.default_dof_pos, torch.tensor([[0, 0, 0, 0, 0, 0]]).float().to(device=self.device)),
            dim=1).to(self.device)
        self.default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.reset()

    def get_state(self):
        results = super(AllegroCardSlidingEnv, self).get_state()
        # card_ori_euler = self._q[:, -3:]
        # card_position = self._q[:, -6:-3]
        card_ori_euler = self._q[:, -1:]
        card_position = self._q[:, -6:-4]
        results['card_ori'] = card_ori_euler
        results['card_position'] = card_position
        # gt_euler = R.from_quat(self.rb_states[-4, 3:7].cpu()).as_euler('XYZ')
        # print(gt_euler, card_ori_euler)
        q = []
        for finger in self.fingers:
            q.append(results[f'{finger}_q'])
        q.append(results['card_position'])
        q.append(results['card_ori'])
        q = torch.cat(q, dim=1)
        results['q'] = q
        return results

    def reset(self, dof_pos=None):
        num_actors = self.actor_rb_states.shape[1]
        global_indices = torch.arange(self.num_envs * num_actors,
                                      dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        default_dof_pos = torch.zeros_like(self.default_dof_pos)
        for i, finger in enumerate(['index', 'middle', 'ring', 'thumb']):
            idx = [i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3]
            default_dof_pos[:, self.finger_to_joint_index[finger]] = self.default_dof_pos[:, idx]
        default_dof_pos[:, 16:] = self.default_dof_pos[:, 16:]
        self.dof_states[:, :, 0] = default_dof_pos
        self.dof_states[:, :, 1] *= 0

        if self.randomize_obj_start:
            # self.dof_states[:, 16:16+2, 0] = default_dof_pos[:, 16:16+2] + 0.05 * torch.randn_like(self.default_dof_pos[:, :16:16+2])
            # self.dof_states[:, -1, 0] = np.pi * 2 * (torch.rand_like(self.default_dof_pos[:, -1]) - 0.5)
            self.dof_states[:, -1, 0] = (random.random() - 0.5) * .5

        robot_ids = global_indices[:, self.handles['allegro'][0]].contiguous()
        # obj_ids = global_indices[:, self.handles['valve'][0]].contiguous()
        # update

        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))

        # rval = self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                      gymtorch.unwrap_tensor(self.dof_states),
        #                                      gymtorch.unwrap_tensor(robot_ids),
        #                                      self.num_envs
        #                                      )
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(default_dof_pos),
                                                        gymtorch.unwrap_tensor(robot_ids),
                                                        self.num_envs
                                                        )
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(torch.zeros_like(self.default_dof_pos)),
                                                        gymtorch.unwrap_tensor(robot_ids),
                                                        self.num_envs
                                                        )
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)
        # to resolve contact
        for _ in range(64):
            self._step_sim()
        self._refresh_tensors()
