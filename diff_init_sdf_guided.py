from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
# from isaac_victor_envs.tasks.allegro_ros import RosAllegroScrewdriverTurningEnv

import sys

sys.path.append('..')
from model import LatentDiffusionModel

import matplotlib.pyplot as plt
from ccai.utils.allegro_utils import *
# from allegro_valve_roll import AllegroValveTurning, AllegroContactProblem, PositionControlConstrainedSVGDMPC, \
#    add_trajectories, add_trajectories_hardware

from ccai.allegro_contact import AllegroManipulationProblem, PositionControlConstrainedSVGDMPC, add_trajectories, \
    add_trajectories_hardware
from ccai.allegro_screwdriver_problem_diffusion import AllegroScrewdriverDiff

from ccai.models.trajectory_samplers import TrajectorySampler

from isaac_victor_envs.utils import get_assets_dir
# from isaac_victor_envs.tasks.allegro import AllegroValveTurningEnv

import numpy as np
import pickle as pkl

import torch
import time
import yaml
import copy
import pathlib
from functools import partial
import itertools
from torch.func import vmap, jacrev, hessian, jacfwd

from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
from ccai.kernels import rbf_kernel, structured_rbf_kernel

from ccai.problem import ConstrainedSVGDProblem, IpoptProblem
from ccai.mpc.csvgd import Constrained_SVGD_MPC
# from ccai.mpc.ipopt import IpoptMPC

import time
import pytorch_volumetric as pv
import pytorch_kinematics as pk
# import pytorch3d.transforms as tf

import matplotlib.pyplot as plt
from ccai.utils.allegro_utils import *
import pytorch_kinematics.transforms as tf

CCAI_PATH = pathlib.Path(__file__).resolve().parents[0]

print("CCAI_PATH", CCAI_PATH)

device = 'cuda:0'
obj_dof = 3
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')


def vector_cos(a, b):
    return torch.dot(a.reshape(-1), b.reshape(-1)) / (torch.norm(a.reshape(-1)) * torch.norm(b.reshape(-1)))


def euler_to_quat(euler):
    matrix = tf.euler_angles_to_matrix(euler, convention='XYZ')
    quat = tf.matrix_to_quaternion(matrix)
    return quat


def euler_to_angular_velocity(current_euler, next_euler):
    current_quat = euler_to_quat(current_euler)
    next_quat = euler_to_quat(next_euler)
    dquat = next_quat - current_quat
    con_quat = - current_quat  # conjugate
    con_quat[..., 0] = current_quat[..., 0]
    omega = 2 * tf.quaternion_raw_multiply(dquat, con_quat)[..., 1:]
    return omega


def do_trial(env, params, fpath, sim_viz_env=None, ros_copy_node=None, inits_noise=None, noise_noise=None, sim=None,
             seed=None):
    """only turn the valve once"""
    num_fingers = len(params['fingers'])
    print('\n get initial state before resetting environment')
    state = env.get_state()
    action_list = []
    # visualize the initial state
    env.frame_fpath = fpath
    env.frame_id = 0

    # state here is 16(4*3(index, middle, thumb) + 4(roll pitch yaw and 1 useless joint))
    start = state['q'].reshape(4 * num_fingers + 4).to(device=params['device'])
    goal = start.clone()
    # index finger is used for stability
    if 'index' in params['fingers']:
        fingers = params['fingers']
    else:
        fingers = ['index'] + params['fingers']
    min_force_dict = None

    # initial grasp
    # make sure the index finger is grasping
    if 'index' in params['fingers']:
        fingers = params['fingers']
    else:
        fingers = ['index'] + params['fingers']

    # warm-starting using learned sampler
    trajectory_sampler = True
    model_path = './data/allegro_screwdriver_diffusion_w_classifier.pt'

    # define the problem for the grasp trajectory
    # we use weak constraint for the diffusion model
    if model_path is not None:
        pregrasp_problem_diff = AllegroScrewdriverDiff(
            start=start[:4 * num_fingers + obj_dof],
            goal=params['valve_goal'],
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.table_pose,
            object_location=params['object_location'],
            object_type=params['object_type'],
            world_trans=env.world_trans,
            regrasp_fingers=fingers,
            contact_fingers=[],
            obj_dof=obj_dof,
            obj_joint_dim=1,
            optimize_force=params['optimize_force'],
        )
        # finger gate index
        index_regrasp_problem_diff = AllegroScrewdriverDiff(
            start=start[:4 * num_fingers + obj_dof],
            goal=params['valve_goal'],
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.table_pose,
            object_location=params['object_location'],
            object_type=params['object_type'],
            world_trans=env.world_trans,
            regrasp_fingers=['index'],
            contact_fingers=['middle', 'thumb'],
            obj_dof=obj_dof,
            obj_joint_dim=1,
            optimize_force=params['optimize_force'],
            default_dof_pos=env.default_dof_pos[:, :16]
        )
        thumb_and_middle_regrasp_problem_diff = AllegroScrewdriverDiff(
            start=start[:4 * num_fingers + obj_dof],
            goal=params['valve_goal'],
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.table_pose,
            object_location=params['object_location'],
            object_type=params['object_type'],
            world_trans=env.world_trans,
            contact_fingers=['index'],
            regrasp_fingers=['middle', 'thumb'],
            obj_dof=obj_dof,
            obj_joint_dim=1,
            optimize_force=params['optimize_force'],
            default_dof_pos=env.default_dof_pos[:, :16]
        )
        turn_problem_diff = AllegroScrewdriverDiff(
            start=start[:4 * num_fingers + obj_dof],
            goal=params['valve_goal'],
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.table_pose,
            object_location=params['object_location'],
            object_type=params['object_type'],
            world_trans=env.world_trans,
            contact_fingers=['index', 'middle', 'thumb'],
            obj_dof=obj_dof,
            obj_joint_dim=1,
            optimize_force=params['optimize_force'],
            default_dof_pos=env.default_dof_pos[:, :16]
        )

        if params['use_partial_constraint']:
            problem_for_sampler = {
                (-1, -1, -1): pregrasp_problem_diff,
                (-1, 1, 1): index_regrasp_problem_diff,
                (1, -1, -1): thumb_and_middle_regrasp_problem_diff,
                (1, 1, 1): turn_problem_diff
            }
        if 'type' not in params:
            params['type'] = 'diffusion'

        vae = None
        model_t = params['type'] == 'latent_diffusion'
        if model_t:
            vae_path = params.get('vae_path', None)
            vae = LatentDiffusionModel(params, None).to(params['device'])
            vae.load_state_dict(torch.load(f'{CCAI_PATH}/{vae_path}'))
            for param in vae.parameters():
                param.requires_grad = False

        trajectory_sampler = TrajectorySampler(T=params['T'] + 1, dx=(15 + 1) if not model_t else params['nzt'],
                                               du=21 if not model_t else 0, type=params['type'],
                                               timesteps=256, hidden_dim=128 if not model_t else 64,
                                               context_dim=3, generate_context=False,
                                               constrain=params['projected'],
                                               problem=problem_for_sampler,
                                               inits_noise=inits_noise, noise_noise=noise_noise,
                                               guided=params['use_guidance'],
                                               vae=None)
        trajectory_sampler.load_state_dict(torch.load(f'{model_path}', map_location=torch.device(params['device'])),
                                           strict=True)
        trajectory_sampler.to(device=params['device'])
        trajectory_sampler.send_norm_constants_to_submodels()
        print('Loaded trajectory sampler')

        # set the initial state to the learned initial state
        state = env.get_state()
        start = state['q'].reshape(4 * num_fingers + 4).to(device=params['device'])

        actual_trajectory = []
        duration = 0

        # define some helper functions
        def _partial_to_full(traj, mode):
            if mode == 'index':
                traj = torch.cat((traj[..., :-6], torch.zeros(*traj.shape[:-1], 3).to(device=params['device']),
                                  traj[..., -6:]), dim=-1)
            if mode == 'thumb_middle':
                traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 6).to(device=params['device'])), dim=-1)
            if mode == 'pregrasp':
                traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 9).to(device=params['device'])), dim=-1)
            return traj

        def _full_to_partial(traj, mode):
            if mode == 'index':
                traj = torch.cat((traj[..., :-9], traj[..., -6:]), dim=-1)
            if mode == 'thumb_middle':
                traj = traj[..., :-6]
            if mode == 'pregrasp':
                traj = traj[..., :-9]
            return traj

        def convert_sine_cosine_to_yaw(xu):
            """
            xu is shape (N, T, 37)
            Replace the sine and cosine in xu with yaw and return the new xu
            """
            sine = xu[..., 15]
            cosine = xu[..., 14]
            yaw = torch.atan2(sine, cosine)
            xu_new = torch.cat([xu[..., :14], yaw.unsqueeze(-1), xu[..., 16:]], dim=-1)
            return xu_new

        def convert_yaw_to_sine_cosine(xu):
            """
            xu is shape (N, T, 36)
            Replace the yaw in xu with sine and cosine and return the new xu
            """
            yaw = xu[14]
            sine = torch.sin(yaw)
            cosine = torch.cos(yaw)
            xu_new = torch.cat([xu[:14], cosine.unsqueeze(-1), sine.unsqueeze(-1), xu[15:]], dim=-1)
            return xu_new

        # execute the grasp trajectory
        # we use initial samples to warm-start the trajectory sampler
        def execute_traj(planner, mode, goal=None, fname=None, initial_samples=True):
            # reset planner
            print('get state ... ...')
            state = env.get_state()
            # we use only first 15 dof(these are all state variables),last one dof is the screwdriver-cap joint
            state = state['q'].reshape(-1)[:16].to(device=params['device'])

            # generate context from mode
            contact = -torch.ones(params['N'], 3).to(device=params['device'])
            if mode == 'thumb_middle':
                contact[:, 0] = 1  # thumb contact and index contact
            elif mode == 'index':
                contact[:, 1] = 1
                contact[:, 2] = 1  # thumb and index and middle contact
            elif mode == 'turn':
                contact[:, :] = 1  # all finger contact

            # generate initial samples with diffusion model
            sim_rollouts = None
            with torch.no_grad():
                start = state.clone()
                # if state[-1] < -1.0:
                #     start[-1] += 0.75
                a = time.perf_counter()
                # start_for_diff = start#convert_yaw_to_sine_cosine(start)
                if params['sine_cosine']:
                    start_for_diff = convert_yaw_to_sine_cosine(start)
                else:
                    start_for_diff = start
                initial_samples, _, _ = trajectory_sampler.sample(N=params['N'],
                                                                  start=start_for_diff.reshape(1, -1),
                                                                  H=params['T'] + 1,
                                                                  constraints=contact)
                if params['sine_cosine']:
                    initial_samples = convert_sine_cosine_to_yaw(initial_samples)
                print('Sampling time', time.perf_counter() - a)

            sim_rollouts = torch.zeros_like(initial_samples)

            initial_samples = _full_to_partial(initial_samples, mode)
            initial_x = initial_samples[:, 1:, :15]
            initial_u = initial_samples[:, :-1, -21:]
            initial_samples = torch.cat((initial_x, initial_u), dim=-1)

            # create list of outputs
            planned_trajectories = []
            actual_trajectory = []
            optimizer_paths = []
            contact_points = {}
            contact_distance = {}
            plans = None

            # size of state is 15
            dx = 15
            # size of action is 21
            du = 21

            for k in range(params['num_steps']):
                print("----------------------------------------------------------------")
                print(f"Step {k + 1}")

                state = env.get_state()
                # roll pitch yaw of the screwdriver
                print('Pose before step:')
                print(state['q'][:, -4:-1].detach().cpu().numpy())
                # using the cloud point to update the pose of the object
                new_pose = env.update_pose_pcd()
                # override the pose of the object with the new pose
                state['q'][:, -4:-1] = new_pose
                state = state['q'].reshape(-1)[:15].to(device=params['device'])

                # Extract the action part of the current step k from the pre-generated action sample sequence
                # And set it as the state of the planner
                # print(initial_samples.shape[0]) # 16
                # choose the first sample as the initial trajectory
                selected_traj = initial_samples[0]

                s = time.time()
                # extract the state and action part of the current step k from the pre-generated action sample sequence
                best_traj = selected_traj[k, :dx + du]
                # make sure the shape is [1, 15+21=36]
                best_traj = best_traj.unsqueeze(0)
                # should be [1, 15+21=36]
                # print(best_traj.shape)  torch.Size([1, 36])
                print(f'Solve time for step {k + 1}', time.time() - s)

                # record the actual trajectory
                if mode == 'turn':
                    index_force = torch.norm(best_traj[..., 27:30], dim=-1)
                    middle_force = torch.norm(best_traj[..., 30:33], dim=-1)
                    thumb_force = torch.norm(best_traj[..., 33:36], dim=-1)
                    print('Middle force:', middle_force)
                    print('Thumb force:', thumb_force)
                    print('Index force:', index_force)
                elif mode == 'index':
                    middle_force = torch.norm(best_traj[..., 27:30], dim=-1)
                    thumb_force = torch.norm(best_traj[..., 30:33], dim=-1)
                    print('Middle force:', middle_force)
                    print('Thumb force:', thumb_force)
                elif mode == 'thumb_middle':
                    index_force = torch.norm(best_traj[..., 27:30], dim=-1)
                    print('Index force:', index_force)

                x = best_traj[0, :dx + du]
                x = x.reshape(1, dx + du)
                action = x[:, dx:du + du].to(device=env.device)

                xu = torch.cat((state[:-1].cpu(), action[0].cpu()))
                actual_trajectory.append(xu)

                action = action[:, :4 * num_fingers]
                action = action.to(device=env.device) + state.unsqueeze(0)[:, :4 * num_fingers].to(device=env.device)

                env.step(action.to(device=env.device))
                state = env.get_state()
                state = state['q'].reshape(-1).to(device=params['device'])
                ori = state[:15][-3:]
                print('ori after step:', ori)

                # make sure contact constraint is satisfied
                # 16 = 4*3(finger number) + 4(roll pitch yaw and 1 screwdriver-cap joint)
                state = env.get_state()
                q_state = state['q'][:16].reshape(-1).to(device=params['device'])
                q_tensor = q_state.reshape(1, 1, 16).clone().detach().requires_grad_(True).to(device=params['device'])
                optimizer = torch.optim.Adam([q_tensor], lr=1e-3)
                # print('q_tensor:', q_tensor.shape)

                from only_sdf_planner import contact_constraints
                # using gradient descent to satisfy the contact constraint
                print('Solving contact constraint...')
                s = time.time()
                # initialize contact scenes make sure it can be used to compute the contact constraint
                threshold = 1e-3
                max_steps = 200
                step = 0
                # for each finger and avoid the infinite loop
                while step < max_steps:
                    step += 1
                    # recompute the contact g, grad_g
                    # reset the gradient of the cost function to zero
                    optimizer.zero_grad()

                    g_list = []  # record the sdf values for each finger
                    for finger_name in (finger_list or ['index', 'middle', 'thumb']):
                        g_f, grad_g_f, _ = contact_constraints(q_tensor,
                                                               finger_name,
                                                               contact_scenes,
                                                               compute_grads=True,
                                                               terminal=False)
                        # update q_tensor by gradient descent(make grad_g_f = 0)
                        g_val = torch.mean(torch.square(g_f))  # cost function for one finger
                        g_list.append(g_val.item())  # record the cost function for each finger

                    cost = sum(torch.mean(torch.square(contact_constraints(q_tensor, f, contact_scenes)[0]))
                               for f in (finger_list or ['index', 'middle', 'thumb']))
                    cost.backward()
                    optimizer.step()

                    # if satisfied, break the loop and print tge sdf values
                    max_g = max(g_list)  # g_list is a list of g_val for each finger
                    if max_g < threshold:
                        # print the sdf values of each finger
                        print(" index: SDF =", g_list[0])
                        print(" middle: SDF =", g_list[1])
                        print(" thumb: SDF =", g_list[2])
                        print('time for solving contact constraint:', time.time() - s)
                        break

                new_state = env.get_state()
                new_state['q'][:, :16] = q_tensor.detach()
                action = new_state['q'][:, :4 * num_fingers]
                action = action.reshape(-1, 4 * num_fingers).to(device=env.device)
                print('force the finger to be in contact with the object')
                env.step(action)

                print("Final SDF values after optimization:")
                for finger_name in ['index', 'middle', 'thumb']:
                    g_f, _, _ = contact_constraints(q_tensor, finger_name, contact_scenes, compute_grads=False)
                    print(f"{finger_name}: SDF =", g_f.detach().cpu().numpy())

            actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=params['device'])
            return actual_trajectory, planned_trajectories, initial_samples, sim_rollouts, optimizer_paths, contact_points, contact_distance

            data = {}
            for t in range(1, 1 + params['T']):
                data[t] = {'plans': [], 'starts': [], 'inits': [], 'init_sim_rollouts': [], 'optimizer_paths': [],
                           'contact_points': [], 'contact_distance': [], 'contact_state': []}

        sample_contact = params.get('sample_contact', False)
        num_stages = 2 + 3 * (params['num_turns'] - 1)
        if not sample_contact:
            # we only focus on the index, thumb_middle, and turn contacts which means turn mode
            contact_sequence = ['turn']
            # for k in range(params['num_turns'] - 1):
            #     contact_options = ['index', 'thumb_middle']
            #     perm = np.random.permutation(2)
            #     contact_sequence += [contact_options[perm[0]], contact_options[perm[1]], 'turn']
        else:
            contact_sequence = None

        contact = None
        next_node = None
        executed_contacts = []
        stages_since_plan = 0

        for stage in range(num_stages):
            print('\n get state before the stage begins ')
            state = env.get_state()
            state = state['q'].reshape(-1)[:15].to(device=params['device'])
            ori = state[:15][-3:]
            yaw = ori[-1]
            _goal = torch.tensor([0, 0, state[-1]]).to(device=params['device'])
            # traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance = execute_traj(
            #     planner=None, mode='diffusion_policy', goal=_goal, fname=f'diffusion_policy_{stage}')

            if stage > len(contact_sequence):
                print('Planner thinks task is complete')
                print(executed_contacts)
                break
            else:
                contact = contact_sequence[stage - 1]
            executed_contacts.append(contact)
            print(f'---Stage == {stage} Contact == {contact}---')

            if contact == 'index':
                _goal = torch.tensor([0, 0, state[-1]]).to(device=params['device'])
                traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance = execute_traj(
                    None, mode='index', goal=_goal, fname=f'index_regrasp_{stage}')
                traj = torch.cat((traj[..., :-6], torch.zeros(*traj.shape[:-1], 3).to(device=params['device']),
                                  traj[..., -6:]), dim=-1)

            elif contact == 'thumb_middle':
                _goal = torch.tensor([0, 0, state[-1]]).to(device=params['device'])
                traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance = execute_traj(
                    None, mode='thumb_middle',
                    goal=_goal, fname=f'thumb_middle_regrasp_{stage}')
                traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 6).to(device=params['device'])), dim=-1)

            elif contact == 'turn':
                _goal = torch.tensor([0, 0, state[-1] - np.pi / 6]).to(device=params['device'])
                traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance = execute_traj(
                    None, mode='turn', goal=_goal, fname=f'turn_{stage}')

            if contact != 'pregrasp':
                actual_trajectory.append(traj)

            env.reset()
    return -1


if __name__ == "__main__":
    config = yaml.safe_load(pathlib.Path('allegro_screwdriver_csvto_only.yaml').read_text())

    from tqdm import tqdm

    sim_env = None
    ros_copy_node = None

    env = AllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                       use_cartesian_controller=False,
                                       viewer=config['visualize'],
                                       steps_per_action=60,
                                       friction_coefficient=config['friction_coefficient'] * 2.5,
                                       # friction_coefficient=1.0,  # DEBUG ONLY, set the friction very high
                                       device=config['sim_device'],
                                       video_save_path=img_save_dir,
                                       joint_stiffness=config['kp'],
                                       fingers=config['fingers'],
                                       gradual_control=False,
                                       gravity=True,  # For data generation only
                                       randomize_obj_start=config.get('randomize_obj_start', False),
                                       )
    sim, gym, viewer = env.get_sim()

    results = {}

    # set up the kinematic chain
    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    ee_names = {
        'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
        'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
        'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
        'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
    }
    config['ee_names'] = ee_names
    config['obj_dof'] = 3
    chain = pk.build_chain_from_urdf(open(asset).read())
    chain = chain.to(device=device)
    robot_sdf = pv.RobotSDF(chain, path_prefix=get_assets_dir() + '/xela_models',
                            use_collision_geometry=False)

    world_trans = env.world_trans
    scene_trans = world_trans.inverse().compose(
        pk.Transform3d(device=device).translate(0, 0, 1.205))  # object_location is ([0, 0, 1.205])

    screwdriver_asset = f'{get_assets_dir()}/screwdriver/screwdriver.urdf'
    chain_object = pk.build_chain_from_urdf(open(screwdriver_asset).read())
    chain_object = chain_object.to(device=device)
    object_sdf = pv.RobotSDF(chain_object, path_prefix=get_assets_dir() + '/screwdriver',
                             use_collision_geometry=False)

    collision_check_links = [ee_names[finger] for finger in config['fingers']]
    contact_scenes = pv.RobotScene(robot_sdf, object_sdf, scene_trans,
                                   collision_check_links=collision_check_links,
                                   softmin_temp=1.0e3,
                                   points_per_link=1000,
                                   )

    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]  # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices,
                           world_trans=env.world_trans)

    forward_kinematics = partial(chain.forward_kinematics,
                                 frame_indices=frame_indices)  # full_to= _partial_state = partial(full_to_partial_state, fingers=config['fingers'])
    partial_to_full_state = partial(partial_to_full_state, fingers=config['fingers'])

    inits_noise, noise_noise = [None] * config['num_trials'], [None] * config['num_trials']
    start_ind = 0  # if config['experiment_name'] == 'allegro_screwdriver_csvto_diff_sine_cosine_eps_.015_2.5_damping_pi_6' else 0
    for i in tqdm(range(start_ind, config['num_trials'])):
        if config['mode'] != 'hardware':
            torch.manual_seed(i)
            np.random.seed(i)

        goal = torch.tensor([0, 0, float(config['goal'])])
        for controller in config['controllers'].keys():
            env.reset()
            now = ''
            fpath = pathlib.Path(
                f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}.{now}/{controller}/trial_{i + 1}')
            if config['mode'] != 'hardware':
                pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
            # set up params
            params = config.copy()
            params.pop('controllers')
            params.update(config['controllers'][controller])
            if torch.cuda.device_count() == 1 and torch.cuda.current_device() == 1:
                params['device'] = 'cuda:0'
            params['controller'] = controller
            params['valve_goal'] = goal.to(device=params['device'])
            params['chain'] = chain.to(device=params['device'])
            object_location = torch.tensor([0, 0, 1.205]).to(
                params['device'])  # TODO: confirm if this is the correct location
            params['object_location'] = object_location
            # If params['device'] is cuda:1 but the computer only has 1 gpu, change to cuda:0
            final_distance_to_goal = do_trial(env, params, fpath, sim_env, ros_copy_node, inits_noise[i],
                                              noise_noise[i], seed=i)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
