from ccai.utils.allegro_utils import *

FINGER_TO_IDX = {
    'index': 0,
    'middle': 1,
    'thumb': 2
}

def finger_wrapper(q_state):
    """
    :param q_state: (1, 1, 16)
    :return: a dictionary containing sdf and grad_sdf for each finger
    """
    fingers = ['index', 'middle', 'thumb']

    q_b = q_state[..., :12].reshape(-1, 4 * len(fingers))
    theta = q_state[..., 12:]  # shape [N, T, 4]
    theta_b = theta.reshape(-1, 4)

    full_q = partial_to_full_state(q_b, fingers=fingers)  # full_q is a tensor of shape [N, T, 16]

    des_q = torch.cat((full_q, theta_b), dim=-1) #[1, 1, 20]
    des_q = des_q.reshape(1, 20)

    return des_q

def _preprocess_fingers(q_state, contact_scenes):
    """
    :param q_state: (1, 1, 16)
    :param contact_scenes: the object provided by scene_collision_check(...)
    :return: data: a dictionary containing sdf and grad_sdf for each finger
    """
    data = {}
    fingers = ['index', 'middle', 'thumb']
    # q: (N, T, 16) = (1, 1, 16)
    N, T, d = q_state.shape
    assert d == 16, f"Expect last dim=16, but got {d}"

    q_b = q_state[..., :12].reshape(-1, 4 * len(fingers))

    theta = q_state[..., 12:15]  # shape [N, T, 3]
    theta_b = theta.reshape(-1, 3)
    theta_obj_joint = torch.zeros((theta_b.shape[0], 1),
                                  device=theta_b.device)
    # add a dimension for the cap of the screwdriver
    theta_b = torch.cat((theta_b, theta_obj_joint), dim=1)

    full_q = partial_to_full_state(q_b, fingers=fingers) # full_q is a tensor of shape [N, T, 16]

    ret_scene = contact_scenes.scene_collision_check(full_q, theta_b,
                                                     compute_gradient=True,
                                                     compute_hessian=False)

    for i, finger in enumerate(fingers):
        data[finger] = {}
        data[finger]['sdf'] = ret_scene['sdf'][:, i].reshape(N, T)
        grad_g_q = ret_scene.get('grad_sdf', None)
        data[finger]['grad_sdf'] = grad_g_q[:, i].reshape(N, T, d)
        data[finger]['grad_env_sdf'] = ret_scene['grad_env_sdf'][:, i, :3]
        data[finger]['closest_pt_world'] = ret_scene['closest_pt_world'][:, i]
        data[finger]['contact_normal'] = ret_scene['contact_normal'][:, i]

    return data


def quat_change_convention(q, current='xyzw'):
    if current == 'xyzw':
        return torch.stack(
            (q[:, 3], q[:, 0], q[:, 1], q[:, 2]), dim=-1)

    if current == 'wxyz':
        return torch.stack((
            q[:, 1], q[:, 2], q[:, 3], q[:, 0]), dim=-1)