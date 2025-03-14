import torch
from ccai.utils.allegro_utils import *

FINGER_TO_IDX = {
    'index': 0,
    'middle': 1,
    'thumb': 2
}

def contact_constraints(q, finger_name, contact_scenes, compute_grads=True, compute_hess=False, terminal=False,
                         projected_diffusion=False):
    """

        :param q: shape [N, T, 16], where the first 12 dimensions are fingers (4 fingers x 3 dof), the last 4 dimensions are objects (roll, pitch, yaw, etc).
        :param finger_name: the name of the current finger, such as "index", "middle", "ring", "thumb"
        :param contact_scenes: the object provided by scene_collision_check(...)
        :param compute_grads: whether to return gradients (usually only required under autograd)
        :param compute_hess: ignore or leave blank here
        :param terminal: if True, only take the last frame
        :param projected_diffusion: if True, do not ignore frame 0; otherwise ignore (usually offset=1)

        return: (g, grad_g, None)
        g: shape [N, M], M = T-offset or 1
        grad_g: shape [N, M, T*d], or the shape you want
    """
    # q: (N, T, 16) = (1, 1, 16)
    N, T, d = q.shape
    assert d == 16, f"Expect last dim=16, but got {d}"

    q_r = q[..., :12] # shape [N, T, 12]
    q_b = q_r.reshape(N*T, 4 * 3)

    theta = q[..., 12:]  # shape [N, T, 4]
    theta_b = theta.reshape(N*T, 4)

    fingers = ['index', 'middle', 'thumb']
    full_q = partial_to_full_state(q_b, fingers=fingers)

    ret_scene = contact_scenes.scene_collision_check(full_q, theta_b,
                                                          compute_gradient=True,
                                                          compute_hessian=False)

    finger_idx = FINGER_TO_IDX[finger_name]
    sdf_2d = ret_scene['sdf'][:, finger_idx] # shape [N*T]
    sdf_3d = sdf_2d.reshape(N, T) #[N, T]

    T_offset = 0 if projected_diffusion else 1
    # Retrieve pre-processed data
    if T_offset < T:
        g = sdf_3d[:, T_offset:]  # shape [N, T - T_offset]
    else:
        g = sdf_3d

    # If terminal, only consider last state
    if terminal and g.shape[1] > 0:
        g = g[:, -1:].reshape(N, 1)

    g = g.reshape(N, -1) #[N, M]

    if not compute_grads:
        return g, None, None

    grad_2d = ret_scene['grad_sdf'][:, finger_idx, :]  # [N*T, 16]
    grad_3d = grad_2d.reshape(N, T, 16)  # => [N, T, 16]
    grad_3d = grad_3d[:, T_offset:]  # => [N, T - offset, 16]
    if terminal and grad_3d.shape[1] > 0:
        grad_3d = grad_3d[:, -1:]  # [N,1,16]

    grad_g = grad_3d

    if compute_hess:
        hess = torch.zeros(N, g.shape[1], T * d, T * d, device=q.device)
        return g, grad_g, hess

    return g, grad_g, None