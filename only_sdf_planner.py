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
    g = sdf_3d

    # we don't use T offset here, since we just use one time step
    # T_offset = 0 if projected_diffusion else 1
    # # Retrieve pre-processed data
    # if T_offset < T:
    #     g = sdf_3d[:, T_offset:]  # shape [N, T - T_offset]
    # else:
    #     g = sdf_3d

    # If terminal, only consider last state
    if terminal and g.shape[1] > 0:
        g = g[:, -1:].reshape(N, 1)

    g = g.reshape(N, -1) #[N, M]

    if not compute_grads:
        return g, None, None

    grad_2d = ret_scene['grad_sdf'][:, finger_idx, :]  # [N*T, 16]
    grad_3d = grad_2d.reshape(N, T, 16)  # => [N, T, 16]
    # grad_3d = grad_3d[:, T_offset:]  # => [N, T - offset, 16] # we don't use T offset here, since we just use one time step
    if terminal and grad_3d.shape[1] > 0:
        grad_3d = grad_3d[:, -1:]  # [N,1,16]

    grad_g = grad_3d

    if compute_hess:
        hess = torch.zeros(N, g.shape[1], T * d, T * d, device=q.device)
        return g, grad_g, hess

    return g, grad_g, None

def combine_3_fingers(q_index, q_middle, q_thumb):
    """
    :param q_index: q_index.shape=[4]
    :param q_middle: q_middle.shape=[4]
    :param q_thumb: q_thumb.shape=[4]
    :return: shape=[1，1，16]
    """
    q_index_flat = q_index.view(-1)  # => [4]
    q_middle_flat = q_middle.view(-1)  # => [4]
    q_thumb_flat = q_thumb.view(-1)  # => [4]
    zeros = torch.zeros_like(q_index_flat)  # => [4]
    cat_fingers = torch.cat([q_index_flat, q_middle_flat, zeros, q_thumb_flat], dim=0)

    # print(f"q_index.shape:{q_index.shape}, q_middle.shape:{q_middle.shape}, q_thumb.shape:{q_thumb.shape}")
    return cat_fingers.view(1, 1, 16)



def compute_finger_sdf_grad(q_index, q_middle, q_thumb, finger_name, contact_scenes):
    """Wrap combine_3_fingers -> contact_constraints -> parse out SDF, grad
    :param q_index: q_index.shape=[4]
    :param q_middle: q_middle.shape=[4]
    :param q_thumb: q_thumb.shape=[4]
    :param finger_name: the name of the current finger, such as "index", "middle", "thumb"
    :param contact_scenes: the object provided by scene_collision_check(...)
    """
    q_full = combine_3_fingers(q_index, q_middle, q_thumb)  # shape=[1,1,16]
    # print(f"q_full.shape:{q_full.shape}")
    g, grad_g, _ = contact_constraints(q_full, finger_name,
                                       contact_scenes,
                                       compute_grads=True,
                                       compute_hess=False,
                                       terminal=False,
                                       projected_diffusion=False)
    # [1, 1, 16]
    import torch.nn.functional as F
    g_eff = F.relu(g)  # g_eff = max(g, 0), a non-negative sdf value
    # partial_grad
    gf_flat = g_eff.view(-1)  # shape=[1]
    if finger_name == 'thumb':
        #
        gradgf_flat = -grad_g.view(-1, grad_g.shape[-1])
    else:
        gradgf_flat = grad_g.view(-1, grad_g.shape[-1])

    # cost=0.5*g^2 => grad(cost)=g * grad(g)
    # this is the gradient of the cost function w.r.t. q_tensor
    partial_grad = (gf_flat.unsqueeze(-1) * gradgf_flat).sum(dim=0)
    # print(f'finger_name:{finger_name}, partial_grad:{partial_grad}')
    partial_grad = partial_grad.view(1, 1, 16)  # => [1,1,16]

    return g_eff.mean().item(), partial_grad  # mean sdf, partial_grad


def update(q_state, contact_scenes, finger_list=('index','middle','thumb'), max_steps=200, threshold=1e-3):
    """
    :param q_state: shape [1, 1, 16], where the first 12 dimensions are fingers (4 fingers x 3 dof), the last 4 dimensions are objects (roll, pitch, yaw, etc).
    :param contact_scenes: the object provided by scene_collision_check(...)
    :param finger_list: the list of fingers to optimize
    :param max_steps: maximum number of optimization steps
    :param threshold: the threshold for early stopping
    :return: the optimized q_tensor
    """
    device = q_state.device
    # here we use a nn.Parameter to make q_tensor a learnable parameter
    q_tensor = q_state.clone().detach().to(device)

    # create separate Param for each finger
    q_param_index = torch.nn.Parameter(q_tensor[:, :, 0:4].clone())  # [1, 1, 4]
    q_param_middle = torch.nn.Parameter(q_tensor[:, :, 4:8].clone()) # [1, 1, 4]
    q_param_thumb = torch.nn.Parameter(q_tensor[:, :, 12:16].clone()) # [1, 1, 4]

    # create separate optimizers
    optimizer_index = torch.optim.Adam([q_param_index], lr=1e-3)
    optimizer_middle = torch.optim.Adam([q_param_middle], lr=1e-3)
    optimizer_thumb = torch.optim.Adam([q_param_thumb], lr=1e-3)

    import time
    step = 0
    s = time.time() # a probe to measure the time for solving contact constraint

    while step < max_steps:
        step += 1
        sdf_vals = {}  # record the sdf values for each finger

        # -------- index --------
        sdf_index, grad_index = compute_finger_sdf_grad(
            q_param_index,
            q_param_middle,
            q_param_thumb,
            'index',
            contact_scenes
        )
        sdf_vals['index'] = sdf_index
        print(f"index sdf={sdf_index:.6f}")
        if sdf_index >= threshold:
            optimizer_index.zero_grad()
            # grad_index [1,1,16], index [0:4]
            q_param_index.grad = grad_index[:, :, 0:4]
            optimizer_index.step()

        # -------- middle --------
        sdf_middle, grad_middle = compute_finger_sdf_grad(
            q_param_index,
            q_param_middle,
            q_param_thumb,
            'middle',
            contact_scenes
        )
        print(f"middle sdf={sdf_middle:.6f}")
        sdf_vals['middle'] = sdf_middle
        if sdf_middle >= threshold:
            optimizer_middle.zero_grad()
            q_param_middle.grad = grad_middle[:, :, 4:8]
            optimizer_middle.step()

        # -------- thumb --------
        sdf_thumb, grad_thumb = compute_finger_sdf_grad(
            q_param_index,
            q_param_middle,
            q_param_thumb,
            'thumb',
            contact_scenes
        )
        sdf_vals['thumb'] = sdf_thumb
        print(f"thumb sdf={sdf_thumb:.6f}")
        if sdf_thumb >= threshold:
            optimizer_thumb.zero_grad()
            q_param_thumb.grad = grad_thumb[:, :, 12:16]
            optimizer_thumb.step()

        # if all fingers < threshold, break
        if all(sdf < threshold for sdf in sdf_vals.values()):
            print(f"All fingers satisfied at step={step}")
            break

        # combine final result
    q_final = combine_3_fingers(q_param_index, q_param_middle, q_param_thumb)
    print(f"Optimization finished in {time.time()-s:.2f}s")
    return q_final