import torch
from ccai.utils.allegro_utils import *

FINGER_TO_IDX = {
    'index': 0,
    'middle': 1,
    'thumb': 2
}
def _preprocess_fingers(q, contact_scenes):
    """
    :param q: (1, 1, 16)
    :param contact_scenes: the object provided by scene_collision_check(...)
    :return: data: a dictionary containing sdf and grad_sdf for each finger
    """
    data = {}
    fingers = ['index', 'middle', 'thumb']
    # q: (N, T, 16) = (1, 1, 16)
    N, T, d = q.shape
    assert d == 16, f"Expect last dim=16, but got {d}"

    q_b = q[..., :12].reshape(-1, 4 * len(fingers))

    theta = q[..., 12:15]  # shape [N, T, 3]
    theta_b = theta.reshape(-1, 3)
    theta_obj_joint = torch.zeros((theta_b.shape[0], 1),
                                  device=theta_b.device)
    # add a dimension for the cap of the screwdriver
    theta_b = torch.cat((theta_b, theta_obj_joint), dim=1)

    full_q = partial_to_full_state(q_b, fingers=fingers)

    ret_scene = contact_scenes.scene_collision_check(full_q, theta_b,
                                                     compute_gradient=True,
                                                     compute_hessian=False)

    for i, finger in enumerate(fingers):
        data[finger] = {}
        data[finger]['sdf'] = ret_scene['sdf'][:, i].reshape(N, T)
        grad_g_q = ret_scene.get('grad_sdf', None)
        data[finger]['grad_sdf'] = grad_g_q[:, i].reshape(N, T, d)
        data[finger]['grad_env_sdf'] = ret_scene['grad_env_sdf'][:, i, :3]

    return data


def contact_constraints(q, finger_name, contact_scenes, compute_grads=True, compute_hess=False, terminal=False,
                         projected_diffusion=False):
    """

        :param q: state tensor shape [N, T, 16],
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
    N, T, _ = q.shape # [1, 1, 16]
    data = _preprocess_fingers(q, contact_scenes)
    ret_scene = data[finger_name]
    g = ret_scene.get('sdf').reshape(N, 1, 1)
    grad_g_q = ret_scene.get('grad_sdf', None)
    grad_g_theta = ret_scene.get('grad_env_sdf', None)
    print(grad_g_theta)

    T_offset = 0 if projected_diffusion else min(1, T - 1)
    d = 32 + 15  # 32 + obj_dof(15) = 47
    if compute_grads:
        T_range = torch.arange(T, device=q.device)
        # compute gradient of sdf
        grad_g = torch.zeros(N, T, T, d, device=q.device)
        grad_g[:, T_range, T_range, :16] = grad_g_q[:, T_offset:]
        grad_g[:, T_range, T_range, 16: 16 + 3] = grad_g_theta.reshape(N, T + T_offset, 3)[:,
                                                             T_offset:]
        grad_g = grad_g.reshape(N, -1, T, d)
        grad_g = grad_g.reshape(N, -1, T * d)
        print(grad_g.shape) # torch.Size([1, 1, 47])
        if terminal:
            grad_g = grad_g[:, -1].reshape(N, 1, T * d)
    else:
        return g, None, None

    if compute_hess:
        hess = torch.zeros(N, g.shape[1], T * d, T * d, device=q.device)
        return g, grad_g, hess

    return g, grad_g, None


def update(q_state, contact_scenes, finger_list=('index','middle','thumb'), max_steps=200, threshold=1e-3):
    """
    :param q_state: shape [1, 1, 16], where the first 12 dimensions are fingers (4 fingers x 3 dof), the last 4 dimensions are objects (roll, pitch, yaw, etc).
    :param contact_scenes: the object provided by scene_collision_check(...)
    :param finger_list: the list of fingers to optimize
    :param max_steps: maximum number of optimization steps
    :param threshold: the threshold for early stopping
    :return: the optimized q_tensor
    """
    q_tensor = q_state.reshape(1, 1, 16).clone().detach()
    q_tensor.requires_grad_(True)
    optimizer = torch.optim.Adam([q_tensor], lr=1e-3)

    step = 0
    import time
    s = time.time()

    while step < max_steps:
        step += 1
        # first we zero the gradients
        optimizer.zero_grad()
        total_grad = torch.zeros_like(q_tensor)

        data = _preprocess_fingers(q_tensor, contact_scenes)
        sdf_vals = {}
        # calculate sdf and grad
        for finger_name in finger_list:
            g = data[finger_name]['sdf']  # => shape [N, T], [1,1]
            grad_sdf = data[finger_name]['grad_sdf']  # => [N, T, 16]

            # print(grad_sdf)
            import torch.nn.functional as F
            g_eff = F.relu(g)  # g_eff = max(g, 0), a non-negative sdf value
            sdf_vals[finger_name] = g_eff.mean().item()  # record
            # partial_grad
            gf_flat = g_eff.view(-1)  # shape=[1]
            if finger_name == 'thumb':
                #
                gradgf_flat = -grad_sdf.view(-1, q_tensor.shape[-1])  # shape=[1, 16]
            else:
                gradgf_flat = grad_sdf.view(-1, q_tensor.shape[-1])  # shape=[1, 16]

            # cost=0.5*g^2 => grad(cost)=g * grad(g)
            # this is the gradient of the cost function w.r.t. q_tensor
            partial_grad = (gf_flat.unsqueeze(-1) * gradgf_flat).sum(dim=0) #[16]
            # print(f'finger_name:{finger_name}, partial_grad:{partial_grad}')
            partial_grad = partial_grad.view(q_tensor.shape)  # shape=[1, 1, 16]
            total_grad += partial_grad

        q_tensor.grad = total_grad
        # q_tensor has been updated
        optimizer.step()  # Adam updates q_tensor

        print(f"Step {step}:")
        for finger in ['index', 'middle', 'thumb']:
            print(f"  {finger}: SDF = {sdf_vals[finger]:.6f}")

        if max(sdf_vals.values()) < threshold:
            print('time for solving contact constraint:', time.time() - s)
            # # print the sdf values of each finger
            # print("Final SDF values after optimization:")
            # for finger in ['index', 'middle', 'thumb']:
            #     print(f"  {finger}: SDF = {sdf_vals[finger]:.6f}")
            break  # if the contact constraint is satisfied, break the loop

        # return the optimized q_tensor
    return q_tensor