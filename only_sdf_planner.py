import torch
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

    sdf = ret_scene.get('sdf', None)
    grad_g_theta = ret_scene.get('grad_env_sdf', None)
    grad_g_q = ret_scene.get('grad_sdf', None)
    return data


def contact_constraints(q_state, finger_name, contact_scenes, compute_grads=True, compute_hess=False, terminal=False,
                        projected_diffusion=False):
    """

        :param q_state: state tensor shape [N, T, 16],
        :param finger_name: the name of the current finger, such as "index", "middle", "thumb"
        :param contact_scenes: the object provided by scene_collision_check(...)
        :param compute_grads: whether to return gradients (usually only required under autograd)
        :param compute_hess: ignore or leave blank here
        :param terminal: if True, only take the last frame
        :param projected_diffusion: if True, do not ignore frame 0; otherwise ignore (usually offset=1)

        return: (g, grad_g, None)
        g: shape [N, M], M = T-offset or 1
        grad_g: shape [N, M, T*d], or the shape you want
    """
    N, T, d = q_state.shape
    data = _preprocess_fingers(q_state, contact_scenes)
    ret_scene = data[finger_name]
    g = ret_scene.get('sdf').reshape(N, 1, 1)
    grad_g_q = ret_scene.get('grad_sdf', None)
    grad_g_theta = ret_scene.get('grad_env_sdf', None)
    if finger_name == 'thumb':
        grad_g_q = - grad_g_q
    elif finger_name == 'index':
        pass
    elif finger_name == '':
        pass
    grad_g = None
    return g, grad_g, None


def update(q_state, contact_scenes, finger_list=('index', 'middle', 'thumb'), max_steps=200, threshold=1e-3, lambda_=1e-2):
    """
    :param q_state: state we will update, shape [1, 1, 16], where the first 12 dimensions are fingers (4 fingers x 3 dof), the last 4
    dimensions are objects (roll, pitch, yaw, and cap joint).
    :param contact_scenes: the object provided by scene_collision_check(...)
    :param finger_list: the list of fingers to optimize
    :param max_steps: maximum number of optimization steps
    :param threshold: the threshold for early stopping
    :param lambda_: weight for the regularization term (keep object pose close to initial pose)
    :return: the optimized q_tensor
    """
    q_tensor = q_state.reshape(1, 1, 16).clone().detach()
    init_object_ori = q_tensor[..., 12:15].detach().clone()
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
            grad_sdf_index = grad_sdf[:, :, :4]  # first four dimensions are for index finger
            grad_sdf_middle = grad_sdf[:, :, 4:8]  # next four dimensions are for middle finger
            # we ignore the ring finger because it is not used.
            grad_sdf_thumb = grad_sdf[:, :, 12:]  # last four dimensions are for thumb finger
            # combine the gradients of all fingers should be [1, 1, 12]
            grad_sdf_finger = torch.cat((grad_sdf_index, grad_sdf_middle, grad_sdf_thumb), dim=2)

            grad_g_theta = data[finger_name]['grad_env_sdf']  # => [1, 1, 3]
            grad_g_theta = grad_g_theta.reshape(1, 1, -1)  # => [1, 1, 3]
            grad_g_theta = torch.cat([grad_g_theta, torch.zeros((1, 1, 1), device=grad_g_theta.device)], dim=2)
            grad_sdf_all = torch.cat((grad_sdf_finger, grad_g_theta), dim=2)  # => [N, T, 16]


            import torch.nn.functional as F
            g_eff = F.relu(g)  # g_eff = max(g, 0), a non-negative sdf value
            sdf_vals[finger_name] = g_eff.mean().item()  # record
            # partial_grad
            gf_flat = g_eff.view(-1)  # shape=[1]
            gradgf_flat = grad_sdf_all.view(-1, q_tensor.shape[-1])  # shape=[1, 16]
            # cost=0.5*g^2 => grad(cost)=g * grad(g)
            # this is the gradient of the cost function w.r.t. q_tensor
            partial_grad = (gf_flat.unsqueeze(-1) * gradgf_flat).sum(dim=0)  # [16]
            # print(f'finger_name:{finger_name}, partial_grad:{partial_grad}')
            partial_grad = partial_grad.view(q_tensor.shape)  # shape=[1, 1, 16]
            total_grad += partial_grad

        # cost_reg = 0.5 * lambda_ * || (theta - theta_init) ||^2
        # => grad( cost_reg, theta ) = lambda_ * (theta - theta_init)
        curr_obj_ori = q_tensor[..., 12:15]   # [1,1,3]
        reg_diff = (curr_obj_ori - init_object_ori)
        reg_grad = lambda_ * reg_diff  # [1,1,3]
        total_grad[..., 12:15]  += reg_grad
        q_tensor.grad = total_grad
        # q_tensor has been updated
        optimizer.step()  # Adam updates q_tensor

        # print(f"Step {step}:")
        # for finger in ['index', 'middle', 'thumb']:
        #     print(f"  {finger}: SDF = {sdf_vals[finger]:.6f}")

        if max(sdf_vals.values()) < threshold:
            print('time for solving contact constraint:', time.time() - s)
            # print the sdf values of each finger
            # print("Final SDF values after optimization:")
            for finger in ['index', 'middle', 'thumb']:
                print(f"  {finger}: SDF = {sdf_vals[finger]:.6f}")
            break  # if the contact constraint is satisfied, break the loop

        # return the optimized q_tensor
    return q_tensor

def quat_change_convention(q, current='xyzw'):
    if current == 'xyzw':
        return torch.stack(
            (q[:, 3], q[:, 0], q[:, 1], q[:, 2]), dim=-1)

    if current == 'wxyz':
        return torch.stack((
            q[:, 1], q[:, 2], q[:, 3], q[:, 0]), dim=-1)