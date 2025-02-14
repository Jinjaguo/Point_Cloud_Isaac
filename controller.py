import numpy as np


class Controller:
    def __init__(self, q,
                 q_,
                 o_pose,
                 o_pose_,
                 o_sdf,
                 o_sdf_,
                 u,
                 u_star,
                 c,
                 pc):
        self.q = q  # state from diffusion model
        self.q_ = q_  # observed state
        self.o_pose = o_pose  # object state from diffusion model
        self.o_pose_ = None  # observed object state
        self.o_sdf = o_sdf  # signed distance field of the object
        self.o_sdf_ = None  # observed signed distance field of the object
        self.u = u  # action from diffusion model
        self.u_star = u_star  # action we want to take
        self.c = c  # contact mode,binary variable
        self.pc = pc  # point cloud of the object

    def get_estimated_pose(self, pc, o_sdf):
        # estimate object pose from point cloud and signed distance field
        sampled_surface_points = self.sample_surface_points(o_sdf)  # TODO: sample surface points from sdf
        self.o_pose_ = self.icp(pc, sampled_surface_points)  # TODO: icp to estimate object pose
        self.o_sdf_ = self.get_sdf(self.o_pose_)  # TODO: get sdf of estimated object pose
        return self.o_pose_, self.o_sdf_

    def pose_function(self):
        w_p_1 = 0.1  # weight for object pose cost
        w_p_2 = 0.1  # weight for object pose cost
        # implement cost function
        J_obj_pose = np.linalg.norm(self.o_pose_ - self.o_pose)
        J_q = np.linalg.norm(self.q - self.q_)
        J_p = w_p_1 * J_obj_pose + w_p_2 * J_q
        return J_p

    def contact_point_function(self):
        w_1 = 0.1  # weight for contact point cost
        # forwards kinematics to get contact point
        ee_point = self.forward_kinematics(self.q)  # TODO get the real contact point
        contact_point = self.get_contact_point(self.q, self.o_sdf)  # TODO: get the contact point from sdf
        J_c = w_1 * np.linalg.norm(contact_point - ee_point)
        return J_c

    def control_cost(self):
        w_u = 0.1  # weight for control cost
        J_u = w_u * np.linalg.norm(self.u)
        return J_u

    def smoothness_cost(self, u_prev):
        w_s = 0.1
        J_s = w_s * np.linalg.norm(self.u - u_prev)
        return J_s

    def dynamics_cost(self, q_dot, torque):
        v_max = 1.5  # 速度限制
        tau_max = 2.0  # 力矩限制
        w_d = 0.1  # 权重

        J_v = w_d * max(0, np.linalg.norm(q_dot) - v_max) ** 2
        J_tau = w_d * max(0, np.linalg.norm(torque) - tau_max) ** 2

        return J_v + J_tau

    def total_cost(self, u_prev, c_prev, q_dot, torque):
        J_total = (
                self.pose_function() +
                self.control_cost() +
                self.smoothness_cost(u_prev) +
                self.dynamics_cost(q_dot, torque)
        )
        return J_total




