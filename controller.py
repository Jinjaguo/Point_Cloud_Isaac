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
        self.q = q # state from diffusion model
        self.q_ = q_ # observed state
        self.o_pose = o_pose # object state from diffusion model
        self.o_pose_ = None # observed object state
        self.o_sdf = o_sdf # signed distance field of the object
        self.o_sdf_ = None # observed signed distance field of the object
        self.u = u #action from diffusion model
        self.u_star = u_star # action we want to take
        self.c = c # contact mode,binary variable
        self.pc = pc # point cloud of the object

    def get_estimated_pose(self, pc, o_sdf):
        # estimate object pose from point cloud and signed distance field
        sampled_surface_points = self.sample_surface_points(o_sdf) #TODO: sample surface points from sdf
        self.o_pose_ = self.icp(pc, sampled_surface_points) #TODO: icp to estimate object pose
        self.o_sdf_ = self.get_sdf(self.o_pose_) #TODO: get sdf of estimated object pose
        return self.o_pose_, self.o_sdf_

    def pose_function(self):
        # implement cost function
        J_obj_pose = np.linalg.norm(self.o_pose_ - self.o_pose)
        J_q = np.linalg.norm(self.q - self.q_)
        J_1 = J_obj_pose + J_q
        return J_1

    def contact_point_function(self):
        # forwards kinematics to get contact point
        ee_point = self.forward_kinematics(self.q)
        contact_point = self.get_contact_point(self.q, self.o_sdf)
        J_2 = np.linalg.norm(contact_point - ee_point)
        return J_2

