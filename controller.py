import numpy as np
import os
import open3d as o3d
from isaac_victor_envs.utils import get_assets_dir
import sys
from pytorch_volumetric import sdf
from sklearn.neighbors import NearestNeighbors


sys.path.append('..')

import pytorch_volumetric as pv
import pytorch_kinematics as pk
from ccai.utils.allegro_utils import *


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

    def contact_point_function(self):
        w_1 = 0.1  # weight for contact point cost
        # forwards kinematics to get contact point
        ee_point = self.forward_kinematics(self.q)  # TODO get the real contact point
        contact_point = self.get_contact_point(self.q, self.o_sdf)  # TODO: get the contact point from sdf
        J_c = w_1 * np.linalg.norm(contact_point - ee_point)
        return J_c


class Sample_Points():

    def __init__(self, point_cloud_file_path, screwdriver_asset):
        self.file_path = point_cloud_file_path
        self.screwdriver_asset = screwdriver_asset

    def get_point_cloud(self, file):
        # get point cloud from path
        file = os.path.join(self.file_path, file)
        pcd = o3d.io.read_point_cloud(file)
        pc = np.asarray(pcd.points)
        return pc

    def get_sample_points_sdf(self, num_points_per_link=1000):
        screwdriver_asset = f'{get_assets_dir()}/screwdriver/screwdriver_6d_back.urdf'
        screwdriver_chain = pk.build_chain_from_urdf(open(screwdriver_asset).read())
        object_sdf = pv.RobotSDF(screwdriver_chain, path_prefix=get_assets_dir() + '/screwdriver',
                                 use_collision_geometry=False, link_sdf_cls=sdf.CylinderSDF)
        print(f"Number of links in SDF: {len(object_sdf.sdf.sdfs)}")
        all_points = []
        for i, link_sdf in enumerate(object_sdf.sdf.sdfs):
            num_points_per_link /= 5  # reduce number of points per link to speed up sampling
            link_surface_points, _ = link_sdf.sample_surface_points(num_points_per_link)
            if link_surface_points is None or link_surface_points.nelement() == 0:
                print(f"Warning: No points sampled for link {i}")
                continue
            all_points.append(link_surface_points)
            # print(f"Link {i} - Sampled points shape: {link_surface_points.shape}")
            # print(f"First few points: \n{link_surface_points[:5]}")

        sampled_surface_points = torch.cat(all_points, dim=0)
        print("Sampled points shape:", sampled_surface_points.shape)

        return sampled_surface_points.numpy()


def get_pose_estimation(point_cloud, sampled_surface_points):
    # estimate object pose from point cloud and signed distance field
    source = o3d.geometry.PointCloud(point_cloud)
    target = o3d.geometry.PointCloud(sampled_surface_points)
    threshold = 0.02
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])

    evaluation = o3d.registration.evaluate_registration(source, target,
                                                        threshold, trans_init)
    print(evaluation)
    print("Apply point-to-point ICP")
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)

def visualize_reg(point_cloud, sampled_surface_points, transformation_matrix):
    # visualize registration result
    source = o3d.geometry.PointCloud(point_cloud)
    target = o3d.geometry.PointCloud(sampled_surface_points)
