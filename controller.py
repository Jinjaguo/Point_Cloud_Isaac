import numpy as np
import os
import open3d as o3d
from isaac_victor_envs.utils import get_assets_dir
import sys
from pytorch_volumetric import sdf
import copy
from sklearn.neighbors import NearestNeighbors
from torch.fx.experimental.unification.multipledispatch.dispatcher import source

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

    def contact_point_function(self):
        w_1 = 0.1  # weight for contact point cost
        # forwards kinematics to get contact point
        ee_point = self.forward_kinematics(self.q)  # TODO get the real contact point
        contact_point = self.get_contact_point(self.q, self.o_sdf)  # TODO: get the contact point from sdf
        J_c = w_1 * np.linalg.norm(contact_point - ee_point)
        return J_c


class Sample_Points():
    def __init__(self, point_cloud_folder, screwdriver_asset, sample_numbers ):
        self.folder = point_cloud_folder
        self.screwdriver_asset = screwdriver_asset
        self.sample_numbers = sample_numbers

    def get_point_cloud(self, filename):
        file_path = os.path.join(self.folder, filename)
        pcd = o3d.io.read_point_cloud(file_path)
        pc = np.asarray(pcd.points)
        return pc

    def get_sample_points_sdf(self):
        sample_points = self.sample_numbers

        screwdriver_asset = self.screwdriver_asset
        screwdriver_chain = pk.build_chain_from_urdf(open(screwdriver_asset).read())
        object_sdf = pv.RobotSDF(screwdriver_chain, path_prefix=get_assets_dir() + '/screwdriver',
                                 use_collision_geometry=False, link_sdf_cls=sdf.CylinderSDF)
        # print(f"Number of links in SDF: {len(object_sdf.sdf.sdfs)}")
        all_points = []

        num_links = len(object_sdf.sdf.sdfs)
        for i, link_sdf in enumerate(object_sdf.sdf.sdfs):
            # TODO check the function of sample_surface_points
            link_surface_points, _ = link_sdf.sample_surface_points(sample_points[i])
            all_points.append(link_surface_points)
            print(f"Link {i} - Sampled points shape: {link_surface_points.shape}")
            print(f"First few points: \n{link_surface_points[:5]}")

        sampled_surface_points = torch.cat(all_points, dim=0)
        print("Sampled points shape:", sampled_surface_points.shape)

        return sampled_surface_points.numpy()


class points_registration():
    def __init__(self):
        pass


    def get_pose_estimation(self, point_cloud, sampled_surface_points):
        # estimate object pose from point cloud and signed distance field
        source_point = point_cloud
        target_point = sampled_surface_points

        threshold = 0.02
        trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                                 [-0.139, 0.967, -0.215, 0.7],
                                 [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
        numbers = min(len(source_point), len(target_point))
        print(f"source_point shape: {source_point.shape}, target_point shape: {target_point.shape}")

        if len(source_point) > len(target_point):
            idx = np.random.choice(target_point.shape[0], min(numbers, target_point.shape[0]), replace=False)
            source_point = source_point[idx]
        else:
            idx = np.random.choice(source_point.shape[0], min(numbers, source_point.shape[0]), replace=False)
            target_point = target_point[idx]

        print(f"source_point shape: {source_point.shape}, target_point shape: {target_point.shape}")
        T, _, _ = self.icp(source_point, target_point, init_pose = trans_init, max_iterations=20, tolerance=threshold)
        print(T)
        self.draw_registration_result(point_cloud, sampled_surface_points, T)


    def draw_registration_result(self,source, target, transformation):
        source_temp = copy.deepcopy(source)
        source_temp = o3d.geometry.PointCloud()
        source_temp.points = o3d.utility.Vector3dVector(source)

        target_temp = copy.deepcopy(target)
        target_temp = o3d.geometry.PointCloud()
        target_temp.points = o3d.utility.Vector3dVector(target)

        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def best_fit_transform(self,A, B):
        '''
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
          A: Nxm numpy array of corresponding points
          B: Nxm numpy array of corresponding points
        Returns:
          T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
          R: mxm rotation matrix
          t: mx1 translation vector
        '''

        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m - 1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R, centroid_A.T)

        # homogeneous transformation
        T = np.identity(m + 1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t

    def nearest_neighbor(self, src, dst):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        '''

        assert src.shape == dst.shape

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()

    def icp(self,A, B, init_pose=None, max_iterations=20, tolerance=0.001):
        '''
        The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
        Input:
            A: Nxm numpy array of source mD points
            B: Nxm numpy array of destination mD point
            init_pose: (m+1)x(m+1) homogeneous transformation
            max_iterations: exit algorithm after max_iterations
            tolerance: convergence criteria
        Output:
            T: final homogeneous transformation that maps A on to B
            distances: Euclidean distances (errors) of the nearest neighbor
            i: number of iterations to converge
        '''

        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m + 1, A.shape[0]))
        dst = np.ones((m + 1, B.shape[0]))
        src[:m, :] = np.copy(A.T)
        dst[:m, :] = np.copy(B.T)

        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)

        prev_error = 0

        for i in range(max_iterations):
            # find the nearest neighbors between the current source and destination points
            distances, indices = self.nearest_neighbor(src[:m, :].T, dst[:m, :].T)

            # compute the transformation between the current source and nearest destination points
            T, _, _ = self.best_fit_transform(src[:m, :].T, dst[:m, indices].T)

            # update the current source
            src = np.dot(T, src)

            # check error
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        # calculate final transformation
        T, _, _ = self.best_fit_transform(A, src[:m, :].T)

        return T, distances, i

if __name__ == "__main__":

    point_cloud_folder = './pointclouds/run_1/screwdriver_only'
    screwdriver_asset = f'{get_assets_dir()}/screwdriver/screwdriver_6d_back.urdf'

    for file in os.listdir(point_cloud_folder):
        if file.endswith(".ply"):
            # table\ stick \ screwdriver body\ cap \ marker
            n = [0, 100, 100, 100, 100]
            sp = Sample_Points(point_cloud_folder, screwdriver_asset, n)
            sample_points = sp.get_sample_points_sdf()
            point_cloud = sp.get_point_cloud(file)

            reg = points_registration()
            reg.get_pose_estimation(point_cloud, sample_points)
            #here we output the transformation matrix
            # TODO: use the transformation matrix to update the object pose in the diffusion model