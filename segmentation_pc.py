#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import sys

def main(ply_file):
    pcd = o3d.io.read_point_cloud(ply_file)
    #show the number of points in the point cloud
    if not pcd.has_points():
        print(f"Warning: The file '{ply_file}' doesn't contain any points!")
        return
    points = np.asarray(pcd.points)
    # print(f"Number of points: {points.shape[0]}")

    # o3d.visualization.draw_geometries([pcd])

    # 对点云进行DBSCAN聚类
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10))
    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")

    screwdriver_cluster_id = 0

    screwdriver_points = points[labels == screwdriver_cluster_id]

    # 创建新的点云对象
    screwdriver_pcd = o3d.geometry.PointCloud()
    screwdriver_pcd.points = o3d.utility.Vector3dVector(screwdriver_points)

    import datetime
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    o3d.io.write_point_cloud(f"./screwdriver_pointclouds/screwdriver_{timestamp}.ply", screwdriver_pcd)
    print("screwdriver_only.ply saved")

    # 可视化螺丝刀点云
    o3d.visualization.draw_geometries([screwdriver_pcd], window_name="Screwdriver PointCloud")


if __name__ == "__main__":
    ply_file = './pointclouds/pointcloud_01-24_22-48-17.ply'
    main(ply_file)
