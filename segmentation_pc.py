#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import os

def process_ply_file(ply_file, save_dir):
    pcd = o3d.io.read_point_cloud(ply_file)

    # 检查点云是否为空
    if not pcd.has_points():
        print(f"Warning: The file '{ply_file}' doesn't contain any points!")
        return

    points = np.asarray(pcd.points)

    # DBSCAN聚类
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10))
    max_label = labels.max()
    print(f"Point cloud '{ply_file}' has {max_label + 1} clusters")

    # 选择特定簇（这里默认选择ID为0的簇，你可能需要调整）
    screwdriver_cluster_id = 0
    screwdriver_points = points[labels == screwdriver_cluster_id]

    # 创建新的点云对象
    screwdriver_pcd = o3d.geometry.PointCloud()
    screwdriver_pcd.points = o3d.utility.Vector3dVector(screwdriver_points)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 生成保存路径
    base_name = os.path.splitext(os.path.basename(ply_file))[0]
    save_path = os.path.join(save_dir, f"{base_name}.ply")

    # 保存点云
    o3d.io.write_point_cloud(save_path, screwdriver_pcd)
    print(f"Saved: {save_path}")

    # 可视化
    # o3d.visualization.draw_geometries([screwdriver_pcd], window_name=f"{base_name} - Screwdriver PointCloud")

def process_folder(folder_path):
    save_dir = os.path.join(folder_path, "screwdriver_only")
    os.makedirs(save_dir, exist_ok=True)

    # 遍历文件夹中的所有 .ply 文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".ply"):
            ply_file = os.path.join(folder_path, file_name)
            process_ply_file(ply_file, save_dir)

if __name__ == "__main__":
    folder_path = './pointclouds/run_3/'
    process_folder(folder_path)
