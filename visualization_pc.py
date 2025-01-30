import open3d as o3d
import os

path = "./screwdriver_pointclouds"
files = os.listdir(path)


files = [file for file in files if file.endswith(('.ply', '.pcd'))]

for file in files:
    pcd = o3d.io.read_point_cloud(os.path.join(path, file))
    o3d.visualization.draw_geometries([pcd])
