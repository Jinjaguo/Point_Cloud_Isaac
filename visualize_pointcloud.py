#!/usr/bin/env python3
import open3d as o3d
import sys

def main(ply_file):
    pcd = o3d.io.read_point_cloud(ply_file)
    if not pcd.has_points():
        print(f"Warning: The file '{ply_file}' doesn't contain any points!")
        return

    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    ply_file = './pointclouds/pointcloud_01-23_23-54.ply'
    main(ply_file)
