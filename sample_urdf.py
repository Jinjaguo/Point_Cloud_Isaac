import numpy as np
import open3d as o3d
from urdfpy import URDF


def sample_box(size, num_points=1000):
    """
    从一个长方体表面均匀采样点云。
    :param size: (length, width, height) 盒子尺寸
    :param num_points: 采样点数
    :return: (N,3) 点云
    """
    l, w, h = size
    n = int(num_points // 6)  # 每个面分配点数

    x = np.random.uniform(-l / 2, l / 2, (n, 1))
    y = np.random.uniform(-w / 2, w / 2, (n, 1))
    z = np.random.uniform(-h / 2, h / 2, (n, 1))

    faces = [
        np.hstack([x, y, np.full_like(z, h / 2)]),  # 顶面
        np.hstack([x, y, np.full_like(z, -h / 2)]),  # 底面
        np.hstack([x, np.full_like(y, w / 2), z]),  # 前面
        np.hstack([x, np.full_like(y, -w / 2), z]),  # 后面
        np.hstack([np.full_like(x, l / 2), y, z]),  # 右面
        np.hstack([np.full_like(x, -l / 2), y, z])  # 左面
    ]
    return np.vstack(faces)


def sample_cylinder(radius, length, num_points=None):
    """
    从圆柱表面均匀采样点云。
    :param radius: 圆柱半径
    :param length: 圆柱长度
    :param num_points: 采样点数
    :return: (N,3) 点云
    """
    # 圆柱体三个部分：侧面、顶部、底部
    if num_points is None:
        num_points = [0, 0, 0]
    num_side, num_top, num_bottom = num_points  # 拆解为侧面、顶部、底部的点数

    # 侧面点 (均匀分布在圆柱侧面)
    theta_side = np.random.uniform(0, 2 * np.pi, (num_side, 1))
    z_side = np.random.uniform(-length / 2, length / 2, (num_side, 1))
    x_side = radius * np.cos(theta_side)
    y_side = radius * np.sin(theta_side)
    side = np.hstack([x_side, y_side, z_side])

    # 顶面点 (均匀填充圆)
    theta_top = np.random.uniform(0, 2 * np.pi, (num_top, 1))
    r_top = np.sqrt(np.random.uniform(0, radius ** 2, (num_top, 1)))  # 采用均匀填充方法
    x_top = r_top * np.cos(theta_top)
    y_top = r_top * np.sin(theta_top)
    z_top = np.full((num_top, 1), length / 2)
    top = np.hstack([x_top, y_top, z_top])

    # 底面点 (均匀填充圆)
    theta_bottom = np.random.uniform(0, 2 * np.pi, (num_bottom, 1))
    r_bottom = np.sqrt(np.random.uniform(0, radius ** 2, (num_bottom, 1)))  # 采用均匀填充方法
    x_bottom = r_bottom * np.cos(theta_bottom)
    y_bottom = r_bottom * np.sin(theta_bottom)
    z_bottom = np.full((num_bottom, 1), -length / 2)
    bottom = np.hstack([x_bottom, y_bottom, z_bottom])

    return np.vstack([side, top, bottom])


def transform_points(points, transform):
    """
    应用 4x4 变换矩阵到点云。
    :param points: (N,3) numpy 数组
    :param transform: (4,4) numpy 数组
    :return: 变换后的 (N,3) 点云
    """
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack([points, ones])
    points_transformed = (transform @ points_hom.T).T[:, :3]
    return points_transformed


def load_urdf_as_pointcloud(urdf_path, n_points_per_primitive=1000):
    """
    读取URDF模型，并对几何原语进行表面点云采样。
    :param urdf_path: URDF 文件路径
    :param n_points_per_primitive: 每个几何体的采样点数
    :return: open3d.geometry.PointCloud 对象
    """
    robot = URDF.load(urdf_path)
    all_points = []

    for link in robot.links:
        for visual in link.visuals:
            geom = visual.geometry
            origin = visual.origin  # (4x4 变换矩阵)

            if geom.box:
                continue  # 忽略盒子
                # size = geom.box.size
                # sampled_points = sample_box(size, n_points_per_primitive)
            elif geom.cylinder:
                radius = geom.cylinder.radius
                length = geom.cylinder.length
                sampled_points = sample_cylinder(radius, length, n_points_per_primitive)
            else:
                continue  # 其他类型不处理

            transformed_points = transform_points(sampled_points, origin)
            all_points.append(transformed_points)

    if len(all_points) == 0:
        print("No valid geometry found in URDF.")
        return None

    all_points = np.vstack(all_points)
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(all_points)

    return pc_o3d


import xml.etree.ElementTree as ET


def remove_inertial_from_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # 遍历所有 <link>，删除其中的 <inertial>
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            link.remove(inertial)  # 直接从 <link> 里删除 <inertial>

    # 生成新的 URDF 文件
    new_urdf_path = urdf_path.replace(".urdf", "_no_inertial.urdf")
    tree.write(new_urdf_path)
    return new_urdf_path


if __name__ == "__main__":
    urdf_file = "./assets/screwdriver/screwdriver.urdf"
    cleaned_urdf = remove_inertial_from_urdf(urdf_file)

    model_pcd = load_urdf_as_pointcloud(cleaned_urdf, [500, 200, 200])

    if model_pcd is not None:
        o3d.visualization.draw_geometries([model_pcd])
    # SAVE THE POINT CLOUD HERE
    o3d.io.write_point_cloud("screwdriver_pcd.ply", model_pcd)
