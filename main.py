import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
import open3d as o3d

gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# 创建环境
env = gym.create_env(sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 1)

# 加载 URDF 机器人
asset_root = "./assets/screwdriver/"
urdf_file = "screwdriver.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset = gym.load_asset(sim, asset_root, urdf_file, asset_options)

# 将机器人添加到环境中
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0)  # 设置初始位置
actor_handle = gym.create_actor(env, asset, pose, "robot", 0, 1)

# 配置虚拟相机
camera_props = gymapi.CameraProperties()
camera_props.width = 640
camera_props.height = 480
camera_props.horizontal_fov = 90
camera_props.near_plane = 0.1  # 最近深度
camera_props.far_plane = 10.0  # 最远深度
camera_handle = gym.create_camera_sensor(env, camera_props)

# 设置相机位置和方向
camera_pose = gymapi.Transform()
camera_pose.p = gymapi.Vec3(1.5, 1.5, 1.0)  # 相机位置
camera_pose.r = gymapi.Quat.from_euler_zyx(-np.pi / 4, 0, np.pi / 4)  # 相机旋转
gym.set_camera_transform(camera_handle, env, camera_pose)

# 仿真步进
gym.simulate(sim)
gym.fetch_results(sim, True)

# 渲染相机图像
gym.render_all_camera_sensors(sim)

# 获取深度图
depth_image = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_DEPTH)
depth_image = np.array(depth_image, dtype=np.float32).reshape(camera_props.height, camera_props.width)

# 打印深度图信息
print(f"Captured depth image with shape: {depth_image.shape}")

# 过滤无效深度值
valid_mask = np.isfinite(depth_image) & (depth_image > 0)
z = depth_image.flatten()[valid_mask]

# 相机内参
fx = fy = 320  # 焦距
cx = camera_props.width / 2  # 图像中心点 x
cy = camera_props.height / 2  # 图像中心点 y

# 只保留有效点的 x 和 y
x, y = np.meshgrid(np.arange(camera_props.width), np.arange(camera_props.height))
x = (x.flatten()[valid_mask] - cx) * z / fx
y = (y.flatten()[valid_mask] - cy) * z / fy

# 生成点云
points = np.stack((x, y, z), axis=-1)

# 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 保存点云
o3d.io.write_point_cloud("pointcloud.ply", pcd)
print("Point cloud saved to pointcloud.ply")

# 可视化点云
o3d.visualization.draw_geometries([pcd])

# 关闭 Isaac Gym
gym.destroy_sim(sim)
