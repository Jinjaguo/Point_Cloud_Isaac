import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
import open3d as o3d

# 获取Gym实例
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.substeps = 2  # 可以根据需要调整子步数
sim_params.dt = 0.01     # 可以根据需要调整时间步长

# 创建仿真
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# 检查是否成功创建仿真
if sim is None:
    print("Failed to create sim")
    quit()

# 创建环境
env = gym.create_env(sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 1)

# 加载URDF机器人模型
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

# 创建并启动可视化
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.subscribe_viewer_camera_image(sim, viewer, camera_handle)

# 创建Open3D的可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window()

# 进行仿真步进并显示相机画面
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(sim, viewer, True)

    # 获取相机图像
    depth_image = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_DEPTH)
    depth_image = np.array(depth_image, dtype=np.float32).reshape(camera_props.height, camera_props.width)

    # 将深度图转换为适合Open3D显示的格式
    depth_image_o3d = o3d.geometry.Image((depth_image / np.max(depth_image) * 255).astype(np.uint8))

    # 清除之前的几何体
    vis.clear_geometries()

    # 添加新的几何体
    vis.add_geometry(depth_image_o3d)

    # 更新可视化窗口
    vis.update_renderer()

# 关闭Open3D的可视化窗口
vis.destroy_window()

# 关闭Isaac Gym的可视化
gym.destroy_viewer(sim, viewer)

# 销毁仿真
gym.destroy_sim(sim)
