import numpy as np


def rotation_matrix_from_euler_xyz(euler_angles):
    """
    将欧拉角转换为旋转矩阵，旋转顺序为 x-y-z.
    参数:
      euler_angles: 包含 [alpha, beta, gamma]（单位为弧度）的列表或数组，
                    分别表示绕 x, y, z 轴的旋转角度.
    返回:
      对应的 3x3 旋转矩阵.
    """
    alpha, beta, gamma = euler_angles

    # 绕 x 轴的旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])

    # 绕 y 轴的旋转矩阵
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])

    # 绕 z 轴的旋转矩阵
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])

    # 按照顺序先绕 x 轴，再绕 y 轴，最后绕 z 轴
    R = Rx @ Ry @ Rz
    return R


# 定义两个欧拉角（单位：弧度）
euler = [0.0021, 0.0133, -0.0790]

# 计算对应的旋转矩阵
R = rotation_matrix_from_euler_xyz(euler)
T_gym = np.zeros((4, 4))
T_gym[:3, :3] = R
T_gym[:3, 3] = [1.3395e-03, -1.5743e-04,  1.3050e+00]
T_gym[3, 3] = 1
print(T_gym)

T_icp = np.array([[-0.38719548, -0.73902718,  0.55128802, -0.33183914],
                  [0.47972286, -0.67209856, -0.56404742, -0.37811116],
                  [0.78736626,  0.04606885,  0.61476177,  0.16819135],
                  [0,          0,          0,          1.        ]])

T = np.linalg.inv(T_icp) @ T_gym

# 输出结果
print("旋转矩阵 R1:")
print(T_gym)
print("\n旋转矩阵 R2:")
print(T_icp)
print("\n相对变换矩阵 T:")
print(T)

# 验证结果
print(np.linalg.norm(T_icp @  T - T_gym))


import numpy as np
from scipy.spatial.transform import Rotation as R

# 定义两个变换矩阵
T_gym = np.array([[-0.44076724,  0.45375013,  0.7800964,   0.94625626],
                  [-0.68998848, -0.72274815,  0.0393831, -0.44557337],
                  [ 0.58168332, -0.5212926,   0.62441857,  0.66100806],
                  [ 0.,          0.,          0.,          1.        ]])

T_icp = np.array([[-0.41842976,  0.45401763,  0.78662858,  0.95474958],
                  [-0.69058019, -0.72158472,  0.04913741, -0.44466947],
                  [ 0.58992842, -0.52266955,  0.61546812,  0.65980761],
                  [ 0.,          0.,          0.,          1.        ]])

np.set_printoptions(precision=20, suppress=False, floatmode='maxprec_equal')
T = np.mean([T_gym, T_icp], axis=0)
print(np.array2string(T, precision=15, suppress_small=False))
for row in T:
    print(["{:.20f}".format(val) for val in row])


# 提取旋转部分和平移部分
R_gym = T_gym[:3, :3]
t_gym = T_gym[:3, 3]

R_icp = T_icp[:3, :3]
t_icp = T_icp[:3, 3]

# 计算平移差异
translation_difference = np.linalg.norm(t_icp - t_gym)

# 计算旋转差异
# 使用 scipy.spatial.transform.Rotation 来计算旋转差异
rotation_gym = R.from_matrix(R_gym)
rotation_icp = R.from_matrix(R_icp)

# 计算两个旋转之间的差异角（单位为弧度）
rotation_difference = rotation_gym.inv() * rotation_icp
angle_difference = rotation_difference.magnitude()

# 输出结果
print("平移差异: ", translation_difference)
print("旋转差异（弧度）: ", angle_difference)
