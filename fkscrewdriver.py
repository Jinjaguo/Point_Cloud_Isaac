def fk_screwdriver(q1, q2, q3, cap_angle):
    """
    输入4个角度(弧度)：joint_1=x, joint_2=y, joint_3=z, cap_joint=z(带offset)
    返回一个4x4变换矩阵：base/table系 -> 螺丝刀指定link(比如cap link)
    """
    import numpy as np
    from math import sin, cos

    # 1. R_x(q1)
    Rx = np.array([
        [1,         0,          0,      0],
        [0,  cos(q1),  -sin(q1),  0],
        [0,  sin(q1),   cos(q1),  0],
        [0,        0,          0,      1]
    ])

    # 2. R_y(q2)
    Ry = np.array([
        [ cos(q2), 0, sin(q2), 0],
        [ 0,       1, 0,       0],
        [-sin(q2), 0, cos(q2), 0],
        [ 0,       0, 0,       1]
    ])

    # 3. R_z(q3)
    Rz = np.array([
        [ cos(q3), -sin(q3), 0, 0],
        [ sin(q3),  cos(q3), 0, 0],
        [ 0,        0,       1, 0],
        [ 0,        0,       0, 1]
    ])

    # stick -> body (fixed joint)
    T_sb = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.1],  # offset in Z=0.1
        [0, 0, 0, 1]
    ])

    # 4. cap_joint: 先平移(0,0,0.1)，再绕z(cap_angle)
    T_bc = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.1],
        [0, 0, 0, 1]
    ])
    Rz4 = np.array([
        [ cos(cap_angle), -sin(cap_angle), 0, 0],
        [ sin(cap_angle),  cos(cap_angle), 0, 0],
        [ 0,               0,              1, 0],
        [ 0,               0,              0, 1]
    ])
    T_bc = T_bc @ Rz4

    # base->cap = R_x(q1)*R_y(q2)*R_z(q3)*T_sb*T_bc
    T_base_cap = Rx @ Ry @ Rz @ T_sb @ T_bc
    return T_base_cap

import numpy as np
from scipy.spatial.transform import Rotation as R


q1 = 1.01278074
q2 = -0.67826671
q3 = 0.55064581
cap = 0.56202943
T_fk = fk_screwdriver(q1, q2, q3, cap)


q1_gym = 0.00160723
q2_gym = 0.01335732
q3_gym = -0.07890692
cap_gym = 1.2259915
T_fk_gym = fk_screwdriver(q1, q2, q3, cap)

T_icp = np.array([[-0.38133537, -0.74663514,  0.54508651, -0.33603399],
                  [ 0.48384859, -0.66362882, -0.57051497, -0.3743772 ],
                  [ 0.78770164,  0.0461818 ,  0.61432351,  0.16754782],
                  [ 0.        ,  0.        ,  0.        ,  1.        ]])


T_delta = np.linalg.inv(T_icp) @ T_fk_gym
print(T_delta)

# 1. 比较平移误差
pos_fk  = T_fk[:3, 3]
pos_icp = T_icp[:3, 3]
pos_gym = T_fk_gym[:3, 3]
trans_error = np.linalg.norm(pos_fk - pos_icp)
trans_error_gym = np.linalg.norm(pos_icp - pos_gym)
trans_err_12 = np.linalg.norm(pos_fk - pos_gym)
print("Translation error:", trans_error, " meters")
print("Translation error (gym):", trans_error_gym, " meters")
print("Translation error (1-2):", trans_err_12, " meters")

# 2. 比较旋转误差
R_fk = R.from_matrix(T_fk[:3,:3])
R_icp= R.from_matrix(T_icp[:3,:3])
R_gym= R.from_matrix(T_fk_gym[:3,:3])
# 用旋转向量差
rotvec_fk = R_fk.as_rotvec()
rotvec_icp= R_icp.as_rotvec()
rotvec_gym= R_gym.as_rotvec()
rot_err = np.linalg.norm(rotvec_fk - rotvec_icp)
rot_err_gym = np.linalg.norm(rotvec_icp - rotvec_gym)
rot_err_12 = np.linalg.norm(rotvec_fk - rotvec_gym)
print("Rotation error:", rot_err, " rad (=", np.degrees(rot_err), "deg)")
print("Rotation error (gym):", rot_err_gym, " rad (=", np.degrees(rot_err_gym), "deg)")
print("Rotation error (1-2):", rot_err_12, " rad (=", np.degrees(rot_err_12), "deg)")
