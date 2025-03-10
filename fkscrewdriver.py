def fk_screwdriver(q1, q2, q3):
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
    '''
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
    '''

    # base->cap = R_x(q1)*R_y(q2)*R_z(q3)*T_sb*T_bc
    T_base_cap = Rx @ Ry @ Rz
    return T_base_cap

import numpy as np
from scipy.spatial.transform import Rotation as R


q1 = 0.00015042
q2 = 0.00043013
q3 = -0.00360273
T_fk = fk_screwdriver(q1, q2, q3)


q1_gym = 0.0090
q2_gym = 0.0040
q3_gym = -0.1880
T_fk_gym = fk_screwdriver(q1_gym, q2_gym, q3_gym)

T_icp = np.array([[9.82924930e-01,  1.82793637e-01,  2.10965162e-02,  7.58128333e-04],
                  [-1.82626510e-01,  9.83135391e-01, -9.61041699e-03, -1.06445113e-03],
                  [-2.24974693e-02,  5.59354129e-03,  9.99731245e-01,  5.66213402e-04],
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
