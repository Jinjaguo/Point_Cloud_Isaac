import numpy as np

def get_reference_point(point_cloud):
    """
    :param point_cloud: (N, 3) numpy array, each row is a 3D point in world/object frame
    :return: p_com, the geometric center (centroid) of the point cloudm used for grasp matrix construction
    """
    p_com = np.mean(point_cloud, axis=0)
    return p_com


def skew(v):
    """反对称矩阵 [v]_x."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])



class DSAttractor:
    def __init__(self, opt_res, contact_points):
        self.fingers = ["index", "middle", "thumb"]
        self.x1 = None
        self.x2 = {}
        self.kt = 300
        self.kn = 600
        self.kx = np.diag([self.kt, self.kt, self.kn])
        self.kx_inv = np.linalg.inv(self.kx)

        self.opt_res = opt_res
        self.contact_points = contact_points


    def get_grasp_attractor(self, pcd):

        p_com = get_reference_point(pcd)  # 计算点云的质心
        lam_vec = self.opt_res["lambda"].reshape(-1)  # shape (3*n_f, )
        lam_per_finger = lam_vec.reshape(3, 3)  # [[fx,fy,fz],...]

        for i, finger in enumerate(self.fingers):
            lam_i = lam_per_finger[i].reshape(3, 1)  # (3×1)

            # 构建 G_i ∈ ℝ^{3×3}（平面只取 fx,fy,τz→当作 Δx 的第三分量）
            r = self.contact_points[finger] - p_com  # (3,)
            G_i = np.vstack([
                np.eye(2, 3),  # 2×3 取 fx, fy
                skew(r)[2, :].reshape(1, 3)  # 1×3 取 τ_z = (r×f)_z
            ])  # 3×3

            # Δx = Kx⁻¹ · (G_i @ λ_i)
            delta_x = (self.kx_inv @ (G_i @ lam_i)).flatten()
            self.x1 = self.contact_points[finger]
            x2 = self.x1 + delta_x.reshape(3,)
            self.x2[finger] = x2

        return self.x2