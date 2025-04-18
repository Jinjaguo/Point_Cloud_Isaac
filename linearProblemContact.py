import numpy as np
import cvxpy as cp
import scipy

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

def build_grasp_matrix(contact_points, p_com):
    """
    返回 G ∈ ℝ^{3 × 3n}，只管 x、y 平移和 z 轴转动
    """
    G_blocks = []
    for finger in ['index', 'middle', 'thumb']:
        r = contact_points[finger] - p_com        # (3,)
        # 平面力 [fx, fy] & 转矩 τz = r × f · z
        G_i = np.vstack([
            np.eye(2, 3),                         # 2×3 取 fx, fy
            skew(r)[2, :].reshape(1, 3)           # 1×3 取 (r×)z
        ])                                        # 3×3
        G_blocks.append(G_i)
    return np.hstack(G_blocks)


def build_friction_cone_rows(contact_frame, mu=0.3, k=4):
    """
    Build primitive friction directions in local frame (z is normal)
    :param contact_frame: 3x3 rotation matrix (columns = x, y, z)
    :param mu: friction coefficient
    :param k: number of tangential directions (excluding normal)
    :return: F_i ∈ ℝ^{3×(k+1)} in world frame
    """
    dirs = [[0, 0, 1]]
    for theta in np.linspace(0, 2 * np.pi, k, endpoint=False):
        dirs.append([mu * np.cos(theta), mu * np.sin(theta), 1])
    Fi = np.asarray(dirs)  # (k+1)×3  在接触局部系
    return Fi @ contact_frame.T



def build_full_F(contact_frames, mu=0.3, k=4):
    fingers = ["index", "middle", "thumb"]
    F_blocks = [build_friction_cone_rows(contact_frames[f], mu, k)
                for f in fingers]
    return scipy.linalg.block_diag(*F_blocks)   # ((k+1)n, 3n)

def build_e_vector(normals):
    """
    normals: list[(3,)] 按 index/middle/thumb 排
    返回 e ∈ ℝ^{3n×1}, 使 eᵀ λ = Σ n_i·f_i
    """
    rows = []
    for n in normals:
        rows.extend(n.tolist())
    e = np.array(rows).reshape(-1,1)
    return e


class LinearProgram:
    def __init__(self, G, F, e, n_f_vec, eta_max=None):
        self.G = G                  # (3 x 3n) grasp matrix
        self.F = F                  # ((k+1)n x 3n) block-diagonal friction cone matrix
        self.e = e                  # (3n x 1) normal force selection vector
        self.n_c = G.shape[1] // 3  # number of contact points
        self.n_f_vec = cp.Constant(n_f_vec.reshape(-1, 1))  # 每指上限 (n_c,1)
        self.eta_max = eta_max

        assert G.shape == (3, 3 * self.n_c)
        assert e.shape == (3 * self.n_c, 1)

        # variables
        self.lambda_ = cp.Variable((3 * self.n_c, 1))
        self.eta = cp.Variable()

    def static_equilibrium(self):
        return self.G @ self.lambda_ == 0

    def friction_cone(self):
        ones = np.ones((self.F.shape[0], 1))
        return self.F @ self.lambda_ - ones * self.eta >= 0

    def positive_constraint(self):
        return self.eta >= 0


    def solve(self):
        constraints = [
            self.static_equilibrium(),
            self.friction_cone(),
            self.positive_constraint(),
        ]
        fz = self.lambda_[2::3]  # 取每 3 个里的第 3 个分量
        constraints += [
            fz >= 0,  # 压力向内
            fz <= self.n_f_vec  # 每指上限 (cp.Constant or np.array shape (n_c,1))
        ]

        if self.eta_max is not None:
            constraints.append(self.eta <= self.eta_max)

        objective = cp.Maximize(self.eta)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)

        return {
            "status": problem.status,
            "optimal_eta": self.eta.value,
            "lambda": self.lambda_.value
        }


def run_linear_program(object_cloud, contact_pts_world, frame_data, n_f_vec, eta_max=0.1):
    p_com = get_reference_point(object_cloud)
    G = build_grasp_matrix(contact_pts_world, p_com)

    contact_frames = {f: frame_data[f]['contact_frame'] for f in frame_data}
    F = build_full_F(contact_frames)

    normals = {f: frame_data[f]['contact_frame'][:, 2] for f in frame_data}
    e = build_e_vector(list(normals.values()))


    lp = LinearProgram(G, F, e, n_f_vec, eta_max=eta_max)
    result = lp.solve()
    print(result['status'], result['optimal_eta'])

    return result


def check_result(result):
    if result["status"] == "optimal":
        # λ is a (3n×1) vector
        lambda_vec = result["lambda"].reshape(-1)  # flatten to 1D vector
        eta_opt = result["optimal_eta"]
        print(f"+++>>> success! λ = {lambda_vec}，η = {eta_opt}")

        # decompose λ into per-finger components
        n_contacts = 3  # 3
        lambda_per_finger = lambda_vec.reshape(n_contacts, 3)
        fingers = ["index", "middle", "thumb"]
        lambda_dict = {
            fingers[i]: lambda_per_finger[i]
            for i in range(n_contacts)
        }
        print("contact forces per finger：", lambda_dict)
    else:
        print("!!!>>> Failed! Status：", result["status"])