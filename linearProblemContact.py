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
    contact_points: dict {finger: (3,)}
    返回 G ∈ ℝ^{6 × 3n}
    """
    G_blocks = []
    for finger in ['index', 'middle', 'thumb']:
        r = contact_points[finger] - p_com           # (3,)
        G_i = np.vstack([np.eye(3), skew(r)])        # (6,3)
        G_blocks.append(G_i)
    return np.hstack(G_blocks)                       # (6, 9)


def build_friction_cone_frame_basis(contact_frame, mu=0.3, k=4):
    """
    Build primitive friction directions in local frame (z is normal)
    :param contact_frame: 3x3 rotation matrix (columns = x, y, z)
    :param mu: friction coefficient
    :param k: number of tangential directions (excluding normal)
    :return: F_i ∈ ℝ^{3×(k+1)} in world frame
    """
    local_dirs = []

    # Add normal direction (unit z)
    normal_local = np.array([0, 0, 1])
    local_dirs.append(normal_local)

    # Tangential directions in local xy-plane
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    for theta in angles:
        t_local = np.array([np.cos(theta), np.sin(theta), mu])
        t_local = t_local / np.linalg.norm(t_local)  # normalize
        local_dirs.append(t_local)

    # Transform to world frame
    F_i = contact_frame @ np.stack(local_dirs, axis=1)
    return F_i


def build_full_F(contact_frames, mu=0.5, k=4):
    """
    contact_frames: dict from finger → contact_frame (3x3)
    returns F ∈ ℝ^{3n × n(k+1)} (block-diagonal)
    """
    fingers = ["index", "middle", "thumb"]
    blocks = []

    for finger in fingers:
        F_i = build_friction_cone_frame_basis(contact_frames[finger], mu=mu, k=k)
        blocks.append(F_i)

    blocks = [F_i.T for F_i in blocks]
    # F_i: 原来形状是 (3 × (k+1))，列向量是各摩擦面元方向
    # F_i.T: 变成 ((k+1) × 3)，行向量是各面元方向，方便写 F_i.T @ f_i ≥ η
    F = scipy.linalg.block_diag(*blocks)  # shape (3n, (k+1)n)
    return F

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
    def __init__(self, G, F, e, n_f):
        self.G = G                  # (6 x 3n) grasp matrix
        self.F = F                  # (3n x 3n) block-diagonal friction cone matrix
        self.e = e                  # (3n x 1) normal force selection vector
        self.n_f = n_f              # scalar upper bound on normal force
        self.n_c = G.shape[1] // 3  # number of contact points
        assert G.shape == (6, 3 * self.n_c)
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

    def normal_force_constraint(self):
        return self.e.T @ self.lambda_ == self.n_f

    def solve(self):
        constraints = [
            self.static_equilibrium(),
            self.friction_cone(),
            self.positive_constraint(),
            self.normal_force_constraint()
        ]
        objective = cp.Maximize(self.eta)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)

        return {
            "status": problem.status,
            "optimal_eta": self.eta.value,
            "lambda": self.lambda_.value
        }


def run_linear_program(object_cloud, contact_pts_world, frame_data):
    p_com = get_reference_point(object_cloud)
    G = build_grasp_matrix(contact_pts_world, p_com)

    contact_frames = {f: frame_data[f]['contact_frame'] for f in frame_data}
    F = build_full_F(contact_frames, mu=0.5, k=4)

    normals = {f: frame_data[f]['contact_frame'][:, 2] for f in frame_data}
    e = build_e_vector(list(normals.values()))
    n_f = 1.0

    lp = LinearProgram(G, F, e, n_f)
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