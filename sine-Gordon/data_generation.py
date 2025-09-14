import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import svds
from rbf.sputils import expand_rows, expand_cols
from rbf.pde.fd import weight_matrix

def sech(x):
    return 1 / np.cosh(x)


def boundary_matrix(XY):
    XY_d, XY_r, XY_u, XY_l, XY_int = [], [], [], [], []
    idx_d, idx_r, idx_u, idx_l, idx_int = [], [], [], [], []
    for idx, xy in enumerate(XY):
        if xy[1] == y_l:
            idx_d.append(idx)
            XY_d.append(xy)
        elif xy[0] == x_r:
            idx_r.append(idx)
            XY_r.append(xy)
        elif xy[1] == y_r:
            idx_u.append(idx)
            XY_u.append(xy)
        elif xy[0] == x_l:
            idx_l.append(idx)
            XY_l.append(xy)
        else:
            idx_int.append(idx)
            XY_int.append(xy)

    I = eye(len(idx_int))  # Ni * Ni
    I = expand_rows(I, idx_int, N)  # N * Ni
    I = expand_cols(I, idx_int, N)  # N * N
    W = weight_matrix(x=XY_int, diffs=[[2, 0], [0, 2]], coeffs=[1, 1], p=XY, n=n, phi=phi, eps=eps, order=order)
    A = expand_rows(W, idx_int, N)
    B = - expand_rows(W, idx_int, N) * dt * dt / 4
    C = expand_rows(W, idx_int, N) * dt * dt / 2
    C += 2 * I  # Ni * Ni -> N * Ni -> N * N
    D = expand_rows(W, idx_int, N) * dt * dt / 4

    W = weight_matrix(x=XY_d, diffs=[[0, 1]], coeffs=[-1], p=XY, n=n, phi=phi, eps=eps, order=order)
    A += expand_rows(W, idx_d, N)
    B += expand_rows(W, idx_d, N)

    W = weight_matrix(x=XY_r, diffs=[[1, 0]], coeffs=[1], p=XY, n=n, phi=phi, eps=eps, order=order)
    A += expand_rows(W, idx_r, N)
    B += expand_rows(W, idx_r, N)

    W = weight_matrix(x=XY_u, diffs=[[0, 1]], coeffs=[1], p=XY, n=n, phi=phi, eps=eps, order=order)
    A += expand_rows(W, idx_u, N)
    B += expand_rows(W, idx_u, N)

    W = weight_matrix(x=XY_l, diffs=[[1, 0]], coeffs=[-1], p=XY, n=n, phi=phi, eps=eps, order=order)
    A += expand_rows(W, idx_l, N)
    B += expand_rows(W, idx_l, N)

    return A, B, C, D, I, set(idx_int)


def RBF_FD(mu_1, mu_2):
    B_mu = B + (1 + mu_1 * dt / 2) * I
    D_mu = D - (1 + mu_1 * dt / 2) * I
    u = np.zeros((N, N_t))
    u[:, 0] = f
    for t in range(1, N_t):
        if t == 1:
            s = [mu_2 * np.sin(f[idx]) if idx in idx_int else 0 for idx, xy in enumerate(XY)]
            s = np.array(s)
            rhs = C.dot(f) + dt * dt * s
            lhs = B_mu - D_mu
            u[:, 1] = spsolve(lhs, rhs)
        else:
            s = [mu_2 * np.sin(u[idx, t - 1]) if idx in idx_int else 0 for idx, xy in enumerate(XY)]
            s = np.array(s)
            rhs = C.dot(u[:, t - 1]) + D_mu.dot(u[:, t - 2]) + dt * dt * s
            u[:, t] = spsolve(B_mu, rhs)
    return u


def snapshot_mat(MU_1, MU_2):
    S = np.zeros((0, N, N_t))
    for mu_1 in MU_1:
        for mu_2 in MU_2:
            print(mu_1, mu_2)
            u_hf = RBF_FD(mu_1, mu_2)
            S = np.concatenate((S, u_hf[np.newaxis, :, :]), axis=0)
    return S

def POD(S, d):
    S = np.transpose(S, axes=(1, 2, 0))  #(n_s, N, N_t) -> (N, N_t, n_s)
    S = S.reshape(-1, S.shape[-1])
    U, Sig, _ = svds(S, k=d, return_singular_vectors='u')
    return U  # N*N_t, d


dx = dy = 0.25
dt = 0.1
x_l = y_l = -7
x_r = y_r = 7
t_l = 0
t_r = 10
N_x = int((x_r - x_l) / dx) + 1
N_y = int((y_r - y_l) / dy) + 1
N_t = int((t_r - t_l) / dt) + 1
N = N_x * N_y
xgrid, ygrid = np.meshgrid(np.linspace(x_l, x_r, N_x), np.linspace(y_l, y_r, N_y))
tgrid = np.linspace(t_l, t_r, N_t)
XY = np.array([xgrid.flatten(), ygrid.flatten()]).T

n = 25
phi = 'imq'
eps = 1
order = -1

n_s_1 = 10
n_s_2 = 10
n_s = n_s_1 * n_s_2
MU_1 = np.linspace(0.01, 0.1, n_s_1)
MU_2 = np.linspace(-0.2, -2, n_s_2)
d = 15

A, B, C, D, I, idx_int = boundary_matrix(XY)

f = [4 * np.arctan(np.exp(xy[0] + 1 - 2 * sech(xy[1] + 7) - 2 * sech(xy[1] - 7))) for xy in XY]
f = np.array(f)

try:
    S = np.load(f'./snapshot_mat_ns{n_s}.npy')
except IOError:
    S = snapshot_mat(MU_1, MU_2)
    np.save(f'./snapshot_mat_ns{n_s}.npy', S)
try:
    V = np.load(f'./POD_mat_ns{n_s}_d{d}.npy')
except IOError:
    V = POD(S, d=d) # N*N_t, d
    np.save(f'./POD_mat_ns{n_s}_d{d}.npy', V)

n_data_1 = 32
n_data_2 = 32
n_data = n_data_1 * n_data_2
MU_1 = np.linspace(0.01, 0.1, n_data_1)
MU_2 = np.linspace(-0.2, -2, n_data_2)
X = np.zeros((0, 2))
Y = np.zeros((0, V.shape[1]))
U_hf = np.zeros((0, N, N_t))

for mu_1 in MU_1:
    print(mu_1)
    for mu_2 in MU_2:
        x = np.array([mu_1, mu_2])
        u_hf = RBF_FD(mu_1, mu_2)  #(N, N_t)
        y = V.T.dot(u_hf.flatten())
        X = np.concatenate((X, x[np.newaxis, :]))
        Y = np.concatenate((Y, y[np.newaxis, :]))
        U_hf = np.concatenate((U_hf, u_hf[np.newaxis, :, :]))
np.savez(f'./data_set_ns{n_s}_d{d}_ndata{n_data}.npz', X=X, Y=Y, U_hf=U_hf)
print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("U_hf.shape:", U_hf.shape)
