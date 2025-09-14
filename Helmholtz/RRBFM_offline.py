import os
import h5py
import numpy as np
from scipy.sparse.linalg import spsolve
from rbf.sputils import expand_rows
from rbf.pde.fd import weight_matrix

def gram_schmidt(A):
    Q = np.zeros_like(A, dtype=np.float64)
    for i in range(A.shape[1]):
        v = A[:, i]
        for j in range(i):
            q = Q[:, j]
            v = v - np.dot(q, v) * q
        Q[:, i] = v / np.linalg.norm(v)
    return Q


discretization = h5py.File('./discretization.mat', mode='r')
XY = np.array(discretization['xy']).T
BDY = np.array(discretization['bdy']).T
NODE = np.concatenate((XY, BDY))
N = NODE.shape[0]



n = 50
phi = 'imq'
eps = 3.0
order = -1

f = [-10 * np.sin(8 * xy[0] * (xy[1] - 1)) for xy in XY] + [0 for bdy in BDY]
f = np.array(f)

n_s_1 = 100
n_s_2 = 100
n_s = n_s_1 * n_s_2
MU_1 = np.linspace(0.1, 4, n_s_1)
MU_2 = np.linspace(0, 2, n_s_2)

eps_POD = 1e-6
V = np.load(f'./POD_mat_ns{100}_epsPOD{eps_POD}.npy')
RRB_dim = V.shape[1]

try:
    data = np.load(f'./RRBFM_offline_ns{n_s}_RRBdim{RRB_dim}.npz')
    L = data['L']
except:
    L = [[None for i in range(n_s_2)] for i in range(n_s_1)]
    A_boundary = weight_matrix(
        x=BDY,
        p=NODE,
        n=1,
        diffs=[0, 0])
    A_boundary = expand_rows(A_boundary, [i for i in range(XY.shape[0], N)], N)
    for i, mu_1 in enumerate(MU_1):
        print(i, mu_1)
        for j, mu_2 in enumerate(MU_2):
            A_interior = weight_matrix(
                x=XY,
                p=NODE,
                n=n,
                diffs=[[2, 0], [0, 2], [0, 0]],
                coeffs=[-1, -mu_1, -mu_2],
                phi=phi,
                eps=eps,
                order=order)
            A = expand_rows(A_interior, [i for i in range(XY.shape[0])], N) + A_boundary
            L[i][j] = A.copy()
    np.savez(f'./RRBFM_offline_ns{n_s}_RRBdim{RRB_dim}.npz', L=L)

RRB_mat = np.empty((N, 0))
RRB_mu = [(MU_1[n_s_1 // 2 - 1], MU_2[n_s_2 // 2 - 1])]
u_hf = spsolve(L[n_s_1 // 2 - 1][n_s_2 // 2 - 1], f)
RRB_mat = np.concatenate((RRB_mat, u_hf[:, np.newaxis]), axis=1)
for _ in range(1, RRB_dim):
    est_err_max = 0
    print(_)
    for i, mu_1 in enumerate(MU_1):
        for j, mu_2 in enumerate(MU_2):
            if (mu_1, mu_2) in RRB_mu:
                continue
            A = L[i][j]
            M = A @ RRB_mat
            MM = M.T @ M
            Mf = M.T.dot(f)
            if _ == 1:
                coef = Mf / MM[0, 0]
            else:
                coef = np.linalg.solve(MM, Mf)
            u = RRB_mat.dot(coef)
            est_err = np.linalg.norm(f - A.dot(u))
            if est_err > est_err_max:
                A_nxt = L[i][j]
                mu_1_nxt = mu_1
                mu_2_nxt = mu_2
                est_err_max = est_err
    u = spsolve(A_nxt, f)
    RRB_mat = np.concatenate((RRB_mat, u[:, np.newaxis]), axis=1)
    RRB_mat = gram_schmidt(RRB_mat)
    RRB_mu.append((mu_1_nxt, mu_2_nxt))
print(RRB_mat.shape)
np.savez(f'./RRBFM_offline_ns{n_s}_RRBdim{RRB_dim}.npz', L=L, RRB_mu=RRB_mu, RRB_mat=RRB_mat)
