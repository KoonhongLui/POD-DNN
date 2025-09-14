import h5py
import numpy as np
from rbf.sputils import expand_rows
from rbf.pde.fd import weight_matrix

def Relative_l2_error(y_pred, y_true):
    l2_error = np.linalg.norm(y_pred - y_true, ord=2)
    l2_norm = np.linalg.norm(y_true, ord=2)
    eps = 1e-8
    relative_l2 = l2_error / (l2_norm + eps)
    return relative_l2

discretization = h5py.File('./discretization.mat', mode='r')
XY = np.array(discretization['xy']).T
BDY = np.array(discretization['bdy']).T
NODE = np.concatenate((XY, BDY))
N = NODE.shape[0]

f = [-10 * np.sin(8 * xy[0] * (xy[1] - 1)) for xy in XY] + [0 for bdy in BDY]
f = np.array(f)

n = 50
phi = 'imq'
eps = 3.0
order = -1

n_s_1 = 100
n_s_2 = 100
n_s = n_s_1 * n_s_2
eps_POD = 1e-06
test_size = 2000
data = np.load(f"./test_dataset_ns{100}_epsPOD{eps_POD}_testsize{test_size}.npz")
X = data['X_test']
Y = data['Y_test']
U_hf = data['U_hf_test']
n_data = X.shape[0]
RRB_dim = Y.shape[1]
RRB_mat = np.load(f'./RRBFM_offline_ns{n_s}_RRBdim{RRB_dim}.npz')['RRB_mat']

A_boundary = weight_matrix(
        x=BDY,
        p=NODE,
        n=1,
        diffs=[0, 0])
A_boundary = expand_rows(A_boundary, [i for i in range(XY.shape[0], N)], N)

relative_l2_error = 0.0
for _ in range(test_size):
    if (_ + 1) % 100 == 0:
        print(_)
    x = X[_]
    u_hf = U_hf[_]
    mu_1 = x[0]
    mu_2 = x[1]
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
    M = A @ RRB_mat
    MM = M.T @ M
    Mf = M.T.dot(f)
    if _ == 1:
        coef = Mf / MM[0, 0]
    else:
        coef = np.linalg.solve(MM, Mf)
    u = RRB_mat.dot(coef)
    relative_l2_error += Relative_l2_error(y_pred=u, y_true=u_hf)

relative_l2_error = relative_l2_error / test_size
print(f"Relative l2 Error: {relative_l2_error:.8f}")
# n_s = 100, Relative l2 Error: 0.00132978
# n_s = 10000, Relative l2 Error: 0.00138450
