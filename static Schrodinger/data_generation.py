import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
from scipy.special import sph_harm
from scipy.sparse.linalg import spsolve
from rbf.pde.fd import weight_matrix
from rbf.sputils import expand_rows

def cartesian_to_spherical(cartesian_coords):
    """cartesian coordinate (x,y,z) to spherical coordinate (θ,ϕ)∈[0,2π]*[0,π]"""
    r = np.linalg.norm(cartesian_coords, ord=2, axis=1)
    x = cartesian_coords[:, 0]
    y = cartesian_coords[:, 1]
    z = cartesian_coords[:, 2]
    theta = np.arctan2(y, x)
    theta = np.where(theta<0, 2*np.pi+theta, theta)
    phi = np.arccos(z / r)
    return theta, phi

def gaussian_random_field(n, xyz, tau, alpha, k):
    """
    sample n functions from gaussian random field (-\Delta_0 + tau^2*I)^(-alpha)
    :param n: return n functions
    :param xyz: N_bdy * 3 matrix, N_bdy evaluation coordinates (x,y,z) of function
    :param tau: control smoothness
    :param alpha: alpha should > d/2 (here d=2)
        tau and alpha control smoothness; the bigger they are, the smoother the
    :param k: use truncated KL expansion with k terms
    :return: N_bdy * n matrix, N_bdy evaluate values of n sample functions
    """
    theta, phi = cartesian_to_spherical(xyz)
    l_list = []
    m_list = []
    for l in range(k):
        for m in range(-l, l + 1):
            l_list.append(l)
            m_list.append(m)
    num_coeffs = len(l_list)
    l_list = np.array(l_list)
    m_list = np.array(m_list)
    coeffs = (l_list * (l_list + 1) + tau ** 2) ** (-alpha / 2)

    # generate n*num_coeffs normal gauss random values
    ksi = np.random.randn(n, num_coeffs)

    coeffs = ksi * coeffs

    # compute the value of sphere hamornics, return num_coeffs*N matrix
    Y = sph_harm(m_list[:, np.newaxis], l_list[:, np.newaxis], theta[np.newaxis, :], phi[np.newaxis, :]).real

    F = coeffs.dot(Y).T

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(x, y, z, c=F[:, 0], cmap='viridis', s=100)
    cbar1 = fig.colorbar(scatter1, ax=ax1, shrink=0.6, pad=0.1)
    cbar1.set_label("Values")

    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(x, y, z, c=F[:, 1], cmap='viridis', s=100)
    cbar2 = fig.colorbar(scatter2, ax=ax2, shrink=0.6, pad=0.1)
    cbar2.set_label("Values")

    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(x, y, z, c=F[:, 2], cmap='viridis', s=100)
    cbar3 = fig.colorbar(scatter3, ax=ax3, shrink=0.6, pad=0.1)
    cbar3.set_label("Values")

    plt.tight_layout()
    plt.savefig('gaussian_random_field.png')
    plt.close()
    return F

def snapshot_mat(A, F):
    S = np.zeros((N, 0))
    for f in F.T:
        u_hf = spsolve(A, f)
        S = np.concatenate((S, u_hf[:, np.newaxis]), axis=1)
    return S


def POD(S, eps_POD):
    U, S, V = np.linalg.svd(S)
    total_energy = (S ** 2).sum()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    energy_sum = [S[0]**2]
    for s in S[1:]:
        energy_sum.append(energy_sum[-1] + s**2)
    ax[0].semilogy(range(len(S)), S)
    ax[1].semilogy(range(len(S)), S / S[0])
    ax[2].semilogy(range(len(S)), 1 - energy_sum / total_energy)
    plt.tight_layout()
    plt.show()

    energy = 0
    d = 0
    while energy / total_energy < 1 - eps_POD ** 2 and d < len(S):
        energy += S[d] ** 2
        d += 1
    return U[:, 0:d]

if __name__ == "__main__":
    np.random.seed(1111)
    discretization = np.load('./discretization.npz')
    XYZ = discretization['xyz']
    BDY = discretization['bdy']
    NODE = np.concatenate((XYZ, BDY))
    N = NODE.shape[0]

    n_data = 500
    train_size = 300
    val_size = 100
    test_size = 100
    eps_POD = 0.03

    tau = 1
    alpha = 2
    k = 50

    try:
        F = np.load(f'./gaussian_random_field_ndata{n_data}_tau{tau}_alpha{alpha}_k{k}.npy')
    except IOError:
        F = gaussian_random_field(n=n_data, xyz=BDY, tau=tau, alpha=alpha, k=k)  # N_bdy * n_data
        np.save(f'./gaussian_random_field_ndata{n_data}_tau{tau}_alpha{alpha}_k{k}.npy', F)

    X = F.T
    F = np.concatenate((np.zeros((XYZ.shape[0], n_data)), F)) # N * n_data

    phi = 'imq'
    eps = 1
    order = -1
    n_stencil = 50

    A_interior = weight_matrix(
        x=XYZ,
        p=NODE,
        n=n_stencil,
        diffs=[[2, 0, 0], [0, 2, 0], [0, 0, 2], [0, 0, 0]],
        coeffs=[-1, -1, -1, 1],
        phi=phi,
        eps=eps,
        order=order)
    A_interior = expand_rows(A_interior, [i for i in range(XYZ.shape[0])], N)
    A_boundary = weight_matrix(
        x=BDY,
        p=NODE,
        n=1,
        diffs=[0, 0, 0])
    A_boundary = expand_rows(A_boundary, [i for i in range(XYZ.shape[0], N)], N)
    A = A_boundary + A_interior

    try:
        S = np.load(f'./snapshot_mat_ndata{n_data}.npy')
    except IOError:
        S = snapshot_mat(A, F) # N * n_data
        np.save(f'./snapshot_mat_ndata{n_data}.npy', S)
    try:
        V = np.load(f'./POD_mat_ndata{n_data}_epsPOD{eps_POD}.npy')
    except IOError:
        V = POD(S[:, 0:train_size], eps_POD=eps_POD)
        np.save(f'./POD_mat_ndata{n_data}_epsPOD{eps_POD}.npy', V)

    Y = S.T @ V
    np.savez(f'./data_set_ndata{n_data}_epsPOD{eps_POD}.npz', X=X, Y=Y, U_hf=S.T)