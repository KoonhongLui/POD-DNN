import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def generate_spiral_points(N):
    """generate spiral points as parametric node set"""
    indices = np.arange(0, N, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / N)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    # spherical coordinate (λ, θ) ∈ [-π,π) × [-π/2,π/2)
    lambda_coord = theta % (2 * np.pi) - np.pi
    theta_coord = np.pi / 2 - phi

    return np.column_stack((lambda_coord, theta_coord))


def spherical_to_cartesian(spherical_coords):
    """spherical coordinate (λ,θ) to cartesian coordinate (x,y,z)"""
    lambda_coord = spherical_coords[:, 0]
    theta_coord = spherical_coords[:, 1]

    x = np.cos(theta_coord) * np.cos(lambda_coord)
    y = np.cos(theta_coord) * np.sin(lambda_coord)
    z = np.sin(theta_coord)

    return np.column_stack((x, y, z))


def polyharmonic_spline(r, m):
    """polyharmonic spline (PHS) kernel"""
    if m % 2 == 0:  # for sphere S^2
        log_r = np.zeros_like(r)
        mask = r > 0
        log_r[mask] = np.log(r[mask])
        return r ** m * log_r
    else:  # for circle S^1
        return r ** m


def build_sbf_interpolant(seed_points, m=6):
    """build SBF interpolant"""
    # turn seed points to spherical coordinates
    x, y, z = seed_points.T
    theta = np.arcsin(z)  # θ ∈ [-π/2, π/2]
    lambda_coord = np.arctan2(y, x)  # λ ∈ [-π, π)

    # parametric node set
    param_nodes = np.column_stack((lambda_coord, theta))

    # interpolation matrix
    cart_prod = seed_points @ seed_points.T
    cart_prod = np.where(cart_prod >= 1, 0, 2 * (1 - cart_prod))
    r = np.sqrt(cart_prod)
    Phi = polyharmonic_spline(r, m)

    # solve the linear system
    coeffs = np.linalg.solve(Phi, seed_points)

    return param_nodes, coeffs


def evaluate_sbf_interpolant(param_nodes, coeffs, eval_points, m=6):
    """evaluate SBF interpolant to get Cartesian coordinate"""
    eval_cart = spherical_to_cartesian(eval_points)
    node_cart = spherical_to_cartesian(param_nodes)

    # distance matrix
    r = np.sqrt(2 * (1 - eval_cart @ node_cart.T))
    Phi = polyharmonic_spline(r, m)

    # evaluate SBF interpolant
    return Phi @ coeffs


def sample_elimination(points, h):
    """sample elimination to make sure that minimum nearest neighbor distance >= h"""
    tree = KDTree(points)
    n = len(points)
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if keep[i]:
            # find all points with distance <= h (including itself)
            idx = tree.query_ball_point(points[i], h)
            # keep current point and eliminate all points with distance <= h
            keep[idx] = False
            keep[i] = True

    return points[keep]

def offset_boundary(boundary_points, h):
    """project boundary points inwards a distance h"""
    return boundary_points - h * boundary_points / np.linalg.norm(boundary_points, ord=2, axis=1)[:, np.newaxis]

def compute_obb(points):
    """use PCA to compute OBB"""
    pca = PCA(n_components=3)
    pca.fit(points)

    # get PCA components and center
    components = pca.components_
    center = pca.mean_

    # transform points to PCA space
    transformed = (points - center) @ components.T

    # compute bounding vertices
    mins = transformed.min(axis=0)
    maxs = transformed.max(axis=0)

    # compute all 8 bounding vertices
    vertices = np.array([
        [mins[0], mins[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [mins[0], maxs[1], mins[2]],
        [mins[0], maxs[1], maxs[2]],
        [maxs[0], mins[1], mins[2]],
        [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], mins[2]],
        [maxs[0], maxs[1], maxs[2]]
    ])

    # transform bounding vertices to original space
    obb_vertices = vertices @ components + center

    return obb_vertices, components, center, (mins, maxs)


def poisson_disk_sampling(obb_vertices, h, k=30):
    """poisson disk sampling in OBB"""
    # get OBB boundary
    mins = obb_vertices.min(axis=0)
    maxs = obb_vertices.max(axis=0)

    # initialize a cell to facilitate searching
    cell_size = h / np.sqrt(3)
    grid_shape = ((maxs - mins) / cell_size).astype(int) + 1
    grid = np.full(grid_shape, -1, dtype=int)

    # choose the center point as the initial point
    initial_point = (mins + maxs) / 2
    samples = [initial_point]
    active = [0]
    grid[tuple(((initial_point - mins) // cell_size).astype(int))] = 0

    while active:
        # choose a random active point
        idx = np.random.choice(active)
        point = samples[idx]
        found = False

        # generate k candidate points
        for _ in range(k):
            # generate k uniform random nodes in the spherical annulus
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.arccos(np.random.uniform(-1, 1))
            r = np.random.uniform(h, 2 * h)

            offset = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])

            candidate = point + offset

            # check if all candidates in the OBB, if not, reject them
            if not (np.all(candidate >= mins) and np.all(candidate <= maxs)):
                continue

            # check possible neighbors with the cell
            grid_coord = ((candidate - mins) // cell_size).astype(int)
            min_coord = np.maximum(grid_coord - 2, 0)
            max_coord = np.minimum(grid_coord + 3, grid_shape)

            valid = True
            for i in range(min_coord[0], max_coord[0]):
                for j in range(min_coord[1], max_coord[1]):
                    for k in range(min_coord[2], max_coord[2]):
                        neighbor_idx = grid[i, j, k]
                        if neighbor_idx != -1:
                            if np.linalg.norm(candidate - samples[neighbor_idx]) < h:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break

            if valid:
                samples.append(candidate)
                new_idx = len(samples) - 1
                active.append(new_idx)
                grid[tuple(grid_coord)] = new_idx
                found = True

        if not found:
            active.remove(idx)

    return np.array(samples)


def generate_uniform_ball_points(Nd, h, tau=2, m=6, k=30):
    """
    generate points on the sphere and in ball
    Nd: number of seed points on the sphere (boundary)
    h: minimum nearest neighbor distance
    tau: the supersampling parameter
    m: degree of PHS
    k: the number of samples to choose before rejection in Poisson disk sampling (the Poisson neighborhood size)
    """
    # 1. generate spherical points
    # 1.1 generate spiral points as parametric node set
    param_nodes = generate_spiral_points(Nd)

    # 1.2 turn spherical coordinate (λ,θ) to cartesian coordinate (x,y,z) as seed points
    seed_points = spherical_to_cartesian(param_nodes)

    # 1.3 build SBF interpolant model
    param_nodes, coeffs = build_sbf_interpolant(seed_points, m)

    # 1.4 estimate the number of boundary points (N_b = s * h^{-2}, for unit sphere, s=4π)
    s = 4 * np.pi
    Nb = int(s / h ** 2)

    # 1.5 supersampling: generate τN_b candidates
    N_supersample = tau * Nb
    lambda_sup = np.random.uniform(-np.pi, np.pi, N_supersample)
    theta_sup = np.arcsin(np.random.uniform(-1, 1, N_supersample))

    eval_points = np.column_stack((lambda_sup, theta_sup))

    # 1.6 evaluate sbf interpolant to get cartesian coordinates
    candidate_points = evaluate_sbf_interpolant(param_nodes, coeffs, eval_points, m)

    # 1.7 sample elimination to make sure that minimum nearest neighbor distance >= h
    surface_points = sample_elimination(candidate_points, h)

    # 2. generate interior points

    # 2.1 generate inner boundary points

    inner_surface_points = offset_boundary(surface_points, h)
    # 2.2 compute OBB of inner boundary
    obb_vertices, _, _, _ = compute_obb(inner_surface_points)

    # 2.3 Poisson disk sampling in OBB
    interior_candidates = poisson_disk_sampling(obb_vertices, h, k)

    # 2.4 retain points within the inner boundary
    norms = np.linalg.norm(interior_candidates, axis=1)
    interior_points = interior_candidates[norms <= 1.0 - h]

    # 3. merge surface points and interior points
    all_points = np.vstack((surface_points, interior_points))

    return all_points, surface_points, interior_points

def nearest_distance_surface_to_interior(surface_points, interior_points):
    '''compute nearest interior neighbor distance for surface points'''
    tree = cKDTree(interior_points)
    dists, _ = tree.query(surface_points, k=1)
    return dists



if __name__ == "__main__":
    np.random.seed(555)
    # parameter setting
    Nd = 100  # number of seed points on the boundary
    h = 0.1  # expected minimum nearest neighbor distance
    tau = 2  # the supersampling parameter
    k = 30  # the number of samples to choose before rejection in Poisson disk sampling (the Poisson neighborhood size)

    # generate uniform points on and in the ball
    all_points, surface_points, interior_points = generate_uniform_ball_points(Nd, h, tau, k=30)

    print(f"Generated {len(all_points)} points in total")
    print(f"Surface points: {len(surface_points)}")
    print(f"Interior points: {len(interior_points)}")

    # check if all points are in the ball
    norms = np.linalg.norm(all_points, axis=1)
    print(f"Max norm: {np.max(norms):.4f}")
    print(f"Min norm: {np.min(norms):.4f}")

    # compute distance statistics
    dists = cdist(all_points, all_points)
    np.fill_diagonal(dists, np.inf)  # ignore distance to itself
    min_dists = np.min(dists, axis=1)

    # compute nearest interior neighbor distance for surface points
    distance_surface_to_interior = nearest_distance_surface_to_interior(surface_points, interior_points)

    print("\nDistance statistics:")
    print(f"Average nearest neighbor distance: {np.mean(min_dists):.4f}")
    print(f"Minimum nearest neighbor distance: {np.min(min_dists):.4f}")
    print(f"Maximum nearest neighbor distance: {np.max(min_dists):.4f}")
    print(f'Minimum nearest interior neighbor distance for surface points: {np.min(distance_surface_to_interior):.4f}')
    print(f'Maximum nearest interior neighbor distance for surface points: {np.max(distance_surface_to_interior):.4f}')

    # np.savez(f'./discretization.npz', xyz=interior_points, bdy=surface_points)

    # visualization
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])

    # plot interior points (blue)
    ax.scatter(interior_points[:, 0], interior_points[:, 1], interior_points[:, 2],
               c='b', s=10, alpha=0.6, label='Interior points')

    # plot surface points (red)
    ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2],
               c='r', s=20, alpha=0.8, label='Surface points')

    ax.set_title(f"Uniform Points in Unit Ball (h={h})")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.show()