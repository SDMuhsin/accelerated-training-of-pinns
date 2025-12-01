"""
Python-based dataset generator for DT-PINN experiments.
Replicates functionality of GenSCAIMats.m for linear Poisson equation.
"""

import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree
import os


def rbf_gaussian(r, eps=1.0):
    """Gaussian RBF: exp(-(eps*r)^2)"""
    return np.exp(-(eps * r) ** 2)


def rbf_gaussian_laplacian(r, eps=1.0):
    """Laplacian of Gaussian RBF"""
    return 4 * eps**2 * (eps**2 * r**2 - 1) * np.exp(-(eps * r) ** 2)


def rbf_gaussian_derivative(r, eps=1.0):
    """First derivative of Gaussian RBF"""
    return -2 * eps**2 * r * np.exp(-(eps * r) ** 2)


def build_rbf_fd_laplacian(X, stencil_size=31, order=2):
    """
    Build RBF-FD approximation of Laplacian operator.

    Args:
        X: Points (Nx2 array)
        stencil_size: Number of nearest neighbors for stencil
        order: Order of accuracy

    Returns:
        L: Sparse Laplacian matrix
    """
    N = X.shape[0]
    tree = KDTree(X)

    # Estimate spacing
    h = 1.0 / np.power(N, 1.0 / X.shape[1])
    eps = 3.0 / h  # Shape parameter

    rows, cols, data = [], [], []

    for i in range(N):
        # Find nearest neighbors
        dists, indices = tree.query(X[i], k=min(stencil_size, N))

        # Build local RBF system for Laplacian
        n_stencil = len(indices)
        A = np.zeros((n_stencil, n_stencil))
        b = np.zeros(n_stencil)

        for j in range(n_stencil):
            for k in range(n_stencil):
                r_jk = np.linalg.norm(X[indices[j]] - X[indices[k]])
                A[j, k] = rbf_gaussian(r_jk, eps)

        # Right-hand side: Laplacian at center point
        for j in range(n_stencil):
            r_0j = np.linalg.norm(X[i] - X[indices[j]])
            b[j] = rbf_gaussian_laplacian(r_0j, eps)

        # Solve for weights
        try:
            weights = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback to least squares if singular
            weights = np.linalg.lstsq(A, b, rcond=None)[0]

        # Add to sparse matrix
        for j, idx in enumerate(indices):
            rows.append(i)
            cols.append(idx)
            data.append(weights[j])

    L = csr_matrix((data, (rows, cols)), shape=(N, N))
    return L


def build_rbf_fd_boundary(X_full, X_b, normals, alpha, beta, stencil_size=31):
    """
    Build RBF-FD approximation for boundary conditions.
    Robin BC: alpha * du/dn + beta * u = g

    Args:
        X_full: All points (interior + boundary + ghost)
        X_b: Boundary points
        normals: Boundary normal vectors
        alpha: Neumann coefficient
        beta: Dirichlet coefficient
        stencil_size: Number of nearest neighbors

    Returns:
        B: Sparse boundary condition matrix
    """
    N_full = X_full.shape[0]
    N_b = X_b.shape[0]
    tree = KDTree(X_full)

    h = 1.0 / np.power(N_full, 1.0 / X_full.shape[1])
    eps = 3.0 / h

    rows, cols, data = [], [], []

    for i in range(N_b):
        # Find nearest neighbors
        dists, indices = tree.query(X_b[i], k=min(stencil_size, N_full))

        n_stencil = len(indices)
        A = np.zeros((n_stencil, n_stencil))
        b = np.zeros(n_stencil)

        # Build RBF interpolation matrix
        for j in range(n_stencil):
            for k in range(n_stencil):
                r_jk = np.linalg.norm(X_full[indices[j]] - X_full[indices[k]])
                A[j, k] = rbf_gaussian(r_jk, eps)

        # Right-hand side: alpha * d/dn + beta * I
        normal = normals[i]
        for j in range(n_stencil):
            delta = X_b[i] - X_full[indices[j]]
            r = np.linalg.norm(delta)

            # Directional derivative in normal direction
            if r > 1e-10:
                grad_rbf = rbf_gaussian_derivative(r, eps) * delta / r
                b[j] = alpha[i] * np.dot(grad_rbf, normal) + beta[i] * rbf_gaussian(r, eps)
            else:
                b[j] = beta[i] * rbf_gaussian(r, eps)

        # Solve for weights
        try:
            weights = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(A, b, rcond=None)[0]

        # Add to sparse matrix
        for j, idx in enumerate(indices):
            rows.append(i)
            cols.append(idx)
            data.append(weights[j])

    B = csr_matrix((data, (rows, cols)), shape=(N_b, N_full))
    return B


def generate_linear_poisson_dataset(order=2, node_file='MatlabSolver/DiskPoissonNodes.mat',
                                    node_idx=0, nonlinear=False):
    """
    Generate dataset for linear Poisson equation.

    PDE: Δu = f
    BC: alpha * du/dn + beta * u = g (Robin BC)

    Exact solution: u = 1 + sin(πx)cos(πy)
    """
    # Load node data
    data = sio.loadmat(node_file)

    # Extract nodes (MATLAB uses cell arrays)
    Xi = data['fullintnodes'][0, node_idx]  # Interior nodes
    Xb = data['bdrynodes'][0, node_idx]     # Boundary nodes
    normals = data['normals'][0, node_idx]  # Boundary normals

    print(f"Loaded {Xi.shape[0]} interior points, {Xb.shape[0]} boundary points")

    # Create ghost points (0.25*h outside boundary)
    N_total = Xi.shape[0] + Xb.shape[0]
    h = 1.0 / np.power(N_total, 1.0 / 2)
    X_g = Xb + 0.25 * h * normals

    # All points for differential operators
    X_full = np.vstack([Xi, Xb, X_g])

    # Define exact solution and source term
    def u_exact(x, y):
        return 1 + np.sin(np.pi * x) * np.cos(np.pi * y)

    def f_exact(x, y):
        # For linear: Δu = -2π²sin(πx)cos(πy)
        return -2 * np.pi**2 * np.sin(np.pi * x) * np.cos(np.pi * y)

    def g_exact(nx, ny, x, y):
        # Robin BC: du/dn + u
        # ∇u = (π cos(πx)cos(πy), -π sin(πx)sin(πy))
        du_dx = np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)
        du_dy = -np.pi * np.sin(np.pi * x) * np.sin(np.pi * y)
        du_dn = nx * du_dx + ny * du_dy
        return du_dn + u_exact(x, y)  # Robin: alpha=1, beta=1

    # Evaluate functions at points
    u = u_exact(X_full[:, 0], X_full[:, 1]).reshape(-1, 1)
    f = f_exact(X_full[:, 0], X_full[:, 1]).reshape(-1, 1)
    g = g_exact(normals[:, 0], normals[:, 1], Xb[:, 0], Xb[:, 1]).reshape(-1, 1)

    # Boundary condition coefficients (Robin: alpha=1, beta=1)
    alpha = np.ones((Xb.shape[0], 1))
    beta = np.ones((Xb.shape[0], 1))

    # Build differential operators
    print("Building Laplacian operator...")
    stencil_size = min(31, X_full.shape[0])
    L = build_rbf_fd_laplacian(X_full, stencil_size=stencil_size, order=order)

    print("Building boundary operator...")
    B = build_rbf_fd_boundary(X_full, Xb, normals, alpha, beta, stencil_size=stencil_size)

    # Create output directory
    training_size = Xi.shape[0] + Xb.shape[0]
    folder_name = f"../scai/files_{order}_{training_size}"
    os.makedirs(folder_name, exist_ok=True)

    # Save all data
    print(f"Saving to {folder_name}...")
    sio.savemat(f'{folder_name}/L1.mat', {'L1': L})
    sio.savemat(f'{folder_name}/B1.mat', {'B1': B})
    sio.savemat(f'{folder_name}/Xi.mat', {'Xi': Xi})
    sio.savemat(f'{folder_name}/Xb.mat', {'Xb': Xb})
    sio.savemat(f'{folder_name}/Xg.mat', {'X_g': X_g})
    sio.savemat(f'{folder_name}/f.mat', {'f': f})
    sio.savemat(f'{folder_name}/g.mat', {'g': g})
    sio.savemat(f'{folder_name}/n.mat', {'n': normals})
    sio.savemat(f'{folder_name}/alpha.mat', {'Neucoeff': alpha})
    sio.savemat(f'{folder_name}/beta.mat', {'Dircoeff': beta})
    sio.savemat(f'{folder_name}/u.mat', {'u': u})

    print(f"Dataset generated successfully!")
    return training_size


if __name__ == "__main__":
    # Generate smallest dataset (node_idx=0 corresponds to ~582 points)
    print("="*70)
    print("Generating Linear Poisson Dataset")
    print("="*70)

    # Generate for order 2 (smallest task)
    training_size = generate_linear_poisson_dataset(order=2, node_idx=0)

    # Also generate test dataset (node_idx=4 for ~21k points)
    print("\n" + "="*70)
    print("Generating Test Dataset")
    print("="*70)

    # Check if we have enough nodes
    data = sio.loadmat('MatlabSolver/DiskPoissonNodes.mat')
    n_nodes = len(data['fullintnodes'][0])
    print(f"Available node sets: {n_nodes}")

    if n_nodes >= 5:
        test_size = generate_linear_poisson_dataset(order=2, node_idx=4)
        # Rename test folder
        import shutil
        src_folder = f"../scai/files_2_{test_size}"
        dst_folder = f"../scai/files_2_{test_size}_test"
        if os.path.exists(dst_folder):
            shutil.rmtree(dst_folder)
        shutil.move(src_folder, dst_folder)
        print(f"Test dataset saved to {dst_folder}")
    else:
        print(f"Not enough node sets for test data (need 5, have {n_nodes})")
