"""
Comprehensive dataset generator for all DT-PINN tasks:
- Linear Poisson
- Nonlinear Poisson
- Heat equation

Uses polyharmonic splines r^m with polynomial augmentation (CORRECT implementation).
"""

import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree
import os


def polyharmonic_rbf(r, m):
    """Polyharmonic spline: r^m"""
    return np.power(r + 1e-10, m)


def polyharmonic_laplacian(r, m, dim=2):
    """Laplacian of polyharmonic spline in d dimensions"""
    # Δ(r^m) = m(m+d-2)r^(m-2)
    return m * (m + dim - 2) * np.power(r + 1e-10, m - 2)


def polynomial_basis_2d(x, y, degree):
    """Generate polynomial basis up to given degree"""
    basis = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            basis.append(x**i * y**j)
    return np.column_stack(basis) if basis else np.zeros((len(x), 0))


def polynomial_laplacian_2d(x, y, degree):
    """Laplacian of polynomial basis"""
    basis = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            # Δ(x^i y^j) = i(i-1)x^(i-2)y^j + j(j-1)x^i y^(j-2)
            lap = np.zeros_like(x)
            if i >= 2:
                lap += i * (i - 1) * x**(i - 2) * y**j
            if j >= 2:
                lap += j * (j - 1) * x**i * y**(j - 2)
            basis.append(lap)
    return np.column_stack(basis) if basis else np.zeros((len(x), 0))


def build_rbf_fd_laplacian(X, order=2):
    """Build RBF-FD Laplacian using polyharmonic splines + polynomial augmentation."""
    N = X.shape[0]
    dim = X.shape[1]

    # Set RBF parameters based on order
    theta = 2  # Laplacian is 2nd order operator
    ell = order + theta - 1  # polynomial degree
    rbfexp = ell - 1 if ell % 2 == 0 else ell
    rbfexp = max(min(rbfexp, 11), 5)  # clamp to [5, 11]

    # Number of polynomial terms
    npoly = (ell + dim) * (ell + dim - 1) // 2
    stencil_size = 2 * npoly + 1

    tree = KDTree(X)
    rows, cols, data = [], [], []

    for i in range(N):
        # Find nearest neighbors
        dists, indices = tree.query(X[i], k=min(stencil_size, N))
        stencil_points = X[indices]
        n_stencil = len(indices)

        # Distance matrix for stencil
        r = np.linalg.norm(stencil_points[:, None] - stencil_points[None, :], axis=2)

        # Build augmented RBF matrix
        Ar = polyharmonic_rbf(r, rbfexp)
        v = polynomial_basis_2d(stencil_points[:, 0], stencil_points[:, 1], ell)

        # Augmented system [Ar, v; v^T, 0]
        A = np.block([
            [Ar, v],
            [v.T, np.zeros((npoly, npoly))]
        ])

        # Right-hand side: Laplacian of RBF at center + polynomial Laplacians
        rhs_rbf = polyharmonic_laplacian(r[0, :], rbfexp, dim=dim)
        rhs_poly = polynomial_laplacian_2d(stencil_points[:, 0], stencil_points[:, 1], ell)
        rhs = np.concatenate([rhs_rbf, rhs_poly[0]])

        # Solve for weights
        try:
            weights = np.linalg.solve(A, rhs)[:n_stencil]
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(A, rhs, rcond=1e-10)[0][:n_stencil]

        # Add to sparse matrix
        for j, idx in enumerate(indices):
            rows.append(i)
            cols.append(idx)
            data.append(weights[j])

    L = csr_matrix((data, (rows, cols)), shape=(N, N))
    return L


def build_rbf_fd_boundary(X_full, X_b, normals, alpha, beta, order=2):
    """Build RBF-FD boundary operator for Robin BC."""
    N_full = X_full.shape[0]
    N_b = X_b.shape[0]
    dim = X_full.shape[1]

    # Set RBF parameters (for gradient operator, theta=1)
    theta = 1
    ell = order + theta - 1
    rbfexp = ell - 1 if ell % 2 == 0 else ell
    rbfexp = max(min(rbfexp, 11), 5)

    npoly = (ell + dim) * (ell + dim - 1) // 2
    stencil_size = 2 * npoly + 1

    tree = KDTree(X_full)
    rows, cols, data = [], [], []

    for i in range(N_b):
        # Find nearest neighbors
        dists, indices = tree.query(X_b[i], k=min(stencil_size, N_full))
        stencil_points = X_full[indices]
        n_stencil = len(indices)

        # Distance matrix
        r = np.linalg.norm(stencil_points[:, None] - stencil_points[None, :], axis=2)

        # Augmented RBF matrix
        Ar = polyharmonic_rbf(r, rbfexp)
        v = polynomial_basis_2d(stencil_points[:, 0], stencil_points[:, 1], ell)

        A = np.block([
            [Ar, v],
            [v.T, np.zeros((npoly, npoly))]
        ])

        # RHS: gradient of RBF in normal direction
        dr_dx = (stencil_points[:, 0] - X_b[i, 0]) / (r[0, :] + 1e-10)
        dr_dy = (stencil_points[:, 1] - X_b[i, 1]) / (r[0, :] + 1e-10)

        # ∂(r^m)/∂n = m*r^(m-1) * (∂r/∂n)
        dr_dn = dr_dx * normals[i, 0] + dr_dy * normals[i, 1]
        drbf_dn = rbfexp * np.power(r[0, :] + 1e-10, rbfexp - 1) * dr_dn

        # Polynomial gradients in normal direction
        dpoly_dx = []
        dpoly_dy = []
        for k in range(ell + 1):
            for l in range(ell + 1 - k):
                dx = k * stencil_points[:, 0]**(k-1 if k > 0 else 0) * stencil_points[:, 1]**l if k > 0 else np.zeros(n_stencil)
                dy = l * stencil_points[:, 0]**k * stencil_points[:, 1]**(l-1 if l > 0 else 0) if l > 0 else np.zeros(n_stencil)
                dpoly_dx.append(dx)
                dpoly_dy.append(dy)

        dpoly_dn = []
        for j in range(len(dpoly_dx)):
            dpoly_dn.append(dpoly_dx[j] * normals[i, 0] + dpoly_dy[j] * normals[i, 1])
        dpoly_dn = np.column_stack(dpoly_dn) if dpoly_dn else np.zeros((n_stencil, 0))

        # Robin BC: alpha * du/dn + beta * u
        rhs = np.concatenate([
            alpha[i, 0] * drbf_dn + beta[i, 0] * polyharmonic_rbf(r[0, :], rbfexp),
            alpha[i, 0] * dpoly_dn[0] + beta[i, 0] * v[0]
        ])

        try:
            weights = np.linalg.solve(A, rhs)[:n_stencil]
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(A, rhs, rcond=1e-10)[0][:n_stencil]

        for j, idx in enumerate(indices):
            rows.append(i)
            cols.append(idx)
            data.append(weights[j])

    B = csr_matrix((data, (rows, cols)), shape=(N_b, N_full))
    return B


def generate_poisson_dataset(task_type='linear', order=2, node_file='MatlabSolver/DiskPoissonNodes.mat',
                             node_idx=0):
    """
    Generate dataset for Poisson equation (linear or nonlinear).

    task_type: 'linear' or 'nonlinear'
    node_idx: row index in the matlab file
    """

    data = sio.loadmat(node_file)
    Xi = data['fullintnodes'][node_idx, 0]
    Xb = data['bdrynodes'][node_idx, 0]
    normals = data['normals'][node_idx, 0]

    print(f"Loaded {Xi.shape[0]} interior points, {Xb.shape[0]} boundary points")

    # Ghost points
    N_total = Xi.shape[0] + Xb.shape[0]
    h = 1.0 / np.power(N_total, 1.0 / 2)
    X_g = Xb + 0.25 * h * normals

    X_full = np.vstack([Xi, Xb, X_g])

    # Exact solution (same for both linear and nonlinear)
    def u_exact(x, y):
        return 1 + np.sin(np.pi * x) * np.cos(np.pi * y)

    if task_type == 'linear':
        # Linear: ∇²u = f
        def f_exact(x, y):
            return -2 * np.pi**2 * np.sin(np.pi * x) * np.cos(np.pi * y)
    else:
        # Nonlinear: ∇²u = f + exp(u)
        def f_exact(x, y):
            # f = ∇²u - exp(u) where u is the exact solution
            laplacian_u = -2 * np.pi**2 * np.sin(np.pi * x) * np.cos(np.pi * y)
            u = u_exact(x, y)
            return laplacian_u - np.exp(u)  # So that ∇²u = f + exp(u)

    def g_exact(nx, ny, x, y):
        du_dx = np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)
        du_dy = -np.pi * np.sin(np.pi * x) * np.sin(np.pi * y)
        du_dn = nx * du_dx + ny * du_dy
        return du_dn + u_exact(x, y)  # Robin: alpha=1, beta=1

    u = u_exact(X_full[:, 0], X_full[:, 1]).reshape(-1, 1)
    f = f_exact(X_full[:, 0], X_full[:, 1]).reshape(-1, 1)
    g = g_exact(normals[:, 0], normals[:, 1], Xb[:, 0], Xb[:, 1]).reshape(-1, 1)

    alpha = np.ones((Xb.shape[0], 1))
    beta = np.ones((Xb.shape[0], 1))

    print("Building Laplacian operator with polyharmonic RBF-FD...")
    L = build_rbf_fd_laplacian(X_full, order=order)

    print("Building boundary operator with polyharmonic RBF-FD...")
    B = build_rbf_fd_boundary(X_full, Xb, normals, alpha, beta, order=order)

    # Test accuracy
    if task_type == 'linear':
        laplacian_exact = f_exact(X_full[:, 0], X_full[:, 1])
        laplacian_approx = L.dot(u).flatten()
        error = np.linalg.norm(laplacian_approx[:Xi.shape[0]] - laplacian_exact[:Xi.shape[0]]) / np.linalg.norm(laplacian_exact[:Xi.shape[0]])
        print(f"Laplacian approximation error: {error:.6f} (should be < 0.01)")

        if error > 0.1:
            print("WARNING: Large approximation error!")

    # Save
    training_size = Xi.shape[0] + Xb.shape[0]
    folder_name = f"{task_type}/files_{order}_{training_size}"
    os.makedirs(folder_name, exist_ok=True)

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

    print(f"{task_type.capitalize()} Poisson dataset generated successfully!")
    return training_size


def generate_heat_dataset(order=2, node_file='MatlabSolver/DiskPoissonNodes.mat',
                         node_idx=1):  # Note: heat uses idx=1 (828 points)
    """Generate dataset for heat equation (time-dependent)."""

    data = sio.loadmat(node_file)
    Xi = data['fullintnodes'][node_idx, 0]
    Xb = data['bdrynodes'][node_idx, 0]
    normals = data['normals'][node_idx, 0]

    print(f"Loaded {Xi.shape[0]} interior points, {Xb.shape[0]} boundary points (heat)")

    # Ghost points
    N_total = Xi.shape[0] + Xb.shape[0]
    h = 1.0 / np.power(N_total, 1.0 / 2)
    X_g = Xb + 0.25 * h * normals

    X_full = np.vstack([Xi, Xb, X_g])

    # Heat equation: u_t = ∇²u
    # Exact solution (time-dependent): u(x,y,t) = exp(-2π²t) * sin(πx) * cos(πy)
    def u_exact(x, y, t):
        return np.exp(-2 * np.pi**2 * t) * np.sin(np.pi * x) * np.cos(np.pi * y)

    def f_exact(x, y, t):
        # u_t - ∇²u = 0 for heat equation, so f = 0
        return np.zeros_like(x)

    def g_exact(nx, ny, x, y, t):
        # Robin BC at all times
        du_dx = np.exp(-2 * np.pi**2 * t) * np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)
        du_dy = np.exp(-2 * np.pi**2 * t) * (-np.pi) * np.sin(np.pi * x) * np.sin(np.pi * y)
        du_dn = nx * du_dx + ny * du_dy
        return du_dn + u_exact(x, y, t)  # Robin: alpha=1, beta=1

    # Time discretization (25 time steps from 0 to 1)
    t_vals = np.linspace(0, 1, 25)

    # For heat equation, store initial condition
    u_heat = u_exact(X_full[:, 0], X_full[:, 1], 0.0).reshape(-1, 1)  # t=0
    f_heat = np.zeros((X_full.shape[0], 1))  # Heat equation has no source term
    g_heat = g_exact(normals[:, 0], normals[:, 1], Xb[:, 0], Xb[:, 1], 0.0).reshape(-1, 1)

    alpha = np.ones((Xb.shape[0], 1))
    beta = np.ones((Xb.shape[0], 1))

    print("Building Laplacian operator with polyharmonic RBF-FD (for spatial derivatives)...")
    L = build_rbf_fd_laplacian(X_full, order=order)

    print("Building boundary operator with polyharmonic RBF-FD...")
    B = build_rbf_fd_boundary(X_full, Xb, normals, alpha, beta, order=order)

    # Save
    training_size = Xi.shape[0] + Xb.shape[0]
    folder_name = f"heat/files_{order}_{training_size}"
    os.makedirs(folder_name, exist_ok=True)

    print(f"Saving to {folder_name}...")
    sio.savemat(f'{folder_name}/L1.mat', {'L1': L})
    sio.savemat(f'{folder_name}/B1.mat', {'B1': B})
    sio.savemat(f'{folder_name}/Xi.mat', {'Xi': Xi})
    sio.savemat(f'{folder_name}/Xb.mat', {'Xb': Xb})
    sio.savemat(f'{folder_name}/Xg.mat', {'X_g': X_g})
    sio.savemat(f'{folder_name}/f_heat.mat', {'f_heat': f_heat})
    sio.savemat(f'{folder_name}/g_heat.mat', {'g_heat': g_heat})
    sio.savemat(f'{folder_name}/u_heat.mat', {'u_heat': u_heat})
    sio.savemat(f'{folder_name}/n.mat', {'n': normals})
    sio.savemat(f'{folder_name}/alpha.mat', {'Neucoeff': alpha})
    sio.savemat(f'{folder_name}/beta.mat', {'Dircoeff': beta})

    print(f"Heat equation dataset generated successfully!")
    return training_size


if __name__ == "__main__":
    print("="*70)
    print("GENERATING ALL DT-PINN DATASETS")
    print("="*70)

    # Linear Poisson (already have this in scai/, skip)

    # Nonlinear Poisson - 582 points (small)
    print("\n" + "="*70)
    print("1. NONLINEAR POISSON (582 points)")
    print("="*70)
    generate_poisson_dataset(task_type='nonlinear', order=2,
                            node_file='MatlabSolver/DiskPoissonNodes.mat',
                            node_idx=0)

    # Nonlinear Poisson - 2236 points (standard)
    print("\n" + "="*70)
    print("2. NONLINEAR POISSON (2236 points)")
    print("="*70)
    generate_poisson_dataset(task_type='nonlinear', order=2,
                            node_file='MatlabSolver/DiskPoissonNodesLarge.mat',
                            node_idx=0)

    # Heat equation - 828 points
    print("\n" + "="*70)
    print("3. HEAT EQUATION (828 points)")
    print("="*70)
    generate_heat_dataset(order=2,
                         node_file='MatlabSolver/DiskPoissonNodes.mat',
                         node_idx=1)

    print("\n" + "="*70)
    print("ALL DATASETS GENERATED SUCCESSFULLY!")
    print("="*70)
