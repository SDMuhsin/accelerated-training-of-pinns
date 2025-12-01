"""
FIXED Python-based dataset generator for DT-PINN experiments.
Now uses polyharmonic splines r^m with polynomial augmentation like MATLAB.
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


def build_rbf_fd_laplacian_correct(X, order=2):
    """
    Build RBF-FD Laplacian using polyharmonic splines + polynomial augmentation.
    Follows MATLAB FormLaplacian.m logic.

    order: desired approximation order (xi in MATLAB)
    """
    N = X.shape[0]
    dim = X.shape[1]

    # Set RBF parameters based on order (from rbffdop.m)
    theta = 2  # Laplacian is 2nd order operator
    ell = order + theta - 1  # polynomial degree
    rbfexp = ell - 1 if ell % 2 == 0 else ell
    rbfexp = max(min(rbfexp, 11), 5)  # clamp to [5, 11]

    # Number of polynomial terms
    npoly = (ell + dim) * (ell + dim - 1) // 2  # binomial(ell+dim, dim)
    stencil_size = 2 * npoly + 1

    print(f"RBF-FD params: order={order}, ell={ell}, rbfexp={rbfexp}, npoly={npoly}, stencil={stencil_size}")

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

        # Polynomial basis
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
            weights = np.linalg.solve(A, rhs)[:n_stencil]  # Only RBF part
        except np.linalg.LinAlgError:
            # Use regularized least squares if singular
            weights = np.linalg.lstsq(A, rhs, rcond=1e-10)[0][:n_stencil]

        # Add to sparse matrix
        for j, idx in enumerate(indices):
            rows.append(i)
            cols.append(idx)
            data.append(weights[j])

    L = csr_matrix((data, (rows, cols)), shape=(N, N))
    return L


def build_rbf_fd_boundary_correct(X_full, X_b, normals, alpha, beta, order=2):
    """
    Build RBF-FD boundary operator for Robin BC.
    Uses polyharmonic splines + polynomial augmentation.
    """
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

        # RHS for Robin BC: alpha * d/dn + beta * I
        # Directional derivative of r^m: m*r^(m-2) * (x-xi) · n
        normal = normals[i]
        delta = X_b[i] - stencil_points  # Points to center
        r_vec = np.linalg.norm(delta, axis=1, keepdims=True)

        # Gradient of r^m: m*r^(m-2) * delta
        grad_rbf = rbfexp * np.power(r[0, :] + 1e-10, rbfexp - 2)[:, None] * delta
        dn_rbf = grad_rbf @ normal

        # Polynomial gradients
        eps = 1e-6
        poly_center = polynomial_basis_2d(np.array([X_b[i, 0]]), np.array([X_b[i, 1]]), ell)
        poly_dx = polynomial_basis_2d(np.array([X_b[i, 0] + eps]), np.array([X_b[i, 1]]), ell)
        poly_dy = polynomial_basis_2d(np.array([X_b[i, 0]]), np.array([X_b[i, 1] + eps]), ell)
        poly_grad_x = (poly_dx - poly_center) / eps
        poly_grad_y = (poly_dy - poly_center) / eps
        dn_poly = poly_grad_x * normal[0] + poly_grad_y * normal[1]

        # Value at boundary
        value_rbf = polyharmonic_rbf(r[0, :], rbfexp)
        value_poly = polynomial_basis_2d(np.array([X_b[i, 0]]), np.array([X_b[i, 1]]), ell)

        # Combine: alpha * dn + beta * value
        rhs = np.concatenate([
            alpha[i, 0] * dn_rbf + beta[i, 0] * value_rbf,
            alpha[i, 0] * dn_poly.flatten() + beta[i, 0] * value_poly.flatten()
        ])

        # Solve
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


def generate_linear_poisson_dataset(order=2, node_file='MatlabSolver/DiskPoissonNodes.mat',
                                    node_idx=0):
    """Generate dataset for linear Poisson equation with CORRECT RBF-FD."""

    data = sio.loadmat(node_file)
    Xi = data['fullintnodes'][0, node_idx]
    Xb = data['bdrynodes'][0, node_idx]
    normals = data['normals'][0, node_idx]

    print(f"Loaded {Xi.shape[0]} interior points, {Xb.shape[0]} boundary points")

    # Ghost points
    N_total = Xi.shape[0] + Xb.shape[0]
    h = 1.0 / np.power(N_total, 1.0 / 2)
    X_g = Xb + 0.25 * h * normals

    X_full = np.vstack([Xi, Xb, X_g])

    # Exact solution
    def u_exact(x, y):
        return 1 + np.sin(np.pi * x) * np.cos(np.pi * y)

    def f_exact(x, y):
        return -2 * np.pi**2 * np.sin(np.pi * x) * np.cos(np.pi * y)

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

    print("Building Laplacian operator with CORRECT polyharmonic RBF-FD...")
    L = build_rbf_fd_laplacian_correct(X_full, order=order)

    print("Building boundary operator with CORRECT polyharmonic RBF-FD...")
    B = build_rbf_fd_boundary_correct(X_full, Xb, normals, alpha, beta, order=order)

    # Test accuracy
    laplacian_exact = f_exact(X_full[:, 0], X_full[:, 1])
    laplacian_approx = L.dot(u).flatten()
    error = np.linalg.norm(laplacian_approx[:Xi.shape[0]] - laplacian_exact[:Xi.shape[0]]) / np.linalg.norm(laplacian_exact[:Xi.shape[0]])
    print(f"Laplacian approximation error: {error:.6f} (should be < 0.01)")

    if error > 0.1:
        print("WARNING: Large approximation error! Matrices may still be inaccurate.")

    # Save
    training_size = Xi.shape[0] + Xb.shape[0]
    folder_name = f"scai/files_{order}_{training_size}"
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

    print(f"Dataset generated successfully with error={error:.6f}!")
    return training_size


if __name__ == "__main__":
    print("="*70)
    print("Generating FIXED Linear Poisson Dataset")
    print("Using polyharmonic splines r^m with polynomial augmentation")
    print("="*70)

    training_size = generate_linear_poisson_dataset(order=2, node_idx=0)

    print("\n" + "="*70)
    print("Generating Test Dataset")
    print("="*70)

    test_size = generate_linear_poisson_dataset(
        order=2,
        node_file='MatlabSolver/DiskPoissonNodesLarge.mat',
        node_idx=0
    )

    import shutil
    src_folder = f"scai/files_2_{test_size}"
    dst_folder = f"scai/files_2_21748_test"
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    shutil.move(src_folder, dst_folder)
    print(f"Test dataset saved to {dst_folder}")
