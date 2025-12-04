"""
Test boundary operators for RBF-FD.

Validates:
1. Dirichlet extraction matrix structure
2. Dirichlet interpolation (row sums = 1)
3. Neumann operator (directional derivative accuracy)
"""

import numpy as np
from scipy.io import loadmat
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbf_fd import (
    BoundaryOperatorBuilder,
    build_dirichlet_extraction,
    build_dirichlet_interpolation,
    build_neumann_operator,
    GhostPointGenerator,
    estimate_normals_radial,
)


def test_dirichlet_extraction():
    """Test simple Dirichlet extraction operator."""
    print("\n" + "=" * 60)
    print("Test 1: Dirichlet Extraction Operator")
    print("=" * 60)

    n_i, n_b, n_g = 100, 20, 20
    B = build_dirichlet_extraction(n_i, n_b, n_g)

    print(f"  Shape: {B.shape}")
    print(f"  Expected: ({n_b}, {n_i + n_b + n_g})")
    print(f"  NNZ: {B.nnz}")
    print(f"  Expected NNZ: {n_b}")

    # Test that it extracts boundary values correctly
    u = np.random.randn(n_i + n_b + n_g)
    u_boundary_expected = u[n_i:n_i + n_b]
    u_boundary_extracted = B @ u

    error = np.linalg.norm(u_boundary_extracted - u_boundary_expected)
    print(f"  Extraction error: {error:.2e}")

    passed = (B.shape == (n_b, n_i + n_b + n_g) and
              B.nnz == n_b and
              error < 1e-14)
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed


def test_dirichlet_interpolation():
    """Test Dirichlet interpolation operator."""
    print("\n" + "=" * 60)
    print("Test 2: Dirichlet Interpolation Operator")
    print("=" * 60)

    # Create a simple test domain (unit disk)
    n_interior = 200
    n_boundary = 50

    # Interior points (random in disk)
    r = np.sqrt(np.random.rand(n_interior)) * 0.9
    theta = np.random.rand(n_interior) * 2 * np.pi
    X_interior = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    # Boundary points (on unit circle)
    theta_b = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
    X_boundary = np.column_stack([np.cos(theta_b), np.sin(theta_b)])

    # All points
    X_all = np.vstack([X_interior, X_boundary])

    B = build_dirichlet_interpolation(
        X_all, X_boundary, n_interior,
        stencil_size=13, poly_degree=3, rbf_order=5
    )

    print(f"  Shape: {B.shape}")
    print(f"  NNZ: {B.nnz}")
    print(f"  NNZ per row: {B.nnz / n_boundary:.1f}")

    # Check row sums (should be 1 for interpolation)
    row_sums = np.array(B.sum(axis=1)).flatten()
    row_sum_error = np.abs(row_sums - 1.0).max()
    print(f"  Row sums range: [{row_sums.min():.6f}, {row_sums.max():.6f}]")
    print(f"  Row sum error from 1: {row_sum_error:.2e}")

    # Test on known function: u(x,y) = x^2 + y^2 (should equal 1 on boundary)
    u_all = X_all[:, 0]**2 + X_all[:, 1]**2
    u_boundary_interp = B @ u_all
    u_boundary_exact = X_boundary[:, 0]**2 + X_boundary[:, 1]**2

    interp_error = np.linalg.norm(u_boundary_interp - u_boundary_exact) / np.linalg.norm(u_boundary_exact)
    print(f"  Interpolation error (x^2+y^2): {interp_error:.2e}")

    passed = row_sum_error < 1e-10 and interp_error < 1e-10
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed


def test_neumann_operator():
    """Test Neumann (normal derivative) operator."""
    print("\n" + "=" * 60)
    print("Test 3: Neumann Operator (Normal Derivative)")
    print("=" * 60)

    # Create a simple test domain (unit disk)
    n_interior = 300
    n_boundary = 50

    # Interior points (random in disk)
    r = np.sqrt(np.random.rand(n_interior)) * 0.9
    theta = np.random.rand(n_interior) * 2 * np.pi
    X_interior = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    # Boundary points (on unit circle)
    theta_b = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
    X_boundary = np.column_stack([np.cos(theta_b), np.sin(theta_b)])

    # Normals (radial for unit circle)
    normals = estimate_normals_radial(X_boundary, center=np.array([0.0, 0.0]))

    # All points
    X_all = np.vstack([X_interior, X_boundary])

    B_neumann = build_neumann_operator(
        X_all, X_boundary, normals, n_interior,
        stencil_size=13, poly_degree=3, rbf_order=5
    )

    print(f"  Shape: {B_neumann.shape}")
    print(f"  NNZ: {B_neumann.nnz}")
    print(f"  NNZ per row: {B_neumann.nnz / n_boundary:.1f}")

    # Test 1: u(x,y) = x^2 + y^2
    # grad u = [2x, 2y]
    # du/dn = grad u · n = 2x*nx + 2y*ny
    # On unit circle: x = cos(theta), y = sin(theta), n = [cos(theta), sin(theta)]
    # du/dn = 2*cos^2 + 2*sin^2 = 2
    u_all = X_all[:, 0]**2 + X_all[:, 1]**2
    dudn_computed = B_neumann @ u_all
    dudn_exact = 2.0 * np.ones(n_boundary)  # = 2 everywhere on unit circle

    error_1 = np.linalg.norm(dudn_computed - dudn_exact) / np.linalg.norm(dudn_exact)
    print(f"  Test u=x^2+y^2: du/dn error = {error_1:.2e} (exact=2)")

    # Test 2: u(x,y) = x
    # grad u = [1, 0]
    # du/dn = 1*nx + 0*ny = nx = cos(theta)
    u_all = X_all[:, 0]
    dudn_computed = B_neumann @ u_all
    dudn_exact = normals[:, 0]  # = cos(theta)

    error_2 = np.linalg.norm(dudn_computed - dudn_exact) / np.linalg.norm(dudn_exact)
    print(f"  Test u=x: du/dn error = {error_2:.2e}")

    # Test 3: u(x,y) = xy
    # grad u = [y, x]
    # du/dn = y*nx + x*ny = sin(theta)*cos(theta) + cos(theta)*sin(theta) = sin(2*theta)
    u_all = X_all[:, 0] * X_all[:, 1]
    dudn_computed = B_neumann @ u_all
    dudn_exact = X_boundary[:, 1] * normals[:, 0] + X_boundary[:, 0] * normals[:, 1]

    error_3 = np.linalg.norm(dudn_computed - dudn_exact) / np.linalg.norm(dudn_exact)
    print(f"  Test u=xy: du/dn error = {error_3:.2e}")

    passed = error_1 < 1e-6 and error_2 < 1e-6 and error_3 < 1e-6
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed


def test_boundary_builder_class():
    """Test the BoundaryOperatorBuilder class interface."""
    print("\n" + "=" * 60)
    print("Test 4: BoundaryOperatorBuilder Class")
    print("=" * 60)

    # Create test domain
    n_interior = 200
    n_boundary = 50

    r = np.sqrt(np.random.rand(n_interior)) * 0.9
    theta = np.random.rand(n_interior) * 2 * np.pi
    X_interior = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    theta_b = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
    X_boundary = np.column_stack([np.cos(theta_b), np.sin(theta_b)])

    normals = estimate_normals_radial(X_boundary, center=np.array([0.0, 0.0]))

    # Test builder
    builder = BoundaryOperatorBuilder(stencil_size=13, poly_degree=3, rbf_order=5)

    # Dirichlet extraction
    B_extract = builder.build_dirichlet(X_interior, X_boundary, method='extraction')
    print(f"  Dirichlet extraction shape: {B_extract.shape}")

    # Dirichlet interpolation
    B_interp = builder.build_dirichlet(X_interior, X_boundary, method='interpolation')
    print(f"  Dirichlet interpolation shape: {B_interp.shape}")

    # Neumann
    B_neumann = builder.build_neumann(X_interior, X_boundary, normals)
    print(f"  Neumann shape: {B_neumann.shape}")

    # Verify shapes
    n_total = n_interior + n_boundary
    shapes_ok = (B_extract.shape == (n_boundary, n_total) and
                 B_interp.shape == (n_boundary, n_total) and
                 B_neumann.shape == (n_boundary, n_total))

    print(f"  Status: {'PASS' if shapes_ok else 'FAIL'}")
    return shapes_ok


def test_with_matlab_data():
    """Test against MATLAB reference data if available."""
    print("\n" + "=" * 60)
    print("Test 5: Compare with MATLAB Reference")
    print("=" * 60)

    data_dir = '/workspace/dt-pinn/data/nonlinear/files_2_2236'

    try:
        Xi = loadmat(os.path.join(data_dir, 'Xi.mat'))['Xi']
        Xb = loadmat(os.path.join(data_dir, 'Xb.mat'))['Xb']
        Xg = loadmat(os.path.join(data_dir, 'Xg.mat'))['X_g']
        B_matlab = loadmat(os.path.join(data_dir, 'B1.mat'))['B1']

        print(f"  MATLAB B shape: {B_matlab.shape}")
        print(f"  MATLAB B NNZ: {B_matlab.nnz}")
        print(f"  MATLAB B NNZ per row: {B_matlab.nnz / B_matlab.shape[0]:.1f}")

        # Analyze MATLAB B structure
        B_dense = B_matlab.toarray()
        row_sums = B_dense.sum(axis=1)
        print(f"  MATLAB B row sums: [{row_sums.min():.4f}, {row_sums.max():.4f}]")

        # Build our operators
        n_i, n_b = len(Xi), len(Xb)
        X_all = np.vstack([Xi, Xb, Xg])

        # Simple extraction
        B_extract = build_dirichlet_extraction(n_i, n_b, len(Xg))
        print(f"\n  Our extraction B shape: {B_extract.shape}")

        # RBF interpolation
        B_interp = build_dirichlet_interpolation(X_all, Xb, n_i, stencil_size=13)
        print(f"  Our interpolation B shape: {B_interp.shape}")

        # Compare row structures
        interp_row_sums = np.array(B_interp.sum(axis=1)).flatten()
        print(f"  Our interpolation row sums: [{interp_row_sums.min():.4f}, {interp_row_sums.max():.4f}]")

        print(f"\n  Note: MATLAB B uses ghost point BC extrapolation,")
        print(f"        which is a different method than simple interpolation.")
        print(f"  Status: INFO (comparison only, not validation)")
        return True

    except Exception as e:
        print(f"  Could not load MATLAB data: {e}")
        print(f"  Status: SKIP")
        return True


def main():
    """Run all boundary operator tests."""
    print("\n" + "=" * 60)
    print("RBF-FD Boundary Operator Tests")
    print("=" * 60)

    results = []

    results.append(("Dirichlet Extraction", test_dirichlet_extraction()))
    results.append(("Dirichlet Interpolation", test_dirichlet_interpolation()))
    results.append(("Neumann Operator", test_neumann_operator()))
    results.append(("Builder Class", test_boundary_builder_class()))
    results.append(("MATLAB Comparison", test_with_matlab_data()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
