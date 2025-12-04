"""
Validation Test for RBF-FD Operators

Compares Python-generated L and B matrices against MATLAB reference.
Success criteria: relative error < 1e-6
"""

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbf_fd import RBFFDOperators


def load_matlab_data(data_dir: str):
    """Load MATLAB data files."""
    Xi = loadmat(os.path.join(data_dir, 'Xi.mat'))['Xi']
    Xb = loadmat(os.path.join(data_dir, 'Xb.mat'))['Xb']
    Xg = loadmat(os.path.join(data_dir, 'Xg.mat'))['X_g']
    L_matlab = csr_matrix(loadmat(os.path.join(data_dir, 'L1.mat'))['L1'])
    B_matlab = csr_matrix(loadmat(os.path.join(data_dir, 'B1.mat'))['B1'])

    return Xi, Xb, Xg, L_matlab, B_matlab


def analyze_matlab_stencil(L_matlab, B_matlab):
    """Analyze MATLAB operator stencil properties."""
    # L matrix analysis
    L_nnz_per_row = np.diff(L_matlab.indptr)
    L_stencil = int(np.median(L_nnz_per_row))

    # B matrix analysis
    B_nnz_per_row = np.diff(B_matlab.indptr)
    B_stencil = int(np.median(B_nnz_per_row))

    return L_stencil, B_stencil


def validate_operators(data_dir: str):
    """Main validation function."""
    print("=" * 60)
    print("RBF-FD Operator Validation")
    print("=" * 60)

    # Load MATLAB data
    print("\n1. Loading MATLAB data...")
    Xi, Xb, Xg, L_matlab, B_matlab = load_matlab_data(data_dir)

    print(f"   Interior points: {Xi.shape}")
    print(f"   Boundary points: {Xb.shape}")
    print(f"   Ghost points: {Xg.shape}")
    print(f"   L_matlab shape: {L_matlab.shape}")
    print(f"   B_matlab shape: {B_matlab.shape}")

    # Analyze MATLAB stencils
    print("\n2. Analyzing MATLAB stencil sizes...")
    L_stencil, B_stencil = analyze_matlab_stencil(L_matlab, B_matlab)
    print(f"   L stencil size: {L_stencil}")
    print(f"   B stencil size: {B_stencil}")

    # Build Python operators
    print("\n3. Building Python RBF-FD operators...")
    print(f"   Using stencil_size={L_stencil}, boundary_stencil_size={B_stencil}")

    # Try different polynomial degrees and RBF orders
    poly_degrees = [2, 3, 4]
    rbf_orders = [3, 5, 7]

    best_L_error = float('inf')
    best_B_error = float('inf')
    best_params = None

    for poly_deg in poly_degrees:
        for rbf_ord in rbf_orders:
            try:
                gen = RBFFDOperators(
                    stencil_size=L_stencil,
                    poly_degree=poly_deg,
                    rbf_order=rbf_ord,
                    boundary_stencil_size=B_stencil,
                )

                L_python, B_python = gen.build_operators(
                    Xi, Xb, Xg, verbose=False
                )

                # Compute relative errors
                L_py = L_python.toarray()
                L_mat = L_matlab.toarray()
                L_rel_err = np.linalg.norm(L_py - L_mat) / np.linalg.norm(L_mat)

                B_py = B_python.toarray()
                B_mat = B_matlab.toarray()
                B_rel_err = np.linalg.norm(B_py - B_mat) / np.linalg.norm(B_mat)

                print(f"   poly_deg={poly_deg}, rbf_ord={rbf_ord}: L_err={L_rel_err:.2e}, B_err={B_rel_err:.2e}")

                if L_rel_err < best_L_error:
                    best_L_error = L_rel_err
                    best_B_error = B_rel_err
                    best_params = (poly_deg, rbf_ord)

            except Exception as e:
                print(f"   poly_deg={poly_deg}, rbf_ord={rbf_ord}: FAILED - {e}")

    print("\n4. Best results:")
    print(f"   Parameters: poly_deg={best_params[0]}, rbf_ord={best_params[1]}")
    print(f"   L relative error: {best_L_error:.6e}")
    print(f"   B relative error: {best_B_error:.6e}")

    # Detailed analysis with best parameters
    print("\n5. Detailed analysis with best parameters...")
    gen = RBFFDOperators(
        stencil_size=L_stencil,
        poly_degree=best_params[0],
        rbf_order=best_params[1],
        boundary_stencil_size=B_stencil,
    )

    L_python, B_python = gen.build_operators(Xi, Xb, Xg, verbose=True)

    L_py = L_python.toarray()
    L_mat = L_matlab.toarray()
    B_py = B_python.toarray()
    B_mat = B_matlab.toarray()

    print(f"\n   L matrix comparison:")
    print(f"     Python shape: {L_python.shape}, NNZ: {L_python.nnz}")
    print(f"     MATLAB shape: {L_matlab.shape}, NNZ: {L_matlab.nnz}")
    print(f"     Relative Frobenius error: {np.linalg.norm(L_py - L_mat) / np.linalg.norm(L_mat):.6e}")
    print(f"     Max absolute diff: {np.abs(L_py - L_mat).max():.6e}")
    print(f"     Python row sum range: [{L_py.sum(axis=1).min():.6e}, {L_py.sum(axis=1).max():.6e}]")
    print(f"     MATLAB row sum range: [{L_mat.sum(axis=1).min():.6e}, {L_mat.sum(axis=1).max():.6e}]")

    print(f"\n   B matrix comparison:")
    print(f"     Python shape: {B_python.shape}, NNZ: {B_python.nnz}")
    print(f"     MATLAB shape: {B_matlab.shape}, NNZ: {B_matlab.nnz}")
    print(f"     Relative Frobenius error: {np.linalg.norm(B_py - B_mat) / np.linalg.norm(B_mat):.6e}")
    print(f"     Max absolute diff: {np.abs(B_py - B_mat).max():.6e}")

    # Check success criteria
    print("\n" + "=" * 60)
    print("VALIDATION RESULT")
    print("=" * 60)

    L_pass = best_L_error < 1e-6
    B_pass = best_B_error < 1e-6

    print(f"   L matrix: {'PASS' if L_pass else 'FAIL'} (error={best_L_error:.2e}, threshold=1e-6)")
    print(f"   B matrix: {'PASS' if B_pass else 'FAIL'} (error={best_B_error:.2e}, threshold=1e-6)")

    if L_pass and B_pass:
        print("\n   ✓ ALL TESTS PASSED!")
    else:
        print("\n   ✗ VALIDATION FAILED - needs debugging")

    return L_pass and B_pass, best_L_error, best_B_error, best_params


if __name__ == '__main__':
    data_dir = '/workspace/dt-pinn/data/nonlinear/files_2_2236'
    success, L_err, B_err, params = validate_operators(data_dir)
    sys.exit(0 if success else 1)
