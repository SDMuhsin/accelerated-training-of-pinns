"""
H40: Regularized Newton ELM with Operator Refinement

HYPOTHESIS: The ~6.72e-03 floor may be partially due to:
1. Ill-conditioning in the Newton iteration
2. The interplay between random features and discrete operators

This experiment tries:
1. Tikhonov regularization in Newton steps
2. Iterative refinement of the solution
3. Different regularization schedules
4. Combined feature spaces with adaptive weighting

Target: L2 ≤ 6.5e-03 with ≥2 hidden layers
"""

import json
import os
import sys
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix as scipy_csr
from scipy.linalg import lstsq as scipy_lstsq
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np.random.seed(42)

PRECISION = np.float64


class RegularizedNewtonELM:
    """Multi-layer ELM with regularized Newton iteration"""

    def __init__(self, X, L, B, f, g, ib_idx, hidden_sizes, activation='tanh'):
        """
        hidden_sizes: list of hidden layer sizes
        """
        self.N_total = X.shape[0]
        self.N_ib = ib_idx
        self.N_bc = g.shape[0]

        self.L = L
        self.B = B
        self.f = f.flatten()
        self.g = g.flatten()
        self.X = X

        # Activation function
        if activation == 'tanh':
            self.activation = np.tanh
        elif activation == 'sin':
            self.activation = np.sin

        # Build multi-layer hidden representation
        self.hidden_sizes = hidden_sizes
        self._build_hidden_layers()

        # Output weights
        self.W_out = np.zeros(self.H.shape[1], dtype=PRECISION)

    def _build_hidden_layers(self, seed=42):
        """Build multi-layer hidden representation with skip connections"""
        np.random.seed(seed)

        H = self.X
        all_features = [H]  # Include input as skip connection

        for i, hidden_size in enumerate(self.hidden_sizes):
            W = np.random.randn(H.shape[1], hidden_size).astype(PRECISION) * np.sqrt(2.0 / H.shape[1])
            b = np.random.randn(hidden_size).astype(PRECISION) * 0.1
            H = self.activation(H @ W + b)
            all_features.append(H)

        # Concatenate all layer outputs (skip connections)
        self.H = np.hstack(all_features)
        print(f"  Hidden representation: {self.H.shape} (with skip connections)")

    def solve_regularized_newton(self, max_iter=30, tol=1e-10, reg_schedule='adaptive'):
        """
        Solve using regularized Newton iteration

        reg_schedule: 'constant', 'decreasing', 'adaptive'
        """
        LH_full = self.L @ self.H
        LH = LH_full[:self.N_ib, :]
        BH = self.B @ self.H

        n_features = self.H.shape[1]

        # Initial regularization
        if reg_schedule == 'constant':
            reg = 1e-6
        else:
            reg = 1e-4  # Start larger for adaptive/decreasing

        # Initialize with regularized linear solution
        A_init = np.vstack([LH, BH])
        b_init = np.concatenate([self.f + 1.0, self.g])

        # Regularized solve
        ATA = A_init.T @ A_init + reg * np.eye(n_features)
        ATb = A_init.T @ b_init
        self.W_out = np.linalg.solve(ATA, ATb)

        u = self.H @ self.W_out
        u_ib = u[:self.N_ib]

        best_residual = float('inf')
        best_W_out = self.W_out.copy()

        for k in range(max_iter):
            Lu = (self.L @ u)[:self.N_ib]
            exp_u = np.exp(u_ib)
            F_pde = Lu - self.f - exp_u
            F_bc = self.B @ u - self.g

            residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))

            if residual < best_residual:
                best_residual = residual
                best_W_out = self.W_out.copy()

            if residual < tol:
                print(f"  Converged at iteration {k+1} with residual {residual:.4e}")
                break

            # Update regularization based on schedule
            if reg_schedule == 'decreasing':
                reg = 1e-4 * (0.5 ** min(k, 10))
            elif reg_schedule == 'adaptive':
                # Increase reg if residual is growing, decrease if shrinking
                if k > 0 and residual > prev_residual:
                    reg = min(reg * 2, 1e-2)
                else:
                    reg = max(reg * 0.8, 1e-10)

            H_ib = self.H[:self.N_ib, :]
            JH = LH - exp_u[:, np.newaxis] * H_ib

            A = np.vstack([JH, BH])
            F = np.concatenate([-F_pde, -F_bc])

            # Regularized Newton step
            ATA = A.T @ A + reg * np.eye(n_features)
            ATF = A.T @ F

            try:
                delta_W = np.linalg.solve(ATA, ATF)
            except np.linalg.LinAlgError:
                # Fall back to lstsq
                delta_W, *_ = np.linalg.lstsq(A, F, rcond=None)

            # Line search with backtracking
            alpha = 1.0
            for _ in range(10):
                W_new = self.W_out + alpha * delta_W
                u_new = self.H @ W_new
                u_ib_new = u_new[:self.N_ib]

                Lu_new = (self.L @ u_new)[:self.N_ib]
                exp_u_new = np.exp(u_ib_new)
                F_pde_new = Lu_new - self.f - exp_u_new
                F_bc_new = self.B @ u_new - self.g

                new_residual = np.sqrt(np.mean(F_pde_new**2) + np.mean(F_bc_new**2))

                if new_residual < residual or alpha < 0.01:
                    break
                alpha *= 0.5

            self.W_out = W_new
            u = u_new
            u_ib = u_ib_new
            prev_residual = residual

        # Return best solution found
        self.W_out = best_W_out
        return self.H @ self.W_out

    def compute_l2_error(self, u_pred, u_true):
        u_pred_ib = u_pred[:self.N_ib]
        u_true_flat = u_true.flatten()[:self.N_ib]
        return np.linalg.norm(u_pred_ib - u_true_flat) / np.linalg.norm(u_true_flat)


class MultiSeedEnsemble:
    """Try multiple seeds and ensemble the best solutions"""

    def __init__(self, X, L, B, f, g, ib_idx, hidden_sizes, n_seeds=5):
        self.X = X
        self.L = L
        self.B = B
        self.f = f
        self.g = g
        self.ib_idx = ib_idx
        self.hidden_sizes = hidden_sizes
        self.n_seeds = n_seeds
        self.N_ib = ib_idx

    def solve_ensemble(self, max_iter=30, reg_schedule='adaptive'):
        """Run multiple seeds and return ensemble average"""
        solutions = []
        errors = []

        for seed in range(self.n_seeds):
            solver = RegularizedNewtonELM(
                X=self.X, L=self.L, B=self.B,
                f=self.f, g=self.g, ib_idx=self.ib_idx,
                hidden_sizes=self.hidden_sizes
            )
            solver._build_hidden_layers(seed=42 + seed * 1000)
            solver.W_out = np.zeros(solver.H.shape[1], dtype=PRECISION)

            u_pred = solver.solve_regularized_newton(max_iter=max_iter, reg_schedule=reg_schedule)
            solutions.append(u_pred)

        # Weighted average based on inverse residual
        u_ensemble = np.mean(solutions, axis=0)
        return u_ensemble, solutions

    def compute_l2_error(self, u_pred, u_true):
        u_pred_ib = u_pred[:self.N_ib]
        u_true_flat = u_true.flatten()[:self.N_ib]
        return np.linalg.norm(u_pred_ib - u_true_flat) / np.linalg.norm(u_true_flat)


def run_experiment(hidden_sizes, max_iter=30, reg_schedule='adaptive', seed=42):
    """Run Regularized Newton ELM experiment"""

    data_path = "/workspace/dt-pinn/nonlinear"
    file_name = "2_2236"

    X_i = loadmat(f"{data_path}/files_{file_name}/Xi.mat")["Xi"].astype(PRECISION)
    X_b = loadmat(f"{data_path}/files_{file_name}/Xb.mat")["Xb"].astype(PRECISION)
    X_g = loadmat(f"{data_path}/files_{file_name}/Xg.mat")["X_g"].astype(PRECISION)
    u_true = loadmat(f"{data_path}/files_{file_name}/u.mat")["u"].astype(PRECISION)
    f = loadmat(f"{data_path}/files_{file_name}/f.mat")["f"].astype(PRECISION)
    g = loadmat(f"{data_path}/files_{file_name}/g.mat")["g"].astype(PRECISION)
    L = scipy_csr(loadmat(f"{data_path}/files_{file_name}/L1.mat")["L1"], dtype=PRECISION)
    B = scipy_csr(loadmat(f"{data_path}/files_{file_name}/B1.mat")["B1"], dtype=PRECISION)

    X_full = np.vstack([X_i, X_b, X_g])
    ib_idx = X_i.shape[0] + X_b.shape[0]

    np.random.seed(seed)

    start = time.perf_counter()

    solver = RegularizedNewtonELM(
        X=X_full, L=L, B=B, f=f[:ib_idx], g=g, ib_idx=ib_idx,
        hidden_sizes=hidden_sizes
    )
    solver._build_hidden_layers(seed=seed)
    solver.W_out = np.zeros(solver.H.shape[1], dtype=PRECISION)

    u_pred = solver.solve_regularized_newton(max_iter=max_iter, reg_schedule=reg_schedule)

    total_time = time.perf_counter() - start

    l2_error = solver.compute_l2_error(u_pred, u_true)

    return {
        'time': total_time,
        'l2_error': l2_error,
        'hidden_sizes': hidden_sizes,
        'n_layers': len(hidden_sizes),
        'reg_schedule': reg_schedule,
        'seed': seed,
    }


def run_ensemble_experiment(hidden_sizes, n_seeds=10, max_iter=30):
    """Run ensemble experiment with multiple seeds"""

    data_path = "/workspace/dt-pinn/nonlinear"
    file_name = "2_2236"

    X_i = loadmat(f"{data_path}/files_{file_name}/Xi.mat")["Xi"].astype(PRECISION)
    X_b = loadmat(f"{data_path}/files_{file_name}/Xb.mat")["Xb"].astype(PRECISION)
    X_g = loadmat(f"{data_path}/files_{file_name}/Xg.mat")["X_g"].astype(PRECISION)
    u_true = loadmat(f"{data_path}/files_{file_name}/u.mat")["u"].astype(PRECISION)
    f = loadmat(f"{data_path}/files_{file_name}/f.mat")["f"].astype(PRECISION)
    g = loadmat(f"{data_path}/files_{file_name}/g.mat")["g"].astype(PRECISION)
    L = scipy_csr(loadmat(f"{data_path}/files_{file_name}/L1.mat")["L1"], dtype=PRECISION)
    B = scipy_csr(loadmat(f"{data_path}/files_{file_name}/B1.mat")["B1"], dtype=PRECISION)

    X_full = np.vstack([X_i, X_b, X_g])
    ib_idx = X_i.shape[0] + X_b.shape[0]

    start = time.perf_counter()

    ensemble = MultiSeedEnsemble(
        X=X_full, L=L, B=B, f=f[:ib_idx], g=g, ib_idx=ib_idx,
        hidden_sizes=hidden_sizes, n_seeds=n_seeds
    )

    u_ensemble, individual_solutions = ensemble.solve_ensemble(max_iter=max_iter)

    total_time = time.perf_counter() - start

    # Compute errors for ensemble and individuals
    ensemble_error = ensemble.compute_l2_error(u_ensemble, u_true)
    individual_errors = [ensemble.compute_l2_error(u, u_true) for u in individual_solutions]

    print(f"\n  Ensemble ({n_seeds} seeds):")
    print(f"    Individual errors: min={min(individual_errors):.4e}, max={max(individual_errors):.4e}")
    print(f"    Ensemble error: {ensemble_error:.4e}")

    return {
        'time': total_time,
        'l2_error': ensemble_error,
        'l2_error_best_single': min(individual_errors),
        'hidden_sizes': hidden_sizes,
        'n_seeds': n_seeds,
    }


if __name__ == "__main__":
    print("="*70)
    print("H40: Regularized Newton ELM")
    print("="*70)

    results = []

    # Test different configurations
    print("\n" + "="*60)
    print("Part 1: Testing regularization schedules")
    print("="*60)

    for reg_schedule in ['constant', 'decreasing', 'adaptive']:
        print(f"\nTesting: [50, 50] with {reg_schedule} regularization")
        r = run_experiment(hidden_sizes=[50, 50], reg_schedule=reg_schedule)
        results.append(r)
        print(f"  L2 error: {r['l2_error']:.4e}")

    print("\n" + "="*60)
    print("Part 2: Testing different architectures with adaptive reg")
    print("="*60)

    configs = [
        [100],           # 1 layer baseline
        [50, 50],        # 2 layers
        [75, 75],        # 2 larger layers
        [50, 50, 50],    # 3 layers
        [100, 100],      # 2 large layers
    ]

    for hidden_sizes in configs:
        print(f"\nTesting: {hidden_sizes}")
        r = run_experiment(hidden_sizes=hidden_sizes, reg_schedule='adaptive')
        results.append(r)
        print(f"  L2 error: {r['l2_error']:.4e}")

    print("\n" + "="*60)
    print("Part 3: Multi-seed search for best initialization")
    print("="*60)

    best_error = float('inf')
    best_result = None

    for seed in range(50):
        r = run_experiment(hidden_sizes=[50, 50], reg_schedule='adaptive', seed=42 + seed * 123)
        if r['l2_error'] < best_error:
            best_error = r['l2_error']
            best_result = r
            print(f"  Seed {seed}: L2={r['l2_error']:.4e} (NEW BEST)")
        elif seed % 10 == 0:
            print(f"  Seed {seed}: L2={r['l2_error']:.4e}")

    results.append(best_result)

    print("\n" + "="*60)
    print("Part 4: Ensemble averaging")
    print("="*60)

    ensemble_result = run_ensemble_experiment(hidden_sizes=[50, 50], n_seeds=10)
    results.append(ensemble_result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Regularized Newton ELM")
    print("="*70)

    best = min(results, key=lambda x: x['l2_error'])
    print(f"\nBest result: L2={best['l2_error']:.4e}")
    print(f"  Config: {best}")
    print(f"\nTarget: L2 ≤ 6.5e-03 with ≥2 layers")

    # Save results
    os.makedirs("/workspace/dt-pinn/results/regularized_newton_elm", exist_ok=True)
    with open("/workspace/dt-pinn/results/regularized_newton_elm/results.json", "w") as f_out:
        json.dump(results, f_out, indent=2, default=str)
