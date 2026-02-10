#!/usr/bin/env python3
"""
RoPINN Benchmarking: Region-Optimized PINN on NS+Smagorinsky

Benchmarks RoPINN against existing methods on the lid-driven cavity
NS+Smagorinsky problem. RoPINN's core innovation is perturbing collocation
points within calibrated trust regions during training.

Variants benchmarked:
1. Autodiff PINN (control) — plain autograd, no perturbation
2. RoPINN + Adam — RoPINN region optimization with Adam optimizer
3. RoPINN + L-BFGS — RoPINN region optimization with L-BFGS (paper's original)

All variants use:
- Same network: 6-layer/64-unit tanh MLP (21,827 params)
- Same evaluation: 51x51 uniform grid, autodiff derivatives
- Same seeds: 42, 43, 44, 45, 46
- Same epochs: 30,000
- Same GPU: NVIDIA A40
- Autograd for all spatial derivatives (not spectral matrices)
"""

import numpy as np
import torch
import torch.nn as nn
import time
import os
import json
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# Configuration
# =============================================================================
SEEDS = [42, 43, 44, 45, 46]
N_EPOCHS = 30000
N_GRID = 50  # Chebyshev grid for collocation
LOG_INTERVAL = 5000
FEASIBILITY_EPOCHS = 200

# Physics
Re = 1000.0
U_lid = 1.0
nu_laminar = U_lid / Re  # 0.001
Cs = 0.1

# RoPINN hyperparameters (paper defaults)
ROPINN_INITIAL_REGION = 1e-4
ROPINN_SAMPLE_NUM = 1
ROPINN_PAST_ITERATIONS = 10
ROPINN_REGION_MAX = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("ROPINN BENCHMARKING: Region-Optimized PINN on NS+Smagorinsky")
print("=" * 70)
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
print(f"Seeds: {SEEDS}")
print(f"Epochs: {N_EPOCHS}")
print(f"Grid: {N_GRID}x{N_GRID}")
print(f"RoPINN: initial_region={ROPINN_INITIAL_REGION}, sample_num={ROPINN_SAMPLE_NUM}, "
      f"past_iterations={ROPINN_PAST_ITERATIONS}, region_max={ROPINN_REGION_MAX}")

# =============================================================================
# Infrastructure
# =============================================================================
def chebyshev_points(N):
    return np.cos(np.pi * np.arange(N) / (N - 1))


def build_collocation_points(N_grid):
    """Build Chebyshev collocation points and classify into interior/boundary."""
    x_ref = chebyshev_points(N_grid)
    x_phys = 0.5 * (x_ref + 1.0)
    xx, yy = np.meshgrid(x_phys, x_phys, indexing='xy')
    xy_grid = np.column_stack([xx.ravel(), yy.ravel()])

    eps = 1e-10
    xc, yc = xy_grid[:, 0], xy_grid[:, 1]
    is_boundary = (xc < eps) | (xc > 1-eps) | (yc < eps) | (yc > 1-eps)
    is_lid = (yc > 1-eps)
    is_wall = is_boundary & ~is_lid

    interior_idx = np.where(~is_boundary)[0]
    lid_idx = np.where(is_lid)[0]
    wall_idx = np.where(is_wall)[0]

    xy_interior = torch.tensor(xy_grid[interior_idx], dtype=torch.float32, device=device)
    xy_lid = torch.tensor(xy_grid[lid_idx], dtype=torch.float32, device=device)
    xy_wall = torch.tensor(xy_grid[wall_idx], dtype=torch.float32, device=device)

    return {
        'xy_interior': xy_interior,
        'xy_lid': xy_lid,
        'xy_wall': xy_wall,
        'N_interior': len(interior_idx),
        'N_lid': len(lid_idx),
        'N_wall': len(wall_idx),
        'N_grid': N_grid,
    }


class PINN_Cavity(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(2, 64), nn.Tanh()]
        for _ in range(5):
            layers.extend([nn.Linear(64, 64), nn.Tanh()])
        layers.append(nn.Linear(64, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


mse = nn.MSELoss()


def gradients(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True)[0]


# =============================================================================
# Autodiff PDE residuals for NS+Smagorinsky
# =============================================================================
def pde_residuals_autodiff(model, xy):
    """Compute PDE residuals using autograd. xy must have requires_grad=True."""
    pred = model(xy)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    grad_u = gradients(u, xy)
    grad_v = gradients(v, xy)
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]

    # Smagorinsky eddy viscosity
    x_coord, y_coord = xy[:, 0:1], xy[:, 1:2]
    d_wall = torch.min(torch.min(x_coord, 1.0 - x_coord),
                       torch.min(y_coord, 1.0 - y_coord))
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs * d_wall)**2 * S_mag

    # Continuity
    continuity = du_dx + dv_dy

    # Convection
    u_conv = u * du_dx + v * du_dy
    v_conv = u * dv_dx + v * dv_dy

    # Pressure gradient
    grad_p = gradients(p, xy)
    dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]

    # Viscous: div(nu_eff * grad(u)) and div(nu_eff * grad(v))
    qx_u, qy_u = nu_eff * du_dx, nu_eff * du_dy
    qx_v, qy_v = nu_eff * dv_dx, nu_eff * dv_dy
    grad_qx_u = gradients(qx_u, xy)
    grad_qy_u = gradients(qy_u, xy)
    grad_qx_v = gradients(qx_v, xy)
    grad_qy_v = gradients(qy_v, xy)
    visc_u = grad_qx_u[:, 0:1] + grad_qy_u[:, 1:2]
    visc_v = grad_qx_v[:, 0:1] + grad_qy_v[:, 1:2]

    # Momentum
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v

    return continuity, mom_u, mom_v


# =============================================================================
# Evaluation (autodiff on 51x51 uniform grid — same as all phases)
# =============================================================================
def evaluate_model(model):
    """Evaluate PDE residual on 51x51 uniform grid using autodiff."""
    nx, ny = 51, 51
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    xy_eval = np.column_stack([X.ravel(), Y.ravel()])
    xy_t = torch.tensor(xy_eval, dtype=torch.float32, device=device, requires_grad=True)

    model.eval()
    pred = model(xy_t)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    grad_u = gradients(u, xy_t)
    grad_v = gradients(v, xy_t)
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]

    x_coord, y_coord = xy_t[:, 0:1], xy_t[:, 1:2]
    d = torch.min(torch.min(x_coord, 1.0 - x_coord),
                  torch.min(y_coord, 1.0 - y_coord))
    Sxx, Syy, Sxy = du_dx, dv_dy, 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs * d)**2 * S_mag
    continuity = du_dx + dv_dy
    u_conv = u * du_dx + v * du_dy
    v_conv = u * dv_dx + v * dv_dy
    grad_p = gradients(p, xy_t)
    dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]
    qx_u = nu_eff * du_dx; qy_u = nu_eff * du_dy
    qx_v = nu_eff * dv_dx; qy_v = nu_eff * dv_dy
    gqxu = gradients(qx_u, xy_t); gqyu = gradients(qy_u, xy_t)
    gqxv = gradients(qx_v, xy_t); gqyv = gradients(qy_v, xy_t)
    visc_u = gqxu[:, 0:1] + gqyu[:, 1:2]
    visc_v = gqxv[:, 0:1] + gqyv[:, 1:2]
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v

    cont_np = continuity.detach().cpu().numpy().flatten()
    mom_u_np = mom_u.detach().cpu().numpy().flatten()
    mom_v_np = mom_v.detach().cpu().numpy().flatten()

    pde_rms = float(np.sqrt(np.mean(cont_np**2 + mom_u_np**2 + mom_v_np**2)))
    cont_rms = float(np.sqrt(np.mean(cont_np**2)))
    mom_rms = float(np.sqrt(np.mean(mom_u_np**2 + mom_v_np**2)))

    model.train()
    return {'pde_rms': pde_rms, 'continuity_rms': cont_rms, 'momentum_rms': mom_rms}


# =============================================================================
# Gradient variance computation (from RoPINN paper)
# =============================================================================
def compute_gradient_variance(gradient_list):
    """
    Compute normalized gradient variance for trust region calibration.

    variance = mean(std(gradients) / (mean(|gradients|) + eps))
    """
    if len(gradient_list) < 2:
        return 1.0

    gradient_array = np.array(gradient_list)
    std_grad = np.std(gradient_array, axis=0)
    mean_abs_grad = np.mean(np.abs(gradient_array), axis=0) + 1e-6
    variance = float((std_grad / mean_abs_grad).mean())

    if variance == 0:
        variance = 1.0
    return variance


# =============================================================================
# METHOD 1: Plain Autodiff PINN (control — no perturbation)
# =============================================================================
def train_autodiff_pinn(seed, g, n_epochs=N_EPOCHS, verbose=True):
    """Plain autodiff PINN: autograd derivatives, no region optimization."""
    torch.manual_seed(seed)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    xy_int = g['xy_interior'].clone().requires_grad_(True)
    xy_lid = g['xy_lid']
    xy_wall = g['xy_wall']
    xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # PDE residual on interior (autograd)
        continuity, mom_u, mom_v = pde_residuals_autodiff(model, xy_int)
        loss_pde = mse(continuity, torch.zeros_like(continuity)) + \
                   mse(mom_u, torch.zeros_like(mom_u)) + \
                   mse(mom_v, torch.zeros_like(mom_v))

        # Boundary conditions
        pred_lid = model(xy_lid)
        loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
                   mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))
        pred_wall = model(xy_wall)
        loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                    mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))
        pred_c = model(xy_center)
        loss_p = mse(pred_c[:, 2:3], torch.zeros_like(pred_c[:, 2:3]))

        loss = loss_pde + loss_lid + loss_wall + loss_p
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % LOG_INTERVAL == 0:
            print(f"      Epoch {epoch+1}: loss={loss.item():.6f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    return model, time.perf_counter() - start


# =============================================================================
# METHOD 2: RoPINN + Adam
# =============================================================================
def train_ropinn_adam(seed, g, n_epochs=N_EPOCHS, verbose=True):
    """RoPINN with Adam: region optimization with Adam optimizer."""
    torch.manual_seed(seed)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    xy_int_base = g['xy_interior']  # Base interior points (never modified)
    xy_lid = g['xy_lid']
    xy_wall = g['xy_wall']
    xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)

    # RoPINN state
    gradient_list = []
    gradient_variance = 1.0

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # --- RoPINN: Trust region calibration ---
        current_region = np.clip(
            ROPINN_INITIAL_REGION / gradient_variance,
            a_min=0,
            a_max=ROPINN_REGION_MAX
        )

        # --- RoPINN: Perturb interior collocation points ---
        perturbation = torch.rand_like(xy_int_base) * current_region
        xy_perturbed = xy_int_base + perturbation
        # Clamp to [0,1]^2 domain
        xy_perturbed = torch.clamp(xy_perturbed, 0.0, 1.0)
        xy_perturbed = xy_perturbed.detach().requires_grad_(True)

        # PDE residual on perturbed interior (autograd)
        continuity, mom_u, mom_v = pde_residuals_autodiff(model, xy_perturbed)
        loss_pde = mse(continuity, torch.zeros_like(continuity)) + \
                   mse(mom_u, torch.zeros_like(mom_u)) + \
                   mse(mom_v, torch.zeros_like(mom_v))

        # Boundary conditions (NOT perturbed — fixed)
        pred_lid = model(xy_lid)
        loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
                   mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))
        pred_wall = model(xy_wall)
        loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                    mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))
        pred_c = model(xy_center)
        loss_p = mse(pred_c[:, 2:3], torch.zeros_like(pred_c[:, 2:3]))

        loss = loss_pde + loss_lid + loss_wall + loss_p
        loss.backward()
        optimizer.step()

        # --- RoPINN: Gradient tracking for trust region calibration ---
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        flat_grad = torch.cat(grads).cpu().numpy()
        gradient_list.append(flat_grad)
        gradient_list = gradient_list[-ROPINN_PAST_ITERATIONS:]
        gradient_variance = compute_gradient_variance(gradient_list)

        if verbose and (epoch + 1) % LOG_INTERVAL == 0:
            print(f"      Epoch {epoch+1}: loss={loss.item():.6f}, "
                  f"region={current_region:.2e}, grad_var={gradient_variance:.4f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    return model, time.perf_counter() - start


# =============================================================================
# METHOD 3: RoPINN + L-BFGS (paper's original optimizer)
# =============================================================================
def train_ropinn_lbfgs(seed, g, n_epochs=N_EPOCHS, verbose=True):
    """RoPINN with L-BFGS: region optimization with L-BFGS (paper's default)."""
    torch.manual_seed(seed)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        line_search_fn='strong_wolfe'
    )

    xy_int_base = g['xy_interior']
    xy_lid = g['xy_lid']
    xy_wall = g['xy_wall']
    xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)

    # RoPINN state
    gradient_list_overall = []
    gradient_list_temp = []
    gradient_variance = 1.0

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(n_epochs):
        # Trust region calibration
        current_region = np.clip(
            ROPINN_INITIAL_REGION / gradient_variance,
            a_min=0,
            a_max=ROPINN_REGION_MAX
        )

        def closure():
            optimizer.zero_grad()

            # RoPINN: Perturb interior collocation points
            perturbation = torch.rand_like(xy_int_base) * current_region
            xy_perturbed = xy_int_base + perturbation
            xy_perturbed = torch.clamp(xy_perturbed, 0.0, 1.0)
            xy_perturbed = xy_perturbed.detach().requires_grad_(True)

            continuity, mom_u, mom_v = pde_residuals_autodiff(model, xy_perturbed)
            loss_pde = mse(continuity, torch.zeros_like(continuity)) + \
                       mse(mom_u, torch.zeros_like(mom_u)) + \
                       mse(mom_v, torch.zeros_like(mom_v))

            pred_lid = model(xy_lid)
            loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
                       mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))
            pred_wall = model(xy_wall)
            loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                        mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))
            pred_c = model(xy_center)
            loss_p = mse(pred_c[:, 2:3], torch.zeros_like(pred_c[:, 2:3]))

            loss = loss_pde + loss_lid + loss_wall + loss_p
            loss.backward()

            # Gradient tracking (L-BFGS may call closure multiple times)
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
            flat_grad = torch.cat(grads).cpu().numpy()
            gradient_list_temp.append(flat_grad)

            return loss

        loss = optimizer.step(closure)

        # Trust region calibration — average gradients from closure calls
        if gradient_list_temp:
            avg_gradient = np.mean(np.array(gradient_list_temp), axis=0)
            gradient_list_overall.append(avg_gradient)
            gradient_list_overall = gradient_list_overall[-ROPINN_PAST_ITERATIONS:]
            gradient_variance = compute_gradient_variance(gradient_list_overall)
            gradient_list_temp.clear()

        if verbose and (epoch + 1) % LOG_INTERVAL == 0:
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            print(f"      Epoch {epoch+1}: loss={loss_val:.6f}, "
                  f"region={current_region:.2e}, grad_var={gradient_variance:.4f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    return model, time.perf_counter() - start


# =============================================================================
# PHASE 1: Feasibility testing (200 epochs each)
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 1: FEASIBILITY TESTING (200 epochs)")
print("=" * 70)

g = build_collocation_points(N_GRID)
print(f"Grid built: {g['N_interior']} interior, {g['N_lid']} lid, {g['N_wall']} wall")

METHODS = [
    ("Autodiff PINN (control)", train_autodiff_pinn),
    ("RoPINN + Adam", train_ropinn_adam),
    ("RoPINN + L-BFGS", train_ropinn_lbfgs),
]

feasibility_results = {}

for method_name, train_fn in METHODS:
    print(f"\n  Testing: {method_name}...")
    try:
        torch.cuda.empty_cache()
        model_test, time_test = train_fn(42, g, n_epochs=FEASIBILITY_EPOCHS, verbose=False)
        time_per_epoch_ms = (time_test / FEASIBILITY_EPOCHS) * 1000
        est_30k_min = (time_per_epoch_ms * N_EPOCHS) / 1000 / 60
        feasibility_results[method_name] = {
            'status': 'OK',
            'time_200ep': round(time_test, 3),
            'time_per_epoch_ms': round(time_per_epoch_ms, 2),
            'estimated_30k_min': round(est_30k_min, 1),
        }
        print(f"    OK: {time_per_epoch_ms:.2f} ms/epoch, est. {est_30k_min:.1f} min for 30K")
        del model_test
        torch.cuda.empty_cache()
    except Exception as e:
        feasibility_results[method_name] = {
            'status': 'FAILED',
            'error': str(e),
        }
        print(f"    FAILED: {e}")
        traceback.print_exc()

print("\n" + "-" * 70)
print("FEASIBILITY SUMMARY")
print("-" * 70)
print(f"{'Method':<30} {'Status':<10} {'ms/epoch':<12} {'Est. 30K (min)':<15}")
print("-" * 67)
for name, res in feasibility_results.items():
    if res['status'] == 'OK':
        print(f"{name:<30} {'OK':<10} {res['time_per_epoch_ms']:<12.2f} {res['estimated_30k_min']:<15.1f}")
    else:
        print(f"{name:<30} {'FAILED':<10} {'--':<12} {'--':<15}")

# Decision gate: skip variants estimated >60 min
MAX_TIME_MIN = 60
viable_methods = []
for method_name, train_fn in METHODS:
    res = feasibility_results.get(method_name, {})
    if res.get('status') == 'OK':
        if res['estimated_30k_min'] <= MAX_TIME_MIN:
            viable_methods.append((method_name, train_fn))
        else:
            print(f"\n  SKIPPING {method_name}: estimated {res['estimated_30k_min']:.1f} min > {MAX_TIME_MIN} min limit")
    else:
        print(f"\n  SKIPPING {method_name}: failed feasibility")

# =============================================================================
# PHASE 2: Full 30K benchmarks
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 2: FULL 30K BENCHMARKS")
print("=" * 70)
print(f"Running {len(viable_methods)} methods x {len(SEEDS)} seeds")

all_results = {}

for method_name, train_fn in viable_methods:
    print(f"\n{'='*70}")
    print(f"METHOD: {method_name}")
    print(f"{'='*70}")

    method_results = []
    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        torch.cuda.empty_cache()
        model, total_time = train_fn(seed, g, n_epochs=N_EPOCHS, verbose=True)
        metrics = evaluate_model(model)

        result = {
            'seed': seed,
            'total_time_s': round(total_time, 2),
            'total_time_min': round(total_time / 60, 2),
            **metrics,
        }
        method_results.append(result)
        print(f"  RESULT: Time={total_time:.1f}s ({total_time/60:.2f}min), "
              f"PDE_RMS={metrics['pde_rms']:.5f}")

        del model
        torch.cuda.empty_cache()

    all_results[method_name] = method_results

# =============================================================================
# PHASE 3: Summary and Pareto analysis
# =============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS: ROPINN BENCHMARKING")
print("=" * 70)

# Reference baselines from previous phases
REFERENCE_BASELINES = {
    'PIELM': {'time_min': 0.8, 'rms': 0.093},
    'Analytical Jacobian': {'time_min': 2.58, 'rms': 0.046},
    'Standard DT-PINN': {'time_min': 10.77, 'rms': 0.030},
    'Autodiff PINN (Phase 2)': {'time_min': 22.4, 'rms': 0.060},
}

print(f"\n{'Method':<30} {'Time (min)':<18} {'PDE RMS':<22} {'Speedup vs AD':<15}")
print("-" * 85)

# Print reference baselines
for name, ref in REFERENCE_BASELINES.items():
    speedup = 22.4 / ref['time_min']
    print(f"{name:<30} {ref['time_min']:<18.2f} {ref['rms']:<22.4f} {speedup:.2f}x")

print("-" * 85)

summary = {}
for method_name, results in all_results.items():
    times = [r['total_time_min'] for r in results]
    rms_vals = [r['pde_rms'] for r in results]
    t_mean, t_std = float(np.mean(times)), float(np.std(times))
    r_mean, r_std = float(np.mean(rms_vals)), float(np.std(rms_vals))
    speedup = 22.4 / t_mean

    time_str = f"{t_mean:.2f} +/- {t_std:.2f}"
    rms_str = f"{r_mean:.4f} +/- {r_std:.4f}"
    print(f"{method_name:<30} {time_str:<18} {rms_str:<22} {speedup:.2f}x")

    summary[method_name] = {
        'time_mean_min': round(t_mean, 2),
        'time_std_min': round(t_std, 2),
        'rms_mean': round(r_mean, 4),
        'rms_std': round(r_std, 4),
        'speedup_vs_autodiff': round(speedup, 2),
        'per_seed': results,
    }

# Pareto analysis
print("\n" + "-" * 70)
print("PARETO FRONTIER ANALYSIS")
print("-" * 70)

all_points = [(name, ref['time_min'], ref['rms']) for name, ref in REFERENCE_BASELINES.items()]
for name, s in summary.items():
    all_points.append((name, s['time_mean_min'], s['rms_mean']))

# Sort by time (ascending)
all_points.sort(key=lambda x: x[1])

# Find Pareto frontier (lower time AND lower RMS is better)
pareto = []
best_rms = float('inf')
for name, t, r in all_points:
    if r < best_rms:
        pareto.append(name)
        best_rms = r

print(f"\n{'Method':<30} {'Time (min)':<12} {'PDE RMS':<12} {'Pareto?':<10}")
print("-" * 64)
for name, t, r in all_points:
    is_pareto = 'YES' if name in pareto else 'no'
    print(f"{name:<30} {t:<12.2f} {r:<12.4f} {is_pareto:<10}")

# =============================================================================
# Save results
# =============================================================================
os.makedirs('results/ropinn_benchmark', exist_ok=True)
output = {
    'feasibility': feasibility_results,
    'full_results': summary,
    'pareto_frontier': pareto,
    'reference_baselines': REFERENCE_BASELINES,
    'ropinn_hyperparams': {
        'initial_region': ROPINN_INITIAL_REGION,
        'sample_num': ROPINN_SAMPLE_NUM,
        'past_iterations': ROPINN_PAST_ITERATIONS,
        'region_max': ROPINN_REGION_MAX,
    },
    'config': {
        'seeds': SEEDS,
        'n_epochs': N_EPOCHS,
        'n_grid': N_GRID,
        'device': str(device),
        'pytorch_version': torch.__version__,
    }
}

output_path = 'results/ropinn_benchmark/results.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

# Key finding
print("\n" + "=" * 70)
print("KEY FINDING")
print("=" * 70)
for name, s in summary.items():
    aj_time, aj_rms = 2.58, 0.046
    if s['time_mean_min'] > aj_time and s['rms_mean'] >= aj_rms:
        print(f"{name}: DOMINATED by Analytical Jacobian "
              f"(slower: {s['time_mean_min']:.1f} vs {aj_time} min, "
              f"same/worse accuracy: {s['rms_mean']:.4f} vs {aj_rms})")
    elif s['time_mean_min'] < aj_time and s['rms_mean'] < aj_rms:
        print(f"{name}: NEW PARETO POINT! "
              f"(faster: {s['time_mean_min']:.1f} vs {aj_time} min, "
              f"better accuracy: {s['rms_mean']:.4f} vs {aj_rms})")
    else:
        print(f"{name}: Pareto-comparable to Analytical Jacobian "
              f"(time: {s['time_mean_min']:.1f} vs {aj_time} min, "
              f"RMS: {s['rms_mean']:.4f} vs {aj_rms})")
