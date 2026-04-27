"""
Speed benchmark: Compare forward+backward time for DT-PINN (spectral),
JVP (forward-mode AD), and pure autograd methods.

This determines which approach is feasible for training.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sage_partner_ns import (
    FNN_NS, build_3d_grid, compute_pde_ns_3d, compute_pde_ns_3d_jvp,
    pde_residuals_autodiff, NU, V0,
)

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
mse = nn.MSELoss()

# Build grid
print("\nBuilding 3D Chebyshev grid (Nx=55, Ny=15, Nt=30)...")
g = build_3d_grid(55, 15, 30, device)
N_all = g['N_all']
ii = g['interior_idx']
print(f"Total points: {N_all}, Interior: {len(ii)}")

# Create model
model = FNN_NS(input_dim=3, output_dim=3, hidden=128, n_layers=6).to(device)
print(f"Model: {sum(p.numel() for p in model.parameters())} params")


def benchmark(name, fn, n_warmup=5, n_runs=20):
    """Benchmark a function (forward + backward)."""
    # Warmup
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    print(f"  {name:30s}: mean={times.mean()*1000:.1f}ms  "
          f"std={times.std()*1000:.1f}ms  "
          f"min={times.min()*1000:.1f}ms  "
          f"max={times.max()*1000:.1f}ms")
    return times.mean()


# =========================================================================
# 1. DT-PINN: spectral forward + autograd backward
# =========================================================================
print("\n" + "=" * 70)
print("SPEED BENCHMARK: Forward + Backward per iteration")
print("=" * 70)

def dtpinn_step():
    model.zero_grad()
    pred_all = model(g['xyt_all'])
    c, mu, mv = compute_pde_ns_3d(pred_all, g)
    loss = (c[ii] ** 2).mean() + (mu[ii] ** 2).mean() + (mv[ii] ** 2).mean()
    loss.backward()
    return loss.item()

t_dtpinn = benchmark("DT-PINN (spectral+autograd bwd)", dtpinn_step)


# =========================================================================
# 2. JVP: forward-mode AD forward + autograd backward
# =========================================================================
xyt_all = g['xyt_all'].detach()  # No requires_grad needed for JVP

def jvp_step_interior():
    """JVP on interior points only (current implementation)."""
    model.zero_grad()
    xyt_int = xyt_all[ii]
    out, cont, mom_u, mom_v = compute_pde_ns_3d_jvp(xyt_int, model.net)
    loss = mse(cont, torch.zeros_like(cont)) + \
           mse(mom_u, torch.zeros_like(mom_u)) + \
           mse(mom_v, torch.zeros_like(mom_v))
    loss.backward()
    return loss.item()

def jvp_step_all():
    """JVP on ALL grid points (proposed fix)."""
    model.zero_grad()
    out, cont, mom_u, mom_v = compute_pde_ns_3d_jvp(xyt_all, model.net)
    loss = mse(cont, torch.zeros_like(cont)) + \
           mse(mom_u, torch.zeros_like(mom_u)) + \
           mse(mom_v, torch.zeros_like(mom_v))
    loss.backward()
    return loss.item()

t_jvp_int = benchmark("JVP (interior only, N=22680)", jvp_step_interior)
t_jvp_all = benchmark("JVP (all points, N=27776)", jvp_step_all)


# =========================================================================
# 3. Pure autograd (double backward)
# =========================================================================
xyt_ag = g['xyt_all'].detach().clone().requires_grad_(True)

def autograd_step_interior():
    """Pure autograd on interior points."""
    model.zero_grad()
    xyt_int_ag = xyt_all[ii].detach().requires_grad_(True)
    pred = model(xyt_int_ag)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    grad_u = torch.autograd.grad(u.sum(), xyt_int_ag, create_graph=True)[0]
    grad_v = torch.autograd.grad(v.sum(), xyt_int_ag, create_graph=True)[0]
    grad_p = torch.autograd.grad(p.sum(), xyt_int_ag, create_graph=True)[0]

    u_x, u_y, u_t = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
    v_x, v_y, v_t = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]
    p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]

    grad_u_x = torch.autograd.grad(u_x.sum(), xyt_int_ag, create_graph=True, retain_graph=True)[0]
    grad_u_y = torch.autograd.grad(u_y.sum(), xyt_int_ag, create_graph=True, retain_graph=True)[0]
    grad_v_x = torch.autograd.grad(v_x.sum(), xyt_int_ag, create_graph=True, retain_graph=True)[0]
    grad_v_y = torch.autograd.grad(v_y.sum(), xyt_int_ag, create_graph=True, retain_graph=True)[0]

    u_xx, u_yy = grad_u_x[:, 0:1], grad_u_y[:, 1:2]
    v_xx, v_yy = grad_v_x[:, 0:1], grad_v_y[:, 1:2]

    cont = u_x + v_y
    mom_u = u_t + u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
    mom_v = v_t + u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)

    loss = mse(cont, torch.zeros_like(cont)) + \
           mse(mom_u, torch.zeros_like(mom_u)) + \
           mse(mom_v, torch.zeros_like(mom_v))
    loss.backward()
    return loss.item()

t_autograd = benchmark("Autograd (interior, N=22680)", autograd_step_interior)


# =========================================================================
# 4. DT-PINN with ALL points (fix boundary exclusion bug)
# =========================================================================
def dtpinn_step_all():
    """DT-PINN with PDE on ALL grid points."""
    model.zero_grad()
    pred_all = model(g['xyt_all'])
    c, mu, mv = compute_pde_ns_3d(pred_all, g)
    loss = (c ** 2).mean() + (mu ** 2).mean() + (mv ** 2).mean()
    loss.backward()
    return loss.item()

t_dtpinn_all = benchmark("DT-PINN (all points, N=27776)", dtpinn_step_all)


# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 70)
print("SUMMARY (lower is faster)")
print("=" * 70)

baseline = t_dtpinn
methods = [
    ("DT-PINN (spectral, interior)", t_dtpinn),
    ("DT-PINN (spectral, all pts)", t_dtpinn_all),
    ("JVP (interior)", t_jvp_int),
    ("JVP (all points)", t_jvp_all),
    ("Autograd (interior)", t_autograd),
]

for name, t in methods:
    print(f"  {name:35s}: {t*1000:7.1f}ms  ({t/baseline:.2f}x vs DT-PINN)")

# Estimate training times (20K Adam epochs + L-BFGS)
print("\n--- Estimated training time (20K Adam epochs only) ---")
for name, t in methods:
    est_min = t * 20000 / 60
    print(f"  {name:35s}: ~{est_min:.1f} min")

print("\nDone.")
