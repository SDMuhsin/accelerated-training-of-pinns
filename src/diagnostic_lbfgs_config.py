"""
Experiment 2: L-BFGS Configuration Test

Tests whether DeepXDE's more aggressive L-BFGS settings improve DT-PINN accuracy.
Retrains DT-PINN from scratch with DeepXDE's L-BFGS parameters while keeping
everything else the same.

DeepXDE L-BFGS: max_iter=1000, max_eval=1250, tolerance_grad=1e-8,
                tolerance_change=0, history_size=100
Our L-BFGS:     max_iter=20, max_eval=25, tolerance_grad=1e-7,
                tolerance_change=1e-9, history_size=50

Note: Even if this improves results, the derivative mismatch from Experiment 1
(1208x gap) is the dominant issue. This experiment tests whether L-BFGS config
is a secondary contributor.
"""

import argparse
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sage_partner_ns import (
    FNN_NS, build_3d_grid, compute_pde_ns_3d, compute_losses, evaluate_ns,
    NU, V0, X_MIN, X_MAX, Y_MIN, Y_MAX, T_MIN, T_MAX, mse, LOG_INTERVAL,
)

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("\n" + "=" * 70)
print("EXPERIMENT 2: L-BFGS Configuration Test")
print("=" * 70)
print("Training DT-PINN with DeepXDE's L-BFGS parameters.")
print("All other settings identical to the original DT-PINN run.")

# =============================================================================
# Build grid and model (same as original)
# =============================================================================
Nx, Ny, Nt = 55, 15, 30
adam_epochs = 20000
lr = 1e-3
lbfgs_steps = 15000  # max outer steps (same as original)

print(f"\nGrid: Nx={Nx}, Ny={Ny}, Nt={Nt}")
print(f"Adam: {adam_epochs} epochs, lr={lr}")
print(f"L-BFGS: max {lbfgs_steps} outer steps, DeepXDE config")

g = build_3d_grid(Nx, Ny, Nt, device)
N_all = g['N_all']
ii = g['interior_idx']
print(f"Total points: {N_all}, Interior: {len(ii)}")

model = FNN_NS(input_dim=3, output_dim=3, hidden=128, n_layers=6).to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters())}")

# =============================================================================
# Adam phase (identical to original)
# =============================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
start = time.perf_counter()

print(f"\n[DT-PINN+DeepXDE-LBFGS] Adam phase: {adam_epochs} epochs")
for epoch in range(adam_epochs):
    optimizer.zero_grad()

    pred_batch = model(g['xyt_batched'])
    pred_all = pred_batch[:N_all]
    c, mu, mv = compute_pde_ns_3d(pred_all, g)
    loss_pde = (c[ii] ** 2).mean() + (mu[ii] ** 2).mean() + (mv[ii] ** 2).mean()

    pred_inlet = pred_batch[g['off_inlet']:g['off_wall']]
    pred_wall = pred_batch[g['off_wall']:g['off_outlet']]
    pred_outlet = pred_batch[g['off_outlet']:g['off_ic']]
    pred_ic = pred_batch[g['off_ic']:]

    inlet_target = torch.zeros_like(pred_inlet)
    inlet_target[:, 0] = V0
    loss_inlet = mse(pred_inlet[:, 0:2], inlet_target[:, 0:2])
    loss_wall = mse(pred_wall[:, 0:2], torch.zeros_like(pred_wall[:, 0:2]))
    loss_outlet = mse(pred_outlet[:, 2:3], torch.zeros_like(pred_outlet[:, 2:3]))
    loss_ic = mse(pred_ic, torch.zeros_like(pred_ic))

    loss = loss_pde + loss_inlet + loss_wall + loss_outlet + loss_ic
    loss.backward()
    optimizer.step()

    if (epoch + 1) % LOG_INTERVAL == 0 or epoch == adam_epochs - 1:
        print(f"  Epoch {epoch+1}: loss={loss.item():.6f} pde={loss_pde.item():.6f}")

adam_time = time.perf_counter() - start
print(f"[DT-PINN+DeepXDE-LBFGS] Adam done in {adam_time:.1f}s ({adam_time/60:.1f} min)")

# =============================================================================
# L-BFGS phase with DeepXDE parameters
# =============================================================================
print(f"\n[DT-PINN+DeepXDE-LBFGS] L-BFGS phase (DeepXDE config)")
print("  max_iter=1000, max_eval=1250, tolerance_grad=1e-8")
print("  tolerance_change=0, history_size=100")

lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=1000,       # DeepXDE: 1000 (ours was 20)
    max_eval=1250,       # DeepXDE: 1250 (ours was 25)
    tolerance_grad=1e-8, # DeepXDE: 1e-8 (ours was 1e-7)
    tolerance_change=0,  # DeepXDE: 0 (ours was 1e-9) — NEVER stop due to ftol!
    history_size=100,    # DeepXDE: 100 (ours was 50)
    line_search_fn='strong_wolfe',
)

lbfgs_state = {'iter': 0, 'loss': float('inf'), 'plateau': 0}

def closure():
    lbfgs.zero_grad()
    pred_batch = model(g['xyt_batched'])
    pred_all = pred_batch[:N_all]
    c, mu, mv = compute_pde_ns_3d(pred_all, g)
    loss_pde = (c[ii] ** 2).mean() + (mu[ii] ** 2).mean() + (mv[ii] ** 2).mean()

    pred_inlet = pred_batch[g['off_inlet']:g['off_wall']]
    pred_wall = pred_batch[g['off_wall']:g['off_outlet']]
    pred_outlet = pred_batch[g['off_outlet']:g['off_ic']]
    pred_ic = pred_batch[g['off_ic']:]

    inlet_target = torch.zeros_like(pred_inlet)
    inlet_target[:, 0] = V0
    loss_inlet = mse(pred_inlet[:, 0:2], inlet_target[:, 0:2])
    loss_wall = mse(pred_wall[:, 0:2], torch.zeros_like(pred_wall[:, 0:2]))
    loss_outlet = mse(pred_outlet[:, 2:3], torch.zeros_like(pred_outlet[:, 2:3]))
    loss_ic = mse(pred_ic, torch.zeros_like(pred_ic))

    loss = loss_pde + loss_inlet + loss_wall + loss_outlet + loss_ic
    loss.backward()

    lbfgs_state['iter'] += 1
    if lbfgs_state['iter'] % 10 == 0:
        print(f"  L-BFGS iter {lbfgs_state['iter']}: loss={loss.item():.6f}")
    return loss

lbfgs_start = time.perf_counter()

for step in range(lbfgs_steps):
    loss_t = lbfgs.step(closure)
    cur_loss = loss_t.item() if loss_t is not None else lbfgs_state['loss']
    if abs(lbfgs_state['loss'] - cur_loss) < 1e-10 * max(1.0, abs(cur_loss)):
        lbfgs_state['plateau'] += 1
    else:
        lbfgs_state['plateau'] = 0
    lbfgs_state['loss'] = cur_loss
    if lbfgs_state['plateau'] >= 50:
        print(f"  L-BFGS converged (plateau) at outer step {step+1}")
        break
    if (step + 1) % 50 == 0:
        print(f"  Outer step {step+1}: loss={cur_loss:.6f}")

lbfgs_time = time.perf_counter() - lbfgs_start

if device.type == 'cuda':
    torch.cuda.synchronize()
total_time = time.perf_counter() - start

peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0.0

# =============================================================================
# Evaluate
# =============================================================================
print(f"\n{'=' * 70}")
print("EVALUATION")
print("=" * 70)

# Training loss (spectral)
with torch.no_grad():
    losses = compute_losses(model(g['xyt_batched']), g)
print(f"\nFinal training loss (spectral): {losses['total']:.6e}")
print(f"  PDE: {losses['pde']:.6e}  BC: {losses['bc']:.6e}")

# Eval PDE RMS (autograd, 161x81x20 grid)
print("\nEvaluating on 161x81x20 uniform grid (autograd)...")
eval_results = evaluate_ns(model, device)
print(f"  PDE RMS: {eval_results['pde_rms']:.4f}")
print(f"  Continuity RMS: {eval_results['continuity_rms']:.4f}")
print(f"  Momentum RMS: {eval_results['momentum_rms']:.4f}")

# Save model
save_path = 'results/sage_partner/model_ns_dtpinn_deepxde_lbfgs.pt'
torch.save(model.state_dict(), save_path)
print(f"\nModel saved to {save_path}")

# =============================================================================
# Comparison
# =============================================================================
print(f"\n{'=' * 70}")
print("COMPARISON")
print("=" * 70)

print(f"\n{'Method':<30s} {'Time (min)':<12s} {'Train Loss':<12s} {'Eval PDE RMS':<12s}")
print("-" * 66)
print(f"{'DT-PINN (original L-BFGS)':<30s} {'19.1':<12s} {'2.04e-2':<12s} {'18.05':<12s}")
print(f"{'DT-PINN (DeepXDE L-BFGS)':<30s} {total_time/60:<12.1f} {losses['total']:<12.6e} {eval_results['pde_rms']:<12.4f}")
print(f"{'DeepXDE':<30s} {'178.8':<12s} {'1.80e-3':<12s} {'0.55':<12s}")

print(f"\n  Adam time: {adam_time/60:.1f} min")
print(f"  L-BFGS time: {lbfgs_time/60:.1f} min")
print(f"  Total time: {total_time/60:.1f} min")
print(f"  Peak GPU memory: {peak_mem:.2f} GB")
print(f"  L-BFGS outer steps: {step+1}")
print(f"  L-BFGS inner iters: {lbfgs_state['iter']}")

improvement = 18.05 / max(eval_results['pde_rms'], 1e-10)
print(f"\n  Improvement over original DT-PINN: {improvement:.2f}x")
if improvement > 2:
    print("  L-BFGS config IS a significant secondary contributor.")
else:
    print("  L-BFGS config is NOT a significant contributor (or the derivative")
    print("  mismatch dominates regardless of optimizer quality).")

print("\nDone.")
