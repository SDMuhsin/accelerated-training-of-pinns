"""
Diagnostic for Priority 3 anomaly: SK-PINN catastrophic on elasticity cells.

Hypothesis:
  SK-PINN's chained sparse RKPM second-derivative operator
  (`Dx @ (Dx @ u)` etc., implemented as `torch.sparse.mm` chains in
  `compute_pde_elasticity_sparse`) is far less accurate than autograd's
  exact second derivative of the network. The model can drive the chained
  RKPM residual to ~1.6e-3 by adopting high-frequency content that the
  chained sparse operator smooths to near-zero, while autograd captures
  the actual derivatives — leading to eval-grid pde_rms ~0.13.

What this script does:
  1. Trains SK-PINN elasticity (PirateNet, seed 42, 30k epochs) — same
     code path as paper.
  2. After training, on the SAME training-grid points (xy_all):
     R_train_RKPM = chained-sparse-matmul residual (what trainer minimised)
     R_train_AD   = autograd second-derivative residual (network's true derivatives)
  3. On the EVAL grid (51x51 uniform via autograd):
     R_eval_AD    = standard evaluate_elasticity output

  Reports the three RMS values plus per-component breakdown.

Decision rule:
  R_train_AD ≫ R_train_RKPM  → confirms hypothesis (chained-sparse-matmul inaccuracy)
  R_train_AD ≈ R_train_RKPM  → hypothesis falsified; issue is off-grid extrapolation
"""
import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.lid_benchmark import (
    train_sk_pinn_elasticity,
    build_sk_data_elasticity,
    compute_pde_elasticity_sparse,
    evaluate_elasticity,
    elasticity_body_forces,
    pde_residuals_elasticity_autodiff,
    lam_e, mu_e,
)


def autograd_residual_at_points(model, xy_pts):
    """Compute elasticity residual at given points via autograd. Returns (eq_x, eq_y) tensors."""
    xy = xy_pts.detach().clone().requires_grad_(True)
    eq_x, eq_y = pde_residuals_elasticity_autodiff(model, xy)
    return eq_x.detach(), eq_y.detach()


def rms(t):
    return float(torch.sqrt((t ** 2).mean()).item())


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")
    print(f"Args: model={args.model}, seed={args.seed}, n_epochs={args.epochs}, "
          f"grid_size={args.grid_size}")

    # =========================================================================
    # 1. Train SK-PINN elasticity (same code path as paper benchmark)
    # =========================================================================
    print("\n[1/4] Building RKPM grid...")
    g = build_sk_data_elasticity(args.grid_size, device)

    print(f"\n[2/4] Training SK-PINN elasticity ({args.model}, seed={args.seed}, "
          f"epochs={args.epochs})...")
    t0 = time.perf_counter()
    model, train_time, final_loss = train_sk_pinn_elasticity(
        seed=args.seed, device=device, n_epochs=args.epochs, lr=args.lr,
        grid_size=args.grid_size, model_name=args.model, grid_data=g,
        tracker=None,
    )
    t1 = time.perf_counter()
    print(f"  Trained in {train_time:.1f}s ({train_time/60:.2f} min). "
          f"final_loss = {final_loss:.6e}")

    model.eval()

    # =========================================================================
    # 2. Train-grid: compute RKPM residual (what trainer minimised)
    # =========================================================================
    print("\n[3/4] Computing residuals...")

    with torch.no_grad():
        pred_all = model(g['xy_all'])
        eq_x_rkpm, eq_y_rkpm = compute_pde_elasticity_sparse(pred_all, g)
        ii = g['interior_idx']
        rkpm_eq_x_rms = rms(eq_x_rkpm[ii])
        rkpm_eq_y_rms = rms(eq_y_rkpm[ii])
        rkpm_total_rms = float(torch.sqrt(
            ((eq_x_rkpm[ii] ** 2).mean() + (eq_y_rkpm[ii] ** 2).mean())
        ).item())

    print(f"  R_train_RKPM (chained sparse matmul, interior only):")
    print(f"    eq_x rms = {rkpm_eq_x_rms:.6e}")
    print(f"    eq_y rms = {rkpm_eq_y_rms:.6e}")
    print(f"    total rms = {rkpm_total_rms:.6e}")

    # 3. Train-grid: compute AUTOGRAD residual (model's true derivatives at same pts)
    xy_int_pts = g['xy_all'][g['interior_idx']]
    eq_x_ad, eq_y_ad = autograd_residual_at_points(model, xy_int_pts)
    ad_eq_x_rms = rms(eq_x_ad)
    ad_eq_y_rms = rms(eq_y_ad)
    ad_total_rms = float(torch.sqrt(
        ((eq_x_ad ** 2).mean() + (eq_y_ad ** 2).mean())
    ).item())

    print(f"  R_train_AD (autograd residual at SAME interior training points):")
    print(f"    eq_x rms = {ad_eq_x_rms:.6e}")
    print(f"    eq_y rms = {ad_eq_y_rms:.6e}")
    print(f"    total rms = {ad_total_rms:.6e}")

    # 4. Eval-grid: standard evaluate_elasticity (autograd at 51x51 uniform)
    eval_metrics = evaluate_elasticity(model, device)
    print(f"  R_eval_AD (standard 51x51 uniform autograd):")
    print(f"    eq_x rms = {eval_metrics['continuity_rms']:.6e}")
    print(f"    eq_y rms = {eval_metrics['momentum_rms']:.6e}")
    print(f"    pde_rms  = {eval_metrics['pde_rms']:.6e}")

    # =========================================================================
    # 5. Diagnosis
    # =========================================================================
    print("\n[4/4] Diagnosis:")
    print(f"  Train RKPM rms       = {rkpm_total_rms:.6e}")
    print(f"  Train AD rms         = {ad_total_rms:.6e}")
    print(f"  Eval AD pde_rms      = {eval_metrics['pde_rms']:.6e}")
    print(f"  AD/RKPM ratio (train) = {ad_total_rms / max(rkpm_total_rms, 1e-30):.2f}x")
    print(f"  Eval/Train AD ratio   = {eval_metrics['pde_rms'] / max(ad_total_rms, 1e-30):.2f}x")

    if ad_total_rms / max(rkpm_total_rms, 1e-30) > 5:
        print("\n  CONCLUSION: AD/RKPM ratio at training points >> 1.")
        print("  → Confirms chained-sparse-matmul operator inaccuracy hypothesis.")
        print("  → The model satisfies the chained sparse residual but its actual")
        print("    autograd derivatives at the same points are much larger.")
    else:
        print("\n  CONCLUSION: AD/RKPM ratio at training points is moderate.")
        print("  → Operator inaccuracy not the dominant cause.")
        print("  → Issue is likely off-grid extrapolation / model overfitting.")

    # Save results
    out_dir = 'results/p3_sk_pinn_diagnostic'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"sk_pinn_elas_{args.model}_s{args.seed}_e{args.epochs}.json",
    )
    payload = {
        'model': args.model,
        'seed': args.seed,
        'epochs': args.epochs,
        'grid_size': args.grid_size,
        'train_time_s': train_time,
        'final_loss': final_loss,
        'train_rkpm_total_rms': rkpm_total_rms,
        'train_rkpm_eq_x_rms': rkpm_eq_x_rms,
        'train_rkpm_eq_y_rms': rkpm_eq_y_rms,
        'train_ad_total_rms': ad_total_rms,
        'train_ad_eq_x_rms': ad_eq_x_rms,
        'train_ad_eq_y_rms': ad_eq_y_rms,
        'eval_pde_rms': eval_metrics['pde_rms'],
        'eval_eq_x_rms': eval_metrics['continuity_rms'],
        'eval_eq_y_rms': eval_metrics['momentum_rms'],
        'eval_u_rms_error': eval_metrics['u_rms_error'],
        'eval_v_rms_error': eval_metrics['v_rms_error'],
        'ad_to_rkpm_ratio_train': ad_total_rms / max(rkpm_total_rms, 1e-30),
        'eval_to_train_ad_ratio': eval_metrics['pde_rms'] / max(ad_total_rms, 1e-30),
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='pirate-net',
                        choices=['mlp', 'tsa-pinn', 'pirate-net'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30000)
    parser.add_argument('--grid-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
