#!/usr/bin/env python3
"""CAN-PINN paper-faithful validator (cavity Re=400 plain NS).

Runs the Chiu et al. 2022 (CMAME 395, 114909) cavity Re=400 demo as faithfully
as possible in PyTorch. This is INDEPENDENT of src/lid_benchmark.py's harness:
the spec asks for a separate validator script so the paper-faithful gate is
not entangled with our Re=1000 NS+Smagorinsky benchmark.

Configuration (from paper Table 1 / upstream notebook cells 21-23):
  - Domain         : x, y in [0, 1]  (unit square)
  - Re             : 400
  - Grid           : 51 x 51 = 2,601 collocation points (uniform; dx = dy = 0.02)
  - Network        : SIREN-style sinusoidal-features MLP with 3 separate heads
                     (u, v, p). 32 sinusoidal features (input -> 64 channels via
                     sin(2 pi (W x + b))), 3 shared sin-act layers of width 20,
                     3 sin-act towers of width 20 per output, final linear no-bias.
  - PDE residual   : can(uw2-conv, cd-p) -- the same scheme implemented for the
                     harness drop-in, but with Smagorinsky disabled.
  - BC             : top u=1 v=0; left/right/bottom u=0 v=0 (corners go to top).
  - Loss           : L = (L_continuity + L_mom_u + L_mom_v) / lambda + L_BC
                     with lambda = 1.0 (paper Table 1 footnote 3).
  - Optimizer      : Adam(lr=1e-3) + ReduceLROnPlateau(factor=0.5, patience=50,
                     min_lr=5e-6, monitor=loss).
  - Iterations     : 200,000 (paper). For Phase 2 smoke, --epochs sets iterations.
  - Mini-batch     : 475 PDE + 25 BC = 500. We sample uniformly without replacement
                     from the 2401 interior + 200 boundary points each iteration
                     (a stochastic mini-batch over the deterministic grid).

After training we report:
  - u-MSE / v-MSE / p-MSE (over the full 51x51 grid) vs. the IDFC ground-truth
    CSV at temp/can-pinn-upstream/d00_data/RE400_LDC_GROUND_TRUTH_51X51.csv.
  - Final training loss + PDE residual.

Phase 2 only requires this script to run without NaN at --epochs 50. Phase 4
will run the full 200k-iteration sweep against the paper's gates.

Usage:
    source env/bin/activate
    python scripts/can_pinn_paper_validation.py --epochs 50          # smoke
    python scripts/can_pinn_paper_validation.py --epochs 200000      # paper-faithful

NOTE on independent reimplementation: this script does NOT import or copy any
upstream CAN-PINN code (license: All Rights Reserved, IHPC/A*STAR). The
algorithm is reconstructed from the published paper and the Phase-1 spec at
llmdocs/trackers/can_pinn_replication_2026-04-29.md.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the can-PINN stencil from lid_benchmark — the algebra is identical
# whether the model is the harness's plain MLP or our SIREN-3-head net.
from src.lid_benchmark import pde_residuals_canpinn_cavity


GROUND_TRUTH_CSV = (
    "temp/can-pinn-upstream/d00_data/RE400_LDC_GROUND_TRUTH_51X51.csv"
)


# Columns written to --out-csv. All numeric fields are formatted with .6e where
# applicable; integer fields plain.
OUT_CSV_COLUMNS = [
    'mode', 'seed', 'epochs', 'wall_clock_s', 'final_loss',
    'u_mse', 'v_mse', 'p_mse', 'iter_per_sec', 'gpu_name', 'git_sha',
]


def _git_sha() -> str:
    """Return short git SHA of the current repo HEAD, or 'unknown'."""
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return out.decode().strip()
    except Exception:
        return 'unknown'


# --- Network architecture: SIREN-3-head sinusoidal-features MLP ----------
class SirenSinusoidalFeatures(nn.Module):
    """Sinusoidal random-Fourier-features projection (paper eq. 3, ff='SIREN').

    input -> Dense(2 * n_ffs, no-bias) -> sin(2 pi *) -> output.
    sigma controls the truncated-normal init stddev for the projection
    weights. The notebook uses TruncatedNormal(stddev=sigma) with
    sigma = 1.0; we approximate with N(0, sigma^2) and clip to +/- 2 sigma.
    """

    def __init__(self, n_ffs: int = 32, sigma: float = 1.0):
        super().__init__()
        # In the upstream notebook, ff='SIREN' uses a Dense(n_ffs*2, linear)
        # followed by sin(2 pi *). Total output dim = 2 * n_ffs = 64.
        self.proj = nn.Linear(2, 2 * n_ffs, bias=True)
        with torch.no_grad():
            torch.nn.init.trunc_normal_(self.proj.weight, std=sigma, a=-2*sigma, b=2*sigma)
            torch.nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(2 * math.pi * self.proj(x))


class SinDense(nn.Module):
    """Linear layer followed by sin activation (notebook ff='SIREN' path)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.lin(x))


class CanPinnSirenMLP(nn.Module):
    """SIREN sin-activation MLP with 3 shared layers + 3 separate towers (u, v, p)."""

    def __init__(self, n_ffs: int = 32, n_nodes: int = 20, sigma: float = 1.0):
        super().__init__()
        self.features = SirenSinusoidalFeatures(n_ffs=n_ffs, sigma=sigma)
        ff_dim = 2 * n_ffs
        # Shared trunk: 3 sin-act layers of width n_nodes.
        self.shared = nn.Sequential(
            SinDense(ff_dim, n_nodes),
            SinDense(n_nodes, n_nodes),
            SinDense(n_nodes, n_nodes),
        )
        # Three towers, one per output (u, v, p). Each is 3 sin-act layers
        # then a single linear no-bias readout.
        def head() -> nn.Sequential:
            return nn.Sequential(
                SinDense(n_nodes, n_nodes),
                SinDense(n_nodes, n_nodes),
                SinDense(n_nodes, n_nodes),
                nn.Linear(n_nodes, 1, bias=False),
            )
        self.head_u = head()
        self.head_v = head()
        self.head_p = head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(self.features(x))
        u = self.head_u(h)
        v = self.head_v(h)
        p = self.head_p(h)
        return torch.cat([u, v, p], dim=1)


# --- Domain helpers ------------------------------------------------------
def build_grid(N: int, device: torch.device, dtype: torch.dtype):
    """Uniform N x N grid on [0, 1]^2; returns interior/lid/wall index arrays.

    Same mask logic as upstream: top edge -> lid (u=1, v=0); left/right/bottom
    -> wall (u=0, v=0). Top corners go to lid (consistent with the upstream
    `_left & ~_top`, `_right & ~_top` notation).
    """
    x_lin = np.linspace(0.0, 1.0, N)
    xx, yy = np.meshgrid(x_lin, x_lin, indexing='xy')
    xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)

    eps = 1e-10
    xc, yc = xy[:, 0], xy[:, 1]
    is_boundary = (xc < eps) | (xc > 1 - eps) | (yc < eps) | (yc > 1 - eps)
    is_top = (yc > 1 - eps)
    is_wall_only = is_boundary & ~is_top

    interior_idx = np.where(~is_boundary)[0]
    top_idx = np.where(is_top)[0]
    wall_idx = np.where(is_wall_only)[0]

    xy_t = torch.tensor(xy, dtype=dtype, device=device)
    return {
        'xy_all': xy_t,
        'xy_int': xy_t[interior_idx].contiguous(),
        'xy_top': xy_t[top_idx].contiguous(),
        'xy_wall': xy_t[wall_idx].contiguous(),
        'interior_idx': interior_idx,
        'top_idx': top_idx,
        'wall_idx': wall_idx,
    }


def load_ground_truth(device: torch.device, dtype: torch.dtype) -> dict:
    """Load the IDFC 51x51 ground-truth CSV. Required at --epochs > 0."""
    if not os.path.exists(GROUND_TRUTH_CSV):
        raise FileNotFoundError(
            f"Ground truth CSV not found at {GROUND_TRUTH_CSV} -- the upstream "
            f"clone at temp/can-pinn-upstream/ must be present.")
    data = np.loadtxt(GROUND_TRUTH_CSV, delimiter=',', skiprows=1)
    # Columns: x, y, u, v, p.
    xy = data[:, 0:2]
    u_gt = data[:, 2:3]
    v_gt = data[:, 3:4]
    p_gt = data[:, 4:5]
    return {
        'xy': torch.tensor(xy, dtype=dtype, device=device),
        'u': torch.tensor(u_gt, dtype=dtype, device=device),
        'v': torch.tensor(v_gt, dtype=dtype, device=device),
        'p': torch.tensor(p_gt, dtype=dtype, device=device),
    }


# --- PDE residuals (mode-aware) ------------------------------------------
def pde_residuals_autodiff_plain_ns(model, xy_int, nu):
    """Plain incompressible NS residuals via pure autograd (no Smagorinsky).

    Used by --mode=autodiff. xy_int must already have requires_grad=True.
    Mirrors scripts/can_pinn_stencil_sanity.py's helper (already validated to
    agree with the can-PDE stencil to within fp64 round-off in the dx -> 0
    limit). All derivatives come from torch.autograd; the only difference vs.
    the can-PINN path is the residual computation. Network, optimizer, batch,
    grid, and BC loss are identical.
    """
    pred = model(xy_int)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    grad_u = torch.autograd.grad(
        u, xy_int, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True)[0]
    grad_v = torch.autograd.grad(
        v, xy_int, grad_outputs=torch.ones_like(v),
        create_graph=True, retain_graph=True)[0]
    grad_p = torch.autograd.grad(
        p, xy_int, grad_outputs=torch.ones_like(p),
        create_graph=True, retain_graph=True)[0]

    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]
    dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]

    # 2nd-order AD for the Laplacian.
    d2u_dx2 = torch.autograd.grad(
        du_dx, xy_int, grad_outputs=torch.ones_like(du_dx),
        create_graph=True, retain_graph=True)[0][:, 0:1]
    d2u_dy2 = torch.autograd.grad(
        du_dy, xy_int, grad_outputs=torch.ones_like(du_dy),
        create_graph=True, retain_graph=True)[0][:, 1:2]
    d2v_dx2 = torch.autograd.grad(
        dv_dx, xy_int, grad_outputs=torch.ones_like(dv_dx),
        create_graph=True, retain_graph=True)[0][:, 0:1]
    d2v_dy2 = torch.autograd.grad(
        dv_dy, xy_int, grad_outputs=torch.ones_like(dv_dy),
        create_graph=True, retain_graph=True)[0][:, 1:2]

    continuity = du_dx + dv_dy
    mom_u = u * du_dx + v * du_dy + dp_dx - nu * (d2u_dx2 + d2u_dy2)
    mom_v = u * dv_dx + v * dv_dy + dp_dy - nu * (d2v_dx2 + d2v_dy2)
    return continuity, mom_u, mom_v


# --- Loss assembly -------------------------------------------------------
def compute_total_loss(model, grid, dx, dy, lam, nu_lam, mode,
                       int_batch_idx=None, top_batch_idx=None,
                       wall_batch_idx=None):
    """Total loss (PDE + BC). Optional mini-batch indices; if None, full grid.

    `mode` selects the PDE residual:
      - 'can-pinn'  : 9-point can(uw2-conv, cd-p) FD stencil (Smagorinsky off).
      - 'autodiff'  : pure torch.autograd derivatives, plain NS, same network.

    Both modes share network, BC, mini-batch sampling, and loss assembly. The
    only thing that changes is the residual computation — apples-to-apples.
    """
    xy_int = grid['xy_int'] if int_batch_idx is None \
        else grid['xy_int'][int_batch_idx]
    xy_top = grid['xy_top'] if top_batch_idx is None \
        else grid['xy_top'][top_batch_idx]
    xy_wall = grid['xy_wall'] if wall_batch_idx is None \
        else grid['xy_wall'][wall_batch_idx]

    if mode == 'can-pinn':
        # PDE residual via can(uw2-conv, cd-p), Smagorinsky off.
        R_c, R_mu, R_mv = pde_residuals_canpinn_cavity(
            model, xy_int, dx, dy,
            Cs_d_sq_int=None, nu_lam=nu_lam,
            use_smagorinsky=False)
    elif mode == 'autodiff':
        # Pure-AD plain-NS residual on the same interior batch. We need a leaf
        # tensor with requires_grad=True for autograd; clone-detach-leaf each
        # iter so the graph is freshly built (matches lid_benchmark's pattern).
        xy_int_ad = xy_int.detach().clone().requires_grad_(True)
        R_c, R_mu, R_mv = pde_residuals_autodiff_plain_ns(
            model, xy_int_ad, nu_lam)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    L_c = (R_c ** 2).mean()
    L_mu = (R_mu ** 2).mean()
    L_mv = (R_mv ** 2).mean()
    L_pde = (L_c + L_mu + L_mv) / lam

    # BC loss (top: u=1, v=0; wall: u=0, v=0).
    pred_top = model(xy_top)
    L_top = ((pred_top[:, 0:1] - 1.0) ** 2).mean() + (pred_top[:, 1:2] ** 2).mean()

    pred_wall = model(xy_wall)
    L_wall = (pred_wall[:, 0:1] ** 2).mean() + (pred_wall[:, 1:2] ** 2).mean()

    L_bc = L_top + L_wall
    return L_pde + L_bc, L_pde, L_bc, (L_c.detach(), L_mu.detach(), L_mv.detach())


# --- Evaluation ----------------------------------------------------------
def evaluate(model, gt) -> dict:
    """Compute u/v/p MSE on the full 2,601-point grid vs. ground truth.

    Pressure is gauge-shifted: subtract the mean-difference (pred_p - gt_p)
    before computing p-MSE so an arbitrary global offset doesn't dominate.
    """
    model.eval()
    with torch.no_grad():
        pred = model(gt['xy'])
    u_pred = pred[:, 0:1]
    v_pred = pred[:, 1:2]
    p_pred = pred[:, 2:3]

    u_mse = ((u_pred - gt['u']) ** 2).mean().item()
    v_mse = ((v_pred - gt['v']) ** 2).mean().item()
    # Mean-shift pressure to remove the gauge ambiguity.
    p_offset = (p_pred - gt['p']).mean().item()
    p_mse = ((p_pred - gt['p'] - p_offset) ** 2).mean().item()
    uv_mse = 0.5 * (u_mse + v_mse)
    model.train()
    return {
        'u_mse': u_mse, 'v_mse': v_mse, 'p_mse': p_mse,
        'uv_mse': uv_mse, 'p_offset': p_offset,
    }


# --- Main ----------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="can-pinn",
                        choices=["can-pinn", "autodiff"],
                        help="PDE residual scheme. 'can-pinn' uses the 9-point "
                             "FD stencil with AD coupling (paper); 'autodiff' "
                             "uses pure torch.autograd on the same network and "
                             "same training protocol (apples-to-apples baseline).")
    parser.add_argument("--epochs", type=int, default=200000,
                        help="Number of optimizer iterations (paper: 200000).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-grid", type=int, default=51,
                        help="N x N uniform grid (paper: 51).")
    parser.add_argument("--n-pde", type=int, default=475,
                        help="Mini-batch size for PDE points (paper: 475).")
    parser.add_argument("--n-bc", type=int, default=25,
                        help="Mini-batch size for BC points (paper: 25).")
    parser.add_argument("--lam", type=float, default=1.0,
                        help="PDE/BC weight lambda (paper: 1.0).")
    parser.add_argument("--re", type=float, default=400.0)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Compute u/v/p MSE vs ground truth every N iterations and "
                             "include in the log line. 0 disables periodic evaluation. "
                             "Adds ~1 forward pass over 2601 points per evaluation.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-batching", action="store_true",
                        help="Use full grid every iteration (skip mini-batching). "
                             "Useful for very small --epochs / debugging.")
    parser.add_argument("--out-csv", default=None,
                        help="Path to append a single result row to. If the "
                             "file does not exist, a header is written.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float32

    print("=" * 70)
    print(f"CAN-PINN PAPER-FAITHFUL VALIDATOR (cavity Re={args.re:g} plain NS)")
    print("=" * 70)
    print(f"Mode:      {args.mode}")
    print(f"Device:    {device}")
    if device.type == 'cuda':
        print(f"GPU:       {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:   {torch.__version__}")
    print(f"Seed:      {args.seed}")
    print(f"N grid:    {args.n_grid} x {args.n_grid}")
    print(f"Epochs:    {args.epochs}")
    print(f"LR:        {args.lr}")
    print(f"Batch:     {args.n_pde} PDE + {args.n_bc} BC "
          f"({'no batching' if args.no_batching else 'random'})")
    print(f"Lambda:    {args.lam}")
    print(f"Git SHA:   {_git_sha()}")
    print("=" * 70)

    grid = build_grid(args.n_grid, device, dtype)
    dx = 1.0 / (args.n_grid - 1)
    dy = dx
    nu_lam = float(1.0 / args.re)

    print(f"dx = dy = {dx:.6e}")
    print(f"N_int = {grid['xy_int'].shape[0]}, "
          f"N_top = {grid['xy_top'].shape[0]}, "
          f"N_wall = {grid['xy_wall'].shape[0]}")

    # Ground-truth load up-front so we fail fast if missing.
    gt = load_ground_truth(device, dtype)
    print(f"Loaded ground truth: {gt['xy'].shape[0]} points "
          f"(from {GROUND_TRUTH_CSV})")

    # Network.
    model = CanPinnSirenMLP(n_ffs=32, n_nodes=20, sigma=1.0).to(device).to(dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model:     CanPinnSirenMLP(n_ffs=32, n_nodes=20) -- {n_params} params")

    # Optimizer + scheduler.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Match Keras 2.x ReduceLROnPlateau semantics from the upstream notebook:
    # scheduler.step is invoked once per "Keras epoch" of 100 mini-batch iterations
    # on the mean loss over that window; patience=50 epochs => 5000 mini-batch iter
    # of plateau before halving. threshold=1e-4 rel matches Keras default.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=5e-6,
        threshold=1e-4, threshold_mode='rel')

    # Boundary index pool (top + wall combined; sample n_bc each iter).
    n_top = grid['xy_top'].shape[0]
    n_wall = grid['xy_wall'].shape[0]
    n_int = grid['xy_int'].shape[0]

    rng = np.random.default_rng(args.seed)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = float('nan')
    loss_window = []  # Keras-epoch averaging window for ReduceLROnPlateau

    for epoch in range(args.epochs):
        optimizer.zero_grad()

        if args.no_batching:
            int_idx = top_idx = wall_idx = None
        else:
            int_idx = torch.tensor(
                rng.choice(n_int, size=min(args.n_pde, n_int), replace=False),
                dtype=torch.long, device=device)
            # Split BC budget between top and wall by their grid-fraction.
            n_top_batch = max(1, int(round(args.n_bc * n_top / (n_top + n_wall))))
            n_wall_batch = max(1, args.n_bc - n_top_batch)
            top_idx = torch.tensor(
                rng.choice(n_top, size=min(n_top_batch, n_top), replace=False),
                dtype=torch.long, device=device)
            wall_idx = torch.tensor(
                rng.choice(n_wall, size=min(n_wall_batch, n_wall), replace=False),
                dtype=torch.long, device=device)

        loss, L_pde, L_bc, _ = compute_total_loss(
            model, grid, dx, dy, args.lam, nu_lam, args.mode,
            int_batch_idx=int_idx, top_batch_idx=top_idx, wall_batch_idx=wall_idx)

        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        loss_window.append(final_loss)
        if (epoch + 1) % 100 == 0:
            scheduler.step(sum(loss_window) / len(loss_window))
            loss_window.clear()

        if (epoch + 1) % args.log_every == 0 or epoch == 0 or epoch == args.epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            extra = ""
            if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
                m = evaluate(model, gt)
                extra = (f" u-MSE={m['u_mse']:.3e} v-MSE={m['v_mse']:.3e} "
                         f"uv-MSE={m['uv_mse']:.3e} p-MSE={m['p_mse']:.3e}")
            print(f"  Epoch {epoch+1:>7d}: loss={final_loss:.6e} "
                  f"L_pde={L_pde.item():.6e} L_bc={L_bc.item():.6e} "
                  f"lr={current_lr:.2e}{extra}")

        if math.isnan(final_loss):
            print(f"FAIL: NaN loss at epoch {epoch+1}.")
            return 1

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start

    print(f"\nTraining done in {train_time:.2f}s ({train_time/60:.2f} min).")
    print(f"Final loss: {final_loss:.6e}")

    metrics = evaluate(model, gt)
    print()
    print("=" * 70)
    print(f"VALIDATION vs IDFC ground truth (51x51, Re=400) -- mode={args.mode}")
    print("=" * 70)
    print(f"u-MSE:     {metrics['u_mse']:.6e}")
    print(f"v-MSE:     {metrics['v_mse']:.6e}")
    print(f"u&v MSE:   {metrics['uv_mse']:.6e}   "
          f"(paper Fig 10a median ~3e-5 for can(uw2,cd))")
    print(f"p-MSE:     {metrics['p_mse']:.6e}   "
          f"(after subtracting gauge offset {metrics['p_offset']:.6e})")
    print("=" * 70)

    iter_per_sec = float(args.epochs) / max(train_time, 1e-9)
    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'

    if args.out_csv is not None:
        os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
        write_header = not os.path.exists(args.out_csv)
        with open(args.out_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=OUT_CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                'mode': args.mode,
                'seed': args.seed,
                'epochs': args.epochs,
                'wall_clock_s': f"{train_time:.6f}",
                'final_loss': f"{final_loss:.6e}",
                'u_mse': f"{metrics['u_mse']:.6e}",
                'v_mse': f"{metrics['v_mse']:.6e}",
                'p_mse': f"{metrics['p_mse']:.6e}",
                'iter_per_sec': f"{iter_per_sec:.4f}",
                'gpu_name': gpu_name,
                'git_sha': _git_sha(),
            })
        print(f"Result row appended to {args.out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
