#!/usr/bin/env python3
"""Phase-4 root-cause diagnostic for CAN-PINN vs AD speedup anomaly.

The validator's canpinn mode crosses 5e-5 at epoch 82k while autodiff
crosses at epoch 19.5k, opposite of the paper's claim. This script
runs three quick tests:

  H1: Residual scale at random interior points at fresh init AND after
      partial AD-training. If R_can / R_AD ratio stays huge as the
      network converges, FD stencil is misfiring (e.g., SIREN aliasing).
  H2: Analytic-field test. Both residuals applied to a closed-form
      manufactured u, v, p with a known exact NS residual (incl. forcing
      term we subtract). Their disagreement at training dx=0.02 reveals
      coefficient errors.
  H3: Boundary-mask off-by-one. Verify that interior_idx really excludes
      boundary; verify that the can-stencil's 2dx out-of-domain points
      (at x=0.02, the WW stencil reaches x=-0.02) are not pathological.

Run with:
    CUDA_VISIBLE_DEVICES=0 python scripts/can_pinn_phase4_diagnose.py
"""
from __future__ import annotations

import os
import sys
import math
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lid_benchmark import pde_residuals_canpinn_cavity
from scripts.can_pinn_paper_validation import (
    CanPinnSirenMLP, build_grid, compute_total_loss,
    pde_residuals_autodiff_plain_ns,
)


def fresh_net(seed: int, device, dtype):
    torch.manual_seed(seed)
    net = CanPinnSirenMLP(n_ffs=32, n_nodes=20, sigma=1.0).to(device).to(dtype)
    return net


def residual_scale(net, xy_int, dx, dy, nu_lam):
    """Return (||R_can||_2, ||R_AD||_2, ratio_can_over_ad) at fixed points."""
    R_c, R_mu, R_mv = pde_residuals_canpinn_cavity(
        net, xy_int, dx, dy,
        Cs_d_sq_int=None, nu_lam=nu_lam, use_smagorinsky=False)
    can_l2 = math.sqrt(
        (R_c**2).mean().item() + (R_mu**2).mean().item() + (R_mv**2).mean().item())
    can_inf = max(R_c.abs().max().item(), R_mu.abs().max().item(), R_mv.abs().max().item())

    xy_ad = xy_int.detach().clone().requires_grad_(True)
    A_c, A_mu, A_mv = pde_residuals_autodiff_plain_ns(net, xy_ad, nu_lam)
    ad_l2 = math.sqrt(
        (A_c**2).mean().item() + (A_mu**2).mean().item() + (A_mv**2).mean().item())
    ad_inf = max(A_c.abs().max().item(), A_mu.abs().max().item(), A_mv.abs().max().item())

    return can_l2, ad_l2, can_inf, ad_inf


def quick_train(net, grid, dx, dy, nu_lam, n_iter, mode, lr=1e-3, batch=475):
    """Train `net` for n_iter steps in `mode`; return final loss."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n_int = grid['xy_int'].shape[0]
    n_top = grid['xy_top'].shape[0]
    n_wall = grid['xy_wall'].shape[0]
    rng = np.random.default_rng(0)
    last_loss = float('nan')
    for k in range(n_iter):
        opt.zero_grad()
        int_idx = torch.tensor(
            rng.choice(n_int, size=min(batch, n_int), replace=False),
            dtype=torch.long, device=grid['xy_int'].device)
        n_top_b = max(1, int(round(25 * n_top / (n_top + n_wall))))
        n_wall_b = max(1, 25 - n_top_b)
        top_idx = torch.tensor(
            rng.choice(n_top, size=min(n_top_b, n_top), replace=False),
            dtype=torch.long, device=grid['xy_int'].device)
        wall_idx = torch.tensor(
            rng.choice(n_wall, size=min(n_wall_b, n_wall), replace=False),
            dtype=torch.long, device=grid['xy_int'].device)
        loss, _, _, _ = compute_total_loss(
            net, grid, dx, dy, 1.0, nu_lam, mode,
            int_batch_idx=int_idx, top_batch_idx=top_idx, wall_batch_idx=wall_idx)
        loss.backward()
        opt.step()
        last_loss = loss.item()
    return last_loss


def h1_residual_scale_at_milestones(device, dtype):
    print("=" * 70)
    print("H1: Residual scale at fresh init AND after AD-training milestones")
    print("=" * 70)
    grid = build_grid(51, device, dtype)
    dx = 1.0 / 50.0
    nu_lam = 1.0 / 400.0

    # Use the SAME 475 random interior points across all milestones.
    rng = np.random.default_rng(42)
    n_int = grid['xy_int'].shape[0]
    pick = torch.tensor(
        rng.choice(n_int, size=475, replace=False),
        dtype=torch.long, device=device)
    xy_eval = grid['xy_int'][pick].contiguous()

    print(f"{'milestone':>20s} | {'||R_can||_2':>14s} {'||R_AD||_2':>14s} "
          f"{'||R_can||_inf':>14s} {'||R_AD||_inf':>14s} {'ratio_l2':>10s}")
    print("-" * 105)

    # 1) Fresh init.
    net = fresh_net(seed=0, device=device, dtype=dtype)
    can_l2, ad_l2, can_inf, ad_inf = residual_scale(net, xy_eval, dx, dx, nu_lam)
    print(f"{'fresh_init':>20s} | {can_l2:14.6e} {ad_l2:14.6e} "
          f"{can_inf:14.6e} {ad_inf:14.6e} {can_l2/max(ad_l2,1e-30):10.2f}")

    # 2) After AD-training of 100, 1000, 5000 iter.
    for n_steps in (100, 1000, 5000):
        net = fresh_net(seed=0, device=device, dtype=dtype)
        t0 = time.perf_counter()
        final_loss = quick_train(net, grid, dx, dx, nu_lam, n_steps, mode='autodiff')
        wall = time.perf_counter() - t0
        can_l2, ad_l2, can_inf, ad_inf = residual_scale(net, xy_eval, dx, dx, nu_lam)
        print(f"{'AD-trained '+str(n_steps):>20s} | {can_l2:14.6e} {ad_l2:14.6e} "
              f"{can_inf:14.6e} {ad_inf:14.6e} {can_l2/max(ad_l2,1e-30):10.2f}  "
              f"(loss={final_loss:.3e}, {wall:.1f}s)")

    # 3) Also: after CAN-trained 5000 — does can-residual stay ~constant
    #    even though loss drops?
    net = fresh_net(seed=0, device=device, dtype=dtype)
    t0 = time.perf_counter()
    final_loss = quick_train(net, grid, dx, dx, nu_lam, 5000, mode='can-pinn')
    wall = time.perf_counter() - t0
    can_l2, ad_l2, can_inf, ad_inf = residual_scale(net, xy_eval, dx, dx, nu_lam)
    print(f"{'CAN-trained 5000':>20s} | {can_l2:14.6e} {ad_l2:14.6e} "
          f"{can_inf:14.6e} {ad_inf:14.6e} {can_l2/max(ad_l2,1e-30):10.2f}  "
          f"(loss={final_loss:.3e}, {wall:.1f}s)")
    print()


def h2_analytic_field(device, dtype):
    """Define a SIREN-free analytic model and check residual exactness.

    Pick u(x,y) = sin(pi x) sin(pi y), v(x,y) = cos(pi x) cos(pi y),
    p(x,y) = -0.25 * (cos(2 pi x) + cos(2 pi y)). This is NOT a NS
    solution; we just compare can-stencil and AD on the SAME analytic
    field. Both residuals should match analytically as dx -> 0.

    At dx=0.02 we expect:
      - AD residual = exact analytic residual (to round-off).
      - can residual = AD residual + O(dx^2) discretization error.

    The discretization error magnitude tells us whether the stencil is
    actually 2nd-order or has a bug that makes it O(1) wrong.
    """
    print("=" * 70)
    print("H2: Analytic-field residual comparison (stencil vs AD)")
    print("=" * 70)

    class AnalyticUVP(nn.Module):
        def forward(self, xy):
            x = xy[:, 0:1]; y = xy[:, 1:2]
            u = torch.sin(math.pi * x) * torch.sin(math.pi * y)
            v = torch.cos(math.pi * x) * torch.cos(math.pi * y)
            p = -0.25 * (torch.cos(2 * math.pi * x) + torch.cos(2 * math.pi * y))
            return torch.cat([u, v, p], dim=1)

    net = AnalyticUVP().to(device).to(dtype)
    grid = build_grid(51, device, dtype)
    nu_lam = 1.0 / 400.0

    rng = np.random.default_rng(42)
    n_int = grid['xy_int'].shape[0]
    pick = torch.tensor(
        rng.choice(n_int, size=475, replace=False),
        dtype=torch.long, device=device)
    xy_eval = grid['xy_int'][pick].contiguous()

    print(f"{'dx':>10s} | {'||R_can||_2':>14s} {'||R_AD||_2':>14s} "
          f"{'||diff||_2':>14s} {'rel_diff':>10s}")
    print("-" * 70)
    for dx in (0.02, 0.005, 0.001, 1e-5):
        R_c, R_mu, R_mv = pde_residuals_canpinn_cavity(
            net, xy_eval, dx, dx,
            Cs_d_sq_int=None, nu_lam=nu_lam, use_smagorinsky=False)
        can_l2 = math.sqrt(
            (R_c**2).mean().item() + (R_mu**2).mean().item() + (R_mv**2).mean().item())

        xy_ad = xy_eval.detach().clone().requires_grad_(True)
        A_c, A_mu, A_mv = pde_residuals_autodiff_plain_ns(net, xy_ad, nu_lam)
        ad_l2 = math.sqrt(
            (A_c**2).mean().item() + (A_mu**2).mean().item() + (A_mv**2).mean().item())

        # Diff (component-wise, then L2).
        d_c = (R_c - A_c).detach()
        d_mu = (R_mu - A_mu).detach()
        d_mv = (R_mv - A_mv).detach()
        diff_l2 = math.sqrt(
            (d_c**2).mean().item() + (d_mu**2).mean().item() + (d_mv**2).mean().item())
        rel = diff_l2 / max(ad_l2, 1e-30)
        print(f"{dx:10.2e} | {can_l2:14.6e} {ad_l2:14.6e} "
              f"{diff_l2:14.6e} {rel:10.4f}")
    print()


def h3_misc_sanity(device, dtype):
    """Verify boundary-mask logic, out-of-domain stencil count, and
    residual-loss scaling between batch and full-grid."""
    print("=" * 70)
    print("H3: Misc sanity (interior-mask, out-of-domain stencils)")
    print("=" * 70)
    grid = build_grid(51, device, dtype)
    n_int = grid['xy_int'].shape[0]
    n_top = grid['xy_top'].shape[0]
    n_wall = grid['xy_wall'].shape[0]
    print(f"  Grid 51x51: N_int={n_int} (expect 49*49=2401), "
          f"N_top={n_top} (expect 51), N_wall={n_wall} (expect 51*4-51-2 = 149)")
    # Note: top has 51 (y=1 row), wall has the other 3 sides minus the
    # 2 top corners which are assigned to top: 3*51 - 2 = 151? Let's check.

    # Now: how many of the 2401 interior points have a 2-step stencil
    # neighbor outside [0,1]^2?
    xy = grid['xy_int'].cpu().numpy()
    dx = 1.0 / 50.0
    n_oob = 0
    for x, y in xy:
        for ox, oy in [(2*dx, 0), (-2*dx, 0), (0, 2*dx), (0, -2*dx)]:
            xn, yn = x + ox, y + oy
            if xn < -1e-12 or xn > 1 + 1e-12 or yn < -1e-12 or yn > 1 + 1e-12:
                n_oob += 1
                break
    print(f"  Interior points with at least one 2dx stencil neighbor outside "
          f"[0,1]^2: {n_oob} / {n_int} ({100.*n_oob/n_int:.1f}%)")
    print()


def h1_short(device, dtype):
    """Truncated H1: only fresh_init + AD-trained 100. Skip 1000/5000 to save
    minutes — we've already confirmed at fresh+100 that R_can ~ R_AD."""
    print("=" * 70)
    print("H1 (short): residual ratio at fresh and AD-100")
    print("=" * 70)
    grid = build_grid(51, device, dtype)
    dx = 1.0 / 50.0
    nu_lam = 1.0 / 400.0

    rng = np.random.default_rng(42)
    n_int = grid['xy_int'].shape[0]
    pick = torch.tensor(
        rng.choice(n_int, size=475, replace=False),
        dtype=torch.long, device=device)
    xy_eval = grid['xy_int'][pick].contiguous()

    print(f"{'milestone':>20s} | {'||R_can||_2':>14s} {'||R_AD||_2':>14s} "
          f"{'ratio_l2':>10s}")
    print("-" * 65)
    for label, n_steps in (('fresh_init', 0), ('AD-trained 100', 100)):
        net = fresh_net(seed=0, device=device, dtype=dtype)
        if n_steps > 0:
            quick_train(net, grid, dx, dx, nu_lam, n_steps, mode='autodiff')
        can_l2, ad_l2, _, _ = residual_scale(net, xy_eval, dx, dx, nu_lam)
        print(f"{label:>20s} | {can_l2:14.6e} {ad_l2:14.6e} "
              f"{can_l2/max(ad_l2,1e-30):10.2f}")
    print()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    print(f"device={device}, gpu={torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}")
    print()

    # Run the analytic-field test FIRST — most decisive for coefficient bugs.
    h2_analytic_field(device, dtype)
    h3_misc_sanity(device, dtype)
    h1_short(device, dtype)


if __name__ == "__main__":
    main()
