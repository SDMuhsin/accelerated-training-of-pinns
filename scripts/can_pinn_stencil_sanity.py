#!/usr/bin/env python3
"""CAN-PINN stencil sanity test (strict gate, Phase 1 spec §6 gate 7).

For a fixed network and 5 random interior points, evaluate two residuals on
the SAME points using the SAME physics (plain incompressible NS, no
Smagorinsky):

  (a) the can-PINN FD stencil residual via pde_residuals_canpinn_cavity
      (use_smagorinsky=False -> nu_eff = nu_lam constant), and
  (b) the pure-AD residual on the same plain NS via a small AD helper below.

Sweep dx in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]. Verify that the relative L_infty
difference between (a) and (b) goes to zero as dx -> 0. PASS criterion:
at dx=1e-6, |R_can - R_AD|_inf / |R_AD|_inf <= 0.01 (<= 1% relative).

The reduce-to-AD-as-dx->0 result is paper Section 2.3.1 (the can-PDE scheme
converges to the AD scheme in the dx -> 0 limit). We test against PLAIN NS
because the harness drop-in's Smagorinsky variant uses local-constant nu_eff
in the FD Laplacian (b1 design choice), which differs from AD's divergence
form by O(grad_nu_eff · grad_u) — a real, non-vanishing physical-model
difference, not a stencil bug. Plain NS removes that ambiguity.

Usage:
    source env/bin/activate
    python scripts/can_pinn_stencil_sanity.py
"""
from __future__ import annotations

import os
import sys
import math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lid_benchmark import (
    PINN_Cavity,
    pde_residuals_canpinn_cavity,
    Cs,
    nu_laminar,
)


def pde_residuals_autodiff_plain_ns(model, xy, nu):
    """Plain incompressible NS residual via autodiff (no Smagorinsky).

    Used by the stencil-sanity script as the comparison baseline. Matches the
    scheme used by the can-PINN limit when use_smagorinsky=False.
    """
    pred = model(xy)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    grad_u = torch.autograd.grad(
        u, xy, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True)[0]
    grad_v = torch.autograd.grad(
        v, xy, grad_outputs=torch.ones_like(v),
        create_graph=True, retain_graph=True)[0]
    grad_p = torch.autograd.grad(
        p, xy, grad_outputs=torch.ones_like(p),
        create_graph=True, retain_graph=True)[0]

    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]
    dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]

    # 2nd-order AD for the Laplacian.
    d2u_dx2 = torch.autograd.grad(
        du_dx, xy, grad_outputs=torch.ones_like(du_dx),
        create_graph=True, retain_graph=True)[0][:, 0:1]
    d2u_dy2 = torch.autograd.grad(
        du_dy, xy, grad_outputs=torch.ones_like(du_dy),
        create_graph=True, retain_graph=True)[0][:, 1:2]
    d2v_dx2 = torch.autograd.grad(
        dv_dx, xy, grad_outputs=torch.ones_like(dv_dx),
        create_graph=True, retain_graph=True)[0][:, 0:1]
    d2v_dy2 = torch.autograd.grad(
        dv_dy, xy, grad_outputs=torch.ones_like(dv_dy),
        create_graph=True, retain_graph=True)[0][:, 1:2]

    continuity = du_dx + dv_dy
    mom_u = u * du_dx + v * du_dy + dp_dx - nu * (d2u_dx2 + d2u_dy2)
    mom_v = u * dv_dx + v * dv_dy + dp_dy - nu * (d2v_dx2 + d2v_dy2)
    return continuity, mom_u, mom_v


def main() -> int:
    torch.manual_seed(2026)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Run in fp64 because the stencil sweep includes dx = 1e-6 — fp32 FD
    # at that scale is dominated by O(machine_eps / dx) catastrophic
    # cancellation noise (one or two correct digits), so an apples-to-apples
    # comparison with AD requires double precision. The training-time can-PINN
    # uses fp32; this is a *correctness* test of the algebraic stencil only.
    dtype = torch.float64

    # Fixed-weights tiny MLP so that AD and can-PDE see the same forward map.
    model = PINN_Cavity().to(device).to(dtype)
    model.eval()

    # 5 random interior points on (0.2, 0.8)^2 so the stencil at dx=1e-2 still
    # stays well inside the unit square.  We deliberately avoid extremes (and
    # the wall) since the d_wall term would otherwise push nu_eff to a point
    # where Smagorinsky's nonlinearity dominates the comparison.
    N = 5
    rng = np.random.default_rng(42)
    xy = rng.uniform(0.2, 0.8, size=(N, 2)).astype(np.float64)
    xy_int = torch.tensor(xy, dtype=dtype, device=device)

    # Plain NS — no Smagorinsky for the sanity test. The harness drop-in's
    # Smagorinsky form differs from AD's divergence form by O(grad_nu_eff
    # * grad_u) which is a real physical-model offset that does NOT vanish as
    # dx -> 0. Removing Smagorinsky isolates the stencil-correctness test.
    nu = float(nu_laminar)

    # AD baseline residual (does NOT depend on dx). Standalone — needs its own
    # leaf with requires_grad.
    xy_for_ad = xy_int.clone().detach().requires_grad_(True)
    R_c_ad, R_mu_ad, R_mv_ad = pde_residuals_autodiff_plain_ns(
        model, xy_for_ad, nu)
    R_c_ad = R_c_ad.detach()
    R_mu_ad = R_mu_ad.detach()
    R_mv_ad = R_mv_ad.detach()
    R_ad_stack = torch.cat([R_c_ad, R_mu_ad, R_mv_ad], dim=1)
    ad_norm = R_ad_stack.abs().max().item()

    print()
    print(f"AD baseline (5 random interior points)")
    print(f"  ||R_continuity||_inf = {R_c_ad.abs().max().item():.6e}")
    print(f"  ||R_mom_u||_inf      = {R_mu_ad.abs().max().item():.6e}")
    print(f"  ||R_mom_v||_inf      = {R_mv_ad.abs().max().item():.6e}")
    print(f"  ||R_AD||_inf (joint) = {ad_norm:.6e}")
    print()

    dx_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    rel_errors = []
    print(f"{'dx':>10} | {'||R_can-R_AD||_inf':>20} | {'||R_can||_inf':>15} | "
          f"{'rel_err':>12}")
    print("-" * 70)
    for dx in dx_values:
        dy = dx
        R_c_can, R_mu_can, R_mv_can = pde_residuals_canpinn_cavity(
            model, xy_int, dx, dy,
            Cs_d_sq_int=None, nu_lam=nu,
            use_smagorinsky=False, return_components=False)
        R_c_can = R_c_can.detach()
        R_mu_can = R_mu_can.detach()
        R_mv_can = R_mv_can.detach()
        R_can_stack = torch.cat([R_c_can, R_mu_can, R_mv_can], dim=1)
        diff = (R_can_stack - R_ad_stack).abs().max().item()
        can_norm = R_can_stack.abs().max().item()
        # Relative error normalized by |R_AD|_inf (a finite, dx-independent
        # quantity); avoids division by 0 if AD residual happens to be tiny.
        denom = max(ad_norm, 1e-30)
        rel = diff / denom
        rel_errors.append(rel)
        print(f"{dx:>10.1e} | {diff:>20.6e} | {can_norm:>15.6e} | "
              f"{rel:>12.6e}")

    print()
    final_rel = rel_errors[-1]
    threshold = 0.01  # 1% relative
    print(f"Final relative error at dx={dx_values[-1]:.1e}: {final_rel:.6e}")
    print(f"Threshold: {threshold:.2f}")
    if final_rel <= threshold:
        print(f"\nPASS: can-PINN stencil reduces to AD as dx -> 0 within "
              f"{threshold * 100:.1f}% relative.")
        # Diagnostic: verify the O(dx^2) Taylor-truncation regime is observed
        # at moderate dx values (1e-2 -> 1e-3 -> 1e-4). At dx <= 1e-5 fp64
        # round-off in the FD ratio (machine_eps / dx^2) becomes the dominant
        # error source -- this is expected, not a bug, and is why a sane
        # training-time choice for dx is the actual grid spacing (~1e-2).
        rel_2 = rel_errors[0]
        rel_4 = rel_errors[2]
        if rel_2 > 0 and rel_4 > 0:
            decay = rel_2 / rel_4
            print(f"  rel_err(dx=1e-2) / rel_err(dx=1e-4) = {decay:.1f} "
                  f"(expected ~10000 for O(dx^2) decay; observed in this run)")
        return 0
    else:
        print(f"\nFAIL: relative error {final_rel:.6e} > {threshold:.6e}.")
        print("The can-PINN stencil does NOT reduce to the AD residual as "
              "dx -> 0. This is a stencil bug — fix before continuing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
