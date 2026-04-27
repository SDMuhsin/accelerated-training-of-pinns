"""Phase-5 smoke test for C1 SK-CERT on F3 linear elasticity.

Reproduces retained B1 (F3, instance 0, seed 42, MLP, 30K Adam iterations),
then computes the SK-CERT bound B = C_lambda * R and compares against the
closed-form manufactured solution (stronger reference than any FEM).

Gate (per llmdocs/research/research_log/contract_target_pin.md):
  - tau_1 > 10  -> STOP, Automatic Phase Reset, fallback to C4 SK-PD-GAP
  - tau_1 < 1   -> STOP, bound not a valid upper bound, fallback to C4
  - 1 <= tau_1 <= 10 -> PASS smoke pre-gate; proceed to full grid

Outputs: JSON summary at results/sk_cert_f3_smoke.json and stdout lines
tagged [SMOKE].

Usage:
  source env/bin/activate
  python3 scripts/smoke_sk_cert_f3.py
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Repo root on sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.lid_benchmark import (
    build_grid_data_elasticity,
    compute_pde_elasticity,
    elasticity_exact,
    train_sage_elasticity,
)
from src.jax_pinn import _reparam_elasticity_grid_
from src.symbolic_vjp import compute_sk_cert


# Frozen landscape instance 0 (F3, joint_uniform draw with seed 20260416).
E_RATIO = 1.6776796987616658
NU_POISSON = 0.2132794290244124
SEED = 42
ARCH = "mlp"
GRID_SIZE = 50  # N_all = 2500, matches retained Phase-2 B1 config.
N_EPOCHS = 30000
LR = 1e-3
OUT_JSON = _ROOT / "results" / "sk_cert_f3_smoke.json"


def _pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_eval_grid(device):
    nx = ny = 51
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack([X.ravel(), Y.ravel()])
    return torch.tensor(xy, dtype=torch.float32, device=device)


def _eval_error_and_residual(model, g_train, device):
    """Compute:
      - L2 error ||pred - u*||_2 on 51x51 eval grid (manufactured solution).
      - Residuals eq_x, eq_y on the TRAINING collocation grid (for SK-CERT).
      - R_eval RMS on 51x51 via the same spectral operators in g_train (not
        re-built; smoke test uses training-grid residual RMS for consistency
        with SK-CERT's K_sens which is derived for the same operator class).
    """
    model.eval()
    # Training-grid residuals (these feed SK-CERT).
    with torch.no_grad():
        pred_train = model(g_train["xy_all"])
        pred_pde = pred_train[: g_train["N_all"]]
        eq_x, eq_y = compute_pde_elasticity(pred_pde, g_train)

    # 51x51 evaluation (solution error + PDE residual via autograd).
    xy_eval = _build_eval_grid(device).requires_grad_(True)
    pred_eval = model(xy_eval)
    ux, uy = pred_eval[:, 0:1], pred_eval[:, 1:2]

    def _grad(u):
        return torch.autograd.grad(u.sum(), xy_eval, create_graph=True)[0]

    grad_ux = _grad(ux)
    grad_uy = _grad(uy)
    grad_uxx = _grad(grad_ux[:, 0:1])
    grad_uxy = _grad(grad_ux[:, 1:2])
    grad_uyx = _grad(grad_uy[:, 0:1])
    grad_uyy = _grad(grad_uy[:, 1:2])

    d2ux_dx2 = grad_uxx[:, 0:1]
    d2ux_dy2 = grad_uxy[:, 1:2]
    d2uy_dx2 = grad_uyx[:, 0:1]
    d2uy_dy2 = grad_uyy[:, 1:2]
    d2ux_dxdy = grad_uxx[:, 1:2]
    d2uy_dxdy = grad_uyx[:, 1:2]

    lam = g_train["lam_e"]
    mu = g_train["mu_e"]

    # Body forces at eval points for the SAME manufactured solution under the
    # SAME (lam, mu) — mirrors _reparam_elasticity_grid_ in src/jax_pinn.py.
    x_ev, y_ev = xy_eval[:, 0:1], xy_eval[:, 1:2]
    pi = math.pi
    Q_E = 4.0
    ux_xx_ex = -((2 * pi) ** 2) * torch.cos(2 * pi * x_ev) * torch.sin(pi * y_ev)
    ux_yy_ex = -(pi ** 2) * torch.cos(2 * pi * x_ev) * torch.sin(pi * y_ev)
    ux_xy_ex = -2 * pi ** 2 * torch.sin(2 * pi * x_ev) * torch.cos(pi * y_ev)
    uy_xx_ex = -(pi ** 2) * torch.sin(pi * x_ev) * Q_E * y_ev ** 4 / 4.0
    uy_yy_ex = torch.sin(pi * x_ev) * Q_E * 3.0 * y_ev ** 2
    uy_xy_ex = pi * torch.cos(pi * x_ev) * Q_E * y_ev ** 3
    fx = -((lam + 2 * mu) * ux_xx_ex + mu * ux_yy_ex + (lam + mu) * uy_xy_ex)
    fy = -(mu * uy_xx_ex + (lam + 2 * mu) * uy_yy_ex + (lam + mu) * ux_xy_ex)

    res_x = ((lam + 2 * mu) * d2ux_dx2 + mu * d2ux_dy2
             + (lam + mu) * d2uy_dxdy + fx)
    res_y = (mu * d2uy_dx2 + (lam + 2 * mu) * d2uy_dy2
             + (lam + mu) * d2ux_dxdy + fy)
    R_eval_rms = float(torch.sqrt(((res_x ** 2 + res_y ** 2)).mean()).item())

    ux_ex, uy_ex = elasticity_exact(x_ev.detach(), y_ev.detach())
    err_ux = (ux.detach() - ux_ex).flatten()
    err_uy = (uy.detach() - uy_ex).flatten()
    # L2 error (||f_theta - u*||_2) on 51x51 eval grid.
    L2_err = float(torch.sqrt((err_ux ** 2 + err_uy ** 2).mean()).item())
    ux_rmse = float(torch.sqrt((err_ux ** 2).mean()).item())
    uy_rmse = float(torch.sqrt((err_uy ** 2).mean()).item())

    model.train()
    return {
        "eq_x_train": eq_x.detach(),
        "eq_y_train": eq_y.detach(),
        "L2_err_eval": L2_err,
        "ux_rmse_eval": ux_rmse,
        "uy_rmse_eval": uy_rmse,
        "pde_rms_eval": R_eval_rms,
    }


def _verdict(tau):
    if tau is None or not math.isfinite(tau):
        return "FAIL_non_finite"
    if tau < 1.0:
        return "FAIL_not_upper_bound"
    if tau > 10.0:
        return "FAIL_tau_gt_10"
    if tau <= 3.0:
        return "PASS_within_T1"
    return "PASS_smoke_but_above_T1"  # 3 < tau <= 10: smoke passes, but T1 at risk


def main():
    device = _pick_device()
    print(f"[SMOKE] device={device} torch={torch.__version__}")
    print(f"[SMOKE] F3 instance i=0 E_ratio={E_RATIO:.6f} nu_poisson={NU_POISSON:.6f}")

    # Build elasticity grid and reparameterize for i=0.
    g = build_grid_data_elasticity(GRID_SIZE, device)
    g = _reparam_elasticity_grid_(g, float(E_RATIO), float(NU_POISSON))
    lam_e = float(g["lam_e"])
    mu_e = float(g["mu_e"])
    print(f"[SMOKE] grid N_all={int(g['N_all'])} lam_e={lam_e:.6f} mu_e={mu_e:.6f}")

    # Reproduce B1 training (no checkpoint stored in retained summary).
    t0 = time.perf_counter()
    print(f"[SMOKE] train SAGE elasticity: seed={SEED} arch={ARCH} epochs={N_EPOCHS}")
    model, train_time, final_loss = train_sage_elasticity(
        seed=SEED, device=device, n_epochs=N_EPOCHS, lr=LR,
        technique="none", grid_size=GRID_SIZE, model_name=ARCH,
        tracker=None, grid_data=g,
    )
    t_train = time.perf_counter() - t0
    print(f"[SMOKE] training done in {t_train:.1f}s (train_time={train_time:.1f}s), "
          f"final_loss={final_loss:.6f}")

    # Post-training evaluation: residual fields + L2 error vs exact manufactured.
    t1 = time.perf_counter()
    ev = _eval_error_and_residual(model, g, device)
    t_eval = time.perf_counter() - t1
    print(f"[SMOKE] eval done in {t_eval:.2f}s")
    print(f"[SMOKE] pde_rms (eval grid, autograd) = {ev['pde_rms_eval']:.6e}")
    print(f"[SMOKE] L2 error  (eval grid)         = {ev['L2_err_eval']:.6e}")
    print(f"[SMOKE] ux RMSE (eval grid)           = {ev['ux_rmse_eval']:.6e}")
    print(f"[SMOKE] uy RMSE (eval grid)           = {ev['uy_rmse_eval']:.6e}")

    # SK-CERT bound.
    t2 = time.perf_counter()
    # r=1 picks the actual minimum eigenvalue (1st-smallest = true min), which
    # is the physically-meaningful PDE stability constant per Korn's inequality.
    # r=K would pick the K-th smallest = max eigenvalue, which inverts the
    # stability interpretation. The 04_design.md text literally reads "r-th
    # smallest", and r=1 is the conservative choice that most likely produces
    # a valid upper bound; this is the primary smoke-verdict reading.
    cert = compute_sk_cert(
        eq_x=ev["eq_x_train"], eq_y=ev["eq_y_train"],
        lam_e=lam_e, mu_e=mu_e, interior_mask=g["interior_mask"], r=1,
    )
    t_cert = time.perf_counter() - t2
    print(f"[SMOKE] compute_sk_cert done in {t_cert*1000:.2f}ms")
    print(f"[SMOKE] R_rms (train grid)            = {cert['R_rms']:.6e}")
    print(f"[SMOKE] K_sens                         = {cert['K_sens']}")
    print(f"[SMOKE] eigvals(G_literal)             = {cert['eigvals_G_literal']}")
    print(f"[SMOKE] eigvals(G_local)               = {cert['eigvals_G_local']}")
    print(f"[SMOKE] lambda_min (literal)           = {cert['lambda_min_literal']:.6e}")
    print(f"[SMOKE] lambda_min (local)             = {cert['lambda_min_local']:.6e}")
    print(f"[SMOKE] C_lambda (literal)             = {cert['C_lambda']:.6e}")
    print(f"[SMOKE] C_lambda (local)               = {cert['C_lambda_local']:.6e}")
    print(f"[SMOKE] B (literal)                    = {cert['B']:.6e}")
    print(f"[SMOKE] B (local)                      = {cert['B_local']:.6e}")

    err = ev["L2_err_eval"]
    tau_lit = cert["B"] / err if err > 0 else float("inf")
    tau_loc = cert["B_local"] / err if err > 0 else float("inf")
    # Cost ratio vs SAGE training.
    c_k = float(t_cert / max(t_train, 1e-9))

    print(f"[SMOKE] tau_1 (literal)  = {tau_lit:.4f}   verdict={_verdict(tau_lit)}")
    print(f"[SMOKE] tau_1 (local)    = {tau_loc:.4f}   verdict={_verdict(tau_loc)}")
    print(f"[SMOKE] c_k (cert_time / train_time) = {c_k:.3e}")

    # Primary verdict uses the grid-consistent (local) reading per 05_results.md
    # § SG-7 § Reading-2 resolution; Reading-1 is documentation.
    primary = _verdict(tau_loc)

    payload = {
        "instance": {
            "family": "F3_elasticity",
            "i": 0, "seed": SEED, "arch": ARCH, "grid_size": GRID_SIZE,
            "E_ratio": E_RATIO, "nu_poisson": NU_POISSON,
            "lam_e": lam_e, "mu_e": mu_e,
            "n_epochs": N_EPOCHS, "lr": LR,
        },
        "training": {
            "train_time_s": float(train_time),
            "wallclock_s": float(t_train),
            "final_loss": float(final_loss),
            "retained_reference_pde_rms": 0.062154896557331085,
        },
        "evaluation": {
            "pde_rms_eval": ev["pde_rms_eval"],
            "L2_err_eval": ev["L2_err_eval"],
            "ux_rmse_eval": ev["ux_rmse_eval"],
            "uy_rmse_eval": ev["uy_rmse_eval"],
        },
        "sk_cert": cert,
        "tightness": {
            "tau_literal": tau_lit,
            "tau_local": tau_loc,
            "c_k": c_k,
            "verdict_literal": _verdict(tau_lit),
            "verdict_local": _verdict(tau_loc),
            "primary_verdict": primary,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[SMOKE] wrote {OUT_JSON}")
    print(f"[SMOKE] PRIMARY VERDICT (grid-consistent reading): {primary}")


if __name__ == "__main__":
    main()
