"""Phase-5 smoke test for C3' KT-LAP on F2 Kovasznay.

Reproduces retained B1 (F2 Kovasznay, instance 0, seed 42, MLP, 30K Adam
SAGE iterations), resumes the SAGE trainer for M additional iterations
caching parameter snapshots, forms the trajectory-ensembled Gauss-Newton
Hessian as an HVP oracle, extracts top-r Ritz pairs via Lanczos with full
reorthogonalisation, constructs the KT-Laplace posterior Sigma = gamma I
+ V Lambda^{-1} V^T, draws n_sample parameter samples, and measures the
expected-coverage gap against the closed-form Kovasznay solution on the
51x51 X_eval grid (Kovasznay domain [-0.5, 1.0] x [-0.5, 1.5]).

Gate (per llmdocs/research/research_log/contract_target_pin.md
     SMOKE-TEST PRE-GATE):
  1. Validity: finite samples, empirical coverage in [0, 1].
  2. T1 smoke: coverage gap G_smoke <= 0.3.
  3. T2 smoke: cost ratio c_smoke <= 0.25.
  4. P2 smoke: layer-wise ablation coverage gap >= 5 * C3' coverage gap.

Any failure triggers Automatic Phase Reset + escalation to user per
contract section DEADLINE (attempt 2 is final).

Usage:
  source env/bin/activate
  python3 scripts/smoke_kt_lap_f2.py
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.lid_benchmark import (
    build_grid_data_kovasznay,
    compute_pde_kovasznay,
    train_sage_kovasznay,
    make_model,
    model_reg_loss,
    _get_generated_backward_kovasznay,
)
from src.jax_pinn import _reparam_kovasznay_grid_
from src.symbolic_vjp import (
    flatten_params,
    unflatten_into_model,
    param_layer_slices,
    gn_hvp_pinn,
    gn_hvp_pinn_layerwise,
    lanczos_topk,
    kt_laplace_sample,
)


# ---------------------------------------------------------------------------
# Frozen smoke cell per contract_target_pin.md SMOKE-TEST PRE-GATE.
# F2 Kovasznay, landscape instance 0 (log-uniform draw, seed=20260416).
# Re ~= 88.328 per program_B_arc_closed_2026-04-17/02_landscape_instances.json
# F2_Kovasznay.values[0].
# ---------------------------------------------------------------------------
RE_PARAM = 88.3277170958644
SEED = 42
ARCH = "mlp"
GRID_SIZE = 50         # N_all = 2500, matches retained Phase-2 B1 config.
N_EPOCHS = 30_000
LR = 1e-3

# C3' KT-LAP hyperparameters pinned per contract_target_pin.md section T4.
M_TRAJ = 5
BETAS = [1.0 / M_TRAJ] * M_TRAJ
LANCZOS_RANK = 20
LANCZOS_MAX_ITERS = 100
GAMMA = 1e-6
N_SAMPLE = 100
NOMINAL_COVERAGE = 0.9

OUT_JSON = _ROOT / "results" / "kt_lap_f2_smoke.json"


def _pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _kovasznay_reference(x, y, re_param):
    """Closed-form Kovasznay at Reynolds number re_param.
    lam = Re/2 - sqrt(Re^2/4 + 4 pi^2) per _reparam_kovasznay_grid_.
    This overrides the module-level lambda_kov (Re=40 legacy).
    """
    lam = re_param / 2.0 - math.sqrt(re_param ** 2 / 4.0 + 4.0 * math.pi ** 2)
    u = 1.0 - torch.exp(lam * x) * torch.cos(2.0 * math.pi * y)
    v = (lam / (2.0 * math.pi)) * torch.exp(lam * x) * torch.sin(2.0 * math.pi * y)
    p = 0.5 * (1.0 - torch.exp(2.0 * lam * x))
    return u, v, p


def _build_eval_grid(device, nx=51, ny=51):
    """51x51 uniform grid on Kovasznay domain [-0.5, 1.0] x [-0.5, 1.5].

    CRITICAL per memory/research_v3_phase5_c3_kt_lap_entry.md discipline note 1:
    the template at scripts/smoke_sk_cert_f3.py:59-65 uses [0,1]^2 (F3 domain);
    Kovasznay's non-square domain is [-0.5, 1.0] x [-0.5, 1.5], so we build
    the grid directly from the Kovasznay bounds rather than reusing the F3
    helper.
    """
    x = np.linspace(-0.5, 1.0, nx)
    y = np.linspace(-0.5, 1.5, ny)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack([X.ravel(), Y.ravel()])
    return torch.tensor(xy, dtype=torch.float32, device=device)


def _resume_for_trajectory(model, g, n_extra, lr, device):
    """Resume Adam SAGE training for n_extra iterations, returning parameter
    snapshots after each step. Mirrors the training loop in
    src/lid_benchmark.py::train_sage_kovasznay so the trajectory extends the
    retained B1 training consistently.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    backward_fn = _get_generated_backward_kovasznay()
    snapshots = []
    model.train()
    for epoch in range(n_extra):
        optimizer.zero_grad()
        pred_batch = model(g['xy_batched'])
        with torch.no_grad():
            pred_pde = pred_batch[:g['N_all']]
            pred_bc = pred_batch[g['off_bc']:g['off_center']]
            pred_c = pred_batch[g['off_center']:]
            grad_pde = backward_fn(pred_pde, g)
            N_bc = g['N_bc']
            grad_bc = 2.0 * (pred_bc - g['bc_target']) / (N_bc * pred_bc.shape[1])
            grad_center = torch.zeros(1, 3, device=device)
            grad_center[:, 2:3] = 2.0 * (pred_c[:, 2:3] - g['p_center_exact'])
            upstream = torch.cat([grad_pde, grad_bc, grad_center], dim=0)
        pred_batch.backward(gradient=upstream)
        reg = model_reg_loss(model)
        if isinstance(reg, torch.Tensor):
            reg.backward()
        optimizer.step()
        snapshots.append(flatten_params(model).detach().clone())
    return snapshots


def _pinn_loss_closure(model, g):
    """Zero-arg closure returning the PINN residual-squared loss at the
    current model parameters. The Gauss-Newton HVP in src/symbolic_vjp.py
    differentiates through this closure twice (Pearlmutter double-back)."""
    def closure():
        pred = model(g['xy_all'])
        r_c, r_mu, r_mv = compute_pde_kovasznay(pred, g)
        mask = g['interior_mask']
        # MSE-per-component matching train_sage_kovasznay's logging loss
        # structure (mean over interior points per component; Pearlmutter
        # HVP factors are all that matter up to a scalar).
        ii = g['interior_idx']
        loss = (r_c[ii] ** 2).mean() + (r_mu[ii] ** 2).mean() + (r_mv[ii] ** 2).mean()
        return loss
    return closure


def _make_trajectory_hvp(model_factory, device, snapshots, betas, g,
                         layerwise=False):
    """Build an HVP oracle that applies H = sum_t beta_t H_t v where each
    H_t is the PINN-loss Hessian at parameter snapshot theta_t. If
    layerwise=True, each H_t is restricted to its block-diagonal (one
    block per nn.Parameter tensor), implementing the layer-wise K-FAC
    analogue for the P2 SAGE-ablation partner.
    """
    base_model = model_factory().to(device)
    base_model.train()
    hvp_fn = gn_hvp_pinn_layerwise if layerwise else gn_hvp_pinn

    def oracle(v_flat):
        out = torch.zeros_like(v_flat)
        for theta_t, beta_t in zip(snapshots, betas):
            unflatten_into_model(base_model, theta_t)
            loss_closure = _pinn_loss_closure(base_model, g)
            out = out + beta_t * hvp_fn(base_model, loss_closure, v_flat)
        return out
    return oracle, base_model


def _evaluate_coverage(samples, base_model, xy_eval, u_ref_eval,
                       theta_star, nominal=0.9):
    """Forward each sample through the PINN, form per-point per-component
    5/95 credible intervals on the 51x51 X_eval grid, measure empirical
    coverage against the analytic Kovasznay reference. Returns coverage,
    gap, and diagnostics about per-sample finite-ness and relative
    perturbation vs theta_star.
    """
    base_model.eval()
    n_sample = samples.shape[0]

    # theta* prediction for Shape-gamma sanity diagnostic.
    unflatten_into_model(base_model, theta_star)
    with torch.no_grad():
        pred_star = base_model(xy_eval).detach()

    preds = torch.empty((n_sample,) + pred_star.shape, device=xy_eval.device,
                        dtype=pred_star.dtype)
    any_nonfinite = False
    rel_pert_vals = []
    for s in range(n_sample):
        unflatten_into_model(base_model, samples[s])
        with torch.no_grad():
            out = base_model(xy_eval)
        preds[s] = out
        if not torch.isfinite(out).all():
            any_nonfinite = True
        denom = max(float(pred_star.norm().item()), 1e-30)
        rel_pert_vals.append(float((out - pred_star).norm().item()) / denom)
    # Restore base_model to theta_star for subsequent evals.
    unflatten_into_model(base_model, theta_star)

    rel_pert_mean = float(np.mean(rel_pert_vals))
    rel_pert_med = float(np.median(rel_pert_vals))

    # Replace any non-finite predictions with theta* prediction for the
    # coverage computation (still flagged via any_nonfinite).
    mask_nonfinite = ~torch.isfinite(preds)
    if mask_nonfinite.any():
        preds[mask_nonfinite] = pred_star.expand_as(preds)[mask_nonfinite]

    # Per-point/per-component quantiles.
    q05 = torch.quantile(preds, 0.05, dim=0)   # (N_eval, 3)
    q95 = torch.quantile(preds, 0.95, dim=0)
    inside = (u_ref_eval >= q05) & (u_ref_eval <= q95)

    coverage_scalar = float(inside.float().mean().item())
    coverage_per_comp = [float(inside[:, c].float().mean().item())
                         for c in range(inside.shape[1])]
    gap = abs(coverage_scalar - nominal)

    # Interval-width and RMS-error diagnostics for reviewer transparency.
    width_mean = float((q95 - q05).mean().item())
    err_star_rms = float(torch.sqrt(((pred_star - u_ref_eval) ** 2).mean()).item())

    return {
        "coverage": coverage_scalar,
        "coverage_per_component": coverage_per_comp,
        "gap": gap,
        "any_nonfinite_sample": bool(any_nonfinite),
        "rel_perturbation_mean": rel_pert_mean,
        "rel_perturbation_median": rel_pert_med,
        "interval_width_mean": width_mean,
        "theta_star_err_rms_vs_ref": err_star_rms,
        "q05_mean": float(q05.mean().item()),
        "q95_mean": float(q95.mean().item()),
    }


def _run_posterior_branch(label, factory, snapshots, betas, g, theta_star,
                          xy_eval, u_ref_eval, P, device, layerwise=False):
    """Run one KT-LAP branch (C3' SAGE-native or B_no_SAGE layer-wise) from
    an already-computed trajectory snapshot list. Returns timings + a
    coverage diagnostic dict.
    """
    print(f"[SMOKE][{label}] building HVP oracle (layerwise={layerwise})")
    oracle, base_model = _make_trajectory_hvp(factory, device, snapshots,
                                              betas, g, layerwise=layerwise)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_lan = time.perf_counter()
    V, Lam = lanczos_topk(oracle, P, LANCZOS_RANK,
                          max_iters=LANCZOS_MAX_ITERS,
                          device=device, dtype=torch.float32,
                          seed=SEED)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_lan = time.perf_counter() - t_lan
    print(f"[SMOKE][{label}] Lanczos r={LANCZOS_RANK} done in {t_lan:.2f}s "
          f"(m_used rows={V.shape[1]}); eigvals min={float(Lam.min()):.3e} "
          f"max={float(Lam.max()):.3e}")

    # Clamp negative eigenvalues that can arise from full (non-GN) Hessian
    # curvature; sampling uses 1/sqrt(Lam) and negatives must not appear.
    Lam_clamped = torch.clamp(Lam, min=1e-30)

    t_samp = time.perf_counter()
    samples = kt_laplace_sample(theta_star, V, Lam_clamped, GAMMA, N_SAMPLE,
                                device=device, dtype=theta_star.dtype)
    t_samp = time.perf_counter() - t_samp
    print(f"[SMOKE][{label}] drew {N_SAMPLE} posterior samples in "
          f"{t_samp*1000:.1f}ms")

    t_cov = time.perf_counter()
    diag = _evaluate_coverage(samples, base_model, xy_eval, u_ref_eval,
                              theta_star, NOMINAL_COVERAGE)
    t_cov = time.perf_counter() - t_cov
    print(f"[SMOKE][{label}] coverage={diag['coverage']:.4f} "
          f"gap={diag['gap']:.4f} width_mean={diag['interval_width_mean']:.3e} "
          f"rel_pert_mean={diag['rel_perturbation_mean']:.3e}")
    diag["timings_s"] = {
        "lanczos": t_lan, "sampling": t_samp, "coverage_eval": t_cov,
    }
    diag["lanczos_rank_effective"] = int(V.shape[1])
    diag["eigvals_top5"] = [float(v) for v in Lam[:5].tolist()]
    diag["eigvals_bottom5"] = [float(v) for v in Lam[-5:].tolist()]
    return diag


def main():
    device = _pick_device()
    print(f"[SMOKE] device={device} torch={torch.__version__}")
    print(f"[SMOKE] F2 Kovasznay Re={RE_PARAM:.6f}  seed={SEED}  arch={ARCH}")

    # Build Chebyshev grid at size 50, reparam for Re=88.33.
    g = build_grid_data_kovasznay(GRID_SIZE, device)
    g = _reparam_kovasznay_grid_(g, float(RE_PARAM))
    print(f"[SMOKE] grid N_all={int(g['N_all'])}  N_int={int(g['M'])}  "
          f"nu_kov={g['nu_kov']:.6f}")

    # --- B1 training ---
    t0 = time.perf_counter()
    print(f"[SMOKE] train_sage_kovasznay: n_epochs={N_EPOCHS} lr={LR}")
    model, train_time, final_loss = train_sage_kovasznay(
        seed=SEED, device=device, n_epochs=N_EPOCHS, lr=LR,
        technique="none", grid_size=GRID_SIZE, model_name=ARCH,
        tracker=None, grid_data=g,
    )
    t_train_wall = time.perf_counter() - t0
    print(f"[SMOKE] training done in {t_train_wall:.1f}s "
          f"(reported train_time={train_time:.1f}s); final_loss={final_loss:.6f}")

    # Capture theta_star as post-30K-iter point then resume for M more.
    theta_star_30k = flatten_params(model).detach().clone()
    P = theta_star_30k.numel()
    print(f"[SMOKE] P = {P}")

    # --- Trajectory: M extra SAGE iterations, cache parameter snapshots ---
    t1 = time.perf_counter()
    snapshots = _resume_for_trajectory(model, g, M_TRAJ, LR, device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_resume = time.perf_counter() - t1
    print(f"[SMOKE] resumed {M_TRAJ} SAGE iterations in {t_resume:.2f}s; "
          f"theta_norm@T-M+1..T moved "
          f"{(snapshots[-1] - theta_star_30k).norm().item():.3e}")

    theta_star = snapshots[-1].detach().clone()   # use post-iteration T as MAP

    def factory():
        # Deterministic re-creation: torch.manual_seed(SEED) ensures the same
        # init parameter layout (needed for unflatten consistency across
        # multiple model instances; init values are overwritten immediately
        # by unflatten_into_model).
        torch.manual_seed(SEED)
        return make_model(ARCH)

    # Build 51x51 eval grid on Kovasznay domain.
    xy_eval = _build_eval_grid(device)
    u_ex, v_ex, p_ex = _kovasznay_reference(xy_eval[:, 0:1], xy_eval[:, 1:2],
                                            RE_PARAM)
    u_ref_eval = torch.cat([u_ex, v_ex, p_ex], dim=1)
    print(f"[SMOKE] X_eval shape={tuple(xy_eval.shape)}  "
          f"u_ref range u:[{u_ex.min().item():.3f},{u_ex.max().item():.3f}]")

    # --- Branch A: C3' SAGE-native (full cross-layer HVP) ---
    diag_c3 = _run_posterior_branch(
        "C3'", factory, snapshots, BETAS, g, theta_star,
        xy_eval, u_ref_eval, P, device, layerwise=False,
    )

    # --- Branch B: B_no_SAGE layer-wise (P2 ablation partner) ---
    diag_b2 = _run_posterior_branch(
        "B_no_SAGE", factory, snapshots, BETAS, g, theta_star,
        xy_eval, u_ref_eval, P, device, layerwise=True,
    )

    # --- Cost + gate computation ---
    t_uq_total = (t_resume
                  + diag_c3["timings_s"]["lanczos"]
                  + diag_c3["timings_s"]["sampling"]
                  + diag_c3["timings_s"]["coverage_eval"])
    c_smoke = t_uq_total / max(train_time, 1e-9)
    print(f"[SMOKE] c_smoke = {c_smoke:.4e} "
          f"(trajectory+lanczos+sample+eval = {t_uq_total:.2f}s / "
          f"train = {train_time:.1f}s)")

    g_smoke = diag_c3["gap"]
    g_b2 = diag_b2["gap"]
    p2_ratio = g_b2 / g_smoke if g_smoke > 0 else float("inf")

    # --- Pass/fail verdicts per contract_target_pin.md SMOKE-TEST PRE-GATE ---
    verdict_validity = ("PASS" if (not diag_c3["any_nonfinite_sample"]
                                   and 0.0 <= diag_c3["coverage"] <= 1.0)
                        else "FAIL_non_finite_or_out_of_range")
    verdict_t1 = "PASS" if g_smoke <= 0.3 else "FAIL_G_smoke_gt_0.3"
    verdict_t2 = "PASS" if c_smoke <= 0.25 else "FAIL_c_smoke_gt_0.25"
    verdict_p2 = "PASS" if p2_ratio >= 5.0 else "FAIL_p2_lt_5x"

    all_pass = all(v.startswith("PASS") for v in (
        verdict_validity, verdict_t1, verdict_t2, verdict_p2))
    primary = "PASS_smoke" if all_pass else "FAIL_smoke"

    print(f"[SMOKE] VERDICT validity = {verdict_validity}")
    print(f"[SMOKE] VERDICT T1 gap<=0.3 = {verdict_t1}  (G_smoke={g_smoke:.4f})")
    print(f"[SMOKE] VERDICT T2 c<=0.25 = {verdict_t2}  (c_smoke={c_smoke:.4e})")
    print(f"[SMOKE] VERDICT P2 ratio>=5x = {verdict_p2}  "
          f"(gap_B_no_SAGE/gap_C3' = {p2_ratio:.3f})")
    print(f"[SMOKE] PRIMARY VERDICT = {primary}")

    payload = {
        "instance": {
            "family": "F2_Kovasznay",
            "i": 0, "Re": RE_PARAM, "seed": SEED, "arch": ARCH,
            "grid_size": GRID_SIZE, "N_all": int(g['N_all']),
            "N_int": int(g['M']), "nu_kov": float(g['nu_kov']),
            "n_epochs": N_EPOCHS, "lr": LR,
            "P_params": int(P),
        },
        "hyperparameters": {
            "M_trajectory": M_TRAJ, "betas": BETAS,
            "lanczos_rank": LANCZOS_RANK,
            "lanczos_max_iters": LANCZOS_MAX_ITERS,
            "gamma": GAMMA, "n_sample": N_SAMPLE,
            "nominal_coverage": NOMINAL_COVERAGE,
        },
        "training": {
            "train_time_s": float(train_time),
            "wallclock_s": float(t_train_wall),
            "final_loss": float(final_loss),
            "resume_trajectory_s": float(t_resume),
            "theta_trajectory_drift_norm": float(
                (snapshots[-1] - theta_star_30k).norm().item()),
        },
        "c3_prime_sage_native": diag_c3,
        "b_no_sage_layerwise": diag_b2,
        "gating": {
            "c_smoke": c_smoke,
            "g_smoke": g_smoke,
            "g_b_no_sage": g_b2,
            "p2_ratio_b2_over_c3": p2_ratio,
            "verdict_validity": verdict_validity,
            "verdict_t1": verdict_t1,
            "verdict_t2": verdict_t2,
            "verdict_p2": verdict_p2,
            "primary_verdict": primary,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[SMOKE] wrote {OUT_JSON}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
