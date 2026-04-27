"""Phase-5 full-grid SK-CERT driver for F3 linear elasticity.

Iterates 10 instances x 3 seeds x 2 archs = 60 cells. For each cell:
  1. Reproduce B1 by retraining (no checkpoints retained from Phase 2)
     using PyTorch SAGE (train_sage_elasticity) at 30K Adam steps.
  2. Compute SK-CERT bound under the grid-consistent reading
     (RULING-2026-04-18-A in contract_interpretations.md): primary
     B is B_local = (1/sqrt(lambda_min(K_sens^T K_sens))) * R_rms.
  3. Compare against closed-form `elasticity_exact()` reference on
     the 51x51 evaluation grid.
  4. Record per-cell tau_k = B_local / L2_err and c_k = T_cert / T_train.

Per-instance pass = mean(tau_k over 3 seeds) <= T1_TAU_BAR (=3).
Per-arch T1 verdict = (>= 8 of 10 instances pass) per family.

Resilience: writes per-cell records to OUT_LOG (append, JSONL) so that
re-running the script skips already-completed cells. Final aggregate
written to OUT_SUMMARY when the grid finishes.

Usage:
  source env/bin/activate
  python3 scripts/run_sk_cert_grid_f3.py
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
    build_grid_data_elasticity,
    compute_pde_elasticity,
    elasticity_exact,
    train_sage_elasticity,
)
from src.jax_pinn import _reparam_elasticity_grid_
from src.symbolic_vjp import compute_sk_cert


# Frozen F3 instances (joint_uniform draw with seed 20260416), from
# llmdocs/research/archive/program_B_arc_closed_2026-04-17/02_landscape_instances.json
F3_INSTANCES = [
    (1.6776796987616658, 0.2132794290244124),
    (1.9793369907113483, 0.21337385130083905),
    (0.978266457559178,  0.2634363163439426),
    (1.6937249264724823, 0.40772943590958466),
    (1.880555919480621,  0.25742531731352636),
    (1.2743102191189024, 0.41550280208488843),
    (1.646839895132734,  0.3398802516709645),
    (0.6329144137471119, 0.26192624991969227),
    (1.2507970437979055, 0.39996548330540826),
    (0.5216613278636457, 0.3326467560748658),
]

# Retained Phase-2 used these three seeds (per analyzed.jsonl).
SEEDS = [0, 1, 42]
ARCHS = ["mlp", "pirate-net"]
GRID_SIZE = 50
N_EPOCHS = 30000
LR = 1e-3
T1_TAU_BAR = 3.0
T2_C_K_BAR = 0.25

OUT_LOG = _ROOT / "results" / "sk_cert_grid_f3_log.jsonl"
OUT_SUMMARY = _ROOT / "results" / "sk_cert_grid_f3_summary.json"


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
    """Match smoke driver semantics exactly:
      - eq_x_train, eq_y_train: training-grid residuals (input to SK-CERT).
      - L2_err_eval: ||f_theta - u*||_2 on 51x51 grid (closed-form ref).
      - pde_rms_eval: residual RMS on 51x51 via autograd (diagnostic).
    """
    model.eval()
    with torch.no_grad():
        pred_train = model(g_train["xy_all"])
        pred_pde = pred_train[: g_train["N_all"]]
        eq_x, eq_y = compute_pde_elasticity(pred_pde, g_train)

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


def _load_completed_keys(log_path):
    """Read existing JSONL log and return set of completed (i, seed, arch) keys."""
    if not log_path.exists():
        return set()
    keys = set()
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                keys.add((int(rec["instance_idx"]), int(rec["seed"]), str(rec["arch"])))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return keys


def _run_one_cell(i, seed, arch, device):
    """Train + cert + eval one cell. Returns flat record dict."""
    E_ratio, nu_poisson = F3_INSTANCES[i]
    print(f"[GRID-F3] cell i={i} seed={seed} arch={arch} "
          f"E_ratio={E_ratio:.6f} nu={nu_poisson:.6f}", flush=True)

    g = build_grid_data_elasticity(GRID_SIZE, device)
    g = _reparam_elasticity_grid_(g, float(E_ratio), float(nu_poisson))
    lam_e = float(g["lam_e"])
    mu_e = float(g["mu_e"])

    t0 = time.perf_counter()
    model, train_time, final_loss = train_sage_elasticity(
        seed=seed, device=device, n_epochs=N_EPOCHS, lr=LR,
        technique="none", grid_size=GRID_SIZE, model_name=arch,
        tracker=None, grid_data=g,
    )
    t_train = time.perf_counter() - t0

    t1 = time.perf_counter()
    ev = _eval_error_and_residual(model, g, device)
    t_eval = time.perf_counter() - t1

    t2 = time.perf_counter()
    cert = compute_sk_cert(
        eq_x=ev["eq_x_train"], eq_y=ev["eq_y_train"],
        lam_e=lam_e, mu_e=mu_e, interior_mask=g["interior_mask"], r=1,
    )
    t_cert = time.perf_counter() - t2

    err = ev["L2_err_eval"]
    tau_lit = cert["B"] / err if err > 0 else float("inf")
    tau_loc = cert["B_local"] / err if err > 0 else float("inf")
    c_k = float(t_cert / max(t_train, 1e-9))

    rec = {
        "family": "F3_elasticity",
        "instance_idx": i,
        "seed": seed,
        "arch": arch,
        "E_ratio": E_ratio,
        "nu_poisson": nu_poisson,
        "lam_e": lam_e,
        "mu_e": mu_e,
        "n_epochs": N_EPOCHS,
        "lr": LR,
        "train_time_s": float(train_time),
        "wallclock_train_s": float(t_train),
        "eval_time_s": float(t_eval),
        "cert_time_s": float(t_cert),
        "final_loss": float(final_loss),
        "L2_err_eval": ev["L2_err_eval"],
        "ux_rmse_eval": ev["ux_rmse_eval"],
        "uy_rmse_eval": ev["uy_rmse_eval"],
        "pde_rms_eval": ev["pde_rms_eval"],
        "B_literal": cert["B"],
        "B_local": cert["B_local"],
        "C_lambda_literal": cert["C_lambda"],
        "C_lambda_local": cert["C_lambda_local"],
        "R_rms_train": cert["R_rms"],
        "lambda_min_literal": cert["lambda_min_literal"],
        "lambda_min_local": cert["lambda_min_local"],
        "K_sens": cert["K_sens"],
        "tau_literal": tau_lit,
        "tau_local": tau_loc,
        "c_k": c_k,
    }

    print(f"[GRID-F3]   t_train={t_train:.1f}s  L2_err={err:.4e}  "
          f"B_local={cert['B_local']:.4e}  tau_local={tau_loc:.4f}  "
          f"c_k={c_k:.3e}", flush=True)

    # Free GPU memory between cells.
    del model, g, ev, cert
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return rec


def _aggregate(records):
    """Compute per-cell mean(tau_local) over 3 seeds; per-arch T1 verdict."""
    by_arch_inst = {}  # (arch, i) -> list of tau_local
    by_arch_inst_c = {}
    for rec in records:
        key = (rec["arch"], rec["instance_idx"])
        by_arch_inst.setdefault(key, []).append(rec["tau_local"])
        by_arch_inst_c.setdefault(key, []).append(rec["c_k"])

    per_cell = []
    for arch in ARCHS:
        for i in range(len(F3_INSTANCES)):
            taus = by_arch_inst.get((arch, i), [])
            cks = by_arch_inst_c.get((arch, i), [])
            if not taus:
                per_cell.append({
                    "arch": arch, "instance_idx": i, "n_seeds": 0,
                    "tau_mean": None, "tau_std": None, "c_k_mean": None,
                    "passes_T1": False,
                })
                continue
            taus_arr = np.asarray(taus, dtype=float)
            cks_arr = np.asarray(cks, dtype=float)
            tau_mean = float(taus_arr.mean())
            tau_std = float(taus_arr.std(ddof=0))
            c_k_mean = float(cks_arr.mean())
            per_cell.append({
                "arch": arch,
                "instance_idx": i,
                "n_seeds": len(taus),
                "tau_mean": tau_mean,
                "tau_std": tau_std,
                "c_k_mean": c_k_mean,
                "passes_T1": (tau_mean <= T1_TAU_BAR),
            })

    per_arch = {}
    for arch in ARCHS:
        cells = [c for c in per_cell if c["arch"] == arch]
        n_pass = sum(1 for c in cells if c["passes_T1"])
        c_k_vals = [c["c_k_mean"] for c in cells if c["c_k_mean"] is not None]
        per_arch[arch] = {
            "n_instances_pass_T1": n_pass,
            "n_instances_total": len(cells),
            "T1_verdict": "PASS" if n_pass >= 8 else "FAIL",
            "T2_c_k_max": (max(c_k_vals) if c_k_vals else None),
            "T2_verdict": (
                "PASS" if c_k_vals and max(c_k_vals) <= T2_C_K_BAR else "FAIL"
            ),
        }

    return {"per_cell": per_cell, "per_arch": per_arch}


def main():
    device = _pick_device()
    print(f"[GRID-F3] device={device}", flush=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)

    completed = _load_completed_keys(OUT_LOG)
    print(f"[GRID-F3] {len(completed)} cell(s) already completed in {OUT_LOG}",
          flush=True)

    cells = [(i, s, a) for a in ARCHS for i in range(len(F3_INSTANCES))
             for s in SEEDS]
    todo = [c for c in cells if c not in completed]
    print(f"[GRID-F3] running {len(todo)} cell(s) "
          f"({len(cells)} total, {len(completed)} skipped)", flush=True)

    t_start = time.perf_counter()
    for k, (i, seed, arch) in enumerate(todo, start=1):
        try:
            rec = _run_one_cell(i, seed, arch, device)
        except Exception as e:
            print(f"[GRID-F3] FAILED cell i={i} seed={seed} arch={arch}: {e}",
                  flush=True)
            err_rec = {
                "family": "F3_elasticity",
                "instance_idx": i, "seed": seed, "arch": arch,
                "error": repr(e),
            }
            with open(OUT_LOG, "a") as f:
                f.write(json.dumps(err_rec) + "\n")
            continue
        with open(OUT_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        elapsed = time.perf_counter() - t_start
        eta = elapsed * (len(todo) - k) / max(k, 1)
        print(f"[GRID-F3] [{k}/{len(todo)}] elapsed={elapsed/60:.1f}min  "
              f"eta={eta/60:.1f}min", flush=True)

    # Aggregate from log file (handles resume from prior partial runs).
    records = []
    with open(OUT_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "tau_local" in rec:
                    records.append(rec)
            except json.JSONDecodeError:
                continue

    summary = {
        "family": "F3_elasticity",
        "n_records_used": len(records),
        "n_cells_expected": len(cells),
        "T1_tau_bar": T1_TAU_BAR,
        "T2_c_k_bar": T2_C_K_BAR,
        "n_epochs": N_EPOCHS,
        "lr": LR,
        "grid_size": GRID_SIZE,
        "seeds": SEEDS,
        "archs": ARCHS,
        "ruling_applied": "RULING-2026-04-18-A (grid-consistent reading)",
    }
    summary.update(_aggregate(records))

    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[GRID-F3] wrote {OUT_SUMMARY}", flush=True)
    for arch, v in summary["per_arch"].items():
        print(f"[GRID-F3] {arch}: T1 {v['T1_verdict']} "
              f"({v['n_instances_pass_T1']}/{v['n_instances_total']}), "
              f"T2 {v['T2_verdict']} (c_k_max={v['T2_c_k_max']:.3e})",
              flush=True)


if __name__ == "__main__":
    main()
