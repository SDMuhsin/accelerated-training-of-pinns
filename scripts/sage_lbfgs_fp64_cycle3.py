"""SAGE+L-BFGS+fp64 ablation for cycle-3 paper revision.

Runs the SAGE method (auto-generated PDE backward) under L-BFGS in fp64
on elasticity-MLP across 5 seeds (the cell where DT-PINN currently shows
the largest accuracy lead over SAGE-Adam-fp32 in Table III).

Outputs rows to results/lid_benchmark_results.csv with tag
sage_lbfgs_fp64_cycle3_20260427.

Reuses lid_benchmark internals (build_grid_data_elasticity,
_get_generated_backward_elasticity, compute_pde_elasticity, make_model)
so that the experiment differs from the Adam-fp32 SAGE row only in
optimizer/precision, not in PDE/grid/backward generation.

Hardware honesty: this script targets the locally available NVIDIA A40
(set via CUDA_VISIBLE_DEVICES). The H100 MIG 2g.20gb partition that
produced multiseed_20260427 is on a remote cluster. We log the actual
GPU name so the CSV is unambiguous.
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.lid_benchmark import (
    build_grid_data_elasticity,
    compute_pde_elasticity,
    _get_generated_backward_elasticity,
    elasticity_exact,
    elasticity_body_forces,
    make_model,
    model_reg_loss,
    chebyshev_diff_matrix,
    chebyshev_points,
    lam_e,
    mu_e,
)


CSV_PATH = REPO / "results" / "lid_benchmark_results.csv"
TAG = "sage_lbfgs_fp64_cycle3_20260427"


def build_grid_data_elasticity_fp64(N_grid: int, device: torch.device) -> dict:
    """Mirror of build_grid_data_elasticity but with all tensors in float64.

    Cast policy: differentiation matrices are formed in fp64 numpy then
    pushed to torch.float64 (no fp32 round-trip). Coordinates, body
    forces, and BC targets are also fp64. The SAGE-generated backward is
    dtype-polymorphic (it just calls Dxx @ adj etc.), so this drop-in
    suffices.
    """
    D1d = chebyshev_diff_matrix(N_grid) * 2.0
    I_mat = np.eye(N_grid)
    Dx_np = np.kron(I_mat, D1d)
    Dy_np = np.kron(D1d, I_mat)

    Dxx_np = Dx_np @ Dx_np
    Dyy_np = Dy_np @ Dy_np
    Dxy_np = Dy_np @ Dx_np

    x_ref = chebyshev_points(N_grid)
    x_phys = 0.5 * (x_ref + 1.0)
    xx, yy = np.meshgrid(x_phys, x_phys, indexing="xy")
    xy_grid = np.column_stack([xx.ravel(), yy.ravel()])

    eps = 1e-10
    xc, yc = xy_grid[:, 0], xy_grid[:, 1]
    is_boundary = (xc < eps) | (xc > 1 - eps) | (yc < eps) | (yc > 1 - eps)
    interior_idx = np.where(~is_boundary)[0]
    bc_idx = np.where(is_boundary)[0]

    N_all = len(xy_grid)
    N_bc = len(bc_idx)
    M = len(interior_idx)

    dt = torch.float64

    Dx = torch.tensor(Dx_np, dtype=dt, device=device)
    Dy = torch.tensor(Dy_np, dtype=dt, device=device)
    DxT = Dx.T.contiguous()
    DyT = Dy.T.contiguous()

    Dxx = torch.tensor(Dxx_np, dtype=dt, device=device)
    Dyy = torch.tensor(Dyy_np, dtype=dt, device=device)
    Dxy = torch.tensor(Dxy_np, dtype=dt, device=device)
    DxxT = Dxx.T.contiguous()
    DyyT = Dyy.T.contiguous()
    DxyT = Dxy.T.contiguous()

    xy_all = torch.tensor(xy_grid, dtype=dt, device=device)
    xy_bc = xy_all[bc_idx]

    interior_mask = torch.zeros(N_all, 1, device=device, dtype=dt)
    interior_mask[interior_idx] = 1.0

    fx_all, fy_all = elasticity_body_forces(xy_all[:, 0:1], xy_all[:, 1:2])
    fx_all = fx_all.to(dtype=dt)
    fy_all = fy_all.to(dtype=dt)

    ux_ex, uy_ex = elasticity_exact(xy_bc[:, 0:1], xy_bc[:, 1:2])
    bc_target = torch.cat([ux_ex.to(dtype=dt), uy_ex.to(dtype=dt)], dim=1)

    xy_batched = torch.cat([xy_all, xy_bc], dim=0)
    off_bc = N_all

    return {
        "Dx": Dx, "Dy": Dy, "DxT": DxT, "DyT": DyT,
        "Dxx": Dxx, "Dyy": Dyy, "Dxy": Dxy,
        "DxxT": DxxT, "DyyT": DyyT, "DxyT": DxyT,
        "xy_all": xy_all, "xy_bc": xy_bc, "xy_batched": xy_batched,
        "interior_idx": interior_idx, "bc_idx": bc_idx,
        "interior_mask": interior_mask,
        "fx": fx_all, "fy": fy_all,
        "bc_target": bc_target,
        "N_all": N_all, "N_bc": N_bc, "M": M,
        "off_bc": off_bc,
        "N_grid": N_grid,
        "lam_e": float(lam_e),
        "mu_e": float(mu_e),
    }


def evaluate_pde_rms(model: torch.nn.Module, g: dict) -> tuple[float, float, float]:
    """Returns (pde_rms, eq_x_rms, eq_y_rms) on the interior collocation
    points, all in fp64."""
    model.eval()
    with torch.no_grad():
        pred = model(g["xy_all"])
        eq_x, eq_y = compute_pde_elasticity(pred, g)
        ii = g["interior_idx"]
        eq_x_int = eq_x[ii]
        eq_y_int = eq_y[ii]
        eq_x_rms = float(torch.sqrt((eq_x_int ** 2).mean()).item())
        eq_y_rms = float(torch.sqrt((eq_y_int ** 2).mean()).item())
        pde_rms = float(
            torch.sqrt(((eq_x_int ** 2).mean() + (eq_y_int ** 2).mean()) / 2.0).item()
        )
    model.train()
    return pde_rms, eq_x_rms, eq_y_rms


def train_sage_lbfgs_fp64(
    seed: int,
    n_outer_steps: int,
    lr: float,
    grid_size: int,
    device: torch.device,
    log_interval: int = 250,
):
    """SAGE + L-BFGS + fp64 on elasticity (Navier-Cauchy), MLP architecture.

    Mirrors the DT-PINN paper-faithful protocol (raw L-BFGS, lr=0.04,
    5,000 outer steps) but uses the SAGE-generated dense Chebyshev
    backward in place of the DT-PINN sparse RBF-FD backward.
    Auto-restart with halving lr on stuck-at-init (matches DT-PINN
    wrapper) so the comparison stays apples-to-apples.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    g = build_grid_data_elasticity_fp64(grid_size, device)
    n_int = len(g["interior_idx"])
    n_bc = g["N_bc"]
    n_all = g["N_all"]

    generated_backward = _get_generated_backward_elasticity()

    MAX_RETRIES = 6
    ABORT_CHECKS = [(50, 0.5), (200, 0.01)]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    wall_start = time.perf_counter()

    cur_lr = lr
    final_loss = float("nan")
    final_model: torch.nn.Module | None = None
    retry_attempts = 0

    best_pde_rms = float("inf")
    best_state = None
    best_step = -1
    eval_interval = 100  # match Table-III best-checkpoint protocol

    while True:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = make_model("mlp", output_dim=2).to(device).to(torch.float64)
        optimizer = torch.optim.LBFGS(model.parameters(), lr=cur_lr)

        n_params = sum(p.numel() for p in model.parameters())

        # Closure: compute scalar loss + populate .grad via the SAGE-generated
        # PDE backward composed with the network reverse pass. This is the
        # one-line bridge between SAGE (which emits a vector adjoint) and
        # L-BFGS (which expects loss.backward() inside step()).
        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=False)

            pred_batch = model(g["xy_batched"])
            pred_pde = pred_batch[: g["N_all"]]
            pred_bc = pred_batch[g["off_bc"]: g["off_bc"] + g["N_bc"]]

            # Scalar loss for L-BFGS line search (computed without graph).
            with torch.no_grad():
                eq_x, eq_y = compute_pde_elasticity(pred_pde, g)
                ii = g["interior_idx"]
                loss_pde = (eq_x[ii] ** 2).mean() + (eq_y[ii] ** 2).mean()
                loss_bc = ((pred_bc - g["bc_target"]) ** 2).mean()
                reg = model_reg_loss(model)
                reg_val = reg.item() if isinstance(reg, torch.Tensor) else float(reg)
                scalar_loss_val = loss_pde.item() + loss_bc.item() + reg_val

                # SAGE-generated PDE backward returns d(L_pde)/d(pred_pde).
                grad_pde = generated_backward(pred_pde, g)
                # BC backward: d/d(pred_bc) of mean-square BC loss.
                n_out = pred_bc.shape[1]
                grad_bc = 2.0 * (pred_bc - g["bc_target"]) / (n_bc * n_out)
                upstream = torch.cat([grad_pde, grad_bc], dim=0)

            pred_batch.backward(gradient=upstream)
            reg = model_reg_loss(model)
            if isinstance(reg, torch.Tensor):
                reg.backward()

            # L-BFGS uses the returned scalar for both line search and the
            # convergence test. We pack the precomputed value into a leaf
            # tensor so torch.optim.LBFGS can treat it as a loss value.
            return torch.tensor(scalar_loss_val, dtype=torch.float64)

        initial_loss = float("nan")
        local_final_loss = float("nan")
        aborted = False

        for step in range(n_outer_steps):
            loss_val = optimizer.step(closure)
            local_final_loss = (
                loss_val.item() if torch.is_tensor(loss_val) else float(loss_val)
            )

            if step == 0:
                initial_loss = local_final_loss

            if (step + 1) % eval_interval == 0 or step == n_outer_steps - 1:
                pde_rms, eq_x_rms, eq_y_rms = evaluate_pde_rms(model, g)
                if pde_rms < best_pde_rms and math.isfinite(pde_rms):
                    best_pde_rms = pde_rms
                    best_step = step + 1
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            if (step + 1) % log_interval == 0 or step == 0:
                pde_rms, _, _ = evaluate_pde_rms(model, g)
                print(
                    f"  Step {step + 1:5d}: loss={local_final_loss:.6e} "
                    f"pde_rms={pde_rms:.6e} (best={best_pde_rms:.6e} @{best_step})",
                    flush=True,
                )

            triggered = False
            if retry_attempts < MAX_RETRIES:
                for check_step, max_ratio in ABORT_CHECKS:
                    if step == check_step:
                        threshold = max_ratio * initial_loss
                        if (
                            math.isnan(local_final_loss)
                            or local_final_loss > threshold
                        ):
                            print(
                                f"  No progress at step {step + 1}: "
                                f"loss={local_final_loss:.3e} > {threshold:.3e}; "
                                f"halving lr to {cur_lr/2.0} and retrying",
                                flush=True,
                            )
                            cur_lr = cur_lr / 2.0
                            aborted = True
                            triggered = True
                        break
            if triggered:
                break

        if not aborted:
            final_model = model
            final_loss = local_final_loss
            break

        retry_attempts += 1
        # Reset best-checkpoint tracker on restart so it reflects only the
        # successful run (mirrors DT-PINN's _dtpinn_train_loop semantics).
        best_pde_rms = float("inf")
        best_state = None
        best_step = -1

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time_s = time.perf_counter() - wall_start

    # Restore best checkpoint and re-evaluate (mirrors restore_best in
    # other lid_benchmark training paths).
    if best_state is not None:
        final_model.load_state_dict(best_state)
        pde_rms, eq_x_rms, eq_y_rms = evaluate_pde_rms(final_model, g)
    else:
        pde_rms, eq_x_rms, eq_y_rms = evaluate_pde_rms(final_model, g)
        best_step = n_outer_steps

    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        reserved_mem_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
    else:
        peak_mem_mb = 0.0
        reserved_mem_mb = 0.0

    return {
        "model": final_model,
        "train_time_s": train_time_s,
        "final_loss": final_loss,
        "pde_rms": pde_rms,
        "eq_x_rms": eq_x_rms,
        "eq_y_rms": eq_y_rms,
        "best_step": best_step,
        "n_params": sum(p.numel() for p in final_model.parameters()),
        "peak_mem_mb": peak_mem_mb,
        "reserved_mem_mb": reserved_mem_mb,
        "retry_attempts": retry_attempts,
        "final_lr": cur_lr,
    }


def append_csv_row(row: dict) -> None:
    """Append one row to lid_benchmark_results.csv preserving header."""
    file_exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(list(row.keys()))
        writer.writerow(list(row.values()))


def existing_header() -> list[str]:
    with CSV_PATH.open("r") as f:
        return next(csv.reader(f))


def write_row_using_existing_header(row_dict: dict) -> None:
    """Write a row to the CSV in column order matching the existing
    header, leaving any missing columns blank."""
    header = existing_header()
    row = [str(row_dict.get(col, "")) for col in header]
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="elasticity", choices=["elasticity"])
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--seeds", default="0,1,7,23,42")
    ap.add_argument("--n-outer", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=4e-2)
    ap.add_argument("--grid-size", type=int, default=30)
    ap.add_argument("--log-interval", type=int, default=500)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"Device: {device} ({gpu_name})")
    print(f"PyTorch: {torch.__version__}")
    print(f"Tag: {TAG}")

    if args.smoke:
        seeds = [42]
        n_outer = 100
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        n_outer = args.n_outer

    pytorch_version = torch.__version__

    for seed in seeds:
        print(f"\n=== seed={seed} ===")
        result = train_sage_lbfgs_fp64(
            seed=seed,
            n_outer_steps=n_outer,
            lr=args.lr,
            grid_size=args.grid_size,
            device=device,
            log_interval=args.log_interval,
        )

        ms_per_step = (result["train_time_s"] * 1000.0) / max(n_outer, 1)
        # PDE RMS already computed from best checkpoint above; eq_x_rms is
        # named continuity_rms in the schema for cavity (because the cavity
        # rows use cont./mom. semantics); for elasticity we follow Table III
        # convention (continuity_rms -> eq_x, momentum_rms -> eq_y).
        timestamp = datetime.now().isoformat()

        row = {
            "timestamp": timestamp,
            "problem": args.problem,
            "method": "sage-lbfgs-fp64",  # NEW METHOD STRING
            "model": args.model,
            "optimizer": "lbfgs",
            "lr": args.lr,
            "epochs": n_outer,  # we record outer-steps, parallel to dtpinn convention
            "seed": seed,
            "grid_size": args.grid_size,
            "technique": "none",
            "tag": TAG,
            "train_time_s": round(result["train_time_s"], 3),
            "train_time_min": round(result["train_time_s"] / 60.0, 3),
            "peak_gpu_memory_mb": round(result["peak_mem_mb"], 1),
            "gpu_memory_reserved_mb": round(result["reserved_mem_mb"], 1),
            "ms_per_epoch": round(ms_per_step, 1),
            "n_params": result["n_params"],
            "pde_rms": f"{result['pde_rms']:.6f}",
            "continuity_rms": f"{result['eq_x_rms']:.6f}",
            "momentum_rms": f"{result['eq_y_rms']:.6f}",
            "final_loss": f"{result['final_loss']:.6f}",
            "best_epoch": result["best_step"],
            "status": "OK",
            "device": str(device),
            "gpu_name": gpu_name,
            "pytorch_version": pytorch_version,
        }

        if not args.smoke:
            write_row_using_existing_header(row)
        else:
            print("(smoke mode: not appending to CSV)")
        for k, v in row.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
