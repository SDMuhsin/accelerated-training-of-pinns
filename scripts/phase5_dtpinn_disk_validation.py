"""Phase 5 validation: faithful DT-PINN on the paper's linear-Poisson-on-disk
benchmark.

Sweeps the three node-set sizes the paper validates extensively in Table 1 and
records final L2 / Linf / training time to a JSON file.

Reference run hyperparameters are paper-faithful:
  - 4×50 tanh MLP, fp64
  - L-BFGS, PyTorch defaults — no line search, max_iter=20, tolerance_grad=1e-7,
    tolerance_change=1e-9 (matches temp/dt-pinn/src/dtpinn_cupy_fp64.py:117)
  - lr=0.04 (paper)
  - 5000 outer L-BFGS steps (paper)
  - RBF-FD order p=4 (paper recommends p>2; 4 is the default in our experiments)
  - Robin BC (n·∇ + I) u = g (paper bctype=3)
  - Exact solution u(x,y) = 1 + sin(π x) cos(π y) (paper)
  - Disk node sets from temp/dt-pinn/MatlabSolver/DiskPoissonNodes.mat

Outputs results/dtpinn_phase5_disk/results.json with one entry per grid size.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch
import torch.nn as nn

from src.rbf_fd_operators import build_operators, load_disk_nodes, to_torch_sparse


def exact_solution(x, y):
    u = 1.0 + torch.sin(math.pi * x) * torch.cos(math.pi * y)
    lap = -2.0 * (math.pi ** 2) * torch.sin(math.pi * x) * torch.cos(math.pi * y)
    ux = math.pi * torch.cos(math.pi * x) * torch.cos(math.pi * y)
    uy = -math.pi * torch.sin(math.pi * x) * torch.sin(math.pi * y)
    return u, lap, ux, uy


class MLP(nn.Module):
    def __init__(self, layers=4, width=50):
        super().__init__()
        net = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(layers - 1):
            net.extend([nn.Linear(width, width), nn.Tanh()])
        net.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


def run_one(mat_path: str, k: int, p: int, epochs: int, lr: float, seed: int, device: torch.device):
    print("=" * 74)
    print(f"k={k}, p={p}, epochs={epochs}, lr={lr}, seed={seed}")
    print("=" * 74)
    torch.manual_seed(seed)

    Xi_np, Xb_np, normals_np, h = load_disk_nodes(mat_path, k)
    Ni, Nb = Xi_np.shape[0], Xb_np.shape[0]
    print(f"  Node set: Ni={Ni}, Nb={Nb}, h={h:.5f}")

    t0 = time.perf_counter()
    ops = build_operators(Xi_np, Xb_np, normals_np, p=p, derivs=("lap", (1, 0), (0, 1)))
    op_time = time.perf_counter() - t0
    Xf_np = ops["__metadata__"]["Xf"]
    Lap = to_torch_sparse(ops["lap"], dtype=torch.float64, device=device)
    Dx = to_torch_sparse(ops[(1, 0)], dtype=torch.float64, device=device)
    Dy = to_torch_sparse(ops[(0, 1)], dtype=torch.float64, device=device)

    Xf = torch.tensor(Xf_np, dtype=torch.float64, device=device)
    Xb = torch.tensor(Xb_np, dtype=torch.float64, device=device)
    normals = torch.tensor(normals_np, dtype=torch.float64, device=device)

    u_exact_full, lap_exact_full, _, _ = exact_solution(Xf[:, 0:1], Xf[:, 1:2])
    f = lap_exact_full[: Ni + Nb]
    ub_exact, _, ubx_exact, uby_exact = exact_solution(Xb[:, 0:1], Xb[:, 1:2])
    g_robin = normals[:, 0:1] * ubx_exact + normals[:, 1:2] * uby_exact + ub_exact

    model = MLP(layers=4, width=50).to(device).to(torch.float64)
    # Paper-faithful: temp/dt-pinn/src/dtpinn_cupy_fp64.py:117 builds LBFGS as
    # `optim.LBFGS(w.parameters(), lr=lr)` — no line search, PyTorch defaults
    # for max_iter / tolerance_grad / tolerance_change.
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)

    def closure():
        optimizer.zero_grad()
        u_full = model(Xf)
        pde_residual = torch.sparse.mm(Lap, u_full) - f
        u_xb = u_full[Ni : Ni + Nb]
        ux_xb = torch.sparse.mm(Dx, u_full)[Ni : Ni + Nb]
        uy_xb = torch.sparse.mm(Dy, u_full)[Ni : Ni + Nb]
        bc_residual = normals[:, 0:1] * ux_xb + normals[:, 1:2] * uy_xb + u_xb - g_robin
        loss = (pde_residual ** 2).mean() + (bc_residual ** 2).mean()
        loss.backward()
        return loss

    print("  Starting L-BFGS…")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_train0 = time.perf_counter()
    history = []
    for epoch in range(epochs):
        loss = optimizer.step(closure)
        loss_v = loss.item() if torch.is_tensor(loss) else float(loss)
        if (epoch + 1) % max(1, epochs // 25) == 0 or epoch == 0:
            with torch.no_grad():
                u_pred = model(Xf)
                u_int_bd = u_pred[: Ni + Nb]
                u_true = u_exact_full[: Ni + Nb]
                rel_l2 = (
                    torch.linalg.norm(u_int_bd - u_true)
                    / torch.linalg.norm(u_true)
                ).item()
            t_now = time.perf_counter()
            print(
                f"    epoch {epoch + 1:5d} | loss={loss_v:.4e} | rel_L2_u={rel_l2:.4e} | wall={t_now - t_train0:.1f}s"
            )
            history.append(
                {"epoch": epoch + 1, "loss": loss_v, "rel_l2_u": rel_l2}
            )
        if not np.isfinite(loss_v):
            print(f"    LOSS DIVERGED at epoch {epoch+1}; stopping.")
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t_train0

    with torch.no_grad():
        u_pred = model(Xf)
        u_int_bd = u_pred[: Ni + Nb]
        u_true = u_exact_full[: Ni + Nb]
        rel_l2 = (torch.linalg.norm(u_int_bd - u_true) / torch.linalg.norm(u_true)).item()
        abs_linf = (torch.linalg.norm(u_int_bd - u_true, ord=float("inf"))).item()

    print(f"  FINAL: rel_L2={rel_l2:.4e}, abs_Linf={abs_linf:.4e}, train_time={train_time:.1f}s")
    return {
        "k": k,
        "p": p,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "Ni": Ni,
        "Nb": Nb,
        "h": h,
        "operator_build_time_s": op_time,
        "train_time_s": train_time,
        "final_rel_l2": rel_l2,
        "final_abs_linf": abs_linf,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat-path", default="temp/dt-pinn/MatlabSolver/DiskPoissonNodes.mat")
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4],
                        help="Disk-node-set indices (1-based) — k=2/3/4 in the small file = N=828/1663/3196")
    parser.add_argument("--p", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/dtpinn_phase5_disk/results.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    results = []
    for k in args.ks:
        r = run_one(
            args.mat_path, k, args.p, args.epochs, args.lr, args.seed, device
        )
        results.append(r)
        # Stream-write after each grid in case we get interrupted
        with open(args.out, "w") as f:
            json.dump({
                "config": vars(args),
                "results": results,
            }, f, indent=2)

    print(f"\n{'=' * 70}\nAll results saved to {args.out}\n{'=' * 70}")
    for r in results:
        print(f"  k={r['k']} (N={r['Ni']+r['Nb']}): rel_L2={r['final_rel_l2']:.3e}, "
              f"abs_Linf={r['final_abs_linf']:.3e}, train={r['train_time_s']:.0f}s")


if __name__ == "__main__":
    main()
