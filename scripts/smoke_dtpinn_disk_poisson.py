"""Smoke test for the faithful DT-PINN: linear Poisson on the unit disk.

Replicates the headline experiment of Sharma & Shankar 2022 with the
paper's hyperparameters:
  - 4×50 tanh MLP, fp64
  - L-BFGS, lr=0.04, PyTorch defaults (no line search; matches
    temp/dt-pinn/src/dtpinn_cupy_fp64.py:117)
  - 5K outer L-BFGS steps  (override via --epochs for smoke runs)
  - RBF-FD order p=4
  - Robin BC: (n·∇ + I) u = g, exact solution u = 1 + sin(πx)cos(πy)
  - Pre-stored disk node sets from temp/dt-pinn/MatlabSolver/DiskPoissonNodes.mat

Goal: training loss should decrease and reach a Poisson-residual L2 error
around the paper's reported numbers (Table 1 of the paper, ~1e-3 to 1e-4
range for our N=828–1663).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

# allow `python scripts/smoke_dtpinn_disk_poisson.py` from project root
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch
import torch.nn as nn

from src.rbf_fd_operators import build_operators, load_disk_nodes, to_torch_sparse


def exact_solution(x, y):
    """u(x, y) = 1 + sin(πx) cos(πy). Returns u, ∂_xx u + ∂_yy u, n·∇u + u (RHS for Robin BC)."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat-path", default="temp/dt-pinn/MatlabSolver/DiskPoissonNodes.mat")
    parser.add_argument("--k", type=int, default=2, help="1-based set index (k=2 → N=828)")
    parser.add_argument("--p", type=int, default=4, help="RBF-FD order")
    parser.add_argument("--epochs", type=int, default=200, help="L-BFGS outer steps")
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Nodes + operators
    Xi_np, Xb_np, normals_np, h = load_disk_nodes(args.mat_path, args.k)
    Ni, Nb = Xi_np.shape[0], Xb_np.shape[0]
    print(f"Loaded disk node set k={args.k}: Ni={Ni}, Nb={Nb}, h={h:.5f}")

    t0 = time.perf_counter()
    ops = build_operators(Xi_np, Xb_np, normals_np, p=args.p, derivs=("lap", (1, 0), (0, 1)))
    t1 = time.perf_counter()
    print(f"Built RBF-FD operators (p={args.p}) in {t1 - t0:.2f}s")

    Xf_np = ops["__metadata__"]["Xf"]
    print(f"Xf={Xf_np.shape}, ghost nodes: {ops['__metadata__']['Ng']}")

    # Convert to torch fp64 sparse
    Lap = to_torch_sparse(ops["lap"], dtype=torch.float64, device=device)
    Dx = to_torch_sparse(ops[(1, 0)], dtype=torch.float64, device=device)
    Dy = to_torch_sparse(ops[(0, 1)], dtype=torch.float64, device=device)

    Xf = torch.tensor(Xf_np, dtype=torch.float64, device=device)
    Xb = torch.tensor(Xb_np, dtype=torch.float64, device=device)
    normals = torch.tensor(normals_np, dtype=torch.float64, device=device)

    # Targets — Robin BC: (n·∇ + I) u = g.  Poisson: Δu = f.
    u_exact_full, lap_exact_full, ux_full, uy_full = exact_solution(Xf[:, 0:1], Xf[:, 1:2])
    f = lap_exact_full[: Ni + Nb]  # shape (Ni+Nb, 1)
    # boundary BC: ub-portion of Xf is rows Ni:Ni+Nb, but we already have Xb
    ub_exact, _, ubx_exact, uby_exact = exact_solution(Xb[:, 0:1], Xb[:, 1:2])
    g_robin = normals[:, 0:1] * ubx_exact + normals[:, 1:2] * uby_exact + ub_exact

    # Network in fp64
    model = MLP(layers=4, width=50).to(device).to(torch.float64)
    print(f"Model: {sum(p.numel() for p in model.parameters())} fp64 params")

    # Paper-faithful: temp/dt-pinn/src/dtpinn_cupy_fp64.py:117 builds LBFGS as
    # `optim.LBFGS(w.parameters(), lr=lr)` — no line search, PyTorch defaults
    # for max_iter / tolerance_grad / tolerance_change.
    optimizer = torch.optim.LBFGS(model.parameters(), lr=args.lr)

    def closure():
        optimizer.zero_grad()
        u_full = model(Xf)  # (Ntot, 1) fp64
        # PDE residual (interior + boundary as in paper)
        pde_residual = torch.sparse.mm(Lap, u_full) - f
        # Robin BC residual — at boundary points, indices [Ni : Ni+Nb] in the (Ni+Nb)-row operators
        u_xb = u_full[Ni : Ni + Nb]
        ux_xb = torch.sparse.mm(Dx, u_full)[Ni : Ni + Nb]
        uy_xb = torch.sparse.mm(Dy, u_full)[Ni : Ni + Nb]
        bc_residual = normals[:, 0:1] * ux_xb + normals[:, 1:2] * uy_xb + u_xb - g_robin

        loss = (pde_residual ** 2).mean() + (bc_residual ** 2).mean()
        loss.backward()
        return loss

    print(f"Starting L-BFGS (lr={args.lr}, paper-faithful: no line search, max_iter=20 per step)…")
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        loss = optimizer.step(closure)
        if (epoch + 1) % max(1, args.epochs // 20) == 0 or epoch == 0:
            with torch.no_grad():
                u_pred = model(Xf)
                u_int_bd = u_pred[: Ni + Nb]
                u_true = u_exact_full[: Ni + Nb]
                rel_l2_u = (
                    torch.linalg.norm(u_int_bd - u_true) / torch.linalg.norm(u_true)
                ).item()
                # Also compute autograd-pde residual at interior+boundary for sanity
                pde_disc = torch.sparse.mm(Lap, u_pred) - f
                pde_l2 = torch.linalg.norm(pde_disc).item()
            t1 = time.perf_counter()
            print(
                f"  epoch {epoch + 1:5d} | loss={loss.item():.4e} | rel_L2_u={rel_l2_u:.4e} | pde_disc_l2={pde_l2:.4e} | wall={t1 - t0:.1f}s"
            )

    t1 = time.perf_counter()
    print(f"\nTotal training time: {t1 - t0:.1f}s")
    with torch.no_grad():
        u_pred = model(Xf)
        u_int_bd = u_pred[: Ni + Nb]
        u_true = u_exact_full[: Ni + Nb]
        rel_l2_u = (torch.linalg.norm(u_int_bd - u_true) / torch.linalg.norm(u_true)).item()
        rel_inf_u = (torch.linalg.norm(u_int_bd - u_true, ord=float("inf"))).item()
        print(f"FINAL  rel_L2 = {rel_l2_u:.4e}    abs_Linf = {rel_inf_u:.4e}")


if __name__ == "__main__":
    main()
