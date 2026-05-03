#!/usr/bin/env python3
"""Quick MSE check: train AD-mode for N iters and eval u/v/p MSE.

Tests whether the AD-mode 'fast convergence to 5e-5 loss' is reaching
the cavity solution or an under-constrained PDE+BC minimum that's NOT
the cavity flow.
"""
from __future__ import annotations

import os
import sys
import time
import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.can_pinn_paper_validation import (
    CanPinnSirenMLP, build_grid, compute_total_loss, load_ground_truth, evaluate,
)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    print(f"device={device}, gpu={torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}")

    torch.manual_seed(0); np.random.seed(0)
    grid = build_grid(51, device, dtype)
    dx = 1.0 / 50.0
    nu_lam = 1.0 / 400.0
    gt = load_ground_truth(device, dtype)

    n_int = grid['xy_int'].shape[0]
    n_top = grid['xy_top'].shape[0]
    n_wall = grid['xy_wall'].shape[0]
    rng = np.random.default_rng(0)

    # Train AD-mode for N iters; eval at milestones.
    net = CanPinnSirenMLP(n_ffs=32, n_nodes=20, sigma=1.0).to(device).to(dtype)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    print(f"\n{'epoch':>8s} {'wall_s':>8s} {'loss':>14s} {'L_pde':>14s} {'L_bc':>14s}"
          f"  {'u_mse':>14s} {'v_mse':>14s} {'p_mse':>14s}")

    t0 = time.perf_counter()
    milestones = {5000, 10000, 15000, 20000, 25000}
    last = 0.0
    target_epochs = 25000
    for k in range(target_epochs):
        opt.zero_grad()
        int_idx = torch.tensor(
            rng.choice(n_int, size=475, replace=False),
            dtype=torch.long, device=device)
        n_top_b = max(1, int(round(25 * n_top / (n_top + n_wall))))
        n_wall_b = max(1, 25 - n_top_b)
        top_idx = torch.tensor(
            rng.choice(n_top, size=min(n_top_b, n_top), replace=False),
            dtype=torch.long, device=device)
        wall_idx = torch.tensor(
            rng.choice(n_wall, size=min(n_wall_b, n_wall), replace=False),
            dtype=torch.long, device=device)
        loss, L_pde, L_bc, _ = compute_total_loss(
            net, grid, dx, dx, 1.0, nu_lam, 'autodiff',
            int_batch_idx=int_idx, top_batch_idx=top_idx, wall_batch_idx=wall_idx)
        loss.backward()
        opt.step()
        last = loss.item()
        ep = k + 1
        if ep in milestones:
            wall = time.perf_counter() - t0
            m = evaluate(net, gt)
            print(f"{ep:>8d} {wall:>8.1f} {last:14.6e} {L_pde.item():14.6e} {L_bc.item():14.6e}"
                  f"  {m['u_mse']:14.6e} {m['v_mse']:14.6e} {m['p_mse']:14.6e}", flush=True)
            if wall > 600:
                print(f"  (time budget exceeded; stopping at epoch {ep})")
                break


if __name__ == "__main__":
    main()
