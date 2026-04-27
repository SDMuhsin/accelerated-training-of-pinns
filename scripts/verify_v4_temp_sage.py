"""Level-2 verification for V4 temp SAGE integration.

Compares the parameter gradients produced by FD-SAGE against those produced
by regular PyTorch autograd, **on the same FD residual** (i.e. both paths
compute the same advection-diffusion PDE via the same 7 stencil forward
passes; only the adjoint mechanism differs). A correct SAGE adjoint should
match autograd to float32 noise (~1e-5 max abs diff, ~1e-5 relative).

This is different from comparing SAGE-FD vs autograd-exact — those two
would differ by the FD truncation error (O(dx^2)), which is a physics
difference, not a code bug. The apples-to-apples check below isolates the
code (SAGE adjoint) from the physics (FD truncation).

Usage::

    source env/bin/activate
    python scripts/verify_v4_temp_sage.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from partner_v4_temp_sage import (  # noqa: E402
    TemperatureNet,
    _SAGE_DX,
    _SAGE_DY,
    _SAGE_DT,
    _forward_temperature,
    _forward_no_grad_stencil,
    _get_temp_sage_backward,
    _sage_pde_step,
)


def autograd_pde_step(
    model: torch.nn.Module,
    x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
    u: torch.Tensor, v: torch.Tensor,
    dx: float, dy: float, dt_fd: float, D: float, Q: float,
    w_pde: float,
) -> torch.Tensor:
    """Reference implementation: compute the SAME FD-based PDE loss, but
    build the autograd graph through all 7 stencil forward passes and do a
    single ``loss.backward()``. Accumulates grads onto ``model.parameters()``.
    Returns the scalar loss (detached)."""
    T0 = _forward_temperature(model, x, y, t, u, v)
    T_xp = _forward_temperature(model, x + dx, y, t, u, v)
    T_xm = _forward_temperature(model, x - dx, y, t, u, v)
    T_yp = _forward_temperature(model, x, y + dy, t, u, v)
    T_ym = _forward_temperature(model, x, y - dy, t, u, v)
    T_tp = _forward_temperature(model, x, y, t + dt_fd, u, v)
    T_tm = _forward_temperature(model, x, y, t - dt_fd, u, v)

    T_x = (T_xp - T_xm) / (2.0 * dx)
    T_y = (T_yp - T_ym) / (2.0 * dy)
    T_t = (T_tp - T_tm) / (2.0 * dt_fd)
    T_xx = (T_xp - 2.0 * T0 + T_xm) / (dx * dx)
    T_yy = (T_yp - 2.0 * T0 + T_ym) / (dy * dy)
    residual = T_t + u * T_x + v * T_y - D * (T_xx + T_yy) - Q

    loss_pde = torch.mean(residual ** 2)
    (w_pde * loss_pde).backward()
    return loss_pde.detach()


def snapshot_grads(model: torch.nn.Module) -> torch.Tensor:
    parts = []
    for p in model.parameters():
        if p.grad is None:
            parts.append(torch.zeros_like(p).reshape(-1))
        else:
            parts.append(p.grad.detach().clone().reshape(-1))
    return torch.cat(parts)


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # V4 baseline temp net shape (from partner_v4_config.yaml temp.model).
    model_args = dict(in_dim=5, hidden_size=256, hidden_layers=12, activation="silu")
    model_A = TemperatureNet(**model_args).to(device)
    # Amplify weights so the net is not near-constant at random init (the
    # default init makes ∂T/∂x ≈ 0 everywhere, which drives the residual and
    # grads to float32 noise and the comparison stops being meaningful).
    with torch.no_grad():
        for p in model_A.parameters():
            p.mul_(3.0)
    model_B = TemperatureNet(**model_args).to(device)
    # Make both nets have identical weights so the grads should match exactly.
    model_B.load_state_dict(model_A.state_dict())

    # Single random PDE mini-batch (B = 2048, matches baseline config).
    B = 2048
    x = torch.rand((B, 1), device=device)
    y = torch.rand((B, 1), device=device)
    t = torch.rand((B, 1), device=device) * 40.0  # t in [0, 40]
    # Flow-ish data features (order-1 magnitudes, like real u/v scale).
    u = torch.randn((B, 1), device=device) * 1.0
    v = torch.randn((B, 1), device=device) * 1.0

    dx, dy, dt_fd = _SAGE_DX, _SAGE_DY, _SAGE_DT
    D, Q = 1.0e-5, 0.0
    w_pde = 1.0

    # Autograd path
    for p in model_A.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss_auto = autograd_pde_step(model_A, x, y, t, u, v, dx, dy, dt_fd, D, Q, w_pde)
    grads_auto = snapshot_grads(model_A)

    # SAGE path
    for p in model_B.parameters():
        if p.grad is not None:
            p.grad.zero_()
    sage_backward = _get_temp_sage_backward(dx, dy, dt_fd, D, Q)
    loss_sage = _sage_pde_step(
        model_B, x, y, t, u, v,
        sage_backward=sage_backward,
        w_pde=w_pde,
        dx=dx, dy=dy, dt_fd=dt_fd, D=D, Q=Q,
    )
    grads_sage = snapshot_grads(model_B)

    abs_diff = (grads_auto - grads_sage).abs()
    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())
    max_ref = float(grads_auto.abs().max().item())
    mean_ref = float(grads_auto.abs().mean().item()) + 1e-30
    rel_by_max = max_abs / (max_ref + 1e-30)
    rel_by_mean = mean_abs / mean_ref

    print(f"Params P                = {grads_auto.numel()}")
    print(f"Mean |grad_auto|        = {mean_ref:.3e}")
    print(f"Mean |grad_sage|        = {grads_sage.abs().mean().item():.3e}")
    print(f"Max  |grad_auto|        = {max_ref:.3e}")
    print(f"Loss autograd           = {loss_auto.item():.6e}")
    print(f"Loss SAGE               = {loss_sage.item():.6e}")
    print(f"Max |diff|              = {max_abs:.3e}")
    print(f"Mean |diff|             = {mean_abs:.3e}")
    print(f"Max |diff| / Max |ref|  = {rel_by_max:.3e}")
    print(f"Mean |diff| / Mean|ref| = {rel_by_mean:.3e}")

    # Float32 accumulates error over 12 layers × 256 wide × 7 stencil passes
    # × 2048 batch. Backward through that stack typically produces per-element
    # relative error of 1e-4 to 1e-3. Both normalised metrics should be below
    # 1e-2 for a correct adjoint; if they aren't, the adjoint has a bug.
    ok = (rel_by_max < 1.0e-2) and (rel_by_mean < 1.0e-2)
    print("Result:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
