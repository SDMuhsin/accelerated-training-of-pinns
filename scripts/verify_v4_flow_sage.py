"""Level-2 verification for V4 flow SAGE NS adjoint.

Compares parameter gradients on the same FD-based NS residual between:
- Autograd: ``loss.backward()`` through the 5-stencil autograd graph.
- FD-SAGE: 5 no_grad forwards → SAGE adjoint → 5 live
  ``pred.backward(gradient=…)`` calls.

Both paths compute the identical quantity (FD NS residual); only the
adjoint mechanism differs. Agreement must be at float32 noise level.

Usage::

    source env/bin/activate
    python scripts/verify_v4_flow_sage.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sage_ns_v4 import (  # noqa: E402
    _SAGE_DX_DEFAULT as _SAGE_DX,
    _SAGE_DY_DEFAULT as _SAGE_DY,
    build_v4_ns_sage_backward as _get_flow_sage_backward,
    flow_sage_pde_step as _flow_sage_pde_step,
)
from physicsnemo.sym.key import Key  # noqa: E402
from physicsnemo.sym.models.fully_connected import FullyConnectedArch  # noqa: E402


def autograd_flow_pde_step(
    flow_net: torch.nn.Module,
    batch,
    dx: float, dy: float, inv_Lx: float, inv_Ly: float, rho: float, nu: float,
    w_cont: float, w_momx: float, w_momy: float,
) -> torch.Tensor:
    """Reference: compute the same FD NS residual, let autograd differentiate
    through all 5 stencil forward passes, then single ``loss.backward()``."""
    x = batch["x"]; y = batch["y"]
    dw = batch["dw"]; sin = batch["sin"]; sout = batch["sout"]

    inv_rho = 1.0 / float(rho)
    inv_Lx2 = inv_Lx * inv_Lx
    inv_Ly2 = inv_Ly * inv_Ly

    def _fwd(xp, yp):
        out = flow_net({"x": xp, "y": yp, "dw": dw, "sin": sin, "sout": sout})
        return out["u"], out["v"], out["p"]

    u0,   v0,   p0   = _fwd(x,          y)
    u_xp, v_xp, p_xp = _fwd(x + dx,     y)
    u_xm, v_xm, p_xm = _fwd(x - dx,     y)
    u_yp, v_yp, p_yp = _fwd(x,          y + dy)
    u_ym, v_ym, p_ym = _fwd(x,          y - dy)

    inv_2dx = 1.0 / (2.0 * dx); inv_2dy = 1.0 / (2.0 * dy)
    inv_dx2 = 1.0 / (dx * dx); inv_dy2 = 1.0 / (dy * dy)

    du_dx = (u_xp - u_xm) * inv_2dx * inv_Lx
    du_dy = (u_yp - u_ym) * inv_2dy * inv_Ly
    dv_dx = (v_xp - v_xm) * inv_2dx * inv_Lx
    dv_dy = (v_yp - v_ym) * inv_2dy * inv_Ly
    dp_dx = (p_xp - p_xm) * inv_2dx * inv_Lx
    dp_dy = (p_yp - p_ym) * inv_2dy * inv_Ly
    d2u_dx2 = (u_xp + u_xm - 2.0 * u0) * inv_dx2 * inv_Lx2
    d2u_dy2 = (u_yp + u_ym - 2.0 * u0) * inv_dy2 * inv_Ly2
    d2v_dx2 = (v_xp + v_xm - 2.0 * v0) * inv_dx2 * inv_Lx2
    d2v_dy2 = (v_yp + v_ym - 2.0 * v0) * inv_dy2 * inv_Ly2

    continuity = du_dx + dv_dy
    mom_x = u0 * du_dx + v0 * du_dy + inv_rho * dp_dx - nu * (d2u_dx2 + d2u_dy2)
    mom_y = u0 * dv_dx + v0 * dv_dy + inv_rho * dp_dy - nu * (d2v_dx2 + d2v_dy2)

    loss = (w_cont * torch.mean(continuity ** 2)
            + w_momx * torch.mean(mom_x ** 2)
            + w_momy * torch.mean(mom_y ** 2))
    loss.backward()
    return loss.detach()


def snapshot_grads(model):
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

    # Match baseline flow net shape.
    arch_args = dict(
        input_keys=[Key(k) for k in ("x", "y", "dw", "sin", "sout")],
        output_keys=[Key(k) for k in ("u", "v", "p")],
        layer_size=512, nr_layers=12,
        activation_fn=torch.nn.SiLU(),
    )
    net_A = FullyConnectedArch(**arch_args).to(device)
    # Mild init amplification so FD residuals are non-trivial.
    with torch.no_grad():
        for p in net_A.parameters():
            p.mul_(1.5)
    net_B = FullyConnectedArch(**arch_args).to(device)
    net_B.load_state_dict(net_A.state_dict())

    # Random batch — baseline uses flow_pde_batch_size = 4096.
    B = 4096
    batch = {
        "x": torch.rand((B, 1), device=device),
        "y": torch.rand((B, 1), device=device),
        "dw": torch.rand((B, 1), device=device) * 0.1,
        "sin": torch.rand((B, 1), device=device),
        "sout": torch.rand((B, 1), device=device),
    }

    dx, dy = _SAGE_DX, _SAGE_DY
    # Match partner_v4_config.yaml defaults.
    rho = 1076.0; nu_stage = 1e-3
    # Representative V4 domain: Lx ~ 591, Ly ~ 415 (based on the smoke log:
    # inv_Lx ≈ 1.69e-3, inv_Ly ≈ 2.41e-3).
    inv_Lx = 1.69205e-3
    inv_Ly = 2.40964e-3

    w_cont = 1.0; w_momx = 1.0; w_momy = 1.0

    # Autograd path
    for p in net_A.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss_auto = autograd_flow_pde_step(
        net_A, batch, dx, dy, inv_Lx, inv_Ly, rho, nu_stage,
        w_cont, w_momx, w_momy,
    )
    grads_auto = snapshot_grads(net_A)

    # SAGE path
    for p in net_B.parameters():
        if p.grad is not None:
            p.grad.zero_()
    sage_backward = _get_flow_sage_backward(dx, dy, inv_Lx, inv_Ly, rho)
    loss_sage = _flow_sage_pde_step(
        net_B, batch, sage_backward,
        rho=rho, nu_stage=nu_stage,
        inv_Lx=inv_Lx, inv_Ly=inv_Ly,
        dx=dx, dy=dy,
        w_cont=w_cont, w_momx=w_momx, w_momy=w_momy,
    )
    grads_sage = snapshot_grads(net_B)

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

    ok = (rel_by_max < 1.0e-2) and (rel_by_mean < 1.0e-2)
    print("Result:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
