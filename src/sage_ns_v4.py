"""SAGE drop-in for the V4 SteadyNavierStokes2DScaled PDE.

Replaces the baseline's SymPy → PhysicsNeMo autograd path for the
2-D steady NS PDE residual with a 5-point FD stencil (x±dx, y±dy,
centre) plus a SAGE-emitted analytical adjoint. Everything else in
PhysicsNeMo's training loop — Domain, Solver, PointwiseDataset,
PointwiseLossNorm, Sum aggregator, Adam, ExponentialLR, AMP, all
non-PDE constraints — is untouched; this module only replaces the
``SteadyNavierStokes2DScaled`` PDE class.

Public entry points:
- ``build_v4_ns_sage_backward`` — generate (and cache) the SAGE
  external-seed backward function for a given geometry / physics.
- ``V4NSFDSAGEFunction`` — the ``torch.autograd.Function`` that
  runs 4 no_grad stencil forwards and SAGE-adjoint backward; used
  internally by ``V4NSFDSAGEEvaluate``.
- ``V4NSFDSAGEEvaluate`` — ``nn.Module`` callable wired into a
  PhysicsNeMo ``Node`` as the evaluate function.
- ``SteadyNavierStokes2DScaledFDSAGE`` — the drop-in PDE subclass.
  Its ``make_nodes()`` returns a single Node that outputs the
  three residuals ``continuity``, ``momentum_x``, ``momentum_y``.

Apples-to-apples constraint (binding): every hyperparameter,
every sample, every seed, every schedule matches the V4 baseline
exactly. Only the gradient engine changes."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.node import Node

from symbolic_vjp import emit_backward, trace_pde_forward


# FD stencil step in normalised (x, y) coordinates. Must match
# ``_SAGE_DX`` / ``_SAGE_DY`` in the trainer so the traced backward
# and the runtime forward agree.
_SAGE_DX_DEFAULT = 1.0e-3
_SAGE_DY_DEFAULT = 1.0e-3

# Column order for the stacked ``pred`` tensor handed to the SAGE
# backward: 3 outputs × 5 stencil positions = 15 columns.
_SAGE_INPUT_NAMES = [
    "u0", "v0", "p0",
    "u_xp", "v_xp", "p_xp",
    "u_xm", "v_xm", "p_xm",
    "u_yp", "v_yp", "p_yp",
    "u_ym", "v_ym", "p_ym",
]


# ---------------------------------------------------------------------------
# SAGE backward generation (pure function, cached)
# ---------------------------------------------------------------------------

def _make_v4_ns_fd_forward(dx: float, dy: float, inv_Lx: float, inv_Ly: float,
                            rho: float):
    """Return a TracedVar-compatible forward of the V4 NS residual.

    FD step (dx, dy) and domain scaling (invLx, invLy) are baked in
    at trace time via Python-scalar ``smul`` ops. Viscosity varies
    per stage, so it flows through the tape as a ``g['nu_stage']``
    TracedVar constant and the emitted backward reads it from the
    runtime ``g`` dict at call time. The tape is agnostic to which
    stage we're in."""
    inv_2dx = 1.0 / (2.0 * dx)
    inv_2dy = 1.0 / (2.0 * dy)
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    inv_rho = 1.0 / float(rho)
    inv_Lx2 = inv_Lx * inv_Lx
    inv_Ly2 = inv_Ly * inv_Ly

    def compute_fn(pred, g):
        u0   = pred[:, 0:1];   v0   = pred[:, 1:2];   p0   = pred[:, 2:3]
        u_xp = pred[:, 3:4];   v_xp = pred[:, 4:5];   p_xp = pred[:, 5:6]
        u_xm = pred[:, 6:7];   v_xm = pred[:, 7:8];   p_xm = pred[:, 8:9]
        u_yp = pred[:, 9:10];  v_yp = pred[:, 10:11]; p_yp = pred[:, 11:12]
        u_ym = pred[:, 12:13]; v_ym = pred[:, 13:14]; p_ym = pred[:, 14:15]

        nu = g["nu_stage"]  # per-stage TracedVar constant

        # Normalised-space FD derivatives scaled to physical space via
        # inv_Lx / inv_Ly (Python floats baked in via smul).
        du_dx = (u_xp - u_xm) * (inv_2dx * inv_Lx)
        du_dy = (u_yp - u_ym) * (inv_2dy * inv_Ly)
        dv_dx = (v_xp - v_xm) * (inv_2dx * inv_Lx)
        dv_dy = (v_yp - v_ym) * (inv_2dy * inv_Ly)
        dp_dx = (p_xp - p_xm) * (inv_2dx * inv_Lx)
        dp_dy = (p_yp - p_ym) * (inv_2dy * inv_Ly)

        d2u_dx2 = (u_xp + u_xm - 2.0 * u0) * (inv_dx2 * inv_Lx2)
        d2u_dy2 = (u_yp + u_ym - 2.0 * u0) * (inv_dy2 * inv_Ly2)
        d2v_dx2 = (v_xp + v_xm - 2.0 * v0) * (inv_dx2 * inv_Lx2)
        d2v_dy2 = (v_yp + v_ym - 2.0 * v0) * (inv_dy2 * inv_Ly2)

        continuity = du_dx + dv_dy
        mom_x = (
            u0 * du_dx
            + v0 * du_dy
            + inv_rho * dp_dx
            - nu * (d2u_dx2 + d2u_dy2)
        )
        mom_y = (
            u0 * dv_dx
            + v0 * dv_dy
            + inv_rho * dp_dy
            - nu * (d2v_dx2 + d2v_dy2)
        )
        return (continuity, mom_x, mom_y)

    return compute_fn


_cached_v4_ns_sage_backward: Dict[Tuple[float, float, float, float, float], object] = {}


def build_v4_ns_sage_backward(dx: float, dy: float,
                               inv_Lx: float, inv_Ly: float, rho: float):
    """Build (and cache) the SAGE external-seed backward for V4 NS.

    The returned callable has signature
    ``fn(pred_det, g, grad_cont, grad_mom_x, grad_mom_y) -> adj_pred``
    where ``pred_det`` is the (B, 15) stacked stencil pred, ``g``
    must carry ``'nu_stage'`` (per-stage viscosity) and ``'N_all'``
    (= B), and the three grads are PhysicsNeMo's
    ``∂loss/∂residual_k`` coming out of PointwiseLossNorm."""
    key = (float(dx), float(dy), float(inv_Lx), float(inv_Ly), float(rho))
    cached = _cached_v4_ns_sage_backward.get(key)
    if cached is not None:
        return cached

    compute_fn = _make_v4_ns_fd_forward(*key)
    tape: list = []
    outputs, inputs = trace_pde_forward(
        compute_fn,
        N_all=None,
        tape=tape,
        sparse=False,
        constants=["nu_stage"],
        input_names=_SAGE_INPUT_NAMES,
    )
    _source, fn = emit_backward(
        tape,
        list(outputs),
        seed_names=["dc", "dmu", "dmv"],
        input_vars=inputs,
        sparse=False,
        func_name="generated_v4_ns_backward_ext",
        input_names=_SAGE_INPUT_NAMES,
        external_seeds=True,
    )
    print(
        f"[SAGE-V4-NS] Built external-seed backward: {len(tape)} tape ops, "
        f"dx={dx:g}, dy={dy:g}, invLx={inv_Lx:g}, invLy={inv_Ly:g}, rho={rho:g}"
    )
    _cached_v4_ns_sage_backward[key] = fn
    return fn


# ---------------------------------------------------------------------------
# torch.autograd.Function: FD-stencil forward + SAGE adjoint backward
# ---------------------------------------------------------------------------

class V4NSFDSAGEFunction(torch.autograd.Function):
    """Custom autograd Function for the V4 steady NS PDE residual.

    Forward: 4 no_grad stencil forwards on ``flow_net`` at
        ``(x±dx, y)`` and ``(x, y±dy)`` with ``dw, sin, sout`` held
        fixed; central differences give the 10 physical derivatives;
        residuals are assembled as in ``SteadyNavierStokes2DScaled``.
    Backward: upstream grads ``(grad_cont, grad_mom_x, grad_mom_y)``
        are PhysicsNeMo's ``∂loss/∂residual_k`` (from PointwiseLossNorm).
        We stack the center + 4 stencil (u, v, p) tensors into a
        (B, 15) pred and call the SAGE-generated external-seed
        backward to get ``adj_pred``. Its first three columns (center
        u, v, p) are returned as the Function's output grads so they
        flow back through the existing autograd graph PhysicsNeMo
        built when evaluating ``flow_net`` at the center. The other
        12 columns correspond to stencil-point u/v/p; we re-forward
        ``flow_net`` at each stencil position under ``enable_grad``
        and call ``pred_k.backward(gradient=adj_k)`` so those
        parameter-gradient contributions accumulate onto
        ``flow_net.parameters()``.

    The total gradient on ``flow_net.parameters()`` after this
    backward equals center_path + xp_path + xm_path + yp_path +
    ym_path, which is the correct full chain rule for the FD NS
    residual.
    """

    @staticmethod
    def forward(ctx, u_c, v_c, p_c, x, y, dw, sin_, sout_, sage_ctx):
        flow_net = sage_ctx.flow_net
        dx = sage_ctx.dx
        dy = sage_ctx.dy
        inv_Lx = sage_ctx.inv_Lx
        inv_Ly = sage_ctx.inv_Ly
        inv_rho = sage_ctx.inv_rho
        nu = sage_ctx.nu

        # Evaluate flow_net at the 4 stencil positions under no_grad.
        # The centre (u_c, v_c, p_c) was already evaluated by PhysicsNeMo
        # upstream — we receive its output via the input args.
        u_cd, v_cd, p_cd = u_c.detach(), v_c.detach(), p_c.detach()
        with torch.no_grad():
            out_xp = flow_net({"x": x + dx, "y": y,      "dw": dw, "sin": sin_, "sout": sout_})
            out_xm = flow_net({"x": x - dx, "y": y,      "dw": dw, "sin": sin_, "sout": sout_})
            out_yp = flow_net({"x": x,      "y": y + dy, "dw": dw, "sin": sin_, "sout": sout_})
            out_ym = flow_net({"x": x,      "y": y - dy, "dw": dw, "sin": sin_, "sout": sout_})
        u_xp, v_xp, p_xp = out_xp["u"], out_xp["v"], out_xp["p"]
        u_xm, v_xm, p_xm = out_xm["u"], out_xm["v"], out_xm["p"]
        u_yp, v_yp, p_yp = out_yp["u"], out_yp["v"], out_yp["p"]
        u_ym, v_ym, p_ym = out_ym["u"], out_ym["v"], out_ym["p"]

        # Physical-space FD derivatives.
        inv_2dx = 1.0 / (2.0 * dx); inv_2dy = 1.0 / (2.0 * dy)
        inv_dx2 = 1.0 / (dx * dx);  inv_dy2 = 1.0 / (dy * dy)
        inv_Lx2 = inv_Lx * inv_Lx;  inv_Ly2 = inv_Ly * inv_Ly

        du_dx = (u_xp - u_xm) * (inv_2dx * inv_Lx)
        du_dy = (u_yp - u_ym) * (inv_2dy * inv_Ly)
        dv_dx = (v_xp - v_xm) * (inv_2dx * inv_Lx)
        dv_dy = (v_yp - v_ym) * (inv_2dy * inv_Ly)
        dp_dx = (p_xp - p_xm) * (inv_2dx * inv_Lx)
        dp_dy = (p_yp - p_ym) * (inv_2dy * inv_Ly)

        d2u_dx2 = (u_xp + u_xm - 2.0 * u_cd) * (inv_dx2 * inv_Lx2)
        d2u_dy2 = (u_yp + u_ym - 2.0 * u_cd) * (inv_dy2 * inv_Ly2)
        d2v_dx2 = (v_xp + v_xm - 2.0 * v_cd) * (inv_dx2 * inv_Lx2)
        d2v_dy2 = (v_yp + v_ym - 2.0 * v_cd) * (inv_dy2 * inv_Ly2)

        continuity = du_dx + dv_dy
        momentum_x = u_cd * du_dx + v_cd * du_dy + inv_rho * dp_dx \
                     - nu * (d2u_dx2 + d2u_dy2)
        momentum_y = u_cd * dv_dx + v_cd * dv_dy + inv_rho * dp_dy \
                     - nu * (d2v_dx2 + d2v_dy2)

        # Save for backward: centres (detached), stencil outputs, FD inputs.
        ctx.save_for_backward(
            u_cd, v_cd, p_cd,
            u_xp, v_xp, p_xp,
            u_xm, v_xm, p_xm,
            u_yp, v_yp, p_yp,
            u_ym, v_ym, p_ym,
            x, y, dw, sin_, sout_,
        )
        ctx.sage_ctx = sage_ctx

        return continuity, momentum_x, momentum_y

    @staticmethod
    def backward(ctx, grad_cont, grad_mom_x, grad_mom_y):
        (u_cd, v_cd, p_cd,
         u_xp, v_xp, p_xp,
         u_xm, v_xm, p_xm,
         u_yp, v_yp, p_yp,
         u_ym, v_ym, p_ym,
         x, y, dw, sin_, sout_) = ctx.saved_tensors
        sage_ctx = ctx.sage_ctx
        flow_net = sage_ctx.flow_net
        dx = sage_ctx.dx; dy = sage_ctx.dy

        # Stack the 15 per-point values into a single (B, 15) tensor
        # in the column order declared by _SAGE_INPUT_NAMES.
        pred_stack = torch.cat(
            [u_cd, v_cd, p_cd,
             u_xp, v_xp, p_xp,
             u_xm, v_xm, p_xm,
             u_yp, v_yp, p_yp,
             u_ym, v_ym, p_ym],
            dim=1,
        )
        B = pred_stack.shape[0]

        # SAGE external-seed backward: pred_det, g, dc, dmu, dmv
        g_meta = {"nu_stage": sage_ctx.nu, "N_all": B}
        adj_pred = sage_ctx.sage_backward(
            pred_stack, g_meta,
            grad_cont, grad_mom_x, grad_mom_y,
        )

        # Split adjoint columns. Order matches _SAGE_INPUT_NAMES.
        adj_u_c  = adj_pred[:, 0:1]
        adj_v_c  = adj_pred[:, 1:2]
        adj_p_c  = adj_pred[:, 2:3]
        adj_u_xp = adj_pred[:, 3:4]
        adj_v_xp = adj_pred[:, 4:5]
        adj_p_xp = adj_pred[:, 5:6]
        adj_u_xm = adj_pred[:, 6:7]
        adj_v_xm = adj_pred[:, 7:8]
        adj_p_xm = adj_pred[:, 8:9]
        adj_u_yp = adj_pred[:, 9:10]
        adj_v_yp = adj_pred[:, 10:11]
        adj_p_yp = adj_pred[:, 11:12]
        adj_u_ym = adj_pred[:, 12:13]
        adj_v_ym = adj_pred[:, 13:14]
        adj_p_ym = adj_pred[:, 14:15]

        # --- Stencil-point grads: side-effect accumulate onto flow_net params.
        # The 4 stencil forwards inside our forward() ran under no_grad and
        # have no autograd graph. We re-forward them under enable_grad and
        # back-propagate with the adjoint as the upstream gradient.
        stencil_specs = (
            (x + dx, y,      adj_u_xp, adj_v_xp, adj_p_xp),
            (x - dx, y,      adj_u_xm, adj_v_xm, adj_p_xm),
            (x,      y + dy, adj_u_yp, adj_v_yp, adj_p_yp),
            (x,      y - dy, adj_u_ym, adj_v_ym, adj_p_ym),
        )
        with torch.enable_grad():
            for xk, yk, adj_uk, adj_vk, adj_pk in stencil_specs:
                out_k = flow_net({"x": xk, "y": yk, "dw": dw,
                                  "sin": sin_, "sout": sout_})
                pred_k = torch.cat([out_k["u"], out_k["v"], out_k["p"]], dim=1)
                grad_k = torch.cat([adj_uk, adj_vk, adj_pk], dim=1).detach()
                pred_k.backward(gradient=grad_k)

        # --- Centre-point grads: PyTorch routes these back through the
        # upstream autograd graph to flow_net.parameters(). Also return
        # None for the spatial inputs (x, y, dw, sin, sout) and for the
        # non-Tensor sage_ctx arg.
        return (adj_u_c, adj_v_c, adj_p_c,
                None, None, None, None, None,
                None)


# ---------------------------------------------------------------------------
# Shim carrying per-stage closure state for the autograd.Function
# ---------------------------------------------------------------------------

class _V4NSFDSAGECtx:
    """Lightweight, non-Tensor, non-Module closure carrier.

    Holds the flow net, per-stage viscosity, geometry scaling and the
    SAGE backward kernel. We pass one of these as the last arg to
    ``V4NSFDSAGEFunction.apply`` so PyTorch sees a single non-tensor
    opaque argument (returns None for it in ``backward``).
    """

    __slots__ = (
        "flow_net", "rho", "nu", "inv_rho",
        "inv_Lx", "inv_Ly", "dx", "dy", "sage_backward",
    )

    def __init__(self, flow_net, rho, nu, inv_Lx, inv_Ly, dx, dy, sage_backward):
        self.flow_net = flow_net
        self.rho = float(rho)
        self.nu = float(nu)
        self.inv_rho = 1.0 / max(self.rho, 1.0e-30)
        self.inv_Lx = float(inv_Lx)
        self.inv_Ly = float(inv_Ly)
        self.dx = float(dx)
        self.dy = float(dy)
        self.sage_backward = sage_backward


# ---------------------------------------------------------------------------
# PhysicsNeMo Node.evaluate wrapper (nn.Module required by ModuleList)
# ---------------------------------------------------------------------------

class V4NSFDSAGEEvaluate(torch.nn.Module):
    """nn.Module callable wired into a PhysicsNeMo Node.

    Receives PhysicsNeMo's invar/outvar batch dict, calls the
    ``torch.autograd.Function`` wrapper and returns a dict with the
    three residual keys. Flow net is stored via ``object.__setattr__``
    to avoid registering it as a submodule — it is already a submodule
    of the ``FullyConnectedArch`` Node which the Graph tracks for
    optimisation; double-registration would be invisible to the
    optimiser (our node has ``optimize=False``) but would still
    traverse extra params at save time.
    """

    def __init__(self, flow_net, rho, nu, Lx, Ly, dx, dy, sage_backward):
        super().__init__()
        inv_Lx = 1.0 / max(float(Lx), 1.0e-12)
        inv_Ly = 1.0 / max(float(Ly), 1.0e-12)
        # Store as a plain Python attribute (not registered as a submodule).
        object.__setattr__(
            self, "_sage_ctx",
            _V4NSFDSAGECtx(flow_net, rho, nu, inv_Lx, inv_Ly, dx, dy, sage_backward),
        )
        # Metadata consumed by PhysicsNeMo's Solver save/load path.
        self.saveable = False
        self.name = "SteadyNavierStokes2DScaledFDSAGE"

    def forward(self, var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cont, mom_x, mom_y = V4NSFDSAGEFunction.apply(
            var["u"], var["v"], var["p"],
            var["x"], var["y"],
            var["dw"], var["sin"], var["sout"],
            self._sage_ctx,
        )
        return {"continuity": cont, "momentum_x": mom_x, "momentum_y": mom_y}


# ---------------------------------------------------------------------------
# Drop-in PDE subclass
# ---------------------------------------------------------------------------

class SteadyNavierStokes2DScaledFDSAGE(PDE):
    """Apples-to-apples SAGE replacement for ``SteadyNavierStokes2DScaled``.

    Same residual identities (continuity, scaled momentum_x/y with
    Lx/Ly normalisation) but evaluated via a 5-point FD stencil whose
    adjoint is a SAGE-emitted analytical VJP rather than
    PhysicsNeMo's SymPy → ``torch.autograd.grad(create_graph=True)``
    chain. Everything downstream — Node inputs/outputs, Domain,
    Solver, constraints — is byte-identical to the baseline.
    """

    name = "SteadyNavierStokes2DScaledFDSAGE"

    def __init__(
        self,
        rho=1.0,
        nu=1.0e-3,
        Lx=1.0,
        Ly=1.0,
        flow_net=None,
        dx: float = _SAGE_DX_DEFAULT,
        dy: float = _SAGE_DY_DEFAULT,
        sage_backward=None,
    ):
        if flow_net is None:
            raise ValueError(
                "SteadyNavierStokes2DScaledFDSAGE requires a `flow_net` to "
                "evaluate at FD stencil positions."
            )
        super().__init__()

        self.rho = float(rho)
        self.nu = float(nu)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.flow_net = flow_net
        self.dx = float(dx)
        self.dy = float(dy)
        inv_Lx = 1.0 / max(self.Lx, 1.0e-12)
        inv_Ly = 1.0 / max(self.Ly, 1.0e-12)
        if sage_backward is None:
            sage_backward = build_v4_ns_sage_backward(
                self.dx, self.dy, inv_Lx, inv_Ly, self.rho,
            )
        self.sage_backward = sage_backward
        # Intentionally leave self.equations as the empty dict set by
        # PDE.__init__ — make_nodes() below bypasses the SymPy path.

    def make_nodes(self, create_instances: int = 1, freeze_terms=None,
                   detach_names=None, return_as_dict: bool = False):
        """Return a single Node whose evaluate returns all 3 residuals.

        The baseline PDE returns 3 Nodes (one per equation) because each
        SymPy equation is lambdified independently. Here a single FD
        stencil produces all 3 residuals in one call — so a single Node
        with 3 outputs avoids triplicate Function invocations per step.
        """
        if create_instances != 1:
            raise NotImplementedError(
                "SteadyNavierStokes2DScaledFDSAGE does not support "
                "create_instances != 1 (no use case in V4 pipeline)."
            )
        evaluate = V4NSFDSAGEEvaluate(
            flow_net=self.flow_net,
            rho=self.rho, nu=self.nu,
            Lx=self.Lx, Ly=self.Ly,
            dx=self.dx, dy=self.dy,
            sage_backward=self.sage_backward,
        )
        node = Node(
            inputs=["x", "y", "dw", "sin", "sout", "u", "v", "p"],
            outputs=["continuity", "momentum_x", "momentum_y"],
            evaluate=evaluate,
            name="SteadyNavierStokes2DScaledFDSAGE",
            optimize=False,
        )
        if return_as_dict:
            # Partner code does not use return_as_dict for this PDE, but
            # for completeness map every output key to the same Node.
            return {
                "continuity": node,
                "momentum_x": node,
                "momentum_y": node,
            }
        return [node]


# ---------------------------------------------------------------------------
# Verification helper used by scripts/verify_v4_flow_sage.py
# ---------------------------------------------------------------------------

def flow_sage_pde_step(
    flow_net,
    batch: Dict[str, torch.Tensor],
    sage_backward,
    *,
    rho: float,
    nu_stage: float,
    inv_Lx: float,
    inv_Ly: float,
    dx: float,
    dy: float,
    w_cont: float = 1.0,
    w_momx: float = 1.0,
    w_momy: float = 1.0,
) -> torch.Tensor:
    """Run one SAGE PDE step and accumulate grads onto flow_net.parameters().

    Used by the Level-2 verification script to compare the
    SAGE-adjoint parameter gradients against an autograd reference on
    the same FD residual. Not used by the training loop — the
    training loop goes through ``V4NSFDSAGEEvaluate`` / PhysicsNeMo.
    """
    x = batch["x"]; y = batch["y"]
    dw = batch["dw"]; sin_ = batch["sin"]; sout_ = batch["sout"]

    # Evaluate flow_net at the centre with grad enabled so the centre
    # path can flow back through autograd.
    out0 = flow_net({"x": x, "y": y, "dw": dw, "sin": sin_, "sout": sout_})
    u0, v0, p0 = out0["u"], out0["v"], out0["p"]

    sage_ctx = _V4NSFDSAGECtx(
        flow_net=flow_net, rho=rho, nu=nu_stage,
        inv_Lx=inv_Lx, inv_Ly=inv_Ly, dx=dx, dy=dy,
        sage_backward=sage_backward,
    )
    continuity, momentum_x, momentum_y = V4NSFDSAGEFunction.apply(
        u0, v0, p0, x, y, dw, sin_, sout_, sage_ctx,
    )
    loss = (
        w_cont * torch.mean(continuity ** 2)
        + w_momx * torch.mean(momentum_x ** 2)
        + w_momy * torch.mean(momentum_y ** 2)
    )
    loss.backward()
    return loss.detach()
