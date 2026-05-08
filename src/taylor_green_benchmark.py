"""3D Taylor-Green Vortex PINN benchmark — Phase 1 (AD) + Phase 2 (LES) + Phase 3 (PirateNet/causal/SOAP).

Faithful port of NVIDIA PhysicsNeMo-Sym's TGV reference at
``temp/physicsnemo-sym/examples/taylor_green/`` (Apache-2.0). Re=500 vanilla
incompressible Navier-Stokes on [0, 2 pi]^3 x [0, 10] with triply periodic BCs
and the Taylor-Green initial condition. Trained with the moving-time-window
scheme (10 windows x T_w=1.0); each window matches the previous window's
terminal state at its own t_local=0.

Phase 1 scope (--les-cs=0, --model=mlp, --optimizer=adam, --causal-eps=0):
    AD baseline only.
Phase 2 scope (--les-cs>0): adds Smagorinsky LES closure with effective
    viscosity ``nu_eff = nu_lam + (cs*delta)^2 |S|`` and the full
    stress-tensor viscous form ``visc_i = d_j(nu_eff*(d_j u_i + d_i u_j))``.
Phase 3 scope:
    --model=pirate-net  PirateNet residual architecture (Wang et al. 2024),
                        port of temp/jaxpi-pirate/jaxpi/archs.py:342.
                        (Hyphenated name matches src/lid_benchmark.py 2D
                        sweep-system convention.)
    --causal-eps>0      Causal training loss (Wang et al. 2022); per-chunk
                        temporal weighting w_i = exp(-eps * cumsum(l_i)) / w_0
                        applied to the PDE residual. Port of
                        temp/physicsnemo/physicsnemo/sym/loss/loss.py:271.
    --optimizer=soap    SOAP optimizer (Vyas et al. 2024); vendored at
                        src/soap.py from github.com/nikhilvyas/SOAP (Apache-2.0).
Phase 3 target: Re=1600, head-to-head with Wang/Perdikaris arXiv:2507.08972.

Bit-equivalence regression gates (CONTEXT.md §0.6/§4.3):
    --model=mlp --les-cs=0  --optimizer=adam --causal-eps=0  ==> Phase 1 row.
    --model=mlp --les-cs>0  --optimizer=adam --causal-eps=0  ==> Phase 2 row.

Reference (NVIDIA): https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/intermediate/moving_time_window.html
Validation: NGC spectral-solver TKE decay curves at Re=500, N=128 and N=256.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

LOG_INTERVAL = 500


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="3D Taylor-Green Vortex PINN benchmark (Phase 1 AD baseline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--method", default="autodiff", choices=["autodiff"],
                   help="Gradient-engine method. AD only for now.")
    p.add_argument("--model", default="mlp", choices=["mlp", "pirate-net"],
                   help="Network architecture. 'mlp' is the Phase 1/2 Fourier MLP; "
                        "'pirate-net' enables Phase 3 PirateNet residual arch "
                        "(naming matches src/lid_benchmark.py for sweep-system compatibility).")
    p.add_argument("--re", type=float, default=500.0, help="Reynolds number")
    p.add_argument("--domain-length", type=float, default=2 * math.pi,
                   help="Cubic domain side length L; full domain is [0,L]^3.")
    p.add_argument("--num-windows", type=int, default=10,
                   help="Number of sequential time windows.")
    p.add_argument("--window-size", type=float, default=1.0,
                   help="Length of each time window (T_w). Total time = num_windows * T_w.")
    p.add_argument("--hidden-dim", type=int, default=256, help="MLP hidden width.")
    p.add_argument("--num-layers", type=int, default=6, help="MLP hidden depth (Linear+Tanh blocks).")
    p.add_argument("--epochs-per-window", type=int, default=30000,
                   help="Adam epochs per window. PhysicsNeMo default is 300k total / 11 windows ≈ 27k/window.")
    p.add_argument("--batch-interior", type=int, default=4096,
                   help="Interior collocation points per epoch.")
    p.add_argument("--batch-ic", type=int, default=4096,
                   help="Initial-condition / window-match points per epoch.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-decay-rate", type=float, default=0.95,
                   help="Exponential LR decay multiplier per --lr-decay-steps.")
    p.add_argument("--lr-decay-steps", type=int, default=3000)
    p.add_argument("--ic-weight", type=float, default=100.0,
                   help="Loss weight on IC and previous-window-match terms (PhysicsNeMo default: 100).")
    p.add_argument("--les-cs", type=float, default=0.0,
                   help="Smagorinsky constant. 0.0 disables LES (Phase 1 laminar path); "
                        "0.1 is the canonical Smagorinsky value used for TGV.")
    p.add_argument("--les-delta", type=float, default=2 * math.pi / 64.0,
                   help="LES filter width Delta. Defaults to L/64 (=2 pi/64) for the "
                        "canonical [0, 2 pi]^3 TGV domain. Ignored when --les-cs=0.")
    p.add_argument("--les-eps", type=float, default=1e-12,
                   help="Stabiliser added inside sqrt(.) when computing |S|.")
    # ----- Phase 3 knobs (PirateNet / causal / SOAP). Defaults preserve Phase 1/2. -----
    p.add_argument("--pirate-num-layers", type=int, default=3,
                   help="PirateNet bottleneck count (default 3 per Wang Kolmogorov config). "
                        "Ignored when --model=mlp.")
    p.add_argument("--pirate-hidden-dim", type=int, default=256,
                   help="PirateNet hidden width inside each bottleneck. Ignored when --model=mlp.")
    p.add_argument("--pirate-nonlinearity", type=float, default=0.0,
                   help="PirateNet alpha init (0.0 => identity init at start, the 'physics-informed init' trick).")
    p.add_argument("--causal-eps", type=float, default=0.0,
                   help="Causal-loss eps (>0 enables temporal causal weighting; "
                        "0.0 = no causal, route through the existing mean-PDE-loss path "
                        "for bit-equivalence with Phase 1/2).")
    p.add_argument("--causal-chunks", type=int, default=10,
                   help="Number of temporal chunks for the causal loss (PhysicsNeMo default 10; "
                        "Wang Kolmogorov config uses 16). Ignored when --causal-eps=0.")
    p.add_argument("--optimizer", default="adam", choices=["adam", "soap"],
                   help="Optimizer choice. 'adam' is the Phase 1/2 default; 'soap' enables "
                        "the SOAP optimizer (Vyas et al. 2024) for Phase 3.")
    p.add_argument("--soap-betas", type=str, default="0.9,0.999",
                   help="SOAP betas (b1,b2) as comma-separated string. Default 0.9,0.999 "
                        "matches Wang Kolmogorov SOAP config.")
    p.add_argument("--soap-shampoo-beta", type=float, default=-1.0,
                   help="SOAP shampoo_beta (preconditioner moving average). -1 = use betas[1].")
    p.add_argument("--soap-eps", type=float, default=1e-8)
    p.add_argument("--soap-weight-decay", type=float, default=0.0,
                   help="SOAP weight decay; default 0 to match Adam baseline.")
    p.add_argument("--soap-precondition-frequency", type=int, default=10,
                   help="SOAP preconditioner eigendecomp update frequency (default 10).")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-csv", default="results/taylor_green_results.csv")
    p.add_argument("--tag", default="")
    p.add_argument("--track", action="store_true",
                   help="Periodic per-epoch eval (TKE at end-of-window) into a tracking CSV.")
    p.add_argument("--track-interval", type=int, default=1000)
    p.add_argument("--eval-grid", type=int, default=64,
                   help="Uniform grid resolution per axis for end-of-run TKE evaluation.")
    p.add_argument("--eval-times-per-window", type=int, default=5,
                   help="Number of t_local samples per window for the TKE(t) trajectory.")
    p.add_argument("--save-checkpoint-dir", default="",
                   help="If set, saves per-window state dicts to this directory.")
    return p.parse_args(argv)


# =============================================================================
# Problem definition
# =============================================================================
@dataclass
class TGVProblem:
    """3D Taylor-Green Vortex on [0, L]^3 × [0, T_total]."""
    L: float = 2 * math.pi
    Re: float = 500.0
    rho: float = 1.0
    T_total: float = 10.0

    @property
    def nu(self) -> float:
        return 1.0 / self.Re

    def initial_condition(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        """Taylor-Green IC at t=0. Inputs are (B,) tensors; returns (u, v, w, p) each (B,)."""
        u = torch.sin(x) * torch.cos(y) * torch.cos(z)
        v = -torch.cos(x) * torch.sin(y) * torch.cos(z)
        w = torch.zeros_like(x)
        p = (1.0 / 16.0) * (torch.cos(2 * x) + torch.cos(2 * y)) * (torch.cos(2 * z) + 2)
        return u, v, w, p


# =============================================================================
# Network: periodic Fourier-feature MLP
# =============================================================================
class TGVFourierMLP(nn.Module):
    """MLP whose (x,y,z) inputs are replaced by [sin(ωx), cos(ωx)] with ω=2π/L
    so periodicity in x/y/z is built into the architecture (matches PhysicsNeMo's
    FullyConnectedArch with periodicity={"x":(0,L),"y":(0,L),"z":(0,L)}).

    Input: tensor of shape (B, 4) representing (x, y, z, t_local).
    Output: tensor of shape (B, 4) representing (u, v, w, p).
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 6,
                 period: float = 2 * math.pi):
        super().__init__()
        self.period = period
        self.omega = 2.0 * math.pi / period  # 1.0 when period = 2π
        in_dim = 7  # sin/cos for x,y,z + raw t
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 4))
        self.net = nn.Sequential(*layers)

    def fourier_features(self, xyzt: torch.Tensor) -> torch.Tensor:
        x, y, z, t = xyzt[..., 0:1], xyzt[..., 1:2], xyzt[..., 2:3], xyzt[..., 3:4]
        w = self.omega
        return torch.cat([
            torch.sin(w * x), torch.cos(w * x),
            torch.sin(w * y), torch.cos(w * y),
            torch.sin(w * z), torch.cos(w * z),
            t,
        ], dim=-1)

    def forward(self, xyzt: torch.Tensor) -> torch.Tensor:
        return self.net(self.fourier_features(xyzt))


# =============================================================================
# Network: PirateNet (Phase 3) — PyTorch port of temp/jaxpi-pirate/jaxpi/archs.py:342
# =============================================================================
def _glorot_normal_zero_bias_(linear: nn.Linear) -> None:
    """Match Flax Dense default: kernel = glorot_normal, bias = zeros.

    PyTorch's nn.Linear default is Kaiming-uniform with uniform bias; the JAX
    reference uses Glorot (Xavier) normal kernel and zero bias. This helper
    aligns the init so that the PyTorch port behaves like the Flax reference
    at initialization.
    """
    nn.init.xavier_normal_(linear.weight)
    nn.init.zeros_(linear.bias)


class PIModifiedBottleneck(nn.Module):
    """One PirateNet residual block.

    Connectivity (mirroring the Flax PIModifiedBottleneck at
    ``temp/jaxpi-pirate/jaxpi/archs.py:240``):

        identity = x
        x = Tanh(fc1(x))                # (B, embedding_dim) -> (B, hidden_dim)
        x = x * u + (1 - x) * v         # gate-mix at hidden_dim
        x = Tanh(fc2(x))                # (B, hidden_dim) -> (B, hidden_dim)
        x = x * u + (1 - x) * v
        x = Tanh(fc3(x))                # (B, hidden_dim) -> (B, embedding_dim)
        x = alpha * x + (1 - alpha) * identity     # alpha learnable, init=nonlinearity

    The two gating tensors ``u, v`` come from the outer PirateNet body and have
    shape ``(B, hidden_dim)``. The bottleneck preserves the embedding-dim
    residual stream and applies a learnable ``alpha`` skip-blend; the default
    init ``alpha = nonlinearity = 0.0`` makes the block an exact identity at
    step 0 ("physics-informed init", Wang et al. 2024 §3).
    """

    def __init__(self, embedding_dim: int, hidden_dim: int,
                 nonlinearity: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embedding_dim)
        for layer in (self.fc1, self.fc2, self.fc3):
            _glorot_normal_zero_bias_(layer)
        self.alpha = nn.Parameter(torch.full((1,), float(nonlinearity)))

    def forward(self, x: torch.Tensor, u: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
        identity = x
        x = torch.tanh(self.fc1(x))
        x = x * u + (1.0 - x) * v
        x = torch.tanh(self.fc2(x))
        x = x * u + (1.0 - x) * v
        x = torch.tanh(self.fc3(x))
        x = self.alpha * x + (1.0 - self.alpha) * identity
        return x


class TGVPirateNet(nn.Module):
    """PirateNet for the TGV problem.

    Connectivity (mirroring Flax PirateNet at ``temp/jaxpi-pirate/jaxpi/archs.py:342``,
    using the periodic Fourier embedding only — no FourierEmbs/RFF, no
    weight_fact reparam; those are deferred to Phase 3.5 per CONTEXT.md §0.3.4):

        x = periodic_fourier(xyzt)                # (B, 7)
        u = Tanh(gate_u(x))                       # (B, hidden_dim)
        v = Tanh(gate_v(x))                       # (B, hidden_dim)
        for _ in range(num_layers):
            x = PIModifiedBottleneck(...)(x, u, v)   # (B, 7) preserved
        y = head(x)                               # (B, 4) -> (u, v, w, p)

    Input: ``(B, 4)`` tensor representing ``(x, y, z, t_local)``.
    Output: ``(B, 4)`` tensor representing ``(u, v, w, p)``.

    The 7-dim residual stream (sin/cos for x,y,z + raw t) is the deliberate
    consequence of "reuse the periodic Fourier embedding" per CONTEXT.md §0.3.1
    + the deferral of multi-scale Random Fourier Features (Phase 3.5). A
    future widening (--pirate-embed-dim with single-scale RFF) is planned.
    """

    EMBEDDING_DIM = 7  # 2 (sin/cos) per spatial axis * 3 + 1 raw t

    def __init__(self, hidden_dim: int = 256, num_layers: int = 3,
                 nonlinearity: float = 0.0, period: float = 2 * math.pi):
        super().__init__()
        self.period = period
        self.omega = 2.0 * math.pi / period
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity

        self.gate_u = nn.Linear(self.EMBEDDING_DIM, hidden_dim)
        self.gate_v = nn.Linear(self.EMBEDDING_DIM, hidden_dim)
        _glorot_normal_zero_bias_(self.gate_u)
        _glorot_normal_zero_bias_(self.gate_v)

        self.bottlenecks = nn.ModuleList([
            PIModifiedBottleneck(
                embedding_dim=self.EMBEDDING_DIM,
                hidden_dim=hidden_dim,
                nonlinearity=nonlinearity,
            )
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(self.EMBEDDING_DIM, 4)
        _glorot_normal_zero_bias_(self.head)

    def fourier_features(self, xyzt: torch.Tensor) -> torch.Tensor:
        x, y, z, t = xyzt[..., 0:1], xyzt[..., 1:2], xyzt[..., 2:3], xyzt[..., 3:4]
        w = self.omega
        return torch.cat([
            torch.sin(w * x), torch.cos(w * x),
            torch.sin(w * y), torch.cos(w * y),
            torch.sin(w * z), torch.cos(w * z),
            t,
        ], dim=-1)

    def forward(self, xyzt: torch.Tensor) -> torch.Tensor:
        x = self.fourier_features(xyzt)
        u = torch.tanh(self.gate_u(x))
        v = torch.tanh(self.gate_v(x))
        for block in self.bottlenecks:
            x = block(x, u, v)
        return self.head(x)


def make_model(args, problem) -> nn.Module:
    """Build a fresh model per --model. Used both for the trained model and
    the model_template handed to `evaluate_tke_trajectory`."""
    if args.model == "mlp":
        return TGVFourierMLP(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            period=problem.L,
        )
    if args.model == "pirate-net":
        return TGVPirateNet(
            hidden_dim=args.pirate_hidden_dim,
            num_layers=args.pirate_num_layers,
            nonlinearity=args.pirate_nonlinearity,
            period=problem.L,
        )
    raise ValueError(f"Unknown --model: {args.model!r}")


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# AD-based residual for incompressible NS in 3D + time
# =============================================================================
def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """First derivative dy/dx as a vector matching y, with create_graph=True."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True,
    )[0]


def autodiff_residual(
    model: nn.Module,
    xyzt: torch.Tensor,
    problem: TGVProblem,
    *,
    cs: float = 0.0,
    delta: float = 0.0,
    eps: float = 1e-12,
):
    """Plain-AD NS residual at interior points.

    xyzt: (B, 4) float tensor with requires_grad=True.
    Returns (continuity, mom_x, mom_y, mom_z), each (B, 1).

    When ``cs > 0.0`` the viscous term uses the Smagorinsky LES closure with
    effective viscosity ``nu_eff = nu_lam + (cs*delta)^2 * S_mag`` and the
    full stress-tensor form ``visc_i = d_j(nu_eff*(d_j u_i + d_i u_j))``.
    When ``cs == 0.0`` the laminar Phase 1 path runs unchanged so that the
    Phase 1 baseline is reproduced bit-for-bit (acceptance gate 5).
    """
    pred = model(xyzt)
    u = pred[..., 0:1]
    v = pred[..., 1:2]
    w = pred[..., 2:3]
    p = pred[..., 3:4]

    grad_u = _grad(u, xyzt)
    grad_v = _grad(v, xyzt)
    grad_w = _grad(w, xyzt)
    grad_p = _grad(p, xyzt)

    u_x, u_y, u_z, u_t = grad_u[..., 0:1], grad_u[..., 1:2], grad_u[..., 2:3], grad_u[..., 3:4]
    v_x, v_y, v_z, v_t = grad_v[..., 0:1], grad_v[..., 1:2], grad_v[..., 2:3], grad_v[..., 3:4]
    w_x, w_y, w_z, w_t = grad_w[..., 0:1], grad_w[..., 1:2], grad_w[..., 2:3], grad_w[..., 3:4]
    p_x, p_y, p_z = grad_p[..., 0:1], grad_p[..., 1:2], grad_p[..., 2:3]

    if cs > 0.0:
        # 3D Smagorinsky LES branch — exercises the sqrt(.) VJP rule.
        Sxx, Syy, Szz = u_x, v_y, w_z
        Sxy = 0.5 * (u_y + v_x)
        Sxz = 0.5 * (u_z + w_x)
        Syz = 0.5 * (v_z + w_y)
        S_mag = torch.sqrt(
            2.0 * (Sxx ** 2 + Syy ** 2 + Szz ** 2
                   + 2.0 * (Sxy ** 2 + Sxz ** 2 + Syz ** 2)) + eps
        )
        nu_lam = problem.nu
        rho = problem.rho
        nu_eff = nu_lam + (cs * delta) ** 2 * S_mag

        # Stress tensor q_ij = nu_eff * (d_j u_i + d_i u_j) = 2 * nu_eff * S_ij;
        # symmetric in (i,j) so we materialise only the 6 distinct entries.
        q_xx = 2.0 * nu_eff * Sxx
        q_yy = 2.0 * nu_eff * Syy
        q_zz = 2.0 * nu_eff * Szz
        q_xy = 2.0 * nu_eff * Sxy
        q_xz = 2.0 * nu_eff * Sxz
        q_yz = 2.0 * nu_eff * Syz

        grad_qxx = _grad(q_xx, xyzt)
        grad_qxy = _grad(q_xy, xyzt)
        grad_qxz = _grad(q_xz, xyzt)
        grad_qyy = _grad(q_yy, xyzt)
        grad_qyz = _grad(q_yz, xyzt)
        grad_qzz = _grad(q_zz, xyzt)

        visc_u = grad_qxx[..., 0:1] + grad_qxy[..., 1:2] + grad_qxz[..., 2:3]
        visc_v = grad_qxy[..., 0:1] + grad_qyy[..., 1:2] + grad_qyz[..., 2:3]
        visc_w = grad_qxz[..., 0:1] + grad_qyz[..., 1:2] + grad_qzz[..., 2:3]

        continuity = u_x + v_y + w_z
        mom_x = u_t + (u * u_x + v * u_y + w * u_z) + p_x / rho - visc_u
        mom_y = v_t + (u * v_x + v * v_y + w * v_z) + p_y / rho - visc_v
        mom_z = w_t + (u * w_x + v * w_y + w * w_z) + p_z / rho - visc_w
        return continuity, mom_x, mom_y, mom_z

    # Phase 1 laminar branch — unchanged so cs=0.0 is bit-equivalent to Phase 1.
    u_xx = _grad(u_x, xyzt)[..., 0:1]
    u_yy = _grad(u_y, xyzt)[..., 1:2]
    u_zz = _grad(u_z, xyzt)[..., 2:3]
    v_xx = _grad(v_x, xyzt)[..., 0:1]
    v_yy = _grad(v_y, xyzt)[..., 1:2]
    v_zz = _grad(v_z, xyzt)[..., 2:3]
    w_xx = _grad(w_x, xyzt)[..., 0:1]
    w_yy = _grad(w_y, xyzt)[..., 1:2]
    w_zz = _grad(w_z, xyzt)[..., 2:3]

    nu = problem.nu
    rho = problem.rho

    continuity = u_x + v_y + w_z
    mom_x = u_t + (u * u_x + v * u_y + w * u_z) + p_x / rho - nu * (u_xx + u_yy + u_zz)
    mom_y = v_t + (u * v_x + v * v_y + w * v_z) + p_y / rho - nu * (v_xx + v_yy + v_zz)
    mom_z = w_t + (u * w_x + v * w_y + w * w_z) + p_z / rho - nu * (w_xx + w_yy + w_zz)
    return continuity, mom_x, mom_y, mom_z


# =============================================================================
# Sampling helpers
# =============================================================================
def sample_interior(batch_size: int, problem: TGVProblem, window_size: float,
                    device: torch.device, generator: torch.Generator) -> torch.Tensor:
    """(B, 4) uniformly random in [0, L]^3 × [0, T_w]."""
    rand = torch.rand(batch_size, 4, device=device, generator=generator)
    rand[:, 0] *= problem.L
    rand[:, 1] *= problem.L
    rand[:, 2] *= problem.L
    rand[:, 3] *= window_size
    return rand


def sample_ic_xyz(batch_size: int, problem: TGVProblem,
                  device: torch.device, generator: torch.Generator) -> torch.Tensor:
    """(B, 3) uniformly random in [0, L]^3."""
    rand = torch.rand(batch_size, 3, device=device, generator=generator)
    return rand * problem.L


# =============================================================================
# Training: moving time-window scheme
# =============================================================================
@dataclass
class WindowSnapshot:
    """A frozen copy of the network at the end of one time window."""
    window_idx: int
    state_dict: dict


def deepcopy_model(model: nn.Module) -> nn.Module:
    snap = copy.deepcopy(model)
    for p in snap.parameters():
        p.requires_grad_(False)
    snap.eval()
    return snap


def _build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    *,
    soap_betas=(0.9, 0.999),
    soap_shampoo_beta: float = -1.0,
    soap_eps: float = 1e-8,
    soap_weight_decay: float = 0.0,
    soap_precondition_frequency: int = 10,
) -> torch.optim.Optimizer:
    """Construct the per-window optimizer based on --optimizer.

    'adam' is the Phase 1/2 default and bit-equivalent.
    'soap' instantiates the vendored Vyas et al. (2024) SOAP optimizer
    (see ``src/soap.py``).
    """
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_name == "soap":
        # Local import keeps Phase 1/2 path independent of soap.py.
        from soap import SOAP
        return SOAP(
            model.parameters(),
            lr=lr,
            betas=tuple(soap_betas),
            shampoo_beta=soap_shampoo_beta,
            eps=soap_eps,
            weight_decay=soap_weight_decay,
            precondition_frequency=soap_precondition_frequency,
        )
    raise ValueError(f"Unknown --optimizer: {optimizer_name!r}")


def train_one_window(
    model: nn.Module,
    prev_model: Optional[nn.Module],
    problem: TGVProblem,
    window_size: float,
    epochs: int,
    lr: float,
    lr_decay_rate: float,
    lr_decay_steps: int,
    ic_weight: float,
    batch_interior: int,
    batch_ic: int,
    device: torch.device,
    generator: torch.Generator,
    log_prefix: str,
    *,
    les_cs: float = 0.0,
    les_delta: float = 0.0,
    les_eps: float = 1e-12,
    causal_eps: float = 0.0,
    causal_chunks: int = 10,
    optimizer_name: str = "adam",
    soap_betas=(0.9, 0.999),
    soap_shampoo_beta: float = -1.0,
    soap_eps: float = 1e-8,
    soap_weight_decay: float = 0.0,
    soap_precondition_frequency: int = 10,
) -> dict:
    """Train one time window. Returns a small dict of stats.

    Phase 3 extensions (all default to Phase-1/2 behaviour):
    * ``optimizer_name`` selects 'adam' (default) or 'soap'.
    * ``causal_eps > 0.0`` enables the causal PDE loss with ``causal_chunks``
      temporal slices per epoch (port of ``CausalLossNorm`` at
      ``temp/physicsnemo/physicsnemo/sym/loss/loss.py:271``).
    """
    optimizer = _build_optimizer(
        model, optimizer_name, lr,
        soap_betas=soap_betas,
        soap_shampoo_beta=soap_shampoo_beta,
        soap_eps=soap_eps,
        soap_weight_decay=soap_weight_decay,
        soap_precondition_frequency=soap_precondition_frequency,
    )
    # Per-step gamma so that LR decays by lr_decay_rate every lr_decay_steps steps.
    gamma_per_step = lr_decay_rate ** (1.0 / lr_decay_steps)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_per_step)

    last_loss = float("nan")
    last_pde = float("nan")
    last_ic = float("nan")
    nan_seen = False
    causal_active = causal_eps > 0.0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # --- PDE residual on interior ---
        xyzt_int = sample_interior(batch_interior, problem, window_size, device, generator)
        if causal_active:
            # Sort by t_local ascending and trim to a multiple of causal_chunks
            # so reshape(causal_chunks, -1) gives temporally-contiguous slices.
            sort_idx = torch.argsort(xyzt_int[:, 3])
            n_keep = (xyzt_int.shape[0] // causal_chunks) * causal_chunks
            if n_keep == 0:
                raise ValueError(
                    f"batch_interior={xyzt_int.shape[0]} is smaller than "
                    f"causal_chunks={causal_chunks}; increase --batch-interior."
                )
            xyzt_int = xyzt_int[sort_idx[:n_keep]].contiguous()
        xyzt_int.requires_grad_(True)
        cont, mx, my, mz = autodiff_residual(
            model, xyzt_int, problem,
            cs=les_cs, delta=les_delta, eps=les_eps,
        )
        if causal_active:
            # Wang et al. 2022 ("Respecting causality is all you need...")
            # causal weighting per the paper's Eq. (10) and the
            # Wang/Perdikaris 2025 (arXiv:2507.08972) Kolmogorov-flow code at
            # ``temp/jaxpi-pirate/examples/kolmogorov_flow/models.py:91`` (res_and_w):
            #
            #   l_i  = mean of squared residual components within chunk i
            #   M    = strict-lower-triangular ones, so (M @ l)_i = sum_{j<i} l_j
            #   w_i  = stop_grad(exp(-eps * (M @ l)_i))           (w_0 = exp(0) = 1)
            #   loss = sum_i w_i * l_i
            #
            # We use mean per chunk (rather than the PhysicsNeMo
            # ``CausalLossNorm`` sum-per-chunk form) because at PirateNet's
            # alpha=0 identity init the chunk sums reach O(hundreds) and
            # ``exp(-eps * cumsum)`` underflows to 0 in fp32, producing 0/0
            # NaN. The mean-based formulation keeps chunk loss values O(1)
            # and is what Wang's PINN-paper-faithful JAX-pi reference uses.
            # The weighting is mathematically the original Wang et al. 2022
            # form (M-matrix prefix-strict-before), not the PhysicsNeMo
            # ``cumsum`` and ``w / w[0]`` variant.
            pointwise = (cont.pow(2) + mx.pow(2) + my.pow(2) + mz.pow(2)).reshape(-1)
            chunk_loss = pointwise.reshape(causal_chunks, -1).mean(dim=-1)
            with torch.no_grad():
                # Strict-prefix sum: prefix_sum[i] = sum_{j<i} chunk_loss[j].
                cs = torch.cumsum(chunk_loss, dim=0)
                prefix_sum = torch.cat([
                    torch.zeros(1, device=chunk_loss.device, dtype=chunk_loss.dtype),
                    cs[:-1],
                ])
                w_causal = torch.exp(-causal_eps * prefix_sum)
            loss_pde = (w_causal * chunk_loss).sum()
        else:
            loss_pde = (cont.pow(2).mean() + mx.pow(2).mean()
                        + my.pow(2).mean() + mz.pow(2).mean())

        # --- IC / window-match constraint at t_local=0 ---
        xyz_ic = sample_ic_xyz(batch_ic, problem, device, generator)
        t_zero = torch.zeros(batch_ic, 1, device=device)
        xyzt_ic = torch.cat([xyz_ic, t_zero], dim=-1)
        pred_ic = model(xyzt_ic)

        if prev_model is None:
            # Window 0: analytic Taylor-Green IC.
            u_t, v_t, w_t, p_t = problem.initial_condition(
                xyz_ic[:, 0:1], xyz_ic[:, 1:2], xyz_ic[:, 2:3]
            )
            ic_terms = (
                (pred_ic[..., 0:1] - u_t).pow(2).mean()
                + (pred_ic[..., 1:2] - v_t).pow(2).mean()
                + (pred_ic[..., 2:3] - w_t).pow(2).mean()
                + (pred_ic[..., 3:4] - p_t).pow(2).mean()
            )
        else:
            # Window k>=1: match previous window's terminal state at t_local=T_w.
            t_end = torch.full((batch_ic, 1), window_size, device=device)
            xyzt_prev = torch.cat([xyz_ic, t_end], dim=-1)
            with torch.no_grad():
                prev_pred = prev_model(xyzt_prev)
            ic_terms = (
                (pred_ic[..., 0:1] - prev_pred[..., 0:1]).pow(2).mean()
                + (pred_ic[..., 1:2] - prev_pred[..., 1:2]).pow(2).mean()
                + (pred_ic[..., 2:3] - prev_pred[..., 2:3]).pow(2).mean()
            )
            # Pressure not constrained across window boundaries (PhysicsNeMo
            # uses lambda_weighting={"u_prev_step_diff":100, "v_...", "w_..."} only).
        loss_ic = ic_weight * ic_terms

        loss = loss_pde + loss_ic

        if not torch.isfinite(loss):
            print(f"  {log_prefix} epoch {epoch+1}: NaN/Inf loss — stopping window early.")
            nan_seen = True
            break

        loss.backward()
        optimizer.step()
        scheduler.step()

        last_loss = float(loss.detach())
        last_pde = float(loss_pde.detach())
        last_ic = float(loss_ic.detach())

        if (epoch + 1) % LOG_INTERVAL == 0:
            print(f"  {log_prefix} epoch {epoch+1:>6d}  loss={last_loss:.4e}  "
                  f"pde={last_pde:.4e}  ic={last_ic:.4e}  lr={scheduler.get_last_lr()[0]:.2e}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    return {
        "epochs_run": epoch + 1 if not nan_seen else epoch,
        "wall_time_s": t_end - t_start,
        "final_loss": last_loss,
        "final_pde_loss": last_pde,
        "final_ic_loss": last_ic,
        "nan_seen": nan_seen,
    }


# =============================================================================
# Evaluation: TKE decay curve
# =============================================================================
def evaluate_tke_trajectory(
    snapshots: List[WindowSnapshot],
    model_template: nn.Module,
    problem: TGVProblem,
    window_size: float,
    eval_grid: int,
    eval_times_per_window: int,
    device: torch.device,
) -> List[tuple]:
    """Compute (t_global, TKE) pairs by sampling each window snapshot at
    eval_times_per_window equally spaced t_local values.
    """
    # Build evaluation grid (uniform, periodic-aware: exclude endpoint at L since
    # x=0 and x=L are identified).
    n = eval_grid
    axis = torch.linspace(0, problem.L, n + 1, device=device)[:-1]
    X, Y, Z = torch.meshgrid(axis, axis, axis, indexing="ij")
    xyz_flat = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
    n_points = xyz_flat.shape[0]
    out = []
    for snap in snapshots:
        eval_model = copy.deepcopy(model_template).to(device)
        eval_model.load_state_dict(snap.state_dict)
        eval_model.eval()
        # Sample t_local in [0, window_size]; first sample at 0 only on window 0
        # to avoid double-count at window boundaries.
        t_locals = torch.linspace(0, window_size, eval_times_per_window + 1, device=device)
        if snap.window_idx > 0:
            t_locals = t_locals[1:]  # skip t_local=0 (= prev window's t_local=T_w)
        for tl in t_locals:
            t_col = torch.full((n_points, 1), float(tl), device=device)
            xyzt = torch.cat([xyz_flat, t_col], dim=-1)
            with torch.no_grad():
                # Chunk if needed to avoid OOM at high eval_grid
                chunk = 100_000
                preds = []
                for i in range(0, n_points, chunk):
                    preds.append(eval_model(xyzt[i:i + chunk]))
                pred = torch.cat(preds, dim=0)
            u = pred[..., 0]
            v = pred[..., 1]
            w = pred[..., 2]
            tke = 0.5 * (u.pow(2) + v.pow(2) + w.pow(2)).mean().item()
            t_global = snap.window_idx * window_size + float(tl)
            out.append((t_global, tke))
    return out


# =============================================================================
# CSV append (mirrors lid_benchmark.py race-fix pattern)
# =============================================================================
TGV_CSV_COLUMNS = [
    "timestamp", "method", "model", "Re", "domain_length", "num_windows",
    "window_size", "epochs_per_window", "total_epochs",
    "lr", "lr_decay_rate", "lr_decay_steps", "ic_weight",
    "les_cs", "les_delta", "les_eps",
    "optimizer", "causal_eps", "causal_chunks",
    "batch_interior", "batch_ic", "hidden_dim", "num_layers", "n_params",
    "seed", "tag",
    "wall_time_s", "wall_time_min", "ms_per_epoch", "peak_gpu_memory_mb",
    "final_loss", "final_pde_loss", "final_ic_loss", "nan_windows",
    "tke_t_grid", "tke_values",
    "status", "device", "gpu_name", "pytorch_version",
]


def append_csv_row(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=TGV_CSV_COLUMNS)
            if os.fstat(f.fileno()).st_size == 0:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# =============================================================================
# Main
# =============================================================================
def _parse_soap_betas(s: str):
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"--soap-betas must be 'b1,b2'; got {s!r}")
    return float(parts[0]), float(parts[1])


def main():
    args = parse_args()

    soap_betas_tuple = _parse_soap_betas(args.soap_betas)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    problem = TGVProblem(
        L=args.domain_length,
        Re=args.re,
        rho=1.0,
        T_total=args.num_windows * args.window_size,
    )

    model = make_model(args, problem).to(device)
    n_params = num_params(model)
    print(f"TGV-{args.method} model={args.model} optimizer={args.optimizer} "
          f"Re={problem.Re} params={n_params:,} device={device} "
          f"causal_eps={args.causal_eps} causal_chunks={args.causal_chunks} "
          f"les_cs={args.les_cs}")

    snapshots: List[WindowSnapshot] = []
    prev_model: Optional[nn.Module] = None
    nan_windows = 0
    total_wall = 0.0
    final_loss = float("nan")
    final_pde_loss = float("nan")
    final_ic_loss = float("nan")

    overall_start = time.perf_counter()
    status = "OK"
    try:
        for k in range(args.num_windows):
            log_prefix = f"[window {k+1:>2d}/{args.num_windows}]"
            stats = train_one_window(
                model=model,
                prev_model=prev_model,
                problem=problem,
                window_size=args.window_size,
                epochs=args.epochs_per_window,
                lr=args.lr,
                lr_decay_rate=args.lr_decay_rate,
                lr_decay_steps=args.lr_decay_steps,
                ic_weight=args.ic_weight,
                batch_interior=args.batch_interior,
                batch_ic=args.batch_ic,
                device=device,
                generator=generator,
                log_prefix=log_prefix,
                les_cs=args.les_cs,
                les_delta=args.les_delta,
                les_eps=args.les_eps,
                causal_eps=args.causal_eps,
                causal_chunks=args.causal_chunks,
                optimizer_name=args.optimizer,
                soap_betas=soap_betas_tuple,
                soap_shampoo_beta=args.soap_shampoo_beta,
                soap_eps=args.soap_eps,
                soap_weight_decay=args.soap_weight_decay,
                soap_precondition_frequency=args.soap_precondition_frequency,
            )
            total_wall += stats["wall_time_s"]
            final_loss = stats["final_loss"]
            final_pde_loss = stats["final_pde_loss"]
            final_ic_loss = stats["final_ic_loss"]
            if stats["nan_seen"]:
                nan_windows += 1
                status = "NAN"
                break

            # Snapshot end-of-window state and freeze.
            snap = WindowSnapshot(
                window_idx=k,
                state_dict=copy.deepcopy(model.state_dict()),
            )
            snapshots.append(snap)
            prev_model = deepcopy_model(model)

            if args.save_checkpoint_dir:
                os.makedirs(args.save_checkpoint_dir, exist_ok=True)
                torch.save(snap.state_dict,
                           os.path.join(args.save_checkpoint_dir,
                                        f"window_{k:03d}.pt"))
    except Exception as exc:
        print(f"!! Training raised {type(exc).__name__}: {exc}", file=sys.stderr)
        status = "ERROR"
        raise
    finally:
        overall_wall = time.perf_counter() - overall_start
        total_epochs = sum(args.epochs_per_window for _ in snapshots) + (
            0 if status == "OK" else 0)
        ms_per_epoch = (total_wall / max(total_epochs, 1)) * 1000.0 if total_epochs else float("nan")

        # End-of-run TKE trajectory (skip if no snapshots yet)
        tke_t_str = ""
        tke_v_str = ""
        if snapshots:
            traj = evaluate_tke_trajectory(
                snapshots=snapshots,
                model_template=make_model(args, problem),
                problem=problem,
                window_size=args.window_size,
                eval_grid=args.eval_grid,
                eval_times_per_window=args.eval_times_per_window,
                device=device,
            )
            tke_t_str = ";".join(f"{t:.4f}" for t, _ in traj)
            tke_v_str = ";".join(f"{v:.6e}" for _, v in traj)
            print("\nTKE trajectory (t, TKE):")
            for t, v in traj:
                print(f"  t={t:.3f}  TKE={v:.5e}")

        peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
        gpu_name = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": args.method,
            "model": args.model,
            "Re": problem.Re,
            "domain_length": problem.L,
            "num_windows": args.num_windows,
            "window_size": args.window_size,
            "epochs_per_window": args.epochs_per_window,
            "total_epochs": total_epochs,
            "lr": args.lr,
            "lr_decay_rate": args.lr_decay_rate,
            "lr_decay_steps": args.lr_decay_steps,
            "ic_weight": args.ic_weight,
            "les_cs": args.les_cs,
            "les_delta": args.les_delta,
            "les_eps": args.les_eps,
            "optimizer": args.optimizer,
            "causal_eps": args.causal_eps,
            "causal_chunks": args.causal_chunks,
            "batch_interior": args.batch_interior,
            "batch_ic": args.batch_ic,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "n_params": n_params,
            "seed": args.seed,
            "tag": args.tag,
            "wall_time_s": round(total_wall, 3),
            "wall_time_min": round(total_wall / 60.0, 4),
            "ms_per_epoch": round(ms_per_epoch, 3),
            "peak_gpu_memory_mb": round(peak_mb, 1),
            "final_loss": final_loss,
            "final_pde_loss": final_pde_loss,
            "final_ic_loss": final_ic_loss,
            "nan_windows": nan_windows,
            "tke_t_grid": tke_t_str,
            "tke_values": tke_v_str,
            "status": status,
            "device": str(device),
            "gpu_name": gpu_name,
            "pytorch_version": torch.__version__,
        }
        append_csv_row(args.output_csv, row)
        print(f"\nWrote row to {args.output_csv} (status={status}, wall={total_wall/60:.2f} min).")


if __name__ == "__main__":
    main()
