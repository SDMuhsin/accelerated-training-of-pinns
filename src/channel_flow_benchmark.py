"""3D Turbulent Channel Flow PINN benchmark — Phase C1 (autodiff-MLP scaffold).

Wall-bounded turbulent flow at Re_tau = 590 (Moser-Kim-Mansour 1999 reference)
on the rectangular box [0, L_x] x [-h, +h] x [0, L_z] with periodic BCs in
(x, z) and no-slip walls at y = +/-h. Driven by a constant streamwise body
force f_x = u_tau^2 / h that maintains the friction Reynolds number.

Wall-units convention used throughout this file: u_tau = h = 1, so

    nu       = 1 / Re_tau                                  (kinematic viscosity)
    y+(y)    = (1 - |y|) * Re_tau                          (wall distance, wall units)
    U+(y)    = u(y) / u_tau = u(y)                         (mean velocity, wall units)
    forcing  = f_x = u_tau^2 / h = 1                       (constant body force)

The dimensionless domain size matches MKM99: L_x = 2*pi*h, L_z = pi*h.

Trained with the moving-time-window scheme used by `src/taylor_green_benchmark.py`
(K windows of length T_w each; window k>=1 matches window k-1 at t_local=T_w).

C1 scope: --method=autodiff, --model=mlp only.

Architecture: 6-dim periodic Fourier embedding [sin(om_x*x), cos(om_x*x), y,
sin(om_z*z), cos(om_z*z), t] with om_x = 2*pi/L_x, om_z = 2*pi/L_z. The y
dimension is bounded so it goes in raw; the MLP wraps its (u, v, w)
predictions in a no-slip-enforcing factor (1 - y^2) so walls are satisfied
exactly by construction.

Initial condition: Reichardt mean profile + small deterministic 3D
perturbation that vanishes at walls (turbulence-triggering, not div-free).

Validation: end-of-run mean U+(y+) compared to the universal log-law
"U+ = y+" (viscous sublayer) and "U+ = (1/0.41) ln(y+) + 5.2" (log layer).
MKM99 full-profile validation (urms+, vrms+, wrms+, uv+) is C2's gate.

Bit-equivalence regression gate (CONTEXT.md §0.6/§4.3 standard): seed=0,
defaults, 1 window, 50 epochs reproduces an exact mean_u_at_y_plus value
(recorded in `llmdocs/trackers/channel_flow_2026-05-08.md` after first
clean smoke).
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
from dataclasses import dataclass
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
        description="3D turbulent channel flow PINN benchmark (Phase C1 autodiff-MLP scaffold)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--method", default="autodiff",
                   choices=["autodiff"],
                   help="Gradient-engine method. C1 is autodiff-only; C3 will add "
                        "ropinn / chebyshev-pinn / sk-pinn / can-pinn-faithful / dtpinn / sage.")
    p.add_argument("--model", default="mlp", choices=["mlp", "pirate-net"],
                   help="Network architecture. mlp = ChannelFlowFourierMLP (6-dim periodic "
                        "Fourier embedding + Tanh MLP). pirate-net = ChannelFlowPirateNet "
                        "(Wang et al. 2024 PirateNet, port of src/taylor_green_benchmark.py "
                        "TGVPirateNet adjusted for the 6-dim channel-flow embedding and "
                        "hard no-slip wrap).")
    p.add_argument("--re-tau", type=float, default=590.0,
                   help="Friction Reynolds number Re_tau = u_tau h / nu. "
                        "590 is the MKM99 high-Re canonical case.")
    p.add_argument("--Lx", type=float, default=2.0 * math.pi,
                   help="Streamwise domain length (in units of channel half-height h). "
                        "Default 2*pi*h matches MKM99.")
    p.add_argument("--Lz", type=float, default=math.pi,
                   help="Spanwise domain length (in units of h). Default pi*h matches MKM99.")
    p.add_argument("--num-windows", type=int, default=16,
                   help="Number of sequential time windows.")
    p.add_argument("--window-size", type=float, default=1.0,
                   help="Length of each time window (T_w) in units of h/u_tau. "
                        "Total time = num_windows * T_w (wall units).")
    p.add_argument("--hidden-dim", type=int, default=256, help="MLP hidden width.")
    p.add_argument("--num-layers", type=int, default=6, help="MLP hidden depth (Linear+Tanh blocks).")
    p.add_argument("--epochs-per-window", type=int, default=2000,
                   help="Adam epochs per window. C1 dev-box smoke default 2000; "
                        "paperscale default 5000.")
    p.add_argument("--batch-interior", type=int, default=4096,
                   help="Interior collocation points per epoch.")
    p.add_argument("--batch-ic", type=int, default=4096,
                   help="Initial-condition / window-match points per epoch.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-decay-rate", type=float, default=0.95)
    p.add_argument("--lr-decay-steps", type=int, default=3000)
    p.add_argument("--ic-weight", type=float, default=100.0,
                   help="Loss weight on IC and previous-window-match terms.")
    p.add_argument("--y-stretch", type=float, default=2.5,
                   help="Tanh stretching parameter for wall-clustered y sampling. "
                        "y = tanh(alpha*xi)/tanh(alpha) with xi ~ U[-1,1]. "
                        "Set 0 to use uniform-in-y sampling.")
    p.add_argument("--ic-perturb-amp", type=float, default=0.1,
                   help="Amplitude of the deterministic IC perturbation that breaks "
                        "2D symmetry and seeds turbulence.")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-csv", default="results/channel_flow_results.csv")
    p.add_argument("--tag", default="")
    p.add_argument("--track", action="store_true",
                   help="Periodic per-epoch eval into a tracking CSV (not yet implemented in C1).")
    p.add_argument("--track-interval", type=int, default=1000)
    p.add_argument("--eval-nx", type=int, default=32,
                   help="Eval grid resolution in x (uniform).")
    p.add_argument("--eval-ny", type=int, default=64,
                   help="Eval grid resolution in y (CGL clustered to walls).")
    p.add_argument("--eval-nz", type=int, default=32,
                   help="Eval grid resolution in z (uniform).")
    p.add_argument("--eval-times-per-window", type=int, default=4,
                   help="Number of t_local samples per window for the mean-profile trajectory.")
    p.add_argument("--save-checkpoint-dir", default="",
                   help="If set, saves per-window state dicts to this directory.")
    # ----- PirateNet knobs (active only when --model=pirate-net). Defaults from
    #       Wang et al. 2024 Kolmogorov config; mirror the TGV port. -----
    p.add_argument("--pirate-num-layers", type=int, default=3,
                   help="PirateNet bottleneck count (default 3 per Wang Kolmogorov "
                        "config). Ignored when --model=mlp.")
    p.add_argument("--pirate-hidden-dim", type=int, default=256,
                   help="PirateNet hidden width inside each bottleneck. Ignored "
                        "when --model=mlp.")
    p.add_argument("--pirate-nonlinearity", type=float, default=0.0,
                   help="PirateNet alpha init (0.0 => identity init at start, the "
                        "'physics-informed init' trick from Wang et al. 2024 §3).")
    # ----- Causal-mean loss + SOAP optimizer (Phase B port from TGV).
    #       Defaults preserve the C1 vanilla-Adam path exactly so the locked
    #       MLP@590 bit-equivalence anchor (mean_u[y+=262.21]=27.75303) is
    #       reproduced by --optimizer=adam --causal-eps=0. -----
    p.add_argument("--causal-eps", type=float, default=0.0,
                   help="Wang et al. 2022 causal-loss eps. >0 enables the "
                        "per-chunk strict-prefix-cumsum weighting (mean per "
                        "chunk; matches the TGV Phase 3 formulation, NOT the "
                        "PhysicsNeMo CausalLossNorm sum-per-chunk + w/w[0] "
                        "form which fp32-NaNs at PirateNet's alpha=0 init). "
                        "0 = vanilla per-epoch mean.")
    p.add_argument("--causal-chunks", type=int, default=16,
                   help="Number of temporal chunks per epoch when "
                        "--causal-eps>0. Wang Kolmogorov config uses 16. "
                        "batch_interior must be divisible by causal_chunks. "
                        "Ignored when --causal-eps=0.")
    p.add_argument("--optimizer", default="adam", choices=["adam", "soap"],
                   help="Per-window optimizer. 'adam' is the C1 default and "
                        "bit-equivalent to existing channel-flow runs. 'soap' "
                        "instantiates the SOAP optimizer (Vyas et al. 2024) "
                        "for Phase B (vendored at src/soap.py).")
    p.add_argument("--soap-betas", type=str, default="0.9,0.999",
                   help="SOAP betas (b1,b2) as comma-separated string. "
                        "0.9,0.999 matches Wang Kolmogorov SOAP config.")
    p.add_argument("--soap-shampoo-beta", type=float, default=-1.0,
                   help="SOAP shampoo_beta (preconditioner moving average). "
                        "-1 = use betas[1].")
    p.add_argument("--soap-eps", type=float, default=1e-8)
    p.add_argument("--soap-weight-decay", type=float, default=0.0,
                   help="SOAP weight decay; default 0 to match Adam baseline.")
    p.add_argument("--soap-precondition-frequency", type=int, default=10,
                   help="SOAP preconditioner eigendecomp update frequency.")
    return p.parse_args(argv)


# =============================================================================
# Problem definition
# =============================================================================
@dataclass
class ChannelFlowProblem:
    """3D turbulent channel flow on [0, L_x] x [-h, +h] x [0, L_z] x [0, T_total].

    Wall-units convention (u_tau = h = 1):
        nu     = 1 / Re_tau
        y+(y)  = (1 - |y|) * Re_tau
        U+(y)  = u(y)
    """
    Lx: float = 2.0 * math.pi
    Lz: float = math.pi
    h: float = 1.0
    Re_tau: float = 590.0
    rho: float = 1.0
    T_total: float = 16.0
    f_x: float = 1.0  # constant streamwise body force = u_tau^2 / h = 1

    @property
    def nu(self) -> float:
        return 1.0 / self.Re_tau

    def y_plus(self, y: torch.Tensor) -> torch.Tensor:
        """Wall distance in wall units, valid for y in [-h, +h]. Returns same-shape tensor."""
        return (self.h - y.abs()) * self.Re_tau

    def reichardt_u_plus(self, y_plus: torch.Tensor) -> torch.Tensor:
        """Reichardt's law: empirical mean velocity profile valid through viscous
        sublayer, buffer, and log layer. Reichardt (1951); standard reference
        in the wall-turbulence literature.

        U+(y+) = (1/kappa) ln(1 + kappa y+)
                 + 7.8 [1 - exp(-y+/11) - (y+/11) exp(-0.33 y+)]

        At y+=0 returns 0 exactly. At y+ -> infinity asymptotes to
        (1/kappa) ln(y+) + 5.5 (close to but not identical to the standard
        log-law constant 5.2; Reichardt's blend is what produces the "and 5.5"
        rather than "5.2" — both are within the experimental scatter).
        """
        kappa = 0.41
        return (1.0 / kappa) * torch.log1p(kappa * y_plus) + 7.8 * (
            1.0 - torch.exp(-y_plus / 11.0) - (y_plus / 11.0) * torch.exp(-0.33 * y_plus)
        )

    def initial_condition(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                          perturb_amp: float = 0.1):
        """Channel-flow initial condition at t=0.

        u_0(x,y,z) = U_Reichardt(y+) + delta_u(x,y,z)
        v_0(x,y,z) = delta_v(x,y,z)
        w_0(x,y,z) = delta_w(x,y,z)
        p_0(x,y,z) = 0  (reference pressure; network finds its own)

        The perturbations vanish at the walls (factor (1 - y^2)) so that the
        hard-BC wrap on the network is consistent with the IC target.
        Perturbation harmonics are deliberately mismatched in (x, z) so the
        IC breaks both 2D streamwise and 2D spanwise symmetry — the standard
        numerical channel-flow trick that triggers transition to turbulence
        within a few flow-through times.

        Inputs are (B,) tensors; returns (u, v, w, p) each (B,).
        """
        y_plus = (self.h - y.abs()) * self.Re_tau
        u_mean = self.reichardt_u_plus(y_plus)
        wall_factor = 1.0 - (y / self.h) ** 2  # vanishes at y = +/- h
        omx = 2.0 * math.pi / self.Lx
        omz = 2.0 * math.pi / self.Lz
        du = perturb_amp * torch.sin(2.0 * omx * x) * torch.sin(4.0 * omz * z) * wall_factor
        dv = perturb_amp * torch.cos(3.0 * omx * x) * torch.cos(2.0 * omz * z) * wall_factor
        dw = perturb_amp * torch.sin(omx * x) * torch.cos(4.0 * omz * z) * wall_factor
        u = u_mean + du
        v = dv
        w = dw
        p = torch.zeros_like(x)
        return u, v, w, p


# =============================================================================
# Network: periodic-in-xz Fourier MLP with hard no-slip wrap
# =============================================================================
class ChannelFlowFourierMLP(nn.Module):
    """MLP with periodic Fourier embedding in (x, z) and raw (y, t).

    Input  : (B, 4) tensor (x, y, z, t_local).
    Output : (B, 4) tensor (u, v, w, p).

    The embedding is [sin(om_x*x), cos(om_x*x), y, sin(om_z*z), cos(om_z*z), t]
    (6 dims) with om_x = 2*pi/L_x, om_z = 2*pi/L_z, so the network is exactly
    periodic in x and z by construction.

    No-slip hard constraint: the (u, v, w) outputs are multiplied by
    (1 - (y/h)^2) so they vanish at y = +/- h exactly. Pressure is left raw
    (no Dirichlet BC at the walls; the unsteady wall-y momentum equation gives
    dp/dy = nu d2v/dy2 there, which the loss handles).
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 6,
                 Lx: float = 2.0 * math.pi, Lz: float = math.pi, h: float = 1.0):
        super().__init__()
        self.Lx = Lx
        self.Lz = Lz
        self.h = h
        self.omega_x = 2.0 * math.pi / Lx
        self.omega_z = 2.0 * math.pi / Lz
        in_dim = 6  # sin/cos x + raw y + sin/cos z + raw t
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 4))
        self.net = nn.Sequential(*layers)

    def fourier_features(self, xyzt: torch.Tensor) -> torch.Tensor:
        x = xyzt[..., 0:1]
        y = xyzt[..., 1:2]
        z = xyzt[..., 2:3]
        t = xyzt[..., 3:4]
        return torch.cat([
            torch.sin(self.omega_x * x), torch.cos(self.omega_x * x),
            y,
            torch.sin(self.omega_z * z), torch.cos(self.omega_z * z),
            t,
        ], dim=-1)

    def forward(self, xyzt: torch.Tensor) -> torch.Tensor:
        y = xyzt[..., 1:2]
        raw = self.net(self.fourier_features(xyzt))
        wall_factor = 1.0 - (y / self.h) ** 2
        u = raw[..., 0:1] * wall_factor
        v = raw[..., 1:2] * wall_factor
        w = raw[..., 2:3] * wall_factor
        p = raw[..., 3:4]
        return torch.cat([u, v, w, p], dim=-1)


# =============================================================================
# Network: PirateNet — port of `src/taylor_green_benchmark.py:TGVPirateNet`
# (which itself ports the Flax PirateNet at `temp/jaxpi-pirate/jaxpi/archs.py:342`,
# Apache-2.0). Adjusted for channel flow's 6-dim periodic-in-xz embedding and
# hard no-slip wrap on (u, v, w).
# =============================================================================
def _glorot_normal_zero_bias_(linear: nn.Linear) -> None:
    """Match the Flax PirateNet init: Xavier-normal weight + zero bias."""
    nn.init.xavier_normal_(linear.weight)
    nn.init.zeros_(linear.bias)


class PIModifiedBottleneck(nn.Module):
    """One PirateNet residual block.

    Connectivity (mirroring `src/taylor_green_benchmark.py:PIModifiedBottleneck`,
    which mirrors Flax's PIModifiedBottleneck at `temp/jaxpi-pirate/jaxpi/archs.py:240`):

        identity = x
        x = Tanh(fc1(x))                # (B, embedding_dim) -> (B, hidden_dim)
        x = x * u + (1 - x) * v         # gate-mix at hidden_dim
        x = Tanh(fc2(x))                # (B, hidden_dim) -> (B, hidden_dim)
        x = x * u + (1 - x) * v
        x = Tanh(fc3(x))                # (B, hidden_dim) -> (B, embedding_dim)
        x = alpha * x + (1 - alpha) * identity     # alpha learnable, init=nonlinearity

    With alpha=nonlinearity=0.0, every block is exact identity at step 0
    ("physics-informed init", Wang et al. 2024 §3).
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


class ChannelFlowPirateNet(nn.Module):
    """PirateNet for the 3D turbulent channel-flow problem.

    Connectivity (port of `src/taylor_green_benchmark.py:TGVPirateNet`, with
    the embedding swapped from TGV's 7-dim periodic-in-xyz to the 6-dim
    periodic-in-xz + raw-y + raw-t embedding used by ChannelFlowFourierMLP,
    and the head wrapped in the same hard no-slip factor (1 - (y/h)^2) on
    (u, v, w)):

        feat = periodic_fourier(xyzt)             # (B, 6)
        u = Tanh(gate_u(feat))                    # (B, hidden_dim)
        v = Tanh(gate_v(feat))                    # (B, hidden_dim)
        for _ in range(num_layers):
            feat = PIModifiedBottleneck(...)(feat, u, v)
        raw = head(feat)                          # (B, 4) -> (raw_u, raw_v, raw_w, raw_p)
        wall_factor = 1 - (y/h)^2
        return [raw_u * wall, raw_v * wall, raw_w * wall, raw_p]

    Input  : (B, 4) tensor (x, y, z, t_local).
    Output : (B, 4) tensor (u, v, w, p), with no-slip exact at y = +/- h.

    The 6-dim embedding [sin(om_x*x), cos(om_x*x), y, sin(om_z*z), cos(om_z*z), t]
    is identical to ChannelFlowFourierMLP's embedding so periodicity guarantees
    are bit-equivalent across architectures.
    """

    EMBEDDING_DIM = 6

    def __init__(self, hidden_dim: int = 256, num_layers: int = 3,
                 nonlinearity: float = 0.0,
                 Lx: float = 2.0 * math.pi, Lz: float = math.pi, h: float = 1.0):
        super().__init__()
        self.Lx = Lx
        self.Lz = Lz
        self.h = h
        self.omega_x = 2.0 * math.pi / Lx
        self.omega_z = 2.0 * math.pi / Lz
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
        x = xyzt[..., 0:1]
        y = xyzt[..., 1:2]
        z = xyzt[..., 2:3]
        t = xyzt[..., 3:4]
        return torch.cat([
            torch.sin(self.omega_x * x), torch.cos(self.omega_x * x),
            y,
            torch.sin(self.omega_z * z), torch.cos(self.omega_z * z),
            t,
        ], dim=-1)

    def forward(self, xyzt: torch.Tensor) -> torch.Tensor:
        y_coord = xyzt[..., 1:2]
        feat = self.fourier_features(xyzt)
        u = torch.tanh(self.gate_u(feat))
        v = torch.tanh(self.gate_v(feat))
        for block in self.bottlenecks:
            feat = block(feat, u, v)
        raw = self.head(feat)
        wall_factor = 1.0 - (y_coord / self.h) ** 2
        return torch.cat([
            raw[..., 0:1] * wall_factor,
            raw[..., 1:2] * wall_factor,
            raw[..., 2:3] * wall_factor,
            raw[..., 3:4],
        ], dim=-1)


def make_model(args, problem: ChannelFlowProblem) -> nn.Module:
    if args.model == "mlp":
        return ChannelFlowFourierMLP(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            Lx=problem.Lx,
            Lz=problem.Lz,
            h=problem.h,
        )
    if args.model == "pirate-net":
        return ChannelFlowPirateNet(
            hidden_dim=args.pirate_hidden_dim,
            num_layers=args.pirate_num_layers,
            nonlinearity=args.pirate_nonlinearity,
            Lx=problem.Lx,
            Lz=problem.Lz,
            h=problem.h,
        )
    raise ValueError(f"Unknown --model: {args.model!r}")


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# AD-based PDE residual: incompressible NS in 3D + time, with constant
# streamwise body force f_x.
# =============================================================================
def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True,
    )[0]


def autodiff_residual(model: nn.Module, xyzt: torch.Tensor,
                      problem: ChannelFlowProblem):
    """Plain-AD incompressible NS residual at interior points.

    xyzt: (B, 4) float tensor with requires_grad=True.
    Returns (continuity, mom_x, mom_y, mom_z), each (B, 1).

    The constant streamwise body force f_x = u_tau^2 / h drives the flow:
        mom_x = u_t + (u . grad)u + dp/dx/rho - nu Lap(u) - f_x.
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
    f_x = problem.f_x

    continuity = u_x + v_y + w_z
    mom_x = u_t + (u * u_x + v * u_y + w * u_z) + p_x / rho - nu * (u_xx + u_yy + u_zz) - f_x
    mom_y = v_t + (u * v_x + v * v_y + w * v_z) + p_y / rho - nu * (v_xx + v_yy + v_zz)
    mom_z = w_t + (u * w_x + v * w_y + w * w_z) + p_z / rho - nu * (w_xx + w_yy + w_zz)
    return continuity, mom_x, mom_y, mom_z


# =============================================================================
# Sampling: random interior points + IC points.
# =============================================================================
def _sample_y(batch_size: int, problem: ChannelFlowProblem,
              y_stretch: float, device: torch.device,
              generator: torch.Generator) -> torch.Tensor:
    """Sample y in [-h, +h]. If y_stretch > 0, uses
        y = h * tanh(y_stretch * xi) / tanh(y_stretch)
    with xi ~ U[-1, 1] so points cluster near the walls (where the viscous
    sublayer needs resolution).
    """
    if y_stretch > 0.0:
        xi = 2.0 * torch.rand(batch_size, 1, device=device, generator=generator) - 1.0
        return problem.h * torch.tanh(y_stretch * xi) / math.tanh(y_stretch)
    return (2.0 * torch.rand(batch_size, 1, device=device, generator=generator) - 1.0) * problem.h


def sample_interior(batch_size: int, problem: ChannelFlowProblem,
                    window_size: float, y_stretch: float,
                    device: torch.device,
                    generator: torch.Generator) -> torch.Tensor:
    """(B, 4) random in [0, L_x] x [-h, +h] x [0, L_z] x [0, T_w]."""
    rand_xz = torch.rand(batch_size, 3, device=device, generator=generator)
    x = rand_xz[:, 0:1] * problem.Lx
    z = rand_xz[:, 1:2] * problem.Lz
    t = rand_xz[:, 2:3] * window_size
    y = _sample_y(batch_size, problem, y_stretch, device, generator)
    return torch.cat([x, y, z, t], dim=-1)


def sample_ic_xyz(batch_size: int, problem: ChannelFlowProblem,
                  y_stretch: float, device: torch.device,
                  generator: torch.Generator) -> torch.Tensor:
    """(B, 3) random in [0, L_x] x [-h, +h] x [0, L_z]."""
    rand_xz = torch.rand(batch_size, 2, device=device, generator=generator)
    x = rand_xz[:, 0:1] * problem.Lx
    z = rand_xz[:, 1:2] * problem.Lz
    y = _sample_y(batch_size, problem, y_stretch, device, generator)
    return torch.cat([x, y, z], dim=-1)


# =============================================================================
# Training: moving time-window scheme (mirrors src/taylor_green_benchmark.py)
# =============================================================================
@dataclass
class WindowSnapshot:
    window_idx: int
    state_dict: dict


def deepcopy_model(model: nn.Module) -> nn.Module:
    snap = copy.deepcopy(model)
    for p in snap.parameters():
        p.requires_grad_(False)
    snap.eval()
    return snap


def _parse_soap_betas(s: str):
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"--soap-betas must be 'b1,b2'; got {s!r}")
    return float(parts[0]), float(parts[1])


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

    'adam' is the C1 default and is bit-equivalent to the existing
    channel-flow runs (no weight decay; identical to torch.optim.Adam(...,
    lr=lr)). 'soap' instantiates the vendored Vyas et al. (2024) SOAP
    optimizer (see ``src/soap.py``); SOAP weight decay is 0 by default to
    preserve a vanilla baseline, matching the TGV Phase 3 convention.

    Port of ``src/taylor_green_benchmark.py:_build_optimizer`` (lines
    1923-1960), simplified for C1 (no SK-PINN sub-clause).
    """
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_name == "soap":
        # Local import keeps the Adam path independent of soap.py.
        # Inject src/ on sys.path so the vendored module resolves regardless
        # of cwd (matches the TGV pattern at taylor_green_benchmark.py:1668).
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    problem: ChannelFlowProblem,
    window_size: float,
    epochs: int,
    lr: float,
    lr_decay_rate: float,
    lr_decay_steps: int,
    ic_weight: float,
    batch_interior: int,
    batch_ic: int,
    y_stretch: float,
    ic_perturb_amp: float,
    device: torch.device,
    generator: torch.Generator,
    log_prefix: str,
    *,
    causal_eps: float = 0.0,
    causal_chunks: int = 16,
    optimizer_name: str = "adam",
    soap_betas=(0.9, 0.999),
    soap_shampoo_beta: float = -1.0,
    soap_eps: float = 1e-8,
    soap_weight_decay: float = 0.0,
    soap_precondition_frequency: int = 10,
) -> dict:
    """Train one time window. Returns a small dict of stats.

    Phase B extensions (all default to the C1 vanilla-Adam behaviour, so the
    locked MLP@590 bit-equivalence anchor is reproduced byte-exact when
    ``optimizer_name='adam'`` and ``causal_eps=0.0``):

    * ``optimizer_name`` selects 'adam' (default) or 'soap'.
    * ``causal_eps > 0.0`` enables the Wang et al. 2022 causal-mean PDE loss
      with ``causal_chunks`` temporal slabs per epoch (port of
      ``src/taylor_green_benchmark.py:train_one_window`` at lines 2389-2419).
    """
    optimizer = _build_optimizer(
        model, optimizer_name, lr,
        soap_betas=soap_betas,
        soap_shampoo_beta=soap_shampoo_beta,
        soap_eps=soap_eps,
        soap_weight_decay=soap_weight_decay,
        soap_precondition_frequency=soap_precondition_frequency,
    )
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

        # ---- PDE residual on interior ----
        xyzt_int = sample_interior(batch_interior, problem, window_size,
                                   y_stretch, device, generator)
        if causal_active:
            # Sort by t so each chunk is a contiguous time slab. Truncate to
            # the largest multiple of causal_chunks <= batch_interior.
            t_col = xyzt_int[:, -1]
            sort_idx = torch.argsort(t_col)
            n_keep = (xyzt_int.shape[0] // causal_chunks) * causal_chunks
            if n_keep == 0:
                raise ValueError(
                    f"batch_interior={xyzt_int.shape[0]} is smaller than "
                    f"causal_chunks={causal_chunks}; increase --batch-interior."
                )
            xyzt_int = xyzt_int[sort_idx[:n_keep]].contiguous()
        xyzt_int.requires_grad_(True)
        cont, mx, my, mz = autodiff_residual(model, xyzt_int, problem)
        if causal_active:
            # Wang et al. 2022 ("Respecting causality is all you need...")
            # causal weighting per Eq. (10), matching the form used by
            # src/taylor_green_benchmark.py:2389-2419 (Wang/Perdikaris 2025
            # JAX-pi reference at temp/jaxpi-pirate/examples/kolmogorov_flow/
            # models.py:91 — `res_and_w`):
            #
            #   l_i  = mean of squared residual components within chunk i
            #   M    = strict-lower-triangular ones, so (M @ l)_i = sum_{j<i} l_j
            #   w_i  = stop_grad(exp(-eps * (M @ l)_i))           (w_0 = 1)
            #   loss = sum_i w_i * l_i
            #
            # Mean-per-chunk (not the PhysicsNeMo CausalLossNorm sum-per-chunk
            # + w/w[0] form) — at PirateNet's alpha=0 init the chunk sums are
            # O(hundreds) and exp(-eps*cumsum) underflows to 0 in fp32.
            pointwise = (cont.pow(2) + mx.pow(2) + my.pow(2) + mz.pow(2)).reshape(-1)
            chunk_loss = pointwise.reshape(causal_chunks, -1).mean(dim=-1)
            with torch.no_grad():
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

        # ---- IC / window-match constraint at t_local=0 ----
        xyz_ic = sample_ic_xyz(batch_ic, problem, y_stretch, device, generator)
        t_zero = torch.zeros(batch_ic, 1, device=device)
        xyzt_ic = torch.cat([xyz_ic, t_zero], dim=-1)
        pred_ic = model(xyzt_ic)

        if prev_model is None:
            u_t, v_t, w_t, p_t = problem.initial_condition(
                xyz_ic[:, 0:1], xyz_ic[:, 1:2], xyz_ic[:, 2:3],
                perturb_amp=ic_perturb_amp,
            )
            ic_terms = (
                (pred_ic[..., 0:1] - u_t).pow(2).mean()
                + (pred_ic[..., 1:2] - v_t).pow(2).mean()
                + (pred_ic[..., 2:3] - w_t).pow(2).mean()
                + (pred_ic[..., 3:4] - p_t).pow(2).mean()
            )
        else:
            t_end = torch.full((batch_ic, 1), window_size, device=device)
            xyzt_prev = torch.cat([xyz_ic, t_end], dim=-1)
            with torch.no_grad():
                prev_pred = prev_model(xyzt_prev)
            ic_terms = (
                (pred_ic[..., 0:1] - prev_pred[..., 0:1]).pow(2).mean()
                + (pred_ic[..., 1:2] - prev_pred[..., 1:2]).pow(2).mean()
                + (pred_ic[..., 2:3] - prev_pred[..., 2:3]).pow(2).mean()
            )
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
                  f"pde={last_pde:.4e}  ic={last_ic:.4e}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

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
# MKM99 reference profile loader. Picks the canonical chanN.{means,reystress}
# file closest to args.Re_tau from data_dir. Files downloaded from
# https://turbulence.oden.utexas.edu/data/MKM/chan{180,395,550,590,1000}/profiles/.
# Both means and reystress columns are identical across the canonical files;
# the actual Re_tau in each file differs slightly from the nominal label
# (chan590 -> Re_tau=587.19; chan180 -> Re_tau=178.12; etc.).
# =============================================================================
_MKM99_NOMINAL_RE_TAU = (180, 395, 550, 590, 1000)


def load_mkm99(data_dir: str = "data/mkm99", Re_tau: float = 590.0) -> Optional[dict]:
    """Returns {'y_plus': (N,), 'U_plus': (N,), 'urms': (N,), 'vrms': (N,),
                'wrms': (N,), 'uv': (N,), 'Re_tau_ref': int, 'Re_tau_actual': float}
    or None if no matching files are available.

    Picks the chanN file with N closest to the requested Re_tau, where N is in
    _MKM99_NOMINAL_RE_TAU. MKM99 stores R_uu, R_vv, R_ww as variances; urms =
    sqrt(R_uu) etc. R_uv is the Reynolds shear stress directly. Profiles cover
    the lower half of the channel (y/h in [0, 1], i.e. y+ in [0, Re_tau]); in
    our convention, |y|/h going from 0 (centerline) to 1 (wall) so y+ =
    (1-|y|)*Re_tau is identical to the MKM99 convention.
    """
    chosen = min(_MKM99_NOMINAL_RE_TAU, key=lambda n: abs(n - Re_tau))
    means_path = os.path.join(data_dir, f"chan{chosen}.means")
    rey_path = os.path.join(data_dir, f"chan{chosen}.reystress")
    if not (os.path.exists(means_path) and os.path.exists(rey_path)):
        return None
    m = np.loadtxt(means_path, comments="#")
    r = np.loadtxt(rey_path, comments="#")
    # means columns: y, y+, Umean, dUmean/dy, Wmean, dWmean/dy, Pmean
    # reystress columns: y, y+, R_uu, R_vv, R_ww, R_uv, R_uw, R_vw
    return {
        "y_plus": m[:, 1],
        "U_plus": m[:, 2],
        "urms": np.sqrt(np.maximum(r[:, 2], 0.0)),
        "vrms": np.sqrt(np.maximum(r[:, 3], 0.0)),
        "wrms": np.sqrt(np.maximum(r[:, 4], 0.0)),
        "uv": r[:, 5],
        "Re_tau_ref": int(chosen),
        "Re_tau_actual": float(m[:, 1].max()),
    }


def compare_to_mkm99(profile: dict, mkm: dict) -> dict:
    """Compute relative-mean errors of mean U+, urms, vrms, wrms, <uv> in the
    log layer (30 < y+ < 0.3 Re_tau) using piecewise-linear interpolation of
    the MKM99 reference onto our eval-grid y+ values. Returns a dict of
    metrics (one per quantity).
    """
    yp_pred = np.asarray(profile["y_plus"])
    # Mask: log layer only.
    Re_tau_eval = float(yp_pred.max())
    mask = (yp_pred > 30.0) & (yp_pred < 0.3 * Re_tau_eval)
    if not mask.any():
        return {k: float("nan") for k in
                ("u_mkm99_log", "urms_mkm99_log", "vrms_mkm99_log",
                 "wrms_mkm99_log", "uv_mkm99_log")}
    out = {}
    for name_pred, name_mkm, key in (
        ("mean_u", "U_plus", "u_mkm99_log"),
        ("urms", "urms", "urms_mkm99_log"),
        ("vrms", "vrms", "vrms_mkm99_log"),
        ("wrms", "wrms", "wrms_mkm99_log"),
        ("uv", "uv", "uv_mkm99_log"),
    ):
        ref_interp = np.interp(yp_pred[mask], mkm["y_plus"], mkm[name_mkm])
        pred_arr = np.asarray(profile[name_pred])[mask]
        denom = np.mean(np.abs(ref_interp)) + 1e-12
        out[key] = float(np.mean(np.abs(pred_arr - ref_interp)) / denom)
    return out


# =============================================================================
# Evaluation: mean U+(y+) profile averaged over (x, z, t-in-final-window).
# =============================================================================
def _cgl_grid(N: int, h: float, device: torch.device) -> torch.Tensor:
    """Chebyshev-Gauss-Lobatto grid on [-h, +h] with N+1 points, clustered
    toward y = +/- h. Standard formula y_j = -h cos(j pi / N).
    """
    j = torch.arange(N + 1, device=device, dtype=torch.float32)
    return -h * torch.cos(j * math.pi / N)


def evaluate_mean_profile(
    snapshots: List[WindowSnapshot],
    model_template: nn.Module,
    problem: ChannelFlowProblem,
    window_size: float,
    eval_nx: int, eval_ny: int, eval_nz: int,
    eval_times_per_window: int,
    device: torch.device,
) -> dict:
    """Average u(x,y,z,t) over (x, z, t-in-window) at fixed y on the eval grid.
    Returns mean U(y) and the corresponding y+ array for the FINAL window's
    final time slice (so the profile reflects the trained-up state, not the IC).

    Eval grid: uniform in (x, z), CGL-clustered in y.
    """
    if not snapshots:
        return {"y": None, "y_plus": None, "mean_u": None,
                "log_law_match_log_layer": float("nan"),
                "linear_match_sublayer": float("nan")}
    snap = snapshots[-1]
    eval_model = copy.deepcopy(model_template).to(device)
    eval_model.load_state_dict(snap.state_dict)
    eval_model.eval()

    # Uniform in x, z (periodic; exclude right endpoint to avoid double-count).
    x_axis = torch.linspace(0, problem.Lx, eval_nx + 1, device=device)[:-1]
    z_axis = torch.linspace(0, problem.Lz, eval_nz + 1, device=device)[:-1]
    y_axis = _cgl_grid(eval_ny, problem.h, device)  # (Ny+1,)
    # Time samples in the final window, biased to the end so we report a
    # late-time profile (transient should be over by then).
    t_axis = torch.linspace(
        0, window_size, eval_times_per_window + 1, device=device,
    )[1:]  # exclude t_local=0 (matches prev window's terminal)

    Ny_pts = y_axis.shape[0]
    mean_u = torch.zeros(Ny_pts, device=device)
    mean_v = torch.zeros(Ny_pts, device=device)
    mean_w = torch.zeros(Ny_pts, device=device)
    sq_u = torch.zeros(Ny_pts, device=device)
    sq_v = torch.zeros(Ny_pts, device=device)
    sq_w = torch.zeros(Ny_pts, device=device)
    sq_uv = torch.zeros(Ny_pts, device=device)
    n_avg = eval_nx * eval_nz * t_axis.shape[0]

    chunk_xz = 100_000
    with torch.no_grad():
        for j in range(Ny_pts):
            y_val = y_axis[j].item()
            X, Z = torch.meshgrid(x_axis, z_axis, indexing="ij")
            xyz_xz = torch.stack([X.flatten(), torch.full_like(X.flatten(), y_val),
                                  Z.flatten()], dim=-1)
            n_xz = xyz_xz.shape[0]
            su = sv = sw = 0.0
            su2 = sv2 = sw2 = suv = 0.0
            for tl in t_axis:
                t_col = torch.full((n_xz, 1), float(tl), device=device)
                xyzt = torch.cat([xyz_xz, t_col], dim=-1)
                preds = []
                for i in range(0, n_xz, chunk_xz):
                    preds.append(eval_model(xyzt[i:i + chunk_xz]))
                pred = torch.cat(preds, dim=0)
                u_p = pred[..., 0]
                v_p = pred[..., 1]
                w_p = pred[..., 2]
                su = su + u_p.sum().item()
                sv = sv + v_p.sum().item()
                sw = sw + w_p.sum().item()
                su2 = su2 + (u_p * u_p).sum().item()
                sv2 = sv2 + (v_p * v_p).sum().item()
                sw2 = sw2 + (w_p * w_p).sum().item()
                suv = suv + (u_p * v_p).sum().item()
            mean_u[j] = su / n_avg
            mean_v[j] = sv / n_avg
            mean_w[j] = sw / n_avg
            sq_u[j] = su2 / n_avg
            sq_v[j] = sv2 / n_avg
            sq_w[j] = sw2 / n_avg
            sq_uv[j] = suv / n_avg
    # Reynolds stresses about the (x, z, t)-mean: var = E[u^2] - E[u]^2.
    urms = (sq_u - mean_u * mean_u).clamp(min=0.0).sqrt()
    vrms = (sq_v - mean_v * mean_v).clamp(min=0.0).sqrt()
    wrms = (sq_w - mean_w * mean_w).clamp(min=0.0).sqrt()
    uv = sq_uv - mean_u * mean_v

    y_plus = (problem.h - y_axis.abs()) * problem.Re_tau

    # Log-law match (universal, dimensionless): U+ = (1/0.41) ln(y+) + 5.2 for 30 < y+ < 0.3 Re_tau.
    kappa, B = 0.41, 5.2
    mask_log = (y_plus > 30.0) & (y_plus < 0.3 * problem.Re_tau)
    if mask_log.any():
        u_plus_log_pred = mean_u[mask_log]
        u_plus_log_ref = (1.0 / kappa) * torch.log(y_plus[mask_log]) + B
        log_match = float((u_plus_log_pred - u_plus_log_ref).abs().mean()
                          / u_plus_log_ref.abs().mean())
    else:
        log_match = float("nan")

    # Sublayer match (universal): U+ = y+ for 0 < y+ < 5.
    mask_sub = (y_plus > 0.5) & (y_plus < 5.0)
    if mask_sub.any():
        sublayer_pred = mean_u[mask_sub]
        sublayer_ref = y_plus[mask_sub]
        sub_match = float((sublayer_pred - sublayer_ref).abs().mean()
                          / sublayer_ref.abs().mean())
    else:
        sub_match = float("nan")

    return {
        "y": y_axis.cpu().numpy(),
        "y_plus": y_plus.cpu().numpy(),
        "mean_u": mean_u.cpu().numpy(),
        "mean_v": mean_v.cpu().numpy(),
        "mean_w": mean_w.cpu().numpy(),
        "urms": urms.cpu().numpy(),
        "vrms": vrms.cpu().numpy(),
        "wrms": wrms.cpu().numpy(),
        "uv": uv.cpu().numpy(),
        "log_law_match_log_layer": log_match,
        "linear_match_sublayer": sub_match,
    }


# =============================================================================
# CSV append (mirrors lid_benchmark / taylor_green race-fix pattern)
# =============================================================================
CHANNEL_FLOW_CSV_COLUMNS = [
    "timestamp", "method", "model", "Re_tau", "Lx", "Lz", "num_windows",
    "window_size", "epochs_per_window", "total_epochs",
    "lr", "lr_decay_rate", "lr_decay_steps", "ic_weight",
    "y_stretch", "ic_perturb_amp",
    "batch_interior", "batch_ic", "hidden_dim", "num_layers", "n_params",
    "seed", "tag",
    "wall_time_s", "wall_time_min", "ms_per_epoch", "peak_gpu_memory_mb",
    "final_loss", "final_pde_loss", "final_ic_loss", "nan_windows",
    "y_plus_grid", "mean_u_profile",
    "urms_profile", "vrms_profile", "wrms_profile", "uv_profile",
    "log_law_match_log_layer", "linear_match_sublayer",
    "u_mkm99_log", "urms_mkm99_log", "vrms_mkm99_log", "wrms_mkm99_log", "uv_mkm99_log",
    "optimizer", "causal_eps", "causal_chunks", "soap_betas", "soap_precondition_frequency",
    "status", "device", "gpu_name", "pytorch_version",
]


def append_csv_row(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=CHANNEL_FLOW_CSV_COLUMNS)
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
def main():
    args = parse_args()

    soap_betas_tuple = _parse_soap_betas(args.soap_betas)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    problem = ChannelFlowProblem(
        Lx=args.Lx,
        Lz=args.Lz,
        h=1.0,
        Re_tau=args.re_tau,
        rho=1.0,
        T_total=args.num_windows * args.window_size,
    )

    model = make_model(args, problem).to(device)
    n_params = num_params(model)
    print(f"ChannelFlow-{args.method} model={args.model} "
          f"Re_tau={problem.Re_tau} Lx={problem.Lx:.4f} Lz={problem.Lz:.4f} "
          f"params={n_params:,} device={device}")

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
                y_stretch=args.y_stretch,
                ic_perturb_amp=args.ic_perturb_amp,
                device=device,
                generator=generator,
                log_prefix=log_prefix,
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
        total_epochs = sum(args.epochs_per_window for _ in snapshots)
        ms_per_epoch = (total_wall / max(total_epochs, 1)) * 1000.0 if total_epochs else float("nan")

        # Compute mean profile + log-law match metrics.
        profile = evaluate_mean_profile(
            snapshots=snapshots,
            model_template=make_model(args, problem),
            problem=problem,
            window_size=args.window_size,
            eval_nx=args.eval_nx,
            eval_ny=args.eval_ny,
            eval_nz=args.eval_nz,
            eval_times_per_window=args.eval_times_per_window,
            device=device,
        )
        if profile["y_plus"] is not None:
            yp_str = ";".join(f"{yp:.4f}" for yp in profile["y_plus"])
            mu_str = ";".join(f"{mu:.6e}" for mu in profile["mean_u"])
            urms_str = ";".join(f"{v:.6e}" for v in profile["urms"])
            vrms_str = ";".join(f"{v:.6e}" for v in profile["vrms"])
            wrms_str = ";".join(f"{v:.6e}" for v in profile["wrms"])
            uv_str = ";".join(f"{v:.6e}" for v in profile["uv"])
        else:
            yp_str = mu_str = urms_str = vrms_str = wrms_str = uv_str = ""
        log_match = profile["log_law_match_log_layer"]
        sub_match = profile["linear_match_sublayer"]

        # MKM99 reference comparison (if data files are present).
        mkm = load_mkm99(Re_tau=problem.Re_tau)
        mkm_metrics = compare_to_mkm99(profile, mkm) if (mkm is not None and snapshots) else {
            "u_mkm99_log": float("nan"), "urms_mkm99_log": float("nan"),
            "vrms_mkm99_log": float("nan"), "wrms_mkm99_log": float("nan"),
            "uv_mkm99_log": float("nan"),
        }

        if snapshots:
            print("\nmean U+(y+) and Reynolds stresses — final window:")
            yp_arr = profile["y_plus"]
            mu_arr = profile["mean_u"]
            ur_arr = profile["urms"]
            vr_arr = profile["vrms"]
            wr_arr = profile["wrms"]
            uv_arr = profile["uv"]
            step = max(1, len(yp_arr) // 12)
            print(f"  {'y+':>8}  {'U+':>8}  {'urms+':>7}  {'vrms+':>7}  {'wrms+':>7}  {'<uv>+':>8}")
            for i in range(0, len(yp_arr), step):
                print(f"  {yp_arr[i]:>8.3f}  {mu_arr[i]:>8.4f}  "
                      f"{ur_arr[i]:>7.4f}  {vr_arr[i]:>7.4f}  "
                      f"{wr_arr[i]:>7.4f}  {uv_arr[i]:>8.4f}")
            tke_max = float(0.5 * (profile["urms"] ** 2 + profile["vrms"] ** 2 + profile["wrms"] ** 2).max())
            print(f"\nlog-layer rel-mean-error vs (1/0.41) ln(y+) + 5.2 : "
                  f"{log_match:.4f}")
            print(f"sublayer  rel-mean-error vs y+                     : "
                  f"{sub_match:.4f}")
            if mkm is not None:
                mkm_tke_peak = float(0.5 * (mkm["urms"] ** 2 + mkm["vrms"] ** 2 + mkm["wrms"] ** 2).max())
                print(f"max <TKE>+ across y (turbulence indicator)         : "
                      f"{tke_max:.4f}    (MKM99 chan{mkm['Re_tau_ref']} peak ~ {mkm_tke_peak:.2f})")
            else:
                print(f"max <TKE>+ across y (turbulence indicator)         : "
                      f"{tke_max:.4f}")
            if mkm is not None:
                print(f"\nMKM99 chan{mkm['Re_tau_ref']} (Re_tau={mkm['Re_tau_actual']:.2f}) reference comparison "
                      f"(log layer 30 < y+ < {0.3 * problem.Re_tau:.0f}):")
                print(f"  rel-mean-error U+    vs MKM99 : {mkm_metrics['u_mkm99_log']:.4f}")
                print(f"  rel-mean-error urms+ vs MKM99 : {mkm_metrics['urms_mkm99_log']:.4f}")
                print(f"  rel-mean-error vrms+ vs MKM99 : {mkm_metrics['vrms_mkm99_log']:.4f}")
                print(f"  rel-mean-error wrms+ vs MKM99 : {mkm_metrics['wrms_mkm99_log']:.4f}")
                print(f"  rel-mean-error <uv>+ vs MKM99 : {mkm_metrics['uv_mkm99_log']:.4f}")
            else:
                print("\nMKM99 reference data not found at data/mkm99/ — skipping reference comparison.")

        peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
        gpu_name = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": args.method,
            "model": args.model,
            "Re_tau": problem.Re_tau,
            "Lx": problem.Lx,
            "Lz": problem.Lz,
            "num_windows": args.num_windows,
            "window_size": args.window_size,
            "epochs_per_window": args.epochs_per_window,
            "total_epochs": total_epochs,
            "lr": args.lr,
            "lr_decay_rate": args.lr_decay_rate,
            "lr_decay_steps": args.lr_decay_steps,
            "ic_weight": args.ic_weight,
            "y_stretch": args.y_stretch,
            "ic_perturb_amp": args.ic_perturb_amp,
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
            "y_plus_grid": yp_str,
            "mean_u_profile": mu_str,
            "urms_profile": urms_str,
            "vrms_profile": vrms_str,
            "wrms_profile": wrms_str,
            "uv_profile": uv_str,
            "log_law_match_log_layer": log_match,
            "linear_match_sublayer": sub_match,
            "u_mkm99_log": mkm_metrics["u_mkm99_log"],
            "urms_mkm99_log": mkm_metrics["urms_mkm99_log"],
            "vrms_mkm99_log": mkm_metrics["vrms_mkm99_log"],
            "wrms_mkm99_log": mkm_metrics["wrms_mkm99_log"],
            "uv_mkm99_log": mkm_metrics["uv_mkm99_log"],
            "optimizer": args.optimizer,
            "causal_eps": args.causal_eps,
            "causal_chunks": args.causal_chunks,
            "soap_betas": args.soap_betas,
            "soap_precondition_frequency": args.soap_precondition_frequency,
            "status": status,
            "device": str(device),
            "gpu_name": gpu_name,
            "pytorch_version": torch.__version__,
        }
        append_csv_row(args.output_csv, row)
        print(f"\nWrote row to {args.output_csv} (status={status}, wall={total_wall/60:.2f} min).")


if __name__ == "__main__":
    main()
