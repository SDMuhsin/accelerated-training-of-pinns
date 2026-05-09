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
    p.add_argument("--method", default="autodiff",
                   choices=["autodiff", "ropinn", "chebyshev-pinn", "sk-pinn",
                            "can-pinn-faithful", "dtpinn", "sage"],
                   help="Gradient-engine method. 'autodiff' is the Phase 1/2/3 plain-AD "
                        "path; the others mirror the 2D method roster in "
                        "src/lid_benchmark.py (RoPINN region perturbation, Spectral-AD "
                        "via 3D Fourier, SK-PINN sparse RKPM, CAN-PINN Taylor-expansion "
                        "stencils, DT-PINN RBF-FD + L-BFGS, SAGE traced auto-generated "
                        "backward).")
    p.add_argument("--model", default="mlp", choices=["mlp", "pirate-net", "tsa-pinn"],
                   help="Network architecture. 'mlp' is the Phase 1/2 Fourier MLP; "
                        "'pirate-net' enables Phase 3 PirateNet residual arch; "
                        "'tsa-pinn' is the trainable-sinusoidal-activation MLP of "
                        "Khademi (Comput. Phys. Commun. 2025). All three share the "
                        "same periodic Fourier embedding so periodicity is enforced "
                        "exactly across architectures (naming matches "
                        "src/lid_benchmark.py for sweep-system compatibility).")
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
    # ----- TSA-PINN knobs (defaults active only when --model=tsa-pinn). -----
    p.add_argument("--tsa-initial-freq", type=float, default=1.0,
                   help="Initial frequency for trainable sinusoidal activations "
                        "(Khademi 2025). Ignored unless --model=tsa-pinn.")
    p.add_argument("--tsa-reg-weight", type=float, default=1.0,
                   help="Weight on the Dynamic Slope Recovery (DSR) regularizer "
                        "L_reg = 1 / sum_i exp(mean(omega_i)). The original "
                        "Khademi 2025 formulation adds this term unweighted "
                        "(weight=1) to the total loss; set 0 to disable. "
                        "Ignored unless --model=tsa-pinn.")
    # ----- RoPINN knobs (defaults match src/lid_benchmark.py:1665-1668). -----
    p.add_argument("--ropinn-initial-region", type=float, default=1e-4,
                   help="RoPINN initial trust-region radius. Ignored unless "
                        "--method=ropinn.")
    p.add_argument("--ropinn-region-max", type=float, default=0.01,
                   help="RoPINN maximum trust-region radius. Ignored unless "
                        "--method=ropinn.")
    p.add_argument("--ropinn-past-iterations", type=int, default=10,
                   help="RoPINN past-iteration history for gradient-variance "
                        "computation. Ignored unless --method=ropinn.")
    # ----- Spectral-AD knobs (chebyshev-pinn 3D analog: 3D Fourier spectral). -----
    p.add_argument("--spectral-n", type=int, default=16,
                   help="Spatial grid resolution per axis (Nx=Ny=Nz=N) for the "
                        "Spectral-AD (chebyshev-pinn) method. Total spatial points = N**3. "
                        "Larger N improves spectral-derivative accuracy but cost grows as "
                        "N**3. Ignored unless --method=chebyshev-pinn.")
    p.add_argument("--spectral-k", type=int, default=4,
                   help="Number of time samples per epoch on the structured 4D grid "
                        "for Spectral-AD. Total collocation = N**3 * K. When --causal-eps>0 "
                        "the value of --spectral-k must equal --causal-chunks (each time "
                        "slice forms one causal chunk). Ignored unless --method=chebyshev-pinn.")
    # ----- CAN-PINN-faithful knobs (3D analog of pde_residuals_canpinn_cavity). -----
    p.add_argument("--canpinn-n", type=int, default=10,
                   help="Spatial grid resolution per axis for CAN-PINN-faithful. "
                        "dx=dy=dz=L/N so the 7-point stencil neighbors land on adjacent "
                        "grid points modulo periodic wrap. Ignored unless "
                        "--method=can-pinn-faithful.")
    p.add_argument("--canpinn-k", type=int, default=4,
                   help="Number of time samples per epoch for CAN-PINN-faithful. Total "
                        "centers = N**3 * K; total stencil evaluations = 7 * centers. "
                        "When --causal-eps>0, --canpinn-k must equal --causal-chunks. "
                        "Ignored unless --method=can-pinn-faithful.")
    # ----- SK-PINN knobs (3D analog of build_sk_data + train_sk_pinn). -----
    p.add_argument("--skpinn-n", type=int, default=12,
                   help="Spatial grid resolution per axis for SK-PINN. h=L/N is the "
                        "RKPM grid spacing; SPH cutoff radius = 2*1.4*h matches the "
                        "2D cubic-spline-radius scaling. Total grid = N**3, total "
                        "collocation = N**3 * K. Larger N improves O(h**2) accuracy "
                        "but memory grows as N**3 * max_neighbors. Ignored unless "
                        "--method=sk-pinn.")
    p.add_argument("--skpinn-k", type=int, default=4,
                   help="Number of time samples per epoch for SK-PINN. When "
                        "--causal-eps>0, --skpinn-k must equal --causal-chunks. "
                        "Ignored unless --method=sk-pinn.")
    p.add_argument("--skpinn-wd", type=float, default=-1.0,
                   help="SK-PINN weight decay. -1 (default) selects per-model defaults: "
                        "mlp=1e-4, tsa-pinn=5e-4, pirate-net=1e-3 (mirrors the 2D "
                        "_SK_PINN_WD lookup at src/lid_benchmark.py:77). 0.0 = matched "
                        "protocol (no model-specific regularization). Ignored unless "
                        "--method=sk-pinn.")
    p.add_argument("--skpinn-h-factor", type=float, default=1.4,
                   help="SK-PINN kernel smoothing length factor: h = h_factor * dx. "
                        "Default matches the 2D harness. Ignored unless --method=sk-pinn.")
    # ----- DT-PINN knobs (3D periodic analog of dt-pinn / RBF-FD). -----
    p.add_argument("--dtpinn-n", type=int, default=12,
                   help="Spatial grid resolution per axis for DT-PINN (uniform N**3 "
                        "periodic grid). Larger N → smaller h → tighter RBF-FD stencil "
                        "and lower truncation error, but build cost grows as O(N**3 * "
                        "stencil_size**3). Ignored unless --method=dtpinn.")
    p.add_argument("--dtpinn-k", type=int, default=4,
                   help="Number of time samples per epoch for DT-PINN. When "
                        "--causal-eps>0, --dtpinn-k must equal --causal-chunks. "
                        "Ignored unless --method=dtpinn.")
    p.add_argument("--dtpinn-p", type=int, default=2,
                   help="RBF-FD polynomial order (Sharma & Shankar 2022). p>=2; "
                        "p=2 → ell=2 (grad) / 3 (lap), reasonable accuracy at modest "
                        "stencil size. Ignored unless --method=dtpinn.")
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


# =============================================================================
# Network: TSA-PINN (trainable sinusoidal activations) — port of
# src/experiment_dt_elm_pinn/models/tsa_pinn.py for the TGV problem.
# Reference: Khademi, Comput. Phys. Commun. (May 2025); upstream code at
# https://github.com/AmirhosseinnnKhademi/TSA-PINN.
# =============================================================================
class TGVTsaPINN(nn.Module):
    """Trainable-sinusoidal-activation MLP for the TGV problem.

    Each hidden layer applies the Khademi (2025) activation
        h = 0.5 * (sin(omega * z + b) + cos(omega * z + b))
    where ``omega`` is a learnable per-neuron frequency parameter. The
    Dynamic Slope Recovery (DSR) regularizer
        L_reg = 1 / sum_i exp(mean(omega_i))
    penalises small frequencies and is exposed via ``regularization_loss``;
    the training loop adds it to the total loss when
    ``--tsa-reg-weight > 0`` (default 1.0, matching the Khademi reference).

    The network reuses the periodic Fourier embedding from ``TGVFourierMLP``
    so that periodicity in (x, y, z) is enforced by the input mapping —
    consistent with MLP and PirateNet for TGV. The 2D upstream consumes
    raw (x, y); for 3D periodic NS the embedding is the load-bearing piece
    that guarantees exact periodicity, so it stays.
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 6,
                 initial_freq: float = 1.0, period: float = 2 * math.pi):
        super().__init__()
        self.period = period
        self.omega_input = 2.0 * math.pi / period
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        in_dim = 7  # sin/cos for x,y,z + raw t (matches TGVFourierMLP)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.freqs = nn.ParameterList()
        for _ in range(num_layers):
            w = nn.Parameter(torch.empty(in_dim, hidden_dim))
            b = nn.Parameter(torch.zeros(1, hidden_dim))
            f = nn.Parameter(torch.full((1, hidden_dim), float(initial_freq)))
            nn.init.xavier_normal_(w)
            self.weights.append(w)
            self.biases.append(b)
            self.freqs.append(f)
            in_dim = hidden_dim

        self.output_weight = nn.Parameter(torch.empty(hidden_dim, 4))
        self.output_bias = nn.Parameter(torch.zeros(1, 4))
        nn.init.xavier_normal_(self.output_weight)

    def fourier_features(self, xyzt: torch.Tensor) -> torch.Tensor:
        x, y, z, t = xyzt[..., 0:1], xyzt[..., 1:2], xyzt[..., 2:3], xyzt[..., 3:4]
        w = self.omega_input
        return torch.cat([
            torch.sin(w * x), torch.cos(w * x),
            torch.sin(w * y), torch.cos(w * y),
            torch.sin(w * z), torch.cos(w * z),
            t,
        ], dim=-1)

    def forward(self, xyzt: torch.Tensor) -> torch.Tensor:
        h = self.fourier_features(xyzt)
        for w, b, f in zip(self.weights, self.biases, self.freqs):
            z = h @ w
            h = 0.5 * (torch.sin(f * z + b) + torch.cos(f * z + b))
        return h @ self.output_weight + self.output_bias

    def regularization_loss(self) -> torch.Tensor:
        return 1.0 / sum(torch.exp(freq.mean()) for freq in self.freqs)


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
    if args.model == "tsa-pinn":
        return TGVTsaPINN(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            initial_freq=args.tsa_initial_freq,
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
# Spectral-AD (chebyshev-pinn) — 3D Fourier-spectral spatial derivatives
# (period [0, L]^3) + AD on the temporal coordinate.
#
# The 2D chebyshev-pinn baseline in src/lid_benchmark.py uses dense Chebyshev
# differentiation matrices on a Chebyshev tensor-product grid. The natural 3D
# analog for a triply-periodic domain is the Fourier-spectral collocation
# method: derivatives are diagonal in Fourier space, computed via FFT/IFFT
# pairs. Time, which is non-periodic on [0, T_w], stays AD.
#
# torch.fft is differentiable, so the FFT-based residual still propagates
# gradients through model parameters via PyTorch autograd.
# =============================================================================
def build_spectral_grid(N: int, K: int, problem: TGVProblem,
                        window_size: float, device: torch.device):
    """Build the structured (Nx, Ny, Nz, K) 4D collocation grid for Spectral-AD.

    Returns:
        xyz_grid:   (N, N, N, 3) tensor of spatial coordinates.
        t_samples:  (K,) tensor of time samples in [0, T_w], excluding the
                    right endpoint (since the right endpoint of one window
                    is the left endpoint of the next under the moving-window
                    scheme; including it would double-count).
        kx, ky, kz: (N,) angular wavenumbers ``2*pi/L * fftfreq(N, d=L/N) * N``.
    """
    axis = torch.linspace(0, problem.L, N + 1, device=device)[:-1]
    X, Y, Z = torch.meshgrid(axis, axis, axis, indexing="ij")
    xyz_grid = torch.stack([X, Y, Z], dim=-1)  # (N, N, N, 3)

    if K <= 0:
        raise ValueError(f"--spectral-k must be >= 1; got {K}")
    t_samples = torch.linspace(0, window_size, K + 1, device=device)[:-1]

    freqs = 2.0 * math.pi * torch.fft.fftfreq(
        N, d=problem.L / N, device=device,
    )
    return xyz_grid, t_samples, freqs


def _spectral_grad(field_grid: torch.Tensor, k_dir: torch.Tensor) -> torch.Tensor:
    """First spatial derivative via FFT.

    field_grid: (N, N, N, K) real tensor on the structured grid.
    k_dir:      (N, N, N, 1) angular-wavenumber tensor for the desired axis
                (broadcastable over the K dimension).
    Returns:    (N, N, N, K) real tensor of partial derivative.
    """
    # Cast to complex for FFT; multiply by i*k in spectral space; IFFT, take real.
    f_hat = torch.fft.fftn(field_grid, dim=(0, 1, 2))
    d_hat = (1j * k_dir) * f_hat
    return torch.fft.ifftn(d_hat, dim=(0, 1, 2)).real


def spectral_residual(
    model: nn.Module,
    xyz_grid: torch.Tensor,
    t_samples: torch.Tensor,
    freqs: torch.Tensor,
    problem: TGVProblem,
    *,
    cs: float = 0.0,
    delta: float = 0.0,
    eps: float = 1e-12,
):
    """3D-Fourier-spectral spatial + AD-time NS residual on the structured grid.

    xyz_grid:   (N, N, N, 3) spatial mesh.
    t_samples:  (K,) time samples.
    freqs:      (N,) angular wavenumber array (from ``build_spectral_grid``).

    Returns ``(continuity, mom_x, mom_y, mom_z)`` each flattened to ``(N**3*K, 1)``,
    matching ``autodiff_residual``'s return shape so the existing causal-loss
    bookkeeping in ``train_one_window`` stays unchanged.
    """
    N = xyz_grid.shape[0]
    K = t_samples.shape[0]
    device = xyz_grid.device
    n_spatial = N * N * N

    # Build (K * N**3, 4) flattened (x, y, z, t) input.
    # Layout: outermost K (time slowest), then spatial — so reshape(K, N, N, N)
    # gives time-major slices we can later permute to (N, N, N, K) for FFT.
    xyz_flat = xyz_grid.reshape(-1, 3)                               # (N**3, 3)
    xyz_tiled = xyz_flat.unsqueeze(0).expand(K, -1, -1).reshape(-1, 3)  # (K*N**3, 3)
    t_tiled = t_samples.view(K, 1).expand(K, n_spatial).reshape(-1, 1)  # (K*N**3, 1)
    xyzt = torch.cat([xyz_tiled, t_tiled], dim=-1).contiguous()
    xyzt = xyzt.detach().requires_grad_(True)

    pred = model(xyzt)                                                # (K*N**3, 4)
    u_flat = pred[..., 0:1]
    v_flat = pred[..., 1:2]
    w_flat = pred[..., 2:3]
    p_flat = pred[..., 3:4]

    # Time derivatives via AD (last column of the flat-grad).
    u_t_flat = _grad(u_flat, xyzt)[..., 3:4]
    v_t_flat = _grad(v_flat, xyzt)[..., 3:4]
    w_t_flat = _grad(w_flat, xyzt)[..., 3:4]

    # Reshape ``(K*N**3, 1)`` -> ``(N, N, N, K)`` for FFT.
    def _to_grid(field_flat: torch.Tensor) -> torch.Tensor:
        return field_flat.view(K, N, N, N).permute(1, 2, 3, 0).contiguous()

    u_grid = _to_grid(u_flat); v_grid = _to_grid(v_flat)
    w_grid = _to_grid(w_flat); p_grid = _to_grid(p_flat)
    u_t_grid = _to_grid(u_t_flat); v_t_grid = _to_grid(v_t_flat); w_t_grid = _to_grid(w_t_flat)

    # Wavenumber tensors broadcastable over (N, N, N, K).
    KX = freqs.view(N, 1, 1, 1)
    KY = freqs.view(1, N, 1, 1)
    KZ = freqs.view(1, 1, N, 1)

    u_x = _spectral_grad(u_grid, KX); u_y = _spectral_grad(u_grid, KY); u_z = _spectral_grad(u_grid, KZ)
    v_x = _spectral_grad(v_grid, KX); v_y = _spectral_grad(v_grid, KY); v_z = _spectral_grad(v_grid, KZ)
    w_x = _spectral_grad(w_grid, KX); w_y = _spectral_grad(w_grid, KY); w_z = _spectral_grad(w_grid, KZ)
    p_x = _spectral_grad(p_grid, KX); p_y = _spectral_grad(p_grid, KY); p_z = _spectral_grad(p_grid, KZ)

    rho = problem.rho

    if cs > 0.0:
        # Smagorinsky LES in spectral space (full stress-tensor form, as in
        # autodiff_residual). Strain-rate from first derivatives, then
        # divergence of 2*nu_eff*S via further FFT-based derivatives.
        Sxx, Syy, Szz = u_x, v_y, w_z
        Sxy = 0.5 * (u_y + v_x)
        Sxz = 0.5 * (u_z + w_x)
        Syz = 0.5 * (v_z + w_y)
        S_mag = torch.sqrt(
            2.0 * (Sxx ** 2 + Syy ** 2 + Szz ** 2
                   + 2.0 * (Sxy ** 2 + Sxz ** 2 + Syz ** 2)) + eps
        )
        nu_lam = problem.nu
        nu_eff = nu_lam + (cs * delta) ** 2 * S_mag

        q_xx = 2.0 * nu_eff * Sxx; q_yy = 2.0 * nu_eff * Syy; q_zz = 2.0 * nu_eff * Szz
        q_xy = 2.0 * nu_eff * Sxy; q_xz = 2.0 * nu_eff * Sxz; q_yz = 2.0 * nu_eff * Syz

        visc_u = _spectral_grad(q_xx, KX) + _spectral_grad(q_xy, KY) + _spectral_grad(q_xz, KZ)
        visc_v = _spectral_grad(q_xy, KX) + _spectral_grad(q_yy, KY) + _spectral_grad(q_yz, KZ)
        visc_w = _spectral_grad(q_xz, KX) + _spectral_grad(q_yz, KY) + _spectral_grad(q_zz, KZ)
    else:
        # Laminar: Laplacian via spectral diagonal -k^2.
        K_SQ = KX ** 2 + KY ** 2 + KZ ** 2

        def _spectral_lap(field_grid: torch.Tensor) -> torch.Tensor:
            f_hat = torch.fft.fftn(field_grid, dim=(0, 1, 2))
            return torch.fft.ifftn(-K_SQ * f_hat, dim=(0, 1, 2)).real

        nu = problem.nu
        visc_u = nu * _spectral_lap(u_grid)
        visc_v = nu * _spectral_lap(v_grid)
        visc_w = nu * _spectral_lap(w_grid)

    continuity = u_x + v_y + w_z
    mom_x = u_t_grid + (u_grid * u_x + v_grid * u_y + w_grid * u_z) + p_x / rho - visc_u
    mom_y = v_t_grid + (u_grid * v_x + v_grid * v_y + w_grid * v_z) + p_y / rho - visc_v
    mom_z = w_t_grid + (u_grid * w_x + v_grid * w_y + w_grid * w_z) + p_z / rho - visc_w

    # Flatten back to (N**3 * K, 1) with the time-slowest layout the causal
    # loss expects (one chunk = one time slice when causal_chunks == K).
    def _to_flat(field_grid: torch.Tensor) -> torch.Tensor:
        return field_grid.permute(3, 0, 1, 2).contiguous().view(-1, 1)

    return _to_flat(continuity), _to_flat(mom_x), _to_flat(mom_y), _to_flat(mom_z)


# =============================================================================
# CAN-PINN-faithful — 3D port of pde_residuals_canpinn_cavity at
# src/lid_benchmark.py:792 (Chiu et al. 2022). The 2D version uses a 9-point
# stencil but only 5 (C, E/W, N/S) are actually consumed by the FD scheme;
# the 3D analog is a 7-point cross stencil (C, E/W, N/S, U/D), which
# generalizes the same formulas to a third spatial axis. Time is treated by
# autograd (model takes t as an input column).
#
# The scheme combines:
#   * CAN(uw2) upwind-biased Taylor reconstruction at face centers for the
#     convective fluxes,
#   * CAN(cd) central-difference + 1/8 dispersion correction for the
#     pressure gradient,
#   * plain 2nd-order central FD for the viscous Laplacian, and
#   * staggered-face divergence for continuity.
#
# Smagorinsky LES (cs > 0) reuses the AD-derived strain rate at C to compute
# nu_eff at the center; the FD Laplacian is then taken with that local nu_eff
# (matches the 2D harness drop-in design choice b1).
# =============================================================================
def _canpinn_stencil_offsets_3d(dx: float, dy: float, dz: float,
                                device: torch.device,
                                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """7-point 3D axis-aligned cross stencil offsets.

    Order: [C, E, W, N, S, U, D] with E=+dx, W=-dx, N=+dy, S=-dy, U=+dz, D=-dz.
    The t column is always 0 (the stencil is spatial only; time enters via AD).
    """
    return torch.tensor([
        [0.0, 0.0, 0.0, 0.0],   # C
        [+dx, 0.0, 0.0, 0.0],   # E
        [-dx, 0.0, 0.0, 0.0],   # W
        [0.0, +dy, 0.0, 0.0],   # N
        [0.0, -dy, 0.0, 0.0],   # S
        [0.0, 0.0, +dz, 0.0],   # U
        [0.0, 0.0, -dz, 0.0],   # D
    ], dtype=dtype, device=device)


def can_pinn_residual(
    model: nn.Module,
    xyzt_centers: torch.Tensor,
    dx: float, dy: float, dz: float,
    problem: TGVProblem,
    *,
    cs: float = 0.0,
    delta: float = 0.0,
    eps: float = 1e-12,
):
    """3D faithful CAN-PINN residual at the given center points.

    xyzt_centers: (N_int, 4) tensor of centers, assumed to lie on a uniform
    periodic grid with spacings ``dx, dy, dz``. Stencil neighbors are wrapped
    modulo ``problem.L`` along each spatial axis.

    Returns ``(R_continuity, R_mom_u, R_mom_v, R_mom_w)``, each ``(N_int, 1)``.
    """
    N_int = xyzt_centers.shape[0]
    device = xyzt_centers.device
    dtype = xyzt_centers.dtype
    L = problem.L

    offs = _canpinn_stencil_offsets_3d(dx, dy, dz, device, dtype)        # (7, 4)
    xyzt_stencil = xyzt_centers.unsqueeze(0) + offs.unsqueeze(1)         # (7, N_int, 4)
    xyzt_stencil = torch.cat([
        xyzt_stencil[..., 0:3] % L,
        xyzt_stencil[..., 3:4],
    ], dim=-1)
    xyzt_stencil = xyzt_stencil.reshape(7 * N_int, 4)
    xyzt_stencil = xyzt_stencil.detach().requires_grad_(True)

    pred = model(xyzt_stencil)                                           # (7*N_int, 4)
    u_all = pred[:, 0:1]; v_all = pred[:, 1:2]
    w_all = pred[:, 2:3]; p_all = pred[:, 3:4]

    ones_u = torch.ones_like(u_all)
    grad_u = torch.autograd.grad(u_all, xyzt_stencil, ones_u,
                                 create_graph=True, retain_graph=True)[0]
    grad_v = torch.autograd.grad(v_all, xyzt_stencil, ones_u,
                                 create_graph=True, retain_graph=True)[0]
    grad_w = torch.autograd.grad(w_all, xyzt_stencil, ones_u,
                                 create_graph=True, retain_graph=True)[0]
    grad_p = torch.autograd.grad(p_all, xyzt_stencil, ones_u,
                                 create_graph=True, retain_graph=True)[0]

    # Reshape to (7, N_int, 1).
    u_s = u_all.reshape(7, N_int, 1); v_s = v_all.reshape(7, N_int, 1)
    w_s = w_all.reshape(7, N_int, 1); p_s = p_all.reshape(7, N_int, 1)
    ux_s = grad_u[:, 0:1].reshape(7, N_int, 1)
    uy_s = grad_u[:, 1:2].reshape(7, N_int, 1)
    uz_s = grad_u[:, 2:3].reshape(7, N_int, 1)
    ut_s = grad_u[:, 3:4].reshape(7, N_int, 1)
    vx_s = grad_v[:, 0:1].reshape(7, N_int, 1)
    vy_s = grad_v[:, 1:2].reshape(7, N_int, 1)
    vz_s = grad_v[:, 2:3].reshape(7, N_int, 1)
    vt_s = grad_v[:, 3:4].reshape(7, N_int, 1)
    wx_s = grad_w[:, 0:1].reshape(7, N_int, 1)
    wy_s = grad_w[:, 1:2].reshape(7, N_int, 1)
    wz_s = grad_w[:, 2:3].reshape(7, N_int, 1)
    wt_s = grad_w[:, 3:4].reshape(7, N_int, 1)
    px_s = grad_p[:, 0:1].reshape(7, N_int, 1)
    py_s = grad_p[:, 1:2].reshape(7, N_int, 1)
    pz_s = grad_p[:, 2:3].reshape(7, N_int, 1)

    # Stencil index order: 0=C, 1=E, 2=W, 3=N, 4=S, 5=U, 6=D.
    u_C, u_E, u_W, u_N, u_S, u_U, u_D = (u_s[i] for i in range(7))
    v_C, v_E, v_W, v_N, v_S, v_U, v_D = (v_s[i] for i in range(7))
    w_C, w_E, w_W, w_N, w_S, w_U, w_D = (w_s[i] for i in range(7))
    p_C, p_E, p_W, p_N, p_S, p_U, p_D = (p_s[i] for i in range(7))

    # AD gradients we need (only at the relevant stencil indices).
    u_x_C, u_x_E, u_x_W = ux_s[0], ux_s[1], ux_s[2]
    u_y_C, u_y_N, u_y_S = uy_s[0], uy_s[3], uy_s[4]
    u_z_C, u_z_U, u_z_D = uz_s[0], uz_s[5], uz_s[6]
    u_t_C = ut_s[0]
    v_x_C, v_x_E, v_x_W = vx_s[0], vx_s[1], vx_s[2]
    v_y_C, v_y_N, v_y_S = vy_s[0], vy_s[3], vy_s[4]
    v_z_C, v_z_U, v_z_D = vz_s[0], vz_s[5], vz_s[6]
    v_t_C = vt_s[0]
    w_x_C, w_x_E, w_x_W = wx_s[0], wx_s[1], wx_s[2]
    w_y_C, w_y_N, w_y_S = wy_s[0], wy_s[3], wy_s[4]
    w_z_C, w_z_U, w_z_D = wz_s[0], wz_s[5], wz_s[6]
    w_t_C = wt_s[0]
    p_x_C, p_x_E, p_x_W = px_s[0], px_s[1], px_s[2]
    p_y_C, p_y_N, p_y_S = py_s[0], py_s[3], py_s[4]
    p_z_C, p_z_U, p_z_D = pz_s[0], pz_s[5], pz_s[6]

    # Face velocities (paper eq. 7).
    u_face_e = 0.5 * (u_E + u_C); u_face_w = 0.5 * (u_W + u_C)
    v_face_n = 0.5 * (v_N + v_C); v_face_s = 0.5 * (v_S + v_C)
    w_face_u = 0.5 * (w_U + w_C); w_face_d = 0.5 * (w_D + w_C)

    # CAN(uw2) Taylor-reconstructed face values for the convected variable
    # (paper eq. 8/9; notebook lines 444-476). Sign of the face velocity
    # selects the upwind state. /8 dispersion term is OFF here to match the
    # 2D upstream demo's commented-out form.
    half_dx = 0.5 * dx; half_dy = 0.5 * dy; half_dz = 0.5 * dz

    def _upwind_face(left_state, right_state, left_grad_at_left, right_grad_at_right,
                     face_velocity, h_half):
        """Generic upwind reconstruction at a face between left and right cells."""
        from_left = left_state + left_grad_at_left * h_half
        from_right = right_state - right_grad_at_right * h_half
        return torch.where(face_velocity >= 0.0, from_left, from_right)

    # x-direction face values for u, v, w.
    U_e = _upwind_face(u_C, u_E, u_x_C, u_x_E, u_face_e, half_dx)
    U_w = _upwind_face(u_W, u_C, u_x_W, u_x_C, u_face_w, half_dx)
    V_e = _upwind_face(v_C, v_E, v_x_C, v_x_E, u_face_e, half_dx)
    V_w = _upwind_face(v_W, v_C, v_x_W, v_x_C, u_face_w, half_dx)
    W_e = _upwind_face(w_C, w_E, w_x_C, w_x_E, u_face_e, half_dx)
    W_w = _upwind_face(w_W, w_C, w_x_W, w_x_C, u_face_w, half_dx)

    # y-direction face values.
    U_n = _upwind_face(u_C, u_N, u_y_C, u_y_N, v_face_n, half_dy)
    U_s = _upwind_face(u_S, u_C, u_y_S, u_y_C, v_face_s, half_dy)
    V_n = _upwind_face(v_C, v_N, v_y_C, v_y_N, v_face_n, half_dy)
    V_s = _upwind_face(v_S, v_C, v_y_S, v_y_C, v_face_s, half_dy)
    W_n = _upwind_face(w_C, w_N, w_y_C, w_y_N, v_face_n, half_dy)
    W_s = _upwind_face(w_S, w_C, w_y_S, w_y_C, v_face_s, half_dy)

    # z-direction face values.
    U_u = _upwind_face(u_C, u_U, u_z_C, u_z_U, w_face_u, half_dz)
    U_d = _upwind_face(u_D, u_C, u_z_D, u_z_C, w_face_d, half_dz)
    V_u = _upwind_face(v_C, v_U, v_z_C, v_z_U, w_face_u, half_dz)
    V_d = _upwind_face(v_D, v_C, v_z_D, v_z_C, w_face_d, half_dz)
    W_u = _upwind_face(w_C, w_U, w_z_C, w_z_U, w_face_u, half_dz)
    W_d = _upwind_face(w_D, w_C, w_z_D, w_z_C, w_face_d, half_dz)

    # Conservative-form convective fluxes (paper eq. 8/9):
    UU_x = (u_face_e * U_e - u_face_w * U_w) / dx
    VU_y = (v_face_n * U_n - v_face_s * U_s) / dy
    WU_z = (w_face_u * U_u - w_face_d * U_d) / dz
    UV_x = (u_face_e * V_e - u_face_w * V_w) / dx
    VV_y = (v_face_n * V_n - v_face_s * V_s) / dy
    WV_z = (w_face_u * V_u - w_face_d * V_d) / dz
    UW_x = (u_face_e * W_e - u_face_w * W_w) / dx
    VW_y = (v_face_n * W_n - v_face_s * W_s) / dy
    WW_z = (w_face_u * W_u - w_face_d * W_d) / dz

    # CAN(cd) pressure gradient (paper eq. 12/13). /8 dispersion correction ON.
    eighth_dx = dx / 8.0; eighth_dy = dy / 8.0; eighth_dz = dz / 8.0
    p_e = 0.5 * (p_C + p_E) - (p_x_E - p_x_C) * eighth_dx
    p_w = 0.5 * (p_W + p_C) - (p_x_C - p_x_W) * eighth_dx
    p_n = 0.5 * (p_C + p_N) - (p_y_N - p_y_C) * eighth_dy
    p_s = 0.5 * (p_S + p_C) - (p_y_C - p_y_S) * eighth_dy
    p_u = 0.5 * (p_C + p_U) - (p_z_U - p_z_C) * eighth_dz
    p_d = 0.5 * (p_D + p_C) - (p_z_C - p_z_D) * eighth_dz
    P_x = (p_e - p_w) / dx
    P_y = (p_n - p_s) / dy
    P_z = (p_u - p_d) / dz

    # Plain 2nd-order central FD for the viscous Laplacian.
    Uxx = (u_E - 2.0 * u_C + u_W) / (dx * dx)
    Uyy = (u_N - 2.0 * u_C + u_S) / (dy * dy)
    Uzz = (u_U - 2.0 * u_C + u_D) / (dz * dz)
    Vxx = (v_E - 2.0 * v_C + v_W) / (dx * dx)
    Vyy = (v_N - 2.0 * v_C + v_S) / (dy * dy)
    Vzz = (v_U - 2.0 * v_C + v_D) / (dz * dz)
    Wxx = (w_E - 2.0 * w_C + w_W) / (dx * dx)
    Wyy = (w_N - 2.0 * w_C + w_S) / (dy * dy)
    Wzz = (w_U - 2.0 * w_C + w_D) / (dz * dz)

    # Staggered-face divergence for continuity (notebook line 365).
    div = ((u_face_e - u_face_w) / dx
           + (v_face_n - v_face_s) / dy
           + (w_face_u - w_face_d) / dz)

    rho = problem.rho

    if cs > 0.0:
        # AD-derived strain rate at C → local nu_eff (matches 2D harness drop-in).
        Sxx_C = u_x_C; Syy_C = v_y_C; Szz_C = w_z_C
        Sxy_C = 0.5 * (u_y_C + v_x_C)
        Sxz_C = 0.5 * (u_z_C + w_x_C)
        Syz_C = 0.5 * (v_z_C + w_y_C)
        S_mag_C = torch.sqrt(
            2.0 * (Sxx_C ** 2 + Syy_C ** 2 + Szz_C ** 2
                   + 2.0 * (Sxy_C ** 2 + Sxz_C ** 2 + Syz_C ** 2)) + eps
        )
        nu_lam = problem.nu
        nu_eff = nu_lam + (cs * delta) ** 2 * S_mag_C
    else:
        nu_eff = problem.nu

    # Conservative-form momentum residual (paper eq. 14):
    # mom = U_t + (uU)_x + (vU)_y + (wU)_z - nu·(U_xx + U_yy + U_zz)
    #          - U·div + P_x.
    R_continuity = div
    R_mom_u = u_t_C + UU_x + VU_y + WU_z - nu_eff * (Uxx + Uyy + Uzz) - u_C * div + P_x
    R_mom_v = v_t_C + UV_x + VV_y + WV_z - nu_eff * (Vxx + Vyy + Vzz) - v_C * div + P_y
    R_mom_w = w_t_C + UW_x + VW_y + WW_z - nu_eff * (Wxx + Wyy + Wzz) - w_C * div + P_z

    return R_continuity, R_mom_u, R_mom_v, R_mom_w


def build_canpinn_centers(N: int, K: int, problem: TGVProblem,
                          window_size: float, device: torch.device):
    """Build the (N**3 * K, 4) center grid for CAN-PINN-faithful.

    Returns (xyzt_centers, dx) where dx = dy = dz = L/N. Layout is time-slowest
    (so reshape ``(K, N**3)`` gives one chunk per time slice when causal loss
    is on). Same grid is used every epoch within a window.
    """
    axis = torch.linspace(0, problem.L, N + 1, device=device)[:-1]
    X, Y, Z = torch.meshgrid(axis, axis, axis, indexing="ij")
    xyz_flat = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)  # (N**3, 3)
    t_samples = torch.linspace(0, window_size, K + 1, device=device)[:-1]
    xyzt_flat = torch.cat([
        xyz_flat.unsqueeze(0).expand(K, -1, -1).reshape(-1, 3),
        t_samples.view(K, 1).expand(K, N ** 3).reshape(-1, 1),
    ], dim=-1).contiguous()
    dx = float(problem.L) / N
    return xyzt_flat, dx


# =============================================================================
# Shared helpers for the discretized-spatial methods (SK-PINN / DT-PINN / SAGE).
#
# All three methods evaluate the model on a fixed uniform spatial grid at K
# time samples per epoch, then apply the same sparse spatial operators to
# every time slice. The naive form is a Python loop over the K slices, but
# each slice does its own model forward, its own AD time-grad calls, and K
# separate ``torch.sparse.mm`` calls per spatial operator — most of that
# wall-clock is Python / launch overhead rather than arithmetic.
#
# The two helpers below collapse the loop into one call each.
#   - ``_batched_xyzt_K`` builds the (K*N_all, 4) input once.
#   - ``_apply_sparse_op_K`` applies an (N_all, N_all) operator to a
#     time-slowest (K*N_all, 1) tensor in a single sparse mm.
# Mathematics is identical to the per-slice form (each row depends only on
# its own spatial neighbours), so the result is a fewer-launches optimisation,
# not a numerical change in the algorithm.
# =============================================================================
def _batched_xyzt_K(xyz_grid: torch.Tensor, t_samples: torch.Tensor) -> torch.Tensor:
    """Build the time-slowest ``(K*N_all, 4)`` input for a batched forward.

    ``xyz_grid``: ``(N_all, 3)`` spatial coordinates (no time column).
    ``t_samples``: ``(K,)`` time values, one per slice.

    The layout matches the per-slice loop: row ``k*N_all + i`` is
    ``(xyz_grid[i], t_samples[k])``. Returned tensor has ``requires_grad=True``
    so AD time-derivatives can be taken via ``torch.autograd.grad`` against
    column 3.
    """
    N_all = xyz_grid.shape[0]
    K = t_samples.shape[0]
    xyz_rep = xyz_grid.unsqueeze(0).expand(K, -1, -1).reshape(K * N_all, 3)
    t_rep = t_samples.view(K, 1, 1).expand(K, N_all, 1).reshape(K * N_all, 1)
    xyzt = torch.cat([xyz_rep, t_rep], dim=-1).detach()
    xyzt.requires_grad_(True)
    return xyzt


def _apply_sparse_op_K(D: torch.Tensor, x_kn1: torch.Tensor,
                       K: int, N_all: int) -> torch.Tensor:
    """Apply ``(N_all, N_all)`` sparse ``D`` to a time-slowest ``(K*N_all, 1)``
    tensor via a single ``(N_all, K)`` sparse mm.

    Equivalent to ``cat([D @ x_kn1[k*N_all:(k+1)*N_all] for k in range(K)])``
    but uses one CUDA launch instead of K. Output is also ``(K*N_all, 1)``
    in the same time-slowest layout.
    """
    x_NK = x_kn1.view(K, N_all).t().contiguous()             # (N_all, K)
    Dx_NK = torch.sparse.mm(D, x_NK)                         # (N_all, K)
    return Dx_NK.t().contiguous().view(K * N_all, 1)


# =============================================================================
# SK-PINN — 3D port of build_sk_data + train_sk_pinn at src/lid_benchmark.py:503.
# Reproducing-Kernel Particle Method (RKPM) with cubic-spline SPH kernel.
# Constructs sparse derivative matrices Dx, Dy, Dz, Dxx, Dyy, Dzz, Dxy, Dxz,
# Dyz over a uniform 3D periodic grid; spatial derivatives in the residual
# are then sparse matvecs, which is the same computational pattern as 2D.
# Time stays AD (per-time-slice forward + grad on t).
# =============================================================================
_SKPINN_TGV_WD = {
    "mlp": 1e-4,
    "tsa-pinn": 5e-4,
    "pirate-net": 1e-3,
}


def _skpinn_sph_kernel_3d(distances: torch.Tensor, h: float) -> torch.Tensor:
    """3D Monaghan cubic-spline SPH kernel.

    sigma_3d / h^3 normalisation with sigma_3d = 1/pi (vs 15/(7*pi*h^2) in 2D).
    Support radius is 2h; values vanish for q = r/h > 2.
    """
    q = distances / h
    sigma_3d = 1.0 / math.pi
    result = torch.zeros_like(distances, dtype=torch.float64)
    in_close = (q >= 0) & (q <= 1)
    in_far = (q > 1) & (q <= 2)
    if in_close.any():
        q_c = q[in_close]
        result[in_close] = (sigma_3d / h ** 3) * (1.0 - 1.5 * q_c ** 2 + 0.75 * q_c ** 3)
    if in_far.any():
        q_f = q[in_far]
        result[in_far] = (sigma_3d / h ** 3) * 0.25 * (2.0 - q_f) ** 3
    return result


def _skpinn_compute_C_3d(distance_vectors: torch.Tensor, kernel: torch.Tensor,
                         h_scale: float) -> torch.Tensor:
    """RKPM order-2 correction coefficients for 9 derivative operators.

    distance_vectors: (N_all, max_nb, 3) — (Δx, Δy, Δz) per neighbor.
    kernel:           (N_all, max_nb, 1) — per-neighbor SPH kernel value.
    h_scale:          dimensionalisation length (typically the grid spacing).

    Returns C: (N_all, max_nb, 9) — derivative weights per neighbor for the
    operator order (∂x, ∂y, ∂z, ∂²x, ∂²y, ∂²z, ∂xy, ∂xz, ∂yz).
    """
    dx_v = distance_vectors[:, :, 0:1]
    dy_v = distance_vectors[:, :, 1:2]
    dz_v = distance_vectors[:, :, 2:3]

    # Order-2 monomial basis: 1, x, y, z, x², xy, xz, y², yz, z² → 10 terms.
    # Each degree-i term is normalised by h_scale**i so the moment matrix is
    # dimensionless (matches the 2D scaling at src/lid_benchmark.py:481).
    moment_terms = [torch.ones_like(kernel)]
    moment_terms.append(dx_v / h_scale)
    moment_terms.append(dy_v / h_scale)
    moment_terms.append(dz_v / h_scale)
    moment_terms.append(dx_v * dx_v / (h_scale ** 2))
    moment_terms.append(dx_v * dy_v / (h_scale ** 2))
    moment_terms.append(dx_v * dz_v / (h_scale ** 2))
    moment_terms.append(dy_v * dy_v / (h_scale ** 2))
    moment_terms.append(dy_v * dz_v / (h_scale ** 2))
    moment_terms.append(dz_v * dz_v / (h_scale ** 2))
    moment_vector = torch.cat(moment_terms, dim=2)                          # (N_all, max_nb, 10)
    terms_num = moment_vector.shape[-1]

    # H selector: 9 derivative operators expressed in the moment basis. The
    # leading factor inverts the moment-vector normalisation so the resulting
    # weights have units of 1/length (first deriv) or 1/length² (second).
    H = torch.zeros((9, terms_num), dtype=torch.float64,
                    device=moment_vector.device)
    inv_h = 1.0 / h_scale
    inv_h2 = 1.0 / (h_scale ** 2)
    H[0, 1] = inv_h          # ∂x: select x-monomial
    H[1, 2] = inv_h          # ∂y
    H[2, 3] = inv_h          # ∂z
    H[3, 4] = 2.0 * inv_h2   # ∂²x: select x² with the 2 from the Taylor expansion
    H[4, 7] = 2.0 * inv_h2   # ∂²y: select y²
    H[5, 9] = 2.0 * inv_h2   # ∂²z: select z²
    H[6, 5] = inv_h2         # ∂xy: select xy
    H[7, 6] = inv_h2         # ∂xz: select xz
    H[8, 8] = inv_h2         # ∂yz: select yz

    # M = sum_j (m_j m_j^T) * w_j   — moment matrix per node, (N_all, 10, 10).
    matrix = torch.matmul(
        moment_vector.unsqueeze(3),                                          # (N_all, max_nb, 10, 1)
        moment_vector.unsqueeze(2),                                          # (N_all, max_nb, 1, 10)
    ) * kernel.unsqueeze(-1)                                                 # broadcast kernel
    matrix_sum = torch.sum(matrix, dim=1)                                    # (N_all, 10, 10)
    matrix_inverse = torch.inverse(matrix_sum)                               # (N_all, 10, 10)

    # C[i, j, k] = m_j[i] @ M_inv[i] @ H[k]^T.
    C = torch.matmul(
        torch.matmul(moment_vector, matrix_inverse),                          # (N_all, max_nb, 10)
        H.t().view(1, terms_num, 9),                                          # (1, 10, 9)
    )
    return C  # (N_all, max_nb, 9)


def _skpinn_uniform_periodic_neighbors_3d(N: int, L: float, h: float,
                                          radius: float):
    """Enumerate within-radius neighbors on a uniform N**3 periodic grid.

    Returns (neighborhoods, distances, distance_vectors) as torch.float64
    tensors. ``neighborhoods`` is the flat-index neighbor list per node;
    distance vectors are signed and use the minimum-image convention.
    Translation invariance + periodicity makes the offset list identical
    for every node.
    """
    k_max = math.ceil(radius / h)
    candidate_offsets: List[tuple] = []
    for ox in range(-k_max, k_max + 1):
        for oy in range(-k_max, k_max + 1):
            for oz in range(-k_max, k_max + 1):
                d2 = (ox * h) ** 2 + (oy * h) ** 2 + (oz * h) ** 2
                if d2 <= radius ** 2 + 1e-12:
                    candidate_offsets.append((ox, oy, oz, math.sqrt(d2)))
    max_nb = len(candidate_offsets)
    N_all = N ** 3

    # Vectorised flat-index arithmetic for periodic neighbors.
    ii, jj, kk = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    ii_flat = ii.flatten(); jj_flat = jj.flatten(); kk_flat = kk.flatten()

    neighborhoods = np.zeros((N_all, max_nb), dtype=np.int64)
    distance_vectors = np.zeros((N_all, max_nb, 3), dtype=np.float64)
    distances = np.zeros((N_all, max_nb), dtype=np.float64)

    for nb_idx, (ox, oy, oz, d) in enumerate(candidate_offsets):
        ni = (ii_flat + ox) % N
        nj = (jj_flat + oy) % N
        nk = (kk_flat + oz) % N
        neighborhoods[:, nb_idx] = ni * (N * N) + nj * N + nk
        distance_vectors[:, nb_idx, 0] = ox * h
        distance_vectors[:, nb_idx, 1] = oy * h
        distance_vectors[:, nb_idx, 2] = oz * h
        distances[:, nb_idx] = d

    return (
        torch.from_numpy(neighborhoods),
        torch.from_numpy(distances),
        torch.from_numpy(distance_vectors),
    )


def build_skpinn_data_3d(N: int, problem: TGVProblem, h_factor: float,
                         device: torch.device):
    """Assemble the 9 sparse derivative matrices for SK-PINN on a uniform
    N**3 periodic grid. Mirrors src/lid_benchmark.py:build_sk_data() in 3D.

    Returns a dict containing:
        xyz_grid:  (N**3, 3) flat node coordinates.
        Dx, Dy, Dz, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz: sparse (N**3, N**3) tensors.
        h, dx:     scalar grid spacings.
    """
    L = problem.L
    h_grid = L / N                              # uniform grid spacing
    h_kernel = h_grid * h_factor                # SPH smoothing length
    radius = 2.0 * h_kernel
    N_all = N ** 3

    print(f"  SK-PINN: building 3D RKPM matrices for {N}**3={N_all} uniform "
          f"periodic grid; h_grid={h_grid:.4f}, h_kernel={h_kernel:.4f}, "
          f"radius={radius:.4f}.")

    # Uniform [0, L)**3 grid (periodic; no endpoint).
    axis = np.linspace(0, L, N + 1)[:-1]
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    xyz_grid_np = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])

    # Periodic neighbor list (structured, exact for uniform grids).
    nb_t, dist_t, dvec_t = _skpinn_uniform_periodic_neighbors_3d(
        N, L, h_grid, radius,
    )
    max_nb = nb_t.shape[1]

    # SPH kernel + RKPM correction in fp64 for numerical conditioning.
    kernel = _skpinn_sph_kernel_3d(dist_t, h_kernel)             # (N_all, max_nb)
    C = _skpinn_compute_C_3d(dvec_t, kernel.unsqueeze(-1), h_grid)  # (N_all, max_nb, 9)

    # Assemble 9 sparse COO matrices on device.
    rows_arr = np.repeat(np.arange(N_all)[:, None], max_nb, axis=1).flatten()
    cols_arr = nb_t.numpy().flatten()
    indices = torch.tensor(np.stack([rows_arr, cols_arr]), dtype=torch.long, device=device)

    op_names = ["Dx", "Dy", "Dz", "Dxx", "Dyy", "Dzz", "Dxy", "Dxz", "Dyz"]
    matrices = {}
    kernel_flat = kernel.numpy()
    for k_op, name in enumerate(op_names):
        weights = C[:, :, k_op].numpy() * kernel_flat                # (N_all, max_nb)
        matrices[name] = torch.sparse_coo_tensor(
            indices,
            torch.tensor(weights.flatten(), dtype=torch.float32, device=device),
            (N_all, N_all),
        ).coalesce()

    print(f"  SK-PINN: 9 sparse matrices, max_nb={max_nb}, "
          f"nnz_per_op={N_all * max_nb}.")

    return {
        "xyz_grid": torch.tensor(xyz_grid_np, dtype=torch.float32, device=device),
        "h_grid": h_grid,
        "h_kernel": h_kernel,
        "N": N,
        "N_all": N_all,
        **matrices,
    }


def skpinn_residual(
    model: nn.Module,
    sk_data: dict,
    t_samples: torch.Tensor,
    problem: TGVProblem,
    *,
    cs: float = 0.0,
    delta: float = 0.0,
    eps: float = 1e-12,
):
    """3D SK-PINN NS+LES residual via sparse RKPM derivative matrices.

    Single batched forward + 3 AD time-grad calls + ``_apply_sparse_op_K``-
    batched RKPM matvecs. Returns ``(continuity, mom_x, mom_y, mom_z)`` each
    ``(N**3 * K, 1)`` with time-slowest layout.
    """
    N_all = sk_data["N_all"]
    K = t_samples.shape[0]
    rho = problem.rho

    Dx = sk_data["Dx"]; Dy = sk_data["Dy"]; Dz = sk_data["Dz"]
    Dxx = sk_data["Dxx"]; Dyy = sk_data["Dyy"]; Dzz = sk_data["Dzz"]

    xyzt = _batched_xyzt_K(sk_data["xyz_grid"], t_samples)
    pred = model(xyzt)                                              # (K*N_all, 4)
    u = pred[:, 0:1]; v = pred[:, 1:2]
    w = pred[:, 2:3]; p = pred[:, 3:4]

    ones_u = torch.ones_like(u)
    u_t = torch.autograd.grad(u, xyzt, ones_u, create_graph=True, retain_graph=True)[0][:, 3:4]
    v_t = torch.autograd.grad(v, xyzt, ones_u, create_graph=True, retain_graph=True)[0][:, 3:4]
    w_t = torch.autograd.grad(w, xyzt, ones_u, create_graph=True, retain_graph=True)[0][:, 3:4]

    u_x = _apply_sparse_op_K(Dx, u, K, N_all)
    u_y = _apply_sparse_op_K(Dy, u, K, N_all)
    u_z = _apply_sparse_op_K(Dz, u, K, N_all)
    v_x = _apply_sparse_op_K(Dx, v, K, N_all)
    v_y = _apply_sparse_op_K(Dy, v, K, N_all)
    v_z = _apply_sparse_op_K(Dz, v, K, N_all)
    w_x = _apply_sparse_op_K(Dx, w, K, N_all)
    w_y = _apply_sparse_op_K(Dy, w, K, N_all)
    w_z = _apply_sparse_op_K(Dz, w, K, N_all)
    p_x = _apply_sparse_op_K(Dx, p, K, N_all)
    p_y = _apply_sparse_op_K(Dy, p, K, N_all)
    p_z = _apply_sparse_op_K(Dz, p, K, N_all)

    if cs > 0.0:
        # Smagorinsky LES with full stress tensor (mirrors autodiff_residual).
        Sxx, Syy, Szz = u_x, v_y, w_z
        Sxy = 0.5 * (u_y + v_x)
        Sxz = 0.5 * (u_z + w_x)
        Syz = 0.5 * (v_z + w_y)
        S_mag = torch.sqrt(
            2.0 * (Sxx ** 2 + Syy ** 2 + Szz ** 2
                   + 2.0 * (Sxy ** 2 + Sxz ** 2 + Syz ** 2)) + eps
        )
        nu_lam = problem.nu
        nu_eff = nu_lam + (cs * delta) ** 2 * S_mag

        q_xx = 2.0 * nu_eff * Sxx; q_yy = 2.0 * nu_eff * Syy; q_zz = 2.0 * nu_eff * Szz
        q_xy = 2.0 * nu_eff * Sxy; q_xz = 2.0 * nu_eff * Sxz; q_yz = 2.0 * nu_eff * Syz

        visc_u = (_apply_sparse_op_K(Dx, q_xx, K, N_all)
                  + _apply_sparse_op_K(Dy, q_xy, K, N_all)
                  + _apply_sparse_op_K(Dz, q_xz, K, N_all))
        visc_v = (_apply_sparse_op_K(Dx, q_xy, K, N_all)
                  + _apply_sparse_op_K(Dy, q_yy, K, N_all)
                  + _apply_sparse_op_K(Dz, q_yz, K, N_all))
        visc_w = (_apply_sparse_op_K(Dx, q_xz, K, N_all)
                  + _apply_sparse_op_K(Dy, q_yz, K, N_all)
                  + _apply_sparse_op_K(Dz, q_zz, K, N_all))
    else:
        nu = problem.nu
        u_xx = _apply_sparse_op_K(Dxx, u, K, N_all)
        u_yy = _apply_sparse_op_K(Dyy, u, K, N_all)
        u_zz = _apply_sparse_op_K(Dzz, u, K, N_all)
        v_xx = _apply_sparse_op_K(Dxx, v, K, N_all)
        v_yy = _apply_sparse_op_K(Dyy, v, K, N_all)
        v_zz = _apply_sparse_op_K(Dzz, v, K, N_all)
        w_xx = _apply_sparse_op_K(Dxx, w, K, N_all)
        w_yy = _apply_sparse_op_K(Dyy, w, K, N_all)
        w_zz = _apply_sparse_op_K(Dzz, w, K, N_all)
        visc_u = nu * (u_xx + u_yy + u_zz)
        visc_v = nu * (v_xx + v_yy + v_zz)
        visc_w = nu * (w_xx + w_yy + w_zz)

    continuity = u_x + v_y + w_z
    mom_x = u_t + (u * u_x + v * u_y + w * u_z) + p_x / rho - visc_u
    mom_y = v_t + (u * v_x + v * v_y + w * v_z) + p_y / rho - visc_v
    mom_z = w_t + (u * w_x + v * w_y + w * w_z) + p_z / rho - visc_w
    return continuity, mom_x, mom_y, mom_z


# =============================================================================
# DT-PINN — 3D periodic port of Sharma & Shankar 2022 RBF-FD operators on
# scattered nodes. The 2D paper-faithful build at src/rbf_fd_operators.py
# uses Dirichlet boundary + ghost-node augmentation; for the triply-periodic
# TGV domain the boundary set is empty and neighbor search wraps modulo L
# via 27-image augmentation. Operators Dx/Dy/Dz/Dxx/Dyy/Dzz are returned as
# sparse PyTorch tensors and applied via sparse matvecs in the residual.
# =============================================================================
def build_dtpinn_operators_3d_periodic(N: int, problem: TGVProblem,
                                       p_order: int, device: torch.device):
    """Build sparse RBF-FD operators on a uniform N**3 periodic grid.

    Returns a dict with the precomputed grid + sparse derivative matrices,
    matching the structure used by ``build_skpinn_data_3d``.
    """
    # Local imports keep the Phase 1/2 path independent of the RBF-FD code.
    import scipy.linalg
    import scipy.sparse
    from scipy.spatial import cKDTree
    from rbf_fd_operators import (
        RBFFDParams, _phs, _phs_drbf_over_r,
        total_degree_indices, mpoly_eval, _legendre_recurrence, _EPS,
    )

    L = problem.L
    h = L / N
    Ni = N ** 3

    axis = np.linspace(0, L, N + 1)[:-1]
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    Xi = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])

    # 27-image augmentation for periodic neighbor search.
    img_offsets = np.array([
        [ox * L, oy * L, oz * L]
        for ox in (-1, 0, 1) for oy in (-1, 0, 1) for oz in (-1, 0, 1)
    ])  # (27, 3)
    augmented = (Xi[None] + img_offsets[:, None, :]).reshape(-1, 3)  # (27*Ni, 3)
    tree = cKDTree(augmented)

    params_grad = RBFFDParams.from_orders(s_dim=3, p=p_order, theta=1)
    params_lap = RBFFDParams.from_orders(s_dim=3, p=p_order, theta=2)
    n_grad = params_grad.stencil_size
    n_lap = params_lap.stencil_size
    n_query = max(n_grad, n_lap)

    print(f"  DT-PINN: building 3D periodic RBF-FD on {N}**3={Ni} nodes; "
          f"p={p_order}, h={h:.4f}, n_grad={n_grad}, n_lap={n_lap}.")

    _, idx_aug = tree.query(Xi, k=n_query)                                  # (Ni, n_query)
    primary_idx = idx_aug % Ni                                              # (Ni, n_query) primary block

    alpha_grad = total_degree_indices(3, params_grad.ell)
    alpha_lap = total_degree_indices(3, params_lap.ell)

    derivs_grad = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    derivs_lap = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
    all_derivs = derivs_grad + derivs_lap

    rows_acc = {d: [] for d in all_derivs}
    cols_acc = {d: [] for d in all_derivs}
    vals_acc = {d: [] for d in all_derivs}

    for ic in range(Ni):
        je_aug = idx_aug[ic]                                                 # (n_query,)
        stencil_xyz = augmented[je_aug]                                      # (n_query, 3) in image space

        # ---- Gradient operators (theta=1, scaled polynomial coords) ----
        sg = stencil_xyz[:n_grad]
        diffs_g = sg[:, None, :] - sg[None, :, :]
        rd_g = np.sqrt(np.maximum(np.sum(diffs_g * diffs_g, axis=2), 0.0))
        A_rbf_g = _phs(rd_g, params_grad.m)

        w_scale = rd_g[0, n_grad - 1]
        if w_scale <= 0:
            raise RuntimeError(f"degenerate gradient stencil at node {ic}")
        pc_g = (sg - sg[0]) / w_scale
        v_g = mpoly_eval(pc_g, alpha_grad, _legendre_recurrence)
        A_g = np.block([
            [A_rbf_g, v_g],
            [v_g.T, np.zeros((params_grad.poly_count, params_grad.poly_count))],
        ])
        try:
            lu_g = scipy.linalg.lu_factor(A_g)
        except Exception:
            lu_g = None

        D_phs_g = _phs_drbf_over_r(rd_g[0, :], params_grad.m)
        for d in derivs_grad:
            rhs = np.empty(n_grad + params_grad.poly_count, dtype=np.float64)
            ax = 0 if d == (1, 0, 0) else (1 if d == (0, 1, 0) else 2)
            d_comp = sg[0, ax] - sg[:, ax]
            rhs[:n_grad] = d_comp * D_phs_g
            d_p = mpoly_eval(pc_g[:1], alpha_grad, _legendre_recurrence,
                             deriv=list(d))[0] / w_scale
            rhs[n_grad:] = d_p
            if lu_g is not None:
                sol = scipy.linalg.lu_solve(lu_g, rhs)
            else:
                sol = np.linalg.solve(A_g, rhs)
            w = sol[:n_grad]
            rows_acc[d].append(np.full(n_grad, ic, dtype=np.int64))
            cols_acc[d].append(primary_idx[ic, :n_grad].astype(np.int64))
            vals_acc[d].append(w)

        # ---- Laplacian-component operators (theta=2, unscaled polynomial coords) ----
        sl = stencil_xyz[:n_lap]
        diffs_l = sl[:, None, :] - sl[None, :, :]
        rd_l = np.sqrt(np.maximum(np.sum(diffs_l * diffs_l, axis=2), 0.0))
        A_rbf_l = _phs(rd_l, params_lap.m)
        pc_l = sl - sl[0]
        v_l = mpoly_eval(pc_l, alpha_lap, _legendre_recurrence)
        A_l = np.block([
            [A_rbf_l, v_l],
            [v_l.T, np.zeros((params_lap.poly_count, params_lap.poly_count))],
        ])
        try:
            lu_l = scipy.linalg.lu_factor(A_l)
        except Exception:
            lu_l = None

        r_safe = rd_l[0, :] + _EPS
        t_a = params_lap.m * r_safe ** (params_lap.m - 2)
        t_b = params_lap.m * (params_lap.m - 2) * r_safe ** (params_lap.m - 4)

        for d in derivs_lap:
            rhs = np.empty(n_lap + params_lap.poly_count, dtype=np.float64)
            ax = 0 if d == (2, 0, 0) else (1 if d == (0, 2, 0) else 2)
            d_comp = sl[0, ax] - sl[:, ax]
            rhs[:n_lap] = t_a + t_b * (d_comp * d_comp)
            d_p = mpoly_eval(pc_l[:1], alpha_lap, _legendre_recurrence,
                             deriv=list(d))[0]
            rhs[n_lap:] = d_p
            if lu_l is not None:
                sol = scipy.linalg.lu_solve(lu_l, rhs)
            else:
                sol = np.linalg.solve(A_l, rhs)
            w = sol[:n_lap]
            rows_acc[d].append(np.full(n_lap, ic, dtype=np.int64))
            cols_acc[d].append(primary_idx[ic, :n_lap].astype(np.int64))
            vals_acc[d].append(w)

    # Assemble sparse PyTorch operators. coalesce() sums duplicate entries
    # which can arise when two periodic images of the same primary node both
    # appear in a stencil (rare for radius << L/2, but supported).
    op_names = {
        (1, 0, 0): "Dx", (0, 1, 0): "Dy", (0, 0, 1): "Dz",
        (2, 0, 0): "Dxx", (0, 2, 0): "Dyy", (0, 0, 2): "Dzz",
    }
    matrices = {}
    for d, name in op_names.items():
        rows = np.concatenate(rows_acc[d])
        cols = np.concatenate(cols_acc[d])
        vals = np.concatenate(vals_acc[d])
        indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long, device=device)
        matrices[name] = torch.sparse_coo_tensor(
            indices, torch.tensor(vals, dtype=torch.float32, device=device),
            (Ni, Ni),
        ).coalesce()

    print(f"  DT-PINN: assembled 6 sparse operators (Dx/Dy/Dz/Dxx/Dyy/Dzz).")

    return {
        "xyz_grid": torch.tensor(Xi, dtype=torch.float32, device=device),
        "h_grid": h,
        "N": N,
        "N_all": Ni,
        "p_order": p_order,
        **matrices,
    }


def _sage_compute_tgv_pde(pred, g, les_active: bool,
                          nu_lam: float, rho: float,
                          cs_delta_sq: float, eps_les: float):
    """Trace-friendly 3D NS+LES residual on a uniform periodic grid.

    Mirrors ``compute_pde_terms_sparse`` in the 2D SAGE pipeline but with
    7 input columns: the 4 model outputs (u, v, w, p) plus the 3 AD-derived
    time derivatives (u_t, v_t, w_t), which are passed in as additional
    columns of ``pred`` so SAGE can produce a single combined upstream and
    a downstream ``cat``-then-``backward`` propagates through both the
    spatial and temporal-AD branches in one call.

    The function is split-on-``les_active`` at trace time: each LES setting
    produces its own emitted backward (cached upstream).
    """
    Dx = g["Dx"]; Dy = g["Dy"]; Dz = g["Dz"]

    u = pred[:, 0:1]; v = pred[:, 1:2]
    w = pred[:, 2:3]; p = pred[:, 3:4]
    u_t = pred[:, 4:5]; v_t = pred[:, 5:6]; w_t = pred[:, 6:7]

    u_x = torch.sparse.mm(Dx, u); u_y = torch.sparse.mm(Dy, u); u_z = torch.sparse.mm(Dz, u)
    v_x = torch.sparse.mm(Dx, v); v_y = torch.sparse.mm(Dy, v); v_z = torch.sparse.mm(Dz, v)
    w_x = torch.sparse.mm(Dx, w); w_y = torch.sparse.mm(Dy, w); w_z = torch.sparse.mm(Dz, w)
    p_x = torch.sparse.mm(Dx, p); p_y = torch.sparse.mm(Dy, p); p_z = torch.sparse.mm(Dz, p)

    if les_active:
        Sxx, Syy, Szz = u_x, v_y, w_z
        Sxy = 0.5 * (u_y + v_x)
        Sxz = 0.5 * (u_z + w_x)
        Syz = 0.5 * (v_z + w_y)
        S_mag = torch.sqrt(
            2.0 * (Sxx ** 2 + Syy ** 2 + Szz ** 2
                   + 2.0 * (Sxy ** 2 + Sxz ** 2 + Syz ** 2)) + eps_les
        )
        nu_eff = nu_lam + cs_delta_sq * S_mag
        q_xx = 2.0 * nu_eff * Sxx; q_yy = 2.0 * nu_eff * Syy; q_zz = 2.0 * nu_eff * Szz
        q_xy = 2.0 * nu_eff * Sxy; q_xz = 2.0 * nu_eff * Sxz; q_yz = 2.0 * nu_eff * Syz
        visc_u = (torch.sparse.mm(Dx, q_xx) + torch.sparse.mm(Dy, q_xy)
                  + torch.sparse.mm(Dz, q_xz))
        visc_v = (torch.sparse.mm(Dx, q_xy) + torch.sparse.mm(Dy, q_yy)
                  + torch.sparse.mm(Dz, q_yz))
        visc_w = (torch.sparse.mm(Dx, q_xz) + torch.sparse.mm(Dy, q_yz)
                  + torch.sparse.mm(Dz, q_zz))
    else:
        Dxx = g["Dxx"]; Dyy = g["Dyy"]; Dzz = g["Dzz"]
        u_xx = torch.sparse.mm(Dxx, u); u_yy = torch.sparse.mm(Dyy, u); u_zz = torch.sparse.mm(Dzz, u)
        v_xx = torch.sparse.mm(Dxx, v); v_yy = torch.sparse.mm(Dyy, v); v_zz = torch.sparse.mm(Dzz, v)
        w_xx = torch.sparse.mm(Dxx, w); w_yy = torch.sparse.mm(Dyy, w); w_zz = torch.sparse.mm(Dzz, w)
        visc_u = nu_lam * (u_xx + u_yy + u_zz)
        visc_v = nu_lam * (v_xx + v_yy + v_zz)
        visc_w = nu_lam * (w_xx + w_yy + w_zz)

    inv_rho = 1.0 / rho
    continuity = u_x + v_y + w_z
    mom_x = u_t + (u * u_x + v * u_y + w * u_z) + inv_rho * p_x - visc_u
    mom_y = v_t + (u * v_x + v * v_y + w * v_z) + inv_rho * p_y - visc_v
    mom_z = w_t + (u * w_x + v * w_y + w * w_z) + inv_rho * p_z - visc_w
    return continuity, mom_x, mom_y, mom_z


_SAGE_BACKWARD_CACHE: dict = {}


def get_sage_backward_tgv(les_active: bool, nu_lam: float, rho: float,
                          cs_delta_sq: float, eps_les: float,
                          external_seeds: bool = True):
    """Lazy-trace + cache of the SAGE-emitted backward for TGV.

    The returned function has signature
        ``generated_backward(pred_det, g[, dc, dmx, dmy, dmz])``
    and produces the (N_all*K, 7) tensor of adjoints w.r.t.
    ``[u, v, w, p, u_t, v_t, w_t]``. The caller stacks the model outputs
    with the AD time derivatives, calls SAGE for the upstream, and then
    feeds it into ``combined.backward(upstream)`` so propagation through
    both the spatial and temporal-AD branches happens in a single call.

    ``external_seeds=False`` (the non-causal default in ``train_one_window``)
    has SAGE compute the residuals + ``2/M * mask`` seeds itself; the caller
    only supplies ``g`` containing ``M`` and ``interior_mask``. This mirrors
    the 2D ``train_sage`` pattern and avoids the redundant outside-residual
    pass.

    ``external_seeds=True`` keeps the per-chunk weighted seeding flow used
    by the causal-loss path: the caller computes residuals + per-chunk
    weights, builds explicit seeds, and hands them to the emitted backward
    as positional arguments.
    """
    # Local import keeps SAGE machinery off the bit-equivalence path.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from symbolic_vjp import trace_pde_forward, emit_backward

    key = (bool(les_active), float(nu_lam), float(rho),
           float(cs_delta_sq), float(eps_les), bool(external_seeds))
    if key in _SAGE_BACKWARD_CACHE:
        return _SAGE_BACKWARD_CACHE[key]

    tape: list = []
    constants = ["Dx", "Dy", "Dz"] + (
        [] if les_active else ["Dxx", "Dyy", "Dzz"]
    )
    input_names = ["u", "v", "w", "p", "u_t", "v_t", "w_t"]

    def _trace_fn(pred, g):
        return _sage_compute_tgv_pde(
            pred, g,
            les_active=les_active,
            nu_lam=nu_lam, rho=rho,
            cs_delta_sq=cs_delta_sq, eps_les=eps_les,
        )

    outputs, input_vars = trace_pde_forward(
        _trace_fn, None, tape, sparse=True,
        constants=constants, input_names=input_names,
    )
    seed_names = ["dc", "dmx", "dmy", "dmz"]
    source, fn = emit_backward(
        tape, list(outputs), seed_names, input_vars,
        sparse=True, input_names=input_names,
        external_seeds=bool(external_seeds),
    )
    _SAGE_BACKWARD_CACHE[key] = (source, fn)
    return source, fn


def _block_diagonal_sparse(M: torch.Tensor, K: int) -> torch.Tensor:
    """Tile a sparse (n, n) operator into a block-diagonal (K*n, K*n).

    Used by SAGE so that a single emitted backward call can process all
    K time slices of the stacked combined tensor in one pass — applying
    the spatial sparse operator independently within each block.
    """
    M = M.coalesce()
    n = M.shape[0]
    indices = M.indices(); values = M.values()
    rows_blocks, cols_blocks = [], []
    for k in range(K):
        rows_blocks.append(indices[0] + k * n)
        cols_blocks.append(indices[1] + k * n)
    new_rows = torch.cat(rows_blocks)
    new_cols = torch.cat(cols_blocks)
    new_vals = values.repeat(K)
    new_indices = torch.stack([new_rows, new_cols])
    return torch.sparse_coo_tensor(
        new_indices, new_vals, (K * n, K * n),
        device=M.device,
    ).coalesce()


def sage_compute_combined(model: nn.Module, dt_data: dict,
                          t_samples: torch.Tensor) -> torch.Tensor:
    """Build the ``(K*N_all, 7)`` SAGE input tensor with AD graph.

    Single batched forward through the model + 3 batched AD time-grad calls
    (one per ``u``/``v``/``w`` across all K time slices). Columns are
    ``[u, v, w, p, u_t, v_t, w_t]`` in time-slowest layout.

    The spatial branch (sparse matvecs + LES closure) is NOT computed here;
    SAGE's emitted backward owns that path so we don't pay for it twice.
    The caller passes ``combined.detach()`` into the emitted backward, then
    routes the returned upstream into ``combined.backward(gradient=upstream)``
    to propagate ``∂loss_pde/∂model_params`` through both the spatial and
    temporal-AD subgraphs in a single PyTorch backward.
    """
    xyzt = _batched_xyzt_K(dt_data["xyz_grid"], t_samples)
    pred = model(xyzt)                                          # (K*N_all, 4)
    u = pred[:, 0:1]
    ones_u = torch.ones_like(u)
    u_t = torch.autograd.grad(u, xyzt, ones_u, create_graph=True, retain_graph=True)[0][:, 3:4]
    v_t = torch.autograd.grad(pred[:, 1:2], xyzt, ones_u, create_graph=True, retain_graph=True)[0][:, 3:4]
    w_t = torch.autograd.grad(pred[:, 2:3], xyzt, ones_u, create_graph=True, retain_graph=True)[0][:, 3:4]
    return torch.cat([pred, u_t, v_t, w_t], dim=1)              # (K*N_all, 7)


def _sage_compute_tgv_residuals_eval(combined_det: torch.Tensor,
                                     sage_g: dict,
                                     les_active: bool,
                                     nu_lam: float, rho: float,
                                     cs_delta_sq: float, eps_les: float):
    """Evaluate ``(continuity, mom_x, mom_y, mom_z)`` in ``no_grad`` mode.

    Used by the causal-loss path (to build per-chunk weighted seeds) and the
    LOG_INTERVAL printer. Mirrors what SAGE does internally — sharing
    ``_sage_compute_tgv_pde`` keeps the two consistent.
    """
    with torch.no_grad():
        return _sage_compute_tgv_pde(
            combined_det, sage_g,
            les_active=les_active,
            nu_lam=nu_lam, rho=rho,
            cs_delta_sq=cs_delta_sq, eps_les=eps_les,
        )


def dtpinn_residual(
    model: nn.Module,
    dt_data: dict,
    t_samples: torch.Tensor,
    problem: TGVProblem,
    *,
    cs: float = 0.0,
    delta: float = 0.0,
    eps: float = 1e-12,
):
    """3D DT-PINN NS+LES residual via sparse RBF-FD matvecs + AD time.

    Single batched forward + 3 AD time-grad calls (one per ``u``/``v``/``w``
    across all K slices) + ``_apply_sparse_op_K``-batched spatial mms.
    Returns ``(continuity, mom_x, mom_y, mom_z)``, each ``(K*N_all, 1)`` in
    the time-slowest layout that the causal-loss bookkeeping expects.
    """
    N_all = dt_data["N_all"]
    K = t_samples.shape[0]
    rho = problem.rho

    Dx = dt_data["Dx"]; Dy = dt_data["Dy"]; Dz = dt_data["Dz"]
    Dxx = dt_data["Dxx"]; Dyy = dt_data["Dyy"]; Dzz = dt_data["Dzz"]

    xyzt = _batched_xyzt_K(dt_data["xyz_grid"], t_samples)
    pred = model(xyzt)                                              # (K*N_all, 4)
    u = pred[:, 0:1]; v = pred[:, 1:2]
    w = pred[:, 2:3]; p = pred[:, 3:4]

    ones_u = torch.ones_like(u)
    u_t = torch.autograd.grad(u, xyzt, ones_u, create_graph=True, retain_graph=True)[0][:, 3:4]
    v_t = torch.autograd.grad(v, xyzt, ones_u, create_graph=True, retain_graph=True)[0][:, 3:4]
    w_t = torch.autograd.grad(w, xyzt, ones_u, create_graph=True, retain_graph=True)[0][:, 3:4]

    u_x = _apply_sparse_op_K(Dx, u, K, N_all)
    u_y = _apply_sparse_op_K(Dy, u, K, N_all)
    u_z = _apply_sparse_op_K(Dz, u, K, N_all)
    v_x = _apply_sparse_op_K(Dx, v, K, N_all)
    v_y = _apply_sparse_op_K(Dy, v, K, N_all)
    v_z = _apply_sparse_op_K(Dz, v, K, N_all)
    w_x = _apply_sparse_op_K(Dx, w, K, N_all)
    w_y = _apply_sparse_op_K(Dy, w, K, N_all)
    w_z = _apply_sparse_op_K(Dz, w, K, N_all)
    p_x = _apply_sparse_op_K(Dx, p, K, N_all)
    p_y = _apply_sparse_op_K(Dy, p, K, N_all)
    p_z = _apply_sparse_op_K(Dz, p, K, N_all)

    if cs > 0.0:
        Sxx, Syy, Szz = u_x, v_y, w_z
        Sxy = 0.5 * (u_y + v_x)
        Sxz = 0.5 * (u_z + w_x)
        Syz = 0.5 * (v_z + w_y)
        S_mag = torch.sqrt(
            2.0 * (Sxx ** 2 + Syy ** 2 + Szz ** 2
                   + 2.0 * (Sxy ** 2 + Sxz ** 2 + Syz ** 2)) + eps
        )
        nu_lam = problem.nu
        nu_eff = nu_lam + (cs * delta) ** 2 * S_mag

        q_xx = 2.0 * nu_eff * Sxx; q_yy = 2.0 * nu_eff * Syy; q_zz = 2.0 * nu_eff * Szz
        q_xy = 2.0 * nu_eff * Sxy; q_xz = 2.0 * nu_eff * Sxz; q_yz = 2.0 * nu_eff * Syz

        visc_u = (_apply_sparse_op_K(Dx, q_xx, K, N_all)
                  + _apply_sparse_op_K(Dy, q_xy, K, N_all)
                  + _apply_sparse_op_K(Dz, q_xz, K, N_all))
        visc_v = (_apply_sparse_op_K(Dx, q_xy, K, N_all)
                  + _apply_sparse_op_K(Dy, q_yy, K, N_all)
                  + _apply_sparse_op_K(Dz, q_yz, K, N_all))
        visc_w = (_apply_sparse_op_K(Dx, q_xz, K, N_all)
                  + _apply_sparse_op_K(Dy, q_yz, K, N_all)
                  + _apply_sparse_op_K(Dz, q_zz, K, N_all))
    else:
        nu = problem.nu
        u_xx = _apply_sparse_op_K(Dxx, u, K, N_all)
        u_yy = _apply_sparse_op_K(Dyy, u, K, N_all)
        u_zz = _apply_sparse_op_K(Dzz, u, K, N_all)
        v_xx = _apply_sparse_op_K(Dxx, v, K, N_all)
        v_yy = _apply_sparse_op_K(Dyy, v, K, N_all)
        v_zz = _apply_sparse_op_K(Dzz, v, K, N_all)
        w_xx = _apply_sparse_op_K(Dxx, w, K, N_all)
        w_yy = _apply_sparse_op_K(Dyy, w, K, N_all)
        w_zz = _apply_sparse_op_K(Dzz, w, K, N_all)
        visc_u = nu * (u_xx + u_yy + u_zz)
        visc_v = nu * (v_xx + v_yy + v_zz)
        visc_w = nu * (w_xx + w_yy + w_zz)

    continuity = u_x + v_y + w_z
    mom_x = u_t + (u * u_x + v * u_y + w * u_z) + p_x / rho - visc_u
    mom_y = v_t + (u * v_x + v * v_y + w * v_z) + p_y / rho - visc_v
    mom_z = w_t + (u * w_x + v * w_y + w * w_z) + p_z / rho - visc_w
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
# RoPINN gradient-variance trust-region calibration (port of
# src/lid_benchmark.py:1671). The variance is a normalized stat over the last
# ROPINN_PAST_ITERATIONS flattened gradient vectors; small variance means the
# loss surface is locally smooth so we can grow the perturbation radius.
# =============================================================================
def compute_gradient_variance(gradient_list) -> float:
    if len(gradient_list) < 2:
        return 1.0
    arr = np.array(gradient_list)
    std_grad = np.std(arr, axis=0)
    mean_abs_grad = np.mean(np.abs(arr), axis=0) + 1e-6
    variance = float((std_grad / mean_abs_grad).mean())
    if variance == 0.0:
        variance = 1.0
    return variance


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
    weight_decay: float = 0.0,
    soap_betas=(0.9, 0.999),
    soap_shampoo_beta: float = -1.0,
    soap_eps: float = 1e-8,
    soap_weight_decay: float = 0.0,
    soap_precondition_frequency: int = 10,
) -> torch.optim.Optimizer:
    """Construct the per-window optimizer based on --optimizer.

    'adam' is the Phase 1/2 default and bit-equivalent (weight_decay=0).
    'soap' instantiates the vendored Vyas et al. (2024) SOAP optimizer
    (see ``src/soap.py``). The ``weight_decay`` kwarg targets Adam (used by
    SK-PINN's per-model regulariser); SOAP's regulariser remains
    ``soap_weight_decay`` to preserve Phase 3's bit-equivalence at defaults.
    """
    if optimizer_name == "adam":
        if weight_decay > 0.0:
            return torch.optim.Adam(model.parameters(), lr=lr,
                                    weight_decay=weight_decay)
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
    tsa_reg_weight: float = 0.0,
    method: str = "autodiff",
    ropinn_initial_region: float = 1e-4,
    ropinn_region_max: float = 0.01,
    ropinn_past_iterations: int = 10,
    spectral_n: int = 16,
    spectral_k: int = 4,
    canpinn_n: int = 10,
    canpinn_k: int = 4,
    skpinn_n: int = 12,
    skpinn_k: int = 4,
    skpinn_wd: float = 0.0,
    skpinn_h_factor: float = 1.4,
    skpinn_data: Optional[dict] = None,
    dtpinn_k: int = 4,
    dtpinn_data: Optional[dict] = None,
    sage_backward_fn=None,
    sage_les_active: bool = False,
) -> dict:
    """Train one time window. Returns a small dict of stats.

    Phase 3 extensions (all default to Phase-1/2 behaviour):
    * ``optimizer_name`` selects 'adam' (default) or 'soap'.
    * ``causal_eps > 0.0`` enables the causal PDE loss with ``causal_chunks``
      temporal slices per epoch (port of ``CausalLossNorm`` at
      ``temp/physicsnemo/physicsnemo/sym/loss/loss.py:271``).

    Method dispatch (post-Phase-3 baseline ports):
    * ``method='autodiff'`` (default) — fresh-sample PDE residual every epoch,
      bit-equivalent to the Phase 1/2/3 path.
    * ``method='ropinn'`` — region-optimized: holds a fixed (x,y,z,t) base
      sampled once at window start; perturbs it by uniform random in
      ``[0, current_region]`` each epoch. ``current_region`` is calibrated
      from the running gradient variance over the last
      ``ropinn_past_iterations`` epochs (port of
      ``src/lid_benchmark.py:train_ropinn``).
      Periodic dimensions (x, y, z) wrap modulo ``problem.L``; the time
      dimension is clamped to ``[0, window_size]``.
    """
    optimizer = _build_optimizer(
        model, optimizer_name, lr,
        weight_decay=skpinn_wd if method == "sk-pinn" else 0.0,
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

    # ----- RoPINN state: fixed interior base + gradient-variance history -----
    ropinn_active = method == "ropinn"
    if ropinn_active:
        xyzt_int_base = sample_interior(
            batch_interior, problem, window_size, device, generator,
        ).detach()
        ropinn_grad_history: List[np.ndarray] = []
        ropinn_grad_variance = 1.0

    # ----- Spectral-AD state: structured 4D grid + wavenumber array. Built once
    # per window; the same grid is used every epoch (mirrors the 2D
    # chebyshev-pinn pattern of a fixed Chebyshev tensor-product grid). When
    # causal loss is active, --spectral-k must equal --causal-chunks because
    # each time slice contributes one chunk of N**3 points.
    spectral_active = method == "chebyshev-pinn"
    if spectral_active:
        if causal_active and spectral_k != causal_chunks:
            raise ValueError(
                "method=chebyshev-pinn with --causal-eps>0 requires "
                f"--spectral-k == --causal-chunks, but got "
                f"spectral_k={spectral_k}, causal_chunks={causal_chunks}."
            )
        spectral_xyz_grid, spectral_t_samples, spectral_freqs = build_spectral_grid(
            spectral_n, spectral_k, problem, window_size, device,
        )

    # ----- CAN-PINN-faithful state: structured center grid + spacings. Built
    # once per window. Same constraint as Spectral-AD on causal_chunks.
    canpinn_active = method == "can-pinn-faithful"
    if canpinn_active:
        if causal_active and canpinn_k != causal_chunks:
            raise ValueError(
                "method=can-pinn-faithful with --causal-eps>0 requires "
                f"--canpinn-k == --causal-chunks, but got "
                f"canpinn_k={canpinn_k}, causal_chunks={causal_chunks}."
            )
        canpinn_centers, canpinn_dx = build_canpinn_centers(
            canpinn_n, canpinn_k, problem, window_size, device,
        )

    # ----- SK-PINN state: precomputed sparse RKPM derivative matrices +
    # per-window time samples. The matrices are translation-invariant on the
    # uniform periodic grid so they can be reused across windows; the caller
    # passes ``skpinn_data`` once and we just rebuild the time samples here.
    skpinn_active = method == "sk-pinn"
    if skpinn_active:
        if causal_active and skpinn_k != causal_chunks:
            raise ValueError(
                "method=sk-pinn with --causal-eps>0 requires "
                f"--skpinn-k == --causal-chunks, but got "
                f"skpinn_k={skpinn_k}, causal_chunks={causal_chunks}."
            )
        if skpinn_data is None:
            raise RuntimeError(
                "method=sk-pinn requires precomputed RKPM matrices; "
                "build_skpinn_data_3d must be called once before train_one_window."
            )
        skpinn_t_samples = torch.linspace(
            0, window_size, skpinn_k + 1, device=device,
        )[:-1]

    # ----- DT-PINN state: precomputed sparse RBF-FD operators (built once
    # by main, reused across windows). Same causal-chunks compatibility rule.
    dtpinn_active = method == "dtpinn"
    if dtpinn_active:
        if causal_active and dtpinn_k != causal_chunks:
            raise ValueError(
                "method=dtpinn with --causal-eps>0 requires "
                f"--dtpinn-k == --causal-chunks, but got "
                f"dtpinn_k={dtpinn_k}, causal_chunks={causal_chunks}."
            )
        if dtpinn_data is None:
            raise RuntimeError(
                "method=dtpinn requires precomputed RBF-FD operators; "
                "build_dtpinn_operators_3d_periodic must be called once before "
                "train_one_window."
            )
        dtpinn_t_samples = torch.linspace(
            0, window_size, dtpinn_k + 1, device=device,
        )[:-1]

    # ----- SAGE state: shares DT-PINN's RBF-FD matrices + a precompiled
    # SAGE backward function (cached in main()).
    #
    # Two seeding modes are supported. ``causal_active`` selects the form
    # ``main()`` traced + cached via ``external_seeds=True`` (per-chunk
    # weighted seeds built outside SAGE every epoch). The non-causal mode
    # uses ``external_seeds=False``: SAGE recomputes residuals + the
    # ``2/M * mask`` seeds itself in a single forward+backward, mirroring
    # the 2D ``train_sage`` pattern. ``sage_g_dict`` therefore carries
    # ``M`` and ``interior_mask`` for the internal-seeds path; it carries
    # the LES coefficients only for the on-demand residual computation
    # used by the causal seed builder and the LOG_INTERVAL / NaN guard.
    sage_active = method == "sage"
    if sage_active:
        if causal_active and dtpinn_k != causal_chunks:
            raise ValueError(
                "method=sage with --causal-eps>0 requires "
                f"--dtpinn-k == --causal-chunks, but got "
                f"dtpinn_k={dtpinn_k}, causal_chunks={causal_chunks}."
            )
        if dtpinn_data is None:
            raise RuntimeError(
                "method=sage requires precomputed RBF-FD operators; "
                "build_dtpinn_operators_3d_periodic must be called once."
            )
        if sage_backward_fn is None:
            raise RuntimeError(
                "method=sage requires a SAGE-emitted backward function; "
                "main() must call get_sage_backward_tgv before train_one_window."
            )
        sage_t_samples = torch.linspace(
            0, window_size, dtpinn_k + 1, device=device,
        )[:-1]
        # Block-diagonal sparse ops so one matmul covers all K time slices.
        sage_M_total = dtpinn_data["N_all"] * dtpinn_k
        sage_g_dict = {
            "Dx": _block_diagonal_sparse(dtpinn_data["Dx"], dtpinn_k),
            "Dy": _block_diagonal_sparse(dtpinn_data["Dy"], dtpinn_k),
            "Dz": _block_diagonal_sparse(dtpinn_data["Dz"], dtpinn_k),
            "Dxx": _block_diagonal_sparse(dtpinn_data["Dxx"], dtpinn_k),
            "Dyy": _block_diagonal_sparse(dtpinn_data["Dyy"], dtpinn_k),
            "Dzz": _block_diagonal_sparse(dtpinn_data["Dzz"], dtpinn_k),
            "N_all": sage_M_total,
            # Internal-seeds path reads these from g (matching emit_backward's
            # ``mask = g['interior_mask']`` / ``M = g['M']`` lines). All
            # 3D-periodic TGV points are interior so the mask is all-ones.
            "M": sage_M_total,
            "interior_mask": torch.ones(sage_M_total, 1, device=device),
        }
        sage_nu_lam = problem.nu
        sage_rho = problem.rho
        sage_cs_delta_sq = (les_cs * les_delta) ** 2
        sage_eps_les = les_eps

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # ---------------------------------------------------------------
        # SAGE specialised flow.
        #
        # Hot path each epoch (non-causal):
        #   * one batched model forward over (K*N_all, 4),
        #   * three batched AD time-grad calls (one per u/v/w),
        #   * one SAGE-emitted-backward call which itself runs the spatial
        #     forward + builds 2/M*mask seeds + emits the adjoint upstream,
        #   * one IC forward + ``ic_total.backward()``,
        #   * one ``combined.backward(upstream)`` to fold the SAGE-emitted
        #     upstream back through the model + time-AD graph.
        # No outside-residual computation, no ``loss_pde`` tensor, no
        # per-epoch finite check — those happen at LOG_INTERVAL only,
        # mirroring the 2D ``train_sage`` pattern.
        #
        # Causal path keeps the explicit-seeds flow because per-chunk
        # weighting needs residual values to compute ``w_causal`` before
        # SAGE runs. Residuals are computed in ``no_grad`` from
        # ``combined.detach()``, then handed into SAGE as positional seeds.
        # ---------------------------------------------------------------
        if sage_active:
            combined = sage_compute_combined(model, dtpinn_data, sage_t_samples)

            if causal_active:
                cont, mx, my, mz = _sage_compute_tgv_residuals_eval(
                    combined.detach(), sage_g_dict,
                    les_active=sage_les_active,
                    nu_lam=sage_nu_lam, rho=sage_rho,
                    cs_delta_sq=sage_cs_delta_sq, eps_les=sage_eps_les,
                )
                # Wang et al. 2022 causal weighting (matches non-SAGE path).
                pointwise = (cont.pow(2) + mx.pow(2) + my.pow(2) + mz.pow(2)).reshape(-1)
                chunk_loss = pointwise.reshape(causal_chunks, -1).mean(dim=-1)
                cs_sum = torch.cumsum(chunk_loss, dim=0)
                prefix_sum = torch.cat([
                    torch.zeros(1, device=chunk_loss.device, dtype=chunk_loss.dtype),
                    cs_sum[:-1],
                ])
                w_causal = torch.exp(-causal_eps * prefix_sum)

                N_total = cont.shape[0]
                N_per_chunk = N_total // causal_chunks
                per_chunk_scale = 2.0 / N_per_chunk
                w_view = w_causal.view(causal_chunks, 1, 1) * per_chunk_scale
                dc = (cont.view(causal_chunks, N_per_chunk, 1) * w_view).reshape(N_total, 1)
                dmx = (mx.view(causal_chunks, N_per_chunk, 1) * w_view).reshape(N_total, 1)
                dmy = (my.view(causal_chunks, N_per_chunk, 1) * w_view).reshape(N_total, 1)
                dmz = (mz.view(causal_chunks, N_per_chunk, 1) * w_view).reshape(N_total, 1)

                loss_pde_val = float((w_causal * chunk_loss).sum())
                upstream = sage_backward_fn(
                    combined.detach(), sage_g_dict, dc, dmx, dmy, dmz,
                )
            else:
                # Internal-seeds: SAGE forward+seeds+backward in one call.
                upstream = sage_backward_fn(combined.detach(), sage_g_dict)
                loss_pde_val = float("nan")  # set on demand at LOG_INTERVAL

            # IC / window-match (same as standard path; PyTorch AD).
            xyz_ic = sample_ic_xyz(batch_ic, problem, device, generator)
            t_zero = torch.zeros(batch_ic, 1, device=device)
            xyzt_ic = torch.cat([xyz_ic, t_zero], dim=-1)
            pred_ic = model(xyzt_ic)
            if prev_model is None:
                u0, v0, w0, p0 = problem.initial_condition(
                    xyz_ic[:, 0:1], xyz_ic[:, 1:2], xyz_ic[:, 2:3]
                )
                ic_terms = (
                    (pred_ic[..., 0:1] - u0).pow(2).mean()
                    + (pred_ic[..., 1:2] - v0).pow(2).mean()
                    + (pred_ic[..., 2:3] - w0).pow(2).mean()
                    + (pred_ic[..., 3:4] - p0).pow(2).mean()
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

            reg_term = None
            if tsa_reg_weight > 0.0 and hasattr(model, "regularization_loss"):
                reg_term = tsa_reg_weight * model.regularization_loss()
            ic_total = loss_ic + (reg_term if reg_term is not None else 0.0)
            ic_total.backward()
            combined.backward(gradient=upstream)

            optimizer.step()
            scheduler.step()

            last_ic = float(loss_ic.detach())
            log_now = (epoch + 1) % LOG_INTERVAL == 0
            if log_now or epoch == epochs - 1:
                # Compute residuals on demand (cheap; once per LOG_INTERVAL)
                # so the printed loss + the NaN guard reflect the current
                # state without paying for residuals every epoch.
                if not causal_active:
                    cont, mx, my, mz = _sage_compute_tgv_residuals_eval(
                        combined.detach(), sage_g_dict,
                        les_active=sage_les_active,
                        nu_lam=sage_nu_lam, rho=sage_rho,
                        cs_delta_sq=sage_cs_delta_sq, eps_les=sage_eps_les,
                    )
                    loss_pde_val = float(
                        cont.pow(2).mean() + mx.pow(2).mean()
                        + my.pow(2).mean() + mz.pow(2).mean()
                    )
                last_pde = loss_pde_val
                last_loss = last_pde + last_ic + (
                    float(reg_term.detach()) if reg_term is not None else 0.0
                )
                if not math.isfinite(last_loss):
                    print(f"  {log_prefix} epoch {epoch+1}: NaN/Inf loss — stopping window early.")
                    nan_seen = True
                    break
                if log_now:
                    print(f"  {log_prefix} epoch {epoch+1:>6d}  loss={last_loss:.4e}  "
                          f"pde={last_pde:.4e}  ic={last_ic:.4e}  "
                          f"lr={scheduler.get_last_lr()[0]:.2e}  [SAGE]")
            else:
                # Quick non-finite probe on the SAGE upstream (single
                # reduction; ~us on GPU). Catches NaN within one epoch
                # without paying for the full residual computation.
                if not torch.isfinite(upstream).all():
                    print(f"  {log_prefix} epoch {epoch+1}: NaN/Inf in SAGE upstream — stopping window early.")
                    nan_seen = True
                    break
            continue  # skip the standard PDE-loss + .backward() block below

        # --- PDE residual on interior ---
        if spectral_active:
            # Spectral-AD: residual computed on the fixed structured 4D grid
            # using FFT spatial derivatives + AD time derivative. The flat
            # output is laid out time-slowest so that ``reshape(K, N**3)``
            # gives one chunk per time slice when causal loss is on.
            cont, mx, my, mz = spectral_residual(
                model, spectral_xyz_grid, spectral_t_samples, spectral_freqs,
                problem, cs=les_cs, delta=les_delta, eps=les_eps,
            )
        elif canpinn_active:
            # CAN-PINN-faithful: residual at fixed centers using 7-point
            # axis-aligned cross stencil + AD time. Time-slowest layout so the
            # causal loss naturally chunks by time slice.
            cont, mx, my, mz = can_pinn_residual(
                model, canpinn_centers,
                canpinn_dx, canpinn_dx, canpinn_dx,
                problem, cs=les_cs, delta=les_delta, eps=les_eps,
            )
        elif skpinn_active:
            # SK-PINN: spatial via sparse RKPM matvecs, time via AD on the
            # t input column. Time-slowest layout via per-time-slice loop.
            cont, mx, my, mz = skpinn_residual(
                model, skpinn_data, skpinn_t_samples, problem,
                cs=les_cs, delta=les_delta, eps=les_eps,
            )
        elif dtpinn_active:
            # DT-PINN: spatial via sparse RBF-FD matvecs, time via AD.
            cont, mx, my, mz = dtpinn_residual(
                model, dtpinn_data, dtpinn_t_samples, problem,
                cs=les_cs, delta=les_delta, eps=les_eps,
            )
        else:
            if ropinn_active:
                current_region = float(np.clip(
                    ropinn_initial_region / ropinn_grad_variance,
                    a_min=0.0, a_max=ropinn_region_max,
                ))
                perturbation = torch.rand(
                    xyzt_int_base.shape, device=device, generator=generator
                ) * current_region
                xyzt_int = xyzt_int_base + perturbation
                # Periodic in x,y,z; clamp in t.
                xyzt_int = torch.cat([
                    xyzt_int[..., 0:3] % problem.L,
                    xyzt_int[..., 3:4].clamp(0.0, window_size),
                ], dim=-1)
            else:
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

        # Optional architecture-specific regularizer (TSA-PINN's Dynamic Slope
        # Recovery; gated on tsa_reg_weight > 0 so MLP / PirateNet bit-equiv
        # is preserved at default flags).
        if tsa_reg_weight > 0.0 and hasattr(model, "regularization_loss"):
            loss = loss + tsa_reg_weight * model.regularization_loss()

        if not torch.isfinite(loss):
            print(f"  {log_prefix} epoch {epoch+1}: NaN/Inf loss — stopping window early.")
            nan_seen = True
            break

        loss.backward()

        if ropinn_active:
            # Track flattened gradients for trust-region calibration. Adam
            # / SOAP do not modify ``.grad`` so reading after backward (and
            # before the next zero_grad) is fine.
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().view(-1))
            if grads:
                flat_grad = torch.cat(grads).cpu().numpy()
                ropinn_grad_history.append(flat_grad)
                ropinn_grad_history = ropinn_grad_history[-ropinn_past_iterations:]
                ropinn_grad_variance = compute_gradient_variance(ropinn_grad_history)

        optimizer.step()
        scheduler.step()

        last_loss = float(loss.detach())
        last_pde = float(loss_pde.detach())
        last_ic = float(loss_ic.detach())

        if (epoch + 1) % LOG_INTERVAL == 0:
            extra = ""
            if ropinn_active:
                extra = (f"  region={current_region:.2e}"
                         f"  grad_var={ropinn_grad_variance:.3f}")
            print(f"  {log_prefix} epoch {epoch+1:>6d}  loss={last_loss:.4e}  "
                  f"pde={last_pde:.4e}  ic={last_ic:.4e}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}{extra}")

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

    # Resolve SK-PINN weight decay (per-model lookup unless user overrides).
    if args.skpinn_wd < 0.0:
        skpinn_wd_resolved = _SKPINN_TGV_WD.get(args.model, 1e-4)
    else:
        skpinn_wd_resolved = float(args.skpinn_wd)

    # Build SK-PINN derivative matrices once (translation-invariant, reused
    # across windows). Skip for other methods to avoid the build cost.
    skpinn_data = None
    if args.method == "sk-pinn":
        skpinn_data = build_skpinn_data_3d(
            args.skpinn_n, problem, args.skpinn_h_factor, device,
        )
        print(f"  SK-PINN: weight_decay={skpinn_wd_resolved:.1e} "
              f"(model={args.model}).")

    # Build DT-PINN RBF-FD operators once (also translation-invariant).
    # SAGE shares the same operators (its emitted backward references
    # ``g['Dx']`` etc. on the same sparse matrices).
    dtpinn_data = None
    if args.method in ("dtpinn", "sage"):
        dtpinn_data = build_dtpinn_operators_3d_periodic(
            args.dtpinn_n, problem, args.dtpinn_p, device,
        )

    # Trace + emit the SAGE backward (cached). The trace is split on whether
    # the LES branch is active; both branches use the same input layout
    # (4 model outputs + 3 AD time derivatives = 7 columns).
    #
    # Seeding mode is fixed at trace time. ``--causal-eps>0`` uses
    # external seeds (per-chunk weights need residual values to build w_t,
    # so the seeds can't be the standard ``2/M*mask`` form). ``--causal-eps==0``
    # uses internal seeds — SAGE recomputes residuals + standard seeds in one
    # forward+backward, mirroring the 2D pattern.
    sage_backward_fn = None
    sage_les_active_flag = False
    if args.method == "sage":
        sage_les_active_flag = args.les_cs > 0.0
        nu_lam_val = problem.nu
        rho_val = problem.rho
        cs_delta_sq_val = (args.les_cs * args.les_delta) ** 2
        eps_les_val = args.les_eps
        sage_external_seeds = args.causal_eps > 0.0
        sage_source, sage_backward_fn = get_sage_backward_tgv(
            les_active=sage_les_active_flag,
            nu_lam=nu_lam_val, rho=rho_val,
            cs_delta_sq=cs_delta_sq_val, eps_les=eps_les_val,
            external_seeds=sage_external_seeds,
        )
        print(f"  SAGE: traced backward (les_active={sage_les_active_flag}, "
              f"nu={nu_lam_val:.4e}, rho={rho_val}, cs_delta_sq={cs_delta_sq_val:.4e}, "
              f"external_seeds={sage_external_seeds}).")

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
                tsa_reg_weight=(args.tsa_reg_weight if args.model == "tsa-pinn" else 0.0),
                method=args.method,
                ropinn_initial_region=args.ropinn_initial_region,
                ropinn_region_max=args.ropinn_region_max,
                ropinn_past_iterations=args.ropinn_past_iterations,
                spectral_n=args.spectral_n,
                spectral_k=args.spectral_k,
                canpinn_n=args.canpinn_n,
                canpinn_k=args.canpinn_k,
                skpinn_n=args.skpinn_n,
                skpinn_k=args.skpinn_k,
                skpinn_wd=skpinn_wd_resolved,
                skpinn_h_factor=args.skpinn_h_factor,
                skpinn_data=skpinn_data,
                dtpinn_k=args.dtpinn_k,
                dtpinn_data=dtpinn_data,
                sage_backward_fn=sage_backward_fn,
                sage_les_active=sage_les_active_flag,
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
