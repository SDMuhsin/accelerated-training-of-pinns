#!/usr/bin/env python3
"""
Generate training convergence figures for the SAGE paper.

Produces a two-panel figure (training loss + PDE residual RMS vs epoch)
for the lid-driven cavity problem with MLP, comparing all 6 methods.

Output: llmdocs/paper/fig_convergence_cavity_mlp.pdf
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path('results')
OUTPUT_DIR = Path('llmdocs/stream_sage_paper/paper/v2_tetci')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROBLEM = 'cavity'
MODEL = 'mlp'
SEED = 42
TAG = 'multiseed_20260427'
# CAN-PINN was re-run on the H100 MIG partition under a separate tag
# (canpinn_hpc_20260428); per-cell tracking CSVs follow the same convention
# but with method='chebyshev-pinn' (CAN-PINN's internal label) and that tag.
CANPINN_TAG = 'canpinn_hpc_20260428'

# Methods in display order (SAGE last so it draws on top).
# Handcrafted is excluded: not part of the multiseed re-run.
# DT-PINN is excluded: it uses L-BFGS over 5,000 outer steps with a
# different loss scale than the four Adam-trained methods (30,000 epochs);
# we keep this figure to a same-protocol comparison.  DT-PINN's accuracy
# numbers are in Table III.
# SK-PINN is excluded from this figure: its uniform N=200 grid normalizes
# the loss differently from the Chebyshev N=50 grid the other methods use,
# and it applies a model-specific weight decay no other Adam baseline does;
# both panels would mix protocol-incompatible curves.  Numerical comparison
# at SK-PINN's published configuration is in §V's Combined-speedup paragraph.
# CAN-PINN (chebyshev-pinn): same Chebyshev forward / Adam+fp32 / 30k-epoch
# protocol as SAGE; loaded from the canpinn_hpc_20260428 tracking CSVs.
METHODS = {
    'ropinn':         {'label': 'RoPINN',      'color': '#888888', 'ls': (0, (4, 2)),       'lw': 1.0, 'zorder': 2},
    'autodiff':       {'label': 'Autodiff',    'color': '#E69F00', 'ls': '-',               'lw': 1.4, 'zorder': 3},
    'chebyshev-pinn': {'label': 'Spectral-AD', 'color': '#CC79A7', 'ls': (0, (5, 2)),       'lw': 1.4, 'zorder': 4},
    'sage':           {'label': 'SAGE (Ours)', 'color': '#0072B2', 'ls': '-',               'lw': 2.0, 'zorder': 5},
}

# All methods share the Chebyshev grid; both panels can use them.
LOSS_PANEL_METHODS = METHODS

# Smoothing: exponential moving average span (in number of data points)
EMA_SPAN = 25  # ~2500 epochs at 100-epoch intervals


# ============================================================================
# Data loading
# ============================================================================

def load_tracking(problem, method, model, seed, tag=None):
    """Load a single tracking CSV (tagged variant from the released release).

    CAN-PINN (method='chebyshev-pinn') lives under the canpinn_hpc_20260428
    tag; all other methods under multiseed_20260427.  Caller may override
    `tag` explicitly; otherwise the default is chosen by method.
    """
    if tag is None:
        tag = CANPINN_TAG if method == 'chebyshev-pinn' else TAG
    path = RESULTS_DIR / f'tracking_{problem}_{method}_{model}_s{seed}_{tag}.csv'
    return pd.read_csv(path)


def ema(series, span):
    """Exponential moving average."""
    return series.ewm(span=span, adjust=True).mean()


# ============================================================================
# Plotting
# ============================================================================

def make_figure():
    # Global style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Computer Modern Roman'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 7.5,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'text.usetex': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
    })

    fig, (ax_loss, ax_pde) = plt.subplots(
        1, 2, figsize=(7.0, 3.0), sharey=False
    )
    fig.subplots_adjust(wspace=0.35, bottom=0.22)

    # Panel (a): methods on Chebyshev grid only (SK-PINN excluded — its
    # uniform N=200 grid normalizes the loss differently).
    for method, style in LOSS_PANEL_METHODS.items():
        df = load_tracking(PROBLEM, method, MODEL, SEED)
        epochs = df['epoch'].values / 1000.0
        loss_raw = df['train_loss'].values
        loss_smooth = ema(df['train_loss'], EMA_SPAN).values
        ax_loss.plot(epochs, loss_raw, color=style['color'],
                     alpha=0.10, lw=0.4, zorder=style['zorder'] - 1)
        ax_loss.plot(epochs, loss_smooth, color=style['color'],
                     ls=style['ls'], lw=style['lw'], label=style['label'],
                     zorder=style['zorder'])

    # Panel (b): all methods (PDE residual is computed on the actual
    # physical interior so the comparison is meaningful).
    for method, style in METHODS.items():
        df = load_tracking(PROBLEM, method, MODEL, SEED)
        epochs = df['epoch'].values / 1000.0
        pde_raw = df['pde_rms'].values
        pde_smooth = ema(df['pde_rms'], EMA_SPAN).values
        ax_pde.plot(epochs, pde_raw, color=style['color'],
                    alpha=0.10, lw=0.4, zorder=style['zorder'] - 1)
        ax_pde.plot(epochs, pde_smooth, color=style['color'],
                    ls=style['ls'], lw=style['lw'], label=style['label'],
                    zorder=style['zorder'])

    # --- Panel (a): Training Loss ---
    ax_loss.set_yscale('log')
    ax_loss.set_xlabel('Epoch ($\\times 10^3$)')
    ax_loss.set_ylabel('Training Loss')
    ax_loss.set_title('(a) Training Loss', fontweight='normal', pad=8)
    ax_loss.set_xlim(0, 30)
    ax_loss.set_ylim(8e-3, 5e-1)
    ax_loss.set_xticks([0, 5, 10, 15, 20, 25, 30])

    # --- Panel (b): PDE Residual RMS ---
    ax_pde.set_yscale('log')
    ax_pde.set_xlabel('Epoch ($\\times 10^3$)')
    ax_pde.set_ylabel('PDE Residual (RMS)')
    ax_pde.set_title('(b) PDE Residual', fontweight='normal', pad=8)
    ax_pde.set_xlim(0, 30)
    ax_pde.set_ylim(1e-2, 2e-1)
    ax_pde.set_xticks([0, 5, 10, 15, 20, 25, 30])

    # Shared legend below both panels
    handles, labels = ax_pde.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6,
               frameon=False, bbox_to_anchor=(0.5, 0.01),
               columnspacing=1.0, handletextpad=0.4, handlelength=2.0)

    # Light grid and spine cleanup
    for ax in [ax_loss, ax_pde]:
        ax.grid(True, which='major', ls='-', alpha=0.15, color='grey')
        ax.grid(True, which='minor', ls='-', alpha=0.06, color='grey')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=3)

    return fig


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    fig = make_figure()

    out_path = OUTPUT_DIR / 'fig_convergence_cavity_mlp.pdf'
    fig.savefig(out_path)
    print(f'Saved: {out_path}')

    # Also save PNG for quick preview
    png_path = OUTPUT_DIR / 'fig_convergence_cavity_mlp.png'
    fig.savefig(png_path)
    print(f'Saved: {png_path}')

    plt.close(fig)
