#!/usr/bin/env python3
"""
Generate accuracy-vs-speed Pareto frontier figures for the SAGE paper.

Three-panel scatter plot: PDE residual accuracy versus training time for
selected problem-architecture combinations where SAGE dominates.

  (a) Cavity — MLP:       SAGE Pareto-optimal (best accuracy + fastest)
  (b) Cavity — PirateNet: SAGE 18× faster than Autodiff
  (c) Kovasznay — MLP:    SAGE Pareto-optimal (best accuracy + fastest)

Output: llmdocs/paper/fig_pareto_frontiers.pdf
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

RESULTS_CSV = Path('results/lid_benchmark_results.csv')
OUTPUT_DIR = Path('llmdocs/paper')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Panels: (problem, model, title)
PANELS = [
    ('cavity',    'mlp',        '(a) Cavity — MLP'),
    ('cavity',    'pirate-net', '(b) Cavity — PirateNet'),
    ('kovasznay', 'mlp',        '(c) Kovasznay — MLP'),
]

# Per-panel annotation offset (points right, points up) for speedup label
ANNOT_OFFSETS = [(12, 28), (12, 28), (12, 28)]

# Wong colorblind-safe palette (consistent with Fig 1)
METHOD_CFG = {
    'sage':       {'label': 'SAGE (Ours)', 'color': '#0072B2', 'marker': '*',  'ms': 220, 'zorder': 10},
    'analytical': {'label': 'Handcrafted', 'color': '#CC79A7', 'marker': 'D',  'ms': 50,  'zorder': 5},
    'autodiff':   {'label': 'Autodiff',    'color': '#E69F00', 'marker': 'o',  'ms': 50,  'zorder': 5},
    'dtpinn':     {'label': 'DT-PINN',     'color': '#009E73', 'marker': 's',  'ms': 50,  'zorder': 5},
    'ropinn':     {'label': 'RoPINN',      'color': '#888888', 'marker': '^',  'ms': 55,  'zorder': 5},
    'sk-pinn':    {'label': 'SK-PINN',     'color': '#D55E00', 'marker': 'X',  'ms': 50,  'zorder': 5},
}

# Legend ordering (SAGE first, then ours, then baselines by speed)
DISPLAY_ORDER = ['sage', 'analytical', 'autodiff', 'dtpinn', 'ropinn', 'sk-pinn']


# ============================================================================
# Plotting
# ============================================================================

def make_figure():
    df = pd.read_csv(RESULTS_CSV)

    # Global style (matches convergence figure)
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

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0))
    fig.subplots_adjust(wspace=0.42, bottom=0.26, top=0.88)

    legend_handles = {}

    for i, (ax, (problem, model, title)) in enumerate(zip(axes, PANELS)):
        sub = df[(df['problem'] == problem) & (df['model'] == model)]

        # Plot each method
        for method in DISPLAY_ORDER:
            row = sub[sub['method'] == method]
            if row.empty:
                continue
            cfg = METHOD_CFG[method]
            t = row['train_time_min'].values[0]
            pde = row['pde_rms'].values[0]

            h = ax.scatter(
                t, pde,
                c=cfg['color'], marker=cfg['marker'], s=cfg['ms'],
                zorder=cfg['zorder'],
                edgecolors='white', linewidths=0.6,
            )
            if method not in legend_handles:
                legend_handles[method] = h

        # Speedup annotation (SAGE vs Autodiff)
        sage_r = sub[sub['method'] == 'sage']
        auto_r = sub[sub['method'] == 'autodiff']
        if not sage_r.empty and not auto_r.empty:
            sage_t = sage_r['train_time_min'].values[0]
            auto_t = auto_r['train_time_min'].values[0]
            speedup = auto_t / sage_t
            sage_pde = sage_r['pde_rms'].values[0]
            ox, oy = ANNOT_OFFSETS[i]
            ax.annotate(
                f'{speedup:.1f}×',
                xy=(sage_t, sage_pde),
                xytext=(ox, oy),
                textcoords='offset points',
                fontsize=8.5, color='#0072B2', fontweight='bold',
                ha='center', va='bottom',
                arrowprops=dict(
                    arrowstyle='->', color='#0072B2', lw=0.8,
                    shrinkA=0, shrinkB=5,
                ),
                zorder=11,
            )

        # Axes
        ax.set_xscale('log')
        ax.set_xlabel('Training Time (min)')
        if i == 0:
            ax.set_ylabel('PDE Residual (RMS)')
        ax.set_title(title, fontweight='normal', pad=8)

        # Add vertical padding so annotations have room
        ymin_data = sub['pde_rms'].min()
        ymax_data = sub['pde_rms'].max()
        yrange = ymax_data - ymin_data
        ax.set_ylim(ymin_data - 0.08 * yrange, ymax_data + 0.18 * yrange)

        # Tick formatting — avoid scientific notation on y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda v, _: f'{v:.4f}' if v < 0.01 else f'{v:.3f}'
        ))

        # Grid and spine styling (matches Fig 1)
        ax.grid(True, which='major', ls='-', alpha=0.15, color='grey')
        ax.grid(True, which='minor', ls='-', alpha=0.06, color='grey')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=3)

    # Shared legend below all panels
    handles = [legend_handles[m] for m in DISPLAY_ORDER if m in legend_handles]
    labels = [METHOD_CFG[m]['label'] for m in DISPLAY_ORDER if m in legend_handles]
    fig.legend(handles, labels, loc='lower center', ncol=6,
               frameon=False, bbox_to_anchor=(0.5, 0.01),
               columnspacing=1.0, handletextpad=0.4, scatterpoints=1)

    return fig


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    fig = make_figure()

    out_path = OUTPUT_DIR / 'fig_pareto_frontiers.pdf'
    fig.savefig(out_path)
    print(f'Saved: {out_path}')

    png_path = OUTPUT_DIR / 'fig_pareto_frontiers.png'
    fig.savefig(png_path)
    print(f'Saved: {png_path}')

    plt.close(fig)
