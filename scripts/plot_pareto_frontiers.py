#!/usr/bin/env python3
"""
Generate accuracy-vs-speed Pareto frontier figures for the SAGE paper.

Three-panel scatter plot, one per PDE problem, chosen so each axis stress
is visible: (a) cavity-PirateNet shows SAGE's largest wall-clock win
(18.8x); (b) Kovasznay-MLP shows the regime where DT-PINN's L-BFGS+fp64
beats SAGE-Adam-fp32 on residual; (c) elasticity-MLP shows DT-PINN's
largest residual lead (3.5x), the worst case for SAGE's accuracy.  The
DT-PINN annotation calls out its lower residual where it exists.

  (a) Cavity      PirateNet:    SAGE 18.8x faster than Autodiff
  (b) Kovasznay   MLP:          DT-PINN 1.6x lower residual; SAGE 7.0x faster
  (c) Elasticity  MLP (n=4):    DT-PINN 3.5x lower residual; SAGE 8.0x faster

Data source: results/lid_benchmark_results.csv filtered to
tag == 'multiseed_20260427' (5 seeds {0, 1, 7, 23, 42}); points are seed
means.  Elasticity-MLP DT-PINN excludes seed 0 per Table III's $^\dagger$
exclusion rule (PDE RMS > 5x median of remaining seeds).

Output: llmdocs/stream_sage_paper/paper/v2_tetci/fig_pareto_frontiers.pdf
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
OUTPUT_DIR = Path('llmdocs/stream_sage_paper/paper/v2_tetci')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TAG = 'multiseed_20260427'

# Panels: (problem, model, title) -- one per PDE problem to cover the full
# accuracy-speed envelope; elasticity-MLP is the cell where DT-PINN's
# residual lead over SAGE is largest.
PANELS = [
    ('cavity',     'pirate-net', '(a) Cavity — PirateNet'),
    ('kovasznay',  'mlp',        '(b) Kovasznay — MLP'),
    ('elasticity', 'mlp',        '(c) Elasticity — MLP'),
]

# Per-panel annotation offset (points right, points up) for SAGE speedup
# label.  Tuned so the annotation does not collide with the DT-PINN
# residual-lead callout.
ANNOT_OFFSETS = [(12, 28), (12, 28), (12, 28)]

# Wong colorblind-safe palette (consistent with Fig 1)
METHOD_CFG = {
    'sage':       {'label': 'SAGE (Ours)', 'color': '#0072B2', 'marker': '*',  'ms': 220, 'zorder': 10},
    'autodiff':   {'label': 'Autodiff',    'color': '#E69F00', 'marker': 'o',  'ms': 50,  'zorder': 5},
    'dtpinn':     {'label': 'DT-PINN',     'color': '#009E73', 'marker': 's',  'ms': 50,  'zorder': 5},
    'ropinn':     {'label': 'RoPINN',      'color': '#888888', 'marker': '^',  'ms': 55,  'zorder': 5},
    'sk-pinn':    {'label': 'SK-PINN',     'color': '#D55E00', 'marker': 'X',  'ms': 50,  'zorder': 5},
}

# Legend ordering (SAGE first, then baselines by speed)
DISPLAY_ORDER = ['sage', 'autodiff', 'dtpinn', 'ropinn', 'sk-pinn']


# ============================================================================
# Plotting
# ============================================================================

def aggregate_means(df, problem, model):
    """Return seed-mean (train_time_min, pde_rms) per method.

    For elasticity-MLP/-PirateNet under DT-PINN, exclude seed 0 (n=4)
    matching the Table III dagger convention.
    """
    sub = df[(df['problem'] == problem) & (df['model'] == model)]
    rows = []
    for method in DISPLAY_ORDER:
        g = sub[sub['method'] == method]
        if g.empty:
            continue
        # Apply n=4 dagger filter for the elasticity DT-PINN cells.
        if (problem == 'elasticity' and method == 'dtpinn'
                and model in ('mlp', 'pirate-net')):
            g = g[g['seed'] != 0]
        rows.append({
            'method': method,
            'train_time_min': g['train_time_min'].mean(),
            'pde_rms': g['pde_rms'].mean(),
        })
    return pd.DataFrame(rows)


def make_figure():
    df = pd.read_csv(RESULTS_CSV)
    df = df[df['tag'] == TAG].copy()

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
        agg = aggregate_means(df, problem, model)

        # Plot each method
        for method in DISPLAY_ORDER:
            row = agg[agg['method'] == method]
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
        sage_r = agg[agg['method'] == 'sage']
        auto_r = agg[agg['method'] == 'autodiff']
        dt_r = agg[agg['method'] == 'dtpinn']
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

        # DT-PINN residual-lead annotation (only when DT-PINN's PDE RMS
        # is notably below SAGE's; e.g., the Kovasznay/elasticity panels).
        if not sage_r.empty and not dt_r.empty:
            dt_t = dt_r['train_time_min'].values[0]
            dt_pde = dt_r['pde_rms'].values[0]
            sage_pde = sage_r['pde_rms'].values[0]
            ratio = sage_pde / dt_pde if dt_pde > 0 else 1.0
            if ratio >= 1.3:  # DT-PINN at least 1.3x lower residual
                ax.annotate(
                    f'DT-PINN: {ratio:.1f}× lower residual',
                    xy=(dt_t, dt_pde),
                    xytext=(34, 8),
                    textcoords='offset points',
                    fontsize=7.5, color='#009E73', fontweight='normal',
                    ha='center', va='bottom',
                    arrowprops=dict(
                        arrowstyle='->', color='#009E73', lw=0.8,
                        shrinkA=0, shrinkB=4,
                    ),
                    zorder=11,
                )

        # Axes
        ax.set_xscale('log')
        ax.set_xlabel('Training Time (min)')
        if i == 0:
            ax.set_ylabel('PDE Residual (RMS)')
        ax.set_title(title, fontweight='normal', pad=8)

        # Add vertical padding so annotations have room.
        ymin_data = agg['pde_rms'].min()
        ymax_data = agg['pde_rms'].max()
        yrange = ymax_data - ymin_data
        ax.set_ylim(ymin_data - 0.10 * yrange, ymax_data + 0.20 * yrange)

        # Tick formatting --- avoid scientific notation on y-axis
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
