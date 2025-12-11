"""
Generate SPECTO-ELM comparison figures for the paper.
Compares accuracy and timing against baselines.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Data from experiments_specto.csv and experiments.csv
RESULTS = {
    'Poisson': {
        'Vanilla PINN': 2.47e-3,
        'PIELM': 3.15e-1,
        'SPECTO-1L': 3.51e-4,
        'SPECTO-4L': 1.26e-4,
    },
    'Laplace': {
        'Vanilla PINN': 1.02e-3,
        'PIELM': 3.08e-1,
        'SPECTO-1L': 2.45e-4,
        'SPECTO-4L': 4.83e-4,
    },
    'NL-Poisson': {
        'Vanilla PINN': 1.09e-1,
        'PIELM': 1.42e-4,
        'SPECTO-4L': 3.90e-5,
    },
}

TIMING = {
    'Poisson': {
        'Vanilla PINN': 72.74,
        'PIELM': 0.051,
        'SPECTO-1L': 0.012,
        'SPECTO-4L': 10.53,
    },
}


def create_accuracy_comparison(output_path):
    """Create bar chart comparing accuracy across methods."""
    fig, ax = plt.subplots(figsize=(12, 6))

    tasks = list(RESULTS.keys())
    methods = ['Vanilla PINN', 'PIELM', 'SPECTO-1L', 'SPECTO-4L']
    colors = ['#E74C3C', '#F39C12', '#3498DB', '#2ECC71']

    x = np.arange(len(tasks))
    width = 0.2

    for i, method in enumerate(methods):
        errors = []
        for task in tasks:
            if method in RESULTS[task]:
                errors.append(RESULTS[task][method])
            else:
                errors.append(np.nan)

        bars = ax.bar(x + i * width, errors, width, label=method, color=colors[i])

        # Add value labels on bars
        for j, (bar, err) in enumerate(zip(bars, errors)):
            if not np.isnan(err):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                       f'{err:.1e}', ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_yscale('log')
    ax.set_ylabel('L2 Error (log scale)', fontsize=12)
    ax.set_xlabel('PDE Task', fontsize=12)
    ax.set_title('SPECTO-ELM Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(1e-5, 1)

    # Add reference lines
    ax.axhline(y=1e-3, color='gray', linestyle='--', alpha=0.3)
    ax.text(2.5, 1.2e-3, 'Good accuracy threshold', fontsize=9, color='gray', ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy comparison to: {output_path}")


def create_speedup_bar_chart(output_path):
    """Create bar chart showing speedups."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['PIELM', 'SPECTO-1L', 'SPECTO-4L']
    speedups = [
        TIMING['Poisson']['Vanilla PINN'] / TIMING['Poisson']['PIELM'],
        TIMING['Poisson']['Vanilla PINN'] / TIMING['Poisson']['SPECTO-1L'],
        TIMING['Poisson']['Vanilla PINN'] / TIMING['Poisson']['SPECTO-4L'],
    ]
    colors = ['#F39C12', '#3498DB', '#2ECC71']

    x = np.arange(len(methods))
    bars = ax.bar(x, speedups, color=colors, edgecolor='white', linewidth=2)

    # Add value labels
    for bar, speedup in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
               f'{speedup:.0f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Speedup vs Vanilla PINN', fontsize=12)
    ax.set_title('Training Speed Comparison (Poisson Equation)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight SPECTO-1L as best speed
    ax.annotate('Best Speed',
                xy=(1, speedups[1]),
                xytext=(1.5, speedups[1] * 0.7),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'),
                ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved speedup comparison to: {output_path}")


def create_accuracy_vs_speed_scatter(output_path):
    """Create scatter plot of accuracy vs speed tradeoff."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Data points for Poisson task
    methods = {
        'Vanilla PINN': {'time': 72.74, 'error': 2.47e-3, 'color': '#E74C3C', 'marker': 'o'},
        'PIELM': {'time': 0.051, 'error': 3.15e-1, 'color': '#F39C12', 'marker': 's'},
        'SPECTO-1L': {'time': 0.012, 'error': 3.51e-4, 'color': '#3498DB', 'marker': '^'},
        'SPECTO-4L': {'time': 10.53, 'error': 1.26e-4, 'color': '#2ECC71', 'marker': 'D'},
    }

    for name, data in methods.items():
        ax.scatter(data['time'], data['error'],
                  c=data['color'], marker=data['marker'], s=200, label=name, zorder=3)
        # Add label
        offset = (15, 10) if name != 'PIELM' else (-60, 10)
        ax.annotate(name, (data['time'], data['error']),
                   xytext=offset, textcoords='offset points', fontsize=10)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Training Time (seconds, log scale)', fontsize=12)
    ax.set_ylabel('L2 Error (log scale)', fontsize=12)
    ax.set_title('Accuracy vs Speed Tradeoff (Poisson Equation)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add ideal direction arrow
    ax.annotate('', xy=(0.01, 1e-5), xytext=(10, 1e-1),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(0.05, 3e-4, 'Better\n(faster + accurate)', fontsize=9, color='green', ha='center')

    # Highlight SPECTO-1L as Pareto optimal
    ax.add_patch(plt.Circle((0.012, 3.51e-4), 0.008, color='#3498DB', alpha=0.2))

    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0.005, 200)
    ax.set_ylim(1e-5, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy vs speed scatter to: {output_path}")


def create_method_summary_table(output_path):
    """Create a visual table summarizing all methods."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # Table data
    columns = ['Method', 'L2 Error\n(Poisson)', 'Time (s)', 'Speedup', 'Key Characteristic']
    rows = [
        ['Vanilla PINN', '2.47e-3', '72.74', '1x', 'Standard baseline'],
        ['PIELM', '3.15e-1', '0.051', '1,427x', 'Fast but inaccurate on linear PDEs'],
        ['SPECTO-ELM (1L)', '3.51e-4', '0.012', '6,062x', 'Best speed-accuracy tradeoff'],
        ['SPECTO-ELM (4L)', '1.26e-4', '10.53', '6.9x', 'Best accuracy'],
    ]

    # Color coding
    colors = [
        ['#ffffff', '#ffcccc', '#ffcccc', '#ffffff', '#ffffff'],
        ['#ffffff', '#ffcccc', '#ccffcc', '#ccffcc', '#ffffcc'],
        ['#ffffff', '#ccffcc', '#ccffcc', '#ccffcc', '#ccffcc'],
        ['#ffffff', '#ccffcc', '#ffffcc', '#ffffcc', '#ccffcc'],
    ]

    table = ax.table(cellText=rows, colLabels=columns, loc='center',
                    cellLoc='center', colLoc='center',
                    cellColours=colors)

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style header
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title('SPECTO-ELM Method Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved method summary table to: {output_path}")


def create_combined_figure(output_path):
    """Create a combined figure for the paper."""
    fig = plt.figure(figsize=(14, 10))

    # Subplot 1: Accuracy comparison
    ax1 = fig.add_subplot(2, 2, 1)
    tasks = list(RESULTS.keys())
    methods = ['Vanilla PINN', 'PIELM', 'SPECTO-1L', 'SPECTO-4L']
    colors = ['#E74C3C', '#F39C12', '#3498DB', '#2ECC71']
    x = np.arange(len(tasks))
    width = 0.2

    for i, method in enumerate(methods):
        errors = []
        for task in tasks:
            if method in RESULTS[task]:
                errors.append(RESULTS[task][method])
            else:
                errors.append(np.nan)
        ax1.bar(x + i * width, errors, width, label=method, color=colors[i])

    ax1.set_yscale('log')
    ax1.set_ylabel('L2 Error', fontsize=11)
    ax1.set_title('(a) Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(tasks)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(1e-5, 1)

    # Subplot 2: Speedup
    ax2 = fig.add_subplot(2, 2, 2)
    methods_speed = ['PIELM', 'SPECTO-1L', 'SPECTO-4L']
    speedups = [1427, 6062, 6.9]
    ax2.bar(methods_speed, speedups, color=['#F39C12', '#3498DB', '#2ECC71'])
    ax2.set_ylabel('Speedup vs Vanilla PINN', fontsize=11)
    ax2.set_title('(b) Training Speed', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(methods_speed, speedups)):
        ax2.text(i, s + 200, f'{s:.0f}x', ha='center', fontsize=10, fontweight='bold')

    # Subplot 3: Accuracy improvement over Vanilla PINN
    ax3 = fig.add_subplot(2, 2, 3)
    improvements = {
        'Poisson': 2.47e-3 / 1.26e-4,
        'Laplace': 1.02e-3 / 2.45e-4,
        'NL-Poisson': 1.09e-1 / 3.90e-5,
    }
    ax3.bar(improvements.keys(), improvements.values(), color='#2ECC71')
    ax3.set_ylabel('Accuracy Improvement (x)', fontsize=11)
    ax3.set_title('(c) SPECTO-ELM vs Vanilla PINN Accuracy', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for i, (task, imp) in enumerate(improvements.items()):
        ax3.text(i, imp + 50, f'{imp:.0f}x', ha='center', fontsize=10, fontweight='bold')

    # Subplot 4: Timing breakdown
    ax4 = fig.add_subplot(2, 2, 4)
    timing_data = {
        'Vanilla PINN': 72.74,
        'PIELM': 0.051,
        'SPECTO-1L': 0.012,
        'SPECTO-4L': 10.53,
    }
    ax4.barh(list(timing_data.keys()), list(timing_data.values()), color=['#E74C3C', '#F39C12', '#3498DB', '#2ECC71'])
    ax4.set_xlabel('Training Time (seconds)', fontsize=11)
    ax4.set_title('(d) Training Time Comparison', fontsize=12, fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3, axis='x')

    plt.suptitle('SPECTO-ELM: Comprehensive Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined figure to: {output_path}")


def main():
    """Generate all comparison figures."""
    print("="*70)
    print("GENERATING SPECTO-ELM COMPARISON FIGURES")
    print("="*70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'paper', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    # Generate individual figures
    create_accuracy_comparison(os.path.join(output_dir, 'specto_accuracy_comparison.png'))
    create_speedup_bar_chart(os.path.join(output_dir, 'specto_speedup_comparison.png'))
    create_accuracy_vs_speed_scatter(os.path.join(output_dir, 'specto_accuracy_vs_speed.png'))
    create_method_summary_table(os.path.join(output_dir, 'specto_method_summary.png'))
    create_combined_figure(os.path.join(output_dir, 'specto_combined_comparison.png'))

    print("\n" + "="*70)
    print("ALL FIGURES GENERATED")
    print("="*70)


if __name__ == '__main__':
    main()
