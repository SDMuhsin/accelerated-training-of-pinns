"""
Experiment B2: Depth vs Accuracy Curves

Purpose: Systematically show how accuracy changes with network depth.

Method:
- Run experiments with depth = 1, 2, 3, 4, 5, 6 layers (100 neurons each)
- Test on 6 representative tasks (3 where depth helps, 3 where it doesn't)
- Plot L2 error vs depth with error bars (multiple random seeds)
- Include solve time vs depth on secondary y-axis
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiment_dt_elm_pinn'))

from experiment_dt_elm_pinn.tasks import TaskRegistry
from experiment_dt_elm_pinn.models.dt_elm_pinn import DTELMPINN


# Tasks where depth helps
DEPTH_HELPS_TASKS = [
    'laplace-square',           # 73x improvement with 4L
    'nonlinear-poisson-square-sin',  # Significant improvement
    'heat-fast-decay',          # 2.7x improvement
]

# Tasks where depth may not help (or hurts)
DEPTH_NEUTRAL_TASKS = [
    'poisson-disk-sin',         # Slight degradation with depth
    'nonlinear-poisson',        # Minimal change
    'poisson-rbf-fd',           # Already very accurate
]

ALL_TASKS = DEPTH_HELPS_TASKS + DEPTH_NEUTRAL_TASKS
DEPTHS = [1, 2, 3, 4, 5, 6]
N_SEEDS = 5  # Number of random seeds for error bars


def run_depth_experiment(task_name, depths, seeds):
    """Run experiments at different depths with multiple seeds."""
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    task_cls = TaskRegistry.get(task_name)

    results = {depth: {'errors': [], 'times': []} for depth in depths}

    for depth in depths:
        hidden_sizes = [100] * depth
        print(f"  Depth {depth} ({hidden_sizes})...", end=" ")

        for seed in seeds:
            task = task_cls()
            _ = task.load_data()  # Load data once

            model = DTELMPINN(
                task,
                hidden_sizes=hidden_sizes,
                use_skip_connections=True,
                solver='robust',
                seed=seed,
            )

            model.setup()
            result = model.train()

            results[depth]['errors'].append(result.l2_error)
            results[depth]['times'].append(result.train_time)

        mean_error = np.mean(results[depth]['errors'])
        std_error = np.std(results[depth]['errors'])
        mean_time = np.mean(results[depth]['times']) * 1000

        print(f"error: {mean_error:.2e} ± {std_error:.2e}, time: {mean_time:.1f}ms")

    return results


def create_depth_curves(all_results, output_path):
    """Create depth vs accuracy curves for all tasks."""
    n_tasks = len(all_results)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (task_name, results) in enumerate(all_results.items()):
        ax = axes[idx]

        depths = list(results.keys())
        mean_errors = [np.mean(results[d]['errors']) for d in depths]
        std_errors = [np.std(results[d]['errors']) for d in depths]
        mean_times = [np.mean(results[d]['times']) * 1000 for d in depths]

        # Plot error (primary y-axis)
        color_error = '#2E86AB'
        ax.errorbar(depths, mean_errors, yerr=std_errors,
                   marker='o', markersize=8, linewidth=2, capsize=5,
                   color=color_error, label='L2 Error')
        ax.set_yscale('log')
        ax.set_xlabel('Network Depth (layers)', fontsize=11)
        ax.set_ylabel('L2 Error', fontsize=11, color=color_error)
        ax.tick_params(axis='y', labelcolor=color_error)

        # Plot time (secondary y-axis)
        ax2 = ax.twinx()
        color_time = '#C73E1D'
        ax2.plot(depths, mean_times, marker='s', markersize=6, linewidth=2,
                color=color_time, linestyle='--', label='Solve Time')
        ax2.set_ylabel('Solve Time (ms)', fontsize=11, color=color_time)
        ax2.tick_params(axis='y', labelcolor=color_time)

        # Determine if depth helps
        improvement = mean_errors[0] / mean_errors[-1] if mean_errors[-1] > 0 else 1
        if improvement > 2:
            verdict = f"Depth helps ({improvement:.1f}×)"
            title_color = 'green'
        elif improvement < 0.5:
            verdict = f"Depth hurts ({1/improvement:.1f}× worse)"
            title_color = 'red'
        else:
            verdict = "Depth neutral"
            title_color = 'gray'

        ax.set_title(f"{task_name}\n{verdict}", fontsize=11, fontweight='bold', color=title_color)
        ax.set_xticks(depths)
        ax.grid(True, alpha=0.3)

        # Mark optimal depth
        best_depth = depths[np.argmin(mean_errors)]
        ax.axvline(x=best_depth, color='green', linestyle=':', alpha=0.5, linewidth=2)

    # Add legend
    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[0].twinx().get_legend_handles_labels() if hasattr(axes[0], 'twinx') else ([], [])

    plt.suptitle('DISCO-ELM: Effect of Network Depth on Accuracy', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved depth curves to: {output_path}")


def create_summary_chart(all_results, output_path):
    """Create bar chart showing optimal depth and improvement for each task."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    tasks = list(all_results.keys())
    n_tasks = len(tasks)

    # Calculate metrics
    optimal_depths = []
    improvements = []
    error_at_1 = []
    error_at_best = []

    for task_name, results in all_results.items():
        depths = list(results.keys())
        mean_errors = [np.mean(results[d]['errors']) for d in depths]

        best_idx = np.argmin(mean_errors)
        optimal_depths.append(depths[best_idx])
        improvements.append(mean_errors[0] / mean_errors[best_idx] if mean_errors[best_idx] > 0 else 1)
        error_at_1.append(mean_errors[0])
        error_at_best.append(mean_errors[best_idx])

    x = np.arange(n_tasks)

    # Plot 1: Optimal depth
    ax = axes[0]
    colors = ['#4CAF50' if imp > 1.5 else '#F44336' if imp < 0.67 else '#FFC107' for imp in improvements]
    bars = ax.bar(x, optimal_depths, color=colors, edgecolor='white')

    for bar, depth in zip(bars, optimal_depths):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{depth}L', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('-', '\n').replace('nonlinear\npoisson', 'nl-poisson') for t in tasks], fontsize=9)
    ax.set_ylabel('Optimal Depth (layers)', fontsize=12)
    ax.set_title('Optimal Network Depth by Task', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(optimal_depths) + 1)

    # Plot 2: Improvement factor
    ax = axes[1]
    colors = ['#4CAF50' if imp > 1.5 else '#F44336' if imp < 0.67 else '#FFC107' for imp in improvements]
    bars = ax.bar(x, improvements, color=colors, edgecolor='white')

    for bar, imp in zip(bars, improvements):
        label = f'{imp:.1f}×' if imp >= 1 else f'{imp:.2f}×'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('-', '\n').replace('nonlinear\npoisson', 'nl-poisson') for t in tasks], fontsize=9)
    ax.set_ylabel('Improvement (1L error / best error)', fontsize=12)
    ax.set_title('Accuracy Improvement from Depth', fontsize=13, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Depth helps (>1.5×)'),
        Patch(facecolor='#FFC107', label='Depth neutral'),
        Patch(facecolor='#F44336', label='Depth hurts (<0.67×)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary chart to: {output_path}")


def create_markdown_report(all_results, output_path):
    """Create markdown report."""
    lines = []
    lines.append("# B2: Depth vs Accuracy Analysis\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("\n## Summary\n")
    lines.append("This experiment systematically evaluates how network depth affects DISCO-ELM accuracy.")
    lines.append(f"- Depths tested: {DEPTHS}")
    lines.append(f"- Random seeds per depth: {N_SEEDS}")
    lines.append(f"- Total experiments: {len(ALL_TASKS)} tasks × {len(DEPTHS)} depths × {N_SEEDS} seeds = {len(ALL_TASKS) * len(DEPTHS) * N_SEEDS}\n")

    lines.append("\n## Results Table\n")
    lines.append("| Task | 1L Error | Best Error | Best Depth | Improvement | Category |")
    lines.append("|------|----------|------------|------------|-------------|----------|")

    for task_name, results in all_results.items():
        depths = list(results.keys())
        mean_errors = [np.mean(results[d]['errors']) for d in depths]

        best_idx = np.argmin(mean_errors)
        error_1L = mean_errors[0]
        error_best = mean_errors[best_idx]
        best_depth = depths[best_idx]
        improvement = error_1L / error_best if error_best > 0 else 1

        if improvement > 1.5:
            category = "Depth helps"
        elif improvement < 0.67:
            category = "Depth hurts"
        else:
            category = "Neutral"

        lines.append(f"| {task_name} | {error_1L:.2e} | {error_best:.2e} | {best_depth}L | {improvement:.1f}× | {category} |")

    lines.append("\n## Key Findings\n")

    # Calculate statistics
    helps = sum(1 for _, r in all_results.items()
               if np.mean(r[1]['errors']) / np.mean(r[min(r.keys(), key=lambda d: np.mean(r[d]['errors']))]['errors']) > 1.5)
    hurts = sum(1 for _, r in all_results.items()
               if np.mean(r[1]['errors']) / np.mean(r[min(r.keys(), key=lambda d: np.mean(r[d]['errors']))]['errors']) < 0.67)
    neutral = len(all_results) - helps - hurts

    lines.append(f"1. **Depth helps on {helps}/{len(all_results)} tasks** (>1.5× improvement)")
    lines.append(f"2. **Depth hurts on {hurts}/{len(all_results)} tasks** (<0.67× improvement)")
    lines.append(f"3. **Depth neutral on {neutral}/{len(all_results)} tasks**")
    lines.append("")
    lines.append("### When Depth Helps:")
    lines.append("- Square domains with boundary discontinuities (laplace-square)")
    lines.append("- Problems requiring multi-scale features")
    lines.append("- Nonlinear PDEs with complex solution structure")
    lines.append("")
    lines.append("### When Depth Doesn't Help:")
    lines.append("- Simple smooth solutions (poisson-rbf-fd)")
    lines.append("- Already highly accurate single-layer results")
    lines.append("- Some sinusoidal source functions")

    lines.append("\n## Recommendation for Paper\n")
    lines.append("- Report that depth is **task-dependent**")
    lines.append("- Recommend starting with 1-layer for speed, increase depth if accuracy insufficient")
    lines.append("- 4-layer is a good default for challenging problems")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved report to: {output_path}")


def main():
    """Run depth analysis experiment."""
    print("="*70)
    print("EXPERIMENT B2: DEPTH VS ACCURACY ANALYSIS")
    print("="*70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    seeds = list(range(42, 42 + N_SEEDS))

    all_results = {}
    for task_name in ALL_TASKS:
        try:
            results = run_depth_experiment(task_name, DEPTHS, seeds)
            all_results[task_name] = results
        except Exception as e:
            print(f"ERROR on {task_name}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        create_depth_curves(all_results, os.path.join(output_dir, 'b2_depth_curves.png'))
        create_summary_chart(all_results, os.path.join(output_dir, 'b2_depth_summary.png'))
        create_markdown_report(all_results, os.path.join(output_dir, 'b2_depth_analysis.md'))

    print("\n" + "="*70)
    print("EXPERIMENT B2 COMPLETE")
    print("="*70)

    return all_results


if __name__ == '__main__':
    main()
