"""
Experiment D3: Random Seed Sensitivity

Purpose: Quantify variance from random weight initialization.

Method:
- Run each of 14 tasks with 20 different random seeds
- Compute mean, std, min, max of L2 error
- Create box plots showing error distribution per task
- Compare variance for 1-layer vs 4-layer
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


# All 14 tasks
ALL_TASKS = [
    'poisson-rbf-fd',
    'poisson-disk-sin',
    'poisson-disk-quadratic',
    'poisson-square-constant',
    'poisson-square-sin',
    'nonlinear-poisson',
    'nonlinear-poisson-rbf-fd',
    'nonlinear-poisson-disk-sin',
    'nonlinear-poisson-square-constant',
    'nonlinear-poisson-square-sin',
    'laplace-disk',
    'laplace-square',
    'heat-equation',
    'heat-fast-decay',
]

N_SEEDS = 20  # Number of random seeds to test


def run_seed_study(task_name, seeds, depths=[1, 4]):
    """Run experiments with multiple random seeds."""
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    task_cls = TaskRegistry.get(task_name)

    results = {depth: [] for depth in depths}

    for depth in depths:
        hidden_sizes = [100] * depth
        print(f"  Depth {depth} ({hidden_sizes}):", end=" ")

        for seed in seeds:
            task = task_cls()
            task.load_data()

            model = DTELMPINN(
                task,
                hidden_sizes=hidden_sizes,
                use_skip_connections=True,
                solver='robust',
                seed=seed,
            )
            model.setup()
            result = model.train()

            if result.l2_error is not None:
                results[depth].append(result.l2_error)

        errors = results[depth]
        if errors:
            print(f"mean={np.mean(errors):.2e}, std={np.std(errors):.2e}, "
                  f"min={np.min(errors):.2e}, max={np.max(errors):.2e}")
        else:
            print("NO DATA")

    return results


def create_box_plots(all_results, output_path):
    """Create box plots showing error distribution per task."""
    n_tasks = len(all_results)
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    tasks = list(all_results.keys())

    # Prepare data for box plots
    data_1L = []
    data_4L = []
    labels = []

    for task in tasks:
        results = all_results[task]
        if results.get(1):
            data_1L.append(results[1])
        else:
            data_1L.append([np.nan])
        if results.get(4):
            data_4L.append(results[4])
        else:
            data_4L.append([np.nan])
        labels.append(task.replace('-', '\n').replace('nonlinear\npoisson', 'nl-pois'))

    # Plot 1: 1-Layer box plots
    ax = axes[0]
    bp = ax.boxplot(data_1L, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#2E86AB')
        patch.set_alpha(0.7)
    ax.set_yscale('log')
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('DISCO-ELM 1-Layer: Error Distribution Across Random Seeds',
                fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: 4-Layer box plots
    ax = axes[1]
    bp = ax.boxplot(data_4L, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#C73E1D')
        patch.set_alpha(0.7)
    ax.set_yscale('log')
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('DISCO-ELM 4-Layer: Error Distribution Across Random Seeds',
                fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved box plots to: {output_path}")


def create_variance_comparison(all_results, output_path):
    """Create figure comparing variance between 1L and 4L."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    tasks = list(all_results.keys())
    x = np.arange(len(tasks))

    # Calculate coefficient of variation (std/mean) for each task
    cv_1L = []
    cv_4L = []
    means_1L = []
    means_4L = []

    for task in tasks:
        results = all_results[task]

        if results.get(1) and len(results[1]) > 1:
            cv_1L.append(np.std(results[1]) / np.mean(results[1]) * 100)
            means_1L.append(np.mean(results[1]))
        else:
            cv_1L.append(np.nan)
            means_1L.append(np.nan)

        if results.get(4) and len(results[4]) > 1:
            cv_4L.append(np.std(results[4]) / np.mean(results[4]) * 100)
            means_4L.append(np.mean(results[4]))
        else:
            cv_4L.append(np.nan)
            means_4L.append(np.nan)

    width = 0.35

    # Plot 1: Coefficient of variation comparison
    ax = axes[0]
    bars1 = ax.bar(x - width/2, cv_1L, width, label='1-Layer', color='#2E86AB')
    bars2 = ax.bar(x + width/2, cv_4L, width, label='4-Layer', color='#C73E1D')

    ax.set_ylabel('Coefficient of Variation (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('-', '\n').replace('nonlinear\npoisson', 'nl-pois') for t in tasks],
                      fontsize=7, rotation=45, ha='right')
    ax.set_title('Variability: How Much Does Random Seed Affect Accuracy?',
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add threshold line for "stable" (CV < 10%)
    ax.axhline(y=10, color='green', linestyle='--', linewidth=1, label='Stable threshold (10%)')

    # Plot 2: Mean error comparison
    ax = axes[1]
    bars1 = ax.bar(x - width/2, means_1L, width, label='1-Layer', color='#2E86AB')
    bars2 = ax.bar(x + width/2, means_4L, width, label='4-Layer', color='#C73E1D')

    ax.set_yscale('log')
    ax.set_ylabel('Mean L2 Error', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('-', '\n').replace('nonlinear\npoisson', 'nl-pois') for t in tasks],
                      fontsize=7, rotation=45, ha='right')
    ax.set_title('Mean Error Comparison: 1-Layer vs 4-Layer',
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved variance comparison to: {output_path}")


def create_markdown_report(all_results, output_path):
    """Create markdown report."""
    lines = []
    lines.append("# D3: Random Seed Sensitivity Analysis\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("\n## Summary\n")
    lines.append(f"This experiment quantifies the variance from random weight initialization.")
    lines.append(f"- Tasks tested: {len(ALL_TASKS)}")
    lines.append(f"- Random seeds per task: {N_SEEDS}")
    lines.append(f"- Depths compared: 1-layer and 4-layer\n")

    lines.append("\n## Results Table\n")
    lines.append("| Task | 1L Mean | 1L Std | 1L CV% | 4L Mean | 4L Std | 4L CV% |")
    lines.append("|------|---------|--------|--------|---------|--------|--------|")

    stable_1L = 0
    stable_4L = 0
    total_tasks = 0

    for task, results in all_results.items():
        total_tasks += 1

        if results.get(1) and len(results[1]) > 1:
            mean_1L = np.mean(results[1])
            std_1L = np.std(results[1])
            cv_1L = std_1L / mean_1L * 100
            if cv_1L < 10:
                stable_1L += 1
        else:
            mean_1L = std_1L = cv_1L = np.nan

        if results.get(4) and len(results[4]) > 1:
            mean_4L = np.mean(results[4])
            std_4L = np.std(results[4])
            cv_4L = std_4L / mean_4L * 100
            if cv_4L < 10:
                stable_4L += 1
        else:
            mean_4L = std_4L = cv_4L = np.nan

        lines.append(f"| {task} | {mean_1L:.2e} | {std_1L:.2e} | {cv_1L:.1f}% | "
                    f"{mean_4L:.2e} | {std_4L:.2e} | {cv_4L:.1f}% |")

    lines.append("\n## Key Findings\n")
    lines.append(f"1. **1-Layer stability**: {stable_1L}/{total_tasks} tasks have CV < 10%")
    lines.append(f"2. **4-Layer stability**: {stable_4L}/{total_tasks} tasks have CV < 10%")

    # Calculate average CV
    avg_cv_1L = []
    avg_cv_4L = []
    for task, results in all_results.items():
        if results.get(1) and len(results[1]) > 1:
            avg_cv_1L.append(np.std(results[1]) / np.mean(results[1]) * 100)
        if results.get(4) and len(results[4]) > 1:
            avg_cv_4L.append(np.std(results[4]) / np.mean(results[4]) * 100)

    if avg_cv_1L:
        lines.append(f"3. **Average CV (1-Layer)**: {np.mean(avg_cv_1L):.1f}%")
    if avg_cv_4L:
        lines.append(f"4. **Average CV (4-Layer)**: {np.mean(avg_cv_4L):.1f}%")

    lines.append("\n## Stability Classification\n")
    lines.append("| Stability | Description | CV Threshold |")
    lines.append("|-----------|-------------|--------------|")
    lines.append("| High | Very reproducible | CV < 5% |")
    lines.append("| Medium | Moderate variability | 5% ≤ CV < 20% |")
    lines.append("| Low | High variability | CV ≥ 20% |")

    lines.append("\n### Tasks by Stability (1-Layer):\n")
    high_stable = []
    med_stable = []
    low_stable = []
    for task, results in all_results.items():
        if results.get(1) and len(results[1]) > 1:
            cv = np.std(results[1]) / np.mean(results[1]) * 100
            if cv < 5:
                high_stable.append(task)
            elif cv < 20:
                med_stable.append(task)
            else:
                low_stable.append(task)

    lines.append(f"- **High stability**: {', '.join(high_stable) if high_stable else 'None'}")
    lines.append(f"- **Medium stability**: {', '.join(med_stable) if med_stable else 'None'}")
    lines.append(f"- **Low stability**: {', '.join(low_stable) if low_stable else 'None'}")

    lines.append("\n## Recommendation for Paper\n")
    lines.append("- Report mean ± std for all experiments")
    lines.append("- Use 5+ random seeds for statistical significance")
    lines.append("- Note which tasks have high variance (may need more seeds)")
    lines.append("- Consider ensemble methods for high-variance tasks")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved report to: {output_path}")


def main():
    """Run random seed sensitivity experiment."""
    print("="*70)
    print("EXPERIMENT D3: RANDOM SEED SENSITIVITY")
    print("="*70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    seeds = list(range(0, N_SEEDS))

    all_results = {}
    for task_name in ALL_TASKS:
        try:
            results = run_seed_study(task_name, seeds)
            all_results[task_name] = results
        except Exception as e:
            print(f"ERROR on {task_name}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        create_box_plots(all_results, os.path.join(output_dir, 'd3_seed_sensitivity_boxplots.png'))
        create_variance_comparison(all_results, os.path.join(output_dir, 'd3_seed_sensitivity_variance.png'))
        create_markdown_report(all_results, os.path.join(output_dir, 'd3_random_seed_sensitivity.md'))

    print("\n" + "="*70)
    print("EXPERIMENT D3 COMPLETE")
    print("="*70)

    return all_results


if __name__ == '__main__':
    main()
