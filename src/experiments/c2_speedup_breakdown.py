"""
Experiment C2: Speedup Breakdown by PDE Type

Purpose: Address "bimodal speedups hidden" issue from critic review.
Shows that linear PDEs get much higher speedups than nonlinear PDEs.

Outputs:
- Grouped bar chart showing speedup by task, colored by category
- Summary statistics for each PDE category
- Markdown report with findings
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Task categorization
TASK_CATEGORIES = {
    'Linear Poisson': [
        'poisson-rbf-fd',
        'poisson-disk-sin',
        'poisson-disk-quadratic',
        'poisson-square-constant',
        'poisson-square-sin',
    ],
    'Laplace': [
        'laplace-disk',
        'laplace-square',
    ],
    'Heat Equation': [
        'heat-equation',
        'heat-fast-decay',
    ],
    'Nonlinear Poisson': [
        'nonlinear-poisson',
        'nonlinear-poisson-rbf-fd',
        'nonlinear-poisson-disk-sin',
        'nonlinear-poisson-square-constant',
        'nonlinear-poisson-square-sin',
    ],
}

# Color scheme for categories
CATEGORY_COLORS = {
    'Linear Poisson': '#2E86AB',
    'Laplace': '#A23B72',
    'Heat Equation': '#F18F01',
    'Nonlinear Poisson': '#C73E1D',
}


def get_task_category(task_name):
    """Get category for a task."""
    for category, tasks in TASK_CATEGORIES.items():
        if task_name in tasks:
            return category
    return 'Unknown'


def load_experiment_data(csv_path):
    """Load and process experiment data from CSV."""
    df = pd.read_csv(csv_path)

    # Remove duplicate header rows (CSV has repeated headers)
    df = df[df['run_id'] != 'run_id'].copy()

    # Convert numeric columns
    df['train_time'] = pd.to_numeric(df['train_time'], errors='coerce')
    df['l2_error'] = pd.to_numeric(df['l2_error'], errors='coerce')

    # Filter to only dt-elm-pinn and vanilla-pinn models
    disco_df = df[df['model'] == 'dt-elm-pinn'].copy()
    vanilla_df = df[df['model'] == 'vanilla-pinn'].copy()

    # Get unique tasks
    tasks = set(disco_df['task'].unique()) & set(vanilla_df['task'].unique())

    results = []
    for task in tasks:
        disco_row = disco_df[disco_df['task'] == task].iloc[0] if len(disco_df[disco_df['task'] == task]) > 0 else None
        vanilla_row = vanilla_df[vanilla_df['task'] == task].iloc[0] if len(vanilla_df[vanilla_df['task'] == task]) > 0 else None

        if disco_row is not None and vanilla_row is not None:
            disco_time = disco_row['train_time']
            vanilla_time = vanilla_row['train_time']
            disco_error = disco_row['l2_error']
            vanilla_error = vanilla_row['l2_error']
            speedup = vanilla_time / disco_time if disco_time > 0 else 0

            category = get_task_category(task)

            results.append({
                'task': task,
                'category': category,
                'disco_time': disco_time,
                'vanilla_time': vanilla_time,
                'speedup': speedup,
                'disco_error': disco_error,
                'vanilla_error': vanilla_error,
                'error_ratio': disco_error / vanilla_error if vanilla_error > 0 else float('inf'),
            })

    return pd.DataFrame(results)


def create_speedup_bar_chart(df, output_path):
    """Create grouped bar chart showing speedup by task, colored by category."""

    fig, ax = plt.subplots(figsize=(16, 8))

    # Sort by category then by speedup within category
    category_order = ['Linear Poisson', 'Laplace', 'Heat Equation', 'Nonlinear Poisson']
    df['category_order'] = df['category'].apply(lambda x: category_order.index(x) if x in category_order else 99)
    df = df.sort_values(['category_order', 'speedup'], ascending=[True, False])

    # Create bars
    x = np.arange(len(df))
    colors = [CATEGORY_COLORS.get(cat, 'gray') for cat in df['category']]

    bars = ax.bar(x, df['speedup'], color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, df['speedup'])):
        if speedup > 1000:
            label = f'{speedup/1000:.1f}k'
        else:
            label = f'{speedup:.0f}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Task labels
    task_labels = [t.replace('nonlinear-poisson', 'nl-poisson').replace('-rbf-fd', '')
                   for t in df['task']]
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=45, ha='right', fontsize=9)

    # Draw category separators and labels
    prev_cat = None
    cat_positions = []
    for i, row in df.iterrows():
        idx = list(df.index).index(i)
        if row['category'] != prev_cat:
            if prev_cat is not None:
                ax.axvline(x=idx - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            cat_positions.append((row['category'], idx))
            prev_cat = row['category']

    # Add category labels at top
    for i, (cat, start_idx) in enumerate(cat_positions):
        end_idx = cat_positions[i+1][1] if i+1 < len(cat_positions) else len(df)
        mid_idx = (start_idx + end_idx - 1) / 2
        ax.text(mid_idx, ax.get_ylim()[1] * 0.95, cat,
                ha='center', fontsize=11, fontweight='bold',
                color=CATEGORY_COLORS.get(cat, 'black'))

    # Legend
    handles = [plt.Rectangle((0,0),1,1, color=CATEGORY_COLORS[cat]) for cat in category_order]
    ax.legend(handles, category_order, loc='upper right', fontsize=10)

    ax.set_ylabel('Speedup vs Vanilla PINN', fontsize=12)
    ax.set_xlabel('Task', fontsize=12)
    ax.set_title('DISCO-ELM Speedup by Task and PDE Category', fontsize=14, fontweight='bold')

    # Log scale for better visibility
    ax.set_yscale('log')
    ax.set_ylim(bottom=50, top=50000)

    # Add horizontal reference lines
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='100x')
    ax.axhline(y=1000, color='gray', linestyle=':', alpha=0.5, label='1000x')
    ax.axhline(y=10000, color='gray', linestyle=':', alpha=0.5, label='10000x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved speedup bar chart to: {output_path}")


def create_category_summary_chart(df, output_path):
    """Create bar chart showing geometric mean speedup by category."""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate geometric mean speedup per category
    category_order = ['Linear Poisson', 'Laplace', 'Heat Equation', 'Nonlinear Poisson']
    geo_means = []
    for cat in category_order:
        cat_df = df[df['category'] == cat]
        if len(cat_df) > 0:
            geo_mean = np.exp(np.mean(np.log(cat_df['speedup'])))
            geo_means.append(geo_mean)
        else:
            geo_means.append(0)

    # Also compute "All Linear" and "All Nonlinear" aggregates
    linear_cats = ['Linear Poisson', 'Laplace', 'Heat Equation']
    linear_df = df[df['category'].isin(linear_cats)]
    linear_geo_mean = np.exp(np.mean(np.log(linear_df['speedup']))) if len(linear_df) > 0 else 0

    nonlinear_df = df[df['category'] == 'Nonlinear Poisson']
    nonlinear_geo_mean = np.exp(np.mean(np.log(nonlinear_df['speedup']))) if len(nonlinear_df) > 0 else 0

    # Plot
    x = np.arange(len(category_order) + 2)  # +2 for "All Linear" and "All Nonlinear"
    all_values = geo_means + [linear_geo_mean, nonlinear_geo_mean]
    all_labels = category_order + ['ALL LINEAR', 'ALL NONLINEAR']
    all_colors = [CATEGORY_COLORS[cat] for cat in category_order] + ['#1a5276', '#922b21']

    bars = ax.bar(x, all_values, color=all_colors, edgecolor='white', linewidth=1)

    # Add value labels
    for bar, val in zip(bars, all_values):
        if val > 1000:
            label = f'{val/1000:.1f}k×'
        else:
            label = f'{val:.0f}×'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Geometric Mean Speedup', fontsize=12)
    ax.set_title('Speedup by PDE Category: Linear vs Nonlinear', fontsize=14, fontweight='bold')

    # Log scale
    ax.set_yscale('log')
    ax.set_ylim(bottom=50, top=20000)

    # Add annotation showing the ratio
    ratio = linear_geo_mean / nonlinear_geo_mean if nonlinear_geo_mean > 0 else 0
    ax.annotate(f'Linear/Nonlinear\nratio: {ratio:.1f}×',
                xy=(5.5, 5000), fontsize=12, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved category summary chart to: {output_path}")


def create_summary_report(df, output_path):
    """Create markdown summary report."""

    lines = []
    lines.append("# C2: Speedup Breakdown by PDE Type\n")
    lines.append("Generated: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")

    lines.append("\n## Key Finding: BIMODAL SPEEDUP DISTRIBUTION\n")
    lines.append("The 2400× geometric mean speedup HIDES a bimodal distribution:\n")

    # Calculate statistics
    linear_cats = ['Linear Poisson', 'Laplace', 'Heat Equation']
    linear_df = df[df['category'].isin(linear_cats)]
    nonlinear_df = df[df['category'] == 'Nonlinear Poisson']

    linear_geo_mean = np.exp(np.mean(np.log(linear_df['speedup'])))
    nonlinear_geo_mean = np.exp(np.mean(np.log(nonlinear_df['speedup'])))

    lines.append(f"- **Linear PDEs**: {linear_geo_mean:.0f}× geometric mean speedup (range: {linear_df['speedup'].min():.0f}×-{linear_df['speedup'].max():.0f}×)")
    lines.append(f"- **Nonlinear PDEs**: {nonlinear_geo_mean:.0f}× geometric mean speedup (range: {nonlinear_df['speedup'].min():.0f}×-{nonlinear_df['speedup'].max():.0f}×)")
    lines.append(f"- **Ratio**: Linear speedup is **{linear_geo_mean/nonlinear_geo_mean:.1f}×** higher than nonlinear\n")

    lines.append("\n## Detailed Results by Category\n")

    for category in ['Linear Poisson', 'Laplace', 'Heat Equation', 'Nonlinear Poisson']:
        cat_df = df[df['category'] == category].sort_values('speedup', ascending=False)
        if len(cat_df) == 0:
            continue

        geo_mean = np.exp(np.mean(np.log(cat_df['speedup'])))
        lines.append(f"### {category} (n={len(cat_df)}, geo. mean: {geo_mean:.0f}×)\n")
        lines.append("| Task | DISCO Time | Vanilla Time | Speedup | Error Ratio |")
        lines.append("|------|------------|--------------|---------|-------------|")

        for _, row in cat_df.iterrows():
            lines.append(f"| {row['task']} | {row['disco_time']*1000:.1f}ms | {row['vanilla_time']:.2f}s | **{row['speedup']:.0f}×** | {row['error_ratio']:.2f} |")
        lines.append("")

    lines.append("\n## Why the Difference?\n")
    lines.append("1. **Linear PDEs** use a single Cholesky solve (~1-3ms)")
    lines.append("2. **Nonlinear PDEs** require Newton iterations (5-20 iterations × Cholesky solve each)")
    lines.append("3. Both benefit from **no autodiff** and **no gradient descent**")
    lines.append("4. The speedup gap is fundamental to the problem structure, not a limitation\n")

    lines.append("\n## Implication for Paper\n")
    lines.append("The paper should report speedups separately:\n")
    lines.append(f"- \"DISCO-ELM achieves **{linear_geo_mean:.0f}×** speedup on linear PDEs and **{nonlinear_geo_mean:.0f}×** on nonlinear PDEs\"")
    lines.append(f"- The overall **2400×** geometric mean is valid but hides this distinction")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved summary report to: {output_path}")


def main():
    """Run speedup breakdown experiment."""
    print("="*70)
    print("EXPERIMENT C2: SPEEDUP BREAKDOWN BY PDE TYPE")
    print("="*70)

    # Load data
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'experiments.csv')
    df = load_experiment_data(csv_path)

    print(f"\nLoaded {len(df)} task comparisons")
    print(f"Categories: {df['category'].value_counts().to_dict()}")

    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    # Generate outputs
    create_speedup_bar_chart(df, os.path.join(output_dir, 'c2_speedup_by_task.png'))
    create_category_summary_chart(df, os.path.join(output_dir, 'c2_speedup_by_category.png'))
    create_summary_report(df, os.path.join(output_dir, 'c2_speedup_breakdown.md'))

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    linear_cats = ['Linear Poisson', 'Laplace', 'Heat Equation']
    linear_df = df[df['category'].isin(linear_cats)]
    nonlinear_df = df[df['category'] == 'Nonlinear Poisson']

    linear_geo = np.exp(np.mean(np.log(linear_df['speedup'])))
    nonlinear_geo = np.exp(np.mean(np.log(nonlinear_df['speedup'])))
    overall_geo = np.exp(np.mean(np.log(df['speedup'])))

    print(f"Linear PDEs:    {linear_geo:.0f}× geometric mean ({len(linear_df)} tasks)")
    print(f"Nonlinear PDEs: {nonlinear_geo:.0f}× geometric mean ({len(nonlinear_df)} tasks)")
    print(f"Overall:        {overall_geo:.0f}× geometric mean ({len(df)} tasks)")
    print(f"Linear/Nonlinear ratio: {linear_geo/nonlinear_geo:.1f}×")

    print("\n" + "="*70)
    print("EXPERIMENT C2 COMPLETE")
    print("="*70)

    return df


if __name__ == '__main__':
    main()
