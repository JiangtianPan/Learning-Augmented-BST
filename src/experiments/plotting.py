"""
Plotting module for all experiment results.

Generates publication-quality figures:
  1. Synthetic: Expected Cost vs Alpha (line plot)
  2. Synthetic: Wall-clock Time vs Alpha
  3. Robustness: Expected Cost vs Prediction Error Level
  4. Real-data: Bar chart comparison
  5. Dynamic: Bar chart of avg search depth and time
"""

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    'Learning BST': '#e74c3c',
    'AVL Tree': '#3498db',
    'Treap': '#2ecc71',
    'Splay Tree': '#f39c12',
    'Entropy Bound': '#9b59b6',
}
MARKERS = {
    'Learning BST': 'o',
    'AVL Tree': 's',
    'Treap': '^',
    'Splay Tree': 'D',
    'Entropy Bound': '--',
}


def _setup_plot():
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'legend.fontsize': 11,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
    })


def plot_synthetic_results(results, output_dir='results'):
    """Plot synthetic experiment results.

    Args:
        results: dict {alpha: {tree_name: {metric: value}}}
        output_dir: directory to save plots.
    """
    _setup_plot()
    os.makedirs(output_dir, exist_ok=True)

    alphas = sorted(results.keys())
    tree_names = [name for name in results[alphas[0]].keys()
                  if name != 'Entropy Bound']

    # --- Plot 1: Expected Cost vs Alpha ---
    fig, ax = plt.subplots()
    for name in tree_names:
        costs = [results[a][name]['expected_cost'] for a in alphas]
        ax.plot(alphas, costs, marker=MARKERS.get(name, 'o'),
                color=COLORS.get(name, 'gray'), label=name, linewidth=2)

    if 'Entropy Bound' in results[alphas[0]]:
        entropy_costs = [results[a]['Entropy Bound']['expected_cost'] for a in alphas]
        ax.plot(alphas, entropy_costs, '--', color=COLORS['Entropy Bound'],
                label='Entropy Bound H(p)', linewidth=2)

    ax.set_xlabel('Zipfian Parameter α')
    ax.set_ylabel('Expected Cost Σ pᵢ · depth(keyᵢ)')
    ax.set_title('Expected Cost vs. Distribution Skewness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'synthetic_expected_cost.png'))
    plt.close()
    print(f"  Saved: {output_dir}/synthetic_expected_cost.png")

    # --- Plot 2: Average Access Cost vs Alpha ---
    fig, ax = plt.subplots()
    for name in tree_names:
        costs = [results[a][name]['avg_access_cost'] for a in alphas]
        ax.plot(alphas, costs, marker=MARKERS.get(name, 'o'),
                color=COLORS.get(name, 'gray'), label=name, linewidth=2)
    ax.set_xlabel('Zipfian Parameter α')
    ax.set_ylabel('Average Access Cost (depth per query)')
    ax.set_title('Average Query Depth vs. Distribution Skewness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'synthetic_avg_access_cost.png'))
    plt.close()
    print(f"  Saved: {output_dir}/synthetic_avg_access_cost.png")

    # --- Plot 3: Wall-clock Time vs Alpha ---
    fig, ax = plt.subplots()
    for name in tree_names:
        times = [results[a][name]['wall_clock_time'] for a in alphas]
        ax.plot(alphas, times, marker=MARKERS.get(name, 'o'),
                color=COLORS.get(name, 'gray'), label=name, linewidth=2)
    ax.set_xlabel('Zipfian Parameter α')
    ax.set_ylabel('Wall-clock Time (seconds)')
    ax.set_title('Query Time vs. Distribution Skewness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'synthetic_wall_clock.png'))
    plt.close()
    print(f"  Saved: {output_dir}/synthetic_wall_clock.png")

    # --- Plot 4: Tree Height vs Alpha ---
    fig, ax = plt.subplots()
    for name in tree_names:
        heights = [results[a][name]['height'] for a in alphas]
        ax.plot(alphas, heights, marker=MARKERS.get(name, 'o'),
                color=COLORS.get(name, 'gray'), label=name, linewidth=2)
    ax.set_xlabel('Zipfian Parameter α')
    ax.set_ylabel('Tree Height')
    ax.set_title('Tree Height vs. Distribution Skewness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'synthetic_tree_height.png'))
    plt.close()
    print(f"  Saved: {output_dir}/synthetic_tree_height.png")


def plot_robustness_results(results, output_dir='results'):
    """Plot robustness experiment results.

    Args:
        results: dict {error_level: {tree_name: {metric: value}}}
        output_dir: directory to save plots.
    """
    _setup_plot()
    os.makedirs(output_dir, exist_ok=True)

    errors = sorted(results.keys())
    tree_names = [name for name in results[errors[0]].keys()]

    # --- Expected Cost vs Error Level ---
    fig, ax = plt.subplots()
    for name in tree_names:
        costs = [results[e][name]['expected_cost'] for e in errors]
        ax.plot(errors, costs, marker=MARKERS.get(name, 'o'),
                color=COLORS.get(name, 'gray'), label=name, linewidth=2)
    ax.set_xlabel('Prediction Error Level')
    ax.set_ylabel('Expected Cost Σ pᵢ · depth(keyᵢ)')
    ax.set_title('Robustness: Expected Cost vs. Prediction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_expected_cost.png'))
    plt.close()
    print(f"  Saved: {output_dir}/robustness_expected_cost.png")

    # --- Average Access Cost vs Error Level ---
    fig, ax = plt.subplots()
    for name in tree_names:
        costs = [results[e][name]['avg_access_cost'] for e in errors]
        ax.plot(errors, costs, marker=MARKERS.get(name, 'o'),
                color=COLORS.get(name, 'gray'), label=name, linewidth=2)
    ax.set_xlabel('Prediction Error Level')
    ax.set_ylabel('Average Access Cost (depth per query)')
    ax.set_title('Robustness: Avg Query Depth vs. Prediction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_avg_access_cost.png'))
    plt.close()
    print(f"  Saved: {output_dir}/robustness_avg_access_cost.png")


def plot_real_data_results(results, output_dir='results'):
    """Plot real-data experiment results as bar charts.

    Args:
        results: dict {tree_name: {metric: value}}
        output_dir: directory to save plots.
    """
    _setup_plot()
    os.makedirs(output_dir, exist_ok=True)

    tree_names = [name for name in results.keys() if name != 'Entropy Bound']
    x = np.arange(len(tree_names))
    width = 0.5

    # --- Average Access Cost ---
    fig, ax = plt.subplots()
    costs = [results[name]['avg_access_cost'] for name in tree_names]
    colors = [COLORS.get(name, 'gray') for name in tree_names]
    bars = ax.bar(x, costs, width, color=colors)
    if 'Entropy Bound' in results:
        ax.axhline(y=results['Entropy Bound']['avg_access_cost'],
                    color=COLORS['Entropy Bound'], linestyle='--',
                    label='Entropy Bound', linewidth=2)
    ax.set_xlabel('Data Structure')
    ax.set_ylabel('Average Access Cost')
    ax.set_title('Real-Data: Average Access Cost Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(tree_names, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{cost:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'real_data_avg_cost.png'))
    plt.close()
    print(f"  Saved: {output_dir}/real_data_avg_cost.png")

    # --- Wall-clock Time ---
    fig, ax = plt.subplots()
    times = [results[name]['wall_clock_time'] for name in tree_names]
    bars = ax.bar(x, times, width, color=colors)
    ax.set_xlabel('Data Structure')
    ax.set_ylabel('Wall-clock Time (seconds)')
    ax.set_title('Real-Data: Query Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(tree_names, rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{t:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'real_data_wall_clock.png'))
    plt.close()
    print(f"  Saved: {output_dir}/real_data_wall_clock.png")


def plot_dynamic_results(results, output_dir='results'):
    """Plot dynamic update experiment results.

    Args:
        results: dict {tree_name: {metric: value}}
        output_dir: directory to save plots.
    """
    _setup_plot()
    os.makedirs(output_dir, exist_ok=True)

    tree_names = list(results.keys())
    x = np.arange(len(tree_names))
    width = 0.5

    # --- Average Search Depth ---
    fig, ax = plt.subplots()
    depths = [results[name]['avg_search_depth'] for name in tree_names]
    colors = [COLORS.get(name, 'gray') for name in tree_names]
    bars = ax.bar(x, depths, width, color=colors)
    ax.set_xlabel('Data Structure')
    ax.set_ylabel('Average Search Depth')
    ax.set_title('Dynamic Updates: Average Search Depth')
    ax.set_xticks(x)
    ax.set_xticklabels(tree_names, rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, d in zip(bars, depths):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{d:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dynamic_avg_search_depth.png'))
    plt.close()
    print(f"  Saved: {output_dir}/dynamic_avg_search_depth.png")

    # --- Wall-clock Time ---
    fig, ax = plt.subplots()
    times = [results[name]['wall_clock_time'] for name in tree_names]
    bars = ax.bar(x, times, width, color=colors)
    ax.set_xlabel('Data Structure')
    ax.set_ylabel('Wall-clock Time (seconds)')
    ax.set_title('Dynamic Updates: Total Operation Time')
    ax.set_xticks(x)
    ax.set_xticklabels(tree_names, rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{t:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dynamic_wall_clock.png'))
    plt.close()
    print(f"  Saved: {output_dir}/dynamic_wall_clock.png")

    # --- Final Height comparison ---
    fig, ax = plt.subplots()
    heights = [results[name]['final_height'] for name in tree_names]
    bars = ax.bar(x, heights, width, color=colors)
    ax.set_xlabel('Data Structure')
    ax.set_ylabel('Final Tree Height')
    ax.set_title('Dynamic Updates: Final Tree Height')
    ax.set_xticks(x)
    ax.set_xticklabels(tree_names, rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{h}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dynamic_final_height.png'))
    plt.close()
    print(f"  Saved: {output_dir}/dynamic_final_height.png")
