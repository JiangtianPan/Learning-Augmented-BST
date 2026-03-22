"""
Main runner for Learning Augmented BST project.

Runs all experiments and generates plots:
  1. Synthetic Experiment (varying Zipfian alpha)
  2. Robustness Experiment (prediction error simulation)
  3. Real-Data Experiment
  4. Dynamic Updates Experiment

Usage:
    python main.py              # Run all experiments
    python main.py synthetic    # Run only synthetic experiment
    python main.py robustness   # Run only robustness experiment
    python main.py realdata     # Run only real-data experiment
    python main.py dynamic      # Run only dynamic updates experiment
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiments.synthetic_experiment import run_synthetic_experiment
from src.experiments.robustness_experiment import run_robustness_experiment
from src.experiments.real_data_experiment import run_real_data_experiment
from src.experiments.dynamic_experiment import run_dynamic_experiment
from src.experiments.plotting import (plot_synthetic_results,
                                       plot_robustness_results,
                                       plot_real_data_results,
                                       plot_dynamic_results)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def save_results_json(results, filename):
    """Save results to JSON for reproducibility."""
    filepath = os.path.join(RESULTS_DIR, filename)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Convert keys to strings for JSON serialization
    serializable = {}
    for k, v in results.items():
        key_str = str(k)
        if isinstance(v, dict):
            serializable[key_str] = {}
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    serializable[key_str][str(k2)] = {
                        str(k3): (float(v3) if isinstance(v3, (int, float)) else v3)
                        for k3, v3 in v2.items()
                    }
                else:
                    serializable[key_str][str(k2)] = (
                        float(v2) if isinstance(v2, (int, float)) else v2
                    )
        else:
            serializable[key_str] = float(v) if isinstance(v, (int, float)) else v

    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {filepath}")


def run_all():
    experiments = sys.argv[1:] if len(sys.argv) > 1 else ['synthetic', 'robustness', 'realdata', 'dynamic']

    if 'synthetic' in experiments:
        print("=" * 60)
        print("EXPERIMENT 1: Synthetic (Varying Zipfian Alpha)")
        print("=" * 60)
        results = run_synthetic_experiment(n=500, num_queries=50000)
        save_results_json(results, 'synthetic_results.json')
        print("\nGenerating plots...")
        plot_synthetic_results(results, RESULTS_DIR)

    if 'robustness' in experiments:
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: Robustness (Prediction Error)")
        print("=" * 60)
        results = run_robustness_experiment(n=500, alpha=1.5, num_queries=50000)
        save_results_json(results, 'robustness_results.json')
        print("\nGenerating plots...")
        plot_robustness_results(results, RESULTS_DIR)

    if 'realdata' in experiments:
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: Real-Data")
        print("=" * 60)
        results = run_real_data_experiment(data_dir=DATA_DIR, n=1000,
                                            num_interactions=50000)
        save_results_json(results, 'real_data_results.json')
        print("\nGenerating plots...")
        plot_real_data_results(results, RESULTS_DIR)

    if 'dynamic' in experiments:
        print("\n" + "=" * 60)
        print("EXPERIMENT 4: Dynamic Updates")
        print("=" * 60)
        results = run_dynamic_experiment(n=500, alpha=1.5, num_operations=10000)
        save_results_json(results, 'dynamic_results.json')
        print("\nGenerating plots...")
        plot_dynamic_results(results, RESULTS_DIR)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results and plots saved to: {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    run_all()
