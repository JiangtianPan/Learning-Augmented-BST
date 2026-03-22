"""
Robustness Experiment: Prediction Error Simulation.

Train the Learning BST on distribution D, then query with a different
distribution D' (permuted frequencies). Test whether the data structure
degrades to O(n) or maintains O(log n).

Error levels:
  - 0% error: D' = D (perfect prediction)
  - Partial error: D' is a noisy version of D
  - 100% error: D' is a random permutation of D's probabilities
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from src.trees.learning_bst import LearningBST
from src.trees.avl_tree import AVLTree
from src.trees.splay_tree import SplayTree
from src.trees.treap import Treap
from src.metrics.cost import expected_cost, timed_queries
from src.data.zipfian import zipfian_distribution


def _mix_distributions(true_probs, permuted_probs, error_level):
    """Linearly interpolate between true and permuted distributions.

    error_level=0 -> true_probs (perfect prediction)
    error_level=1 -> permuted_probs (completely wrong)
    """
    mixed = (1 - error_level) * true_probs + error_level * permuted_probs
    mixed /= mixed.sum()
    return mixed


def run_robustness_experiment(n=500, alpha=1.5, num_queries=50000,
                               error_levels=None, seed=42):
    """Run the robustness experiment.

    Args:
        n: number of distinct keys.
        alpha: Zipfian alpha for the true query distribution.
        num_queries: number of test queries.
        error_levels: list of error levels to sweep (0.0 to 1.0).
        seed: random seed.

    Returns:
        results: dict {error_level: {tree_name: {metric: value}}}
    """
    if error_levels is None:
        error_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    rng = np.random.default_rng(seed)

    # True query distribution
    true_probs = zipfian_distribution(n, alpha)
    # Permuted distribution (completely wrong predictions)
    perm = rng.permutation(n)
    permuted_probs = true_probs[perm]

    keys = list(range(n))
    results = {}

    for error in error_levels:
        print(f"\n--- Error Level = {error:.1f} ---")

        # Predicted distribution (what the Learning BST is built on)
        predicted_probs = _mix_distributions(true_probs, permuted_probs, error)

        # Actual query distribution remains the true one
        actual_probs = {k: float(true_probs[k]) for k in keys}
        queries = rng.choice(keys, size=num_queries, p=true_probs).tolist()

        error_results = {}

        # --- Learning BST (built on predicted probs) ---
        lbst = LearningBST()
        lbst.build([(k, float(predicted_probs[k])) for k in keys])
        ec = expected_cost(lbst, actual_probs)
        total_depth, elapsed = timed_queries(lbst, queries)
        avg_cost = total_depth / len(queries)
        error_results['Learning BST'] = {
            'expected_cost': ec,
            'avg_access_cost': avg_cost,
            'wall_clock_time': elapsed,
            'height': lbst.height(),
        }
        print(f"  Learning BST: expected_cost={ec:.4f}, avg_depth={avg_cost:.4f}")

        # --- AVL Tree (unaffected by predictions) ---
        avl = AVLTree()
        avl.build(keys)
        ec = expected_cost(avl, actual_probs)
        total_depth, elapsed = timed_queries(avl, queries)
        avg_cost = total_depth / len(queries)
        error_results['AVL Tree'] = {
            'expected_cost': ec,
            'avg_access_cost': avg_cost,
            'wall_clock_time': elapsed,
            'height': avl.height(),
        }
        print(f"  AVL Tree:     expected_cost={ec:.4f}, avg_depth={avg_cost:.4f}")

        # --- Treap ---
        treap = Treap(seed=seed)
        treap.build(keys)
        ec = expected_cost(treap, actual_probs)
        total_depth, elapsed = timed_queries(treap, queries)
        avg_cost = total_depth / len(queries)
        error_results['Treap'] = {
            'expected_cost': ec,
            'avg_access_cost': avg_cost,
            'wall_clock_time': elapsed,
            'height': treap.height(),
        }
        print(f"  Treap:        expected_cost={ec:.4f}, avg_depth={avg_cost:.4f}")

        # --- Splay Tree ---
        splay = SplayTree()
        for k in keys:
            splay.insert(k)
        total_depth, elapsed = timed_queries(splay, queries)
        avg_cost = total_depth / len(queries)
        ec_post = expected_cost(splay, actual_probs)
        error_results['Splay Tree'] = {
            'expected_cost': ec_post,
            'avg_access_cost': avg_cost,
            'wall_clock_time': elapsed,
            'height': splay.height(),
        }
        print(f"  Splay Tree:   expected_cost={ec_post:.4f}, avg_depth={avg_cost:.4f}")

        results[error] = error_results

    return results


if __name__ == '__main__':
    results = run_robustness_experiment()
    print("\n=== Robustness Experiment Complete ===")
