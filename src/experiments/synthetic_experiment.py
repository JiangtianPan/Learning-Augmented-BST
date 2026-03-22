"""
Synthetic Experiment: Compare all BSTs across varying Zipfian alpha.

For each alpha value (from uniform to highly skewed):
  1. Generate a Zipfian distribution over n keys.
  2. Build each tree structure.
  3. Measure expected cost, average query depth, and wall-clock time.
  4. Record the crossover point where Learning BST outperforms Balanced BST.
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from src.trees.learning_bst import LearningBST
from src.trees.splay_tree import SplayTree
from src.trees.treap import Treap
from src.trees.avl_tree import AVLTree
from src.metrics.cost import expected_cost, average_access_cost, timed_queries, entropy_bound
from src.data.zipfian import generate_zipfian_queries


def run_synthetic_experiment(n=500, num_queries=50000,
                              alphas=None, seed=42):
    """Run the full synthetic comparison experiment.

    Args:
        n: number of distinct keys.
        num_queries: number of queries per alpha value.
        alphas: list of alpha values to sweep.
        seed: random seed.

    Returns:
        results: dict with structure:
            {alpha: {tree_name: {metric: value}}}
    """
    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

    results = {}

    for alpha in alphas:
        print(f"\n--- Alpha = {alpha:.2f} ---")
        keys, probs, queries = generate_zipfian_queries(n, alpha, num_queries, seed)

        alpha_results = {}

        # --- Learning BST ---
        lbst = LearningBST()
        lbst.build([(k, probs[k]) for k in keys])
        ec = expected_cost(lbst, probs)
        total_depth, elapsed = timed_queries(lbst, queries)
        avg_cost = total_depth / len(queries)
        alpha_results['Learning BST'] = {
            'expected_cost': ec,
            'avg_access_cost': avg_cost,
            'wall_clock_time': elapsed,
            'height': lbst.height(),
        }
        print(f"  Learning BST: expected_cost={ec:.4f}, avg_depth={avg_cost:.4f}, time={elapsed:.4f}s")

        # --- AVL Tree ---
        avl = AVLTree()
        avl.build(keys)
        ec = expected_cost(avl, probs)
        total_depth, elapsed = timed_queries(avl, queries)
        avg_cost = total_depth / len(queries)
        alpha_results['AVL Tree'] = {
            'expected_cost': ec,
            'avg_access_cost': avg_cost,
            'wall_clock_time': elapsed,
            'height': avl.height(),
        }
        print(f"  AVL Tree:     expected_cost={ec:.4f}, avg_depth={avg_cost:.4f}, time={elapsed:.4f}s")

        # --- Treap ---
        treap = Treap(seed=seed)
        treap.build(keys)
        ec = expected_cost(treap, probs)
        total_depth, elapsed = timed_queries(treap, queries)
        avg_cost = total_depth / len(queries)
        alpha_results['Treap'] = {
            'expected_cost': ec,
            'avg_access_cost': avg_cost,
            'wall_clock_time': elapsed,
            'height': treap.height(),
        }
        print(f"  Treap:        expected_cost={ec:.4f}, avg_depth={avg_cost:.4f}, time={elapsed:.4f}s")

        # --- Splay Tree ---
        splay = SplayTree()
        # Build splay by inserting all keys (not querying with freq)
        for k in keys:
            splay.insert(k)
        # For splay, expected_cost is measured by running the query sequence
        # (splay adapts dynamically, so depth changes with each access)
        total_depth, elapsed = timed_queries(splay, queries)
        avg_cost = total_depth / len(queries)
        # For expected cost, use post-query tree structure
        ec_post = expected_cost(splay, probs)
        alpha_results['Splay Tree'] = {
            'expected_cost': ec_post,
            'avg_access_cost': avg_cost,
            'wall_clock_time': elapsed,
            'height': splay.height(),
        }
        print(f"  Splay Tree:   expected_cost={ec_post:.4f}, avg_depth={avg_cost:.4f}, time={elapsed:.4f}s")

        # --- Entropy bound ---
        h = entropy_bound(probs.values())
        alpha_results['Entropy Bound'] = {
            'expected_cost': h,
            'avg_access_cost': h,
            'wall_clock_time': 0,
            'height': 0,
        }
        print(f"  Entropy H(p): {h:.4f}")

        results[alpha] = alpha_results

    return results


if __name__ == '__main__':
    results = run_synthetic_experiment()
    print("\n=== Experiment Complete ===")
