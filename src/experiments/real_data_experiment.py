"""
Real-Data Experiment.

Uses temporal interaction data (AskUbuntu/StackOverflow or synthetic equivalent):
  1. First 50% of timeline -> estimate key frequencies ("prediction").
  2. Build Learning BST from estimated frequencies.
  3. Run remaining 50% of queries on all trees.
  4. Compare total access cost, average depth, and wall-clock time.
"""

import sys
import os

sys.path.insert(0, '.')
from src.trees.learning_bst import LearningBST
from src.trees.splay_tree import SplayTree
from src.trees.treap import Treap
from src.trees.avl_tree import AVLTree
from src.metrics.cost import expected_cost, timed_queries, entropy_bound
from src.data.real_data import (load_or_generate_dataset, split_dataset,
                                 estimate_frequencies)


def run_real_data_experiment(data_dir='data', n=1000, num_interactions=50000,
                              alpha=1.0, seed=42):
    """Run the real-data experiment.

    Args:
        data_dir: directory containing (or to save) interaction data.
        n: number of keys for synthetic fallback.
        num_interactions: total interactions for synthetic fallback.
        alpha: Zipfian alpha for synthetic fallback.
        seed: random seed.

    Returns:
        results: dict {tree_name: {metric: value}}
    """
    # Load or generate data
    interactions = load_or_generate_dataset(data_dir, n, num_interactions,
                                             alpha, seed)

    # Split chronologically
    train_data, test_data = split_dataset(interactions, train_ratio=0.5)
    print(f"Total interactions: {len(interactions)}")
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Estimate frequencies from training data
    train_freq, train_keys = estimate_frequencies(train_data)

    # Get test query actual frequencies
    test_freq, test_keys = estimate_frequencies(test_data)

    # Use all keys seen in either train or test
    all_keys = sorted(set(train_keys) | set(test_keys))
    print(f"Unique keys: train={len(train_keys)}, test={len(test_keys)}, "
          f"total={len(all_keys)}")

    # For keys not seen in training, assign a small default frequency
    default_freq = 1.0 / (len(all_keys) * 10)
    full_train_freq = {}
    for k in all_keys:
        full_train_freq[k] = train_freq.get(k, default_freq)
    # Re-normalize
    total = sum(full_train_freq.values())
    full_train_freq = {k: v / total for k, v in full_train_freq.items()}

    results = {}

    # --- Learning BST (built on training frequencies) ---
    print("\nBuilding Learning BST...")
    lbst = LearningBST()
    lbst.build([(k, full_train_freq[k]) for k in all_keys])
    total_depth, elapsed = timed_queries(lbst, test_data)
    avg_cost = total_depth / len(test_data)
    ec = expected_cost(lbst, test_freq) if test_freq else 0
    results['Learning BST'] = {
        'total_access_cost': total_depth,
        'avg_access_cost': avg_cost,
        'expected_cost': ec,
        'wall_clock_time': elapsed,
        'height': lbst.height(),
    }
    print(f"  Learning BST: total_cost={total_depth}, avg={avg_cost:.4f}, time={elapsed:.4f}s")

    # --- AVL Tree ---
    print("Building AVL Tree...")
    avl = AVLTree()
    avl.build(all_keys)
    total_depth, elapsed = timed_queries(avl, test_data)
    avg_cost = total_depth / len(test_data)
    ec = expected_cost(avl, test_freq) if test_freq else 0
    results['AVL Tree'] = {
        'total_access_cost': total_depth,
        'avg_access_cost': avg_cost,
        'expected_cost': ec,
        'wall_clock_time': elapsed,
        'height': avl.height(),
    }
    print(f"  AVL Tree:     total_cost={total_depth}, avg={avg_cost:.4f}, time={elapsed:.4f}s")

    # --- Treap ---
    print("Building Treap...")
    treap = Treap(seed=seed)
    treap.build(all_keys)
    total_depth, elapsed = timed_queries(treap, test_data)
    avg_cost = total_depth / len(test_data)
    ec = expected_cost(treap, test_freq) if test_freq else 0
    results['Treap'] = {
        'total_access_cost': total_depth,
        'avg_access_cost': avg_cost,
        'expected_cost': ec,
        'wall_clock_time': elapsed,
        'height': treap.height(),
    }
    print(f"  Treap:        total_cost={total_depth}, avg={avg_cost:.4f}, time={elapsed:.4f}s")

    # --- Splay Tree ---
    print("Building Splay Tree...")
    splay = SplayTree()
    for k in all_keys:
        splay.insert(k)
    total_depth, elapsed = timed_queries(splay, test_data)
    avg_cost = total_depth / len(test_data)
    ec = expected_cost(splay, test_freq) if test_freq else 0
    results['Splay Tree'] = {
        'total_access_cost': total_depth,
        'avg_access_cost': avg_cost,
        'expected_cost': ec,
        'wall_clock_time': elapsed,
        'height': splay.height(),
    }
    print(f"  Splay Tree:   total_cost={total_depth}, avg={avg_cost:.4f}, time={elapsed:.4f}s")

    # --- Entropy bound ---
    if test_freq:
        h = entropy_bound(test_freq.values())
        results['Entropy Bound'] = {
            'total_access_cost': 0,
            'avg_access_cost': h,
            'expected_cost': h,
            'wall_clock_time': 0,
            'height': 0,
        }
        print(f"  Entropy H(p): {h:.4f}")

    return results


if __name__ == '__main__':
    results = run_real_data_experiment()
    print("\n=== Real-Data Experiment Complete ===")
