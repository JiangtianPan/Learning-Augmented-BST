"""
Dynamic Updates Experiment (Extension).

Tests the Insert/Delete operations for Learning BST:
  1. Build initial tree from predicted frequencies.
  2. Perform a sequence of mixed operations (search, insert, delete).
  3. Measure how structural rotations during updates affect cost.
  4. Compare against Splay Tree (which also adapts dynamically).
"""

import sys
import time
import numpy as np

sys.path.insert(0, '.')
from src.trees.learning_bst import LearningBST
from src.trees.splay_tree import SplayTree
from src.trees.treap import Treap
from src.trees.avl_tree import AVLTree
from src.metrics.cost import expected_cost
from src.data.zipfian import zipfian_distribution


def run_dynamic_experiment(n=500, alpha=1.5, num_initial=400,
                            num_operations=10000, seed=42):
    """Run the dynamic update experiment.

    Args:
        n: total key space size.
        alpha: Zipfian alpha for frequency assignment.
        num_initial: number of keys to insert initially.
        num_operations: number of mixed operations after initial build.
        seed: random seed.

    Returns:
        results: dict {tree_name: {metric: value}}
    """
    rng = np.random.default_rng(seed)

    # Generate frequencies for all possible keys
    probs = zipfian_distribution(n, alpha)
    all_keys = list(range(n))

    # Initial keys (randomly selected subset)
    initial_keys = sorted(rng.choice(all_keys, size=num_initial, replace=False).tolist())
    remaining_keys = sorted(set(all_keys) - set(initial_keys))

    # Generate operation sequence: 60% search, 20% insert, 20% delete
    operations = []
    insertable = list(remaining_keys)
    deletable = list(initial_keys)

    for _ in range(num_operations):
        r = rng.random()
        if r < 0.6:
            # Search: pick a key weighted by frequency
            key = rng.choice(all_keys, p=probs)
            operations.append(('search', int(key)))
        elif r < 0.8:
            # Insert
            if insertable:
                idx = rng.integers(0, len(insertable))
                key = insertable.pop(idx)
                deletable.append(key)
                operations.append(('insert', key))
            else:
                key = rng.choice(all_keys, p=probs)
                operations.append(('search', int(key)))
        else:
            # Delete
            if deletable:
                idx = rng.integers(0, len(deletable))
                key = deletable.pop(idx)
                insertable.append(key)
                operations.append(('delete', key))
            else:
                key = rng.choice(all_keys, p=probs)
                operations.append(('search', int(key)))

    print(f"Initial keys: {num_initial}, Operations: {num_operations}")
    op_counts = {}
    for op, _ in operations:
        op_counts[op] = op_counts.get(op, 0) + 1
    print(f"Operation mix: {op_counts}")

    results = {}

    # --- Learning BST ---
    print("\nRunning Learning BST...")
    lbst = LearningBST()
    lbst.build([(k, float(probs[k])) for k in initial_keys])
    total_search_depth = 0
    search_count = 0

    start = time.perf_counter()
    for op, key in operations:
        if op == 'search':
            _, depth = lbst.search(key)
            total_search_depth += depth
            search_count += 1
        elif op == 'insert':
            lbst.insert(key, float(probs[key]))
        elif op == 'delete':
            lbst.delete(key)
    elapsed = time.perf_counter() - start

    avg_search = total_search_depth / search_count if search_count > 0 else 0
    results['Learning BST'] = {
        'avg_search_depth': avg_search,
        'total_search_depth': total_search_depth,
        'search_count': search_count,
        'wall_clock_time': elapsed,
        'final_height': lbst.height(),
        'final_size': len(lbst),
    }
    print(f"  avg_search_depth={avg_search:.4f}, time={elapsed:.4f}s, "
          f"height={lbst.height()}, size={len(lbst)}")

    # --- AVL Tree ---
    print("Running AVL Tree...")
    avl = AVLTree()
    avl.build(initial_keys)
    total_search_depth = 0
    search_count = 0

    start = time.perf_counter()
    for op, key in operations:
        if op == 'search':
            _, depth = avl.search(key)
            total_search_depth += depth
            search_count += 1
        elif op == 'insert':
            avl.insert(key)
        elif op == 'delete':
            avl.delete(key)
    elapsed = time.perf_counter() - start

    avg_search = total_search_depth / search_count if search_count > 0 else 0
    results['AVL Tree'] = {
        'avg_search_depth': avg_search,
        'total_search_depth': total_search_depth,
        'search_count': search_count,
        'wall_clock_time': elapsed,
        'final_height': avl.height(),
        'final_size': len(avl),
    }
    print(f"  avg_search_depth={avg_search:.4f}, time={elapsed:.4f}s, "
          f"height={avl.height()}, size={len(avl)}")

    # --- Treap ---
    print("Running Treap...")
    treap = Treap(seed=seed)
    treap.build(initial_keys)
    total_search_depth = 0
    search_count = 0

    start = time.perf_counter()
    for op, key in operations:
        if op == 'search':
            _, depth = treap.search(key)
            total_search_depth += depth
            search_count += 1
        elif op == 'insert':
            treap.insert(key)
        elif op == 'delete':
            treap.delete(key)
    elapsed = time.perf_counter() - start

    avg_search = total_search_depth / search_count if search_count > 0 else 0
    results['Treap'] = {
        'avg_search_depth': avg_search,
        'total_search_depth': total_search_depth,
        'search_count': search_count,
        'wall_clock_time': elapsed,
        'final_height': treap.height(),
        'final_size': len(treap),
    }
    print(f"  avg_search_depth={avg_search:.4f}, time={elapsed:.4f}s, "
          f"height={treap.height()}, size={len(treap)}")

    # --- Splay Tree ---
    print("Running Splay Tree...")
    splay = SplayTree()
    for k in initial_keys:
        splay.insert(k)
    total_search_depth = 0
    search_count = 0

    start = time.perf_counter()
    for op, key in operations:
        if op == 'search':
            _, depth = splay.search(key)
            total_search_depth += depth
            search_count += 1
        elif op == 'insert':
            splay.insert(key)
        elif op == 'delete':
            splay.delete(key)
    elapsed = time.perf_counter() - start

    avg_search = total_search_depth / search_count if search_count > 0 else 0
    results['Splay Tree'] = {
        'avg_search_depth': avg_search,
        'total_search_depth': total_search_depth,
        'search_count': search_count,
        'wall_clock_time': elapsed,
        'final_height': splay.height(),
        'final_size': len(splay),
    }
    print(f"  avg_search_depth={avg_search:.4f}, time={elapsed:.4f}s, "
          f"height={splay.height()}, size={len(splay)}")

    return results


if __name__ == '__main__':
    results = run_dynamic_experiment()
    print("\n=== Dynamic Update Experiment Complete ===")
