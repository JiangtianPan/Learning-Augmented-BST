"""
Cost metrics for BST comparison experiments.

Core metric: Expected Cost = sum_i p_i * depth(key_i)
where p_i is the actual probability and depth(key_i) is the number
of comparisons to reach the node (root = depth 1).
"""

import time


def expected_cost(tree, key_probs):
    """Compute expected weighted search cost.

    Args:
        tree: A BST with a get_all_depths() method.
        key_probs: dict {key: actual_probability}.

    Returns:
        float: sum of p_i * depth(key_i) for all keys.
    """
    depths = tree.get_all_depths()
    cost = 0.0
    for key, prob in key_probs.items():
        if key in depths:
            cost += prob * depths[key]
    return cost


def average_access_cost(tree, queries):
    """Compute average access cost over a sequence of queries.

    Args:
        tree: A BST with a search(key) -> (found, depth) method.
        queries: list of keys to search.

    Returns:
        float: average depth across all queries.
    """
    if not queries:
        return 0.0
    total_depth = 0
    for key in queries:
        _, depth = tree.search(key)
        total_depth += depth
    return total_depth / len(queries)


def total_access_cost(tree, queries):
    """Compute total access cost (sum of depths) over a query sequence.

    Args:
        tree: A BST with a search(key) -> (found, depth) method.
        queries: list of keys to search.

    Returns:
        int: total depth summed over all queries.
    """
    total = 0
    for key in queries:
        _, depth = tree.search(key)
        total += depth
    return total


def timed_queries(tree, queries):
    """Run queries and measure wall-clock time.

    Args:
        tree: A BST with a search(key) method.
        queries: list of keys to search.

    Returns:
        (total_depth, elapsed_seconds)
    """
    total_depth = 0
    start = time.perf_counter()
    for key in queries:
        _, depth = tree.search(key)
        total_depth += depth
    elapsed = time.perf_counter() - start
    return total_depth, elapsed


def entropy_bound(probs):
    """Compute the entropy lower bound H(p) = sum -p_i * log2(p_i).

    This is the theoretical lower bound for optimal BST expected cost.

    Args:
        probs: iterable of probabilities (must sum to ~1).

    Returns:
        float: entropy in bits.
    """
    import math
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log2(p)
    return h
