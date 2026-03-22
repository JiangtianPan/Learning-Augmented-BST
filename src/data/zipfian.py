"""
Synthetic data generator using Zipfian (power-law) distributions.

Zipf's law: P(rank=k) proportional to 1/k^alpha
- alpha=0 -> uniform distribution
- alpha=1 -> standard Zipf
- alpha>1 -> highly skewed (few keys dominate)
"""

import numpy as np


def zipfian_distribution(n, alpha=1.0):
    """Generate a Zipfian probability distribution over n keys.

    Args:
        n: number of distinct keys.
        alpha: skewness parameter (0=uniform, higher=more skewed).

    Returns:
        numpy array of probabilities summing to 1, indexed by rank.
    """
    ranks = np.arange(1, n + 1, dtype=float)
    weights = 1.0 / np.power(ranks, alpha)
    probs = weights / weights.sum()
    return probs


def generate_zipfian_queries(n, alpha, num_queries, seed=None):
    """Generate a query sequence following a Zipfian distribution.

    Args:
        n: number of distinct keys (keys will be 0..n-1).
        alpha: Zipfian skewness parameter.
        num_queries: number of queries to generate.
        seed: random seed for reproducibility.

    Returns:
        keys: sorted list of all distinct keys [0, 1, ..., n-1].
        probs: dict {key: probability}.
        queries: list of query keys sampled from the distribution.
    """
    rng = np.random.default_rng(seed)
    probs_array = zipfian_distribution(n, alpha)
    keys = list(range(n))
    probs = {k: float(probs_array[k]) for k in keys}

    queries = rng.choice(keys, size=num_queries, p=probs_array).tolist()
    return keys, probs, queries


def generate_permuted_queries(n, alpha_train, alpha_test, num_queries, seed=None):
    """Generate queries where train and test distributions differ.

    Used for the prediction error / robustness experiment:
    - Build tree using train distribution (alpha_train)
    - Query using test distribution (alpha_test) with permuted ranks

    Args:
        n: number of distinct keys.
        alpha_train: Zipfian alpha for the "predicted" distribution.
        alpha_test: Zipfian alpha for the actual query distribution.
        num_queries: number of test queries.
        seed: random seed.

    Returns:
        keys: list of keys.
        train_probs: dict {key: predicted_prob} (for tree construction).
        test_probs: dict {key: actual_prob} (for cost evaluation).
        queries: list of query keys sampled from test distribution.
    """
    rng = np.random.default_rng(seed)

    train_probs_array = zipfian_distribution(n, alpha_train)
    test_probs_array = zipfian_distribution(n, alpha_test)

    # Permute the test distribution: shuffle which keys get which probability
    permuted_indices = rng.permutation(n)
    test_probs_permuted = np.zeros(n)
    for i, pi in enumerate(permuted_indices):
        test_probs_permuted[i] = test_probs_array[pi]
    test_probs_permuted /= test_probs_permuted.sum()

    keys = list(range(n))
    train_probs = {k: float(train_probs_array[k]) for k in keys}
    test_probs = {k: float(test_probs_permuted[k]) for k in keys}

    queries = rng.choice(keys, size=num_queries, p=test_probs_permuted).tolist()
    return keys, train_probs, test_probs, queries
