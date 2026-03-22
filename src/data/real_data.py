"""
Real-world dataset processing pipeline.

Supports loading temporal interaction data (e.g., AskUbuntu, StackOverflow)
and splitting into train/test by timeline for frequency estimation.

Dataset format expected: a text/CSV file where each line represents an
interaction with a key (e.g., a user ID, question ID, or tag).
The pipeline:
  1. Load interactions in chronological order.
  2. Split first 50% as training data -> estimate key frequencies.
  3. Remaining 50% as test queries.
"""

import os
import json
from collections import Counter


def load_real_dataset(filepath, key_column=0, delimiter=None, max_lines=None):
    """Load interaction data from a text/CSV file.

    Each line is an interaction. The key is extracted from the specified column.

    Args:
        filepath: path to the data file.
        key_column: column index to use as the key (default 0).
        delimiter: column delimiter (default None = whitespace).
        max_lines: max number of lines to read (None = all).

    Returns:
        list of keys (in chronological order).
    """
    interactions = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            parts = line.split(delimiter)
            if len(parts) > key_column:
                try:
                    key = int(parts[key_column])
                except ValueError:
                    key = parts[key_column]
                interactions.append(key)
    return interactions


def split_dataset(interactions, train_ratio=0.5):
    """Split interactions chronologically into train and test.

    Args:
        interactions: list of keys in chronological order.
        train_ratio: fraction for training (default 0.5).

    Returns:
        train_interactions: first portion.
        test_interactions: remaining portion.
    """
    split_idx = int(len(interactions) * train_ratio)
    return interactions[:split_idx], interactions[split_idx:]


def estimate_frequencies(interactions):
    """Estimate key frequencies from a list of interactions.

    Args:
        interactions: list of keys.

    Returns:
        freq_dict: dict {key: probability} (normalized).
        keys: sorted list of unique keys.
    """
    counts = Counter(interactions)
    total = sum(counts.values())
    freq_dict = {k: c / total for k, c in counts.items()}
    keys = sorted(counts.keys())
    return freq_dict, keys


def generate_synthetic_real_data(n=1000, num_interactions=50000, alpha=1.0, seed=42):
    """Generate synthetic 'real-world-like' data for testing when
    actual datasets are not available.

    Simulates temporal interaction data where key popularity follows
    a Zipfian distribution that shifts slightly over time.

    Args:
        n: number of distinct keys.
        num_interactions: total number of interactions.
        alpha: Zipfian skewness.
        seed: random seed.

    Returns:
        interactions: list of keys in temporal order.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    ranks = np.arange(1, n + 1, dtype=float)
    base_probs = 1.0 / np.power(ranks, alpha)
    base_probs /= base_probs.sum()

    # Add slight temporal drift: in the second half, shift popularity
    half = num_interactions // 2
    interactions = []

    # First half: base distribution
    first_half = rng.choice(n, size=half, p=base_probs).tolist()
    interactions.extend(first_half)

    # Second half: slightly shifted distribution (rotate ranks by ~10%)
    shift = max(1, n // 10)
    shifted_probs = np.roll(base_probs, shift)
    shifted_probs /= shifted_probs.sum()
    second_half = rng.choice(n, size=num_interactions - half, p=shifted_probs).tolist()
    interactions.extend(second_half)

    return interactions


def save_dataset(interactions, filepath):
    """Save interactions to a file (one key per line)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for key in interactions:
            f.write(f"{key}\n")


def load_or_generate_dataset(data_dir, n=1000, num_interactions=50000,
                              alpha=1.0, seed=42):
    """Load a real dataset if available, otherwise generate synthetic data.

    Looks for 'interactions.txt' in data_dir. If not found, generates
    synthetic data and saves it.

    Args:
        data_dir: directory to look for / save data.
        n, num_interactions, alpha, seed: params for synthetic generation.

    Returns:
        interactions: list of keys.
    """
    filepath = os.path.join(data_dir, 'interactions.txt')
    if os.path.exists(filepath):
        return load_real_dataset(filepath)

    print(f"No real dataset found at {filepath}. Generating synthetic data...")
    interactions = generate_synthetic_real_data(n, num_interactions, alpha, seed)
    save_dataset(interactions, filepath)
    print(f"Saved {len(interactions)} interactions to {filepath}")
    return interactions
