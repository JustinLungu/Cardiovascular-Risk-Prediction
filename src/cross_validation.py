"""Generate reusable cross-validation indices with NumPy."""

import numpy as np


def stratified_k_fold(y, k=5, seed=42):
    """Return a list of (training_indices, validation_indices) for development data.

    Indices refer to positions in y, so use them on the matching development
    features, labels, and IDs. Each row validates once and trains in k-1 folds.
    Each class is divided as evenly as possible; rounding can affect fold sizes.
    The same labels, k, and seed reproduce the same folds.
    """
    y = np.asarray(y)
    # Row positions require a nonempty vector with no missing target labels.
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all():
        raise ValueError("y must be a nonempty 1D array of finite numeric labels.")
    if not isinstance(k, (int, np.integer)) or k < 2:
        raise ValueError("k must be an integer of at least 2.")

    labels, counts = np.unique(y, return_counts=True)
    # Every validation fold needs at least one example from each class.
    if np.any(counts < k):
        raise ValueError("Each class must have at least k rows.")

    rng = np.random.default_rng(seed)
    validation_parts = [[] for _ in range(k)]
    for label in labels:
        class_rows = rng.permutation(np.flatnonzero(y == label))
        # array_split distributes any leftover rows across the folds.
        class_folds = np.array_split(class_rows, k)
        for fold_index in range(k):
            validation_parts[fold_index].append(class_folds[fold_index])

    folds = []
    for groups in validation_parts:
        validation_rows = rng.permutation(np.concatenate(groups))
        # Train on every development row except this fold's validation rows.
        training_mask = np.ones(y.size, dtype=bool)
        training_mask[validation_rows] = False
        training_rows = rng.permutation(np.flatnonzero(training_mask))
        folds.append((training_rows, validation_rows))

    return folds
