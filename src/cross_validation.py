"""Generate shared stratified folds and run model evaluation with NumPy."""

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


class CrossValidator:
    """Evaluate a configured ModelEvaluator on shared development-data folds."""

    def __init__(self, evaluator, folds, preprocessor_factory=None):
        self.evaluator = evaluator
        self.folds = list(folds)
        self.preprocessor_factory = preprocessor_factory

    def run(self, X, y):
        """Return per-fold results and out-of-fold predictions for development data.

        Reuse folds from stratified_k_fold across model candidates. Provide a
        preprocessor_factory that creates a fresh fit/transform instance per
        fold, or None when no learned preprocessing is needed. Never pass the
        reserved holdout here. The returned fold_results follow fold order;
        oof_y_pred and oof_y_score follow the original X/y row order.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(y) == 0 or len(X) != len(y):
            raise ValueError(
                "X and y must have matching nonzero row counts (2D X, 1D y)."
            )
        folds = self._validate_folds(len(y))

        fold_results = []
        oof_y_pred = np.empty(len(y), dtype=int)
        oof_y_score = np.empty(len(y), dtype=float)
        for training_rows, validation_rows in folds:
            # A new instance keeps learned preprocessing separate between folds.
            preprocessor = (
                self.preprocessor_factory()
                if self.preprocessor_factory is not None
                else None
            )
            result = self.evaluator.evaluate_fold(
                X[training_rows],
                y[training_rows],
                X[validation_rows],
                y[validation_rows],
                preprocessor=preprocessor,
            )
            fold_results.append(result)
            # Restore row order: these predictions came from the held-out fold.
            oof_y_pred[validation_rows] = result["y_pred"]
            oof_y_score[validation_rows] = result["y_score"]

        return {
            "fold_results": fold_results,
            "oof_y_pred": oof_y_pred,
            "oof_y_score": oof_y_score,
        }

    def _validate_folds(self, n_rows):
        """Check row separation and complete OOF coverage before any fitting."""
        if len(self.folds) < 2:
            raise ValueError("Cross-validation requires at least two folds.")
        folds = []
        expected_rows = np.arange(n_rows)
        validation_counts = np.zeros(n_rows, dtype=int)
        for training_rows, validation_rows in self.folds:
            training_rows = np.asarray(training_rows)
            validation_rows = np.asarray(validation_rows)
            for rows in (training_rows, validation_rows):
                if rows.ndim != 1 or rows.size == 0 or rows.dtype.kind not in "iu":
                    raise ValueError("Fold indices must be nonempty 1D integer arrays.")
            # Together the two portions must contain every row exactly once.
            # This also rejects overlaps, duplicates, and out-of-range indices.
            all_rows = np.concatenate((training_rows, validation_rows))
            if not np.array_equal(np.sort(all_rows), expected_rows):
                raise ValueError("Each fold must partition all rows without overlap.")
            validation_counts[validation_rows] += 1
            folds.append((training_rows, validation_rows))
        if not np.all(validation_counts == 1):
            raise ValueError("Every development row must validate exactly once.")
        return folds
