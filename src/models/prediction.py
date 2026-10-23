"""Produce class predictions and continuous scores from fitted weights."""

import numpy as np

from src.models.model_inputs import add_intercept


def always_negative_baseline(X):
    """Return (y_pred, y_score), predicting -1 for every feature row in X.

    Only the row count is used; no fitting or preprocessing is needed.
    Decision scores are fixed at -1.0, so all rows receive the same ranking.
    """
    y_pred = np.full(len(X), -1, dtype=int)
    y_score = np.full(len(X), -1.0)
    return y_pred, y_score


def predict_from_weights(X, w, logistic=False, threshold=None):
    """Return (y_pred, y_score) from preprocessed features and fitted weights.

    X must not contain an intercept column; w[0] is the intercept weight.
    Linear scores are X_with_intercept @ w, with default threshold 0.0.
    Logistic scores are positive-class probabilities, with threshold 0.5.
    Scores at or above the threshold predict +1; all others predict -1.
    """
    X_with_intercept = add_intercept(X)
    w = np.asarray(w)
    # One weight is needed for the intercept and one for each feature column.
    if w.ndim != 1 or len(w) != X_with_intercept.shape[1]:
        raise ValueError(
            "Weights must be a 1D array matching the features plus intercept."
        )
    y_score = X_with_intercept @ w
    # Preprocessing and training must produce finite prediction scores.
    if not np.isfinite(y_score).all():
        raise ValueError(
            "Prediction scores must be finite; check preprocessing and weights."
        )
    if logistic:
        # Sigmoid: 1 / (1 + exp(-score)), evaluated without exponential overflow.
        y_score = np.exp(-np.logaddexp(0, -y_score))

    if threshold is None:
        threshold = 0.5 if logistic else 0.0
    if not np.isscalar(threshold) or not np.isfinite(threshold):
        raise ValueError("The prediction threshold must be a finite scalar.")
    y_pred = np.where(y_score >= threshold, 1, -1)
    return y_pred, y_score
