"""Prepare feature matrices and label encodings for the course models."""

# Convert labels to {0, 1} for logistic regression and map predictions back as needed.

import numpy as np


def add_intercept(X):
    """Prepend a column of ones to a 2D feature matrix after preprocessing.

    Input X must not already contain the model's intercept column.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("Features must be a 2D array with one row per sample.")
    return np.column_stack((np.ones(len(X)), X))


def prepare_training_labels(y, logistic=False):
    """Copy {-1, +1} labels, converting them to {0, 1} for logistic training."""
    y = np.asarray(y)
    # Start from the original label encoding so evaluation labels stay unchanged.
    if y.ndim != 1 or not np.isin(y, (-1, 1)).all():
        raise ValueError(
            "Training labels must be a 1D array containing only -1 and +1."
        )
    if logistic:
        return (y == 1).astype(int)
    return y.copy()
