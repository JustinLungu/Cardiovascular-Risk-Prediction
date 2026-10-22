# Implement NumPy classification metrics: confusion counts, accuracy, precision,
# recall, F1, and any additional metric required by the competition.
# State the positive class and expected label encoding; handle zero denominators.
# Keep prediction threshold selection in the training/experiment workflow.

import numpy as np


def confusion_counts(y_true, y_pred):
    """Return tp, tn, fp, and fn as integer counts, with +1 the positive class.

    Inputs are matching 1D arrays of class labels in {-1, +1}. Empty inputs
    return zero counts. Convert prediction scores to class labels beforehand.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    # Each prediction must correspond to exactly one true label.
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("True and predicted labels must be matching 1D arrays.")
    # Reject other encodings, missing labels, and scores that need thresholding.
    if not np.isin(y_true, (-1, 1)).all() or not np.isin(y_pred, (-1, 1)).all():
        raise ValueError("True and predicted labels must contain only -1 and +1.")

    return {
        "tp": int(np.count_nonzero((y_true == 1) & (y_pred == 1))),
        "tn": int(np.count_nonzero((y_true == -1) & (y_pred == -1))),
        "fp": int(np.count_nonzero((y_true == -1) & (y_pred == 1))),
        "fn": int(np.count_nonzero((y_true == 1) & (y_pred == -1))),
    }


def accuracy(y_true, y_pred):
    """Return the fraction of correct predictions, or 0.0 for empty inputs."""
    counts = confusion_counts(y_true, y_pred)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return (counts["tp"] + counts["tn"]) / total


def precision(y_true, y_pred):
    """Return the fraction of predicted positives that are correct (+1 class)."""
    counts = confusion_counts(y_true, y_pred)
    predicted_positive = counts["tp"] + counts["fp"]
    # Without positive predictions, use 0.0 instead of dividing by zero.
    if predicted_positive == 0:
        return 0.0
    return counts["tp"] / predicted_positive


def recall(y_true, y_pred):
    """Return the fraction of actual positives that were detected (+1 class)."""
    counts = confusion_counts(y_true, y_pred)
    actual_positive = counts["tp"] + counts["fn"]
    # Without actual positives, use 0.0 instead of dividing by zero.
    if actual_positive == 0:
        return 0.0
    return counts["tp"] / actual_positive
