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


def f1_score(y_true, y_pred):
    """Return positive-class F1, the harmonic mean of precision and recall."""
    counts = confusion_counts(y_true, y_pred)
    # Equivalent to 2 * precision * recall / (precision + recall).
    denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    # With no actual or predicted positives, use 0.0 instead of dividing by zero.
    if denominator == 0:
        return 0.0
    return 2 * counts["tp"] / denominator


def roc_auc_score(y_true, y_score):
    """Return ROC-AUC for {-1, +1} labels and matching 1D prediction scores.

    Higher scores must indicate +1; use probabilities or decision scores before
    thresholding. Scores must be finite and both classes must be present.
    Empty or single-class labels raise ValueError.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    # Each score must correspond to exactly one true label.
    if y_true.ndim != 1 or y_score.ndim != 1 or y_true.shape != y_score.shape:
        raise ValueError("True labels and scores must be matching 1D arrays.")
    if not np.isin(y_true, (-1, 1)).all():
        raise ValueError("True labels must contain only -1 and +1.")
    # Missing or infinite scores cannot define a valid ROC curve.
    if not np.isfinite(y_score).all():
        raise ValueError("Prediction scores must be finite.")

    n_positive = np.count_nonzero(y_true == 1)
    n_negative = np.count_nonzero(y_true == -1)
    if n_positive == 0 or n_negative == 0:
        raise ValueError("ROC-AUC requires both -1 and +1 labels.")

    # Lowering the threshold includes rows from highest to lowest score.
    order = np.argsort(y_score)[::-1]
    sorted_labels, sorted_scores = y_true[order], y_score[order]
    # Include all rows with the same score together so ties are treated equally.
    threshold_ends = np.append(sorted_scores[1:] != sorted_scores[:-1], True)
    true_positives = np.cumsum(sorted_labels == 1)[threshold_ends]
    false_positives = np.cumsum(sorted_labels == -1)[threshold_ends]

    # Start the ROC curve at (0, 0), where no rows are predicted positive.
    true_positive_rate = np.append(0.0, true_positives / n_positive)
    false_positive_rate = np.append(0.0, false_positives / n_negative)
    return float(np.trapz(true_positive_rate, false_positive_rate))


def evaluate(y_true, y_pred, y_score):
    """Return accuracy, precision, recall, F1, and ROC-AUC in one dictionary.

    Supply matching 1D arrays: true/predicted labels in {-1, +1} and finite
    scores where higher means +1. Choose the prediction threshold beforehand;
    only ROC-AUC uses the continuous scores. Both true classes are required,
    so empty or single-class labels raise ValueError, as in roc_auc_score.
    """
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
    }
