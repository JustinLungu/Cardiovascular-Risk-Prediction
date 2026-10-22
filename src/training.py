# Coordinate preprocessing, model training, prediction, and local evaluation.
# Reuse implementations.py and the validation and metrics modules.
# Convert labels to {0, 1} for logistic regression and map predictions back as needed.
# Record configurations, seeds, metrics, and comparisons for reproducible experiments.
# Support final fitting and saving any parameters needed by run.py.

import numpy as np


def prepare_development_and_holdout(parts):
    """Combine TrainTestSplit's train/val arrays and retain its test partition.

    Return (development, holdout), each containing X, y, and ids. This only
    combines already-loaded arrays; it does not shuffle, preprocess, or save.
    """
    development = {}
    for field in ("X", "y", "ids"):
        development[field] = np.concatenate(
            (parts["train"][field], parts["val"][field]), axis=0
        )
    holdout = parts["test"]

    # Each respondent must appear once, keeping development and test separate.
    all_ids = np.concatenate((development["ids"], holdout["ids"]))
    if np.unique(all_ids).size != all_ids.size:
        raise ValueError("Row IDs must be unique across train, validation, and test.")

    return development, holdout
