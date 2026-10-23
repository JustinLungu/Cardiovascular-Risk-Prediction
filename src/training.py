# Coordinate preprocessing, model training, prediction, and local evaluation.
# Reuse implementations.py and the validation and metrics modules.
# Convert labels to {0, 1} for logistic regression and map predictions back as needed.
# Record configurations, seeds, metrics, and comparisons for reproducible experiments.
# Support final fitting and saving any parameters needed by run.py.

import numpy as np

import implementations


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


def always_negative_baseline(X):
    """Return (y_pred, y_score), predicting -1 for every feature row in X.

    Only the row count is used; no fitting or preprocessing is needed.
    Decision scores are fixed at -1.0, so all rows receive the same ranking.
    """
    y_pred = np.full(len(X), -1, dtype=int)
    y_score = np.full(len(X), -1.0)
    return y_pred, y_score


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


def train_model(X, y, method="least_squares", max_it=1000, gamma=0.01, seed=42):
    """Train a least-squares variant and return the course function's (w, loss).

    X must be preprocessed, finite, and without an intercept column. Labels
    use {-1, +1}. Supported methods are least_squares, mean_squared_error_gd,
    and mean_squared_error_sgd. GD/SGD start from zero weights and use max_it
    and gamma; SGD is seeded per call without changing the caller's RNG state.
    The returned training loss is half the mean squared error.
    """
    X_fit = add_intercept(X)
    y_fit = prepare_training_labels(y)
    # Training needs at least one row, with a label for every feature row.
    if len(y_fit) == 0 or len(X_fit) != len(y_fit):
        raise ValueError(
            "Training features and labels must have matching nonzero lengths."
        )
    # Imputation and other learned preprocessing belong before this call.
    if not np.isfinite(X_fit).all():
        raise ValueError(
            "Training features must be finite; complete preprocessing first."
        )

    if method == "least_squares":
        return implementations.least_squares(y_fit, X_fit)

    w0 = np.zeros(X_fit.shape[1])
    if method == "mean_squared_error_gd":
        return implementations.mean_squared_error_gd(y_fit, X_fit, w0, max_it, gamma)
    if method == "mean_squared_error_sgd":
        # The course SGD uses NumPy's global RNG; restore it after the seeded call.
        previous_state = np.random.get_state()
        try:
            np.random.seed(seed)
            return implementations.mean_squared_error_sgd(
                y_fit, X_fit, w0, max_it, gamma
            )
        finally:
            np.random.set_state(previous_state)

    raise ValueError(f"Unknown training method: {method!r}.")
