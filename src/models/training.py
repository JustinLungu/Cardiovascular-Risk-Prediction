"""Fit the course models using consistent inputs and reproducible settings."""

# Support final fitting and saving any parameters needed by run.py.

import numpy as np

import implementations
from src.models.model_inputs import add_intercept, prepare_training_labels


def train_model(
    X, y, method="least_squares", max_it=1000, gamma=0.01, seed=42, lmbd=1e-6
):
    """Train a model using the course implementations and return (w, loss).

    X must be preprocessed, finite, and without an intercept column. Labels
    use {-1, +1}. Supported methods are least_squares, mean_squared_error_gd,
    mean_squared_error_sgd, ridge_regression, logistic_regression,
    reg_logistic_regression, and svm. Both logistic methods convert labels to
    {0, 1}; SVM keeps {-1, +1} labels.
    Iterative methods start from zero weights and use max_it and gamma; SGD
    preserves the caller's RNG state. Regularized methods use a finite
    nonnegative lmbd and penalize the intercept too. Loss is mean negative
    log-likelihood for both logistic methods, mean hinge loss for SVM, and
    half-MSE for least squares/ridge, excluding any regularization penalty.
    """
    X_fit = add_intercept(X)
    is_logistic = method in ("logistic_regression", "reg_logistic_regression")
    y_fit = prepare_training_labels(y, logistic=is_logistic)
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

    if method in ("ridge_regression", "reg_logistic_regression", "svm"):
        # L2 penalties require a finite, nonnegative strength.
        if not np.isscalar(lmbd) or not np.isfinite(lmbd) or lmbd < 0:
            raise ValueError("lmbd must be a finite, nonnegative scalar.")

    if method == "least_squares":
        return implementations.least_squares(y_fit, X_fit)
    if method == "ridge_regression":
        return implementations.ridge_regression(y_fit, X_fit, lmbd)

    w0 = np.zeros(X_fit.shape[1])
    if method == "mean_squared_error_gd":
        return implementations.mean_squared_error_gd(y_fit, X_fit, w0, max_it, gamma)
    if method == "logistic_regression":
        return implementations.logistic_regression(y_fit, X_fit, w0, max_it, gamma)
    if method == "reg_logistic_regression":
        return implementations.reg_logistic_regression(
            y_fit, X_fit, lmbd, w0, max_it, gamma
        )
    if method == "svm":
        return implementations.svm(y_fit, X_fit, lmbd, w0, max_it, gamma)
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
