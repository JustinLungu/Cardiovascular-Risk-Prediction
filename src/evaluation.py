"""Evaluate models using shared settings and training-only preprocessing."""

# Coordinate preprocessing, model training, prediction, and local evaluation.
# Reuse implementations.py and the validation and metrics modules.
# Record configurations, seeds, metrics, and comparisons for reproducible experiments.

import numpy as np

from src.metrics import evaluate
from src.model_inputs import prepare_training_labels
from src.prediction import always_negative_baseline, predict_from_weights
from src.training import train_model


class ModelEvaluator:
    """Keep model settings together for repeated validation-fold evaluations."""

    def __init__(self, method="least_squares", threshold=None, **training_params):
        self.method = method
        self.threshold = threshold
        self.training_params = training_params

    def evaluate_fold(self, X_train, y_train, X_val, y_val, preprocessor=None):
        """Fit and evaluate one training/validation pair, preserving input arrays.

        Pass a fresh preprocessor with fit(X, y) and transform(X), or None when
        no learned preprocessing is needed. Transforms must preserve row
        order/count and use the same feature columns for both portions.
        Validation labels must contain both classes for ROC-AUC. Return metrics,
        training_loss, y_pred, and y_score in a dictionary; prediction arrays
        follow validation input order. The always_negative_baseline method skips
        preprocessing and has training_loss=None.
        """
        # Copies protect the caller's raw data when a preprocessor edits in place.
        X_train = np.array(X_train, dtype=float, copy=True)
        X_val = np.array(X_val, dtype=float, copy=True)
        y_train = prepare_training_labels(y_train)
        y_val = prepare_training_labels(y_val)
        if X_train.ndim != 2 or X_val.ndim != 2:
            raise ValueError("Training and validation features must be 2D arrays.")
        # Each portion needs nonempty, aligned feature rows and original labels.
        if (
            len(y_train) == 0
            or len(y_val) == 0
            or len(X_train) != len(y_train)
            or len(X_val) != len(y_val)
        ):
            raise ValueError(
                "Each portion must have matching nonzero feature/label lengths."
            )

        if self.method == "always_negative_baseline":
            y_pred, y_score = always_negative_baseline(X_val)
            training_loss = None
        else:
            if preprocessor is not None:
                # Learn from training rows/labels only, then reuse that fitted state.
                preprocessor.fit(X_train, y_train)
                X_train = preprocessor.transform(X_train)
                X_val = preprocessor.transform(X_val)
            w, training_loss = train_model(
                X_train, y_train, self.method, **self.training_params
            )
            is_logistic = self.method in (
                "logistic_regression",
                "reg_logistic_regression",
            )
            y_pred, y_score = predict_from_weights(
                X_val, w, logistic=is_logistic, threshold=self.threshold
            )

        return {
            "metrics": evaluate(y_val, y_pred, y_score),
            "training_loss": training_loss,
            "y_pred": y_pred,
            "y_score": y_score,
        }
