"""Select classification thresholds using development out-of-fold scores."""

from copy import deepcopy

import numpy as np

from src.cross_validation import summarize_fold_results
from src.metrics import evaluate, f1_score
from src.model_inputs import prepare_training_labels


class ThresholdTuner:
    """Keep per-method threshold grids and the original validation folds together."""

    def __init__(self, thresholds_by_method, folds):
        self.thresholds_by_method = {}
        for method, thresholds in thresholds_by_method.items():
            thresholds = np.asarray(thresholds, dtype=float)
            if (
                thresholds.ndim != 1
                or thresholds.size == 0
                or not np.isfinite(thresholds).all()
            ):
                raise ValueError(
                    "Threshold grids must be nonempty 1D arrays of finite values."
                )
            self.thresholds_by_method[method] = thresholds.copy()
        self.validation_rows = [np.array(rows, copy=True) for _, rows in folds]

    def run(self, results, y):
        """Return separately ranked tuned results, preserving the original results.

        Use in-memory ModelComparison results and the same development labels
        and folds, in their original row order. Supply a grid for every trained
        method; the always-negative baseline stays unchanged. No model is refit.
        These are development selection scores: labels used to select a cutoff
        also contribute to its reported F1. Keep the final holdout separate.
        """
        if not results:
            raise ValueError(
                "At least one model result is required for threshold tuning."
            )
        y = prepare_training_labels(y)
        if len(self.validation_rows) < 2:
            raise ValueError("Threshold tuning requires at least two validation folds.")
        for rows in self.validation_rows:
            if rows.ndim != 1 or rows.size == 0 or rows.dtype.kind not in "iu":
                raise ValueError(
                    "Validation indices must be nonempty 1D integer arrays."
                )
        # Every OOF row must belong to exactly one validation fold.
        rows = np.concatenate(self.validation_rows)
        if not np.array_equal(np.sort(rows), np.arange(len(y))):
            raise ValueError(
                "Validation folds must cover each development row exactly once."
            )
        if any(np.unique(y[rows]).size != 2 for rows in self.validation_rows):
            raise ValueError(
                "Each validation fold must contain both classes for ROC-AUC."
            )

        tuned_results = {}
        for name, result in results.items():
            if "oof_y_score" not in result:
                raise ValueError(
                    "Tuning needs in-memory OOF scores; JSON reports omit them."
                )
            scores = np.asarray(result["oof_y_score"])
            if scores.shape != y.shape or not np.isfinite(scores).all():
                raise ValueError(
                    "OOF scores must be finite and match the development labels."
                )
            if len(result["fold_results"]) != len(self.validation_rows):
                raise ValueError(
                    "Results must use the same validation folds as the tuner."
                )
            method = result["configuration"]["method"]
            if method == "always_negative_baseline":
                tuned_results[name] = deepcopy(result)
                continue
            if method not in self.thresholds_by_method:
                raise ValueError(f"Provide a threshold grid for {method}.")
            tuned_results[name] = self._tune_candidate(result, y, scores)

        return dict(
            sorted(
                tuned_results.items(),
                key=lambda item: item[1]["summary"]["f1"]["mean"],
                reverse=True,
            )
        )

    def _tune_candidate(self, result, y, scores):
        method = result["configuration"]["method"]
        original_threshold = float(result["configuration"]["threshold"])
        # Try the original first: ties retain it, then follow supplied grid order.
        thresholds = list(
            dict.fromkeys(
                [original_threshold] + self.thresholds_by_method[method].tolist()
            )
        )
        trials = []
        for threshold in thresholds:
            predictions = np.where(scores >= threshold, 1, -1)
            fold_f1 = [
                f1_score(y[rows], predictions[rows]) for rows in self.validation_rows
            ]
            trials.append({"threshold": threshold, "mean_f1": float(np.mean(fold_f1))})
        selected = trials[int(np.argmax([trial["mean_f1"] for trial in trials]))][
            "threshold"
        ]

        tuned = deepcopy(result)
        tuned["configuration"]["threshold"] = selected
        tuned["oof_y_pred"] = np.where(scores >= selected, 1, -1)
        for rows, fold in zip(self.validation_rows, tuned["fold_results"]):
            fold["y_pred"] = tuned["oof_y_pred"][rows]
            fold["y_score"] = scores[rows].copy()
            fold["metrics"] = evaluate(y[rows], fold["y_pred"], fold["y_score"])
        tuned["summary"] = summarize_fold_results(tuned["fold_results"])
        tuned["threshold_tuning"] = {
            "selection_data": "development_oof",
            "selection_metric": "mean_fold_f1",
            "original_threshold": original_threshold,
            "original_summary": deepcopy(result["summary"]),
            "selected_threshold": selected,
            "candidates": trials,
        }
        return tuned
