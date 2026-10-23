"""Evaluate one frozen configuration on the reserved labeled test partition."""

from copy import deepcopy

import numpy as np

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import confusion_counts
from src.models.model_inputs import prepare_training_labels
from src.reporting.holdout_report import save_holdout_report


class HoldoutEvaluator:
    """Refit on all development data after configuration selection is finished.

    configuration comes from a selected comparison result. Its model settings
    and threshold are copied immediately; this class does no model selection
    or threshold tuning. Call evaluate only for the final reserved-test check.
    A factory must create fresh preprocessing with the development-selected
    settings. Keep those factory settings fixed and describe them in metadata.
    """

    def __init__(self, configuration, preprocessor_factory=None):
        evaluator = ModelEvaluator(
            method=configuration["method"],
            threshold=configuration["threshold"],
            **deepcopy(configuration["training_params"]),
        )
        self._configuration = evaluator.get_config()
        self.preprocessor_factory = preprocessor_factory

    def get_config(self):
        """Return a separate copy of the frozen settings for later refitting."""
        return deepcopy(self._configuration)

    def evaluate(self, development, holdout):
        """Fit on development, then score the holdout without choosing settings.

        Both inputs contain aligned X, y, and ids in the format returned by
        prepare_development_and_holdout. Use original {-1, +1} labels and raw
        features with matching column order. IDs must identify distinct rows;
        this checks their separation, not whether the caller paired them with
        the correct feature/label rows. Both holdout classes are needed for AUC.
        Return predictions/scores aligned with copied holdout IDs, plus metrics,
        confusion counts/matrix, fitting diagnostics, and configuration.
        """
        X_dev, y_dev, dev_ids = self._validate_partition(development, "Development")
        X_test, y_test, test_ids = self._validate_partition(holdout, "Holdout")
        # Reject respondent overlap or duplicates before any preprocessing/fitting.
        all_ids = np.concatenate((dev_ids, test_ids))
        if np.unique(all_ids).size != all_ids.size:
            raise ValueError("Development and holdout IDs must be unique and disjoint.")
        if np.unique(y_test).size != 2:
            raise ValueError("Holdout labels must contain both -1 and +1 for ROC-AUC.")

        config = self.get_config()
        preprocessor = None
        if (
            self.preprocessor_factory is not None
            and config["method"] != "always_negative_baseline"
        ):
            preprocessor = self.preprocessor_factory()
        # Reuse the existing fit/predict path with the full development portion.
        evaluator = ModelEvaluator(
            config["method"], config["threshold"], **config["training_params"]
        )
        result = evaluator.evaluate_fold(
            X_dev, y_dev, X_test, y_test, preprocessor=preprocessor
        )
        counts = confusion_counts(y_test, result["y_pred"])
        result.update(
            configuration=config,
            confusion_counts=counts,
            confusion_matrix={
                "labels": [-1, 1],
                "rows": "actual",
                "columns": "predicted",
                "values": [[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]],
            },
            data={
                "development_rows": len(y_dev),
                "holdout_rows": len(y_test),
                "development_ids": dev_ids.copy(),
                "holdout_ids": test_ids.copy(),
            },
        )
        return result

    def save_report(
        self, result, output_path="results/holdout_evaluation.json", metadata=None
    ):
        """Save JSON metrics and settings, excluding row-level predictions.

        Record dataset/order, development-selection provenance, and the chosen
        preprocessing class/settings in metadata. The saved configuration can
        later be used to refit on all labeled data for the run.py workflow.
        """
        return save_holdout_report(result, output_path, metadata=metadata)

    @staticmethod
    def _validate_partition(partition, name):
        """Require nonempty aligned arrays before fitting the final model."""
        X = np.asarray(partition["X"])
        y = prepare_training_labels(partition["y"])
        ids = np.asarray(partition["ids"])
        if (
            X.ndim != 2
            or ids.ndim != 1
            or len(y) == 0
            or len(X) != len(y)
            or len(ids) != len(y)
        ):
            raise ValueError(f"{name} must have aligned nonempty X, y, and 1D IDs.")
        return X, y, ids
