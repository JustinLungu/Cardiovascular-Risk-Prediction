"""Save the final labeled-holdout evaluation and its frozen configuration."""

from src.reporting.json_io import save_json_report


def save_holdout_report(result, output_path, metadata=None):
    """Save JSON metrics, confusion matrix, configuration, and exact row IDs.

    Prediction arrays stay in memory. Include dataset and preprocessing details
    plus development-selection provenance in metadata. No model is trained here.
    """
    report = {
        "metadata": metadata if metadata is not None else {},
        "evaluation_data": "holdout",
        "positive_label": 1,
    }
    for field in (
        "configuration",
        "metrics",
        "confusion_counts",
        "confusion_matrix",
        "training_loss",
        "training_time_seconds",
        "data",
    ):
        report[field] = result[field]
    return save_json_report(report, output_path)
