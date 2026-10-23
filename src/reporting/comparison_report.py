"""Save model comparison settings and validation measurements as JSON."""

from src.reporting.json_io import save_json_report


def save_comparison_report(results, folds, output_path, fold_seed=None, metadata=None):
    """Save results from ModelComparison.run with the exact folds used to run it.

    Keep each model's configuration, summary, total training time, and per-fold
    metrics/loss/time. Row-level predictions remain in memory. Fold indices
    refer to development row positions, so replay requires the same data/order.
    metadata accepts JSON-compatible dataset and preprocessing descriptions.
    Create parent directories and replace an existing file; return its Path.
    """
    folds = list(folds)
    report = {
        "metadata": metadata if metadata is not None else {},
        "selection_metric": "f1",
        "positive_label": 1,
        "aggregation": {"fold_weighting": "equal", "std_ddof": 0},
        "results": {},
        "cross_validation": {
            "n_folds": len(folds),
            "seed": fold_seed,
            "folds": [
                {"training_rows": training_rows, "validation_rows": validation_rows}
                for training_rows, validation_rows in folds
            ],
        },
    }
    for name, result in results.items():
        report["results"][name] = {
            "configuration": result["configuration"],
            "summary": result["summary"],
            "training_time_seconds": result["training_time_seconds"],
            "fold_results": [
                {
                    "metrics": fold["metrics"],
                    "training_loss": fold["training_loss"],
                    "training_time_seconds": fold["training_time_seconds"],
                }
                for fold in result["fold_results"]
            ],
        }
        if "threshold_tuning" in result:
            report["results"][name]["threshold_tuning"] = result["threshold_tuning"]

    return save_json_report(report, output_path)
