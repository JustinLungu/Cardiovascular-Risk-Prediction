"""Save model comparison settings and validation measurements as JSON."""

import json
from pathlib import Path

import numpy as np

def _json_default(value):
    """Convert NumPy arrays/scalars without rounding their numeric values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON.")


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

    # Serialize before opening the file, so invalid values preserve prior output.
    serialized = json.dumps(
        report, indent=2, default=_json_default, allow_nan=False, ensure_ascii=False
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized + "\n", encoding="utf-8")
    return output_path
