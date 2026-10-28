# Final prediction entry point: load data, apply the selected preprocessing,
# train the selected model or load saved weights, and create the submission CSV.
# Preserve test IDs and output labels in {-1, 1} using the course submission helper.
# Fix randomness and configuration to reproduce the best submitted predictions.
# Keep this file at repo root; reusable logic belongs in src/.

import argparse
import csv
from pathlib import Path

from course_files.helpers import create_csv_submission, load_csv_data
from src.data_prep.feature_selection import feature_selection
from src.data_prep.preprocessing import preproc_MSCOHE
from src.models.prediction import predict_from_weights
from src.models.training import train_model


PROJECT_ROOT = Path(__file__).resolve().parent

# Frozen using development cross-validation, before the final holdout evaluation.
MODEL_CONFIG = {
    "method": "reg_logistic_regression",
    "max_it": 3000,
    "gamma": 0.05,
    "seed": 42,
    "lmbd": 1e-6,
}
THRESHOLD = 0.2


def generate_submission(data_dir, output_path):
    """Refit the selected pipeline on all labeled rows and save benchmark labels.

    data_dir contains the original x_train.csv, y_train.csv and x_test.csv.
    Existing output files are replaced. Return the saved submission's Path.
    """
    data_dir, output_path = Path(data_dir), Path(output_path)
    with (data_dir / "x_train.csv").open(newline="") as source:
        feature_names = next(csv.reader(source))[1:]
    with (data_dir / "x_test.csv").open(newline="") as source:
        benchmark_names = next(csv.reader(source))[1:]
    if feature_names != benchmark_names:
        raise ValueError(
            "Training and benchmark feature names must match in the same order."
        )

    X_train, X_benchmark, y_train, _, benchmark_ids = load_csv_data(data_dir)
    # Both datasets need the same column selection and survey-code cleaning.
    X_train, _, numeric_mask, categorical_mask = feature_selection(
        X_train, feature_names
    )
    X_benchmark, _, _, _ = feature_selection(X_benchmark, benchmark_names)

    # Learn medians, scaling and categories from all labeled rows only.
    preprocessor = preproc_MSCOHE(numeric_mask, categorical_mask)
    X_train = preprocessor.fit_transform(X_train)
    X_benchmark = preprocessor.transform(X_benchmark)
    weights, _ = train_model(X_train, y_train, **MODEL_CONFIG)
    predictions, _ = predict_from_weights(
        X_benchmark, weights, logistic=True, threshold=THRESHOLD
    )

    # The course writer uses zip, which would silently truncate unequal lengths.
    if len(predictions) != len(benchmark_ids):
        raise ValueError("One prediction is required for every benchmark ID.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_csv_submission(benchmark_ids, predictions, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Train the selected model and generate competition predictions."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "submission.csv")
    args = parser.parse_args()
    output_path = generate_submission(args.data_dir, args.output)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
