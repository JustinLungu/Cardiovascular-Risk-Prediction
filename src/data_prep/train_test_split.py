from pathlib import Path
import numpy as np
from course_files.helpers import load_csv_data


class TrainTestSplit:
    """Shuffle and split each class so all sets retain its original proportion.

    A fixed seed makes the split reproducible. Stratification preserves the
    dataset's class imbalance; it does not rebalance it. Fit preprocessing on
    training only.
    """

    def __init__(self, train_size=0.6, val_size=0.2, test_size=0.2, seed=42):
        sizes = np.array([train_size, val_size, test_size])
        
        if (sizes < 0).any() or not np.isclose(sizes.sum(), 1):
            raise ValueError("Split proportions must be nonnegative and sum to 1.")
        
        self.train_size = train_size
        self.val_size = val_size
        self.sizes = sizes
        self.seed = seed

    def split(self, X, y, ids):
        """Return aligned X, y, and ids for each split of already-loaded arrays."""
        if not len(X) == len(y) == len(ids):
            raise ValueError("Features, labels, and IDs must have the same length.")
        
        rng = np.random.default_rng(self.seed)
        split_names = ("train", "val", "test")

        indices = {name: [] for name in split_names}
        for label in np.unique(y):
            # shuffle class's row positions
            rows = rng.permutation(np.flatnonzero(y == label))
            # calcuate where to cut
            train_end = int(len(rows) * self.train_size)
            val_end = train_end + int(len(rows) * self.val_size)
            # split each class separately to retain its share of the dataset.
            groups = np.split(rows, [train_end, val_end])

            for position in range(len(split_names)):
                split_name = split_names[position]
                split_fraction = self.sizes[position]
                split_rows = groups[position]

                # requested split must contain rows from this class.
                if split_fraction > 0 and len(split_rows) == 0:
                    raise ValueError("Too few rows in a class for these proportions.")
                indices[split_name].append(split_rows)

        parts = {}
        for name, groups in indices.items():
            # mix the 1 & -1 classes, keeping features, labels, and IDs aligned.
            rows = rng.permutation(np.concatenate(groups))
            parts[name] = {"X": X[rows], "y": y[rows], "ids": ids[rows]}
        return parts

    def split_csv(self, data_dir="data", output_dir="data/train_test_split"):
        """Use the course loader and save six CSVs. Existing split files are replaced.

        Only the labeled x_train/y_train pair is split; the loader also reads
        the competition benchmark, which we discard here. Return the arrays
        as well so callers can use them directly in a notebook.
        """

        # covert foldernames to paths
        data_dir, output_dir = Path(data_dir), Path(output_dir)
        X, _, y, ids, _ = load_csv_data(data_dir)
        parts = self.split(X, y, ids)

        # read original column names
        with (data_dir / "x_train.csv").open() as source:
            feature_header = source.readline().strip()
        with (data_dir / "y_train.csv").open() as source:
            target_header = source.readline().strip()
        # output folder
        output_dir.mkdir(parents=True, exist_ok=True)

        # process each split and save to CSV
        for name, part in parts.items():
            np.savetxt(
                output_dir / f"x_{name}.csv",
                np.column_stack((part["ids"], part["X"])),
                delimiter=",",
                header=feature_header,
                comments="",
                fmt=["%d"] + ["%.17g"] * X.shape[1],
            )
            np.savetxt(
                output_dir / f"y_{name}.csv",
                np.column_stack((part["ids"], part["y"])),
                delimiter=",",
                header=target_header,
                comments="",
                fmt="%d",
            )
        return parts


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


if __name__ == "__main__":
    parts = TrainTestSplit().split_csv()
    for name, part in parts.items():
        print(
            f"{name}: {len(part['y']):,} rows ({np.mean(part['y'] == 1):.2%} positive)"
        )
