"""Compare configured models using the same development-data folds."""

from src.cross_validation import CrossValidator
from src.reporting import save_comparison_report


class ModelComparison:
    """Keep named evaluators, shared folds, and preprocessing settings together."""

    def __init__(self, candidates, folds, preprocessor_factory=None, fold_seed=None):
        """Accept named evaluators, shared folds, and optional preprocessing.

        fold_seed records the seed used to generate the supplied folds; it does
        not regenerate them. Leave it None if unknown. Reports save exact row
        indices as well, so the folds can be reused without knowing the seed.
        """
        if not candidates:
            raise ValueError("At least one model candidate is required.")
        self.candidates = dict(candidates)
        # Materialize once so every candidate can reuse even a fold iterator.
        self.folds = list(folds)
        self.preprocessor_factory = preprocessor_factory
        self.fold_seed = fold_seed

    def run(self, X, y):
        """Return full CV results by candidate name, ranked by mean positive F1.

        Pass development data only. Each candidate uses the same folds and
        preprocessing factory, which creates a fresh instance for every fold.
        None is appropriate only when no learned preprocessing is needed.
        Equal F1 scores keep the original candidate order. Each value retains
        its summary, individual fold results, and aligned OOF predictions/scores.
        It also records a configuration snapshot and total model-fitting time.
        """
        results = {}
        for name, evaluator in self.candidates.items():
            configuration = evaluator.get_config()
            validator = CrossValidator(evaluator, self.folds, self.preprocessor_factory)
            results[name] = validator.run(X, y)
            results[name]["configuration"] = configuration
            results[name]["training_time_seconds"] = sum(
                fold["training_time_seconds"] for fold in results[name]["fold_results"]
            )

        # Use unrounded scores for ranking; rounding is only for display.
        return dict(
            sorted(
                results.items(),
                key=lambda item: item[1]["summary"]["f1"]["mean"],
                reverse=True,
            )
        )

    def save_report(
        self, results, output_path="reports/model_comparison.json", metadata=None
    ):
        """Save these results with this comparison's original folds and seed.

        Pass results from this comparison and retain its folds unchanged.
        metadata can describe the dataset/order and preprocessing configuration.
        Existing output files are replaced. Return the saved report's Path.
        """
        return save_comparison_report(
            results, self.folds, output_path, fold_seed=self.fold_seed, metadata=metadata
        )


def format_comparison_table(results):
    """Return a Markdown table from the ranked results of ModelComparison.run.

    Cells show the mean and population standard deviation to four decimals.
    Formatting preserves the results, their order, and their full precision.
    """
    metrics = ("f1", "precision", "recall", "accuracy", "roc_auc")
    lines = [
        "| Model | F1 | Precision | Recall | Accuracy | ROC-AUC |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, result in results.items():
        cells = []
        for metric in metrics:
            stats = result["summary"][metric]
            cells.append(f"{stats['mean']:.4f} ± {stats['std']:.4f}")
        # Names may contain Markdown separators or line breaks.
        label = str(name).replace("|", "\\|").replace("\n", " ")
        lines.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(lines)
