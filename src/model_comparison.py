"""Compare configured models using the same development-data folds."""

from src.cross_validation import CrossValidator


class ModelComparison:
    """Keep named evaluators, shared folds, and preprocessing settings together."""

    def __init__(self, candidates, folds, preprocessor_factory=None):
        """Accept a name-to-ModelEvaluator mapping and folds from stratified_k_fold."""
        if not candidates:
            raise ValueError("At least one model candidate is required.")
        self.candidates = dict(candidates)
        # Materialize once so every candidate can reuse even a fold iterator.
        self.folds = list(folds)
        self.preprocessor_factory = preprocessor_factory

    def run(self, X, y):
        """Return full CV results by candidate name, ranked by mean positive F1.

        Pass development data only. Each candidate uses the same folds and
        preprocessing factory, which creates a fresh instance for every fold.
        None is appropriate only when no learned preprocessing is needed.
        Equal F1 scores keep the original candidate order. Each value retains
        its summary, individual fold results, and aligned OOF predictions/scores.
        """
        results = {}
        for name, evaluator in self.candidates.items():
            validator = CrossValidator(evaluator, self.folds, self.preprocessor_factory)
            results[name] = validator.run(X, y)

        # Use unrounded scores for ranking; rounding is only for display.
        return dict(
            sorted(
                results.items(),
                key=lambda item: item[1]["summary"]["f1"]["mean"],
                reverse=True,
            )
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
