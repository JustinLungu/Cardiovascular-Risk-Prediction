# Cardiovascular-Risk-Prediction
From-scratch cardiovascular risk prediction on 300k+ BRFSS records using NumPy, with custom ML algorithms, preprocessing, and cross-validation.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies.

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or via a package manager:

```bash
# macOS (Homebrew)
brew install uv

# pipx
pipx install uv
```

Verify the install:

```bash
uv --version
```

### 2. Set up the project environment

From the repo root:

```bash
uv sync
```

This creates a `.venv/` using Python 3.9, NumPy 1.23.1, and Matplotlib 3.5.2, matching the course grading environment. `uv.lock` pins the resolved dependencies. Development tools (pytest 7.1.2, GitPython 3.1.18, and Black 22.6.0) are included by default for the supplied tests; they are not libraries for the ML implementation. Use `uv sync --no-dev` for just the project dependencies.

### 3. Run code

```bash
uv run python your_script.py
```

or activate the environment directly:

```bash
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

## Download the dataset

To download and prepare the dataset, run:

```bash
bash scripts/prepare_data.sh
```

## Code structure

`implementations.py` and `run.py` stay at the repository root because the course grading tests expect them there. The former contains the six required ML methods; the latter is a thin entry point for reproducing predictions. Other reusable code lives in `src/`, with data preparation in `src/data_prep/` and exploration in `src/data_exploration/`.

| Module | Responsibility |
| --- | --- |
| `src/train_test_split.py` | `TrainTestSplit` and development/holdout preparation. |
| `src/cross_validation.py` | Shared stratified folds, the `CrossValidator` runner, and metric summaries. |
| `src/model_inputs.py` | Intercept columns and training-label conversion. |
| `src/training.py` | Fit the existing course models. |
| `src/prediction.py` | Class predictions, continuous scores, and the baseline. |
| `src/metrics.py` | Classification metric calculations. |
| `src/evaluation.py` | `ModelEvaluator`: model settings and one-fold evaluation. |
| `src/model_comparison.py` | `ModelComparison`: evaluate named candidates on shared folds and format their ranked results. |
| `src/reporting.py` | Save comparison settings, exact folds, metrics, and fitting times as JSON. |
| `src/visualization.py` | Plot mean validation F1 with fold standard deviation error bars. |
| `src/threshold_tuning.py` | Select thresholds from development OOF scores while retaining the original results. |

Exploratory notebooks live in `notebooks/`, starting with [exploration.ipynb](notebooks/exploration.ipynb). When opening a notebook in your editor, select the Python kernel from the project's `.venv` environment.

## Model evaluation protocol

The NumPy model comparison workflow is being implemented on `feat/model-comparison`. The following protocol defines how we will compare candidates and select the final configuration.

### Development data and final test set

Reuse the existing stratified 60/20/20 partitions from `src/train_test_split.py`. Combine training and validation into development data and preserve the existing test partition.

| Dataset | Share of labeled rows | Purpose |
| --- | --- | --- |
| Development | 80% | Compare models and select preprocessing, hyperparameters, and prediction thresholds using cross-validation. |
| Local test | 20% | Evaluate the selected pipeline after model selection. |
| Competition benchmark | Separate, unlabeled data | Generate submission predictions. |

`data/train_test_split/x_test.csv` and its paired labels form the local test set. `data/x_test.csv` is the competition benchmark with hidden labels.

For an already-loaded split dictionary returned by `TrainTestSplit`, prepare the two datasets with:

```python
from src.train_test_split import prepare_development_and_holdout

development, holdout = prepare_development_and_holdout(parts)
X_dev, y_dev = development["X"], development["y"]
X_test, y_test = holdout["X"], holdout["y"]
```

The helper joins train and validation in that order, retaining features, raw labels, missing values, and IDs. It returns the existing test partition unchanged and checks that row IDs are unique across all three input partitions.

### Cross-validation and preprocessing

Use five-fold stratified cross-validation within development data. Generate the fold indices once with seed `42` and reuse identical folds for every candidate. In each round, four folds train the model and one fold validates it: approximately 64% and 16% of all labeled rows, respectively. Every development row appears in validation exactly once; the local test set stays reserved throughout model selection.

Generate the folds from development labels only:

```python
from src.cross_validation import stratified_k_fold

folds = stratified_k_fold(development["y"], k=5, seed=42)
for training_rows, validation_rows in folds:
    X_train, y_train = development["X"][training_rows], development["y"][training_rows]
    X_val, y_val = development["X"][validation_rows], development["y"][validation_rows]
```

The returned list contains row positions relative to development data and can be reused for all models. Each class needs at least `k` rows so every validation fold includes it. Class counts differ by at most one across validation folds; rounding can affect their total sizes.

Within each fold, fit all learned preprocessing on its training rows, including imputation, scaling, categorical encoders, and any data-driven feature selection. Apply those fitted parameters and the same feature-column order to validation rows. Preprocessing integration must provide a way to fit on training rows and transform other rows without refitting.

Control training randomness as well as fold generation, including the sampling used by SGD. Record the seeds with the experiment settings.

### Labels, metrics, and model selection

The positive class is `+1`. Evaluation labels and class predictions use `{-1, +1}`. Convert training labels to `{0, 1}` when fitting logistic regression, then map its class predictions back for evaluation. Use continuous probabilities or decision scores when calculating ROC-AUC.

`confusion_counts(y_true, y_pred)` in `src/metrics.py` returns integer counts under the keys `tp`, `tn`, `fp`, and `fn`: correctly predicted positives, correctly predicted negatives, false alarms, and missed positives. It accepts matching 1D arrays of `{-1, +1}` labels, with zero counts for empty inputs. Prediction scores must be converted to class labels before calling it.

`accuracy(y_true, y_pred)`, `precision(y_true, y_pred)`, and `recall(y_true, y_pred)` use the same label inputs. Accuracy is `(tp + tn) / total`, precision is `tp / (tp + fp)`, and recall is `tp / (tp + fn)`. Each returns a fraction between 0 and 1; a zero denominator returns `0.0`, including for empty inputs.

`f1_score(y_true, y_pred)` combines positive-class precision and recall using their harmonic mean: `2 * tp / (2 * tp + fp + fn)`. It uses the same label inputs and returns `0.0` for a zero denominator. An all-negative predictor therefore receives F1 `0.0`, even when class imbalance gives it high accuracy.

`roc_auc_score(y_true, y_score)` measures how well scores rank positives above negatives across prediction thresholds. Supply finite probabilities or decision scores, with higher scores indicating `+1`. Perfect ranking scores `1.0`, reversed ranking scores `0.0`, and equal scores for all rows give `0.5`. Ties are handled together. Labels must include both classes; empty or single-class inputs raise `ValueError` because ROC-AUC is undefined for them.

`evaluate(y_true, y_pred, y_score)` returns all five metrics in one dictionary with keys `accuracy`, `precision`, `recall`, `f1`, and `roc_auc`. Supply class predictions made using the threshold chosen in the training workflow, plus continuous scores for ROC-AUC. It follows the same input rules and requires both true classes.

```python
from src.metrics import evaluate

results = evaluate(
    y_true=[-1, -1, 1, 1],
    y_pred=[-1, -1, -1, 1],
    y_score=[0.1, 0.4, 0.35, 0.8],
)
# accuracy=0.75, precision=1.0, recall=0.5, f1=2/3, roc_auc=0.75
```

Rank candidate configurations by mean positive-class F1 across the five validation folds. Report fold standard deviations, positive-class precision and recall, accuracy, ROC-AUC, and training time alongside F1. This is our local model selection policy; retain both F1 and accuracy for comparison with competition scores. Keep each model's training loss as a diagnostic, since different objectives such as MSE, logistic loss, and hinge loss are not directly comparable.

Select hyperparameters and prediction thresholds using development cross-validation. For threshold selection, collect out-of-fold scores: each development row's score comes from a model trained on the other folds. Save model settings, thresholds, seeds, and per-fold results so experiments can be reproduced.

### Model inputs and predictions

Use `src/model_inputs.py` for model inputs and `src/prediction.py` for predictions, with these conventions:

| Models | Training labels | Prediction scores | Default threshold |
| --- | --- | --- | --- |
| Least squares (closed form, GD, SGD), ridge, and linear SVM | `{-1, +1}` | Linear decision scores | `0.0` |
| Logistic and regularized logistic regression | `{0, 1}` | Probability of `+1` | `0.5` |

`prepare_training_labels(y, logistic=False)` in `src/model_inputs.py` accepts the original `{-1, +1}` labels and returns a separate training array. Set `logistic=True` to map `-1` to `0` and keep `+1` as `1`; keep the original labels for evaluation.

`add_intercept(X)` in the same module prepends one column of ones after preprocessing, giving the models an intercept weight at `w[0]`. Use it once on training features before passing them to a course training function. The existing regularized implementations penalize all weights, including this intercept.

`predict_from_weights(X, w, logistic=False, threshold=None)` in `src/prediction.py` takes preprocessed prediction features **without** an intercept column and adds it internally. It returns `(y_pred, y_score)` with `{-1, +1}` predictions and continuous scores for ROC-AUC. Set `logistic=True` for logistic weights. A custom `threshold` changes class predictions while preserving scores; equality with the threshold predicts `+1`. Select thresholds using development validation results.

### Training models

`train_model(X, y, method="least_squares", max_it=1000, gamma=0.01, seed=42, lmbd=0.1)` in `src/training.py` calls the existing functions in `implementations.py`. Choose `least_squares`, `mean_squared_error_gd`, or `mean_squared_error_sgd` for three ways to fit the same least-squares model, `ridge_regression` for its L2-regularized version, `logistic_regression` for logistic classification, `reg_logistic_regression` for logistic classification with L2 regularization, or `svm` for linear support vector classification.

Pass preprocessed, finite features **without** an intercept column and the original `{-1, +1}` labels. The helper adds the intercept and prepares a separate label array. It returns `(w, loss)`: final training half-MSE for least squares/ridge, mean negative log-likelihood for both logistic variants, or mean hinge loss for SVM. Returned losses exclude regularization penalties. Use `evaluate(...)` for classification metrics. Closed-form least squares requires linearly independent columns after adding the intercept; a singular matrix raises NumPy's `LinAlgError`.

Iterative methods start from zero weights and use `max_it` updates with learning rate `gamma`. SGD uses `seed` for each call and restores the previous NumPy random state afterward. Select iteration counts and learning rates using development validation results.

For ridge, use `train_model(X_train, y_train, method="ridge_regression", lmbd=0.1)`. The strength `lmbd` must be a finite, nonnegative scalar; larger values apply stronger shrinkage. Setting it to `0.0` matches least squares when the design matrix has full column rank. A positive strength also makes collinear columns solvable. The returned training half-MSE excludes the regularization penalty. Select `lmbd` using development validation results.

```python
from src.training import train_model
from src.prediction import predict_from_weights
from src.metrics import evaluate

w, training_loss = train_model(
    X_train, y_train, method="mean_squared_error_sgd",
    max_it=1000, gamma=0.01, seed=42,
)
y_pred, y_score = predict_from_weights(X_val, w)
validation_metrics = evaluate(y_val, y_pred, y_score)
```

For `method="logistic_regression"` or `method="reg_logistic_regression"`, the training helper converts a copy of the labels to `{0, 1}`. Use `logistic=True` when predicting from its weights to obtain positive-class probabilities and apply the default `0.5` threshold. Keep the original `{-1, +1}` labels for evaluation:

```python
w, training_loss = train_model(
    X_train, y_train, method="logistic_regression", max_it=1000, gamma=0.01,
)
y_pred, probabilities = predict_from_weights(X_val, w, logistic=True)
validation_metrics = evaluate(y_val, y_pred, probabilities)
```

For regularized logistic regression, call `train_model(X_train, y_train, method="reg_logistic_regression", lmbd=0.1)`. The finite, nonnegative `lmbd` controls the L2 penalty on all weights, including the intercept. Setting it to `0.0` matches ordinary logistic regression with the same iteration count and learning rate. Select `lmbd` using development validation results.

For linear SVM, call `train_model(X_train, y_train, method="svm", lmbd=0.1)`. It keeps `{-1, +1}` training labels and uses gradient descent on mean hinge loss plus `(lmbd / 2) * ||w||²`, including the intercept in the penalty. The strength must be finite and nonnegative. Use `predict_from_weights(X_val, w)` to obtain class predictions at threshold `0.0` and continuous decision scores for ROC-AUC. These scores can be negative or greater than `1`. Select `lmbd`, `max_it`, and `gamma` using development validation results.

### Always-negative baseline

`always_negative_baseline(X)` in `src/prediction.py` returns `(y_pred, y_score)`, predicting `-1` for every feature row with constant decision scores of `-1.0`. It needs no fitting or preprocessing. With both classes present, precision, recall, and F1 are `0.0`, ROC-AUC is `0.5`, and accuracy equals the share of negative labels. This reference shows how much accuracy can come from class imbalance alone.

```python
from src.prediction import always_negative_baseline
from src.metrics import evaluate

y_pred, y_score = always_negative_baseline(X_val)
baseline_metrics = evaluate(y_val, y_pred, y_score)
```

### Evaluate one validation fold

`ModelEvaluator` in `src/evaluation.py` keeps the model method, threshold, and training parameters together. Configure it once, then call `evaluator.evaluate_fold(X_train, y_train, X_val, y_val, ...)` on the development rows selected by `stratified_k_fold`. The evaluator combines preprocessing, training, prediction, and metrics. Settings such as `lmbd`, `max_it`, `gamma`, and `seed` are forwarded to `train_model` on each call.

For learned preprocessing, pass a fresh instance as `preprocessor` to each `evaluate_fold` call. It must provide `fit(X_train, y_train)` and `transform(X)`: fitting receives only training features and their original `{-1, +1}` labels, then transformation runs on training and validation separately. Transformations must preserve row counts/order and produce the same feature columns for both portions, without an intercept column. The evaluator copies input arrays before preprocessing so operations in place preserve the caller's raw data.

The production preprocessing module is still a placeholder. Use `preprocessor=None` only when no learned preprocessing is needed, as in this finite synthetic example:

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator(method="ridge_regression", lmbd=0.1)
fold_result = evaluator.evaluate_fold(
    X_train=[[0.0], [1.0]], y_train=[-1, 1],
    X_val=[[0.25], [0.75]], y_val=[-1, 1],
)
print(fold_result["metrics"])
```

The result contains `metrics`, `training_loss`, `training_time_seconds`, `y_pred`, and `y_score`; prediction arrays retain validation row order for later out-of-fold collection. Timing covers the `train_model` call, including its input preparation, and excludes learned preprocessing, prediction, and metric calculation. Configure `ModelEvaluator(method="always_negative_baseline")` for the reference predictor, which skips preprocessing and returns `None` for training loss and `0.0` for training time. Validation labels must include both classes to calculate ROC-AUC.

### Evaluate all development folds

`CrossValidator` in `src/cross_validation.py` repeats one configured evaluator across the shared folds. Generate the folds once and reuse them for every candidate. This example assumes finite features that need no learned preprocessing:

```python
from src.cross_validation import CrossValidator, stratified_k_fold
from src.evaluation import ModelEvaluator

folds = stratified_k_fold(y_dev, k=5, seed=42)
evaluator = ModelEvaluator(method="ridge_regression", lmbd=0.1)
validator = CrossValidator(evaluator, folds)
cv_result = validator.run(X_dev, y_dev)

print(cv_result["summary"]["f1"])  # {"mean": ..., "std": ...}
oof_predictions = cv_result["oof_y_pred"]
oof_scores = cv_result["oof_y_score"]
```

For learned preprocessing, pass `preprocessor_factory` to `CrossValidator`: a callable that constructs a new preprocessor for each fold, such as your preprocessing class or a `lambda` supplying its settings. Each instance follows the `fit(X_train, y_train)` / `transform(X)` interface above. Pass a constructor rather than an already-created instance; fitting stays inside each fold. The production preprocessing implementation is still pending.

`fold_results` contains the one-fold result dictionaries in the same order as `folds`. The out-of-fold (OOF) arrays follow the original development row order, so they align with `y_dev` and development IDs. Each row receives a prediction from a model trained on the other folds. The runner checks that each fold separates training and validation rows and that every development row validates exactly once, before fitting any models. Pass only development data; the reserved local test set stays separate.

`summary` reports `{"mean": ..., "std": ...}` for accuracy, precision, recall, F1, and ROC-AUC. Each fold counts equally, including when rounding produces slightly different validation sizes. Standard deviation uses `ddof=0` and describes how much the observed fold scores vary; it is not a confidence interval. For example, F1 values of `0.6` and `0.8` have mean `0.7` and standard deviation `0.1`. These statistics summarize the individual fold metrics; calculating a metric once over all OOF predictions can give a different result. Training loss stays in each fold result because different model families use different loss functions.

To summarize existing fold results without training again, call `summarize_fold_results(cv_result["fold_results"])` from `src/cross_validation.py`.

### Compare model candidates

`ModelComparison` in `src/model_comparison.py` runs named `ModelEvaluator` instances on the same development folds and returns their complete CV results ranked by mean positive-class F1. Pass the shared folds generated above. For finite features that need no learned preprocessing:

```python
from src.evaluation import ModelEvaluator
from src.model_comparison import ModelComparison, format_comparison_table

candidates = {
    "Baseline": ModelEvaluator("always_negative_baseline"),
    "Least squares": ModelEvaluator("least_squares"),
    "MSE (GD)": ModelEvaluator("mean_squared_error_gd", max_it=1000, gamma=0.01),
    "MSE (SGD)": ModelEvaluator(
        "mean_squared_error_sgd", max_it=1000, gamma=0.01, seed=42,
    ),
    "Ridge": ModelEvaluator("ridge_regression", lmbd=0.1),
    "Logistic": ModelEvaluator("logistic_regression", max_it=1000, gamma=0.01),
    "Regularized logistic": ModelEvaluator(
        "reg_logistic_regression", lmbd=0.1, max_it=1000, gamma=0.01,
    ),
    "Linear SVM": ModelEvaluator("svm", lmbd=0.1, max_it=1000, gamma=0.01),
}
comparison = ModelComparison(candidates, folds, fold_seed=42)
results = comparison.run(X_dev, y_dev)
print(format_comparison_table(results))

ridge_summary = results["Ridge"]["summary"]
ridge_oof_scores = results["Ridge"]["oof_y_score"]
```

For learned preprocessing, pass `preprocessor_factory` to `ModelComparison` using the interface above. Every model uses that factory, with a fresh instance per fold and fitting on training rows only. The real preprocessing implementation is still pending. Keep the reserved local test set outside this workflow.

The table shows F1, precision, recall, accuracy, and ROC-AUC as `mean ± std` to four decimal places. Ranking uses full-precision mean F1; ties keep the original candidate order. Complete summaries, fold results, and OOF arrays remain accessible by candidate name. The example settings demonstrate the API; select the actual model settings using development validation results.

### Compare hyperparameter configurations

Use `ModelComparison.from_configurations` to evaluate small, explicit lists of settings for each method. It creates the named evaluators for the existing comparison runner. Each dictionary is one complete combination to try; omitted arguments use the training helper's defaults, and `[{}]` evaluates a method with defaults. This finite-feature example reuses the same development folds:

```python
configurations = {
    "always_negative_baseline": [{}],
    "ridge_regression": [
        {"lmbd": 0.001},
        {"lmbd": 0.01},
        {"lmbd": 0.1},
    ],
    "logistic_regression": [
        {"max_it": 1000, "gamma": 0.01},
        {"max_it": 5000, "gamma": 0.01},
        {"max_it": 5000, "gamma": 0.05},
    ],
}
comparison = ModelComparison.from_configurations(configurations, folds, fold_seed=42)
results = comparison.run(X_dev, y_dev)
print(format_comparison_table(results))

best_name = next(iter(results))  # Results are ranked by mean validation F1.
best_config = results[best_name]["configuration"]
```

Pass `preprocessor_factory` when learned preprocessing is needed, following the training-only fit rules above. Every configuration uses the same folds and gets fresh preprocessing for each fold. Candidate names include the method, a per-method index, and the supplied settings. Every requested entry is retained, including duplicate settings, with complete effective configuration snapshots, fold metrics, fitting times, and OOF arrays. An empty configuration list raises an error.

The existing report and plot helpers accept these results directly. Select small candidate lists using development data and keep each result for review; the example values are starting choices for demonstrating the API. The example uses default classification thresholds. Threshold selection using OOF scores follows separately, and the reserved local test set stays untouched during selection.

### Select classification thresholds

`ThresholdTuner` in `src/threshold_tuning.py` tries explicit threshold grids against the existing OOF scores. Supply the same folds and development labels used for model comparison. A grid is required for each trained method; configurations of that method share the grid. Logistic scores are probabilities, while ridge, MSE, and SVM use raw decision scores, so choose grids in their respective score units. For the methods in the hyperparameter example:

```python
from src.threshold_tuning import ThresholdTuner

default_results = results
tuner = ThresholdTuner(
    {
        "ridge_regression": [-0.5, -0.25, 0.0, 0.25, 0.5],
        "logistic_regression": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    },
    folds,
)
tuned_results = tuner.run(default_results, y_dev)
print(format_comparison_table(tuned_results))

best_name = next(iter(tuned_results))
best_config = tuned_results[best_name]["configuration"]
comparison.save_report(default_results, "reports/default_thresholds.json")
comparison.save_report(tuned_results, "reports/tuned_thresholds.json")
```

For each candidate, selection maximizes mean positive-class F1 across the original validation folds. The original cutoff is always included. Ties retain it if it is optimal, then follow the supplied grid order. A score at or above the cutoff predicts `+1`. The always-negative baseline remains unchanged. Models are not refit, and their scores, ROC-AUC, training losses, and original fitting times remain unchanged.

The returned results have updated predictions, fold metrics, summaries, and threshold configuration, and are ranked by tuned mean F1. The original results are preserved separately. Each tuned candidate also records `threshold_tuning`: its original cutoff/summary, selected cutoff, and every tried threshold's mean F1. JSON reports retain this information, and the table/plot helpers label tuned output as development threshold-selection results. Tuning requires the in-memory OOF arrays from `comparison.run`; the compact JSON reports omit those arrays.

These tuned metrics are **development selection scores**. Each OOF prediction comes from a model trained on other rows, but choosing its cutoff uses the development labels. The resulting score can be optimistic after selection. Freeze the chosen model settings and cutoff before the one final evaluation on the reserved local test set.

### Save a comparison report

Each comparison result also contains a `configuration` snapshot and `training_time_seconds`, the sum of fitting times across its folds. The configuration records the method, effective threshold, and shared training-helper defaults with any explicit overrides. Defaults such as the SGD seed are therefore recorded even when omitted by the caller. Some helper settings apply only to particular methods; the baseline has no training parameters or threshold. Changing an evaluator's settings after a run preserves the configuration already recorded with that run.

Save the results without training again:

```python
report_path = comparison.save_report(
    results,
    "reports/model_comparison.json",
    metadata={
        "dataset": "Describe the development dataset and its row order here",
        "preprocessing": None,  # No learned preprocessing in the finite example.
    },
)
```

The JSON contains ranked model configurations, metric means and standard deviations, each fold's metrics/loss/fitting time, total fitting times, and the exact training/validation indices in their original order. OOF arrays remain in memory. Add the actual dataset source/order and preprocessing class/settings to `metadata` for your experiment. All metadata values must be JSON-compatible; NumPy arrays and scalars are supported.

`fold_seed` records the seed used when generating the supplied folds; it does not generate or change them. An unknown seed is saved as `null`. The saved indices allow exact fold reuse with the same development data and row order. Reports use Python's standard JSON format and can be reopened with `json.load`; `report["results"]` also works with `format_comparison_table`.

The output directory is created automatically and an existing report at the same path is replaced. Generated files under `reports/` are gitignored.

### Plot a comparison

`plot_model_comparison` in `src/visualization.py` draws a horizontal F1 chart using the existing metric summaries. It keeps the comparison order, with the highest-ranked candidate at the top. Error bars show one population standard deviation across folds (`ddof=0`), describing fold variation rather than a confidence interval. The axis includes the full error bars, even when mean ± standard deviation extends beyond the 0–1 score range.

```python
from pathlib import Path
import matplotlib.pyplot as plt
from src.visualization import plot_model_comparison

fig, ax = plot_model_comparison(results)
Path("reports").mkdir(exist_ok=True)
fig.savefig("reports/model_comparison_f1.png", dpi=150, bbox_inches="tight")
plt.show()
```

To plot a saved report without retraining, load it with `json.load` and pass `report["results"]` instead. The function returns the Matplotlib figure and axes for customization; pass `ax=...` to draw on an existing subplot. The plotting helper preserves the results and leaves saving/display to the caller. Plot files under `reports/` are gitignored.

### Final evaluation and submission

After selecting and freezing a configuration, refit preprocessing and the model on all 80% development data. Apply that fitted pipeline and its selected threshold to the reserved 20% local test set for the final local evaluation.

After recording the final local result, refit the selected pipeline on all labeled data before generating competition benchmark predictions through `run.py`.

The final training and evaluation workflow uses NumPy and the Python standard library, with permitted visualization libraries for plots. The existing Pandas/scikit-learn experiments remain temporary reference work until the NumPy replacement is verified; their removal is scheduled for the end of the model comparison work.
