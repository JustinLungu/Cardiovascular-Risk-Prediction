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
from src.training import prepare_development_and_holdout

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

Use the helpers in `src/training.py` to keep model conventions consistent:

| Models | Training labels | Prediction scores | Default threshold |
| --- | --- | --- | --- |
| Least squares (closed form, GD, SGD), ridge, and linear SVM | `{-1, +1}` | Linear decision scores | `0.0` |
| Logistic and regularized logistic regression | `{0, 1}` | Probability of `+1` | `0.5` |

`prepare_training_labels(y, logistic=False)` accepts the original `{-1, +1}` labels and returns a separate training array. Set `logistic=True` to map `-1` to `0` and keep `+1` as `1`; keep the original labels for evaluation.

`add_intercept(X)` prepends one column of ones after preprocessing, giving the models an intercept weight at `w[0]`. Use it once on training features before passing them to a course training function. The existing regularized implementations penalize all weights, including this intercept.

`predict_from_weights(X, w, logistic=False, threshold=None)` takes preprocessed prediction features **without** an intercept column and adds it internally. It returns `(y_pred, y_score)` with `{-1, +1}` predictions and continuous scores for ROC-AUC. Set `logistic=True` for logistic weights. A custom `threshold` changes class predictions while preserving scores; equality with the threshold predicts `+1`. Select thresholds using development validation results.

### Always-negative baseline

`always_negative_baseline(X)` in `src/training.py` returns `(y_pred, y_score)`, predicting `-1` for every feature row with constant decision scores of `-1.0`. It needs no fitting or preprocessing. With both classes present, precision, recall, and F1 are `0.0`, ROC-AUC is `0.5`, and accuracy equals the share of negative labels. This reference shows how much accuracy can come from class imbalance alone.

```python
from src.training import always_negative_baseline
from src.metrics import evaluate

y_pred, y_score = always_negative_baseline(X_val)
baseline_metrics = evaluate(y_val, y_pred, y_score)
```

### Final evaluation and submission

After selecting and freezing a configuration, refit preprocessing and the model on all 80% development data. Apply that fitted pipeline and its selected threshold to the reserved 20% local test set for the final local evaluation.

After recording the final local result, refit the selected pipeline on all labeled data before generating competition benchmark predictions through `run.py`.

The final training and evaluation workflow uses NumPy and the Python standard library, with permitted visualization libraries for plots. The existing Pandas/scikit-learn experiments remain temporary reference work until the NumPy replacement is verified; their removal is scheduled for the end of the model comparison work.
