"""Plot cross-validation model comparisons with Matplotlib."""

import matplotlib.pyplot as plt
import numpy as np


def plot_model_comparison(results, ax=None):
    """Plot mean positive-class F1 with one fold standard deviation as error bars.

    Accept ranked results from ModelComparison.run or a saved report's results
    dictionary. Keep their order, with the first model at the top. Read only the
    F1 summaries; no training or score recalculation is needed. Return (fig, ax)
    for notebook customization, display, or saving. An optional ax reuses an
    existing Matplotlib axes; otherwise create a figure sized for the model list.
    Tuned results are labeled as development threshold-selection scores.
    """
    if not results:
        raise ValueError("At least one model result is required for plotting.")

    names = list(results)
    means = np.array([result["summary"]["f1"]["mean"] for result in results.values()])
    stds = np.array([result["summary"]["f1"]["std"] for result in results.values()])
    if ax is None:
        _, ax = plt.subplots(
            figsize=(9, max(3, 0.5 * len(names) + 1.8)), constrained_layout=True
        )

    positions = np.arange(len(names))
    ax.barh(positions, means, xerr=stds, capsize=4, color="#4c78a8", ecolor="#28323c")
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    ax.set_ylim(len(names) - 0.5, -0.5)
    # Include the full error bars: mean ± std can extend beyond the 0–1 scores.
    ax.set_xlim(min(0.0, min(means - stds)) - 0.02, max(1.0, max(means + stds)) + 0.02)
    ax.set_xlabel("Positive-class F1")
    ax.set_title("Validation F1: mean ± one fold standard deviation")
    if any("threshold_tuning" in result for result in results.values()):
        ax.set_title(
            "Development F1 after threshold selection\n"
            "Mean ± one fold standard deviation"
        )
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.25)
    return ax.figure, ax
