"""Figure constructors return the panels the README points at."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_parkinsons
from src.visualization import (
    write_correlation_figure,
    write_cv_figure,
    write_importance_figure,
    write_sample_distribution,
    write_subject_median_figure,
)


def test_descriptive_figures_build():
    frame = load_parkinsons()
    distribution = write_sample_distribution(frame)
    medians = write_subject_median_figure(frame)
    correlation = write_correlation_figure(frame)
    assert len(distribution.axes) == 2
    assert len(medians.axes) == 6
    assert len(correlation.axes) == 2  # heatmap plus colorbar axis
    plt.close("all")


def test_result_figures_build_from_a_small_table():
    summary = pd.DataFrame(
        {
            "model": ["majority_baseline", "logistic_regression", "random_forest"],
            "accuracy_mean": [0.75, 0.80, 0.78],
            "accuracy_std": [0.05, 0.04, 0.06],
            "balanced_accuracy_mean": [0.5, 0.70, 0.68],
            "balanced_accuracy_std": [0.0, 0.05, 0.07],
            "roc_auc_mean": [float("nan"), 0.82, 0.80],
            "roc_auc_std": [float("nan"), 0.04, 0.05],
            "f1_mean": [0.86, 0.85, 0.84],
            "f1_std": [0.03, 0.04, 0.05],
        }
    )
    figure = write_cv_figure(summary)
    assert len(figure.axes) == 4
    importance = pd.DataFrame(
        {
            "model": [
                "logistic_regression",
                "logistic_regression",
                "random_forest",
                "random_forest",
            ],
            "feature": ["PPE", "HNR", "PPE", "HNR"],
            "importance_mean": [0.02, 0.01, 0.03, 0.0],
            "importance_std": [0.01, 0.01, 0.02, 0.01],
        }
    )
    importance_figure = write_importance_figure(importance)
    assert len(importance_figure.axes) == 2
    plt.close("all")
