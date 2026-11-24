"""Committed result tables must match a fresh run of the same splits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import RESULTS_DIR, design_matrix, load_parkinsons
from src.train import (
    MODEL_ORDER,
    evaluate_grouped,
    format_mean_std,
)


def test_grouped_metrics_file_matches_a_fresh_run():
    features, labels, groups = design_matrix(load_parkinsons())
    fresh, folds = evaluate_grouped(features, labels, groups)
    saved = pd.read_csv(RESULTS_DIR / "model_metrics.csv")
    assert list(saved["model"]) == list(MODEL_ORDER)
    assert saved["n_folds"].eq(5).all()
    for metric in ("accuracy", "balanced_accuracy", "roc_auc", "f1"):
        for model in MODEL_ORDER:
            fresh_row = fresh.loc[fresh["model"] == model].iloc[0]
            saved_row = saved.loc[saved["model"] == model].iloc[0]
            for suffix in ("mean", "std"):
                column = f"{metric}_{suffix}"
                got = float(fresh_row[column])
                stored = float(saved_row[column])
                if not np.isfinite(got):
                    assert not np.isfinite(stored)
                else:
                    assert stored == pytest_approx(got)
    # A constant majority prediction has balanced accuracy 0.5 when both
    # classes are present, and its ROC-AUC is undefined because the score
    # does not vary. Accuracy stays near the Parkinson's-recording prevalence.
    baseline = saved.loc[saved["model"] == "majority_baseline"].iloc[0]
    assert baseline["balanced_accuracy_mean"] == pytest_approx(0.5)
    assert not np.isfinite(baseline["roc_auc_mean"])
    assert baseline["accuracy_mean"] > 0.7
    assert int(folds["n_subjects_in_both"].max()) == 0


def test_reported_precision_is_three_decimals_of_the_table():
    saved = pd.read_csv(RESULTS_DIR / "model_metrics.csv")
    forest = saved.loc[saved["model"] == "random_forest"].iloc[0]
    rendered = format_mean_std(forest["accuracy_mean"], forest["accuracy_std"])
    readme = (RESULTS_DIR.parents[0] / "README.md").read_text()
    assert rendered in readme


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, abs=1e-8, rel=0)
