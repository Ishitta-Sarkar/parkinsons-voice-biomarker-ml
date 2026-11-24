"""Subject-grouped evaluation of voice-biomarker classifiers.

The question is whether sustained-phonation features separate people with
Parkinson's disease from healthy controls when every recording from a person
is kept on one side of each split. A random split of recordings does not
answer that question: the same voice can land in both the training set and
the test set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep numeric libraries on one thread before they are imported. Combined with
# the fixed seeds below, this is what makes a second run reproduce the tables.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# `python src/train.py` puts src/ on sys.path, not the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json
import platform
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import (
    FIGURES_DIR,
    RESULTS_DIR,
    UCI_PARKINSONS_SHA256,
    algebraic_duplicate_gap,
    cohort_summary,
    design_matrix,
    load_parkinsons,
)

RANDOM_STATE = 42
N_SPLITS = 5
N_ESTIMATORS = 500
LEGACY_N_ESTIMATORS = 100
HOLDOUT_TEST_SIZE = 0.2
PERMUTATION_REPEATS = 20
PRIMARY_SCORING = "roc_auc"

MODEL_ORDER = ("majority_baseline", "logistic_regression", "random_forest")
METRIC_NAMES = ("accuracy", "balanced_accuracy", "roc_auc", "f1")

GROUPED_PROTOCOL = "subject_grouped_stratified_5fold"
RECORD_CV_PROTOCOL = "recording_stratified_5fold"
HOLDOUT_PROTOCOL = "recording_stratified_holdout_20"
LEGACY_HOLDOUT_PROTOCOL = "legacy_recording_stratified_holdout_20"


def grouped_splitter() -> StratifiedGroupKFold:
    """Five stratified folds that never place one subject on both sides."""
    return StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def recording_splitter() -> StratifiedKFold:
    """Five stratified folds of recordings. Subjects are free to leak across."""
    return StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def make_baseline() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def make_logistic_regression() -> Pipeline:
    """L2 logistic regression on features standardised inside each training fold."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            l1_ratio=0.0,
            solver="lbfgs",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
    )


def make_random_forest(n_estimators: int = N_ESTIMATORS) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def grouped_models() -> dict[str, DummyClassifier | Pipeline | RandomForestClassifier]:
    """The three models fixed before looking at test folds."""
    return {
        "majority_baseline": make_baseline(),
        "logistic_regression": make_logistic_regression(),
        "random_forest": make_random_forest(),
    }


def subject_leakage(
    groups: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, float | int]:
    """How many test recordings belong to a subject who is also in training."""
    train_subjects = set(groups.iloc[train_idx])
    test_subjects = set(groups.iloc[test_idx])
    both = train_subjects & test_subjects
    seen = groups.iloc[test_idx].isin(train_subjects)
    return {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_train_subjects": int(len(train_subjects)),
        "n_test_subjects": int(len(test_subjects)),
        "n_subjects_in_both": int(len(both)),
        "fraction_test_recordings_seen_in_train": float(seen.mean()),
    }


def _positive_scores(estimator, features: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(estimator, "predict_proba"):
        return None
    probabilities = estimator.predict_proba(features)
    classes = list(estimator.classes_)
    if 1 not in classes:
        return None
    return probabilities[:, classes.index(1)]


def fold_scores(estimator, features: pd.DataFrame, y_true: pd.Series) -> dict[str, float]:
    """Accuracy, balanced accuracy, ROC-AUC and F1 on one test fold.

    The positive class is status 1 (Parkinson's disease). ROC-AUC is undefined
    when the predicted score does not vary, which is the case for a classifier
    that always emits the majority class.
    """
    predicted = estimator.predict(features)
    scores = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "f1": float(f1_score(y_true, predicted, pos_label=1, zero_division=0)),
        "roc_auc": float("nan"),
    }
    positive = _positive_scores(estimator, features)
    if positive is None or len(np.unique(y_true)) < 2 or len(np.unique(positive)) < 2:
        return scores
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        try:
            scores["roc_auc"] = float(roc_auc_score(y_true, positive))
        except ValueError:
            scores["roc_auc"] = float("nan")
    return scores


def _summarise(fold_frame: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample standard deviation (ddof=1) of each metric across folds."""
    rows = []
    for model, block in fold_frame.groupby("model", sort=False):
        row: dict[str, str | float | int] = {
            "model": str(model),
            "n_folds": int(block["fold"].nunique()),
        }
        for metric in METRIC_NAMES:
            values = block[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            row[f"{metric}_std"] = (
                float(np.std(finite, ddof=1)) if finite.size >= 2 else float("nan")
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["model"] = pd.Categorical(summary["model"], categories=MODEL_ORDER, ordered=True)
    return summary.sort_values("model").reset_index(drop=True)


def _evaluate(
    models: dict,
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    splitter,
    *,
    pass_groups: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict] = []
    splits = (
        splitter.split(features, labels, groups)
        if pass_groups
        else splitter.split(features, labels)
    )
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        leakage = subject_leakage(groups, train_idx, test_idx)
        y_test = labels.iloc[test_idx]
        for name in MODEL_ORDER:
            estimator = clone(models[name])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                estimator.fit(features.iloc[train_idx], labels.iloc[train_idx])
                scores = fold_scores(estimator, features.iloc[test_idx], y_test)
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "n_test_pd": int((y_test == 1).sum()),
                    "n_test_healthy": int((y_test == 0).sum()),
                    **leakage,
                    **scores,
                }
            )
    fold_frame = pd.DataFrame(fold_rows)
    return _summarise(fold_frame), fold_frame


def evaluate_grouped(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate the three models without sharing a subject across a fold."""
    summary, folds = _evaluate(
        grouped_models(),
        features,
        labels,
        groups,
        grouped_splitter(),
        pass_groups=True,
    )
    leaked = int(folds["n_subjects_in_both"].max())
    if leaked != 0:
        raise RuntimeError(f"Grouped folds share subjects (max overlap {leaked}).")
    return summary, folds


def evaluate_record_level(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same models on a recording-level stratified split, which can leak subjects."""
    return _evaluate(
        grouped_models(),
        features,
        labels,
        groups,
        recording_splitter(),
        pass_groups=False,
    )


def recording_holdout_indices(
    labels: pd.Series,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Positions for the stratified 80/20 recording split used as a comparison."""
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=HOLDOUT_TEST_SIZE,
        random_state=random_state,
        stratify=labels,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def evaluate_holdout(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    *,
    n_estimators: int,
    protocol: str,
) -> dict[str, float | int | str]:
    """One stratified 80/20 split of recordings, for comparison with grouped CV."""
    train_idx, test_idx = recording_holdout_indices(labels, RANDOM_STATE)
    estimator = make_random_forest(n_estimators)
    estimator.fit(features.iloc[train_idx], labels.iloc[train_idx])
    scores = fold_scores(estimator, features.iloc[test_idx], labels.iloc[test_idx])
    return {
        "protocol": protocol,
        "model": "random_forest",
        "n_estimators": int(n_estimators),
        **subject_leakage(groups, train_idx, test_idx),
        **scores,
    }


def permutation_importance_grouped(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    *,
    n_repeats: int = PERMUTATION_REPEATS,
) -> pd.DataFrame:
    """Permutation importance on each grouped test fold, then the mean across folds.

    Importance is the drop in ROC-AUC when a column is shuffled in the test
    fold. Training rows are not permuted, and no subject is in both.
    """
    models = {
        "logistic_regression": make_logistic_regression(),
        "random_forest": make_random_forest(),
    }
    rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(
        grouped_splitter().split(features, labels, groups),
        start=1,
    ):
        y_test = labels.iloc[test_idx]
        if y_test.nunique() < 2:
            raise RuntimeError(f"Fold {fold} has one class; importance is undefined.")
        for name, estimator in models.items():
            fitted = clone(estimator)
            fitted.fit(features.iloc[train_idx], labels.iloc[train_idx])
            result = permutation_importance(
                fitted,
                features.iloc[test_idx],
                y_test,
                scoring=PRIMARY_SCORING,
                n_repeats=n_repeats,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            for feature, importance in zip(features.columns, result.importances_mean, strict=True):
                rows.append(
                    {
                        "model": name,
                        "fold": fold,
                        "feature": feature,
                        "importance": float(importance),
                    }
                )
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby(["model", "feature"], sort=False)["importance"]
        .agg(importance_mean="mean", importance_std="std", n_folds="count")
        .reset_index()
    )
    return summary


def format_mean_std(mean: float, std: float) -> str:
    """Format a fold mean and its sample SD, or mark an undefined score."""
    if not np.isfinite(mean):
        return "undefined"
    if not np.isfinite(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def summary_lines(summary: pd.DataFrame) -> list[str]:
    lines = []
    for row in summary.itertuples(index=False):
        bits = [str(row.model)]
        for metric in METRIC_NAMES:
            bits.append(
                format_mean_std(getattr(row, f"{metric}_mean"), getattr(row, f"{metric}_std"))
            )
        lines.append(" | ".join(bits))
    return lines


def _package_versions() -> dict[str, str]:
    import matplotlib
    import numpy
    import pandas
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_analysis(frame: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Fit, score, and write the tables and figures used in the README."""
    # Importing pyplot only when an analysis is requested keeps `import src.train`
    # from selecting a GUI backend during tests.
    os.environ.setdefault("MPLBACKEND", "Agg")
    from src.visualization import (
        save_figure,
        write_correlation_figure,
        write_cv_figure,
        write_importance_figure,
        write_sample_distribution,
        write_subject_median_figure,
    )

    data = load_parkinsons() if frame is None else frame
    features, labels, groups = design_matrix(data)
    cohort = cohort_summary(data)
    duplicates = algebraic_duplicate_gap(data)

    grouped_summary, grouped_folds = evaluate_grouped(features, labels, groups)
    record_summary, record_folds = evaluate_record_level(features, labels, groups)
    holdout = pd.DataFrame(
        [
            evaluate_holdout(
                features,
                labels,
                groups,
                n_estimators=N_ESTIMATORS,
                protocol=HOLDOUT_PROTOCOL,
            ),
            evaluate_holdout(
                features,
                labels,
                groups,
                n_estimators=LEGACY_N_ESTIMATORS,
                protocol=LEGACY_HOLDOUT_PROTOCOL,
            ),
        ]
    )
    importance = permutation_importance_grouped(features, labels, groups)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _write_table(grouped_summary, RESULTS_DIR / "model_metrics.csv")
    _write_table(grouped_folds, RESULTS_DIR / "fold_metrics.csv")
    _write_table(record_summary, RESULTS_DIR / "record_level_cv.csv")
    _write_table(record_folds, RESULTS_DIR / "record_level_folds.csv")
    _write_table(holdout, RESULTS_DIR / "holdout_comparison.csv")
    _write_table(importance, RESULTS_DIR / "feature_importance.csv")

    save_figure(write_sample_distribution(data), FIGURES_DIR / "sample_distribution.png")
    save_figure(write_subject_median_figure(data), FIGURES_DIR / "subject_feature_medians.png")
    save_figure(write_correlation_figure(data), FIGURES_DIR / "feature_correlation.png")
    save_figure(write_cv_figure(grouped_summary), FIGURES_DIR / "grouped_cv_metrics.png")
    save_figure(write_importance_figure(importance), FIGURES_DIR / "feature_importance.png")

    metadata = {
        "data_sha256": UCI_PARKINSONS_SHA256,
        "positive_class": 1,
        "positive_class_meaning": "Parkinson's disease",
        "primary_metric": PRIMARY_SCORING,
        "std_definition": "sample standard deviation of fold scores (ddof=1)",
        "grouped_cv": {
            "splitter": "StratifiedGroupKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "group": "subject id parsed from name",
        },
        "models": {
            "majority_baseline": "DummyClassifier(strategy='most_frequent')",
            "logistic_regression": (
                "StandardScaler + LogisticRegression(C=1, l1_ratio=0, "
                "solver='lbfgs', max_iter=5000)"
            ),
            "random_forest": f"RandomForestClassifier(n_estimators={N_ESTIMATORS})",
        },
        "permutation_importance": {
            "scoring": PRIMARY_SCORING,
            "n_repeats": PERMUTATION_REPEATS,
            "random_state": RANDOM_STATE,
            "where": "each grouped test fold, then mean and sample SD across folds",
        },
        "cohort": cohort,
        "algebraic_duplicate_gap": duplicates,
        "versions": _package_versions(),
        "package_version": "1.0.0",
    }
    (RESULTS_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("Cohort")
    for key, value in cohort.items():
        print(f"  {key}: {value}")
    print("Algebraic duplicate gaps")
    for key, value in duplicates.items():
        print(f"  {key}: {value:.3e}")
    print("Subject-grouped 5-fold CV (mean ± sample SD)")
    for line in summary_lines(grouped_summary):
        print(f"  {line}")
    print("Recording-level 5-fold CV (subjects may sit in both sides of a fold)")
    for line in summary_lines(record_summary):
        print(f"  {line}")
    print("Recording-level 80/20 holdout")
    for row in holdout.itertuples(index=False):
        print(
            f"  {row.protocol}: n_estimators={row.n_estimators} "
            f"accuracy={row.accuracy:.3f} balanced_accuracy={row.balanced_accuracy:.3f} "
            f"roc_auc={row.roc_auc:.3f} f1={row.f1:.3f} "
            f"subjects_in_both={row.n_subjects_in_both} "
            f"fraction_test_recordings_seen_in_train="
            f"{row.fraction_test_recordings_seen_in_train:.3f}"
        )
    return {
        "grouped_summary": grouped_summary,
        "grouped_folds": grouped_folds,
        "record_summary": record_summary,
        "record_folds": record_folds,
        "holdout": holdout,
        "importance": importance,
    }


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
