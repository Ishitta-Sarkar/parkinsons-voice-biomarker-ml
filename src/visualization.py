"""Figures for the cohort, the grouped cross-validation, and feature importance."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.data_loader import cohort_summary, subject_feature_medians

STATUS_COLORS = {0: "#0072B2", 1: "#D55E00"}
STATUS_LABELS = {0: "Healthy", 1: "Parkinson's disease"}
MODEL_ORDER = ("majority_baseline", "logistic_regression", "random_forest")
MODEL_LABELS = {
    "majority_baseline": "Majority baseline",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
}
MODEL_COLORS = {
    "majority_baseline": "#6B6B6B",
    "logistic_regression": "#0072B2",
    "random_forest": "#009E73",
}
# One column from each pre-specified family. These were not chosen by ranking
# a fitted model: frequency, jitter, shimmer, noise-to-harmonics, and two
# nonlinear measures described with the UCI table.
FAMILY_FEATURES = (
    ("MDVP:Fo(Hz)", "Fundamental frequency (Hz)"),
    ("MDVP:Jitter(%)", "Jitter (%)"),
    ("MDVP:Shimmer", "Shimmer"),
    ("HNR", "Harmonics-to-noise ratio (dB)"),
    ("RPDE", "RPDE"),
    ("PPE", "Pitch period entropy"),
)
CV_METRICS = (
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("roc_auc", "ROC-AUC"),
    ("f1", "F1"),
)

_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.titlesize": 13,
}


def save_figure(fig: Figure, path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return destination


def _counts_panel(ax: Axes, counts: list[int], title: str, ylabel: str) -> None:
    labels = [STATUS_LABELS[0], STATUS_LABELS[1]]
    colors = [STATUS_COLORS[0], STATUS_COLORS[1]]
    bars = ax.bar(labels, counts, color=colors, width=0.62)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ceiling = max(counts) * 1.2 if counts else 1
    ax.set_ylim(0, ceiling)
    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )


def write_sample_distribution(frame: pd.DataFrame) -> Figure:
    """Recording counts beside subject counts. The two denominators differ."""
    summary = cohort_summary(frame)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0))
        _counts_panel(
            axes[0],
            [summary["n_healthy_recordings"], summary["n_pd_recordings"]],
            "Recordings",
            "Number of recordings",
        )
        _counts_panel(
            axes[1],
            [summary["n_healthy_subjects"], summary["n_pd_subjects"]],
            "Subjects",
            "Number of subjects",
        )
        fig.suptitle("UCI Parkinsons voice recordings", y=1.02)
        fig.tight_layout()
    return fig


def write_subject_median_figure(frame: pd.DataFrame) -> Figure:
    """Subject-level medians for one feature from each acoustic family."""
    medians = subject_feature_medians(frame)
    status = medians["status"].to_numpy()
    rng = np.random.default_rng(42)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.2))
        for ax, (column, label) in zip(axes.ravel(), FAMILY_FEATURES, strict=True):
            values = medians[column].to_numpy(dtype=float)
            groups = [values[status == 0], values[status == 1]]
            ax.boxplot(
                groups,
                positions=[0, 1],
                widths=0.55,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.4},
            )
            for position, status_value in enumerate((0, 1)):
                subset = values[status == status_value]
                jitter = rng.uniform(-0.08, 0.08, size=subset.size)
                ax.scatter(
                    np.full(subset.size, position) + jitter,
                    subset,
                    s=28,
                    color=STATUS_COLORS[status_value],
                    alpha=0.85,
                    zorder=3,
                    linewidths=0,
                )
            ax.set_xticks([0, 1], ["Healthy", "PD"])
            ax.set_title(label)
        fig.suptitle("Subject medians of representative voice features", y=1.01)
        fig.tight_layout()
    return fig


def write_correlation_figure(frame: pd.DataFrame) -> Figure:
    """Pearson correlations of subject-level median features."""
    medians = subject_feature_medians(frame).drop(columns=["status"])
    corr = medians.corr(method="pearson")
    display = corr.to_numpy(dtype=float).copy()
    display[np.triu_indices_from(display, k=1)] = np.nan
    cmap = plt.cm.RdBu_r.with_extremes(bad="#f7f7f7")
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(9.2, 7.6))
        image = ax.imshow(display, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks(range(len(corr.columns)), list(corr.columns), rotation=90, fontsize=7)
        ax.set_yticks(range(len(corr.columns)), list(corr.columns), fontsize=7)
        ax.set_title("Pearson correlation of subject-level feature medians")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
    return fig


def write_cv_figure(summary: pd.DataFrame) -> Figure:
    """Mean ± sample SD of each metric under subject-grouped cross-validation."""
    indexed = summary.set_index("model")
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.8), sharey=True)
        for ax, (metric, title) in zip(axes, CV_METRICS, strict=True):
            for position, model in enumerate(MODEL_ORDER):
                mean = float(indexed.loc[model, f"{metric}_mean"])
                std = float(indexed.loc[model, f"{metric}_std"])
                color = MODEL_COLORS[model]
                if not np.isfinite(mean):
                    ax.text(
                        position,
                        0.5,
                        "undefined",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                    )
                    continue
                yerr = std if np.isfinite(std) else None
                ax.errorbar(
                    position,
                    mean,
                    yerr=yerr,
                    fmt="o",
                    color=color,
                    markersize=7,
                    capsize=3,
                    linewidth=1.4,
                )
            ax.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)
            ax.set_xticks(range(len(MODEL_ORDER)), ["Baseline", "Logistic", "Forest"])
            ax.set_title(title)
            ax.set_ylim(0, 1.05)
            ax.yaxis.grid(True, linestyle=":", color="#cccccc")
            ax.set_axisbelow(True)
        axes[0].set_ylabel("Score")
        fig.suptitle(
            "Subject-grouped 5-fold cross-validation (mean ± sample SD)",
            y=1.05,
        )
        fig.tight_layout()
    return fig


def write_importance_figure(summary: pd.DataFrame) -> Figure:
    """Grouped-test permutation importance for the two fitted model families."""
    order = summary.groupby("feature")["importance_mean"].mean().sort_values(ascending=True).index
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 7.2), sharex=True)
        for ax, model in zip(axes, ("logistic_regression", "random_forest"), strict=True):
            block = summary.loc[summary["model"] == model].set_index("feature").loc[list(order)]
            positions = np.arange(len(block))
            ax.barh(
                positions,
                block["importance_mean"].to_numpy(dtype=float),
                xerr=block["importance_std"].to_numpy(dtype=float),
                color=MODEL_COLORS[model],
                height=0.72,
                capsize=2,
                error_kw={"linewidth": 0.8, "ecolor": "#333333"},
            )
            ax.axvline(0, color="#333333", linewidth=0.8)
            ax.set_yticks(positions, list(block.index), fontsize=8)
            ax.set_title(MODEL_LABELS[model])
            ax.set_xlabel("Drop in ROC-AUC")
        fig.suptitle(
            "Permutation importance on subject-grouped test folds (mean ± sample SD)",
            y=1.01,
        )
        fig.tight_layout()
    return fig
