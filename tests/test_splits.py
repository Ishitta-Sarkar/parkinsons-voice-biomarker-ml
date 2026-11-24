"""Subject leakage: grouped folds must be disjoint, a recording split is not."""

from __future__ import annotations

import numpy as np

from src.data_loader import design_matrix, load_parkinsons
from src.train import (
    grouped_splitter,
    recording_holdout_indices,
    recording_splitter,
    subject_leakage,
)


def _xy():
    return design_matrix(load_parkinsons())


def test_grouped_folds_do_not_share_subjects():
    features, labels, groups = _xy()
    seen_test: list[str] = []
    for train_idx, test_idx in grouped_splitter().split(features, labels, groups):
        info = subject_leakage(groups, train_idx, test_idx)
        assert info["n_subjects_in_both"] == 0
        assert info["fraction_test_recordings_seen_in_train"] == 0.0
        assert labels.iloc[test_idx].nunique() == 2
        seen_test.extend(groups.iloc[test_idx].unique())
    assert len(seen_test) == len(set(seen_test)) == groups.nunique()


def test_recording_holdout_puts_subjects_on_both_sides():
    _features, labels, groups = _xy()
    train_idx, test_idx = recording_holdout_indices(labels)
    info = subject_leakage(groups, train_idx, test_idx)
    assert info["n_subjects_in_both"] > 0
    assert info["fraction_test_recordings_seen_in_train"] > 0
    assert not np.isclose(info["fraction_test_recordings_seen_in_train"], 0.0)


def test_recording_level_cv_folds_share_subjects():
    features, labels, groups = _xy()
    overlaps = []
    for train_idx, test_idx in recording_splitter().split(features, labels):
        info = subject_leakage(groups, train_idx, test_idx)
        overlaps.append(info["n_subjects_in_both"])
    assert all(count > 0 for count in overlaps)
