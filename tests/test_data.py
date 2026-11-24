"""Properties of the committed UCI Parkinsons file and the subject parser."""

from __future__ import annotations

import hashlib

import pytest

from src.data_loader import (
    DEFAULT_DATA_PATH,
    ROOT,
    UCI_PARKINSONS_SHA256,
    algebraic_duplicate_gap,
    cohort_summary,
    design_matrix,
    load_parkinsons,
    subject_id,
)


def test_file_matches_documented_uci_checksum():
    digest = hashlib.sha256(DEFAULT_DATA_PATH.read_bytes()).hexdigest()
    assert digest == UCI_PARKINSONS_SHA256
    assert UCI_PARKINSONS_SHA256 in (ROOT / "data" / "README.md").read_text()


def test_table_shape_and_cohort_counts():
    frame = load_parkinsons()
    assert frame.shape == (195, 24)
    summary = cohort_summary(frame)
    assert summary["n_recordings"] == 195
    assert summary["n_columns"] == 24
    assert summary["n_features"] == 22
    # The UCI prose says 31 people and 23 with Parkinson's disease. The name
    # column of this file encodes 32 subject tokens, 24 of them labelled PD.
    assert summary["n_subjects"] == 32
    assert summary["n_pd_subjects"] == 24
    assert summary["n_healthy_subjects"] == 8
    assert summary["n_pd_recordings"] == 147
    assert summary["n_healthy_recordings"] == 48
    assert summary["recordings_per_subject_min"] == 6
    assert summary["recordings_per_subject_max"] == 7


def test_subject_parser_examples():
    assert subject_id("phon_R01_S01_1") == "S01"
    assert subject_id("phon_R01_S50_6") == "S50"
    with pytest.raises(ValueError):
        subject_id("subject_01")


def test_every_recording_name_parses_and_features_are_numeric():
    frame = load_parkinsons()
    features, labels, groups = design_matrix(frame)
    assert len(features) == len(labels) == len(groups) == 195
    assert features.shape[1] == 22
    assert set(labels.unique()) == {0, 1}
    assert groups.nunique() == 32
    assert labels.groupby(groups).nunique().max() == 1
    assert not features.isna().any().any()


def test_jitter_and_shimmer_pairs_are_algebraic_duplicates():
    gaps = algebraic_duplicate_gap(load_parkinsons())
    assert gaps["jitter_ddp_minus_3rap_max_abs"] < 1e-4
    assert gaps["shimmer_dda_minus_3apq3_max_abs"] < 1e-4


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_parkinsons(tmp_path / "absent.data")
