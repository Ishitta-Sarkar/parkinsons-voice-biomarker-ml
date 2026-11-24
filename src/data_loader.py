"""Load the UCI Parkinsons file and identify the subject behind each recording."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "parkinsons.data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# SHA-256 of the file distributed by the UCI Machine Learning Repository at
# https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data
UCI_PARKINSONS_SHA256 = "a36116c2deac07895b79ce97a2f99a1b3720ac4892995ff5c3b9ccb397ed2bc5"

# phon_R01_S01_1 -> subject S01, recording 1. The visit token (R01) does not vary.
_RECORDING_NAME = re.compile(r"^phon_R01_(S\d+)_(\d+)$")

TARGET_COLUMN = "status"
ID_COLUMN = "name"


def subject_id(name: str) -> str:
    """Return the subject token from a UCI recording name.

    ``phon_R01_S01_1`` is recording 1 from subject ``S01``. The UCI catalogue
    text says the file holds 31 people. The ``name`` column encodes 32 subject
    tokens, and those tokens are the grouping unit used for cross-validation.
    """
    match = _RECORDING_NAME.fullmatch(str(name).strip())
    if match is None:
        raise ValueError(f"Expected a UCI recording name like 'phon_R01_S01_1', got {name!r}.")
    return match.group(1)


def load_parkinsons(path: Path | str | None = None) -> pd.DataFrame:
    """Read the UCI Parkinsons table.

    The committed file is the UCI ``parkinsons.data`` table, unchanged.
    """
    data_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not data_path.is_file():
        raise FileNotFoundError(
            f"UCI Parkinsons file not found at {data_path}. "
            "The path is data/raw/parkinsons.data inside the repository."
        )
    frame = pd.read_csv(data_path)
    missing = {ID_COLUMN, TARGET_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(f"Parkinsons file is missing columns: {sorted(missing)}")
    return frame.reset_index(drop=True)


def design_matrix(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return features, the status label, and the subject id for each recording.

    ``status`` is 1 for Parkinson's disease and 0 for a healthy control.
    Subject ids are parsed from ``name`` and are not used as features.
    """
    table = frame.reset_index(drop=True)
    groups = table[ID_COLUMN].map(subject_id).astype(str)
    y = table[TARGET_COLUMN].astype(int)
    unknown = set(y.unique()) - {0, 1}
    if unknown:
        raise ValueError(f"status must be 0 or 1, found {sorted(unknown)}")
    mixed = y.groupby(groups).nunique()
    if bool((mixed > 1).any()):
        raise ValueError("At least one subject has both status labels.")
    features = table.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    if bool(features.isna().any().any()):
        raise ValueError("Feature matrix contains missing values.")
    y.index = features.index
    groups.index = features.index
    return features, y, groups


def cohort_summary(frame: pd.DataFrame) -> dict[str, int]:
    """Counts of recordings and subjects, overall and by status."""
    _features, y, groups = design_matrix(frame)
    subject_status = y.groupby(groups).first()
    sizes = groups.value_counts()
    return {
        "n_recordings": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "n_features": int(frame.shape[1] - 2),
        "n_subjects": int(subject_status.shape[0]),
        "n_pd_subjects": int((subject_status == 1).sum()),
        "n_healthy_subjects": int((subject_status == 0).sum()),
        "n_pd_recordings": int((y == 1).sum()),
        "n_healthy_recordings": int((y == 0).sum()),
        "recordings_per_subject_min": int(sizes.min()),
        "recordings_per_subject_max": int(sizes.max()),
    }


def subject_feature_medians(frame: pd.DataFrame) -> pd.DataFrame:
    """Median of each acoustic feature within a subject, plus that subject's status.

    Recordings from one person are repeats, so subject medians are the unit
    used for the exploratory plots.
    """
    features, y, groups = design_matrix(frame)
    medians = features.groupby(groups).median().sort_index()
    medians.insert(0, TARGET_COLUMN, y.groupby(groups).first().astype(int))
    medians.index.name = "subject_id"
    return medians


def algebraic_duplicate_gap(frame: pd.DataFrame) -> dict[str, float]:
    """How far two feature pairs sit from an exact linear identity.

    In this table ``Jitter:DDP`` is three times ``MDVP:RAP``, and
    ``Shimmer:DDA`` is three times ``Shimmer:APQ3``, up to rounding.
    """
    rap = frame["MDVP:RAP"].to_numpy(dtype=float)
    ddp = frame["Jitter:DDP"].to_numpy(dtype=float)
    apq3 = frame["Shimmer:APQ3"].to_numpy(dtype=float)
    dda = frame["Shimmer:DDA"].to_numpy(dtype=float)
    return {
        "jitter_ddp_minus_3rap_max_abs": float(np.max(np.abs(ddp - 3.0 * rap))),
        "shimmer_dda_minus_3apq3_max_abs": float(np.max(np.abs(dda - 3.0 * apq3))),
    }
