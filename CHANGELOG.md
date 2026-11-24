# Changelog

All notable changes to this project are documented here.
The format is based on Keep a Changelog, and the version follows Semantic Versioning.

## [1.0.0] - 2026-11-24

### Changed

- Evaluate classifiers with subject-grouped stratified cross-validation, so no person contributes recordings to both sides of a fold.
- Report a majority-class baseline beside L2 logistic regression and a random forest. The metrics are accuracy, balanced accuracy, ROC-AUC and F1, as the mean and sample standard deviation across five folds.
- Move the unchanged UCI file to `data/raw/parkinsons.data` and the analysis code under `src/`.

### Added

- Permutation importance computed on the grouped test folds, cohort counts, and a check that two feature pairs are algebraic duplicates.
- Figures for class counts, subject-level feature medians, feature correlation, grouped metrics, and permutation importance.
- Pinned `requirements.txt`, pytest, GitHub Actions for lint and tests, `CITATION.cff`, and this changelog.

## [0.1.0] - 2026-06-25

### Added

- Initial data loading, a recording-level train/test split, and a sample-distribution figure.
- Metrics from that split are superseded by 1.0.0. A split of recordings does not estimate performance on people the model has not seen.
