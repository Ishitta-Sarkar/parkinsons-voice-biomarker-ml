# UCI Parkinsons voice recordings

Biomedical voice measurements from sustained phonations, donated to the UCI Machine Learning Repository by Max Little.

| | |
|---|---|
| Dataset | Parkinsons |
| DOI | [10.24432/C59C74](https://doi.org/10.24432/C59C74) |
| Landing page | https://archive.ics.uci.edu/dataset/174/parkinsons |
| File URL | https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data |
| Local file | `parkinsons.data` |
| SHA-256 | `a36116c2deac07895b79ce97a2f99a1b3720ac4892995ff5c3b9ccb397ed2bc5` |
| Licence | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Rows | 195 recordings |
| Columns | 24 (`name`, 22 acoustic features, `status`) |

The file in this directory is byte-identical to the UCI `parkinsons.data` download. It is small enough to keep in the repository so a fresh clone runs without a network fetch. Recompute the checksum with `sha256sum data/raw/parkinsons.data`.

## What a row is

Each row is one sustained-phonation recording. `name` looks like `phon_R01_S01_1`: subject `S01`, recording 1. About six or seven recordings belong to each subject. `status` is 1 for Parkinson's disease and 0 for a healthy control.

The UCI description says the table holds 31 people, 23 with Parkinson's disease. Parsing `name` in this file yields **32 subject ids, 24 labelled Parkinson's disease and 8 healthy** (147 and 48 recordings). No subject id has both labels. Cross-validation in this project groups on those 32 ids. The ids are not collapsed to force the catalogue count of 31.

## Columns

| Column | Role |
|---|---|
| `name` | Recording id. The subject token is the `S##` field. |
| `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)` | Average, maximum and minimum vocal fundamental frequency. |
| `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP` | Variation in fundamental frequency. `Jitter:DDP` equals `3 × MDVP:RAP` up to rounding. |
| `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA` | Variation in amplitude. `Shimmer:DDA` equals `3 × Shimmer:APQ3` up to rounding. |
| `NHR`, `HNR` | Ratio of noise to tonal components. |
| `status` | 1 = Parkinson's disease, 0 = healthy. This is the classification target. |
| `RPDE`, `D2` | Nonlinear dynamical complexity measures. |
| `DFA` | Signal fractal scaling exponent. |
| `spread1`, `spread2`, `PPE` | Nonlinear measures of fundamental frequency variation. |

There are no missing values.

## Please cite the data

Little, M. (2007). Parkinsons [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59C74

The measures are introduced and applied in:

- Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection. *BioMedical Engineering OnLine*. 2007;6:23. https://doi.org/10.1186/1475-925X-6-23
- Little MA, McSharry PE, Hunter EJ, Spielman J, Ramig LO. Suitability of Dysphonia Measurements for Telemonitoring of Parkinson's Disease. *IEEE Transactions on Biomedical Engineering*. 2009;56(4):1015–1022. https://doi.org/10.1109/TBME.2008.2005954
