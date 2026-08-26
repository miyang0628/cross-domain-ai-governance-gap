# Cross-Domain AI Governance Gap Analysis

Replication code for the study *Cross-Domain AI Governance Gap Analysis: Comparing
Fairness Gaps between Credit Scoring and Insurance Underwriting under High-Impact AI
Standards.*

This repository accompanies an anonymous submission and is provided for peer review.
Author and affiliation information has been removed for double-blind review.

## Overview

The study introduces the **Governance Gap Index (GGI)**, a measure that quantifies the
distance between a model's realized behavior and the requirements imposed by high-impact
AI regulations. The GGI is computed for each (dataset x regulation) pair using a
threshold-distance backbone with normalized per-component gaps:

- For component *k* under regulation *r*, with required threshold `tau[r,k]` and a model
  score `s_k` oriented so that higher is better:
  `g[r,k] = max(0, tau[r,k] - s_k) / tau[r,k]`
- The per-regulation index aggregates the component gaps as a Euclidean distance:
  `GGI[r] = sqrt( sum_k g[r,k]^2 )` (lower is better; 0 = full compliance)

Regulation differences are encoded as a modeled strictness gradient across thresholds
(KR AI Basic Act < EU AI Act < NAIC); these thresholds are scenarios, not codified legal
constants.

Component scores: discriminative adequacy (from AUC), worst-case fairness (Disparate
Impact and Statistical Parity across all sensitive attributes), and explainability
(entropy-based SHAP concentration). A fourth framework axis, recourse feasibility, is
demonstrated on datasets with semantically interpretable features.

## Datasets

The datasets are **not redistributed** here. Obtain them from the original sources and
place the files in `data/` as described below.

| Domain    | Dataset                          | Source                    | Expected file(s) in `data/`                     |
|-----------|----------------------------------|---------------------------|-------------------------------------------------|
| Credit    | German Credit (Statlog)          | UCI ML Repository         | `german.data` (downloaded automatically)        |
| Credit    | Credit Card Approval Prediction  | Kaggle (dataset)          | `application_record.csv`, `credit_record.csv`   |
| Insurance | Medical Cost Personal            | Kaggle (dataset)          | `insurance.csv`                                  |
| Insurance | Porto Seguro Safe Driver         | Kaggle (competition)      | `porto_train.csv` (rename from `train.csv`)     |

Notes:
- `german.data` is fetched automatically from the UCI repository by the first notebook.
- Kaggle files require a Kaggle account. Competition data additionally requires accepting
  the competition rules before download.
- For the Credit Card Approval data, the binary target is derived from `credit_record.csv`
  using a 30-days-past-due default definition (a 60-days variant is also produced).
- Porto Seguro is stratified-subsampled to align sample sizes across domains.

## Repository structure

```
.
├── data/                 # place raw datasets here (not tracked)
│   └── processed/        # intermediate parquet files (generated)
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_fairness_metrics.ipynb
│   ├── 04_ggi.ipynb
│   ├── 05_sensitivity.ipynb
│   ├── 06_visualization.ipynb
│   └── 07_recourse.ipynb
└── results/
    ├── figures/          # generated figures (PNG + PDF, 600 dpi, grayscale)
    └── tables/           # generated result tables (CSV)
```

## Requirements

Python 3.10+ with:

```
numpy
pandas
scikit-learn
xgboost
lightgbm
catboost
shap
dice-ml
seaborn
matplotlib
pyarrow
xlrd>=2.0.1
kaggle
```

Install with:

```
pip install numpy pandas scikit-learn xgboost lightgbm catboost shap dice-ml \
            seaborn matplotlib pyarrow "xlrd>=2.0.1" kaggle
```

## Reproduction

Run the notebooks in order from the `notebooks/` directory:

1. `01_data_loading.ipynb` — load datasets, derive targets, write processed parquet files
2. `02_preprocessing.ipynb` — impute, encode, scale, split; subsample large datasets
3. `03_fairness_metrics.ipynb` — unified baseline and group-fairness metrics
4. `04_ggi.ipynb` — component scores and the Governance Gap Index
5. `05_sensitivity.ipynb` — robustness checks (thresholds, aggregation, bootstrap)
6. `06_visualization.ipynb` — publication figures
7. `07_recourse.ipynb` — recourse feasibility (supplementary axis)

A fixed random seed (42) is used throughout. Outputs are written to `results/`.

## Notes on interpretation

- Regulation thresholds represent a modeled strictness gradient, not exact legal
  requirements; results should be read as relative comparisons.
- Two of the four datasets (Medical Cost, Porto Seguro) yield low model discrimination;
  their fairness and gap estimates are indicative rather than confirmatory, and this is
  reflected in the bootstrap confidence intervals.
- Recourse feasibility is undefined for datasets with anonymized proxy features and is
  reported as such.

## License

Provided for peer review. A license will be added upon publication.
