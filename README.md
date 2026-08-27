# Cross-Domain AI Governance Gap Analysis

Reproducibility repository for the paper **“Cross-Domain AI Governance Gap
Analysis: Comparing Fairness Gaps between Credit Scoring and Insurance
Underwriting under High-Impact AI Standards.”**

> **Status:** anonymized for peer review.
> Anonymous mirror: <https://anonymous.4open.science/r/cross-domain-ai-governance-gap-3BC4>

---

## Overview

Credit scoring and insurance underwriting are both designated as
*high-impact AI* under emerging AI-governance regimes, yet they differ
sharply in regulatory environment, data structure, and bias patterns. This
project asks a measurement question that normative governance prescriptions
leave open: **when an identical high-impact AI standard is applied, does the
governance gap — the distance between regulatory requirements and actual
model behavior — differ across domains, and across regulatory regimes?**

To answer it, the repository implements the **Governance Gap Index (GGI)**, a
measurement instrument that quantifies how far a deployed model falls short
of a regulation’s thresholds along four axes, and applies it to four public
datasets under three regulatory threshold scenarios.

The analysis is **illustrative rather than confirmatory**: two of the four
datasets have weak discriminative power, and the regulatory thresholds are
strictness scenarios rather than enacted constants. Results should be read
as demonstrating what the GGI can reveal, not as definitive audits.

---

## The Governance Gap Index (GGI)

Each of four axes is mapped to a component score `s_k ∈ [0, 1]` (higher is
better):

| Axis | Score | Definition |
|------|-------|------------|
| Performance | `s_PERF` | `max(0, (AUC − 0.5) / 0.5)` |
| Disparate impact (worst-case) | `s_DI` | `min(1, min_a DI_a)` |
| Statistical parity (worst-case) | `s_SP` | `1 − max_a |SPD_a|` |
| Explainability | `s_EXPL` | `1 − normalized SHAP entropy` |

Given a regulation `r` with per-axis thresholds `τ[r,k]`, the per-axis gap and
the aggregated index are:

```
g[r,k] = max(0, τ[r,k] − s_k) / τ[r,k]        # normalized shortfall, in [0, 1]
GGI[r] = sqrt( Σ_k g[r,k]^2 )                 # Euclidean; lower is better, 0 = full compliance
```

The fairness axes use a **worst-case** aggregation over sensitive attributes
(minimum DI, maximum |SPD|), which is what lets the index capture hidden
disparities a single-proxy audit would miss.

### Regulatory threshold scenarios

Strictness scenarios, **not** legal constants — used for relative comparison
only. All three regimes are applied to all four datasets (applying an
insurance-oriented standard to credit data, or vice versa, is a
counterfactual comparison).

| Regulation | τ_PERF | τ_DI | τ_SP | τ_EXPL |
|------------|:------:|:----:|:----:|:------:|
| KR AI Basic Act | 0.30 | 0.80 | 0.90 | 0.50 |
| EU AI Act | 0.40 | 0.85 | 0.92 | 0.60 |
| NAIC | 0.40 | 0.90 | 0.95 | 0.55 |

---

## Datasets

Data are **not redistributed**. Download each from its original source and
place the files in `data/` as noted below.

| Domain | Dataset | Source | Sensitive attribute(s) | File(s) in `data/` |
|--------|---------|--------|------------------------|--------------------|
| Credit | German Credit (Statlog) | UCI (auto-downloaded) | sex, age group | `german.data` |
| Credit | Credit Card Approval Prediction | Kaggle | sex | `application_record.csv`, `credit_record.csv` |
| Insurance | Medical Cost Personal | Kaggle | sex, age group, smoker | `insurance.csv` |
| Insurance | Porto Seguro Safe Driver | Kaggle | anonymized proxies (`ps_ind_*`) | `porto_train.csv` |

Notes:
- **German Credit** is downloaded automatically from UCI by
  `01_data_loading.ipynb` if missing. The sex attribute is mapped from the
  combined `personal_status_sex` variable; this mapping has a known coding
  limitation (Grömping, 2019), and the corrected *South German Credit*
  release is not used.
- **Credit Card Approval** labels are derived from `credit_record.csv`; the
  primary label marks an account high-risk if it ever reaches 30 days past
  due (`target_30dpd`).
- **Medical Cost** is binarized by labeling the top 25% of `charges` as
  high-risk.
- **Porto Seguro** (`train.csv`, unzip and rename to `porto_train.csv`) is
  stratified-subsampled to 40,000 rows to align sample sizes across domains.

Kaggle sources:
- Credit Card Approval — `rikdifos/credit-card-approval-prediction`
- Medical Cost — `mirichoi0218/insurance`
- Porto Seguro — `competitions/porto-seguro-safe-driver-prediction`

---

## Repository structure

```
.
├── data/                     # place raw inputs here (not tracked); processed/ is generated
│   └── processed/            # parquet splits + preprocess_meta.json (created by nbs 01–02)
├── notebooks/
│   ├── 01_data_loading.ipynb      # load 4 datasets, build targets, cross-domain summary
│   ├── 02_preprocessing.ipynb     # shared impute/scale/encode, subsample, stratified split
│   ├── 03_fairness_metrics.ipynb  # unified XGBoost baseline + group-fairness metrics
│   ├── 04_ggi.ipynb               # component scores + 4-axis GGI matrix
│   ├── 05_sensitivity.ipynb       # Studies A–D (threshold / aggregation / expl. / bootstrap)
│   ├── 06_visualization.ipynb     # publication figures (grayscale, 600 dpi, PNG + PDF)
│   └── 07_recourse.ipynb          # DiCE recourse feasibility (auxiliary axis)
└── results/
    ├── tables/               # CSV outputs consumed by the manuscript
    └── figures/              # fig1–fig5 (PNG + PDF)
```

Run the notebooks **in order** (`01 → 07`); each persists intermediate
artifacts the next one reads.

---

## Reproducing the analysis

### Environment

- Python 3.10
- Core packages: `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `xgboost`,
  `shap`, `dice-ml`, `matplotlib`, `seaborn`

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install numpy pandas pyarrow scikit-learn xgboost shap dice-ml matplotlib seaborn
```

### Run

```bash
# from the repository root, after placing raw files in data/
jupyter lab            # then execute notebooks 01 → 07 in order
```

Key configuration (in the notebooks): `RANDOM_STATE = 42`,
`TEST_SIZE = 0.25`, `MAX_ROWS = 40000`, `N_BOOTSTRAP = 200`. The unified
baseline is a single XGBoost configuration applied identically to every
dataset, with `scale_pos_weight` for imbalance and a prevalence-based
decision threshold.

---

## Key results

**GGI by dataset and regulation** (point estimates; lower is better):

| Domain | Dataset | KR | EU | NAIC |
|--------|---------|:--:|:--:|:----:|
| Credit | Credit Card Approval | 0.533 | 0.612 | 0.577 |
| Credit | German Credit | 0.682 | 0.735 | 0.712 |
| Insurance | Medical Cost | 0.971 | 1.076 | 1.071 |
| Insurance | Porto Seguro | 1.457 | 1.514 | 1.508 |
| — | **Credit mean** | 0.607 | 0.674 | 0.645 |
| — | **Insurance mean** | 1.214 | 1.295 | 1.289 |

- **RQ1 — domain gap.** The insurance-domain gap is roughly **twice** the
  credit-domain gap under every regime. This is driven mainly by the low
  discriminative difficulty of the empirical insurance data (“prediction
  difficulty”), not by inherent bias.
- **RQ2 — regime reordering.** Within each domain the ordering is
  **EU > NAIC > KR**, driven by the explainability axis (the EU scenario sets
  the strictest explainability threshold). NAIC is strictest on fairness but
  does not produce the largest gaps, because the fairness shortfalls in these
  data are small relative to the explainability shortfall.
- **RQ3 — proxy auditing.** For Porto Seguro the disparate-impact ratio
  ranges from 0.000 to 1.681 across candidate proxies; a single-proxy audit
  would report near-parity and miss it. Worst-case aggregation drives
  `s_DI = 0`, so the hidden disparity enters the index.

**Robustness (Studies A–D).** The domain ordering (insurance > credit) holds
under threshold scaling (gap ≈ 0.62), across aggregation rules
(gap 0.62 / 0.36 / 0.22 for Euclidean / mean / max — sign invariant), and
across the explainability-threshold sweep (gap 0.96 → 0.58). Bootstrap 95%
CIs separate the two domains overall; the Medical Cost interval is wide
(`N_test = 335`), so robustness is carried mainly by Porto Seguro.

**Recourse (auxiliary).** DiCE feasibility is high across the three
interpretable-feature datasets (German Credit 1.00, Medical Cost 1.00,
Credit Card Approval 0.84; Porto Seguro N/A over anonymized proxies), so the
recourse axis is demonstrative rather than diagnostic and does not change the
main conclusion.

---

## Outputs

Running the notebooks regenerates:

- `results/tables/` — cross-domain summary, preprocessing summary, primary
  fairness metrics, proxy supplement, GGI matrix, sensitivity studies, and
  bootstrap CIs (CSV).
- `results/figures/` — `fig1_ggi_heatmap`, `fig2_component_stack`,
  `fig3_domain_bars`, `fig4_bootstrap_ci`, `fig5_sensitivity_expl`
  (PNG + PDF, grayscale, 600 dpi).

---

## Limitations

The results are exemplary, not confirmatory. The two insurance datasets are
weakly identified; the regulatory thresholds are scenarios, not statutes;
the benchmark datasets are dated and imperfectly documented; Porto Seguro’s
proxies cannot be mapped to real demographic groups; the GGI tends to be
governed by the explainability axis (a property of gradient-boosted models);
and the recourse axis is a binary feasibility check rather than a cost-based
measure. See the manuscript’s Limitations section for details.

---

## Citation

A citation block will be added upon de-anonymization. For now, please refer
to the anonymized manuscript and this repository.

## License

To be specified upon de-anonymization.
