# Empirical Asset Pricing via Machine Learning: A Replication Study

**ECON5130: Machine Learning in Finance: Group 11**

Replication of [Gu, Kelly, and Xiu (2020)](https://doi.org/10.1093/rfs/hhaa009), *The Review of Financial Studies*, 33(5), 2223–2273.

## Overview

This project evaluates whether machine learning methods can improve equity risk premium prediction relative to traditional models. We apply six models to a panel of ~38,000 U.S. equities (1960–2024) from the CRSP database with 114 firm-level predictor signals.

## Models

| Model | Type | Key Hyperparameters |
|-------|------|-------------------|
| CAPM | Benchmark | Market Beta only |
| OLS | Linear | All ~80 predictors |
| LASSO | Penalised Linear | λ tuned over 50 values |
| PCR | Dimension Reduction | K ∈ {1, ..., 20} components |
| Random Forest | Tree Ensemble | Depth, leaf size, feature fraction |
| Neural Networks | Deep Learning | NN1 [128,64], NN3 [64,32,16], NN5 [32,16,8,4,2] |

## Key Results

| Model | Monthly OOS R² (%) | Annual OOS R² (%) |
|-------|-------------------|-------------------|
| PCR | +0.093 | +1.238 |
| NN3 | +0.072 | +1.334 |
| CAPM | +0.065 | +1.194 |
| LASSO | +0.052 | +1.248 |
| RF | −0.141 | −2.279 |
| OLS | −0.221 | −0.682 |

NN5 [32,16,8,4,2] achieves the highest test R² of +0.163% but NN3 was selected by validation criterion.

## Methodology

- **Data Split**: Train (1960–1999) / Validation (2000–2010) / Test (2011–2024)
- **OOS R²**: Benchmarks against zero forecast, following the paper's Section 1.8
- **Preprocessing**: 40% missing threshold, monthly median imputation, cross-sectional rank normalisation to [−1, 1]
- **NN Stabilisation**: Huber loss, gradient clipping, target winsorisation (1st/99th percentile), ensemble averaging (3 seeds)
- **Evaluation**: Diebold-Mariano tests, decile portfolio analysis, cumulative long-short returns

## Repository Structure

```
├── Group11_ECON5130_1.ipynb      # Main Jupyter notebook (all models + analysis)
├── Group11_ECON5130_1.pdf        # Report (8 pages)
├── README.md                     # This file
└── data/                         # Data files (not included — see below)
    ├── crspm_and_predictors.csv  # CRSP monthly stock data with predictors
    ├── T-Bill_FRED.csv           # 3-month Treasury Bill rates
    └── SignalDoc.csv             # Signal documentation
```

## Data

The dataset is not included in this repository due to size (~4.7 GB). To replicate:

1. Download `crspm_and_predictors.csv` from the [Open Asset Pricing](https://www.openassetpricing.com/) portal
2. Download `TB3MS` (3-Month Treasury Bill) from [FRED](https://fred.stlouisfed.org/series/TB3MS)
3. Download `SignalDoc.csv` from the Open Asset Pricing portal
4. Place all three files in the same directory as the notebook

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
torch
```

## Reference

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *The Review of Financial Studies*, 33(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa009
