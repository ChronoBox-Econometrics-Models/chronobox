# ARDL Examples

Examples and tutorials for the ARDL module of **chronobox**, covering Autoregressive Distributed Lag (ARDL) estimation, Pesaran-Shin-Smith bounds test for cointegration, and Error Correction Models (ECM).

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `notebooks/01_ardl_bounds_test.ipynb` | Introduction to ARDL models and bounds testing: model specification, lag selection, Pesaran-Shin-Smith bounds test for cointegration with mixed I(0)/I(1) regressors, and interpretation of results. |
| 2 | `notebooks/02_ecm_long_run.ipynb` | Error Correction Model (ECM) from ARDL: estimation of long-run coefficients, short-run dynamics, speed of adjustment, forecasting, and diagnostic checks. |

## Datasets

| File | Description | Observations |
|------|-------------|-------------|
| `data/us_macro_quarterly.csv` | Synthetic US macro (gdp, inflation, fed_funds, unemployment) — quarterly, 1975Q1–2024Q4 | 200 |
| `data/ardl_synthetic.csv` | Synthetic mixed-integration data (y, x1, x2, x3) — quarterly, 1970Q1–2019Q4 | 200 |

The `ardl_synthetic.csv` dataset includes variables with different integration orders:
- **y** (I(1)): dependent variable, cointegrated with x1
- **x1** (I(1)): cointegrated with y (long-run: y = 1.5 + 0.6*x1)
- **x2** (I(0)): stationary regressor
- **x3** (I(1)): independent random walk (not cointegrated with y)

## Cross-Validation Scripts

- `R/` - R scripts using `dynamac`, `ARDL`, and `urca` packages for comparison
- `Stata/` - Stata `.do` files using native `ardl` and `ec` commands

## Pre-requisites

### Python
```bash
pip install chronobox pandas numpy matplotlib statsmodels scipy
```

### R (for cross-validation)
```r
install.packages(c("dynamac", "ARDL", "urca", "tseries", "lmtest"))
```

### Stata (for cross-validation)
Stata 15+ with `ardl` command (install via `ssc install ardl`).

## Quick Start

1. Generate datasets (or use the pre-generated CSVs):
   ```bash
   cd examples/ardl
   python data/generate_datasets.py
   ```

2. Open the notebooks:
   ```bash
   jupyter notebook notebooks/
   ```

3. Run cross-validation scripts (optional):
   ```bash
   Rscript R/01_ardl_bounds_test.R
   stata -b do Stata/01_ardl_bounds_test.do
   ```

## Directory Structure

```
ardl/
  README.md              # This file
  notebooks/             # Jupyter notebook tutorials
  solutions/             # Completed notebooks with outputs
  R/                     # R cross-validation scripts
  Stata/                 # Stata cross-validation scripts
  data/                  # Datasets (CSV)
  utils/                 # Python utilities (data generators, plot helpers)
  outputs/               # Saved outputs (plots, tables, comparison results)
```
