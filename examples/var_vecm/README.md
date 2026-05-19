# VAR/VECM Examples

Examples and tutorials for the VAR/VECM module of **chronobox**, covering VAR estimation, VECM with Johansen cointegration, impulse response functions (IRF), forecast error variance decomposition (FEVD), and Granger causality testing.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `notebooks/01_var_basics.ipynb` | Introduction to VAR models: specification, estimation, lag selection, stability analysis, and forecasting using the Canada macro dataset. |
| 2 | `notebooks/02_vecm_cointegration.ipynb` | VECM modeling and Johansen cointegration test: rank determination, estimation of cointegrating vectors, and error correction dynamics. |
| 3 | `notebooks/03_irf_fevd.ipynb` | Impulse response functions and forecast error variance decomposition: orthogonalized and structural IRFs, bootstrapped confidence bands. |
| 4 | `notebooks/04_granger_causality.ipynb` | Granger causality testing: pairwise and block tests, interpretation of causal relationships, and comparison with structural approaches. |

## Datasets

| File | Description | Observations |
|------|-------------|-------------|
| `data/canada_macro.csv` | Synthetic Canada macro (e, prod, rw, U) — quarterly, 1980Q1–2000Q4 | 84 |
| `data/us_macro_quarterly.csv` | Synthetic US macro (gdp, inflation, fed_funds, unemployment) — quarterly, 1975Q1–2024Q4 | 200 |

## Cross-Validation Scripts

- `R/` - R scripts using `vars`, `urca`, `tsDyn`, and `lmtest` packages for comparison
- `Stata/` - Stata `.do` files using native `var`, `vec`, and `vargranger` commands

## Pre-requisites

### Python
```bash
pip install chronobox pandas numpy matplotlib statsmodels scipy
```

### R (for cross-validation)
```r
install.packages(c("vars", "urca", "tsDyn", "lmtest", "tseries"))
```

### Stata (for cross-validation)
Stata 14+ with `var`, `vec`, and `vargranger` commands (included in standard installation).

## Quick Start

1. Generate datasets (or use the pre-generated CSVs):
   ```bash
   cd examples/var_vecm
   python data/generate_datasets.py
   ```

2. Open the notebooks:
   ```bash
   jupyter notebook notebooks/
   ```

3. Run cross-validation scripts (optional):
   ```bash
   Rscript R/01_var_basics.R
   stata -b do Stata/01_var_basics.do
   ```

## Directory Structure

```
var_vecm/
  README.md              # This file
  notebooks/             # Jupyter notebook tutorials
  solutions/             # Completed notebooks with outputs
  R/                     # R cross-validation scripts
  Stata/                 # Stata cross-validation scripts
  data/                  # Datasets (CSV)
  utils/                 # Python utilities (data generators, plot helpers)
  outputs/               # Saved outputs (plots, tables, comparison results)
```
