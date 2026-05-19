# SVAR/BVAR Examples

Examples and tutorials for the SVAR and BVAR modules of **chronobox**, covering Structural VAR identification (Cholesky, AB model, Blanchard-Quah, sign restrictions) and Bayesian VAR with Minnesota prior.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `notebooks/01_svar_cholesky_ab.ipynb` | SVAR identification with short-run restrictions: Cholesky decomposition and the AB model. Includes recursive identification for monetary policy and demand-supply systems. |
| 2 | `notebooks/02_svar_bq_sign.ipynb` | Long-run identification (Blanchard-Quah) and sign restrictions. Covers permanent vs transitory shocks and set-identified models with sign constraints. |
| 3 | `notebooks/03_bvar_minnesota.ipynb` | Bayesian VAR with Minnesota/Litterman prior: prior specification, posterior estimation, forecasting, and comparison with frequentist VAR. |

## Datasets

| File | Description | Observations |
|------|-------------|-------------|
| `data/us_macro_quarterly.csv` | Synthetic US macro (gdp, inflation, fed_funds, unemployment) — quarterly, 1975Q1–2024Q4 | 200 |
| `data/demand_supply.csv` | Synthetic demand-supply system (price, quantity) — quarterly | 200 |
| `data/monetary_policy.csv` | Synthetic monetary policy system (output_gap, inflation, interest_rate) — quarterly | 200 |
| `data/blanchard_quah.csv` | Synthetic BQ-style system (output_growth, unemployment) — quarterly | 300 |
| `data/sign_restriction.csv` | Synthetic sign-restriction DGP (output, inflation, rate) — quarterly | 200 |

## Cross-Validation Scripts

- `R/` - R scripts using `vars`, `svars`, and `bvartools` packages for comparison
- `Stata/` - Stata `.do` files using native `svar` and `var` commands

## Pre-requisites

### Python
```bash
pip install chronobox pandas numpy matplotlib statsmodels scipy
```

### R (for cross-validation)
```r
install.packages(c("vars", "svars", "bvartools", "urca"))
```

### Stata (for cross-validation)
Stata 14+ with `var`, `svar`, and `irf` commands (included in standard installation).

## Quick Start

1. Generate datasets (or use the pre-generated CSVs):
   ```bash
   cd examples/svar
   python data/generate_datasets.py
   ```

2. Open the notebooks:
   ```bash
   jupyter notebook notebooks/
   ```

3. Run cross-validation scripts (optional):
   ```bash
   Rscript R/01_svar_cholesky.R
   stata -b do Stata/01_svar_cholesky.do
   ```

## Directory Structure

```
svar/
  README.md              # This file
  notebooks/             # Jupyter notebook tutorials
  solutions/             # Completed notebooks with outputs
  R/                     # R cross-validation scripts
  Stata/                 # Stata cross-validation scripts
  data/                  # Datasets (CSV)
  utils/                 # Python utilities (data generators, plot helpers)
  outputs/               # Saved outputs (plots, tables, comparison results)
```
