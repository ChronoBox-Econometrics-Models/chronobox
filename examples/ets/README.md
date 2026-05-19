# ETS Examples

Examples and tutorials for the ETS module of **chronobox**, covering Exponential Smoothing (ETS), Holt-Winters, Theta method, and Auto-ETS models.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `notebooks/01_ets_basics.ipynb` | Introduction to Exponential Smoothing: Simple ES, Holt's linear trend, and Holt-Winters additive/multiplicative models with the airline dataset. |
| 2 | `notebooks/02_theta_method.ipynb` | The Theta method for forecasting: decomposition into Theta lines, parameter estimation, and comparison with ETS on the IPCA dataset. |
| 3 | `notebooks/03_auto_ets.ipynb` | Automatic ETS model selection: information criteria (AIC/BIC), error taxonomy (A/M), and comparison with manual Holt-Winters on the synthetic dataset. |

## Datasets

| File | Description | Observations |
|------|-------------|-------------|
| `data/airline.csv` | Classic Box-Jenkins airline passengers (monthly, 1949-1960) | 144 |
| `data/brazil_ipca.csv` | Synthetic Brazilian IPCA monthly inflation (2004-2023, seed=42) | 240 |
| `data/ets_synthetic.csv` | Synthetic series with trend and multiplicative seasonality (2009-2023, seed=42) | 180 |

## Cross-Validation Scripts

- `R/` - R scripts using `forecast` and `smooth` packages for comparison
- `Stata/` - Stata `.do` files using native `tssmooth` and `ets` commands

## Pre-requisites

### Python
```bash
pip install chronobox pandas numpy matplotlib statsmodels scipy
```

### R (for cross-validation)
```r
install.packages(c("forecast", "smooth", "tseries", "ggplot2"))
```

### Stata (for cross-validation)
Stata 14+ with `tssmooth` commands (included in standard installation).

## Quick Start

1. Generate datasets (or use the pre-generated CSVs):
   ```bash
   cd examples/ets
   python data/generate_datasets.py
   ```

2. Open the notebooks:
   ```bash
   jupyter notebook notebooks/
   ```

3. Run cross-validation scripts (optional):
   ```bash
   Rscript R/01_ets_basics.R
   stata -b do Stata/01_ets_basics.do
   ```

## Directory Structure

```
ets/
  README.md              # This file
  notebooks/             # Jupyter notebook tutorials
  solutions/             # Completed notebooks with outputs
  R/                     # R cross-validation scripts
  Stata/                 # Stata cross-validation scripts
  data/                  # Datasets (CSV)
  utils/                 # Python utilities (data generators, plot helpers)
  outputs/               # Saved outputs (plots, tables, comparison results)
```
