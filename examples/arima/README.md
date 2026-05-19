# ARIMA Examples

Examples and tutorials for the ARIMA module of **chronobox**, covering ARIMA, SARIMA, Auto-ARIMA, and ARFIMA models.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `notebooks/01_arima_basics.ipynb` | Introduction to ARIMA models: identification, estimation, diagnostics, and forecasting using the airline and Nile datasets. |
| 2 | `notebooks/02_sarima.ipynb` | Seasonal ARIMA (SARIMA) modeling with the airline passengers dataset, including seasonal decomposition and model selection. |
| 3 | `notebooks/03_auto_arima.ipynb` | Automatic model selection with Auto-ARIMA: information criteria (AIC/BIC), stepwise search, and comparison with manual selection. |
| 4 | `notebooks/04_arfima.ipynb` | Fractionally integrated ARIMA (ARFIMA) for long-memory processes: estimating the fractional differencing parameter and forecasting. |

## Datasets

| File | Description | Observations |
|------|-------------|-------------|
| `data/airline.csv` | Classic Box-Jenkins airline passengers (monthly, 1949-1960) | 144 |
| `data/nile.csv` | Nile River annual flow at Aswan (1871-1970) | 100 |
| `data/brazil_ipca.csv` | Synthetic Brazilian IPCA monthly inflation (2004-2023, seed=42) | 240 |

## Cross-Validation Scripts

- `R/` - R scripts using `forecast`, `urca`, and `fracdiff` packages for comparison
- `Stata/` - Stata `.do` files using native `arima` and `arfima` commands

## Pre-requisites

### Python
```bash
pip install chronobox pandas numpy matplotlib statsmodels scipy
```

### R (for cross-validation)
```r
install.packages(c("forecast", "urca", "fracdiff", "tseries", "lmtest"))
```

### Stata (for cross-validation)
Stata 14+ with `arima` and `arfima` commands (included in standard installation).

## Quick Start

1. Generate datasets (or use the pre-generated CSVs):
   ```bash
   cd examples/arima
   python data/generate_datasets.py
   ```

2. Open the notebooks:
   ```bash
   jupyter notebook notebooks/
   ```

3. Run cross-validation scripts (optional):
   ```bash
   Rscript R/01_arima_basics.R
   stata -b do Stata/01_arima_basics.do
   ```

## Directory Structure

```
arima/
  README.md              # This file
  notebooks/             # Jupyter notebook tutorials
  solutions/             # Completed notebooks with outputs
  R/                     # R cross-validation scripts
  Stata/                 # Stata cross-validation scripts
  data/                  # Datasets (CSV)
  utils/                 # Python utilities (data generators, plot helpers)
  outputs/               # Saved outputs (plots, tables, comparison results)
```
