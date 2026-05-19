# Complete Workflow Examples

End-to-end time series analysis pipelines using **chronobox**, demonstrating how to combine all library components in integrated univariate and multivariate workflows.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `notebooks/01_univariate_workflow.ipynb` | Complete univariate pipeline: stationarity tests, decomposition, filters, model selection (ARIMA/ETS/Theta), forecasting, and cross-validation with R and Stata. Uses the airline and Brazil GDP datasets. |
| 2 | `notebooks/02_multivariate_workflow.ipynb` | Complete multivariate pipeline: unit-root tests, cointegration, VAR/VECM estimation, SVAR identification, IRF/FEVD, forecasting, and cross-validation. Uses the US macro quarterly dataset. |

## Datasets

| File | Description | Frequency | Observations | Variables |
|------|-------------|-----------|-------------|-----------|
| `data/airline.csv` | Box-Jenkins airline passengers (1949-1960) | Monthly | 144 | passengers |
| `data/us_macro_quarterly.csv` | US macro series: GDP, inflation, fed funds, unemployment (1975-2024) | Quarterly | 200 | gdp, inflation, fed_funds, unemployment |
| `data/brazil_gdp.csv` | Brazilian real GDP (2000-2029) | Quarterly | 120 | gdp_real, gdp_log |

All datasets use seed=42 for reproducibility. See `data/all_datasets_index.json` for full metadata.

## Cross-Validation Scripts

- `R/` — R scripts using `forecast`, `vars`, `urca`, and `tseries` packages
- `Stata/` — Stata `.do` files using native `arima`, `var`, and `vec` commands

## Utilities

| File | Description |
|------|-------------|
| `utils/pipeline.py` | `univariate_pipeline()` and `multivariate_pipeline()` orchestrators |
| `utils/plot_helpers.py` | Summary plots for pipeline results |
| `utils/report_generator.py` | HTML report generator for cross-validation comparison |

## Pre-requisites

### Python
```bash
pip install chronobox pandas numpy matplotlib statsmodels scipy
```

### R (for cross-validation)
```r
install.packages(c("forecast", "vars", "urca", "tseries", "lmtest"))
```

### Stata (for cross-validation)
Stata 14+ with `arima`, `var`, and `vec` commands.

## Quick Start

```python
import pandas as pd
import sys
sys.path.insert(0, "..")

from complete_workflow.utils import univariate_pipeline, plot_pipeline_summary

# Load data
airline = pd.read_csv("data/airline.csv", parse_dates=["date"], index_col="date")
airline.index.freq = "MS"

# Run full pipeline
results = univariate_pipeline(
    airline["passengers"],
    tests=["adf", "kpss", "pp"],
    models=["auto_arima", "ets", "theta"],
    forecast_horizon=12,
    seasonal_period=12,
)

# Visualise
fig = plot_pipeline_summary(airline["passengers"], results)
fig.savefig("outputs/univariate_summary.png", dpi=150)
```

## Directory Structure

```
complete_workflow/
  README.md
  notebooks/          # Jupyter tutorial notebooks
  solutions/          # Notebook solutions with outputs
  R/                  # R cross-validation scripts
  Stata/              # Stata cross-validation .do files
  data/               # Datasets and index
  utils/              # Pipeline, plotting, and report utilities
  outputs/            # Generated plots and reports
```
