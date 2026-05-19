# Decomposition Examples

Examples and tutorials for the decomposition module of **chronobox**, covering classical decomposition (additive/multiplicative), STL (Seasonal-Trend decomposition using LOESS), and X-13 ARIMA-SEATS.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `notebooks/01_classical_decomposition.ipynb` | Classical additive and multiplicative decomposition using the airline and CO2 datasets. Comparison of methods, seasonal adjustment, and diagnostics. |
| 2 | `notebooks/02_stl_decomposition.ipynb` | STL decomposition: parameter tuning (seasonal/trend windows), robustness to outliers, and multiple seasonalities using CO2 and synthetic data. |
| 3 | `notebooks/03_x13_arima_seats.ipynb` | X-13 ARIMA-SEATS seasonal adjustment: automatic model selection, trading day and holiday effects, and comparison with STL. |

## Datasets

| File | Description | Observations |
|------|-------------|-------------|
| `data/airline.csv` | Classic Box-Jenkins airline passengers (monthly, 1949-1960) | 144 |
| `data/co2.csv` | Mauna Loa CO2 monthly averages (1959-1997, seed=42) | 468 |
| `data/brazil_ipca.csv` | Synthetic Brazilian IPCA monthly inflation (2004-2023, seed=42) | 240 |

## Cross-Validation Scripts

- `R/` - R scripts using `stats::stl`, `stats::decompose`, and `seasonal` (X-13) packages for comparison
- `Stata/` - Stata `.do` files using `tsfilter` and seasonal adjustment commands

## Pre-requisites

### Python
```bash
pip install chronobox pandas numpy matplotlib statsmodels scipy
```

### R (for cross-validation)
```r
install.packages(c("stats", "stl2", "seasonal", "forecast", "tseries"))
```

### Stata (for cross-validation)
Stata 14+ with `tsfilter` and time series commands.

## Quick Start

1. Generate datasets (or use the pre-generated CSVs):
   ```bash
   cd examples/decomposition
   python data/generate_datasets.py
   ```

2. Open the notebooks:
   ```bash
   jupyter notebook notebooks/
   ```

3. Run cross-validation scripts (optional):
   ```bash
   Rscript R/01_classical_validation.R
   stata -b do Stata/01_classical_validation.do
   ```

## Directory Structure

```
decomposition/
  README.md              # This file
  notebooks/             # Jupyter notebook tutorials
  solutions/             # Completed notebooks with outputs
  R/                     # R cross-validation scripts
  Stata/                 # Stata cross-validation scripts
  data/                  # Datasets (CSV)
  utils/                 # Python utilities (data generators, plot helpers)
  outputs/               # Saved outputs (plots, tables, comparison results)
```
