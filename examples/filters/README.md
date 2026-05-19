# Filters Examples

Examples demonstrating the chronobox filters module: Hodrick-Prescott (HP),
Baxter-King (BK), Christiano-Fitzgerald (CF), Hamilton regression filter,
Beveridge-Nelson (BN) decomposition, and Diebold-Yilmaz spillover index.

## Notebooks

### 1. Trend-Cycle Decomposition (`notebooks/01_trend_cycle_decomposition.ipynb`)

Applies HP, BK, CF, and Hamilton filters to US GDP data, comparing the
extracted trend and cyclical components side by side. Covers parameter
sensitivity (e.g., HP lambda) and discusses pros/cons of each filter.

### 2. Beveridge-Nelson Decomposition (`notebooks/02_beveridge_nelson.ipynb`)

Demonstrates the Beveridge-Nelson decomposition on Brazil GDP data.
Compares the BN permanent/transitory split with HP-filtered components and
discusses identification in the presence of unit roots.

### 3. Spillover Analysis (`notebooks/03_spillover_analysis.ipynb`)

Uses multivariate synthetic data to compute Diebold-Yilmaz spillover indices.
Builds a connectedness table, plots directional spillovers, and shows how
the spillover index evolves over a rolling window.

## Datasets

| File | Description | Observations | Frequency |
|------|-------------|-------------|-----------|
| `data/us_gdp_quarterly.csv` | Synthetic US real GDP | 200 | Quarterly |
| `data/brazil_gdp.csv` | Synthetic Brazil GDP | 120 | Quarterly |

All datasets are generated with `seed=42` for reproducibility. Re-generate
with:

```bash
python data/generate_datasets.py
```

## Cross-validation

- `R/` — R scripts using `mFilter`, `vars`, and `tseries` for reference results
- `Stata/` — Stata `.do` files using native `tsfilter` commands

## Directory Structure

```
filters/
  notebooks/       Jupyter tutorial notebooks
  solutions/       Completed notebook versions
  R/               R validation scripts
  Stata/           Stata validation scripts
  data/            Datasets and generator
  utils/           Data generators and plot helpers
  outputs/         Saved figures and tables
```
