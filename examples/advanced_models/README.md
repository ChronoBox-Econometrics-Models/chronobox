# Advanced Models Examples

Examples for advanced time series models in chronobox: **FAVAR**, **TVP-VAR**, **GVAR**, and **Historical Decomposition**.

## Notebooks

### 1. FAVAR - Factor-Augmented VAR
- Estimate latent factors from a large panel of macroeconomic series
- Augment a standard VAR with extracted factors
- Impulse response analysis with factor-augmented identification
- **Notebook**: `notebooks/01_favar.ipynb`

### 2. TVP-VAR - Time-Varying Parameter VAR
- Estimate VAR models with coefficients that evolve over time
- Visualize parameter instability and structural change
- Compare time-varying vs. constant-parameter impulse responses
- **Notebook**: `notebooks/02_tvp_var.ipynb`

### 3. GVAR - Global VAR
- Multi-country VAR framework with trade-weighted foreign variables
- Estimate country-specific VARX models linked through a weight matrix
- Cross-country spillover analysis and global impulse responses
- **Notebook**: `notebooks/03_gvar_historical_decomposition.ipynb`

## Datasets

| File | Description |
|------|-------------|
| `data/us_macro_quarterly.csv` | US macroeconomic data (GDP, inflation, fed funds, unemployment) - 200 quarterly observations |
| `data/us_macro_panel.csv` | Multi-country panel (US, UK, DE, JP, BR) with 4 variables each - 80 quarterly observations |

## Cross-Validation

- `R/` - R scripts using reference packages for validation
- `Stata/` - Stata .do files for validation

## Structure

```
advanced_models/
  README.md
  notebooks/          # Jupyter tutorial notebooks
  solutions/          # Solution notebooks with outputs
  R/                  # R validation scripts
  Stata/              # Stata validation scripts
  data/               # Datasets and generators
  utils/              # Data generators and plot helpers
  outputs/            # Saved results for cross-validation
```

## Data Generators

The `utils/data_generators.py` module provides:

- `generate_factor_model(n, n_series, n_factors, seed)` - synthetic factor model for FAVAR
- `generate_tvp_var(n, k, seed)` - VAR with time-varying parameters
- `generate_gvar_data(n, n_countries, k, seed)` - multi-country panel for GVAR

All generators use `seed=42` by default for reproducibility.
