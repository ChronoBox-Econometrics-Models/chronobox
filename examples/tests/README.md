# Statistical Tests Examples

Examples and tutorials for the **chronobox** statistical testing module, covering unit root tests, cointegration tests, and structural break detection.

## Notebooks

### 1. Unit Root Tests (`notebooks/01_unit_root_tests.ipynb`)
Comprehensive guide to testing for stationarity using ADF, Phillips-Perron, KPSS, ERS, and Zivot-Andrews tests. Covers the distinction between I(0), I(1), and I(2) processes, trend-stationary vs. difference-stationary series, and how to interpret conflicting test results.

### 2. Cointegration Tests (`notebooks/02_cointegration_tests.ipynb`)
Tutorial on Engle-Granger two-step method and Johansen cointegration tests. Demonstrates how to identify long-run equilibrium relationships between non-stationary variables, determine cointegrating rank, and estimate the cointegrating vector.

### 3. Structural Break Tests (`notebooks/03_structural_break_tests.ipynb`)
Guide to detecting structural breaks using Chow test, CUSUM/CUSUM-of-squares, and Bai-Perron multiple break detection. Shows how breaks affect unit root test inference and how to test for breaks at known and unknown dates.

### 4. Integrated Testing Strategy (`notebooks/04_integrated_testing.ipynb`)
Putting it all together: a practical workflow combining unit root, cointegration, and structural break tests for applied time series analysis. Uses the US and Brazil GDP datasets to demonstrate a complete testing pipeline.

## Datasets

| File | Description | Observations | Frequency |
|------|-------------|:------------:|:---------:|
| `data/us_gdp_quarterly.csv` | Synthetic US real GDP | 200 | Quarterly |
| `data/brazil_gdp.csv` | Synthetic Brazil GDP | 120 | Quarterly |

Both datasets are generated with `seed=42` for full reproducibility. To regenerate:

```bash
cd examples/tests
python data/generate_datasets.py
```

## Directory Structure

```
tests/
├── README.md
├── notebooks/          # Jupyter notebooks with exercises
├── solutions/          # Completed notebook solutions
├── R/                  # R validation scripts (forecast, urca, vars)
├── Stata/              # Stata validation scripts (.do files)
├── data/               # Synthetic datasets
│   ├── us_gdp_quarterly.csv
│   ├── brazil_gdp.csv
│   └── generate_datasets.py
├── utils/              # Helper functions
│   ├── __init__.py
│   ├── data_generators.py
│   └── plot_helpers.py
└── outputs/            # Generated plots and results
```

## Cross-Validation

Each notebook has companion scripts in `R/` and `Stata/` that replicate the same tests using reference implementations, allowing direct comparison of chronobox results against established packages:

- **R**: `urca` (unit root, Johansen), `tseries` (ADF, KPSS), `strucchange` (structural breaks)
- **Stata**: `dfuller`, `pperron`, `kpss`, `vecrank`, `estat sbcusum`

## Prerequisites

```bash
pip install chronobox matplotlib pandas numpy
```
