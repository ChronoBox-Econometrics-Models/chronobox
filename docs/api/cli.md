---
title: "CLI API"
description: "API reference for the chronobox command-line interface — estimate, test, forecast, decompose, filter"
---

# CLI API Reference

!!! info "Module"
    **Entry point**: `chronobox` (installed via `pip install chronobox`)
    **Source**: `chronobox/cli/main.py`

## Overview

| Command | Description |
|---------|-------------|
| `chronobox estimate` | Fit a time series model |
| `chronobox test` | Run a statistical test |
| `chronobox forecast` | Generate forecasts |
| `chronobox decompose` | Decompose a time series |
| `chronobox filter` | Apply a time series filter |

All commands support `--format` (`text`, `json`, `csv`) for output formatting
and `--data` to specify a CSV file or built-in dataset name.

```bash
chronobox --help
chronobox --version
chronobox <command> --help
```

---

## chronobox estimate

Fit a time series model and print summary results.

```bash
chronobox estimate <model_type> --data <path_or_dataset> [options]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `model_type` | Yes | `arima`, `auto_arima`, `var`, `vecm`, `ardl` |
| `--data` | Yes | CSV file path or built-in dataset name |
| `--column` | No | Column name for univariate extraction |
| `--order` | No | Model order, e.g., `0,1,1` |
| `--seasonal` | No | Seasonal order, e.g., `0,1,1,12` |
| `--m` | No | Seasonal period for auto_arima |
| `--lags` | No | Number of lags (VAR/VECM/ARDL) |
| `--maxlags` | No | Maximum lags for VAR lag selection |
| `--rank` | No | Cointegration rank (VECM) |
| `--format` | No | Output format: `text` (default), `json`, `csv` |

### Examples

```bash
# ARIMA(0,1,1) on built-in airline data
chronobox estimate arima --data airline --order 0,1,1

# SARIMA(1,1,1)(1,1,1)[12] on CSV file
chronobox estimate arima --data monthly_sales.csv --order 1,1,1 --seasonal 1,1,1,12

# Auto-ARIMA with seasonal period 12
chronobox estimate auto_arima --data airline --m 12

# VAR on multivariate data
chronobox estimate var --data canada --maxlags 4

# VECM with cointegration rank 1
chronobox estimate vecm --data denmark --lags 2 --rank 1

# JSON output
chronobox estimate arima --data airline --order 1,1,1 --format json
```

---

## chronobox test

Run a statistical test on time series data.

```bash
chronobox test <test_type> --data <path_or_dataset> [options]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `test_type` | Yes | `adf`, `pp`, `kpss`, `ers`, `ljungbox`, `engle_granger` |
| `--data` | Yes | CSV file path or built-in dataset name |
| `--column` | No | Column name for univariate extraction |
| `--lags` | No | Number of lags for the test |
| `--format` | No | Output format: `text`, `json`, `csv` |

### Examples

```bash
# ADF test on Nile data
chronobox test adf --data nile

# KPSS test on specific column
chronobox test kpss --data us_macro_quarterly --column gdp

# Phillips-Perron test on CSV
chronobox test pp --data gdp_data.csv

# Ljung-Box with 20 lags
chronobox test ljungbox --data airline --lags 20

# Engle-Granger cointegration (multivariate data)
chronobox test engle_granger --data denmark

# JSON output for programmatic use
chronobox test adf --data nile --format json
```

---

## chronobox forecast

Fit a model and produce multi-step forecasts.

```bash
chronobox forecast --model <model_type> --data <path_or_dataset> [options]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--model` | Yes | `arima`, `auto_arima`, `var` |
| `--data` | Yes | CSV file path or built-in dataset name |
| `--column` | No | Column name for univariate extraction |
| `--steps` | No | Forecast horizon (default: 12) |
| `--order` | No | ARIMA order, e.g., `0,1,1` |
| `--seasonal` | No | Seasonal order, e.g., `0,1,1,12` |
| `--m` | No | Seasonal period for auto_arima |
| `--maxlags` | No | Maximum lags for VAR |
| `--output` | No | Save forecast to CSV file |
| `--format` | No | Output format: `text`, `json`, `csv` |

### Examples

```bash
# ARIMA forecast 24 steps
chronobox forecast --model arima --data airline --order 0,1,1 --steps 24

# Auto-ARIMA seasonal forecast
chronobox forecast --model auto_arima --data airline --m 12 --steps 12

# VAR forecast
chronobox forecast --model var --data canada --maxlags 4 --steps 8

# Save forecast to CSV
chronobox forecast --model arima --data airline --order 1,1,1 --steps 12 --output forecast.csv

# JSON output
chronobox forecast --model arima --data airline --order 1,1,1 --format json
```

---

## chronobox decompose

Decompose a time series into trend, seasonal, and residual components.

```bash
chronobox decompose --method <method> --data <path_or_dataset> [options]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--method` | Yes | `stl`, `classical` |
| `--data` | Yes | CSV file path or built-in dataset name |
| `--column` | No | Column name for univariate extraction |
| `--period` | Yes | Seasonal period |
| `--type` | No | Decomposition type: `additive` (default), `multiplicative` |
| `--plot` | No | Show plot (flag) |
| `--output` | No | Save components to CSV |
| `--format` | No | Output format: `text`, `json`, `csv` |

### Examples

```bash
# STL decomposition with period 12
chronobox decompose --method stl --data airline --period 12

# Classical multiplicative decomposition
chronobox decompose --method classical --data airline --period 12 --type multiplicative

# Save components and show plot
chronobox decompose --method stl --data co2 --period 12 --output components.csv --plot
```

---

## chronobox filter

Apply a time series filter to extract trend and cycle components.

```bash
chronobox filter <filter_type> --data <path_or_dataset> [options]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `filter_type` | Yes | `hp`, `bk`, `cf`, `hamilton` |
| `--data` | Yes | CSV file path or built-in dataset name |
| `--column` | No | Column name for univariate extraction |
| `--lamb` | No | Lambda for HP filter (default: 1600) |
| `--low` | No | Lower period for BK/CF (default: 6) |
| `--high` | No | Upper period for BK/CF (default: 32) |
| `--K` | No | Truncation for BK filter (default: 12) |
| `--h` | No | Horizon for Hamilton filter (default: 8) |
| `--p` | No | Lags for Hamilton filter (default: 4) |
| `--output` | No | Save filtered output to CSV |
| `--format` | No | Output format: `text`, `json`, `csv` |

### Examples

```bash
# HP filter with quarterly lambda
chronobox filter hp --data us_gdp --lamb 1600

# HP filter for annual data
chronobox filter hp --data us_gdp --lamb 6.25

# Baxter-King band-pass
chronobox filter bk --data us_gdp --low 6 --high 32 --K 12

# Christiano-Fitzgerald
chronobox filter cf --data us_gdp --low 6 --high 32

# Hamilton filter
chronobox filter hamilton --data us_gdp --h 8 --p 4

# Save to CSV
chronobox filter hp --data us_gdp --lamb 1600 --output hp_output.csv --format csv
```

---

## Data Input

All commands accept `--data` which can be either:

1. **CSV file path** — any `.csv` file with the first column as index:

    ```bash
    chronobox estimate arima --data /path/to/mydata.csv --order 1,1,1
    ```

2. **Built-in dataset name** — any name from `list_datasets()`:

    ```bash
    chronobox test adf --data airline
    chronobox estimate var --data canada
    ```

For multivariate datasets, use `--column` to select a specific column
for univariate operations:

```bash
chronobox test adf --data us_macro_quarterly --column gdp
```

---

## Output Formats

All commands support three output formats via `--format`:

| Format | Description | Use Case |
|--------|-------------|----------|
| `text` | Human-readable summary (default) | Terminal inspection |
| `json` | JSON output | Programmatic consumption, pipelines |
| `csv` | CSV output | Data export, spreadsheets |

### Piping and Scripting

```bash
# Pipe JSON to jq
chronobox test adf --data nile --format json | jq '.pvalue'

# Chain with CSV processing
chronobox forecast --model arima --data airline --order 1,1,1 --format csv > forecast.csv

# Use in shell scripts
RESULT=$(chronobox test adf --data nile --format json)
echo $RESULT | python3 -c "import json, sys; d=json.load(sys.stdin); print(d)"
```

---

## Python API

The CLI is built on the same Python API. For programmatic use, prefer
the Python interface directly:

```python
# CLI equivalent: chronobox estimate arima --data airline --order 0,1,1
from chronobox import ARIMA
from chronobox.datasets import load_dataset

data = load_dataset("airline")
results = ARIMA(order=(0, 1, 1)).fit(data.values)
print(results.summary())
```

The `main()` function can also be called from Python:

```python
from chronobox.cli import main

# Programmatic CLI invocation
main(["estimate", "arima", "--data", "airline", "--order", "0,1,1"])
```

::: chronobox.cli.main.main
    options:
      show_root_heading: false
      show_source: true

---

## See Also

- [Datasets API](datasets.md) -- Available built-in datasets
- [ARIMA API](arima.md) -- ARIMA Python API
- [VAR API](var.md) -- VAR Python API
- [Filters API](filters.md) -- Filter Python API
- [Diagnostics API](diagnostics.md) -- Statistical tests Python API
- [Quick Start](../getting-started/quickstart.md) -- Getting started guide
