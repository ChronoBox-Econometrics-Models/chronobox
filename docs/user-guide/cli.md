# CLI - Command Line Interface

## Overview

ChronoBox provides a command-line interface for quick analysis without
writing Python code.

## Commands

### Estimate Models

```bash
# ARIMA
chronobox estimate arima --order 0,1,1 --data airline

# SARIMA
chronobox estimate arima --order 0,1,1 --seasonal-order 0,1,1,12 --data airline

# Auto-ARIMA
chronobox estimate auto-arima --data airline --seasonal --m 12

# VAR
chronobox estimate var --data canada --maxlags 4
```

### Statistical Tests

```bash
# Unit root tests
chronobox test adf --data airline
chronobox test kpss --data airline
chronobox test pp --data airline

# Diagnostics
chronobox test ljung-box --data residuals.csv --lags 10
```

### Forecasting

```bash
chronobox forecast --model arima --order 0,1,1 --steps 12 --data airline
```

### Data Operations

```bash
# List available datasets
chronobox datasets list

# List by category
chronobox datasets list --category classic

# Describe a dataset
chronobox datasets describe airline
```

## Input Data

The CLI accepts:

- Built-in dataset names (e.g., `airline`, `canada`)
- CSV file paths (e.g., `data.csv`)
- Excel file paths (e.g., `data.xlsx`)

```bash
# From CSV
chronobox estimate arima --order 1,1,1 --data /path/to/data.csv

# From built-in dataset
chronobox estimate arima --order 1,1,1 --data airline
```

## Output

Results are printed to stdout. Use `--output` to save:

```bash
chronobox estimate arima --order 0,1,1 --data airline --output results.json
```
