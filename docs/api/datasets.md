---
title: "Datasets API"
description: "API reference for built-in datasets — classic, macro, Brazil, finance, and simulated time series"
---

# Datasets API Reference

!!! info "Module"
    **Import**: `from chronobox.datasets import load_dataset, list_datasets`
    **Source**: `chronobox/datasets/__init__.py`

## Overview

| Function | Description |
|----------|-------------|
| `load_dataset(name)` | Load a built-in dataset by name |
| `list_datasets(category)` | List available datasets with metadata |

---

## load_dataset

Load a built-in dataset by name. Returns a `pd.Series` for univariate datasets
and a `pd.DataFrame` for multivariate datasets.

```python
load_dataset(
    name: str,
) -> pd.Series | pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Dataset name (see [Available Datasets](#available-datasets) below) |

**Returns**: `pd.Series` (univariate) or `pd.DataFrame` (multivariate)

**Raises**:

- `KeyError` — if dataset name is not found
- `FileNotFoundError` — if CSV file is missing (non-simulated datasets)

### Example

```python
from chronobox.datasets import load_dataset

# Univariate — returns pd.Series
airline = load_dataset("airline")
print(airline.shape)       # (144,)
print(airline.index[:3])   # DatetimeIndex

# Multivariate — returns pd.DataFrame
canada = load_dataset("canada")
print(canada.columns)      # ['e', 'prod', 'rw', 'U']

# Simulated — generated on-the-fly
var2 = load_dataset("var2")
print(var2.shape)           # (500, 2)
```

::: chronobox.datasets.load_dataset
    options:
      show_root_heading: false
      show_source: true

---

## list_datasets

List all available datasets, optionally filtered by category.

```python
list_datasets(
    category: str | None = None,
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `category` | `str \| None` | `None` | Filter: `'classic'`, `'macro'`, `'brazil'`, `'finance'`, `'simulated'`. If None, returns all |

**Returns**: `pd.DataFrame` with columns `name`, `category`, `description`, `bundled`

### Example

```python
from chronobox.datasets import list_datasets

# All datasets
all_ds = list_datasets()
print(f"Total datasets: {len(all_ds)}")

# Only Brazilian macro datasets
brazil = list_datasets(category="brazil")
print(brazil[["name", "description"]])
```

::: chronobox.datasets.list_datasets
    options:
      show_root_heading: false
      show_source: true

---

## Available Datasets

### Classic

| Name | Description | Type |
|------|-------------|------|
| `airline` | Monthly airline passengers (1949–1960) | Series |
| `nile` | Annual Nile river flow (1871–1970) | Series |
| `sunspot` | Annual sunspot numbers (1700–2008) | Series |
| `lynx` | Annual Canadian lynx trappings (1821–1934) | Series |
| `co2` | Monthly CO₂ Mauna Loa (1959–1997) | Series |
| `uspop` | US population decennial (1790–2000) | Series |

### Macro International

| Name | Description | Type |
|------|-------------|------|
| `us_gdp` | US quarterly GDP (1947–2023) | Series |
| `us_macro_quarterly` | US macro quarterly panel | DataFrame |
| `us_consumption` | US personal consumption monthly | Series |
| `canada` | Canada dataset (e, prod, rw, U) | DataFrame |
| `uk_drivers` | UK road casualties (1969–1984) | Series |
| `uk_gas` | UK quarterly gas consumption | Series |
| `denmark` | Denmark money, income, prices, interest | DataFrame |

### Macro Brazil

| Name | Description | Type |
|------|-------------|------|
| `ipca` | IPCA monthly inflation | Series |
| `pib_brasil` | Brazil quarterly GDP | Series |
| `selic` | SELIC target rate | Series |
| `cambio` | BRL/USD exchange rate | Series |
| `desemprego` | Unemployment rate (PNAD) | Series |
| `producao_industrial` | Industrial production index | Series |
| `macro_brasil` | Brazil macro panel | DataFrame |

### Finance

| Name | Description | Type |
|------|-------------|------|
| `sp500_returns` | S&P 500 daily returns | Series |
| `ibovespa_returns` | Ibovespa daily returns | Series |
| `exchange_rates` | Multiple exchange rates | DataFrame |
| `vix` | VIX volatility index | Series |

### Simulated

Simulated datasets are generated on-the-fly with a fixed seed for
reproducibility. No CSV files are needed.

| Name | Description | Type |
|------|-------------|------|
| `arma11` | Simulated ARMA(1,1) process | Series |
| `var2` | Simulated bivariate VAR(2) | DataFrame |
| `cointegrated` | Simulated cointegrated pair | DataFrame |
| `structural_break` | Simulated structural break | Series |
| `long_memory` | Simulated long memory (ARFIMA) | Series |

---

## Usage Recipes

### Quick exploratory analysis

```python
from chronobox.datasets import load_dataset
from chronobox.visualization import plot_series, set_theme

set_theme("professional")

# Load and visualize
data = load_dataset("airline")
fig = plot_series(data, title="Airline Passengers", ylabel="Thousands")
```

### Unit root testing on built-in data

```python
from chronobox.datasets import load_dataset
from chronobox.tests_stat import adf_test, kpss_test

nile = load_dataset("nile")
print(adf_test(nile.values).summary())
print(kpss_test(nile.values).summary())
```

### Multivariate analysis

```python
from chronobox.datasets import load_dataset
from chronobox.models.var import VAR

canada = load_dataset("canada")
model = VAR(maxlags=4)
results = model.fit(canada)
print(results.summary())
```

---

## See Also

- [Core API](core.md) -- `TimeSeriesData` container
- [Experiment API](experiment.md) -- Using datasets with `ChronoExperiment`
- [Quick Start](../getting-started/quickstart.md) -- Getting started with chronobox
