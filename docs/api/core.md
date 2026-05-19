---
title: "Core API"
description: "API reference for chronobox.core — TimeSeriesData, TimeSeriesResults, ChronoBoxConfig, LagPolynomial"
---

# Core API Reference

!!! info "Module"
    **Import**: `from chronobox.core import TimeSeriesData, TimeSeriesResults, LagPolynomial`
    **Source**: `chronobox/core/`

## Overview

The core module provides the foundational classes used across all ChronoBox models:

| Class | Description | Source |
|-------|-------------|--------|
| `TimeSeriesData` | Time series container with index and frequency metadata | `chronobox/core/tsdata.py` |
| `TimeSeriesResults` | Base result class returned by all model `.fit()` methods | `chronobox/core/results.py` |
| `ChronoBoxConfig` | Global configuration singleton | `chronobox/core/config.py` |
| `LagPolynomial` | Lag polynomial algebra for ARMA representations | `chronobox/core/lag_polynomial.py` |

---

## TimeSeriesData {#timeseries-data}

Time series container with frequency metadata, transformations, and basic analysis.

```python
TimeSeriesData(
    data: ndarray | pd.Series | pd.DataFrame | list[float],
    index: DatetimeIndex | RangeIndex | None = None,
    frequency: str | None = None,
    name: str | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `ndarray \| pd.Series \| pd.DataFrame \| list` | *required* | Input data. If DataFrame, uses the first numeric column |
| `index` | `DatetimeIndex \| RangeIndex \| None` | `None` | Temporal index. Inferred from data if possible |
| `frequency` | `str \| None` | `None` | Frequency string (`'M'`, `'Q'`, `'A'`, `'W'`, `'D'`). Auto-detected if None |
| `name` | `str \| None` | `None` | Name of the series |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `values` | `ndarray` | Numeric values of the series |
| `index` | `DatetimeIndex \| RangeIndex` | Temporal index |
| `frequency` | `str \| None` | Detected or provided frequency |
| `name` | `str` | Series name |
| `nobs` | `int` | Number of observations |
| `missing_mask` | `ndarray[bool]` | Boolean mask of missing (NaN) values |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `diff(d=1)` | `TimeSeriesData` | Regular differencing of order `d` |
| `seasonal_diff(s)` | `TimeSeriesData` | Seasonal differencing $y_t - y_{t-s}$ |
| `log()` | `TimeSeriesData` | Natural log transformation |
| `boxcox(lam=None)` | `(TimeSeriesData, float)` | Box-Cox transformation with optional lambda estimation |
| `split(test_size)` | `(TimeSeriesData, TimeSeriesData)` | Train/test split preserving temporal order |
| `describe()` | `pd.Series` | Descriptive statistics |
| `plot(**kwargs)` | `matplotlib.Axes` | Plot the series |
| `to_pandas()` | `pd.Series` | Convert to pandas Series |
| `from_pandas(series, frequency=None)` | `TimeSeriesData` | Class method: create from pandas Series |

### Example

```python
import numpy as np
import pandas as pd
from chronobox.core import TimeSeriesData

# From numpy array
y = TimeSeriesData(
    data=np.random.randn(120),
    index=pd.date_range("2010-01", periods=120, freq="M"),
    name="GDP Growth"
)
print(y)
# TimeSeriesData(name='GDP Growth', nobs=120, frequency='ME')

# Transformations
y_diff = y.diff(1)
y_log = y.log() if (y.values > 0).all() else y

# Train/test split
train, test = y.split(test_size=24)

# Descriptive statistics
print(y.describe())
```

```python
# From pandas Series
import pandas as pd
series = pd.Series([1.2, 3.4, 2.1, 4.5], name="sales")
ts = TimeSeriesData.from_pandas(series)
```

::: chronobox.core.tsdata.TimeSeriesData
    options:
      show_root_heading: false
      show_source: true
      show_bases: true
      members: false

---

## TimeSeriesResults {#timeseries-results}

Container for model estimation results. Returned by all model `.fit()` methods.

```python
TimeSeriesResults(
    params: ndarray,
    param_names: list[str],
    se: ndarray,
    loglike: float,
    nobs: int,
    nobs_effective: int,
    residuals: ndarray,
    fitted_values: ndarray,
    model_name: str,
    forecast_func: callable | None = None,
    model: object | None = None,
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `ndarray` | Estimated parameters |
| `param_names` | `list[str]` | Names of the parameters |
| `se` | `ndarray` | Standard errors |
| `tvalues` | `ndarray` | t-statistics ($\text{params} / \text{se}$) |
| `pvalues` | `ndarray` | Two-sided p-values |
| `loglike` | `float` | Maximized log-likelihood |
| `nobs` | `int` | Total number of observations |
| `nobs_effective` | `int` | Effective observations used in estimation |
| `residuals` | `ndarray` | Model residuals |
| `fitted_values` | `ndarray` | In-sample fitted values |
| `model_name` | `str` | Model name (e.g., `"ARIMA(1,1,1)"`) |
| `aic` | `float` | Akaike Information Criterion: $-2\ell + 2k$ |
| `bic` | `float` | Bayesian IC: $-2\ell + k\ln(n)$ |
| `aicc` | `float` | Corrected AIC: $\text{AIC} + \frac{2k(k+1)}{n-k-1}$ |
| `hqic` | `float` | Hannan-Quinn IC: $-2\ell + 2k\ln(\ln(n))$ |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Formatted summary with parameters, SE, t-values, p-values, ICs |
| `forecast(steps=1, alpha=0.05)` | `dict` | Out-of-sample forecasts with keys `'forecast'`, `'lower'`, `'upper'` |
| `plot_diagnostics(figsize=(12,8))` | `Figure` | 2x2 diagnostic panel (residuals, ACF, QQ, histogram) |
| `plot_forecast(steps=12, alpha=0.05)` | `Figure` | Forecast fan chart |
| `to_dataframe()` | `pd.DataFrame` | Parameters as DataFrame (estimate, std_err, t_value, p_value) |
| `save(path)` | `None` | Serialize results to file |
| `load(path)` | `TimeSeriesResults` | Static method: load results from file |

### Example

```python
from chronobox import ARIMA

model = ARIMA(order=(1, 1, 1))
results = model.fit(y)

# Summary table
print(results.summary())

# Information criteria
print(f"AIC: {results.aic:.2f}, BIC: {results.bic:.2f}")

# Parameter table as DataFrame
df = results.to_dataframe()
print(df)

# Forecasting
fc = results.forecast(steps=12, alpha=0.05)
print(fc["forecast"])    # point forecasts
print(fc["lower"])       # lower confidence bound
print(fc["upper"])       # upper confidence bound

# Diagnostics
results.plot_diagnostics()

# Save/load
results.save("my_model.pkl")
loaded = TimeSeriesResults.load("my_model.pkl")
```

::: chronobox.core.results.TimeSeriesResults
    options:
      show_root_heading: false
      show_source: true
      show_bases: true
      members: false

---

## Configuration {#configuration}

Global configuration singleton for default settings.

```python
from chronobox.core.config import config

# View defaults
print(config.default_optimizer)           # "L-BFGS-B"
print(config.default_information_criterion)  # "aicc"

# Modify defaults
config.default_alpha = 0.10              # change default significance level
config.max_iterations = 1000             # increase optimizer iterations
```

### ChronoBoxConfig

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_optimizer` | `str` | `"L-BFGS-B"` | Default optimization algorithm |
| `max_iterations` | `int` | `500` | Maximum optimizer iterations |
| `tolerance` | `float` | `1e-8` | Convergence tolerance |
| `default_information_criterion` | `str` | `"aicc"` | Default IC for model selection |
| `default_trend` | `str` | `"n"` | Default trend specification |
| `default_alpha` | `float` | `0.05` | Default significance level |

::: chronobox.core.config.ChronoBoxConfig
    options:
      show_root_heading: false
      show_source: true
      members: true

---

## LagPolynomial {#lag-polynomial}

Algebraic operations on lag polynomials for ARMA model representations.

A lag polynomial $\phi(L) = 1 - \phi_1 L - \phi_2 L^2 - \cdots - \phi_p L^p$
represents the AR or MA operator in ARMA models.

::: chronobox.core.lag_polynomial.LagPolynomial
    options:
      show_root_heading: true
      show_source: true
      show_bases: true

---

## See Also

- [ARIMA API](arima.md) -- ARIMA, SARIMA, ARFIMA models
- [ETS API](ets.md) -- Exponential smoothing models
- [VAR API](var.md) -- Vector autoregression models
- [Getting Started: Core Concepts](../getting-started/core-concepts.md) -- Introduction to the API patterns
