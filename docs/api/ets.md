---
title: "ETS API"
description: "API reference for ETS, HoltWinters, ThetaMethod, and auto_ets — exponential smoothing state-space models"
---

# ETS Family API Reference

!!! info "Module"
    **Import**: `from chronobox import ETS, HoltWinters, ThetaMethod, auto_ets`
    **Source**: `chronobox/models/ets.py`, `chronobox/models/holtwinters.py`, `chronobox/models/theta.py`, `chronobox/selection/auto_ets.py`

## Overview

| Class / Function | Description | Use Case |
|------------------|-------------|----------|
| `ETS` | All 30 ETS(Error, Trend, Seasonal) state-space models | Full taxonomy with MLE estimation |
| `HoltWinters` | Classical Holt-Winters exponential smoothing | Seasonal forecasting with trend |
| `ThetaMethod` | Theta method (Assimakopoulos & Nikolopoulos 2000) | Simple, competitive forecasting |
| `auto_ets` | Automatic ETS model selection via AICc | When model form is unknown |

---

## ETS

Exponential Smoothing State Space Model supporting all 30 combinations of
Error $\times$ Trend $\times$ Seasonal components.

```python
ETS(
    error: str = "A",
    trend: str = "A",
    seasonal: str = "N",
    seasonal_period: int | None = None,
    damped: bool = False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `error` | `str` | `"A"` | Error type: `'A'` (additive) or `'M'` (multiplicative) |
| `trend` | `str` | `"A"` | Trend type: `'N'` (none), `'A'` (additive), `'M'` (multiplicative) |
| `seasonal` | `str` | `"N"` | Seasonal type: `'N'` (none), `'A'` (additive), `'M'` (multiplicative) |
| `seasonal_period` | `int \| None` | `None` | Seasonal period $m$. Required if `seasonal != 'N'` |
| `damped` | `bool` | `False` | Use damped trend (appends `'d'` to trend: `'Ad'` or `'Md'`) |

### The 30 ETS Models

All valid combinations of Error(A,M) $\times$ Trend(N,A,Ad,M,Md) $\times$ Seasonal(N,A,M):

| Error | Trend | Seasonal | Model Name | Classical Name |
|-------|-------|----------|------------|----------------|
| A | N | N | ETS(A,N,N) | Simple Exponential Smoothing |
| A | A | N | ETS(A,A,N) | Holt's Linear Trend |
| A | Ad | N | ETS(A,Ad,N) | Damped Trend |
| A | A | A | ETS(A,A,A) | Additive Holt-Winters |
| A | A | M | ETS(A,A,M) | Multiplicative Seasonal |
| A | Ad | A | ETS(A,Ad,A) | Damped Additive Holt-Winters |
| A | Ad | M | ETS(A,Ad,M) | Damped Multiplicative Seasonal |
| M | A | M | ETS(M,A,M) | Multiplicative Holt-Winters |
| M | Ad | M | ETS(M,Ad,M) | Damped Multiplicative HW |
| ... | ... | ... | ... | (30 total combinations) |

### `.fit()` Method

```python
ETS.fit(
    endog: array-like,
    optimized: bool = True,
) -> ETSResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `array-like` | *required* | Time series data |
| `optimized` | `bool` | `True` | Optimize parameters via L-BFGS-B. If False, uses heuristic values |

**Returns**: `ETSResults`

!!! warning "Data requirements"
    - Multiplicative error (`error='M'`) requires all positive data
    - Multiplicative seasonality (`seasonal='M'`) requires all positive data
    - Seasonal models need at least $2m$ observations

### `.simulate()` Method

```python
ETS.simulate(
    nsimulations: int,
    alpha: float = 0.1,
    beta: float | None = None,
    gamma: float | None = None,
    phi: float | None = None,
    l0: float = 100.0,
    b0: float | None = None,
    s0: ndarray | None = None,
    sigma: float = 1.0,
    seed: int | None = None,
) -> ndarray
```

Simulate from an ETS model with given parameters.

### ETSResults

| Attribute | Type | Description |
|-----------|------|-------------|
| `error` | `str` | Error type (`'A'` or `'M'`) |
| `trend` | `str` | Trend type (`'N'`, `'A'`, `'Ad'`, `'M'`, `'Md'`) |
| `seasonal` | `str` | Seasonal type (`'N'`, `'A'`, `'M'`) |
| `seasonal_period` | `int` | Seasonal period $m$ |
| `alpha` | `float` | Level smoothing parameter |
| `beta` | `float \| None` | Trend smoothing parameter |
| `gamma` | `float \| None` | Seasonal smoothing parameter |
| `phi` | `float \| None` | Damping parameter |
| `l0` | `float` | Initial level |
| `b0` | `float \| None` | Initial trend |
| `s0` | `ndarray \| None` | Initial seasonal components (length $m$) |
| `sigma2` | `float` | Innovation variance |
| `states` | `ndarray` | State matrix $(T+1 \times \text{num\_states})$ |
| `aic` | `float` | Akaike Information Criterion |
| `bic` | `float` | Bayesian Information Criterion |
| `aicc` | `float` | Corrected AIC |
| `loglik` | `float` | Log-likelihood |
| `resid` | `ndarray` | Innovation residuals |
| `fittedvalues` | `ndarray` | One-step-ahead fitted values |
| `nobs` | `int` | Number of observations |

| Method | Description |
|--------|-------------|
| `summary()` | Formatted summary with smoothing parameters, initial states, ICs |
| `forecast(steps)` | Multi-step-ahead point forecasts |

### Example

```python
import numpy as np
from chronobox import ETS

# Simple Exponential Smoothing
ses = ETS(error="A", trend="N", seasonal="N")
results = ses.fit(y)

# Holt's Linear Trend with damping
holt = ETS(error="A", trend="A", seasonal="N", damped=True)
results = holt.fit(y)
print(f"alpha={results.alpha:.4f}, beta={results.beta:.4f}, phi={results.phi:.4f}")

# Additive Holt-Winters for monthly data
hw = ETS(error="A", trend="A", seasonal="A", seasonal_period=12)
results = hw.fit(monthly_data)
fc = results.forecast(steps=24)

# Multiplicative seasonality
hw_mult = ETS(error="M", trend="A", seasonal="M", seasonal_period=12, damped=True)
results = hw_mult.fit(monthly_data)
print(results.summary())
```

::: chronobox.models.ets.ETS
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit
        - simulate

::: chronobox.models.ets.ETSResults
    options:
      show_root_heading: true
      show_source: false
      members:
        - summary
        - forecast

---

## Holt-Winters {#holt-winters}

Classical Holt-Winters exponential smoothing with additive or multiplicative seasonality.
Parameters optimized via L-BFGS-B.

```python
HoltWinters(
    seasonal_period: int,
    method: str = "additive",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seasonal_period` | `int` | *required* | Seasonal period $m$ |
| `method` | `str` | `"additive"` | `'additive'` or `'multiplicative'` |

### `.fit()` Method

```python
HoltWinters.fit(endog: array-like) -> HoltWintersResults
```

### HoltWintersResults

| Attribute | Type | Description |
|-----------|------|-------------|
| `method` | `str` | `'additive'` or `'multiplicative'` |
| `seasonal_period` | `int` | Seasonal period |
| `alpha` | `float` | Level smoothing parameter |
| `beta` | `float` | Trend smoothing parameter |
| `gamma` | `float` | Seasonal smoothing parameter |
| `l0`, `b0` | `float` | Initial level and trend |
| `s0` | `ndarray` | Initial seasonal indices (length $m$) |
| `level` | `ndarray` | Estimated level at each time point |
| `trend` | `ndarray` | Estimated trend at each time point |
| `season` | `ndarray` | Estimated seasonal component at each time point |
| `resid` | `ndarray` | Residuals |
| `fittedvalues` | `ndarray` | Fitted values |
| `sse`, `mse`, `rmse` | `float` | Error metrics |

| Method | Description |
|--------|-------------|
| `summary()` | Formatted summary |
| `forecast(steps)` | Multi-step forecasts |

### Example

```python
from chronobox import HoltWinters

# Additive Holt-Winters
hw = HoltWinters(seasonal_period=12, method="additive")
results = hw.fit(monthly_sales)
print(results.summary())

# Forecast next year
fc = results.forecast(steps=12)

# Multiplicative Holt-Winters
hw_mult = HoltWinters(seasonal_period=4, method="multiplicative")
results = hw_mult.fit(quarterly_gdp)
```

::: chronobox.models.holtwinters.HoltWinters
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit

::: chronobox.models.holtwinters.HoltWintersResults
    options:
      show_root_heading: true
      show_source: false
      members:
        - summary
        - forecast

---

## Theta Method {#theta-method}

The Theta method of Assimakopoulos & Nikolopoulos (2000), which is equivalent
to SES with drift (Hyndman & Billah 2003).

```python
ThetaMethod()
```

No constructor parameters -- the method automatically estimates the SES smoothing
parameter and OLS drift.

### `.fit()` Method

```python
ThetaMethod.fit(endog: array-like) -> ThetaResults
```

### ThetaResults

| Attribute | Type | Description |
|-----------|------|-------------|
| `ses_level` | `float` | Final SES level $l_T$ |
| `drift` | `float` | Drift term (half the OLS slope) |
| `alpha` | `float` | SES smoothing parameter |
| `slope` | `float` | OLS trend slope |
| `intercept` | `float` | OLS trend intercept |
| `fittedvalues` | `ndarray` | Fitted values |
| `resid` | `ndarray` | Residuals |
| `nobs` | `int` | Number of observations |

| Method | Description |
|--------|-------------|
| `summary()` | Formatted summary |
| `forecast(steps)` | SES-with-drift forecasts |

### Example

```python
from chronobox import ThetaMethod

theta = ThetaMethod()
results = theta.fit(y)
print(results.summary())

fc = results.forecast(steps=12)
```

::: chronobox.models.theta.ThetaMethod
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit

::: chronobox.models.theta.ThetaResults
    options:
      show_root_heading: true
      show_source: false
      members:
        - summary
        - forecast

---

## Auto-ETS {#auto-ets}

Automatic ETS model selection by fitting all valid model combinations and
selecting the best by AICc.

```python
auto_ets(
    y: array-like,
    seasonal_period: int = 1,
    error: str | None = None,
    trend: str | None = None,
    seasonal: str | None = None,
    damped: bool | None = None,
    information_criterion: str = "aicc",
) -> AutoETSResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `array-like` | *required* | Time series data |
| `seasonal_period` | `int` | `1` | Seasonal period $m$ (1 for non-seasonal) |
| `error` | `str \| None` | `None` | Fix error type (`'A'` or `'M'`). None = try both |
| `trend` | `str \| None` | `None` | Fix trend type. None = try all |
| `seasonal` | `str \| None` | `None` | Fix seasonal type. None = try all |
| `damped` | `bool \| None` | `None` | Fix damping. None = try both |
| `information_criterion` | `str` | `"aicc"` | Selection criterion: `'aic'`, `'aicc'`, `'bic'` |

**Returns**: `AutoETSResult`

### AutoETSResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `best_model` | `ETSResults` | Results of the best-fitting model |
| `best_spec` | `tuple[str, str, str]` | (error, trend, seasonal) of the best model |
| `all_results` | `list` | All models with AICc values, sorted |
| `n_models_tried` | `int` | Number of models successfully fitted |
| `n_models_failed` | `int` | Number of models that failed |

| Method | Description |
|--------|-------------|
| `summary()` | Selection summary with top-5 models and best model details |
| `forecast(steps)` | Forecast using the best model |

### Example

```python
from chronobox import auto_ets

# Fully automatic selection for monthly data
result = auto_ets(monthly_data, seasonal_period=12)
print(result.summary())
print(f"Best model: ETS({result.best_spec[0]},{result.best_spec[1]},{result.best_spec[2]})")

# Constrain search: only additive error
result = auto_ets(y, seasonal_period=12, error="A")

# Non-seasonal selection
result = auto_ets(y, seasonal_period=1)
fc = result.forecast(steps=12)
```

::: chronobox.selection.auto_ets.auto_ets
    options:
      show_root_heading: false
      show_source: true

::: chronobox.selection.auto_ets.AutoETSResult
    options:
      show_root_heading: true
      show_source: false
      members:
        - summary
        - forecast

---

## See Also

- [Core API](core.md) -- `TimeSeriesResults` attributes and methods
- [ARIMA API](arima.md) -- ARIMA family models
- [Decomposition API](decomposition.md) -- STL and X-13 decomposition
- [ETS Theory](../theory/ets-theory.md) -- State-space formulation and taxonomy
- [ETS User Guide](../user-guide/ets/index.md) -- Step-by-step usage guides
