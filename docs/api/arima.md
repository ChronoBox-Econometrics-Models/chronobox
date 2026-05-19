---
title: "ARIMA API"
description: "API reference for ARIMA, SARIMA, ARFIMA, and auto_arima — univariate autoregressive integrated moving average models"
---

# ARIMA Family API Reference

!!! info "Module"
    **Import**: `from chronobox import ARIMA, ARFIMA, auto_arima`
    **Source**: `chronobox/models/arima.py`, `chronobox/models/arfima.py`, `chronobox/selection/auto_arima.py`

## Overview

| Class / Function | Description | Use Case |
|------------------|-------------|----------|
| `ARIMA` | ARIMA(p,d,q) with optional seasonal SARIMA(p,d,q)(P,D,Q)[s] | Standard univariate forecasting |
| `ARFIMA` | Fractionally integrated ARIMA(p,d,q) with $d \in (-0.5, 0.5)$ | Long-memory processes |
| `auto_arima` | Automatic order selection via Hyndman-Khandakar algorithm | When order is unknown |

---

## ARIMA

ARIMA(p,d,q) with optional seasonal extension SARIMA(p,d,q)(P,D,Q)[s].
Estimation via conditional sum of squares (CSS) and/or exact maximum likelihood
through the Kalman filter.

```python
ARIMA(
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: str | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `order` | `tuple[int, int, int]` | `(1, 0, 0)` | $(p, d, q)$ -- AR order, differencing order, MA order |
| `seasonal_order` | `tuple[int, int, int, int]` | `(0, 0, 0, 0)` | $(P, D, Q, s)$ -- Seasonal AR, differencing, MA orders, and period |
| `trend` | `str \| None` | `None` | `'n'` (none), `'c'` (constant), `'t'` (linear), `'ct'` (both). Default: `'c'` if $d+D=0$, else `'n'` |

### `.fit()` Method

```python
ARIMA.fit(
    endog: array-like,
    method: str = "css-mle",
    maxiter: int = 500,
) -> TimeSeriesResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `array-like` | *required* | Endogenous time series |
| `method` | `str` | `"css-mle"` | Estimation method: `'css'`, `'mle'`, or `'css-mle'` |
| `maxiter` | `int` | `500` | Maximum optimizer iterations |

**Returns**: [`TimeSeriesResults`](core.md#timeseries-results)

### `.simulate()` Method

```python
ARIMA.simulate(n: int, seed: int | None = None) -> ndarray
```

Simulate `n` observations from the fitted model.

### `.loglike()` Method

```python
ARIMA.loglike(params: ndarray) -> float
```

Compute log-likelihood for a given parameter vector.

### Estimation Methods

| Method | Description |
|--------|-------------|
| `'css'` | Conditional Sum of Squares -- fast, approximate |
| `'mle'` | Exact MLE via Kalman filter with Lyapunov initialization |
| `'css-mle'` | CSS for initial values, then MLE refinement (recommended) |

!!! tip "SARIMA via `seasonal_order`"
    To fit a seasonal ARIMA model, set `seasonal_order=(P, D, Q, s)`:
    ```python
    model = ARIMA(order=(1,1,1), seasonal_order=(1,1,1,12))
    ```
    This fits a SARIMA(1,1,1)(1,1,1)[12] model.

### Example

```python
import numpy as np
from chronobox import ARIMA

# Simulate AR(1) data
rng = np.random.default_rng(42)
y = np.cumsum(rng.normal(size=200))

# Fit ARIMA(1,1,1)
model = ARIMA(order=(1, 1, 1))
results = model.fit(y)
print(results.summary())

# Forecast 12 steps ahead
fc = results.forecast(steps=12, alpha=0.05)
print(fc["forecast"])

# SARIMA for monthly data
sarima = ARIMA(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results_s = sarima.fit(monthly_data, method="css-mle")
```

::: chronobox.models.arima.ARIMA
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit
        - simulate
        - loglike
        - model_name

---

## ARFIMA

Fractionally Integrated ARIMA allowing $d \in (-0.5, 0.5)$ for long-memory processes.

The fractional differencing operator is defined as:

$$
(1 - L)^d = \sum_{k=0}^{\infty} \pi_k L^k, \quad \pi_0 = 1, \quad \pi_k = \pi_{k-1} \frac{k - 1 - d}{k}
$$

```python
ARFIMA(
    order: tuple[int, float, int] = (0, 0.0, 0),
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `order` | `tuple[int, float, int]` | `(0, 0.0, 0)` | $(p, d, q)$ where $d$ can be fractional |

### `.fit()` Method

```python
ARFIMA.fit(
    endog: array-like,
    method: str = "css",
    estimate_d: bool = False,
) -> ARFIMAResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `array-like` | *required* | Time series data |
| `method` | `str` | `"css"` | `'css'` or `'mle'` |
| `estimate_d` | `bool` | `False` | If True, estimate $d$ jointly with AR/MA parameters |

### `.estimate_d()` Method

```python
ARFIMA.estimate_d(
    endog: array-like,
    method: str = "gph",
    bandwidth_exp: float | None = None,
) -> float
```

Estimate the fractional differencing parameter $d$ semi-parametrically.

| Method | Description | Reference |
|--------|-------------|-----------|
| `'gph'` | GPH log-periodogram regression | Geweke & Porter-Hudak (1983) |
| `'whittle'` | Local Whittle estimator | Robinson (1995) |

### ARFIMAResults

| Attribute | Type | Description |
|-----------|------|-------------|
| `d` | `float` | Estimated fractional differencing parameter |
| `ar_params` | `ndarray` | AR coefficients |
| `ma_params` | `ndarray` | MA coefficients |
| `sigma2` | `float` | Innovation variance |
| `order` | `tuple` | $(p, d, q)$ |
| `aic` | `float` | Akaike Information Criterion |
| `bic` | `float` | Bayesian Information Criterion |
| `aicc` | `float` | Corrected AIC |
| `loglik` | `float` | Log-likelihood |
| `resid` | `ndarray` | Residuals |
| `fittedvalues` | `ndarray` | Fitted values |

| Method | Description |
|--------|-------------|
| `summary()` | Formatted summary |
| `forecast(steps)` | Multi-step-ahead forecasts |

### Example

```python
import numpy as np
from chronobox import ARFIMA

# Fit ARFIMA(1, d, 0) with fixed d
model = ARFIMA(order=(1, 0.3, 0))
results = model.fit(y, method="css")
print(f"d = {results.d:.4f}")
print(results.summary())

# Estimate d jointly
model2 = ARFIMA(order=(1, 0.0, 0))
results2 = model2.fit(y, estimate_d=True)
print(f"Estimated d = {results2.d:.4f}")

# Semi-parametric d estimation
d_hat = model2.estimate_d(y, method="gph")
print(f"GPH estimate: d = {d_hat:.4f}")

# Forecast
fc = results.forecast(steps=12)
```

::: chronobox.models.arfima.ARFIMA
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit
        - estimate_d

::: chronobox.models.arfima.ARFIMAResults
    options:
      show_root_heading: true
      show_source: false
      members:
        - summary
        - forecast

---

## Auto-ARIMA {#auto-arima}

Automatic ARIMA model selection using the Hyndman-Khandakar (2008) stepwise algorithm.

```python
auto_arima(
    y: array-like,
    seasonal: bool = True,
    m: int = 1,
    d: int | None = None,
    D: int | None = None,
    max_p: int = 5,
    max_q: int = 5,
    max_P: int = 2,
    max_Q: int = 2,
    max_d: int = 2,
    max_D: int = 1,
    information_criterion: str = "aicc",
    stepwise: bool = True,
    trace: bool = False,
) -> TimeSeriesResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `array-like` | *required* | Time series data |
| `seasonal` | `bool` | `True` | Consider seasonal models |
| `m` | `int` | `1` | Seasonal period (12 for monthly, 4 for quarterly) |
| `d` | `int \| None` | `None` | Differencing order. If None, determined via ADF test |
| `D` | `int \| None` | `None` | Seasonal differencing order. If None, auto-determined |
| `max_p` | `int` | `5` | Maximum AR order |
| `max_q` | `int` | `5` | Maximum MA order |
| `max_P` | `int` | `2` | Maximum seasonal AR order |
| `max_Q` | `int` | `2` | Maximum seasonal MA order |
| `max_d` | `int` | `2` | Maximum differencing order |
| `max_D` | `int` | `1` | Maximum seasonal differencing order |
| `information_criterion` | `str` | `"aicc"` | Selection criterion: `'aic'`, `'aicc'`, `'bic'` |
| `stepwise` | `bool` | `True` | Stepwise search (True) or exhaustive (False) |
| `trace` | `bool` | `False` | Print each model tested |

**Returns**: [`TimeSeriesResults`](core.md#timeseries-results) from the best model.

### Example

```python
from chronobox import auto_arima

# Non-seasonal auto-ARIMA
results = auto_arima(y, seasonal=False)
print(results.model_name)  # e.g., "ARIMA(1,1,1)"
print(results.summary())

# Seasonal auto-ARIMA for monthly data
results = auto_arima(y, seasonal=True, m=12, trace=True)
fc = results.forecast(steps=24)

# Custom search space
results = auto_arima(
    y,
    seasonal=True,
    m=4,
    max_p=3,
    max_q=3,
    information_criterion="bic",
)
```

::: chronobox.selection.auto_arima.auto_arima
    options:
      show_root_heading: false
      show_source: true

---

## Helper Functions

### `fractional_diff`

Apply fractional differencing $(1-L)^d$ to a time series.

::: chronobox.models.arfima.fractional_diff
    options:
      show_root_heading: true
      show_source: false

### `simulate_arfima`

Simulate an ARFIMA(p,d,q) process.

::: chronobox.models.arfima.simulate_arfima
    options:
      show_root_heading: true
      show_source: false

---

## See Also

- [Core API](core.md) -- `TimeSeriesResults` attributes and methods
- [ETS API](ets.md) -- Alternative exponential smoothing models
- [ARIMA Theory](../theory/arima-theory.md) -- Mathematical foundations
- [ARIMA User Guide](../user-guide/arima/arima.md) -- Step-by-step usage guide
- [Auto-ARIMA Guide](../user-guide/arima/auto-arima.md) -- Automatic selection guide
