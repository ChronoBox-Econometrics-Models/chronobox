---
title: "Filters API"
description: "API reference for economic filters — HP, Baxter-King, Christiano-Fitzgerald, Hamilton, Beveridge-Nelson"
---

# Filters API Reference

!!! info "Module"
    **Import**: `from chronobox.filters import hp_filter, bk_filter, cf_filter, hamilton_filter, bn_decomposition`
    **Source**: `chronobox/filters/`

## Overview

| Function | Description | Use Case |
|----------|-------------|----------|
| `hp_filter` | Hodrick-Prescott filter | Trend-cycle decomposition (standard) |
| `bk_filter` | Baxter-King band-pass filter | Isolating business cycle frequencies |
| `cf_filter` | Christiano-Fitzgerald band-pass filter | Asymmetric band-pass, no endpoint loss |
| `hamilton_filter` | Hamilton (2018) regression filter | Robust alternative to HP filter |
| `bn_decomposition` | Beveridge-Nelson decomposition | Permanent-transitory decomposition |

---

## hp_filter

Hodrick-Prescott filter decomposing a time series into trend and cycle components
by solving:

$$
\min_{\tau} \left\{ \sum_{t=1}^{T}(y_t - \tau_t)^2 + \lambda \sum_{t=2}^{T-1}[(\tau_{t+1} - \tau_t) - (\tau_t - \tau_{t-1})]^2 \right\}
$$

```python
hp_filter(
    y: ndarray,
    lamb: float | None = None,
    frequency: str | None = None,
) -> tuple[ndarray, ndarray]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Input time series |
| `lamb` | `float \| None` | `None` | Smoothing parameter $\lambda$. If `None`, inferred from `frequency` |
| `frequency` | `str \| None` | `None` | `'annual'` ($\lambda = 6.25$), `'quarterly'` ($\lambda = 1600$), `'monthly'` ($\lambda = 129600$) |

**Returns**: `(trend, cycle)` — tuple of ndarrays

!!! tip "Choosing $\lambda$"
    The standard values follow Ravn & Uhlig (2002): $\lambda = 1600$ for quarterly data
    is the most common. For annual data, $\lambda = 6.25$ is recommended over $100$.
    You can also pass a custom value directly via `lamb`.

### Example

```python
import numpy as np
from chronobox.filters import hp_filter

# Simulate GDP-like series
rng = np.random.default_rng(42)
trend_true = np.linspace(100, 120, 200) + 0.5 * np.cumsum(rng.normal(0, 0.1, 200))
cycle_true = 2 * np.sin(2 * np.pi * np.arange(200) / 32)
y = trend_true + cycle_true

# HP filter with quarterly lambda
trend, cycle = hp_filter(y, frequency="quarterly")
print(f"Trend range: [{trend.min():.1f}, {trend.max():.1f}]")
print(f"Cycle std: {cycle.std():.4f}")

# Custom lambda
trend_smooth, cycle_smooth = hp_filter(y, lamb=10000)
```

::: chronobox.filters.hp.hp_filter
    options:
      show_root_heading: false
      show_source: true

---

## bk_filter

Baxter-King symmetric band-pass filter (Baxter & King, 1999). Isolates cyclical
components within a specified frequency band.

The ideal band-pass filter weights are approximated by a finite moving average
of order $2K + 1$:

$$
y_t^c = \sum_{j=-K}^{K} a_j \, y_{t-j}
$$

where $a_j$ are chosen to pass frequencies between $\frac{2\pi}{\text{high}}$ and
$\frac{2\pi}{\text{low}}$.

```python
bk_filter(
    y: ndarray,
    low: int = 6,
    high: int = 32,
    trunc: int = 12,
) -> ndarray
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Input time series |
| `low` | `int` | `6` | Minimum period (in data frequency units) |
| `high` | `int` | `32` | Maximum period (in data frequency units) |
| `trunc` | `int` | `12` | Truncation lag $K$. Output loses $K$ observations at each end |

**Returns**: `ndarray` — Cyclical component (length $T - 2K$)

!!! warning "Endpoint loss"
    The BK filter trims `trunc` observations from each end. With the default
    `trunc=12`, you lose 24 observations total. Use `cf_filter` if endpoint
    preservation is important.

### Example

```python
import numpy as np
from chronobox.filters import bk_filter

y = np.random.randn(200).cumsum()

# Standard business cycle band (6-32 quarters)
cycle = bk_filter(y, low=6, high=32, trunc=12)
print(f"Input length: {len(y)}, Output length: {len(cycle)}")  # 200, 176
print(f"Cycle std: {cycle.std():.4f}")
```

### bk_filter_weights

Compute the symmetric BK filter weights without applying them.

```python
bk_filter_weights(
    low: int = 6,
    high: int = 32,
    trunc: int = 12,
) -> ndarray
```

**Returns**: `ndarray` — Filter weights (shape `(2*trunc + 1,)`)

::: chronobox.filters.bk.bk_filter
    options:
      show_root_heading: false
      show_source: true

::: chronobox.filters.bk.bk_filter_weights
    options:
      show_root_heading: false
      show_source: true

---

## cf_filter

Christiano-Fitzgerald asymmetric band-pass filter (Christiano & Fitzgerald, 2003).
Unlike BK, this filter is asymmetric and uses the full sample — no endpoint loss.

The filter approximates the ideal band-pass using a time-varying projection that
accounts for the non-stationarity of the series.

```python
cf_filter(
    y: ndarray,
    low: int = 6,
    high: int = 32,
    drift: bool = False,
) -> ndarray
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Input time series |
| `low` | `int` | `6` | Minimum period |
| `high` | `int` | `32` | Maximum period |
| `drift` | `bool` | `False` | If `True`, assume the series has a linear drift |

**Returns**: `ndarray` — Cyclical component (same length as input)

### Example

```python
import numpy as np
from chronobox.filters import cf_filter

y = np.random.randn(200).cumsum()

# CF filter — no endpoint loss
cycle = cf_filter(y, low=6, high=32)
print(f"Input length: {len(y)}, Output length: {len(cycle)}")  # 200, 200
print(f"Cycle std: {cycle.std():.4f}")

# With drift assumption
cycle_drift = cf_filter(y, low=6, high=32, drift=True)
```

::: chronobox.filters.cf.cf_filter
    options:
      show_root_heading: false
      show_source: true

---

## hamilton_filter

Hamilton (2018) regression-based filter. Projects $y_{t+h}$ on $p$ lags of $y_t$
and takes the residual as the cyclical component:

$$
y_{t+h} = \beta_0 + \beta_1 y_t + \beta_2 y_{t-1} + \cdots + \beta_p y_{t-p+1} + v_{t+h}
$$

The fitted values form the trend and the residuals form the cycle.

```python
hamilton_filter(
    y: ndarray,
    h: int = 8,
    p: int = 4,
) -> tuple[ndarray, ndarray]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Input time series |
| `h` | `int` | `8` | Forecast horizon (Hamilton recommends $h=8$ for quarterly data) |
| `p` | `int` | `4` | Number of autoregressive lags |

**Returns**: `(trend, cycle)` — tuple of ndarrays

!!! tip "Hamilton vs HP"
    Hamilton (2018) argues that the HP filter creates spurious cycles and proposes
    this regression filter as a robust alternative. For quarterly data, $h=8, p=4$
    captures business cycle frequencies without the endpoint instability of HP.

### hamilton_filter_detailed

Returns a `HamiltonFilterResult` dataclass with additional information.

```python
hamilton_filter_detailed(
    y: ndarray,
    h: int = 8,
    p: int = 4,
) -> HamiltonFilterResult
```

### HamiltonFilterResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `trend` | `ndarray` | Trend component (fitted values) |
| `cycle` | `ndarray` | Cyclical component (residuals) |
| `coefficients` | `ndarray` | OLS regression coefficients $[\beta_0, \beta_1, \ldots, \beta_p]$ |
| `h` | `int` | Forecast horizon used |
| `p` | `int` | Number of lags used |

### Example

```python
import numpy as np
from chronobox.filters import hamilton_filter, hamilton_filter_detailed

y = np.random.randn(200).cumsum()

# Simple usage
trend, cycle = hamilton_filter(y, h=8, p=4)
print(f"Cycle std: {cycle[~np.isnan(cycle)].std():.4f}")

# Detailed results
result = hamilton_filter_detailed(y, h=8, p=4)
print(f"Coefficients: {result.coefficients}")
print(f"h={result.h}, p={result.p}")
```

::: chronobox.filters.hamilton.hamilton_filter
    options:
      show_root_heading: false
      show_source: true

::: chronobox.filters.hamilton.hamilton_filter_detailed
    options:
      show_root_heading: false
      show_source: true

::: chronobox.filters.hamilton.HamiltonFilterResult
    options:
      show_root_heading: true
      show_source: false

---

## bn_decomposition

Beveridge-Nelson (1981) decomposition into permanent (trend) and transitory (cycle)
components. Based on the Wold representation of an ARIMA process:

$$
\Delta y_t = \mu + \Psi(L) \varepsilon_t
$$

The permanent component is:

$$
\tau_t = \tau_{t-1} + \mu + \Psi(1) \varepsilon_t
$$

where $\Psi(1) = \sum_{j=0}^{\infty} \psi_j$ is the long-run multiplier.

```python
bn_decomposition(
    y: ndarray,
    p: int = 2,
    q: int = 0,
    n_ma_terms: int = 200,
) -> tuple[ndarray, ndarray]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Input time series (should be $I(1)$) |
| `p` | `int` | `2` | AR order for the $\Delta y_t$ process |
| `q` | `int` | `0` | MA order (0 = pure AR approximation) |
| `n_ma_terms` | `int` | `200` | Number of MA($\infty$) terms to approximate |

**Returns**: `(trend, cycle)` — tuple of ndarrays

### bn_decomposition_detailed

Returns a `BNDecompositionResult` dataclass with full estimation details.

```python
bn_decomposition_detailed(
    y: ndarray,
    p: int = 2,
    q: int = 0,
    n_ma_terms: int = 200,
) -> BNDecompositionResult
```

### BNDecompositionResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `trend` | `ndarray` | Permanent (trend) component |
| `cycle` | `ndarray` | Transitory (cyclical) component |
| `psi_one` | `float` | Long-run multiplier $\Psi(1)$ |
| `ar_coeffs` | `ndarray` | Estimated AR coefficients |
| `ma_coeffs` | `ndarray` | Estimated MA coefficients (empty if $q=0$) |
| `drift` | `float` | Estimated drift $\mu$ |
| `residuals` | `ndarray` | Residuals from the ARMA fit |

### Example

```python
import numpy as np
from chronobox.filters import bn_decomposition, bn_decomposition_detailed

# Simulate I(1) process
rng = np.random.default_rng(42)
y = np.cumsum(0.5 + rng.normal(0, 1, 200))

# Simple decomposition
trend, cycle = bn_decomposition(y, p=4)
print(f"Trend range: [{trend.min():.1f}, {trend.max():.1f}]")
print(f"Cycle std: {cycle.std():.4f}")

# Detailed results
result = bn_decomposition_detailed(y, p=4)
print(f"Long-run multiplier Psi(1): {result.psi_one:.4f}")
print(f"Drift: {result.drift:.4f}")
print(f"AR coefficients: {result.ar_coeffs}")
```

::: chronobox.filters.bn.bn_decomposition
    options:
      show_root_heading: false
      show_source: true

::: chronobox.filters.bn.bn_decomposition_detailed
    options:
      show_root_heading: false
      show_source: true

::: chronobox.filters.bn.BNDecompositionResult
    options:
      show_root_heading: true
      show_source: false

---

## Filter Comparison

| Filter | Symmetric | Endpoint Loss | Stationarity Assumption | Best For |
|--------|-----------|---------------|------------------------|----------|
| HP | Yes | No | Level-stationary trend | Quick trend extraction |
| BK | Yes | Yes ($2K$ obs) | Stationary cycle | Clean band-pass isolation |
| CF | No | No | Random walk or drift | Full-sample band-pass |
| Hamilton | No | $h + p - 1$ NaN | None | Robust cycle extraction |
| BN | No | $p$ obs | $I(1)$ input | Permanent-transitory split |

---

## See Also

- [Filters Theory](../theory/filters-theory.md) -- Mathematical background
- [Filters User Guide](../user-guide/filters/index.md) -- Step-by-step guide
- [Decomposition API](decomposition.md) -- STL, X-13, Classical decomposition
