---
title: "Decomposition API"
description: "API reference for STL, X13Wrapper, and ClassicalDecomposition — time series decomposition methods"
---

# Decomposition API Reference

!!! info "Module"
    **Import**: `from chronobox import STL, ClassicalDecomposition`
    **Source**: `chronobox/decomposition/`

## Overview

| Class | Description | Use Case |
|-------|-------------|----------|
| `STL` | Seasonal-Trend decomposition using LOESS (Cleveland et al. 1990) | Robust, flexible decomposition |
| `X13Wrapper` | X-13 ARIMA-SEATS wrapper (US Census Bureau) | Official seasonal adjustment |
| `ClassicalDecomposition` | Classical additive/multiplicative decomposition | Simple baseline decomposition |

All decomposition methods return a `DecompositionResult` containing the three components:

$$y_t = T_t + S_t + R_t \quad \text{(additive)}$$

$$y_t = T_t \times S_t \times R_t \quad \text{(multiplicative)}$$

---

## STL

Seasonal and Trend decomposition using LOESS. Robust, iterative method that separates
a time series into trend, seasonal, and remainder components.

```python
STL(
    period: int,
    seasonal: int = 7,
    trend: int | None = None,
    low_pass: int | None = None,
    robust: bool = False,
    seasonal_deg: int = 1,
    trend_deg: int = 1,
    low_pass_deg: int = 1,
    n_inner: int = 2,
    n_outer: int = 0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | `int` | *required* | Seasonal period (e.g., 12 for monthly data) |
| `seasonal` | `int` | `7` | LOESS window for seasonal extraction. Must be odd $\geq 7$ |
| `trend` | `int \| None` | `None` | LOESS window for trend. Default computed from `period` and `seasonal` |
| `low_pass` | `int \| None` | `None` | LOESS window for low-pass filter. Default: `period` (made odd) |
| `robust` | `bool` | `False` | Use robustness iterations to downweight outliers |
| `seasonal_deg` | `int` | `1` | LOESS polynomial degree for seasonal (0 or 1) |
| `trend_deg` | `int` | `1` | LOESS polynomial degree for trend (0 or 1) |
| `low_pass_deg` | `int` | `1` | LOESS polynomial degree for low-pass (0 or 1) |
| `n_inner` | `int` | `2` | Number of inner loop iterations |
| `n_outer` | `int` | `0` | Number of outer robustness iterations (0 = non-robust) |

### `.fit()` Method

```python
STL.fit(endog: array-like) -> DecompositionResult
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `endog` | `array-like` | Time series data (must have $\geq 2 \times \text{period}$ observations) |

**Returns**: `DecompositionResult`

!!! tip "Choosing the seasonal window"
    Larger values of `seasonal` produce smoother seasonal components. The default of 7
    works well for most monthly and quarterly data. Increase for noisy data, decrease
    for rapidly changing seasonality.

### Example

```python
import numpy as np
from chronobox import STL

# Generate seasonal data
rng = np.random.default_rng(42)
t = np.arange(144)
y = 100 + 0.5 * t + 20 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 5, 144)

# Standard STL decomposition
stl = STL(period=12)
result = stl.fit(y)
print(result.summary())

# Access components
print(result.trend[:5])
print(result.seasonal[:12])
print(result.remainder[:5])

# Robust STL for data with outliers
stl_robust = STL(period=12, robust=True, n_outer=5)
result = stl_robust.fit(y_with_outliers)
```

::: chronobox.decomposition.stl.STL
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit

---

## X-13 ARIMA-SEATS {#x-13-arima-seats}

Wrapper for the X-13 ARIMA-SEATS seasonal adjustment program from the US Census Bureau.

!!! warning "External dependency"
    Requires the `x13as` executable to be installed and accessible on your system.
    Download from: [census.gov/data/software/x13as.html](https://www.census.gov/data/software/x13as.html)

```python
X13Wrapper(
    x13_path: str | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x13_path` | `str \| None` | `None` | Path to the X-13 executable. Auto-detected if None |

### `.seasonal_adjust()` Method

```python
X13Wrapper.seasonal_adjust(
    endog: array-like,
    period: int = 12,
    model: str = "auto",
) -> DecompositionResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `array-like` | *required* | Time series data |
| `period` | `int` | `12` | Seasonal period |
| `model` | `str` | `"auto"` | ARIMA model for RegARIMA (`"auto"` for automatic selection) |

**Returns**: `DecompositionResult`

### Example

```python
from chronobox.decomposition.x13_wrapper import X13Wrapper

# Auto-detect x13as executable
x13 = X13Wrapper()

# Seasonal adjustment
result = x13.seasonal_adjust(monthly_data, period=12)
print(result.summary())

# Specify path to executable
x13 = X13Wrapper(x13_path="/usr/local/bin/x13as")
result = x13.seasonal_adjust(y, period=12)
```

::: chronobox.decomposition.x13_wrapper.X13Wrapper
    options:
      show_root_heading: false
      show_source: true
      members:
        - seasonal_adjust

---

## Classical Decomposition {#classical-decomposition}

Classical decomposition using moving average for trend extraction.

```python
ClassicalDecomposition(
    period: int,
    model: str = "additive",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | `int` | *required* | Seasonal period |
| `model` | `str` | `"additive"` | `'additive'` or `'multiplicative'` |

### `.fit()` Method

```python
ClassicalDecomposition.fit(endog: array-like) -> DecompositionResult
```

### Example

```python
from chronobox import ClassicalDecomposition

# Additive decomposition
decomp = ClassicalDecomposition(period=12, model="additive")
result = decomp.fit(monthly_data)
print(result.summary())

# Multiplicative decomposition (for data with proportional seasonality)
decomp_mult = ClassicalDecomposition(period=4, model="multiplicative")
result = decomp_mult.fit(quarterly_data)
```

::: chronobox.decomposition.classical.ClassicalDecomposition
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit

---

## DecompositionResult

Shared result class for all decomposition methods.

| Attribute | Type | Description |
|-----------|------|-------------|
| `observed` | `ndarray` | Original series |
| `trend` | `ndarray` | Trend component $T_t$ |
| `seasonal` | `ndarray` | Seasonal component $S_t$ |
| `remainder` | `ndarray` | Remainder component $R_t$ |
| `weights` | `ndarray \| None` | Robustness weights (STL only, None if non-robust) |
| `period` | `int` | Seasonal period |
| `model` | `str` | `'additive'` or `'multiplicative'` |

| Method | Description |
|--------|-------------|
| `summary()` | Formatted summary with component statistics |

::: chronobox.decomposition.stl.DecompositionResult
    options:
      show_root_heading: false
      show_source: false
      members:
        - summary

---

## See Also

- [Core API](core.md) -- Foundation classes
- [ETS API](ets.md) -- ETS models with built-in trend/seasonal components
- [Filter API](filters.md) -- Economic filters (HP, BK, CF, Hamilton)
- [STL User Guide](../user-guide/decomposition/stl.md) -- Step-by-step STL guide
- [X-13 User Guide](../user-guide/decomposition/x13.md) -- X-13 ARIMA-SEATS guide
