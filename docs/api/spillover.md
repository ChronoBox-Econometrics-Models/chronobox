---
title: "Spillover API"
description: "API reference for SpilloverIndex — Diebold-Yilmaz connectedness and spillover analysis"
---

# Spillover API Reference

!!! info "Module"
    **Import**: `from chronobox.analysis.spillover import SpilloverIndex`
    **Source**: `chronobox/analysis/spillover.py`

## Overview

| Class | Description | Use Case |
|-------|-------------|----------|
| `SpilloverIndex` | Static spillover table (Diebold & Yilmaz, 2012) | Measuring total and directional connectedness |
| `SpilloverResult` | Results from static spillover analysis | Accessing spillover measures and visualization |
| `RollingSpilloverResult` | Time-varying spillover from rolling windows | Tracking connectedness dynamics over time |

---

## SpilloverIndex

Computes the Diebold-Yilmaz (2012) spillover index based on generalized forecast
error variance decompositions (Pesaran & Shin, 1998). The approach:

1. Fits a VAR($p$) to the data
2. Computes generalized FEVD at horizon $H$
3. Normalizes rows to sum to 1
4. Derives total, directional, and net spillover measures

The total spillover index is:

$$
S = \frac{\sum_{i \neq j} \tilde{d}_{ij}^H}{\sum_{i,j} \tilde{d}_{ij}^H} \times 100
$$

where $\tilde{d}_{ij}^H$ is the normalized $H$-step generalized FEVD contribution
of shock $j$ to variable $i$.

```python
SpilloverIndex(
    lags: int = 2,
    horizon: int = 10,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lags` | `int` | `2` | VAR lag order $p$ |
| `horizon` | `int` | `10` | Forecast horizon $H$ for FEVD |

### `.fit()` Method

Compute the static (full-sample) spillover table.

```python
SpilloverIndex.fit(
    data: ndarray,
) -> SpilloverResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `ndarray(T, K)` | *required* | Multivariate time series |

**Returns**: `SpilloverResult`

### `.rolling()` Method

Compute time-varying spillover using a rolling window.

```python
SpilloverIndex.rolling(
    data: ndarray,
    window: int = 200,
) -> RollingSpilloverResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `ndarray(T, K)` | *required* | Multivariate time series |
| `window` | `int` | `200` | Rolling window size |

**Returns**: `RollingSpilloverResult`

---

## SpilloverResult

Results from the static (full-sample) spillover analysis.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `fevd_table` | `ndarray(K, K)` | Normalized generalized FEVD table. `fevd_table[i, j]` = contribution of shock $j$ to FEV of variable $i$ |
| `total_spillover` | `float` | Total spillover index $S$ (percentage) |
| `directional_from` | `ndarray(K,)` | Spillover received by each variable FROM others: $S_{i \leftarrow \bullet} = \sum_{j \neq i} \tilde{d}_{ij}^H$ |
| `directional_to` | `ndarray(K,)` | Spillover transmitted by each variable TO others: $S_{\bullet \leftarrow i} = \sum_{j \neq i} \tilde{d}_{ji}^H$ |
| `net_spillover` | `ndarray(K,)` | Net spillover: $S_i^{\text{net}} = S_{\bullet \leftarrow i} - S_{i \leftarrow \bullet}$ (positive = net transmitter) |
| `pairwise_spillover` | `ndarray(K, K)` | Pairwise net spillover: $S_{ij}^{\text{net}} = \tilde{d}_{ji}^H - \tilde{d}_{ij}^H$ |
| `horizon` | `int` | Forecast horizon $H$ used |
| `var_lags` | `int` | VAR lag order $p$ used |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Formatted spillover table with FROM/TO/NET rows and total index |
| `plot_table()` | | Heatmap visualization of the spillover table |

### Example

```python
import numpy as np
from chronobox.analysis.spillover import SpilloverIndex

# Simulate 4-variable system
rng = np.random.default_rng(42)
T = 500
data = rng.normal(size=(T, 4))
# Add cross-dependencies
data[:, 1] += 0.3 * data[:, 0]
data[:, 2] += 0.2 * data[:, 0] + 0.4 * data[:, 1]
data[:, 3] += 0.1 * data[:, 1]

# Static spillover table
si = SpilloverIndex(lags=2, horizon=10)
result = si.fit(data)

# Total connectedness
print(f"Total spillover index: {result.total_spillover:.2f}%")

# Spillover table
print(result.summary())

# Directional measures
print(f"Spillover FROM others: {result.directional_from}")
print(f"Spillover TO others:   {result.directional_to}")
print(f"Net spillover:         {result.net_spillover}")

# Pairwise: net spillover from variable 0 to variable 1
print(f"Net spillover 0 -> 1: {result.pairwise_spillover[0, 1]:.4f}")

# Heatmap
result.plot_table()
```

::: chronobox.analysis.spillover.SpilloverIndex
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit
        - rolling

::: chronobox.analysis.spillover.SpilloverResult
    options:
      show_root_heading: true
      show_source: false
      members:
        - summary
        - plot_table

---

## RollingSpilloverResult

Results from the rolling-window spillover analysis for tracking time-varying
connectedness.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `total_spillover` | `ndarray(T - window + 1,)` | Time-varying total spillover index |
| `directional_from` | `ndarray(T - window + 1, K)` | Time-varying spillover received by each variable |
| `directional_to` | `ndarray(T - window + 1, K)` | Time-varying spillover transmitted by each variable |
| `net_spillover` | `ndarray(T - window + 1, K)` | Time-varying net spillover per variable |
| `window_size` | `int` | Rolling window size used |
| `dates` | `ndarray \| None` | Date index (if available) |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `plot_total()` | | Plot time-varying total spillover index |

### Example

```python
import numpy as np
from chronobox.analysis.spillover import SpilloverIndex

# Simulate regime-change data
rng = np.random.default_rng(42)
T = 800
data = rng.normal(size=(T, 3))
# Stronger coupling in second half (regime change)
data[400:, 1] += 0.8 * data[400:, 0]
data[400:, 2] += 0.6 * data[400:, 0]

# Rolling spillover with 200-period window
si = SpilloverIndex(lags=2, horizon=10)
rolling = si.rolling(data, window=200)

print(f"Number of rolling windows: {len(rolling.total_spillover)}")
print(f"Spillover range: [{rolling.total_spillover.min():.1f}%, "
      f"{rolling.total_spillover.max():.1f}%]")

# Plot time-varying total spillover
rolling.plot_total()

# Track net transmitter/receiver status over time
for i in range(3):
    net = rolling.net_spillover[:, i]
    pct_transmitter = (net > 0).mean() * 100
    print(f"Variable {i}: net transmitter {pct_transmitter:.0f}% of the time")
```

::: chronobox.analysis.spillover.RollingSpilloverResult
    options:
      show_root_heading: true
      show_source: false
      members:
        - plot_total

---

## See Also

- [Spillover Theory](../theory/spillover-theory.md) -- Diebold-Yilmaz framework and generalized FEVD
- [Spillover User Guide](../user-guide/spillover/index.md) -- Step-by-step spillover guide
- [VAR & VECM API](var.md) -- Underlying VAR estimation
- [FEVD](var.md#forecast-error-variance-decomposition) -- Standard FEVD (Cholesky-based)
