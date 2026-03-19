# HP Filter (Hodrick-Prescott)

## Overview

The Hodrick-Prescott filter decomposes a time series into trend and
cyclical components by minimizing a penalized least squares criterion.

## Usage

```python
from chronobox.filters import hp_filter
from chronobox.datasets import load_dataset

gdp = load_dataset('us_gdp')
trend, cycle = hp_filter(gdp, lamb=1600)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `y` | required | Time series data |
| `lamb` | `1600` | Smoothing parameter ($\lambda$) |

## Smoothing Parameter Guidelines

| Frequency | Recommended $\lambda$ |
|-----------|----------------------|
| Annual | 6.25 |
| Quarterly | 1600 |
| Monthly | 129,600 |

## Mathematical Formulation

The HP filter minimizes:

$$
\min_{\tau} \left\{ \sum_{t=1}^{T}(y_t - \tau_t)^2 + \lambda \sum_{t=2}^{T-1}(\tau_{t+1} - 2\tau_t + \tau_{t-1})^2 \right\}
$$

## Visualization

```python
from chronobox.visualization import plot_decomposition

plot_decomposition(gdp, trend, cycle, title="HP Filter Decomposition")
```

!!! warning
    The HP filter suffers from endpoint bias and can produce spurious
    cycles. Consider the Hamilton filter as an alternative.
