# Time Series Decomposition

## Overview

ChronoBox provides methods to decompose a time series into trend,
seasonal, and residual components.

## STL Decomposition

Seasonal and Trend decomposition using Loess:

```python
from chronobox.decomposition import STL
from chronobox.datasets import load_dataset

airline = load_dataset('airline')

stl = STL(airline, period=12)
result = stl.fit()

print(result.trend)
print(result.seasonal)
print(result.resid)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Time series data |
| `period` | required | Seasonal period |

## Classical Decomposition

Additive or multiplicative decomposition:

```python
from chronobox.decomposition import ClassicalDecomposition

# Additive: Y = T + S + R
decomp = ClassicalDecomposition(airline, period=12, model='additive')
result = decomp.fit()

# Multiplicative: Y = T * S * R
decomp = ClassicalDecomposition(airline, period=12, model='multiplicative')
result = decomp.fit()
```

## Visualization

```python
from chronobox.visualization import plot_decomposition

plot_decomposition(result)
```

## Beveridge-Nelson Decomposition

For non-stationary series, decomposes into permanent and transitory components:

```python
from chronobox.filters import bn_decomposition

permanent, transitory = bn_decomposition(data, ar_order=4)
```
