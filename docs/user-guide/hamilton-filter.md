# Hamilton Filter

## Overview

The Hamilton filter (Hamilton, 2018) provides a regression-based alternative
to the HP filter that avoids spurious cyclicality and endpoint bias.

## Usage

```python
from chronobox.filters import hamilton_filter
from chronobox.datasets import load_dataset

gdp = load_dataset('us_gdp')
trend, cycle = hamilton_filter(gdp, h=8, p=4)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `y` | required | Time series data |
| `h` | `8` | Forecast horizon |
| `p` | `4` | Number of AR lags |

## Recommended Parameters

| Frequency | h | p |
|-----------|---|---|
| Quarterly | 8 | 4 |
| Monthly | 24 | 12 |
| Annual | 2 | 1 |

## Mathematical Formulation

The Hamilton filter estimates:

$$
y_{t+h} = \beta_0 + \beta_1 y_t + \beta_2 y_{t-1} + \cdots + \beta_p y_{t-p+1} + v_{t+h}
$$

- **Trend**: Fitted values $\hat{y}_{t+h}$
- **Cycle**: Residuals $\hat{v}_{t+h}$

## Detailed Output

```python
from chronobox.filters import hamilton_filter_detailed

result = hamilton_filter_detailed(gdp, h=8, p=4)
print(f"R-squared: {result.r_squared}")
print(f"Coefficients: {result.coefficients}")
```

## Advantages over HP Filter

- No spurious cyclicality
- No endpoint bias
- Based on standard regression
- Clear statistical properties
