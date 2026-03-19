# VAR - Vector Autoregression

## Overview

VAR models capture linear interdependencies among multiple time series.
Each variable is modeled as a linear function of its own past values
and the past values of all other variables.

## Basic Usage

```python
from chronobox import VAR
from chronobox.datasets import load_dataset

canada = load_dataset('canada')
model = VAR(canada)
results = model.fit(maxlags=2)
print(results.summary())
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Multivariate time series (pd.DataFrame) |
| `maxlags` | `None` | Maximum lags to consider |
| `ic` | `"aic"` | Information criterion for lag selection |
| `trend` | `"c"` | `"n"` (none), `"c"` (constant), `"ct"` (constant+trend) |

## Lag Selection

```python
model = VAR(data)
results = model.fit(maxlags=8, ic='aic')
print(f"Selected lag order: {results.k_ar}")
```

## Impulse Response Functions (IRF)

```python
irf = results.irf(periods=20)

from chronobox.visualization import plot_irf
plot_irf(irf)
```

## Forecast Error Variance Decomposition (FEVD)

```python
fevd = results.fevd(periods=20)

from chronobox.visualization import plot_fevd
plot_fevd(fevd)
```

## Granger Causality

```python
from chronobox.analysis import granger_causality

gc = granger_causality(results, caused='gdp', causing='inflation')
print(f"F-statistic: {gc.f_stat}")
print(f"p-value: {gc.p_value}")
```

## Forecasting

```python
forecast = results.forecast(steps=8)
print(forecast)
```

## Example: Canadian Macro Data

```python
from chronobox import VAR
from chronobox.datasets import load_dataset
from chronobox.visualization import plot_irf, plot_fevd

canada = load_dataset('canada')

model = VAR(canada)
results = model.fit(maxlags=4, ic='aic')
print(results.summary())

# IRF analysis
irf = results.irf(periods=20)
plot_irf(irf)

# FEVD analysis
fevd = results.fevd(periods=20)
plot_fevd(fevd)
```
