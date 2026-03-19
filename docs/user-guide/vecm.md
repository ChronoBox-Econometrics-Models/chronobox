# VECM - Vector Error Correction Model

## Overview

The VECM is used when variables are cointegrated - they share common
stochastic trends. VECM captures both short-run dynamics and long-run
equilibrium relationships.

## Basic Usage

```python
from chronobox import VECM
from chronobox.datasets import load_dataset

data = load_dataset('cointegrated')
model = VECM(data)
results = model.fit(k_ar_diff=2, coint_rank=1)
print(results.summary())
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Multivariate data (pd.DataFrame) |
| `k_ar_diff` | required | Number of lagged differences |
| `coint_rank` | required | Cointegration rank |
| `deterministic` | `"ci"` | Deterministic terms |

## Cointegration Rank

Use the Johansen test to determine the cointegration rank:

```python
from chronobox.tests_stat import johansen_test

result = johansen_test(data, k_ar_diff=2)
print(result)
```

## Cointegrating Vectors

```python
# Alpha (adjustment coefficients)
print(results.alpha)

# Beta (cointegrating vectors)
print(results.beta)
```

## Forecasting

```python
forecast = results.forecast(steps=12)
```

## Example

```python
from chronobox import VECM
from chronobox.datasets import load_dataset

data = load_dataset('cointegrated')

model = VECM(data)
results = model.fit(k_ar_diff=2, coint_rank=1)

print("Cointegrating vector (beta):")
print(results.beta)

print("Adjustment coefficients (alpha):")
print(results.alpha)

forecast = results.forecast(steps=10)
print(forecast)
```
