# ARDL - Autoregressive Distributed Lag

## Overview

ARDL models allow testing for long-run relationships between variables
regardless of whether they are I(0), I(1), or a mixture, using the
Pesaran-Shin-Smith bounds testing approach.

## Basic Usage

```python
from chronobox.models.ardl import ARDL

model = ARDL(endog=y, exog=X, lags=2, order={0: 2, 1: 1})
result = model.fit()
print(result.summary())
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `endog` | Dependent variable (pd.Series) |
| `exog` | Independent variables (pd.DataFrame) |
| `lags` | AR lags for dependent variable |
| `order` | Dict mapping column index to lag order |

## Bounds Test

The Pesaran-Shin-Smith (PSS) bounds test checks for long-run relationships:

```python
from chronobox.tests_stat import bounds_test

bt = bounds_test(result)
print(f"F-statistic: {bt.f_stat}")
print(f"Decision: {bt.decision}")
```

Interpretation:

- F > upper bound: Reject null, long-run relationship exists
- F < lower bound: Cannot reject null
- Between bounds: Inconclusive

## Error Correction Form

If cointegration is found, estimate the ECM:

```python
from chronobox.models.ecm import ECM

ecm = ECM(endog=y, exog=X, lags=2)
ecm_result = ecm.fit()
print(f"Speed of adjustment: {ecm_result.ec_coefficient}")
```

## Long-run Coefficients

```python
print("Long-run coefficients:")
print(result.long_run_coefficients)

print("Short-run coefficients:")
print(result.short_run_coefficients)
```
