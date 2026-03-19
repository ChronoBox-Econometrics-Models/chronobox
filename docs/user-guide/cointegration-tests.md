# Cointegration Tests

## Overview

Cointegration tests determine whether non-stationary series share common
stochastic trends, implying a long-run equilibrium relationship.

## Available Tests

| Test | Description |
|------|-------------|
| Engle-Granger | Two-step residual-based test |
| Phillips-Ouliaris | Residual-based test with PP correction |
| Johansen | System-based test (trace and max eigenvalue) |
| Gregory-Hansen | With structural break |
| Bounds Test (PSS) | For ARDL models |

## Engle-Granger Test

Two-step procedure: (1) estimate long-run regression, (2) test residuals for unit root.

```python
from chronobox.tests_stat import engle_granger_test

result = engle_granger_test(y, X)
print(f"Test statistic: {result.statistic}")
print(f"p-value: {result.p_value}")
print(f"Cointegrated: {result.p_value < 0.05}")
```

## Phillips-Ouliaris Test

```python
from chronobox.tests_stat import phillips_ouliaris_test

result = phillips_ouliaris_test(y, X)
print(result)
```

## Gregory-Hansen Test

Tests for cointegration allowing for a structural break:

```python
from chronobox.tests_stat import gregory_hansen_test

result = gregory_hansen_test(y, X, model='regime')
print(f"Break date: {result.break_date}")
print(f"p-value: {result.p_value}")
```

## Bounds Test (ARDL)

```python
from chronobox.tests_stat import bounds_test

result = bounds_test(ardl_result)
print(f"F-stat: {result.f_stat}")
print(f"Lower bound (5%): {result.lower_bound}")
print(f"Upper bound (5%): {result.upper_bound}")
```

## Workflow

```python
from chronobox.tests_stat import adf_test, engle_granger_test

# Step 1: Check integration order
adf_y = adf_test(y)
adf_x = adf_test(x)

# Step 2: If both I(1), test for cointegration
if adf_y.p_value > 0.05 and adf_x.p_value > 0.05:
    eg = engle_granger_test(y, x)
    if eg.p_value < 0.05:
        print("Cointegrated - use VECM or ECM")
    else:
        print("Not cointegrated - use VAR in differences")
```
