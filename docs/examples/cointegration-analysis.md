# Example: Cointegration Analysis

## Overview

Testing for cointegration and estimating a VECM for a system of
non-stationary variables.

## Load Data

```python
from chronobox.datasets import load_dataset

data = load_dataset('cointegrated')
print(data.describe())
```

## Step 1: Check Integration Order

```python
from chronobox.tests_stat import adf_test

# Test levels
for col in data.columns:
    result = adf_test(data[col])
    print(f"{col} (level): p={result.p_value:.4f}")

# Test first differences
for col in data.columns:
    result = adf_test(data[col].diff().dropna())
    print(f"{col} (diff): p={result.p_value:.4f}")
```

All variables should be I(1) - non-stationary in levels, stationary in differences.

## Step 2: Engle-Granger Test

```python
from chronobox.tests_stat import engle_granger_test

y = data.iloc[:, 0]
X = data.iloc[:, 1:]

eg = engle_granger_test(y, X)
print(f"Engle-Granger statistic: {eg.statistic:.4f}")
print(f"p-value: {eg.p_value:.4f}")
```

## Step 3: Johansen Test

```python
from chronobox.tests_stat.cointegration import johansen_test

joh = johansen_test(data, k_ar_diff=2)
print(joh)
```

## Step 4: Estimate VECM

```python
from chronobox import VECM

model = VECM(data)
results = model.fit(k_ar_diff=2, coint_rank=1)
print(results.summary())

print("Cointegrating vector (beta):")
print(results.beta)

print("Adjustment coefficients (alpha):")
print(results.alpha)
```

## Step 5: Diagnostics

```python
from chronobox.tests_stat import ljung_box_test

for i, col in enumerate(data.columns):
    lb = ljung_box_test(results.residuals[:, i], lags=10)
    print(f"{col} residuals: Ljung-Box p={lb.p_value:.4f}")
```

## Step 6: IRF from VECM

```python
irf = results.irf(periods=30)

from chronobox.visualization import plot_irf
plot_irf(irf)
```

## Step 7: Forecast

```python
forecast = results.forecast(steps=12)
print(forecast)
```

## Alternative: ARDL Bounds Test

```python
from chronobox.models.ardl import ARDL
from chronobox.tests_stat import bounds_test

y = data.iloc[:, 0]
X = data.iloc[:, 1:]

ardl = ARDL(endog=y, exog=X, lags=2, order={0: 2})
ardl_result = ardl.fit()

bt = bounds_test(ardl_result)
print(f"F-stat: {bt.f_stat:.4f}")
print(f"Decision: {bt.decision}")
```
