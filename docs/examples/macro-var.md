# Example: Macroeconomic VAR Analysis

## Overview

VAR analysis of Canadian macroeconomic data with impulse responses
and forecast error variance decomposition.

## Load Data

```python
from chronobox.datasets import load_dataset

canada = load_dataset('canada')
print(canada.columns.tolist())
print(canada.describe())
```

## Step 1: Check Stationarity

```python
from chronobox.tests_stat import adf_test

for col in canada.columns:
    result = adf_test(canada[col])
    print(f"{col}: ADF stat={result.statistic:.3f}, p={result.p_value:.3f}")
```

## Step 2: Fit VAR

```python
from chronobox import VAR

model = VAR(canada)
results = model.fit(maxlags=8, ic='aic')
print(f"Selected lag order: {results.k_ar}")
print(results.summary())
```

## Step 3: Granger Causality

```python
from chronobox.analysis import granger_causality

for caused in canada.columns:
    for causing in canada.columns:
        if caused != causing:
            gc = granger_causality(results, caused=caused, causing=causing)
            if gc.p_value < 0.05:
                print(f"{causing} -> {caused}: p={gc.p_value:.4f} *")
```

## Step 4: Impulse Response Functions

```python
irf = results.irf(periods=20)

from chronobox.visualization import plot_irf
plot_irf(irf)
```

## Step 5: FEVD

```python
fevd = results.fevd(periods=20)

from chronobox.visualization import plot_fevd
plot_fevd(fevd)
```

## Step 6: Forecast

```python
forecast = results.forecast(steps=8)
print(forecast)
```

## Step 7: Spillover Analysis

```python
from chronobox.analysis import SpilloverIndex

si = SpilloverIndex(results, horizon=10)
static = si.compute()
print(f"Total Spillover Index: {static.total_index:.2f}%")

from chronobox.visualization import plot_heatmap
plot_heatmap(static)
```

## Step 8: Generate Report

```python
from chronobox.reports import ReportManager

rm = ReportManager()
report = rm.generate(results, format='html')
report.save('canada_var_report.html')
```
