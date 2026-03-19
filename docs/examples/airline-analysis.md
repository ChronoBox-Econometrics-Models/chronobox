# Example: Airline Passengers Analysis

## Overview

Complete Box-Jenkins analysis of the classic airline passengers dataset.

## Load Data

```python
from chronobox.datasets import load_dataset

airline = load_dataset('airline')
print(airline.describe())
```

## Step 1: Visualize

```python
from chronobox.visualization import plot_series

plot_series(airline, title="Monthly Airline Passengers (1949-1960)")
```

## Step 2: Test for Stationarity

```python
from chronobox.tests_stat import adf_test, kpss_test

print("Original series:")
print(adf_test(airline))
print(kpss_test(airline))
```

The series is clearly non-stationary with trend and seasonality.

## Step 3: Differencing

```python
import numpy as np

# Log transform (stabilize variance)
log_airline = np.log(airline)

# First difference + seasonal difference
diff = log_airline.diff(1).diff(12).dropna()

print("After differencing:")
print(adf_test(diff))
```

## Step 4: Fit ARIMA

```python
from chronobox import ARIMA

# Classic airline model: ARIMA(0,1,1)(0,1,1)[12]
model = ARIMA(order=(0,1,1), seasonal_order=(0,1,1,12))
results = model.fit(airline)
print(results.summary())
```

## Step 5: Diagnostics

```python
from chronobox.visualization import plot_diagnostics
from chronobox.tests_stat import ljung_box_test, jarque_bera_test

plot_diagnostics(results.residuals)

lb = ljung_box_test(results.residuals, lags=24)
print(f"Ljung-Box p-value: {lb.p_value}")

jb = jarque_bera_test(results.residuals)
print(f"Jarque-Bera p-value: {jb.p_value}")
```

## Step 6: Auto-ARIMA

```python
from chronobox import auto_arima

best = auto_arima(airline, seasonal=True, m=12)
print(best.summary())
print(f"Selected order: {best.order}")
print(f"Selected seasonal order: {best.seasonal_order}")
```

## Step 7: Forecast

```python
from chronobox.visualization import plot_forecast

forecast = results.forecast(steps=24)
plot_forecast(results, steps=24)
```

## Step 8: Generate Report

```python
from chronobox.reports import ReportManager

rm = ReportManager()
report = rm.generate(results, format='html')
report.save('airline_report.html')
```
