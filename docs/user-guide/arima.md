# ARIMA & SARIMA

## Overview

ARIMA (AutoRegressive Integrated Moving Average) is the foundation
of univariate time series modeling in ChronoBox.

## ARIMA(p,d,q)

```python
from chronobox import ARIMA

model = ARIMA(order=(p, d, q))
results = model.fit(data)
```

### Parameters

- `order`: Tuple (p, d, q) where:
    - `p`: AR order (number of autoregressive terms)
    - `d`: Differencing order
    - `q`: MA order (number of moving average terms)

## SARIMA(p,d,q)(P,D,Q)[m]

```python
model = ARIMA(
    order=(p, d, q),
    seasonal_order=(P, D, Q, m),
)
results = model.fit(data)
```

### Parameters

- `seasonal_order`: Tuple (P, D, Q, m) where:
    - `P`: Seasonal AR order
    - `D`: Seasonal differencing order
    - `Q`: Seasonal MA order
    - `m`: Seasonal period (e.g., 12 for monthly)

## Example: Airline Passengers

```python
from chronobox import ARIMA
from chronobox.datasets import load_dataset

airline = load_dataset('airline')
model = ARIMA(order=(0,1,1), seasonal_order=(0,1,1,12))
results = model.fit(airline)

# Summary
print(results.summary())

# Diagnostics
results.plot_diagnostics()

# Forecast
forecast = results.forecast(steps=12)
```

## Model Selection

Use information criteria to compare models:

```python
print(f"AIC: {results.aic}")
print(f"BIC: {results.bic}")
print(f"AICc: {results.aicc}")
```

## Estimation Method

ChronoBox uses Maximum Likelihood Estimation (MLE) via the Kalman filter
provided by `kalmanbox`. The ARIMA model is cast into state-space form
and the Kalman filter computes the exact likelihood.

## Residual Diagnostics

After fitting, check model adequacy:

```python
from chronobox.tests_stat import ljung_box_test, jarque_bera_test

# Serial correlation
lb = ljung_box_test(results.residuals, lags=10)
print(f"Ljung-Box p-value: {lb.p_value}")

# Normality
jb = jarque_bera_test(results.residuals)
print(f"Jarque-Bera p-value: {jb.p_value}")
```
