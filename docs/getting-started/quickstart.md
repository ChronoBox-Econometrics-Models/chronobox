# Quick Start

## Univariate Analysis

### ARIMA

```python
from chronobox import ARIMA
from chronobox.datasets import load_dataset

airline = load_dataset('airline')
model = ARIMA(order=(0,1,1), seasonal_order=(0,1,1,12))
results = model.fit(airline)
print(results.summary())
forecast = results.forecast(steps=12)
```

### Auto-ARIMA

```python
from chronobox import auto_arima

best = auto_arima(airline, seasonal=True, m=12)
print(best.summary())
```

## Multivariate Analysis

### VAR

```python
from chronobox import VAR
from chronobox.datasets import load_dataset

canada = load_dataset('canada')
model = VAR(canada)
results = model.fit(maxlags=2)
irf = results.irf(periods=20)
```

## Statistical Tests

```python
from chronobox.tests_stat import adf_test, kpss_test

result = adf_test(airline)
print(result)
```

## Filters

```python
from chronobox.filters import hp_filter

trend, cycle = hp_filter(airline, lamb=1600)
```

## CLI

```bash
chronobox estimate arima --order 0,1,1 --data airline
chronobox test adf --data airline
chronobox forecast --model arima --steps 12 --data airline
```
