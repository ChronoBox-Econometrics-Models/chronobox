# Auto-ARIMA

## Overview

Auto-ARIMA automates the model selection process by searching over
ARIMA orders and selecting the best model based on information criteria.

## Basic Usage

```python
from chronobox import auto_arima
from chronobox.datasets import load_dataset

airline = load_dataset('airline')
best = auto_arima(airline, seasonal=True, m=12)
print(best.summary())
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `y` | required | Time series data (pd.Series) |
| `seasonal` | `False` | Include seasonal terms |
| `m` | `1` | Seasonal period |
| `max_p` | `5` | Maximum AR order |
| `max_d` | `2` | Maximum differencing order |
| `max_q` | `5` | Maximum MA order |
| `max_P` | `2` | Maximum seasonal AR order |
| `max_D` | `1` | Maximum seasonal differencing |
| `max_Q` | `2` | Maximum seasonal MA order |
| `information_criterion` | `"aicc"` | Criterion: `"aic"`, `"bic"`, `"aicc"`, `"hqic"` |
| `stepwise` | `True` | Use stepwise algorithm (faster) |

## Stepwise Algorithm

The stepwise algorithm uses a heuristic search instead of exhaustive grid
search. It starts from a base model and explores neighboring orders,
reducing computation from $O(n^6)$ to typically 15-30 model fits.

```python
# Stepwise (default, fast)
best = auto_arima(data, stepwise=True)

# Grid search (exhaustive, slow)
best = auto_arima(data, stepwise=False)
```

## Unit Root Pre-testing

Auto-ARIMA automatically determines the differencing order `d` using
unit root tests (KPSS for stationarity, ADF for unit root):

```python
# Uses KPSS by default
best = auto_arima(data)

# The selected order
print(f"Selected order: {best.order}")
print(f"Selected seasonal order: {best.seasonal_order}")
```

## Example: Seasonal Data

```python
from chronobox import auto_arima
from chronobox.datasets import load_dataset

airline = load_dataset('airline')

# Automatic seasonal ARIMA
best = auto_arima(
    airline,
    seasonal=True,
    m=12,
    information_criterion="aicc",
)

print(best.summary())
forecast = best.forecast(steps=24)
```
