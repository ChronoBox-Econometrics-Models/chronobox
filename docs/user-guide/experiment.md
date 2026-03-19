# ChronoExperiment

## Overview

ChronoExperiment provides a systematic framework for comparing multiple
models through cross-validation, backtesting, and information criteria.

## Basic Usage

```python
from chronobox.experiment import ChronoExperiment
from chronobox import ARIMA, auto_arima
from chronobox.datasets import load_dataset

airline = load_dataset('airline')
exp = ChronoExperiment(airline)
```

## Adding Models

```python
exp.add_model('ARIMA(1,1,1)', ARIMA(order=(1,1,1)))
exp.add_model('ARIMA(0,1,1)x(0,1,1,12)',
              ARIMA(order=(0,1,1), seasonal_order=(0,1,1,12)))
```

## Model Comparison

```python
comparison = exp.compare()
print(comparison.ranking)
print(comparison.metrics)
```

## Cross-validation

```python
cv_result = exp.cross_validate(
    n_splits=5,
    horizon=12,
)
print(cv_result.summary())
```

## Metrics

ChronoExperiment evaluates models on:

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **AIC/BIC**: Information criteria

## Full Workflow

```python
from chronobox.experiment import ChronoExperiment
from chronobox import ARIMA
from chronobox.datasets import load_dataset

airline = load_dataset('airline')
exp = ChronoExperiment(airline)

# Add candidate models
exp.add_model('ARIMA(1,1,1)', ARIMA(order=(1,1,1)))
exp.add_model('ARIMA(0,1,1)', ARIMA(order=(0,1,1)))
exp.add_model('ARIMA(2,1,1)', ARIMA(order=(2,1,1)))

# Compare
comparison = exp.compare()
print(comparison.ranking)

# Cross-validate best
cv = exp.cross_validate(n_splits=5, horizon=12)
print(cv.summary())
```
