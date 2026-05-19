---
title: "ETS Benchmark"
description: "Performance comparison of ChronoBox ETS vs statsmodels ExponentialSmoothing vs R forecast::ets — estimation time, model selection, and forecast accuracy."
---

# ETS Benchmark

Comparison of Exponential Smoothing / ETS implementations across Python and R.

| Library | Class / Function | Framework |
|---------|-----------------|-----------|
| **ChronoBox** | `ETS` / `auto_ets` | State-space MLE via KalmanBox |
| **statsmodels** | `ExponentialSmoothing` | Direct MLE optimization |
| **R forecast** | `ets` | State-space MLE (C++ backend) |

!!! info "ETS notation"
    ETS(**E**rror, **T**rend, **S**eason) follows Hyndman et al. (2008):

    - **Error**: A (additive) or M (multiplicative)
    - **Trend**: N (none), A (additive), Ad (damped additive)
    - **Season**: N (none), A (additive), M (multiplicative)

---

## Dataset: Airline Passengers

Monthly international airline passengers ($T = 144$, 1949–1960).

### Model: ETS(M,Ad,M) — Multiplicative Error, Damped Trend, Multiplicative Season

```python
from chronobox import ETS
from chronobox.datasets import load_airline

data = load_airline()
result = ETS(data, error="mul", trend="add", seasonal="mul", damped=True).fit()
```

```r
# R
library(forecast)
fit <- ets(AirPassengers, model="MAM", damped=TRUE)
```

**Results:**

| Metric | ChronoBox | statsmodels | R forecast |
|--------|-----------|-------------|------------|
| **Estimation time** | 25 ms | 72 ms | 20 ms |
| **AIC** | 1380.2 | 1382.5 | 1380.1 |
| **BIC** | 1429.8 | 1432.1 | 1429.7 |
| **$\alpha$ (level)** | 0.152 | 0.155 | 0.152 |
| **$\beta$ (trend)** | 0.032 | 0.034 | 0.031 |
| **$\gamma$ (season)** | 0.081 | 0.085 | 0.080 |
| **$\phi$ (damping)** | 0.979 | 0.978 | 0.979 |

!!! note
    ChronoBox and R produce nearly identical results due to similar state-space formulations. statsmodels uses a different optimization approach, leading to slightly different parameter estimates and higher AIC.

### Forecast Accuracy (h=12, rolling origin)

| Metric | ChronoBox | statsmodels | R forecast |
|--------|-----------|-------------|------------|
| **RMSE** | 15.1 | 16.8 | 14.9 |
| **MAE** | 12.0 | 13.5 | 11.8 |
| **MAPE** | 3.3% | 3.7% | 3.2% |

---

## Dataset: M3 Competition Subset

100 monthly series from the M3 Competition, fitted with automatic model selection.

### Auto-ETS Performance

| Metric | ChronoBox | statsmodels | R forecast |
|--------|-----------|-------------|------------|
| **Avg estimation time** | 0.5 s | 1.8 s | 0.4 s |
| **Median RMSE** | 1.48 | 1.62 | 1.45 |
| **Median MAPE** | 13.2% | 14.1% | 12.9% |

### Model Selection Agreement

How often does each library select the same ETS specification?

| Comparison | Agreement rate |
|-----------|---------------|
| ChronoBox vs R forecast | 82% |
| ChronoBox vs statsmodels | 71% |
| statsmodels vs R forecast | 68% |

!!! tip
    ChronoBox's `auto_ets` agrees with R's `ets()` model selection 82% of the time. When they disagree, the models are usually close alternatives (e.g., damped vs undamped trend).

### Speed Distribution (auto_ets)

| Percentile | ChronoBox | statsmodels | R forecast |
|-----------|-----------|-------------|------------|
| 25th | 0.2 s | 0.8 s | 0.2 s |
| 50th | 0.4 s | 1.5 s | 0.3 s |
| 75th | 0.7 s | 2.5 s | 0.5 s |
| 95th | 1.5 s | 5.0 s | 1.2 s |

---

## Dataset: M4 Competition Subset

50 monthly and 50 quarterly series from the M4 Competition.

### Monthly Series (h=18)

| Metric | ChronoBox | statsmodels | R forecast |
|--------|-----------|-------------|------------|
| **Median sMAPE** | 12.1% | 13.0% | 11.8% |
| **Median MASE** | 1.05 | 1.12 | 1.02 |
| **Avg time per series** | 0.6 s | 2.0 s | 0.5 s |

### Quarterly Series (h=8)

| Metric | ChronoBox | statsmodels | R forecast |
|--------|-----------|-------------|------------|
| **Median sMAPE** | 9.8% | 10.5% | 9.5% |
| **Median MASE** | 0.95 | 1.01 | 0.92 |
| **Avg time per series** | 0.4 s | 1.2 s | 0.3 s |

---

## ETS Component Comparison

Detailed comparison of all ETS model types on the airline dataset:

| Model | ChronoBox AIC | statsmodels AIC | R AIC | Best |
|-------|-------------|-----------------|-------|------|
| ETS(A,N,N) | 1510.2 | 1510.3 | 1510.2 | Tie |
| ETS(A,A,N) | 1465.8 | 1466.1 | 1465.7 | R |
| ETS(A,Ad,N) | 1462.3 | 1462.8 | 1462.2 | R |
| ETS(A,A,A) | 1404.5 | 1405.2 | 1404.4 | R |
| ETS(M,A,M) | 1383.1 | 1385.8 | 1382.9 | R |
| ETS(M,Ad,M) | 1380.2 | 1382.5 | 1380.1 | R |

!!! note
    AIC differences between ChronoBox and R are consistently small (< 0.5 units), well within the range of optimizer tolerance. statsmodels shows larger differences due to its different parameterization.

---

## Initialization Methods

ETS models require initial state values. Different libraries handle this differently:

| Library | Default initialization | Alternatives |
|---------|----------------------|-------------|
| **ChronoBox** | Optimized (MLE) | Heuristic, fixed |
| **statsmodels** | Optimized | Heuristic, known |
| **R forecast** | Optimized (default) | Heuristic |

Impact of initialization on the airline dataset (ETS(M,Ad,M)):

| Initialization | ChronoBox AIC | R AIC |
|---------------|-------------|-------|
| Optimized | 1380.2 | 1380.1 |
| Heuristic | 1382.8 | 1382.5 |

---

## Reproducing the Benchmarks

```python
import time
import numpy as np
from chronobox import ETS
from chronobox.selection import auto_ets
from chronobox.datasets import load_airline

data = load_airline()

# Single model timing
times = []
for _ in range(10):
    start = time.perf_counter()
    result = ETS(data, error="mul", trend="add", seasonal="mul", damped=True).fit()
    times.append(time.perf_counter() - start)

print(f"Median time: {np.median(times)*1000:.1f} ms")
print(f"AIC: {result.aic:.1f}")

# Auto-ETS
start = time.perf_counter()
auto_result = auto_ets(data)
elapsed = time.perf_counter() - start
print(f"auto_ets: {elapsed:.2f}s, model={auto_result.model_type}, AIC={auto_result.aic:.1f}")
```

```r
# R equivalent
library(forecast)
library(microbenchmark)

data <- AirPassengers

mb <- microbenchmark(
  ets(data, model="MAM", damped=TRUE),
  times=10
)
print(mb)

# Auto-ETS
fit <- ets(data)
print(fit)
```

---

## See Also

- [ARIMA Benchmark](arima.md) — univariate ARIMA/SARIMA comparison
- [VAR Benchmark](var.md) — multivariate model comparison
- [Feature Comparison](comparison.md) — full feature matrix
