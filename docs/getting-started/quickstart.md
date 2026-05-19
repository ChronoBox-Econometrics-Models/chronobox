---
title: Quick Start
description: Fit your first ARIMA model with ChronoBox in 5 minutes
---

# Quick Start

This guide takes you from zero to a working time series model with diagnostics and forecasts --- all in under 10 minutes.

## What You'll Learn

- Load and explore a classic time series dataset
- Test for stationarity with the ADF test
- Fit a seasonal ARIMA model (CSS + MLE estimation)
- Run residual diagnostics (Ljung-Box test)
- Generate forecasts with confidence intervals
- Use `auto_arima` for automatic model selection

---

## Step 1: Load Data

ChronoBox ships with classic econometric datasets. We'll use the **Airline Passengers** dataset: monthly international airline passengers from 1949 to 1960 (144 observations).

```python
from chronobox.datasets import load_dataset

data = load_dataset("airline")
print(f"Shape: {data.shape}")
print(data.head())
```

```text
Shape: (144, 2)
          month  passengers
0    1949-01-01         112
1    1949-02-01         118
2    1949-03-01         132
3    1949-04-01         129
4    1949-05-01         121
```

!!! info "Built-in datasets"
    ChronoBox includes several classic time series datasets for learning and testing:
    `"airline"`, `"canada"`, `"sunspot"`, among others.
    Use `load_dataset(name)` to load any of them as a pandas DataFrame.

---

## Step 2: Explore the Data

Visualize the series and its autocorrelation structure:

```python
from chronobox.visualization import plot_series, plot_acf_pacf

# Plot the raw series
plot_series(data["passengers"], title="Airline Passengers (1949-1960)")
```

The plot reveals two key features:

- **Upward trend**: passengers increase over time
- **Seasonal pattern**: regular peaks every 12 months (summer travel)

Now examine the ACF and PACF of the differenced series:

```python
# ACF/PACF of the original series
plot_acf_pacf(data["passengers"], lags=36)
```

!!! tip "Reading ACF/PACF plots"
    - **ACF** (Autocorrelation Function): slow decay suggests non-stationarity;
      significant spikes at seasonal lags (12, 24, 36) confirm seasonality.
    - **PACF** (Partial Autocorrelation Function): helps identify the AR order ---
      the number of significant lags before cutoff.

---

## Step 3: Test for Stationarity

Before fitting an ARIMA model, check whether the series is stationary using the **Augmented Dickey-Fuller (ADF)** test:

```python
from chronobox.tests_stat import adf_test

result = adf_test(data["passengers"])
print(result)
```

```text
Augmented Dickey-Fuller Test
H0: Series has a unit root (non-stationary)
statistic: 0.815, p-value: 0.9918
Decision: Fail to reject H0 → Series is non-stationary
```

!!! note "Interpreting the ADF test"
    **p > 0.05**: Fail to reject the null --- the series is non-stationary and needs differencing.
    **p <= 0.05**: Reject the null --- the series is stationary.

    The airline series is clearly non-stationary (p = 0.99), confirming we need
    differencing ($d \geq 1$). The seasonal pattern also requires seasonal differencing ($D = 1$).

---

## Step 4: Fit an ARIMA Model

Fit a **SARIMA(0,1,1)(0,1,1)$_{12}$** model --- the classic Box-Jenkins specification for the airline data. ChronoBox estimates parameters in two stages:

1. **CSS** (Conditional Sum of Squares) for initial parameter values
2. **MLE** (Maximum Likelihood Estimation) for final estimates

```python
from chronobox import ARIMA

model = ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
result = model.fit(data["passengers"])
print(result.summary())
```

```text
================================================================================
                         ARIMA Estimation Results
================================================================================
Dependent Variable:          passengers    No. Observations:             144
Model:               ARIMA(0,1,1)(0,1,1)  Estimation Method:        CSS-MLE
                              [12]        Log-Likelihood:          -504.92
Date:                    2026-04-09        AIC:                    1015.84
                                           BIC:                    1024.76
================================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
ma.L1            -0.4018      0.087     -4.614      0.000      -0.573      -0.231
ma.S.L12         -0.5569      0.073     -7.636      0.000      -0.700      -0.414
sigma2          132.3729     10.411     12.716      0.000     112.968     153.778
================================================================================
```

The model specification in mathematical notation:

$$
(1-B)(1-B^{12}) y_t = (1 + \theta_1 B)(1 + \Theta_1 B^{12}) \varepsilon_t
$$

where $B$ is the backshift operator, $\theta_1$ is the MA(1) coefficient, and $\Theta_1$ is the seasonal MA(1) coefficient.

| Parameter | Estimate | Meaning |
|-----------|----------|---------|
| **ma.L1** ($\theta_1$) | -0.4018 | Non-seasonal MA coefficient |
| **ma.S.L12** ($\Theta_1$) | -0.5569 | Seasonal MA coefficient at lag 12 |
| **sigma2** ($\sigma^2$) | 132.37 | Variance of residuals |

!!! info "CSS-MLE estimation"
    ChronoBox first uses **Conditional Sum of Squares** (CSS) to find good starting
    values, then refines them with **Maximum Likelihood Estimation** (MLE). This
    two-stage approach is more robust than MLE alone, especially for seasonal models.

---

## Step 5: Check Diagnostics

Verify that the model residuals behave like white noise:

```python
from chronobox.tests_stat import ljung_box_test

# Ljung-Box test on residuals
lb = ljung_box_test(result.residuals, lags=24)
print(lb)
```

```text
Ljung-Box Test
H0: Residuals are independently distributed (no autocorrelation)
statistic: 19.42, p-value: 0.7312 (lags=24)
Decision: Fail to reject H0 → Residuals are white noise
```

!!! tip "Good diagnostics"
    **Ljung-Box p > 0.05** means we fail to reject the null of no autocorrelation ---
    the residuals are white noise and the model has captured the serial dependence
    in the data. This is what we want to see.

You can also visualize the diagnostics:

```python
result.plot_diagnostics()
```

This produces a panel with four plots:

1. **Standardized residuals** --- should look like white noise
2. **Histogram** --- should approximate a normal distribution
3. **Q-Q plot** --- points should follow the 45-degree line
4. **Correlogram** --- ACF of residuals should show no significant lags

---

## Step 6: Generate Forecasts

Forecast 12 months ahead with 95% confidence intervals:

```python
forecast = result.forecast(steps=12)
print(forecast)
```

```text
            forecast     lower     upper
1961-01-01    419.87    389.52    450.22
1961-02-01    413.45    378.41    448.49
1961-03-01    467.28    427.85    506.71
...
1961-12-01    461.53    399.81    523.25
```

Access forecast components programmatically:

```python
# Point forecasts
print(forecast["forecast"])

# Confidence interval bounds
print(forecast["lower"])   # 95% lower bound
print(forecast["upper"])   # 95% upper bound
```

---

## Step 7: Plot the Forecast

Visualize the forecast alongside the historical data:

```python
result.plot_forecast(forecast, title="Airline Passengers - 12-Month Forecast")
```

This produces a plot with:

- **Historical data** in solid blue
- **Point forecast** as a dashed line
- **95% confidence interval** as a shaded region

!!! note "Confidence intervals widen over time"
    The confidence bands grow wider as the forecast horizon increases. This reflects
    the increasing uncertainty in longer-term predictions --- a fundamental property
    of time series forecasting.

---

## Step 8: Automatic Model Selection

Instead of manually specifying the ARIMA order, use `auto_arima` to search for the best specification:

```python
from chronobox import auto_arima

best = auto_arima(
    data["passengers"],
    seasonal=True,
    m=12,
    information_criterion="aicc",
)
print(best.summary())
```

`auto_arima` performs a stepwise search over $(p,d,q) \times (P,D,Q)_m$ combinations, selecting the model that minimizes the chosen information criterion (AICc by default).

!!! tip "When to use `auto_arima`"
    - **Exploratory analysis**: let `auto_arima` find a good starting point
    - **Production pipelines**: automated model selection for many series
    - **Learning**: compare `auto_arima`'s choice with your manual specification

    For critical research, always validate `auto_arima`'s choice with domain knowledge
    and diagnostic tests.

---

## Complete Script

Here's the full workflow in a single script:

```python
from chronobox import ARIMA, auto_arima
from chronobox.datasets import load_dataset
from chronobox.tests_stat import adf_test, ljung_box_test
from chronobox.visualization import plot_series, plot_acf_pacf

# 1. Load data
data = load_dataset("airline")

# 2. Explore
plot_series(data["passengers"], title="Airline Passengers")
plot_acf_pacf(data["passengers"], lags=36)

# 3. Test stationarity
adf = adf_test(data["passengers"])
print(adf)

# 4. Fit model
model = ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
result = model.fit(data["passengers"])
print(result.summary())

# 5. Diagnostics
lb = ljung_box_test(result.residuals, lags=24)
print(lb)
result.plot_diagnostics()

# 6. Forecast
forecast = result.forecast(steps=12)

# 7. Plot forecast
result.plot_forecast(forecast, title="12-Month Forecast")

# 8. Alternative: automatic selection
best = auto_arima(data["passengers"], seasonal=True, m=12)
print(best.summary())
```

---

## Next Steps

<div class="grid cards" markdown>

- :material-book-open-variant: **[Core Concepts](core-concepts.md)**

    Understand stationarity, lag polynomials, and the ChronoBox workflow

- :material-map-marker-path: **[Choosing a Model](choosing-model.md)**

    Decision guide: ARIMA vs ETS vs VAR and when to use each

- :material-sine-wave: **[ETS Models](../user-guide/ets/index.md)**

    Explore exponential smoothing with 30 model specifications

- :material-chart-timeline-variant: **[VAR Models](../user-guide/var/index.md)**

    Move to multivariate analysis with VAR, VECM, and IRF

</div>
