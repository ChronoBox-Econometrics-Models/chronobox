---
title: Core Concepts
description: Understand TimeSeriesData, Results objects, CSS vs MLE estimation, state-space models, and the ChronoBox workflow
---

# Core Concepts

ChronoBox is designed around a small set of composable abstractions that stay consistent across all model families --- from univariate ARIMA to multivariate SVAR. Understanding these building blocks once means you can work with any model in the library without re-learning the API.

This page introduces the five ideas you need before diving into specific models: how data is represented, what comes back from estimation, how parameters are estimated, how state-space models power the engine, and the typical workflow that ties everything together.

---

## TimeSeriesData

`TimeSeriesData` is the central data container in ChronoBox. It wraps a numeric array together with temporal metadata --- dates, frequency, and a human-readable name --- so that every downstream operation (differencing, estimation, forecasting) can reason about the time dimension automatically.

### Creating from Different Sources

=== "NumPy Array"

    ```python
    import numpy as np
    from chronobox import TimeSeriesData

    values = np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119])

    ts = TimeSeriesData(
        values=values,
        frequency="M",
        start_date="1949-01-01",
        name="passengers",
    )
    print(ts)
    ```

    ```text
    TimeSeriesData: passengers
    Frequency: M | Observations: 10
    Start: 1949-01-01 | End: 1949-10-01
    ```

=== "pandas Series"

    ```python
    import pandas as pd
    from chronobox import TimeSeriesData

    s = pd.Series(
        [112, 118, 132, 129, 121],
        index=pd.date_range("1949-01", periods=5, freq="MS"),
        name="passengers",
    )

    ts = TimeSeriesData.from_pandas(s)
    ```

=== "pandas DataFrame"

    ```python
    import pandas as pd
    from chronobox import TimeSeriesData

    df = pd.DataFrame({
        "date": pd.date_range("1949-01", periods=5, freq="MS"),
        "passengers": [112, 118, 132, 129, 121],
    })

    ts = TimeSeriesData.from_pandas(df["passengers"])
    ```

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `values` | `np.ndarray` | The raw numeric array |
| `dates` | `pd.DatetimeIndex` | Datetime index for each observation |
| `frequency` | `str` | Frequency code (`"M"`, `"Q"`, `"A"`, `"W"`, `"D"`) |
| `name` | `str` | Human-readable label for the series |
| `nobs` | `int` | Number of observations |

!!! note "Automatic frequency detection"
    When creating from a pandas object with a `DatetimeIndex`, ChronoBox infers the frequency automatically. You can always override it with the `frequency` parameter.

---

## Model and Results

All ChronoBox models follow the same two-step pattern: **configure** then **fit**.

```python
from chronobox import ARIMA

# Step 1: Configure the model
model = ARIMA(order=(1, 1, 1), seasonal_order=(0, 1, 1, 12))

# Step 2: Fit to data
result = model.fit(data)
```

The `result` object is the gateway to everything you need after estimation: parameter values, diagnostics, forecasts, and reports.

### Results Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `np.ndarray` | Estimated model parameters |
| `loglik` | `float` | Maximized log-likelihood |
| `aic` | `float` | Akaike Information Criterion |
| `bic` | `float` | Bayesian Information Criterion |
| `aicc` | `float` | Corrected AIC (small-sample adjustment) |
| `hqc` | `float` | Hannan-Quinn Criterion |
| `residuals` | `np.ndarray` | Model residuals |
| `nobs` | `int` | Number of observations used |

### Results Methods

```python
# Full estimation summary table
print(result.summary())

# Out-of-sample forecast with confidence intervals
forecast = result.forecast(steps=12)

# Diagnostic plots (residuals, ACF, Q-Q, histogram)
result.plot_diagnostics()
```

### Information Criteria

All criteria balance goodness-of-fit against model complexity. Lower values indicate a better model.

| Criterion | Formula | Penalty Strength |
|-----------|---------|-----------------|
| **AIC** | $\text{AIC} = -2\ell(\hat{\theta}) + 2k$ | Light |
| **AICc** | $\text{AICc} = \text{AIC} + \dfrac{2k(k+1)}{T - k - 1}$ | Moderate (adjusts for small $T$) |
| **BIC** | $\text{BIC} = -2\ell(\hat{\theta}) + k \ln T$ | Heavy |
| **HQC** | $\text{HQC} = -2\ell(\hat{\theta}) + 2k \ln(\ln T)$ | Between AIC and BIC |

Where $\ell(\hat{\theta})$ is the maximized log-likelihood, $k$ is the number of estimated parameters, and $T$ is the number of observations.

!!! tip "Which criterion to use?"
    - **AICc** is the default in ChronoBox and the best general choice --- it equals AIC as $T \to \infty$ but corrects for overfitting in small samples.
    - **BIC** is more conservative and preferred when you want a parsimonious model.
    - **HQC** offers a middle ground, often used in VAR lag selection.

---

## CSS vs MLE Estimation

ChronoBox supports two estimation methods. Understanding the difference helps you choose the right one --- and diagnose problems when convergence fails.

### Conditional Sum of Squares (CSS)

CSS minimizes the sum of squared residuals, conditioning on initial values:

$$
S(\theta) = \sum_{t=1}^{T} e_t^2(\theta)
$$

where $e_t(\theta)$ are the one-step-ahead prediction errors given parameters $\theta$. CSS is **fast** because it avoids computing the full likelihood, but it **ignores the contribution of initial observations** and does not produce a proper likelihood value.

### Maximum Likelihood Estimation (MLE)

MLE maximizes the exact log-likelihood, computed via the Kalman filter:

$$
\ell(\theta) = -\frac{T}{2}\log(2\pi) - \frac{1}{2}\sum_{t=1}^{T} \left( \log f_t + \frac{v_t^2}{f_t} \right)
$$

where $v_t$ is the innovation (prediction error) and $f_t$ is the innovation variance at time $t$, both obtained from the Kalman filter recursions. MLE produces **exact** parameter estimates, proper standard errors, and a valid log-likelihood for information criteria.

### Comparison

| | CSS | MLE |
|---|-----|-----|
| **Speed** | Fast | Slower |
| **Accuracy** | Approximate | Exact |
| **Log-likelihood** | Not available | Exact |
| **Standard errors** | Approximate | Exact (from observed information) |
| **Information criteria** | Not reliable | Valid AIC, BIC, AICc, HQC |
| **Best for** | Quick exploration, initial values | Final estimation, model comparison |

### CSS-MLE: The Default Strategy

ChronoBox uses a **two-stage** approach by default:

```python
# Default: CSS for initialization, then MLE for final estimates
result = model.fit(data)

# Force CSS-only (faster, less accurate)
result = model.fit(data, method="css")

# Force MLE-only (may struggle without good starting values)
result = model.fit(data, method="mle")
```

!!! warning "When CSS fails"
    CSS can produce poor estimates when the model has **near-unit roots** (AR or MA coefficients close to 1) or when the series is **short** ($T < 50$). In these cases, the initial-value approximation breaks down. If CSS-initialized MLE fails to converge, try:

    1. Increasing `maxiter` (e.g., `model.fit(data, maxiter=500)`)
    2. Using MLE directly with manual starting values
    3. Simplifying the model (reduce $p$, $q$, $P$, or $Q$)

---

## State-Space and kalmanbox

Under the hood, ChronoBox represents every model as a **state-space system** and delegates the heavy numerical work to **kalmanbox** --- a dedicated Kalman filter library optimized for time series econometrics.

### The State-Space Representation

Any linear time series model can be written in state-space form with two equations:

**Observation equation:**

$$
y_t = Z_t \, \alpha_t + \varepsilon_t, \qquad \varepsilon_t \sim N(0, H_t)
$$

**State transition equation:**

$$
\alpha_{t+1} = T_t \, \alpha_t + R_t \, \eta_t, \qquad \eta_t \sim N(0, Q_t)
$$

| Symbol | Meaning |
|--------|---------|
| $y_t$ | Observed value at time $t$ |
| $\alpha_t$ | Unobserved state vector |
| $Z_t$ | Observation (design) matrix |
| $T_t$ | State transition matrix |
| $R_t$ | State noise selection matrix |
| $H_t$ | Observation noise covariance |
| $Q_t$ | State noise covariance |

!!! info "Why state-space?"
    ARIMA, ETS, and structural time series models are all special cases of the state-space form. By casting every model into this unified framework, ChronoBox can use a single estimation engine (the Kalman filter) for all model families.

### How kalmanbox Computes the Likelihood

The Kalman filter processes observations one at a time, producing at each step:

1. **Prediction**: forecast the next state and observation
2. **Innovation**: compute the prediction error $v_t = y_t - Z_t \, a_{t|t-1}$
3. **Update**: revise the state estimate using the new observation

The log-likelihood is accumulated from the innovations:

$$
\ell(\theta) = -\frac{T}{2}\log(2\pi) - \frac{1}{2}\sum_{t=1}^{T} \left( \log f_t + \frac{v_t^2}{f_t} \right)
$$

ChronoBox then uses a numerical optimizer (L-BFGS-B) to find the parameters $\theta$ that maximize this likelihood.

### Conceptual Flow

```text
                        ChronoBox                              kalmanbox
                  ┌─────────────────────┐              ┌─────────────────────┐
                  │                     │              │                     │
  User defines    │  Model spec (p,d,q) │   maps to    │  State-space form   │
  ARIMA(1,1,1) ──►│  + data             │─────────────►│  Z, T, R, H, Q      │
                  │                     │              │                     │
                  └─────────┬───────────┘              └─────────┬───────────┘
                            │                                    │
                            │  optimizer                         │  Kalman filter
                            │  adjusts θ                         │  recursions
                            │                                    │
                  ┌─────────▼───────────┐              ┌─────────▼───────────┐
                  │                     │   receives    │                     │
                  │  Results object     │◄─────────────│  log-likelihood     │
                  │  params, AIC, BIC   │              │  innovations, MSE   │
                  │                     │              │                     │
                  └─────────────────────┘              └─────────────────────┘
```

This separation of concerns means ChronoBox focuses on model specification and user experience, while kalmanbox handles the numerically intensive filtering and smoothing.

---

## The ChronoBox Workflow

Every time series analysis in ChronoBox follows the same eight-step workflow, regardless of the model family:

```text
  Load ─► Explore ─► Test ─► Model ─► Fit ─► Diagnose ─► Forecast ─► Report
```

### Step by Step

| Step | Action | ChronoBox Tools |
|------|--------|----------------|
| **1. Load** | Import your data | `load_dataset()`, `TimeSeriesData.from_pandas()` |
| **2. Explore** | Visualize the series, ACF/PACF | `plot_series()`, `plot_acf_pacf()` |
| **3. Test** | Check stationarity, unit roots | `adf_test()`, `kpss_test()`, `pp_test()` |
| **4. Model** | Choose and configure the model | `ARIMA()`, `ETS()`, `VAR()`, ... |
| **5. Fit** | Estimate parameters | `model.fit(data)` |
| **6. Diagnose** | Validate residuals and stability | `ljung_box_test()`, `plot_diagnostics()` |
| **7. Forecast** | Generate out-of-sample predictions | `result.forecast(steps=h)` |
| **8. Report** | Summarize and export | `result.summary()`, `plot_forecast()` |

```python
from chronobox import ARIMA
from chronobox.datasets import load_dataset
from chronobox.tests_stat import adf_test, ljung_box_test
from chronobox.visualization import plot_series, plot_acf_pacf

# 1. Load
data = load_dataset("airline")

# 2. Explore
plot_series(data["passengers"])
plot_acf_pacf(data["passengers"], lags=36)

# 3. Test
adf = adf_test(data["passengers"])
print(adf)

# 4-5. Model & Fit
model = ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
result = model.fit(data["passengers"])

# 6. Diagnose
lb = ljung_box_test(result.residuals, lags=24)
print(lb)
result.plot_diagnostics()

# 7. Forecast
forecast = result.forecast(steps=12)

# 8. Report
print(result.summary())
result.plot_forecast(forecast)
```

!!! tip "This pattern scales"
    The same eight steps apply whether you are fitting a simple ARIMA, a seasonal ETS, or a multivariate SVAR. The function names change, but the workflow stays the same.

---

## Next Steps

<div class="grid cards" markdown>

- :material-map-marker-path: **[Choosing a Model](choosing-model.md)**

    Decision guide for selecting the right model family for your data

- :material-sine-wave: **[ARIMA Models](../user-guide/arima/index.md)**

    Univariate modeling with ARIMA, SARIMA, ARFIMA, and Auto-ARIMA

- :material-chart-timeline-variant: **[ETS Models](../user-guide/ets/index.md)**

    Exponential smoothing with 30 model specifications

- :material-vector-polyline: **[VAR Models](../user-guide/var/index.md)**

    Multivariate analysis with VAR, VECM, and impulse responses

</div>
