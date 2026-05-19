---
title: "General FAQ"
description: "Frequently asked questions about ChronoBox — installation, data formats, model selection, and basic usage."
---

# General FAQ

Common questions for beginners and intermediate users of ChronoBox.

!!! tip "Looking for more?"
    - **Advanced econometrics**: [Advanced FAQ](advanced.md)
    - **Error messages and debugging**: [Troubleshooting](troubleshooting.md)

---

## Getting Started

??? question "How do I install ChronoBox?"

    Install from PyPI:

    ```bash
    pip install chronobox
    ```

    ChronoBox depends on [KalmanBox](https://github.com/NodesEcon/kalmanbox) for state-space estimation. It is installed automatically as a dependency.

    Verify the installation:

    ```python
    import chronobox
    print(chronobox.__version__)
    ```

    For development installations and optional dependencies, see the [Installation Guide](../getting-started/installation.md).

??? question "What Python versions are supported?"

    ChronoBox supports **Python 3.10 and later**. We recommend Python 3.11+ for the best performance, especially with Numba JIT acceleration.

??? question "What is KalmanBox and why is it required?"

    [KalmanBox](https://github.com/NodesEcon/kalmanbox) is a specialized library for state-space models and Kalman filtering, developed by the same NodesEcon team.

    ChronoBox relies on KalmanBox for:

    - **ARIMA estimation** via exact maximum likelihood (Kalman filter)
    - **ETS models** through state-space representation
    - **TVP-VAR** time-varying parameter estimation

    KalmanBox is installed automatically when you install ChronoBox. If you encounter import errors, install it explicitly:

    ```bash
    pip install kalmanbox
    ```

---

## Data Formats

??? question "What data formats does ChronoBox accept?"

    ChronoBox works with several input formats through `TimeSeriesData`:

    === "pandas Series"

        ```python
        import pandas as pd
        from chronobox import TimeSeriesData

        series = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020", periods=3, freq="M"))
        ts = TimeSeriesData(series)
        ```

    === "pandas DataFrame"

        ```python
        import pandas as pd
        from chronobox import TimeSeriesData

        df = pd.DataFrame({"gdp": [100, 102, 105], "inflation": [2.1, 2.3, 2.0]},
                          index=pd.date_range("2020", periods=3, freq="Q"))
        ts = TimeSeriesData(df)
        ```

    === "numpy array"

        ```python
        import numpy as np
        from chronobox import TimeSeriesData

        data = np.array([1.0, 2.0, 3.0, 4.0])
        ts = TimeSeriesData(data)
        ```

    Key requirements:

    - **Numeric data**: values must be `float` or `int`
    - **No missing values** in the estimation sample (handle NaN before fitting)
    - **DatetimeIndex** recommended for time series features (frequency detection, seasonal handling)

??? question "How do I load my own data?"

    Use pandas to read your data, then pass it to ChronoBox:

    ```python
    import pandas as pd
    from chronobox import ARIMA

    # From CSV
    df = pd.read_csv("data.csv", index_col="date", parse_dates=True)
    df.index.freq = "M"  # set frequency if not detected

    # Fit a model
    model = ARIMA(df["gdp"], order=(1, 1, 1))
    result = model.fit()
    ```

    ChronoBox also provides built-in datasets for experimentation:

    ```python
    from chronobox.datasets import load_airline, load_macro

    airline = load_airline()       # univariate monthly data
    macro = load_macro()           # multivariate quarterly data
    ```

??? question "How do I export results?"

    ChronoBox results can be exported in multiple formats:

    ```python
    # Summary table as DataFrame
    summary_df = result.summary()

    # Forecasts as pandas Series/DataFrame
    forecasts = result.forecast(steps=12)

    # Save to CSV
    forecasts.to_csv("forecasts.csv")

    # Generate a full report
    from chronobox.reports import ReportGenerator
    report = ReportGenerator(result)
    report.to_html("report.html")
    report.to_latex("report.tex")
    ```

---

## Model Selection

??? question "What is the difference between ARIMA and ETS?"

    Both are univariate forecasting models, but they differ in philosophy:

    | Feature | ARIMA | ETS |
    |---------|-------|-----|
    | **Approach** | Autoregressive (past values + errors) | Exponential smoothing (weighted averages) |
    | **Stationarity** | Requires differencing ($d > 0$) | Handles trends natively |
    | **Seasonality** | Multiplicative via SARIMA | Additive or multiplicative |
    | **Interpretation** | ACF/PACF driven | Level/trend/seasonal components |
    | **Best for** | Stationary or differenced data | Data with clear trend/seasonal patterns |

    **Rule of thumb**: use Auto-ARIMA and Auto-ETS, then compare AIC or out-of-sample RMSE.

    ```python
    from chronobox.selection import auto_arima, auto_ets

    arima_result = auto_arima(data)
    ets_result = auto_ets(data)

    print(f"ARIMA AIC: {arima_result.aic:.2f}")
    print(f"ETS AIC:   {ets_result.aic:.2f}")
    ```

??? question "How do I choose between CSS and MLE estimation?"

    ChronoBox supports two estimation methods for ARIMA:

    | Method | Description | When to Use |
    |--------|-------------|-------------|
    | **MLE** (Maximum Likelihood) | Exact likelihood via Kalman filter | Default. Most accurate, slower |
    | **CSS** (Conditional Sum of Squares) | Conditional on initial values | Fast approximation, good starting values |

    ```python
    from chronobox import ARIMA

    # Exact MLE (default)
    result_mle = ARIMA(data, order=(1, 1, 1)).fit(method="mle")

    # CSS (faster, approximate)
    result_css = ARIMA(data, order=(1, 1, 1)).fit(method="css")

    # CSS-MLE (CSS for starting values, then MLE refinement)
    result_cssmle = ARIMA(data, order=(1, 1, 1)).fit(method="css-mle")
    ```

    **Recommendation**: use the default `"mle"` unless estimation is too slow or you need quick exploration. CSS can be unreliable with short series or near-unit-root models.

??? question "When should I use VAR vs ARIMA?"

    | Criterion | ARIMA | VAR |
    |-----------|-------|-----|
    | **Number of variables** | 1 (univariate) | 2+ (multivariate) |
    | **Cross-variable dynamics** | No | Yes (Granger causality, IRF) |
    | **Forecasting focus** | Single series | System of series |
    | **Stationarity** | Handles via differencing | Requires stationary data (or use VECM) |

    Use VAR when you want to model **interactions** between variables (e.g., GDP, inflation, interest rate). Use ARIMA when forecasting a **single series** without needing cross-variable dynamics.

---

## Working with Results

??? question "How do I interpret the model summary?"

    Every fitted model returns a results object with a `summary()` method:

    ```python
    result = model.fit()
    print(result.summary())
    ```

    Key elements in the summary:

    - **Coefficients**: estimated parameters with standard errors and p-values
    - **AIC / BIC**: information criteria (lower is better)
    - **Log-likelihood**: maximized log-likelihood value
    - **Residual diagnostics**: Ljung-Box test, Jarque-Bera normality test

    !!! tip
        A p-value < 0.05 suggests the coefficient is statistically significant. Use AIC/BIC for model comparison — they penalize model complexity.

??? question "How do I generate forecasts?"

    ```python
    # Point forecasts
    forecasts = result.forecast(steps=12)

    # Forecasts with confidence intervals
    forecasts_ci = result.forecast(steps=12, alpha=0.05)
    ```

    For visualization:

    ```python
    from chronobox.visualization import plot_forecast
    plot_forecast(result, steps=12)
    ```

---

## Next Steps

- **Advanced topics**: [Advanced FAQ](advanced.md) — cointegration, SVAR, BVAR priors
- **Error resolution**: [Troubleshooting](troubleshooting.md) — convergence errors, data issues
- **Hands-on learning**: [Tutorials](../tutorials/index.md) — end-to-end workflows
