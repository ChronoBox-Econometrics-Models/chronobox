---
title: "Troubleshooting"
description: "Solutions for common ChronoBox errors — convergence failures, import errors, data issues, and performance problems."
---

# Troubleshooting

Practical solutions for the most common issues encountered when using ChronoBox.

!!! tip "Looking for conceptual answers?"
    - **Basic usage**: [General FAQ](general.md)
    - **Advanced econometrics**: [Advanced FAQ](advanced.md)

---

## Convergence Errors

??? question "ConvergenceWarning: MLE optimization did not converge"

    **Cause**: The likelihood optimizer failed to find a maximum within the iteration limit.

    **Solutions** (try in order):

    1. **Use CSS-MLE** to get better starting values:

        ```python
        result = ARIMA(data, order=(2, 1, 2)).fit(method="css-mle")
        ```

    2. **Increase iterations**:

        ```python
        result = ARIMA(data, order=(2, 1, 2)).fit(maxiter=500)
        ```

    3. **Simplify the model** — reduce AR or MA order:

        ```python
        result = ARIMA(data, order=(1, 1, 1)).fit()
        ```

    4. **Rescale the data** — large values create numerical issues:

        ```python
        import numpy as np
        result = ARIMA(np.log(data), order=(1, 1, 1)).fit()
        ```

    5. **Let auto_arima search** for a well-behaved specification:

        ```python
        from chronobox.selection import auto_arima
        result = auto_arima(data, max_p=5, max_q=5)
        ```

??? question "LinAlgError: Singular matrix in VAR estimation"

    **Cause**: The data matrix is rank-deficient — often due to collinear or constant variables.

    **Solutions:**

    1. **Check for constant columns**:

        ```python
        print(data.std())
        # Drop columns with zero variance
        data = data.loc[:, data.std() > 0]
        ```

    2. **Check for perfect collinearity**:

        ```python
        import numpy as np
        corr = data.corr()
        high_corr = (corr.abs() > 0.99) & (corr != 1.0)
        print(high_corr.any())
        ```

    3. **Reduce lag order** — too many lags relative to observations:

        ```python
        # Rule of thumb: T > k * (k * p + 1) where k = variables, p = lags
        model = VAR(data, lags=1).fit()  # try fewer lags
        ```

??? question "BVAR: MCMC chain did not converge (low acceptance rate)"

    **Cause**: The proposal distribution is poorly tuned for the posterior.

    **Solutions:**

    1. **Increase burn-in and draws**:

        ```python
        result = BVAR(data, lags=2).fit(n_draws=20000, n_burn=5000)
        ```

    2. **Tighten the prior** to regularize:

        ```python
        result = BVAR(data, lags=2, prior="minnesota",
                      prior_params={"lambda1": 0.05}).fit()
        ```

    3. **Reduce model size** — fewer variables or lags

    4. **Check trace plots** for mixing:

        ```python
        result.plot_trace()
        ```

---

## Import Errors

??? question "ModuleNotFoundError: No module named 'kalmanbox'"

    **Cause**: KalmanBox is not installed or the environment is wrong.

    **Solutions:**

    1. **Install KalmanBox**:

        ```bash
        pip install kalmanbox
        ```

    2. **Verify you are in the correct environment**:

        ```bash
        which python
        pip list | grep -E "chronobox|kalmanbox"
        ```

    3. **Reinstall ChronoBox** (pulls KalmanBox as a dependency):

        ```bash
        pip install --force-reinstall chronobox
        ```

??? question "ImportError: cannot import name 'SVAR' from 'chronobox'"

    **Cause**: Outdated version of ChronoBox, or the feature is in a submodule.

    **Solutions:**

    1. **Update ChronoBox**:

        ```bash
        pip install --upgrade chronobox
        ```

    2. **Use the full import path**:

        ```python
        from chronobox.models.svar import SVAR
        ```

    3. **Check available classes**:

        ```python
        import chronobox
        print(dir(chronobox))
        ```

---

## Data Problems

??? question "ValueError: Input contains NaN"

    **Cause**: Missing values in the estimation sample.

    **Solutions:**

    1. **Identify missing values**:

        ```python
        print(data.isna().sum())
        print(data[data.isna().any(axis=1)])
        ```

    2. **Drop missing rows** (simplest):

        ```python
        data_clean = data.dropna()
        ```

    3. **Interpolate** (preserves sample size):

        ```python
        data_clean = data.interpolate(method="time")
        ```

    !!! warning
        Interpolation introduces artificial smoothness. For small gaps it is acceptable; for large gaps, consider shortening the sample.

??? question "TypeError: cannot convert Series to float"

    **Cause**: Passing a DataFrame where a Series is expected, or non-numeric columns.

    **Solutions:**

    1. **Select the correct column**:

        ```python
        # Wrong — passes entire DataFrame
        result = ARIMA(df, order=(1,1,1)).fit()

        # Correct — passes a single Series
        result = ARIMA(df["gdp"], order=(1,1,1)).fit()
        ```

    2. **Ensure numeric types**:

        ```python
        df["gdp"] = pd.to_numeric(df["gdp"], errors="coerce")
        ```

??? question "Frequency not detected / FutureWarning about frequency"

    **Cause**: The DatetimeIndex does not have a recognized frequency.

    **Solutions:**

    1. **Set frequency explicitly**:

        ```python
        data.index.freq = "M"   # Monthly
        data.index.freq = "QS"  # Quarterly
        ```

    2. **Infer from data**:

        ```python
        data = data.asfreq(pd.infer_freq(data.index))
        ```

    3. **Common frequency codes**:

        | Code | Meaning |
        |------|---------|
        | `"D"` | Daily |
        | `"W"` | Weekly |
        | `"M"` | Monthly |
        | `"QS"` | Quarterly (start) |
        | `"YS"` | Annual (start) |

---

## Unexpected Results

??? question "AIC is extremely large or negative"

    **Possible causes and fixes:**

    1. **Data scale issue** — AIC depends on the log-likelihood, which scales with the data. Compare AIC only between models fitted on the **same data**:

        ```python
        # These AICs are NOT comparable
        aic_log = ARIMA(np.log(data), order=(1,1,1)).fit().aic
        aic_raw = ARIMA(data, order=(1,1,1)).fit().aic  # different scale!
        ```

    2. **Different sample lengths** — CSS and MLE use different effective sample sizes. Compare only models with the same method and sample.

    3. **Overflow with large datasets** — with very large $T$, AIC can be large in absolute terms. This is normal; focus on **relative differences** between models.

??? question "Residuals are autocorrelated (Ljung-Box rejects)"

    **Cause**: The model does not capture all serial dependence.

    **Solutions:**

    1. **Increase AR or MA order**:

        ```python
        # Check ACF/PACF of residuals to guide order selection
        from chronobox.visualization import plot_acf, plot_pacf
        plot_acf(result.resid)
        plot_pacf(result.resid)
        ```

    2. **Add seasonal component** if residual ACF shows seasonal spikes:

        ```python
        result = ARIMA(data, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
        ```

    3. **Consider GARCH** if residuals show volatility clustering (autocorrelated squared residuals).

    4. **Use auto_arima** to find a better specification:

        ```python
        from chronobox.selection import auto_arima
        result = auto_arima(data, seasonal=True, m=12)
        ```

??? question "IRF confidence bands are extremely wide"

    **Cause**: Imprecise estimation, often from small samples or too many parameters.

    **Solutions:**

    1. **Reduce lag order** — fewer parameters, tighter estimates
    2. **Use BVAR** — Bayesian shrinkage stabilizes IRF estimates:

        ```python
        from chronobox import BVAR
        result = BVAR(data, lags=2, prior="minnesota").fit()
        irf = result.irf(periods=20)
        irf.plot()
        ```

    3. **Increase bootstrap replications** for more stable confidence bands:

        ```python
        irf = result.irf(periods=20, n_boot=2000)
        ```

---

## Performance

??? question "BVAR is very slow with many variables"

    **Cause**: BVAR scales cubically with the number of variables ($O(k^3)$) and draws.

    **Solutions:**

    1. **Reduce variable count** — keep only essential variables

    2. **Use SSVS prior** for automatic variable selection:

        ```python
        result = BVAR(data, lags=2, prior="ssvs").fit(n_draws=10000)
        ```

    3. **Reduce lag order** — the parameter count is $k^2 p$:

        | Variables ($k$) | Lags ($p$) | Parameters |
        |-----------------|-----------|------------|
        | 3 | 4 | 36 |
        | 7 | 4 | 196 |
        | 10 | 4 | 400 |

    4. **Reduce MCMC draws** for exploratory analysis (increase for final results)

??? question "auto_arima is taking too long"

    **Cause**: Large search space with high `max_p`, `max_q`, or seasonal models.

    **Solutions:**

    1. **Constrain the search space**:

        ```python
        from chronobox.selection import auto_arima
        result = auto_arima(data, max_p=3, max_q=3, max_P=1, max_Q=1)
        ```

    2. **Use stepwise search** (default, much faster than exhaustive):

        ```python
        result = auto_arima(data, stepwise=True)
        ```

    3. **Disable seasonality** if not needed:

        ```python
        result = auto_arima(data, seasonal=False)
        ```

---

## Debugging Tips

??? question "How do I enable verbose output?"

    Enable detailed logging to diagnose issues:

    ```python
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("chronobox")
    logger.setLevel(logging.DEBUG)

    # Now fit — you'll see optimization steps
    result = ARIMA(data, order=(1,1,1)).fit()
    ```

??? question "How do I inspect the internal state of a fitted model?"

    Every results object exposes useful attributes:

    ```python
    result = model.fit()

    # Coefficients
    print(result.params)

    # Standard errors
    print(result.bse)

    # Residuals
    print(result.resid.describe())

    # Log-likelihood
    print(result.llf)

    # Information criteria
    print(f"AIC={result.aic:.2f}, BIC={result.bic:.2f}")

    # Covariance matrix of estimates
    print(result.cov_params())
    ```

??? question "How do I report a bug?"

    Include the following in your [GitHub issue](https://github.com/NodesEcon/chronobox/issues):

    1. **ChronoBox and dependency versions**:

        ```python
        import chronobox
        print(chronobox.__version__)

        import kalmanbox
        print(kalmanbox.__version__)

        import numpy, scipy, pandas
        print(numpy.__version__, scipy.__version__, pandas.__version__)
        ```

    2. **Minimal reproducible example** — the smallest code that triggers the bug

    3. **Full traceback** — copy the complete error message

    4. **Data description** — shape, types, frequency (share data if possible, or use a built-in dataset)

---

## Next Steps

- **Basic questions**: [General FAQ](general.md) — installation, data formats, model choice
- **Advanced topics**: [Advanced FAQ](advanced.md) — cointegration, SVAR, BVAR priors
- **Tutorials**: [Getting Started](../tutorials/index.md) — end-to-end workflows
