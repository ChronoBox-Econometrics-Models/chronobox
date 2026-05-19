---
title: "Advanced FAQ"
description: "Advanced econometrics questions — cointegration, SVAR identification, BVAR priors, IRF interpretation, and more."
---

# Advanced FAQ

Questions for users working with multivariate models, cointegration, structural identification, and Bayesian estimation.

!!! tip "Looking for basics?"
    - **Installation, data, model choice**: [General FAQ](general.md)
    - **Error messages and debugging**: [Troubleshooting](troubleshooting.md)

---

## Convergence and Estimation

??? question "What do I do when ARIMA does not converge?"

    Convergence failures in MLE estimation usually stem from one of these causes:

    **1. Model is over-parameterized**

    Reduce the order — especially the MA component:

    ```python
    from chronobox import ARIMA

    # Try a simpler model
    result = ARIMA(data, order=(1, 1, 0)).fit()
    ```

    **2. Starting values are poor**

    Use CSS-MLE, which finds good starting values via conditional sum of squares:

    ```python
    result = ARIMA(data, order=(2, 1, 2)).fit(method="css-mle")
    ```

    **3. Data scale is extreme**

    Standardize or take logs before fitting:

    ```python
    import numpy as np
    result = ARIMA(np.log(data), order=(1, 1, 1)).fit()
    ```

    **4. Near-unit-root or near-cancellation**

    An AR root near 1.0 or AR/MA roots that nearly cancel make the likelihood surface flat. Consider differencing or simplifying:

    ```python
    # Check roots
    print(result.arroots)
    print(result.maroots)
    # Roots with modulus close to 1.0 signal potential issues
    ```

    !!! tip
        Use `auto_arima` to search over orders automatically — it avoids problematic specifications by penalizing via AIC/BIC.

??? question "How do I choose between CSS and MLE for difficult series?"

    | Scenario | Recommendation |
    |----------|----------------|
    | Short series ($T < 50$) | MLE — CSS loses too many initial observations |
    | Long series, quick exploration | CSS — fast and usually close to MLE |
    | Near-unit-root | CSS-MLE — CSS for starting values, then MLE refinement |
    | Seasonal ARIMA | MLE — CSS can be unreliable with seasonal MA terms |

    The hybrid `"css-mle"` method is often the best compromise:

    ```python
    result = ARIMA(data, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12)).fit(method="css-mle")
    ```

---

## Cointegration

??? question "How do I interpret the Johansen cointegration test?"

    The Johansen test determines how many cointegrating vectors exist in a system of $k$ variables. ChronoBox reports both the **trace** and **maximum eigenvalue** statistics.

    ```python
    from chronobox.tests_stat import johansen_test

    result = johansen_test(data, det_order=0, k_ar_diff=2)
    print(result.summary())
    ```

    **Reading the output:**

    | Null hypothesis | Trace stat | Critical value (5%) | Decision |
    |-----------------|-----------|---------------------|----------|
    | $r = 0$ (no cointegration) | 45.3 | 29.8 | **Reject** → at least 1 |
    | $r \leq 1$ | 12.1 | 15.5 | Fail to reject → exactly 1 |
    | $r \leq 2$ | 3.2 | 3.8 | Fail to reject |

    **Conclusion**: there is **1 cointegrating relationship**. You should estimate a VECM with `rank=1`.

    ```python
    from chronobox import VECM

    model = VECM(data, k_ar_diff=2, coint_rank=1)
    result = model.fit()
    ```

    !!! warning
        Trace and max-eigenvalue tests can disagree. When they do, prefer the **trace test** — it is more robust in small samples.

??? question "How do I interpret an inconclusive bounds test (ARDL)?"

    The Pesaran bounds test for cointegration yields three zones:

    $$
    F\text{-statistic} \begin{cases}
    > I(1) \text{ bound} & \Rightarrow \text{cointegration (reject } H_0\text{)} \\
    < I(0) \text{ bound} & \Rightarrow \text{no cointegration (fail to reject)} \\
    \text{between bounds} & \Rightarrow \text{inconclusive}
    \end{cases}
    $$

    **When the result is inconclusive:**

    1. **Increase sample size** — the test has low power with small $T$
    2. **Check integration orders** — the bounds test assumes a mix of $I(0)$ and $I(1)$; verify with ADF/KPSS
    3. **Try alternative lag lengths** — results can be sensitive to $p$
    4. **Use Johansen as a cross-check** — if Johansen finds cointegration, trust it

    ```python
    from chronobox import ARDL

    model = ARDL(y, x, lags=4, auto=True)
    result = model.fit()
    bounds = result.bounds_test()
    print(bounds.summary())

    # If inconclusive, try different lags
    for p in [2, 3, 4, 5, 6]:
        r = ARDL(y, x, lags=p).fit()
        bt = r.bounds_test()
        print(f"Lags={p}: F={bt.fstat:.3f}, I(0)={bt.lower:.3f}, I(1)={bt.upper:.3f}")
    ```

---

## Structural Models

??? question "When should I use SVAR instead of VAR?"

    | Criterion | VAR (reduced form) | SVAR (structural) |
    |-----------|-------------------|-------------------|
    | **Goal** | Forecasting, Granger causality | Structural analysis, policy shocks |
    | **Identification** | Not required | Required (short-run, long-run, or sign restrictions) |
    | **IRF interpretation** | Correlation-based | Causal (with valid identification) |
    | **Use case** | "How do variables co-move?" | "What is the causal effect of a monetary policy shock?" |

    Use SVAR when you need **causal interpretation** of impulse responses. A reduced-form VAR conflates contemporaneous effects — SVAR disentangles them.

    ```python
    from chronobox import VAR, SVAR
    import numpy as np

    # Reduced-form VAR
    var_result = VAR(data, lags=2).fit()

    # Structural VAR with short-run (Cholesky) restrictions
    A = np.array([[1, 0, 0],
                  ["nan", 1, 0],
                  ["nan", "nan", 1]])
    svar_result = SVAR(data, lags=2, A=A).fit()
    ```

    !!! note
        Cholesky decomposition is the simplest identification strategy. It imposes a recursive ordering — the first variable is not contemporaneously affected by any other. Choose the ordering carefully based on economic theory.

??? question "What is the difference between orthogonal and generalized IRF?"

    | Feature | Orthogonal IRF (OIRF) | Generalized IRF (GIRF) |
    |---------|----------------------|----------------------|
    | **Identification** | Requires Cholesky ordering | Order-invariant |
    | **Shock normalization** | One standard deviation of structural shock | One standard deviation of reduced-form error |
    | **Interpretation** | Causal (if ordering is correct) | Correlational |
    | **Sensitivity to ordering** | High | None |

    **Orthogonal IRF** (default in SVAR):

    $$
    \text{OIRF}_{ij}(h) = \frac{\partial y_{i,t+h}}{\partial \varepsilon_{j,t}}
    $$

    where $\varepsilon_t = P^{-1} u_t$ and $P$ is the Cholesky factor.

    **Generalized IRF** (Pesaran & Shin, 1998):

    $$
    \text{GIRF}_{ij}(h) = \frac{\Phi_h \Sigma e_j}{\sqrt{\sigma_{jj}}}
    $$

    where $\Phi_h$ are the MA coefficient matrices and $\Sigma$ is the residual covariance.

    ```python
    # Orthogonal IRF (requires ordering)
    oirf = svar_result.irf(periods=20, method="cholesky")

    # Generalized IRF (order-invariant)
    girf = var_result.irf(periods=20, method="generalized")
    ```

    !!! tip
        Use GIRF for exploratory analysis when you are unsure about the causal ordering. Use OIRF when you have a theory-based ordering and want causal interpretation.

??? question "TVP-VAR vs VAR with dummies — when to use each?"

    | Feature | TVP-VAR | VAR with dummies |
    |---------|---------|-----------------|
    | **Parameter evolution** | Continuous, gradual | Discrete, at break dates |
    | **Flexibility** | Parameters change every period | Shift at known dates only |
    | **Data requirements** | Large $T$ | Moderate $T$, known break dates |
    | **Estimation** | Kalman filter + MCMC | OLS/MLE |
    | **Computation** | Slow (Bayesian) | Fast |

    **Use TVP-VAR when:**

    - You suspect parameters drift over time (e.g., changing monetary policy regimes)
    - You don't know the break dates

    **Use VAR with dummies when:**

    - Break dates are known (e.g., policy change, financial crisis)
    - Sample size is limited
    - Computational speed matters

    ```python
    from chronobox import TVPVAR, VAR

    # TVP-VAR (time-varying parameters)
    tvp = TVPVAR(data, lags=2)
    tvp_result = tvp.fit(n_draws=5000, n_burn=1000)

    # VAR with structural break dummy
    import pandas as pd
    dummy = pd.Series(0, index=data.index)
    dummy.loc["2008-09":] = 1  # post-Lehman
    var_result = VAR(data, lags=2, exog=dummy).fit()
    ```

---

## Bayesian Models

??? question "How do I configure priors in BVAR?"

    ChronoBox supports several prior specifications for Bayesian VAR:

    | Prior | Description | When to Use |
    |-------|-------------|-------------|
    | **Minnesota** | Shrinks toward random walk | Macroeconomic forecasting (default) |
    | **Normal-Wishart** | Conjugate prior on coefficients and covariance | Analytical posterior, fast |
    | **Diffuse** | Non-informative flat prior | Minimal prior information |
    | **SSVS** | Stochastic search variable selection | Sparse VARs, many variables |

    **Minnesota prior** (most common):

    The hyperparameters control shrinkage:

    - $\lambda_1$ (tightness): overall shrinkage — smaller values shrink more toward prior
    - $\lambda_2$ (cross-variable): shrinkage on other-variable lags — usually $\lambda_2 < \lambda_1$
    - $\lambda_3$ (lag decay): penalizes distant lags — $1$ for linear decay, $2$ for quadratic

    ```python
    from chronobox import BVAR

    model = BVAR(data, lags=4, prior="minnesota",
                 prior_params={
                     "lambda1": 0.1,   # tight overall shrinkage
                     "lambda2": 0.5,   # moderate cross-variable shrinkage
                     "lambda3": 1,     # linear lag decay
                 })
    result = model.fit(n_draws=10000, n_burn=2000)
    ```

    !!! tip
        For forecasting, a tighter Minnesota prior ($\lambda_1 \approx 0.1$) often outperforms OLS-VAR, especially with many variables. For structural analysis, loosen the prior ($\lambda_1 \approx 0.5$) to let data speak.

    **SSVS prior** (variable selection):

    ```python
    model = BVAR(data, lags=4, prior="ssvs",
                 prior_params={
                     "tau0": 0.01,  # spike (near zero)
                     "tau1": 1.0,   # slab (unrestricted)
                     "p0": 0.5,    # prior inclusion probability
                 })
    result = model.fit(n_draws=20000, n_burn=5000)

    # Posterior inclusion probabilities
    print(result.inclusion_probs)
    ```

---

## Next Steps

- **Basic questions**: [General FAQ](general.md) — installation, data formats, model choice
- **Error resolution**: [Troubleshooting](troubleshooting.md) — convergence errors, data issues
- **Theory**: [VAR Theory](../theory/var-theory.md) · [SVAR Theory](../theory/svar-theory.md) · [BVAR Theory](../theory/bvar-theory.md)
