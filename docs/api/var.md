---
title: "VAR & VECM API"
description: "API reference for VAR, VECM, IRF, FEVD, and granger_causality — multivariate time series models"
---

# VAR & VECM API Reference

!!! info "Module"
    **Import**: `from chronobox import VAR, VECM`
    **Source**: `chronobox/models/var.py`, `chronobox/models/vecm.py`, `chronobox/analysis/`

## Overview

| Class / Function | Description | Use Case |
|------------------|-------------|----------|
| `VAR` | Vector Autoregression VAR(p) estimated by OLS | Multivariate forecasting, IRF, FEVD |
| `VECM` | Vector Error Correction Model via Johansen procedure | Cointegrated systems |
| `IRF` | Impulse Response Functions (Cholesky or generalized) | Dynamic shock analysis |
| `FEVD` | Forecast Error Variance Decomposition | Variance attribution |
| `granger_causality` | Granger causality test | Predictive causality testing |

---

## VAR

Vector Autoregression model estimated by equation-by-equation OLS.

$$
\mathbf{y}_t = \mathbf{c} + A_1 \mathbf{y}_{t-1} + A_2 \mathbf{y}_{t-2} + \cdots + A_p \mathbf{y}_{t-p} + \mathbf{u}_t
$$

where $\mathbf{y}_t$ is a $K \times 1$ vector, $A_i$ are $K \times K$ coefficient matrices,
and $\mathbf{u}_t \sim N(\mathbf{0}, \Sigma_u)$.

```python
VAR(
    lags: int | None = None,
    trend: str = "c",
    maxlags: int | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lags` | `int \| None` | `None` | Number of lags $p$. If None and `maxlags` provided, selected via AIC |
| `trend` | `str` | `"c"` | `'n'` (none), `'c'` (constant), `'ct'` (constant + trend), `'ctt'` (+ quadratic) |
| `maxlags` | `int \| None` | `None` | Maximum lags for automatic selection (only when `lags=None`) |

### `.fit()` Method

```python
VAR.fit(
    endog: ndarray | pd.DataFrame,
    names: list[str] | None = None,
) -> VARResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `ndarray(T, K) \| DataFrame` | *required* | Multivariate time series |
| `names` | `list[str] \| None` | `None` | Variable names. Inferred from DataFrame columns if not given |

**Returns**: `VARResults`

### VARResults

| Attribute | Type | Description |
|-----------|------|-------------|
| `coefs` | `ndarray(p, K, K)` | Coefficient matrices $A_1, \ldots, A_p$ |
| `sigma_u` | `ndarray(K, K)` | Residual covariance (bias-corrected) |
| `sigma_u_ml` | `ndarray(K, K)` | Residual covariance (ML) |
| `intercept` | `ndarray(K,)` | Intercept terms |
| `resid` | `ndarray(T', K)` | Residual matrix |
| `endog` | `ndarray(T, K)` | Original data |
| `nobs` | `int` | Effective observations $T'$ |
| `nobs_total` | `int` | Total observations $T$ |
| `k_ar` | `int` | VAR order $p$ |
| `neqs` | `int` | Number of variables $K$ |
| `names` | `list[str]` | Variable names |
| `trend` | `str` | Trend specification |
| `aic` | `float` | Akaike IC |
| `bic` | `float` | Bayesian IC |
| `hqic` | `float` | Hannan-Quinn IC |
| `fpe` | `float` | Final Prediction Error |
| `is_stable` | `bool` | Whether all companion eigenvalues are inside the unit circle |
| `roots` | `ndarray` | Companion matrix eigenvalues |

| Method | Description |
|--------|-------------|
| `summary()` | Formatted summary per equation |
| `forecast(steps, alpha=0.05)` | Multi-step forecasts with confidence intervals |
| `irf(periods=20, method='cholesky', ...)` | Compute IRF (shortcut) |
| `fevd(periods=20)` | Compute FEVD (shortcut) |
| `granger_causality(caused, causing, signif=0.05)` | Granger test (shortcut) |
| `select_order(maxlags)` | Lag order selection via IC |

### Example

```python
import numpy as np
import pandas as pd
from chronobox import VAR

# Create data
data = pd.DataFrame({
    "GDP": np.cumsum(np.random.randn(200)),
    "INF": np.cumsum(np.random.randn(200)),
    "RATE": np.cumsum(np.random.randn(200)),
})

# Fit VAR(2) with constant
model = VAR(lags=2, trend="c")
results = model.fit(data)
print(results.summary())

# Information criteria
print(f"AIC: {results.aic:.4f}, BIC: {results.bic:.4f}")
print(f"Stable: {results.is_stable}")

# Automatic lag selection
model_auto = VAR(lags=None, maxlags=8)
results_auto = model_auto.fit(data)
print(f"Selected lag order: {results_auto.k_ar}")

# Forecast
fc = results.forecast(steps=12)
```

::: chronobox.models.var.VAR
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit

::: chronobox.models.var.VARResults
    options:
      show_root_heading: true
      show_source: false
      members:
        - summary
        - forecast
        - irf
        - fevd
        - granger_causality
        - select_order

---

## VECM

Vector Error Correction Model estimated via the Johansen (1988, 1991) procedure.

$$
\Delta \mathbf{y}_t = \alpha \beta' \mathbf{y}_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta \mathbf{y}_{t-i} + \mathbf{c} + \mathbf{u}_t
$$

where $\alpha$ is the $K \times r$ loading matrix, $\beta$ contains the $r$ cointegrating
vectors, and $\Gamma_i$ are short-run coefficient matrices.

```python
VECM(
    lags: int = 1,
    coint_rank: int | None = None,
    deterministic: str = "ci",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lags` | `int` | `1` | Number of lags in levels (VECM uses $p-1$ lagged differences) |
| `coint_rank` | `int \| None` | `None` | Cointegration rank $r$. If None, determined via Johansen trace test |
| `deterministic` | `str` | `"ci"` | Deterministic specification (see below) |

### Deterministic Specifications

| Value | Description |
|-------|-------------|
| `'nc'` | No constant, no trend |
| `'ci'` | Constant inside ECM (restricted to cointegrating space) |
| `'co'` | Constant outside ECM (unrestricted) |
| `'li'` | Linear trend inside ECM + unrestricted constant |
| `'lo'` | Linear trend outside ECM + unrestricted constant |

### `.fit()` Method

```python
VECM.fit(
    endog: ndarray | pd.DataFrame,
    names: list[str] | None = None,
) -> VECMResults
```

### `.johansen_test()` Method

```python
VECM.johansen_test(endog: ndarray | pd.DataFrame) -> JohansenResults
```

Run the Johansen cointegration test without fitting the full VECM.

### VECMResults

| Attribute | Type | Description |
|-----------|------|-------------|
| `alpha` | `ndarray(K, r)` | Loading matrix (adjustment speeds) |
| `beta` | `ndarray(K, r)` | Cointegrating vectors |
| `gamma` | `list[ndarray(K, K)]` | Short-run matrices $\Gamma_1, \ldots, \Gamma_{p-1}$ |
| `pi` | `ndarray(K, K)` | Long-run matrix $\Pi = \alpha \beta'$ |
| `sigma_u` | `ndarray(K, K)` | Residual covariance |
| `resid` | `ndarray(T', K)` | Residuals |
| `coint_rank` | `int` | Cointegration rank $r$ |
| `eigenvalues` | `ndarray(K,)` | Johansen eigenvalues |
| `trace_stat` | `ndarray(K,)` | Trace test statistics |
| `max_eig_stat` | `ndarray(K,)` | Max-eigenvalue statistics |
| `trace_crit` | `ndarray(K, 3)` | Critical values (90%, 95%, 99%) |
| `nobs` | `int` | Effective observations |
| `k_ar` | `int` | VAR lag order in levels |
| `neqs` | `int` | Number of variables $K$ |
| `names` | `list[str]` | Variable names |
| `deterministic` | `str` | Deterministic specification |

| Method | Description |
|--------|-------------|
| `summary()` | Formatted summary with Johansen test, loading matrix, cointegrating vectors |
| `forecast(steps)` | Multi-step forecasts in levels |

### Example

```python
import numpy as np
from chronobox import VECM

# Generate cointegrated data
rng = np.random.default_rng(42)
e = rng.normal(size=(200, 3))
data = np.cumsum(e, axis=0)
data[:, 1] += 0.5 * data[:, 0]  # cointegration

# Fit VECM with automatic rank determination
model = VECM(lags=2, deterministic="ci")
results = model.fit(data, names=["x1", "x2", "x3"])
print(results.summary())

# Johansen test only
johansen = model.johansen_test(data)
print(f"Cointegration rank: {johansen.rank}")

# Fixed rank
model_r1 = VECM(lags=2, coint_rank=1, deterministic="co")
results_r1 = model_r1.fit(data)

# Access cointegrating vectors
print("Beta (cointegrating vectors):")
print(results.beta)
print("Alpha (loading matrix):")
print(results.alpha)

# Forecast
fc = results.forecast(steps=12)
```

::: chronobox.models.vecm.VECM
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit
        - johansen_test

::: chronobox.models.vecm.VECMResults
    options:
      show_root_heading: true
      show_source: false
      members:
        - summary
        - forecast

---

## Impulse Response Functions {#impulse-response-functions}

Compute and analyze impulse response functions from a fitted VAR model.

$$
\text{IRF}(h)_{ij} = \frac{\partial y_{i,t+h}}{\partial u_{j,t}}
$$

```python
IRF(
    var_results: VARResults,
    periods: int = 20,
    method: str = "cholesky",
    sigs: float = 0.95,
    runs: int = 1000,
    seed: int | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `var_results` | `VARResults` | *required* | Fitted VAR model |
| `periods` | `int` | `20` | Number of IRF horizons |
| `method` | `str` | `"cholesky"` | `'cholesky'` (orthogonalized) or `'generalized'` (Pesaran-Shin 1998) |
| `sigs` | `float` | `0.95` | Confidence level for bootstrap bands |
| `runs` | `int` | `1000` | Bootstrap replications. Set to 0 to skip |
| `seed` | `int \| None` | `None` | Random seed for bootstrap |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `irfs` | `ndarray(H+1, K, K)` | Point estimates. `irfs[h, i, j]` = response of $i$ to shock in $j$ at horizon $h$ |
| `lower` | `ndarray \| None` | Lower confidence band |
| `upper` | `ndarray \| None` | Upper confidence band |
| `cum_irfs` | `ndarray(H+1, K, K)` | Cumulative IRFs |

### Methods

| Method | Description |
|--------|-------------|
| `plot(impulse, response)` | Plot IRF for specific impulse-response pair |
| `to_dataframe()` | Convert to tidy DataFrame |
| `summary()` | Formatted summary |

### Example

```python
from chronobox.analysis.irf import IRF

# From fitted VAR
irf = IRF(var_results, periods=24, method="cholesky", runs=1000, seed=42)

# Access IRF values
# Response of variable 0 to shock in variable 1 at horizon 10
print(irf.irfs[10, 0, 1])

# Cumulative IRF
print(irf.cum_irfs[20, :, 0])

# Plot
irf.plot(impulse="GDP", response="INF")

# Via VARResults shortcut
irf = results.irf(periods=20, method="cholesky")
```

::: chronobox.analysis.irf.IRF
    options:
      show_root_heading: false
      show_source: true
      members:
        - plot
        - to_dataframe
        - summary

---

## Forecast Error Variance Decomposition {#forecast-error-variance-decomposition}

Decomposes the forecast error variance of each variable into contributions
from orthogonalized shocks.

$$
\text{FEVD}(h)_{ik} = \frac{\sum_{s=0}^{h} (\Theta_s)_{ik}^2}{\sum_{s=0}^{h} \sum_{j=1}^{K} (\Theta_s)_{ij}^2}
$$

where $\Theta_s = \Phi_s P$ and $P$ is the Cholesky factor of $\Sigma_u$.

```python
FEVD(
    var_results: VARResults,
    periods: int = 20,
    method: str = "cholesky",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `var_results` | `VARResults` | *required* | Fitted VAR model |
| `periods` | `int` | `20` | Number of forecast horizons |
| `method` | `str` | `"cholesky"` | Orthogonalization method |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `decomp` | `ndarray(H+1, K, K)` | `decomp[h, i, k]` = fraction of FEV of variable $i$ at horizon $h$ from shock $k$ |

### Methods

| Method | Description |
|--------|-------------|
| `plot(variable)` | Stacked area plot for a specific variable |
| `to_dataframe()` | Convert to tidy DataFrame |
| `summary()` | Formatted summary table |

### Example

```python
from chronobox.analysis.fevd import FEVD

# From fitted VAR
fevd = FEVD(var_results, periods=20)

# Variance decomposition of variable 0 at horizon 10
print(fevd.decomp[10, 0, :])  # contributions from each shock

# Plot
fevd.plot(variable="GDP")

# Via VARResults shortcut
fevd = results.fevd(periods=20)
```

::: chronobox.analysis.fevd.FEVD
    options:
      show_root_heading: false
      show_source: true
      members:
        - plot
        - to_dataframe
        - summary

---

## Granger Causality {#granger-causality}

Test whether one variable Granger-causes another in a fitted VAR model.

- $H_0$: `causing` does NOT Granger-cause `caused`
- $H_1$: `causing` Granger-causes `caused`

The test checks whether all coefficients of the causing variable in the
equation of the caused variable are jointly zero.

```python
granger_causality(
    var_results: VARResults,
    caused: str | int,
    causing: str | int,
    signif: float = 0.05,
) -> GrangerResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `var_results` | `VARResults` | *required* | Fitted VAR model |
| `caused` | `str \| int` | *required* | Name or index of the caused (dependent) variable |
| `causing` | `str \| int` | *required* | Name or index of the causing variable |
| `signif` | `float` | `0.05` | Significance level |

### GrangerResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `fstat` | `float` | F-test statistic |
| `pvalue` | `float` | p-value from F distribution |
| `df` | `tuple[int, int]` | Degrees of freedom (numerator, denominator) |
| `reject` | `bool` | Whether to reject $H_0$ at the given significance level |
| `wald_stat` | `float` | Wald test statistic |
| `wald_pvalue` | `float` | p-value from $\chi^2$ distribution |
| `caused` | `str` | Name of the caused variable |
| `causing` | `str` | Name of the causing variable |
| `signif` | `float` | Significance level used |

### Example

```python
from chronobox.analysis.granger import granger_causality

# Test: does GDP Granger-cause inflation?
result = granger_causality(var_results, caused="INF", causing="GDP")
print(result)
# GrangerResult(GDP -> INF: F=5.1234, p=0.0012, REJECT H0 at 5%)

if result.reject:
    print(f"GDP Granger-causes INF (F={result.fstat:.4f}, p={result.pvalue:.4f})")

# Test all pairwise Granger causalities
names = var_results.names
for caused in names:
    for causing in names:
        if caused != causing:
            r = granger_causality(var_results, caused, causing)
            if r.reject:
                print(f"{r.causing} -> {r.caused}: p={r.pvalue:.4f}")

# Via VARResults shortcut
result = results.granger_causality(caused="INF", causing="GDP")
```

::: chronobox.analysis.granger.granger_causality
    options:
      show_root_heading: false
      show_source: true

::: chronobox.models.var.GrangerResult
    options:
      show_root_heading: true
      show_source: false

---

## See Also

- [Core API](core.md) -- Foundation classes
- [SVAR API](svar.md) -- Structural VAR, Bayesian VAR, FAVAR, TVPVAR
- [VAR Theory](../theory/var-theory.md) -- VAR algebra and estimation
- [VECM Theory](../theory/vecm-theory.md) -- Cointegration and Johansen procedure
- [VAR User Guide](../user-guide/var/index.md) -- Step-by-step VAR guide
- [IRF User Guide](../user-guide/var/irf.md) -- Impulse response analysis
- [FEVD User Guide](../user-guide/var/fevd.md) -- Variance decomposition guide
- [Granger Causality Guide](../user-guide/var/granger.md) -- Granger causality testing
