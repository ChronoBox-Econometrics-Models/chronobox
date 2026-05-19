---
title: "ARDL & ECM API"
description: "API reference for ARDL, ECM, and bounds test — cointegration analysis with mixed integration orders"
---

# ARDL & ECM API Reference

!!! info "Module"
    **Import**: `from chronobox.models.ardl import ARDL`; `from chronobox.models.ecm import ECM`
    **Source**: `chronobox/models/ardl.py`, `chronobox/models/ecm.py`, `chronobox/tests_stat/cointegration/bounds_test.py`

## Overview

| Class / Function | Description | Use Case |
|------------------|-------------|----------|
| `ARDL` | Autoregressive Distributed Lag model with automatic lag selection | Long-run relationships with $I(0)$/$I(1)$ regressors |
| `ECM` | Error Correction Model | Short-run dynamics with equilibrium adjustment |
| `bounds_test` | Pesaran-Shin-Smith bounds test for cointegration | Testing cointegration in ARDL framework |

---

## ARDL

Autoregressive Distributed Lag model ARDL($p, q_1, \ldots, q_k$) with optional
automatic lag selection via information criteria.

$$
y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{j=1}^{k} \sum_{\ell=0}^{q_j} \beta_{j,\ell} x_{j,t-\ell} + u_t
$$

The ARDL approach (Pesaran, Shin & Smith, 2001) is particularly useful because:

- Valid regardless of whether regressors are $I(0)$, $I(1)$, or mutually cointegrated
- Provides a framework for bounds testing for cointegration
- Easily reparameterized as an Error Correction Model

```python
ARDL(
    max_p: int = 4,
    max_q: int = 4,
    criterion: str = "aic",
    uniform_q: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_p` | `int` | `4` | Maximum AR lags to consider for $y$ |
| `max_q` | `int` | `4` | Maximum distributed lags to consider for $x$ |
| `criterion` | `str` | `"aic"` | Information criterion: `'aic'` or `'bic'` |
| `uniform_q` | `bool` | `True` | If `True`, same lag order for all regressors; if `False`, select independently |

### `.fit()` Method

```python
ARDL.fit(
    y: ndarray,
    x: ndarray,
    p: int | None = None,
    x_lags: int | list[int] | None = None,
) -> ARDLResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray(T,)` | *required* | Dependent variable |
| `x` | `ndarray(T, k)` | *required* | Regressors matrix |
| `p` | `int \| None` | `None` | Fixed AR lag order. If `None`, selected automatically |
| `x_lags` | `int \| list[int] \| None` | `None` | Fixed distributed lag order(s). If `None`, selected automatically |

**Returns**: `ARDLResult`

### ARDLResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `coefficients` | `ndarray` | Estimated coefficients (intercept, AR lags, distributed lags) |
| `residuals` | `ndarray` | OLS residuals |
| `fitted_values` | `ndarray` | Fitted values $\hat{y}_t$ |
| `y_lags` | `int` | Selected AR lag order $p$ |
| `x_lags` | `int \| list[int]` | Selected distributed lag order(s) $q$ |
| `aic` | `float` | Akaike Information Criterion |
| `bic` | `float` | Bayesian Information Criterion |
| `sigma2` | `float` | Residual variance $\hat{\sigma}^2$ |
| `r_squared` | `float` | $R^2$ |
| `adj_r_squared` | `float` | Adjusted $R^2$ |
| `nobs` | `int` | Number of effective observations |
| `k_params` | `int` | Number of estimated parameters |
| `se` | `ndarray` | Standard errors |
| `t_stats` | `ndarray` | $t$-statistics |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Formatted text summary with coefficients, standard errors, $t$-stats |
| `bounds_test()` | `dict` | PSS bounds test for cointegration (convenience wrapper) |
| `to_ecm()` | `ECMResult` | Reparameterize as Error Correction Model |
| `long_run_coefficients` | `ndarray` | Long-run multipliers $\theta_j = \frac{\sum_\ell \beta_{j,\ell}}{1 - \sum_i \phi_i}$ (property) |

### Example

```python
import numpy as np
from chronobox.models.ardl import ARDL

# Simulate data
rng = np.random.default_rng(42)
T = 200
x = rng.normal(size=(T, 2)).cumsum(axis=0)
y = 0.5 * x[:, 0] + 0.3 * x[:, 1] + rng.normal(0, 0.5, T).cumsum()

# Automatic lag selection via AIC
model = ARDL(max_p=4, max_q=4, criterion="aic")
result = model.fit(y, x)
print(result.summary())
print(f"Selected: ARDL({result.y_lags}, {result.x_lags})")
print(f"AIC: {result.aic:.4f}, BIC: {result.bic:.4f}")

# Long-run coefficients
lr = result.long_run_coefficients
print(f"Long-run coefficients: {lr}")

# Fixed lag specification
model_fixed = ARDL()
result_fixed = model_fixed.fit(y, x, p=2, x_lags=[3, 1])

# Bounds test from fitted ARDL
bt = result.bounds_test()
print(f"F-statistic: {bt['f_statistic']:.4f}")

# Convert to ECM form
ecm_result = result.to_ecm()
print(f"Speed of adjustment: {ecm_result.speed_of_adjustment:.4f}")
```

::: chronobox.models.ardl.ARDL
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit

::: chronobox.models.ardl.ARDLResult
    options:
      show_root_heading: true
      show_source: false
      members:
        - summary
        - bounds_test
        - to_ecm
        - long_run_coefficients

---

## ECM

Error Correction Model estimated via OLS. Captures short-run dynamics and
the speed of adjustment toward long-run equilibrium:

$$
\Delta y_t = \alpha + \pi_{yy} y_{t-1} + \boldsymbol{\pi}_{yx}' \mathbf{x}_{t-1} + \sum_{i=1}^{p-1} \gamma_i \Delta y_{t-i} + \sum_{j=1}^{k} \sum_{\ell=0}^{q_j-1} \delta_{j,\ell} \Delta x_{j,t-\ell} + u_t
$$

where $\pi_{yy}$ is the speed of adjustment (should be negative for stability).

```python
ECM(
    lags: int = 1,
    x_lags: int | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lags` | `int` | `1` | Number of lagged differences for $\Delta y$ |
| `x_lags` | `int \| None` | `None` | Number of lagged differences for $\Delta x$. If `None`, same as `lags` |

### `.fit()` Method

```python
ECM.fit(
    y: ndarray,
    x: ndarray,
) -> ECMResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray(T,)` | *required* | Dependent variable |
| `x` | `ndarray(T, k)` | *required* | Regressors matrix |

**Returns**: `ECMResult`

### ECMResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `coefficients` | `ndarray` | Full coefficient vector |
| `residuals` | `ndarray` | OLS residuals |
| `fitted_values` | `ndarray` | Fitted values |
| `speed_of_adjustment` | `float` | $\pi_{yy}$ — speed of adjustment coefficient |
| `long_run_coefficients` | `ndarray` | Implied long-run coefficients $-\boldsymbol{\pi}_{yx} / \pi_{yy}$ |
| `pi_yy` | `float` | Error correction coefficient for $y_{t-1}$ |
| `pi_yx` | `ndarray` | Error correction coefficients for $\mathbf{x}_{t-1}$ |
| `short_run_y` | `ndarray` | Coefficients on $\Delta y$ lags |
| `short_run_x` | `ndarray` | Coefficients on $\Delta x$ lags |
| `sigma2` | `float` | Residual variance |
| `r_squared` | `float` | $R^2$ |
| `nobs` | `int` | Number of effective observations |
| `k_params` | `int` | Number of parameters |
| `se` | `ndarray` | Standard errors |
| `t_stats` | `ndarray` | $t$-statistics |
| `aic` | `float` | Akaike Information Criterion |
| `bic` | `float` | Bayesian Information Criterion |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Formatted text summary |
| `bounds_test_pss()` | `dict` | PSS bounds test on the ECM specification |

### Example

```python
import numpy as np
from chronobox.models.ecm import ECM

# Simulate cointegrated data
rng = np.random.default_rng(42)
T = 200
x = rng.normal(size=(T, 2)).cumsum(axis=0)
u = rng.normal(0, 0.5, T)
y = 2.0 * x[:, 0] - 1.0 * x[:, 1] + u.cumsum()

# Fit ECM
ecm = ECM(lags=2)
result = ecm.fit(y, x)
print(result.summary())

# Speed of adjustment (should be negative)
print(f"Speed of adjustment: {result.speed_of_adjustment:.4f}")
print(f"Long-run coefficients: {result.long_run_coefficients}")

# Short-run dynamics
print(f"Short-run y coeffs: {result.short_run_y}")
print(f"Short-run x coeffs: {result.short_run_x}")

# Bounds test
bt = result.bounds_test_pss()
```

::: chronobox.models.ecm.ECM
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit

::: chronobox.models.ecm.ECMResult
    options:
      show_root_heading: true
      show_source: false
      members:
        - summary
        - bounds_test_pss

---

## bounds_test

Pesaran, Shin & Smith (2001) bounds test for cointegration within the ARDL framework.

Tests the null hypothesis of no long-run relationship:

$$
H_0: \pi_{yy} = 0 \text{ and } \boldsymbol{\pi}_{yx} = \mathbf{0}
$$

The test produces an $F$-statistic and a $t$-statistic, compared against two sets
of critical bounds:

- **Lower bound**: assumes all regressors are $I(0)$
- **Upper bound**: assumes all regressors are $I(1)$

| Decision | Condition |
|----------|-----------|
| Reject $H_0$ (cointegration) | $F > \text{upper bound}$ |
| Cannot reject $H_0$ | $F < \text{lower bound}$ |
| Inconclusive | $F$ between bounds |

```python
bounds_test(
    y: ndarray,
    x: ndarray,
    lags: int | None = None,
    case: int = 3,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray(T,)` | *required* | Dependent variable |
| `x` | `ndarray(T, k)` | *required* | Regressors matrix |
| `lags` | `int \| None` | `None` | Number of lags. If `None`, selected via AIC |
| `case` | `int` | `3` | Deterministic specification (1-5). Case 3 = unrestricted intercept, no trend |

### Deterministic Cases

| Case | Intercept | Trend | Description |
|------|-----------|-------|-------------|
| 1 | No | No | No intercept, no trend |
| 2 | Restricted | No | Restricted intercept |
| 3 | Unrestricted | No | Unrestricted intercept (most common) |
| 4 | Unrestricted | Restricted | Unrestricted intercept, restricted trend |
| 5 | Unrestricted | Unrestricted | Unrestricted intercept and trend |

### TestResult

The returned `TestResult` includes `additional_info` with:

| Key | Type | Description |
|-----|------|-------------|
| `f_statistic` | `float` | Joint $F$-statistic for $\pi_{yy} = \boldsymbol{\pi}_{yx} = 0$ |
| `t_statistic` | `float` | $t$-statistic for $\pi_{yy} = 0$ |
| `bounds` | `dict` | Critical value bounds at 10%, 5%, 1% |
| `decision` | `str` | `'cointegration'`, `'no_cointegration'`, or `'inconclusive'` |

### Example

```python
import numpy as np
from chronobox.tests_stat.cointegration.bounds_test import bounds_test

# Simulate data
rng = np.random.default_rng(42)
T = 200
x = rng.normal(size=(T, 2)).cumsum(axis=0)
y = 1.5 * x[:, 0] - 0.8 * x[:, 1] + rng.normal(0, 0.5, T).cumsum()

# Run bounds test (Case III — unrestricted intercept, no trend)
result = bounds_test(y, x, case=3)
print(f"F-statistic: {result.additional_info['f_statistic']:.4f}")
print(f"t-statistic: {result.additional_info['t_statistic']:.4f}")
print(f"Decision: {result.additional_info['decision']}")
print(f"Bounds (5%): {result.additional_info['bounds']['5pct']}")

# With fixed lags
result_fixed = bounds_test(y, x, lags=4, case=3)
```

::: chronobox.tests_stat.cointegration.bounds_test.bounds_test
    options:
      show_root_heading: false
      show_source: true

---

## See Also

- [ARDL Theory](../theory/ardl-theory.md) -- ARDL bounds testing and ECM derivation
- [ARDL User Guide](../user-guide/ardl/index.md) -- Step-by-step ARDL guide
- [VAR & VECM API](var.md) -- Multivariate cointegration (Johansen approach)
- [Core API](core.md) -- Foundation classes
