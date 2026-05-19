---
title: "Diagnostics API"
description: "API reference for statistical tests — unit root, cointegration, specification, structural breaks, and VAR diagnostics"
---

# Diagnostics API Reference

!!! info "Module"
    **Import**: `from chronobox.tests_stat import adf_test, kpss_test, ljung_box_test, ...`
    **Source**: `chronobox/tests_stat/`

## Overview

| Function | Category | Description |
|----------|----------|-------------|
| `adf_test` | Unit Root | Augmented Dickey-Fuller test |
| `pp_test` | Unit Root | Phillips-Perron test |
| `kpss_test` | Unit Root | KPSS stationarity test |
| `ers_test` | Unit Root | Elliott-Rothenberg-Stock / DF-GLS test |
| `zivot_andrews_test` | Unit Root | Zivot-Andrews with structural break |
| `lee_strazicich_test` | Unit Root | Lee-Strazicich LM unit root with breaks |
| `hegy_test` | Unit Root | HEGY seasonal unit root test |
| `engle_granger_test` | Cointegration | Engle-Granger two-step test |
| `gregory_hansen_test` | Cointegration | Gregory-Hansen with structural break |
| `bounds_test` | Cointegration | Pesaran-Shin-Smith ARDL bounds test |
| `phillips_ouliaris_test` | Cointegration | Phillips-Ouliaris residual-based test |
| `ljung_box_test` | Specification | Ljung-Box autocorrelation test |
| `breusch_godfrey_test` | Specification | Breusch-Godfrey serial correlation LM test |
| `durbin_watson_test` | Specification | Durbin-Watson autocorrelation test |
| `bds_test` | Specification | BDS test for independence |
| `arch_lm_test` | Specification | ARCH-LM conditional heteroskedasticity |
| `white_test` | Specification | White heteroskedasticity test |
| `jarque_bera_test` | Specification | Jarque-Bera normality test |
| `reset_test` | Specification | Ramsey RESET functional form test |
| `chow_test` | Structural Break | Chow test for known break point |
| `bai_perron_test` | Structural Break | Bai-Perron multiple structural changes |
| `cusum_test` | Structural Break | CUSUM parameter stability |
| `cusumsq_test` | Structural Break | CUSUM of squares test |
| `qlr_test` | Structural Break | Quandt Likelihood Ratio (sup-Wald) test |

All test functions return a standardized [`TestResult`](#testresult) dataclass.

---

## TestResult

Standardized result container returned by all statistical tests.

```python
@dataclass
class TestResult:
    test_name: str
    statistic: float
    pvalue: float | None
    critical_values: dict[str, float]
    null_hypothesis: str
    alternative_hypothesis: str
    reject_at_5pct: bool
    lags_used: int | None = None
    additional_info: dict[str, Any] = {}
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `test_name` | `str` | Name of the test (e.g., `"ADF"`, `"KPSS"`, `"Ljung-Box"`) |
| `statistic` | `float` | Value of the test statistic |
| `pvalue` | `float \| None` | p-value (None if only critical values available) |
| `critical_values` | `dict[str, float]` | Critical values, e.g., `{'1%': -3.43, '5%': -2.86, '10%': -2.57}` |
| `null_hypothesis` | `str` | Description of $H_0$ |
| `alternative_hypothesis` | `str` | Description of $H_1$ |
| `reject_at_5pct` | `bool` | `True` if $H_0$ is rejected at 5% significance |
| `lags_used` | `int \| None` | Number of lags used (if applicable) |
| `additional_info` | `dict[str, Any]` | Test-specific extra information (e.g., break dates, vectors) |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Formatted multi-line summary with statistic, p-value, critical values, and decision |

### Example

```python
from chronobox.tests_stat import adf_test
import numpy as np

y = np.cumsum(np.random.default_rng(42).normal(size=200))
result = adf_test(y)

print(result)
# TestResult(ADF: stat=-1.2345, pval=0.6543, Fail to reject H0)

print(result.summary())
# ============================================================
#   ADF Test
# ============================================================
#   Test statistic : -1.234567
#   p-value        : 0.654321
#   ...
#   Decision (5%)  : Fail to reject H0
# ============================================================

print(result.reject_at_5pct)   # False
print(result.critical_values)  # {'1%': -3.43, '5%': -2.86, '10%': -2.57}
```

::: chronobox.tests_stat.base.TestResult
    options:
      show_root_heading: false
      show_source: true
      members:
        - summary

---

## Unit Root Tests

### adf_test

Augmented Dickey-Fuller test for a unit root. Tests $H_0$: the series has a unit root
against $H_1$: the series is stationary.

The ADF regression:

$$
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t
$$

```python
adf_test(
    y: ndarray,
    maxlag: int | None = None,
    autolag: str = "AIC",
    trend: str = "c",
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Time series data |
| `maxlag` | `int \| None` | `None` | Maximum lag order. If None, auto-selected |
| `autolag` | `str` | `"AIC"` | Lag selection criterion: `'AIC'`, `'BIC'`, `'t-stat'` |
| `trend` | `str` | `"c"` | `'n'` (none), `'c'` (constant), `'ct'` (constant + trend) |

### Example

```python
import numpy as np
from chronobox.tests_stat import adf_test

rng = np.random.default_rng(42)
y = np.cumsum(rng.normal(size=200))  # Unit root process

result = adf_test(y, trend="c")
print(f"ADF statistic: {result.statistic:.4f}")
print(f"p-value: {result.pvalue:.4f}")
print(f"Reject H0? {result.reject_at_5pct}")
```

::: chronobox.tests_stat.unit_root.adf_test
    options:
      show_root_heading: false
      show_source: true

---

### pp_test

Phillips-Perron test for a unit root. Non-parametric correction for serial correlation
(does not add lagged differences to the regression).

```python
pp_test(
    y: ndarray,
    trend: str = "c",
    lags: int | None = None,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Time series data |
| `trend` | `str` | `"c"` | `'n'`, `'c'`, or `'ct'` |
| `lags` | `int \| None` | `None` | Bandwidth for Newey-West. If None, auto-selected |

::: chronobox.tests_stat.unit_root.pp_test
    options:
      show_root_heading: false
      show_source: true

---

### kpss_test

Kwiatkowski-Phillips-Schmidt-Shin (KPSS) stationarity test.
Tests $H_0$: the series is stationary against $H_1$: the series has a unit root.

!!! warning "Reversed hypotheses"
    Unlike ADF and PP, KPSS tests the null of **stationarity**. Rejection means
    evidence of a unit root. Use KPSS together with ADF for confirmatory analysis.

```python
kpss_test(
    y: ndarray,
    trend: str = "c",
    lags: str | int = "auto",
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Time series data |
| `trend` | `str` | `"c"` | `'c'` (level stationarity) or `'ct'` (trend stationarity) |
| `lags` | `str \| int` | `"auto"` | Bandwidth. `'auto'` uses Newey-West automatic selection |

::: chronobox.tests_stat.unit_root.kpss_test
    options:
      show_root_heading: false
      show_source: true

---

### ers_test

Elliott-Rothenberg-Stock / DF-GLS test. GLS-detrended version of ADF with
improved power against near-unit-root alternatives.

```python
ers_test(
    y: ndarray,
    maxlag: int | None = None,
    trend: str = "c",
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Time series data |
| `maxlag` | `int \| None` | `None` | Maximum lag order |
| `trend` | `str` | `"c"` | `'c'` (constant) or `'ct'` (constant + trend) |

::: chronobox.tests_stat.unit_root.ers_test
    options:
      show_root_heading: false
      show_source: true

---

### zivot_andrews_test

Zivot-Andrews unit root test allowing for a single structural break in the
intercept, trend, or both.

```python
zivot_andrews_test(
    y: ndarray,
    trim: float = 0.15,
    model: str = "c",
    maxlag: int | None = None,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Time series data |
| `trim` | `float` | `0.15` | Fraction of data to trim from each end |
| `model` | `str` | `"c"` | `'c'` (intercept break), `'t'` (trend break), `'ct'` (both) |
| `maxlag` | `int \| None` | `None` | Maximum lag order |

!!! tip "Break date"
    The estimated break date is stored in `result.additional_info['break_index']`.

::: chronobox.tests_stat.unit_root.zivot_andrews_test
    options:
      show_root_heading: false
      show_source: true

---

### lee_strazicich_test

Lee-Strazicich LM unit root test with one or two endogenous structural breaks.
Unlike Zivot-Andrews, this test is valid under both the null and alternative.

```python
lee_strazicich_test(
    y: ndarray,
    model: str = "break",
    breaks: int = 1,
    trim: float = 0.15,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Time series data |
| `model` | `str` | `"break"` | Type of break: `'break'` (level), `'trend'` (level + trend) |
| `breaks` | `int` | `1` | Number of breaks (1 or 2) |
| `trim` | `float` | `0.15` | Trimming fraction |

::: chronobox.tests_stat.unit_root.lee_strazicich_test
    options:
      show_root_heading: false
      show_source: true

---

### hegy_test

HEGY (Hylleberg-Engle-Granger-Yoo) test for seasonal unit roots.
Tests for unit roots at seasonal frequencies.

```python
hegy_test(
    y: ndarray,
    period: int = 4,
    maxlag: int | None = None,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Time series data |
| `period` | `int` | `4` | Seasonal period (4 for quarterly, 12 for monthly) |
| `maxlag` | `int \| None` | `None` | Maximum lag order |

::: chronobox.tests_stat.unit_root.hegy_test
    options:
      show_root_heading: false
      show_source: true

---

## Cointegration Tests

### engle_granger_test

Engle-Granger two-step cointegration test. Step 1: OLS regression of $y$ on $x$.
Step 2: ADF test on the residuals.

```python
engle_granger_test(
    y: ndarray,
    x: ndarray,
    trend: str = "c",
    maxlag: int | None = None,
    autolag: str = "AIC",
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Dependent variable |
| `x` | `ndarray` | *required* | Independent variable(s) |
| `trend` | `str` | `"c"` | Trend in the cointegrating regression |
| `maxlag` | `int \| None` | `None` | Maximum lag for ADF on residuals |
| `autolag` | `str` | `"AIC"` | Lag selection criterion |

### Example

```python
import numpy as np
from chronobox.tests_stat import engle_granger_test

rng = np.random.default_rng(42)
# Cointegrated pair: y = 2*x + noise, both I(1)
x = np.cumsum(rng.normal(size=200))
y = 2 * x + rng.normal(0, 0.5, 200)

result = engle_granger_test(y, x)
print(f"EG statistic: {result.statistic:.4f}")
print(f"Reject no cointegration? {result.reject_at_5pct}")
```

::: chronobox.tests_stat.cointegration.engle_granger_test
    options:
      show_root_heading: false
      show_source: true

---

### gregory_hansen_test

Gregory-Hansen cointegration test allowing for a structural break in the
cointegrating relationship.

```python
gregory_hansen_test(
    y: ndarray,
    x: ndarray,
    model: str = "c",
    trim: float = 0.15,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Dependent variable |
| `x` | `ndarray` | *required* | Independent variable(s) |
| `model` | `str` | `"c"` | `'c'` (level shift), `'ct'` (level + trend shift), `'cs'` (regime shift) |
| `trim` | `float` | `0.15` | Trimming fraction |

::: chronobox.tests_stat.cointegration.gregory_hansen_test
    options:
      show_root_heading: false
      show_source: true

---

### bounds_test

Pesaran-Shin-Smith (PSS) bounds test for cointegration in an ARDL framework.
Tests $H_0$: no long-run relationship.

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
| `y` | `ndarray` | *required* | Dependent variable |
| `x` | `ndarray` | *required* | Independent variable(s) |
| `lags` | `int \| None` | `None` | ARDL lag order. If None, auto-selected |
| `case` | `int` | `3` | PSS case (1-5). Case 3: unrestricted intercept, no trend |

!!! tip "Interpreting bounds test"
    If $F > I(1)$ upper bound → reject $H_0$ (cointegration exists).
    If $F < I(0)$ lower bound → fail to reject $H_0$.
    If $F$ falls between bounds → inconclusive.

::: chronobox.tests_stat.cointegration.bounds_test
    options:
      show_root_heading: false
      show_source: true

---

### phillips_ouliaris_test

Phillips-Ouliaris residual-based cointegration test.

```python
phillips_ouliaris_test(
    y: ndarray,
    x: ndarray,
    trend: str = "c",
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Dependent variable |
| `x` | `ndarray` | *required* | Independent variable(s) |
| `trend` | `str` | `"c"` | Trend specification |

::: chronobox.tests_stat.cointegration.phillips_ouliaris_test
    options:
      show_root_heading: false
      show_source: true

---

## Specification Tests

### ljung_box_test

Ljung-Box portmanteau test for autocorrelation in residuals.
Tests $H_0$: no autocorrelation up to lag $m$.

$$
Q_{LB} = T(T+2) \sum_{k=1}^{m} \frac{\hat{\rho}_k^2}{T-k} \sim \chi^2(m - p - q)
$$

```python
ljung_box_test(
    residuals: ndarray,
    lags: int = 10,
    model_df: int = 0,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `residuals` | `ndarray` | *required* | Residual series |
| `lags` | `int` | `10` | Number of lags $m$ to test |
| `model_df` | `int` | `0` | Degrees of freedom used by the model ($p + q$ for ARMA) |

### Example

```python
import numpy as np
from chronobox import ARIMA
from chronobox.tests_stat import ljung_box_test

rng = np.random.default_rng(42)
y = np.cumsum(rng.normal(size=200))

model = ARIMA(order=(1, 1, 1))
results = model.fit(y)

# Test residuals for remaining autocorrelation
lb = ljung_box_test(results.resid, lags=20, model_df=2)
print(f"Ljung-Box Q: {lb.statistic:.4f}, p-value: {lb.pvalue:.4f}")
print(f"White noise? {not lb.reject_at_5pct}")
```

::: chronobox.tests_stat.specification.ljung_box_test
    options:
      show_root_heading: false
      show_source: true

---

### breusch_godfrey_test

Breusch-Godfrey LM test for serial correlation. More general than Ljung-Box
as it is valid with lagged dependent variables.

```python
breusch_godfrey_test(
    residuals: ndarray,
    nlags: int = 1,
    x: ndarray | None = None,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `residuals` | `ndarray` | *required* | Residual series |
| `nlags` | `int` | `1` | Number of lags to test |
| `x` | `ndarray \| None` | `None` | Regressors from the original model |

::: chronobox.tests_stat.specification.breusch_godfrey_test
    options:
      show_root_heading: false
      show_source: true

---

### durbin_watson_test

Durbin-Watson test for first-order autocorrelation.

$$
DW = \frac{\sum_{t=2}^{T}(\hat{e}_t - \hat{e}_{t-1})^2}{\sum_{t=1}^{T}\hat{e}_t^2}
$$

$DW \approx 2$ indicates no autocorrelation; $DW < 2$ positive; $DW > 2$ negative.

```python
durbin_watson_test(
    residuals: ndarray,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `residuals` | `ndarray` | *required* | Residual series |

::: chronobox.tests_stat.specification.durbin_watson_test
    options:
      show_root_heading: false
      show_source: true

---

### arch_lm_test

ARCH-LM test for autoregressive conditional heteroskedasticity (Engle, 1982).
Tests $H_0$: no ARCH effects.

```python
arch_lm_test(
    residuals: ndarray,
    lags: int = 1,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `residuals` | `ndarray` | *required* | Residual series |
| `lags` | `int` | `1` | Number of ARCH lags to test |

::: chronobox.tests_stat.specification.arch_lm_test
    options:
      show_root_heading: false
      show_source: true

---

### white_test

White test for heteroskedasticity. Tests $H_0$: homoskedastic errors.

```python
white_test(
    residuals: ndarray,
    x: ndarray,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `residuals` | `ndarray` | *required* | Residual series |
| `x` | `ndarray` | *required* | Regressors from the original model |

::: chronobox.tests_stat.specification.white_test
    options:
      show_root_heading: false
      show_source: true

---

### jarque_bera_test

Jarque-Bera test for normality based on skewness and kurtosis.
Tests $H_0$: residuals are normally distributed.

$$
JB = \frac{T}{6}\left(S^2 + \frac{(K - 3)^2}{4}\right) \sim \chi^2(2)
$$

```python
jarque_bera_test(
    residuals: ndarray,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `residuals` | `ndarray` | *required* | Residual series |

::: chronobox.tests_stat.specification.jarque_bera_test
    options:
      show_root_heading: false
      show_source: true

---

### bds_test

BDS test for independence and identical distribution (Brock, Dechert & Scheinkman).

```python
bds_test(
    residuals: ndarray,
    max_dim: int = 6,
    epsilon: float | None = None,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `residuals` | `ndarray` | *required* | Residual series |
| `max_dim` | `int` | `6` | Maximum embedding dimension |
| `epsilon` | `float \| None` | `None` | Distance threshold. If None, uses standard deviation |

::: chronobox.tests_stat.specification.bds_test
    options:
      show_root_heading: false
      show_source: true

---

### reset_test

Ramsey RESET test for functional form misspecification.
Tests $H_0$: model is correctly specified.

```python
reset_test(
    residuals: ndarray,
    x: ndarray,
    fitted: ndarray,
    power: int = 3,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `residuals` | `ndarray` | *required* | Residual series |
| `x` | `ndarray` | *required* | Original regressors |
| `fitted` | `ndarray` | *required* | Fitted values |
| `power` | `int` | `3` | Highest power of fitted values to include |

::: chronobox.tests_stat.specification.reset_test
    options:
      show_root_heading: false
      show_source: true

---

## Structural Break Tests

### chow_test

Chow test for a structural break at a known date.
Tests $H_0$: no structural break.

```python
chow_test(
    y: ndarray,
    x: ndarray,
    break_point: int,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Dependent variable |
| `x` | `ndarray` | *required* | Regressors |
| `break_point` | `int` | *required* | Index of the break point |

::: chronobox.tests_stat.structural_breaks.chow_test
    options:
      show_root_heading: false
      show_source: true

---

### bai_perron_test

Bai-Perron test for multiple structural changes. Identifies up to $m$ break
dates by minimizing the global sum of squared residuals.

```python
bai_perron_test(
    y: ndarray,
    x: ndarray | None = None,
    max_breaks: int = 5,
    trim: float = 0.15,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Dependent variable |
| `x` | `ndarray \| None` | `None` | Regressors. If None, uses intercept only |
| `max_breaks` | `int` | `5` | Maximum number of breaks to test |
| `trim` | `float` | `0.15` | Minimum segment fraction |

!!! tip "Break dates"
    Estimated break dates are in `result.additional_info['break_dates']`.

::: chronobox.tests_stat.structural_breaks.bai_perron_test
    options:
      show_root_heading: false
      show_source: true

---

### cusum_test

CUSUM test for parameter stability (Brown, Durbin & Evans, 1975).
Tracks cumulative sum of recursive residuals.

```python
cusum_test(
    y: ndarray,
    x: ndarray | None = None,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Dependent variable |
| `x` | `ndarray \| None` | `None` | Regressors. If None, uses intercept only |

::: chronobox.tests_stat.structural_breaks.cusum_test
    options:
      show_root_heading: false
      show_source: true

---

### cusumsq_test

CUSUM of squares test. Tracks cumulative sum of squared recursive residuals
for detecting changes in variance.

```python
cusumsq_test(
    y: ndarray,
    x: ndarray | None = None,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Dependent variable |
| `x` | `ndarray \| None` | `None` | Regressors |

::: chronobox.tests_stat.structural_breaks.cusumsq_test
    options:
      show_root_heading: false
      show_source: true

---

### qlr_test

Quandt Likelihood Ratio (sup-Wald) test for an unknown structural break.
Computes Chow statistics over a range of candidate break dates and takes
the supremum.

```python
qlr_test(
    y: ndarray,
    x: ndarray,
    trim: float = 0.15,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `ndarray` | *required* | Dependent variable |
| `x` | `ndarray` | *required* | Regressors |
| `trim` | `float` | `0.15` | Fraction of data to trim from each end |

::: chronobox.tests_stat.structural_breaks.qlr_test
    options:
      show_root_heading: false
      show_source: true

---

## Testing Strategy Example

A comprehensive diagnostic workflow combining multiple tests:

```python
import numpy as np
from chronobox import ARIMA
from chronobox.tests_stat import (
    adf_test,
    kpss_test,
    ljung_box_test,
    arch_lm_test,
    jarque_bera_test,
)

rng = np.random.default_rng(42)
y = np.cumsum(rng.normal(size=300))

# Step 1: Unit root tests (confirmatory approach)
adf = adf_test(y, trend="c")
kpss = kpss_test(y, trend="c")
print(f"ADF  → Reject unit root? {adf.reject_at_5pct}")
print(f"KPSS → Reject stationarity? {kpss.reject_at_5pct}")

# Step 2: Fit model
model = ARIMA(order=(1, 1, 1))
results = model.fit(y)

# Step 3: Residual diagnostics
lb = ljung_box_test(results.resid, lags=20, model_df=2)
arch = arch_lm_test(results.resid, lags=5)
jb = jarque_bera_test(results.resid)

print(f"Ljung-Box   → White noise? {not lb.reject_at_5pct}")
print(f"ARCH-LM     → No ARCH? {not arch.reject_at_5pct}")
print(f"Jarque-Bera → Normal? {not jb.reject_at_5pct}")

# Step 4: ADF on residuals confirms stationarity after differencing
adf_resid = adf_test(results.resid)
print(f"ADF on residuals → Stationary? {adf_resid.reject_at_5pct}")
```

---

## See Also

- [Diagnostics Guide](../diagnostics/index.md) -- Step-by-step diagnostic workflows
- [Core API](core.md) -- `TimeSeriesResults` with built-in diagnostics
- [Visualization API](visualization.md) -- `plot_diagnostics()`, `plot_cusum()`
- [ARIMA Theory](../theory/arima-theory.md) -- Unit root and ARIMA background
- [VAR Theory](../theory/var-theory.md) -- VAR stability and Granger causality
