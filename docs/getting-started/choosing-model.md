---
title: Choosing a Model
description: Decision guide for selecting the right time series model in ChronoBox based on your data and analysis goals
---

# Choosing a Model

Selecting the right model is the most consequential decision in a time series analysis. A misspecified model can produce misleading forecasts, spurious impulse responses, or invalid policy conclusions --- no amount of diagnostic tweaking will fix a fundamentally wrong choice.

This guide walks you through the decision process: from the structure of your data to the specific ChronoBox model that fits your needs.

---

## Decision Tree

Start here. Answer the questions in order and follow the arrows.

```text
                         How many series?
                              │
              ┌───────────────┴───────────────┐
              │                               │
          1 series                        2+ series
         UNIVARIATE                     MULTIVARIATE
              │                               │
     ┌────────┴────────┐             ┌────────┴────────┐
     │                 │             │                  │
 Seasonality?    Want automatic?   Cointegration?    Special structure?
     │              │                │                  │
  ┌──┴──┐      ┌───┴───┐       ┌───┴───┐         ┌───┴──────────────┐
  │     │      │       │       │       │         │    │    │    │    │
 Yes   No    auto_   auto_   Yes     No       SVAR BVAR FAVAR TVP  GVAR
  │     │    arima    ets      │       │              -VAR
  │     │                    VECM    VAR
  │     │
SARIMA ARIMA
ETS    ETS
(seasonal) (simple)
```

!!! tip "Start simple"
    When in doubt, start with the simplest model that matches your data structure. A well-specified ARIMA often outperforms a complex VAR with too many parameters. You can always move to more complex models if diagnostics suggest the simpler one is inadequate.

---

## Univariate Models

Use these when you have a **single time series** and want to model its dynamics or generate forecasts.

### ARIMA vs ETS vs Theta

| Feature | ARIMA | ETS | Theta |
|---------|-------|-----|-------|
| **Approach** | Differencing + ARMA errors | Error-Trend-Seasonality decomposition | Modified exponential smoothing |
| **Seasonality** | SARIMA with $(P,D,Q)_m$ | Built-in seasonal components | Limited |
| **Interpretability** | AR/MA coefficients | Level, trend, seasonal states | Single $\theta$ parameter |
| **Long memory** | ARFIMA extension | No | No |
| **Best for** | Stationary/differenced data, Box-Jenkins methodology | Data with clear trend/seasonal patterns | Quick benchmarks, M-competition style |

#### When to Use Each

=== "ARIMA"

    Choose ARIMA when:

    - Your data becomes stationary after differencing
    - You want to follow the Box-Jenkins methodology (identify, estimate, diagnose)
    - You need fractional integration (ARFIMA) for long-memory processes
    - You're working with economic/financial data where AR and MA components have clear interpretations

    ```python
    from chronobox import ARIMA

    model = ARIMA(order=(1, 1, 1), seasonal_order=(0, 1, 1, 12))
    result = model.fit(data)
    ```

=== "ETS"

    Choose ETS when:

    - Your data has clear trend and/or seasonal components
    - You want a model that naturally decomposes the series into interpretable parts
    - You need to choose between additive and multiplicative seasonality
    - Forecasting accuracy is the primary goal (ETS is competitive in forecast competitions)

    ```python
    from chronobox import ETS

    # ETS(A,Ad,M): additive error, damped trend, multiplicative season
    model = ETS(error="add", trend="add", seasonal="mul", damped_trend=True)
    result = model.fit(data)
    ```

=== "Theta"

    Choose Theta when:

    - You need a quick, robust baseline forecast
    - Model simplicity is valued over interpretability
    - You're running large-scale comparisons across many series

    ```python
    from chronobox import Theta

    model = Theta()
    result = model.fit(data)
    ```

### Automatic Selection

When you're unsure about the model specification, let ChronoBox search for you:

```python
from chronobox import auto_arima, auto_ets

# Automatic ARIMA: searches over (p,d,q) × (P,D,Q)_m
best_arima = auto_arima(data, seasonal=True, m=12, information_criterion="aicc")

# Automatic ETS: searches over error, trend, seasonal combinations
best_ets = auto_ets(data, seasonal_period=12)
```

!!! info "How automatic selection works"
    Both `auto_arima` and `auto_ets` perform a structured search over model specifications, selecting the one that minimizes an information criterion (AICc by default). `auto_arima` uses a stepwise algorithm inspired by Hyndman & Khandakar (2008), while `auto_ets` evaluates all valid ETS combinations.

---

## Multivariate Models

Use these when you have **two or more time series** that may influence each other.

### VAR vs VECM

The first question for multivariate data: **are the series cointegrated?**

- **Cointegration** means the series share a common stochastic trend --- they may wander individually, but a linear combination of them is stationary. Classic example: short-term and long-term interest rates.

| Feature | VAR | VECM |
|---------|-----|------|
| **Data requirement** | Stationary (or first-differenced) | Non-stationary with cointegration |
| **Long-run relationships** | Not captured | Explicitly modeled via cointegrating vectors |
| **Short-run dynamics** | Yes | Yes |
| **Impulse responses** | Standard IRF | IRF with long-run constraints |
| **When to use** | Stationary system, no long-run equilibrium needed | Series are $I(1)$ and cointegrated |

```python
from chronobox.tests_stat import johansen_test

# Step 1: Test for cointegration
coint = johansen_test(data, det_order=0, k_ar_diff=2)
print(coint)

# Step 2: Choose based on result
if coint.r > 0:
    # Cointegration found → use VECM
    from chronobox import VECM
    model = VECM(k_ar_diff=2, coint_rank=coint.r)
else:
    # No cointegration → use VAR on stationary data
    from chronobox import VAR
    model = VAR(maxlags=4, ic="aic")

result = model.fit(data)
```

!!! warning "Don't difference away cointegration"
    If your series are cointegrated, fitting a VAR in first differences discards the long-run equilibrium information. Use VECM instead --- it captures both the short-run dynamics and the error-correction mechanism that pulls the system back toward equilibrium.

### Structural and Advanced Models

When standard VAR/VECM is not enough, ChronoBox offers specialized extensions:

| Model | When to Use | Key Feature |
|-------|-------------|-------------|
| **SVAR** | You have economic theory about contemporaneous relationships | Structural identification via short-run or long-run restrictions |
| **BVAR** | Small samples, many variables, or you want to incorporate prior beliefs | Bayesian estimation with Minnesota, Normal-Wishart, or custom priors |
| **FAVAR** | Large datasets with many potential predictors | Combines factor analysis with VAR --- latent factors summarize large panels |
| **TVP-VAR** | Relationships change over time (regime shifts, evolving transmission) | Time-varying parameters estimated via Kalman filter |
| **GVAR** | Multi-country or multi-region analysis | Models interdependencies across countries with trade-weighted foreign variables |

=== "SVAR"

    Use when you need **causally interpretable** impulse responses. Standard VAR IRFs are not structural --- they depend on the ordering of variables (Cholesky decomposition). SVAR imposes theoretically motivated restrictions.

    ```python
    from chronobox import SVAR

    # Short-run restrictions: A @ e_t = B @ u_t
    model = SVAR(var_result, svar_type="AB")
    result = model.fit(A=A_matrix, B=B_matrix)
    ```

=== "BVAR"

    Use when you have **more variables than observations** or want to shrink parameters toward a prior. Minnesota priors are the standard choice for macroeconomic VARs.

    ```python
    from chronobox import BVAR

    model = BVAR(lags=4, prior="minnesota", tightness=0.1)
    result = model.fit(data)
    ```

=== "FAVAR"

    Use when you want to include information from a **large set of indicators** without estimating a huge VAR. FAVAR extracts a few latent factors from the panel and includes them in the VAR.

    ```python
    from chronobox import FAVAR

    model = FAVAR(n_factors=3, lags=4)
    result = model.fit(y=policy_vars, x=large_panel)
    ```

=== "TVP-VAR"

    Use when you suspect the **transmission mechanism has changed** over time --- e.g., monetary policy effectiveness before and after inflation targeting.

    ```python
    from chronobox import TVPVAR

    model = TVPVAR(lags=2)
    result = model.fit(data)
    ```

=== "GVAR"

    Use for **cross-country** or **cross-region** macro analysis. Each country has its own VARX model, linked through trade-weighted foreign variables.

    ```python
    from chronobox import GVAR

    model = GVAR(lags_domestic=2, lags_foreign=1)
    result = model.fit(country_data, weight_matrix=trade_weights)
    ```

---

## Complementary Tools

These are not standalone models but essential tools that complement your modeling workflow.

### Economic Filters

Extract trend and cyclical components from a time series:

| Filter | Best For | Key Characteristic |
|--------|----------|-------------------|
| **Hodrick-Prescott** | Trend-cycle decomposition (standard in macro) | Smoothness parameter $\lambda$ controls trend flexibility |
| **Baxter-King** | Isolating business cycle frequencies | Band-pass filter with symmetric weights |
| **Christiano-Fitzgerald** | Asymmetric band-pass filtering | Allows different frequency bands, handles endpoints better |
| **Hamilton** | Robust trend extraction without endpoint bias | Regression-based, avoids the HP filter's well-known problems |
| **Beveridge-Nelson** | Permanent vs transitory decomposition | Based on the ARIMA representation of the series |

!!! tip "HP filter or Hamilton filter?"
    The Hodrick-Prescott filter is ubiquitous in macroeconomics but has well-documented problems: endpoint instability, spurious cycles, and sensitivity to $\lambda$. Hamilton (2018) proposes a regression-based alternative that avoids these issues. For research, consider reporting both.

### ARDL & Error Correction

| Model | When to Use |
|-------|-------------|
| **ARDL** | Mixed integration orders --- some variables $I(0)$, some $I(1)$ |
| **ECM** | After establishing cointegration, model the adjustment process |

The ARDL bounds test (Pesaran, Shin & Smith, 2001) is especially useful when you're unsure about the integration order of your variables --- unlike Johansen, it works regardless of whether regressors are $I(0)$ or $I(1)$.

```python
from chronobox import ARDL

model = ARDL(lags=4, order={"x1": 2, "x2": 3})
result = model.fit(y=y, exog=exog)

# Bounds test for cointegration
bounds = result.bounds_test()
print(bounds)
```

### Spillover Analysis

Measure **connectedness** and **risk transmission** across markets or variables:

| Type | Use Case |
|------|----------|
| **Static Spillover** | Overall connectedness at a point in time (Diebold & Yilmaz, 2012) |
| **Dynamic Spillover** | How connectedness evolves over rolling windows |

```python
from chronobox import Spillover

sp = Spillover(var_result, horizon=10)
table = sp.connectedness_table()
sp.plot_dynamic(window=200)
```

---

## Model Comparison Table

A comprehensive reference of all models available in ChronoBox:

| Model | Type | Series | Seasonality | Cointegration | Primary Use Case |
|-------|------|--------|-------------|---------------|-----------------|
| **ARIMA** | Univariate | 1 | No | --- | General univariate modeling |
| **SARIMA** | Univariate | 1 | Yes | --- | Seasonal univariate data |
| **ARFIMA** | Univariate | 1 | No | --- | Long-memory processes |
| **Auto-ARIMA** | Univariate | 1 | Optional | --- | Automatic order selection |
| **ETS** | Univariate | 1 | Optional | --- | Trend-seasonal decomposition & forecasting |
| **Auto-ETS** | Univariate | 1 | Optional | --- | Automatic ETS specification |
| **Theta** | Univariate | 1 | Limited | --- | Quick baseline forecasts |
| **VAR** | Multivariate | 2+ | No | No | Stationary multivariate dynamics |
| **VECM** | Multivariate | 2+ | No | Yes | Cointegrated systems |
| **SVAR** | Structural | 2+ | No | No | Structural identification & causal IRF |
| **BVAR** | Structural | 2+ | No | No | Bayesian shrinkage, small samples |
| **FAVAR** | Structural | 2+ (large panel) | No | No | Factor-augmented VAR |
| **TVP-VAR** | Structural | 2+ | No | No | Time-varying parameters |
| **GVAR** | Structural | 2+ (multi-country) | No | Optional | Cross-country macro linkages |
| **ARDL** | Multivariate | 1 dep + regressors | No | Mixed $I(0)/I(1)$ | Mixed integration, bounds test |
| **HP Filter** | Filter | 1 | --- | --- | Trend-cycle decomposition |
| **BK Filter** | Filter | 1 | --- | --- | Business cycle isolation |
| **CF Filter** | Filter | 1 | --- | --- | Asymmetric band-pass |
| **Hamilton** | Filter | 1 | --- | --- | Robust trend extraction |
| **BN Decomposition** | Filter | 1 | --- | --- | Permanent-transitory decomposition |
| **Spillover** | Connectedness | 2+ | No | No | Risk transmission, market connectedness |

---

## By Use Case

Not sure which model fits your research question? Start from the use case:

### Forecasting a Single Variable

> *"I need to forecast monthly CPI inflation 12 months ahead."*

**Recommended**: `auto_arima` or `auto_ets` for automatic selection; manual SARIMA if you have domain knowledge about the seasonal structure.

```python
from chronobox import auto_arima

best = auto_arima(cpi, seasonal=True, m=12)
forecast = best.forecast(steps=12)
```

!!! tip "Forecast combination"
    When ARIMA and ETS produce different forecasts, consider averaging them. Simple forecast combinations often outperform individual models (Timmermann, 2006).

### Monetary Policy Analysis

> *"I want to estimate the effect of an interest rate shock on output and inflation."*

**Recommended**: **SVAR** with structural identification. Use Cholesky ordering for a baseline, then impose theory-motivated restrictions for robustness.

See: [SVAR User Guide](../user-guide/svar/svar.md) | [SVAR Theory](../theory/svar-theory.md)

### Shock Transmission Across Markets

> *"How do volatility shocks transmit across equity markets?"*

**Recommended**: **Spillover** analysis (Diebold-Yilmaz framework). Static table for overall connectedness, dynamic rolling for evolution over time.

See: [Spillover User Guide](../user-guide/spillover/index.md) | [Spillover Theory](../theory/spillover-theory.md)

### Business Cycle Extraction

> *"I need to extract the output gap from real GDP."*

**Recommended**: **Hamilton filter** for a robust estimate, **HP filter** for comparability with the literature. Report both.

See: [Filters User Guide](../user-guide/filters/index.md) | [Filter Theory](../theory/filters-theory.md)

### Long-Run Equilibrium Relationships

> *"Do consumption and income share a long-run equilibrium?"*

**Recommended**: Test with **Johansen** cointegration test. If cointegrated, use **VECM**. If integration orders are mixed or uncertain, use **ARDL bounds test**.

See: [VECM User Guide](../user-guide/var/vecm.md) | [ARDL User Guide](../user-guide/ardl/ardl.md) | [VECM Theory](../theory/vecm-theory.md)

### Large-Scale Macro Modeling

> *"I have 100+ macroeconomic indicators and want to study monetary transmission."*

**Recommended**: **FAVAR** --- extract latent factors from the large panel, then estimate a VAR with the policy variable and the factors.

See: [FAVAR User Guide](../user-guide/svar/favar.md)

### Cross-Country Analysis

> *"How do shocks in the US propagate to European economies?"*

**Recommended**: **GVAR** --- each country has its own model, linked through trade weights. Captures both domestic dynamics and international spillovers.

See: [GVAR User Guide](../user-guide/svar/gvar.md)

---

## Quick Reference Flowchart

Use this condensed checklist when starting a new analysis:

1. **Count your series**: 1 series $\to$ univariate; 2+ series $\to$ multivariate
2. **Check stationarity**: `adf_test()`, `kpss_test()` --- determines differencing order
3. **If univariate**: try `auto_arima` or `auto_ets` first; refine manually if needed
4. **If multivariate, test cointegration**: `johansen_test()` --- VAR if no, VECM if yes
5. **If you need structure**: SVAR for restrictions, BVAR for priors, FAVAR for large panels
6. **Add filters**: HP/Hamilton for trend-cycle, Spillover for connectedness
7. **Always diagnose**: Ljung-Box, stability checks, residual plots --- regardless of the model

!!! note "No single model is best"
    The "best" model depends on the question, the data, and the assumptions you're willing to make. Diagnostics tell you whether a model is *adequate* --- they can't tell you it's the *right* model for your research question. Let theory guide the choice; let diagnostics validate it.

---

## Next Steps

<div class="grid cards" markdown>

- :material-chart-bell-curve-cumulative: **[ARIMA Models](../user-guide/arima/index.md)**

    Univariate modeling with ARIMA, SARIMA, ARFIMA, and Auto-ARIMA

- :material-sine-wave: **[ETS Models](../user-guide/ets/index.md)**

    Exponential smoothing with 30 model specifications

- :material-vector-polyline: **[VAR & VECM](../user-guide/var/index.md)**

    Multivariate dynamics, cointegration, impulse responses

- :material-axis-arrow: **[SVAR & Advanced](../user-guide/svar/index.md)**

    Structural identification, Bayesian VARs, FAVAR, TVP-VAR, GVAR

- :material-filter-outline: **[Economic Filters](../user-guide/filters/index.md)**

    HP, Baxter-King, Christiano-Fitzgerald, Hamilton, Beveridge-Nelson

- :material-transit-connection-variant: **[Spillover Analysis](../user-guide/spillover/index.md)**

    Diebold-Yilmaz connectedness and dynamic spillover plots

</div>
