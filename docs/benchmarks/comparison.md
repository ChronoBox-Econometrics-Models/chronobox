---
title: "Feature Comparison"
description: "Comprehensive feature matrix comparing ChronoBox with statsmodels, R forecast, and R vars — models, tests, visualization, and API design."
---

# Feature Comparison

Side-by-side comparison of ChronoBox with the main alternatives in Python and R.

---

## Univariate Models

| Feature | ChronoBox | statsmodels | R forecast | R tseries |
|---------|:---------:|:-----------:|:----------:|:---------:|
| ARIMA | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| SARIMA | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |
| ARFIMA (fractional) | :white_check_mark: | :x: | :white_check_mark: | :x: |
| auto_arima | :white_check_mark: | :x:^1^ | :white_check_mark: | :x: |
| ETS (state-space) | :white_check_mark: | :white_check_mark:^2^ | :white_check_mark: | :x: |
| auto_ets | :white_check_mark: | :x: | :white_check_mark: | :x: |
| Theta method | :white_check_mark: | :x: | :white_check_mark: | :x: |
| Holt-Winters | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |
| CSS estimation | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |
| CSS-MLE hybrid | :white_check_mark: | :x: | :white_check_mark: | :x: |
| Exact MLE (Kalman) | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |

<small>
^1^ statsmodels has no built-in `auto_arima` — users typically use `pmdarima` (a wrapper).<br>
^2^ statsmodels `ExponentialSmoothing` does not use full state-space formulation.
</small>

---

## Multivariate Models

| Feature | ChronoBox | statsmodels | R vars | R tsDyn |
|---------|:---------:|:-----------:|:------:|:-------:|
| VAR (reduced form) | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| SVAR (structural) | :white_check_mark: | :x:^3^ | :white_check_mark: | :x: |
| VECM (cointegrated) | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| BVAR (Bayesian) | :white_check_mark: | :x: | :x:^4^ | :x: |
| FAVAR | :white_check_mark: | :x: | :x: | :x: |
| TVP-VAR | :white_check_mark: | :x: | :x: | :white_check_mark: |
| GVAR (global) | :white_check_mark: | :x: | :x: | :x: |
| ARDL / Bounds test | :white_check_mark: | :white_check_mark: | :x:^5^ | :x: |
| Short-run restrictions | :white_check_mark: | :x: | :white_check_mark: | :x: |
| Long-run restrictions | :white_check_mark: | :x: | :white_check_mark: | :x: |
| Sign restrictions | :white_check_mark: | :x: | :x: | :x: |

<small>
^3^ statsmodels does not have a dedicated SVAR class.<br>
^4^ R `bvarsv` and `BVAR` packages exist but are not part of `vars`.<br>
^5^ R `dynamac` and `ARDL` packages provide this functionality separately.
</small>

---

## Analysis Tools

| Feature | ChronoBox | statsmodels | R vars | R forecast |
|---------|:---------:|:-----------:|:------:|:----------:|
| IRF (orthogonal) | :white_check_mark: | :white_check_mark: | :white_check_mark: | — |
| IRF (generalized) | :white_check_mark: | :white_check_mark: | :white_check_mark: | — |
| FEVD | :white_check_mark: | :white_check_mark: | :white_check_mark: | — |
| Historical decomposition | :white_check_mark: | :x: | :x:^6^ | — |
| Counterfactual analysis | :white_check_mark: | :x: | :x: | — |
| Granger causality | :white_check_mark: | :white_check_mark: | :white_check_mark: | — |
| Spillover analysis (DY) | :white_check_mark: | :x: | :x:^7^ | — |
| Connectedness measures | :white_check_mark: | :x: | :x: | — |

<small>
^6^ R `svars` package provides historical decomposition.<br>
^7^ R `frequencyConnectedness` package provides DY spillover analysis.
</small>

---

## Statistical Tests

| Test | ChronoBox | statsmodels | R (package) |
|------|:---------:|:-----------:|:-----------:|
| ADF (unit root) | :white_check_mark: | :white_check_mark: | :white_check_mark: (tseries) |
| PP (unit root) | :white_check_mark: | :white_check_mark: | :white_check_mark: (tseries) |
| KPSS (stationarity) | :white_check_mark: | :white_check_mark: | :white_check_mark: (tseries) |
| Zivot-Andrews | :white_check_mark: | :white_check_mark: | :white_check_mark: (urca) |
| Lee-Strazicich | :white_check_mark: | :x: | :white_check_mark: (strucchange) |
| Johansen cointegration | :white_check_mark: | :white_check_mark: | :white_check_mark: (urca) |
| Engle-Granger | :white_check_mark: | :white_check_mark: | :white_check_mark: (tseries) |
| Bounds test (ARDL) | :white_check_mark: | :x: | :white_check_mark: (dynamac) |
| Ljung-Box | :white_check_mark: | :white_check_mark: | :white_check_mark: (stats) |
| Jarque-Bera | :white_check_mark: | :white_check_mark: | :white_check_mark: (tseries) |
| ARCH-LM | :white_check_mark: | :white_check_mark: | :white_check_mark: (FinTS) |
| Breusch-Godfrey | :white_check_mark: | :white_check_mark: | :white_check_mark: (lmtest) |

---

## Filters and Decomposition

| Feature | ChronoBox | statsmodels | R (package) |
|---------|:---------:|:-----------:|:-----------:|
| Hodrick-Prescott | :white_check_mark: | :white_check_mark: | :white_check_mark: (mFilter) |
| Baxter-King | :white_check_mark: | :white_check_mark: | :white_check_mark: (mFilter) |
| Christiano-Fitzgerald | :white_check_mark: | :white_check_mark: | :white_check_mark: (mFilter) |
| Hamilton filter | :white_check_mark: | :x: | :white_check_mark: (neverhpfilter) |
| Beveridge-Nelson | :white_check_mark: | :x: | :white_check_mark: (mFilter) |
| STL decomposition | :white_check_mark: | :white_check_mark: | :white_check_mark: (stats) |
| Classical decomposition | :white_check_mark: | :white_check_mark: | :white_check_mark: (stats) |
| X-13 ARIMA-SEATS | :white_check_mark: | :x:^8^ | :white_check_mark: (seasonal) |

<small>
^8^ statsmodels has `x13_arima_analysis` but requires external X-13 binary.
</small>

---

## Visualization

| Feature | ChronoBox | statsmodels | R (package) |
|---------|:---------:|:-----------:|:-----------:|
| Time series plots | :white_check_mark: | :white_check_mark: | :white_check_mark: (base) |
| ACF / PACF | :white_check_mark: | :white_check_mark: | :white_check_mark: (forecast) |
| IRF plots | :white_check_mark: | :white_check_mark: | :white_check_mark: (vars) |
| FEVD plots | :white_check_mark: | :white_check_mark: | :white_check_mark: (vars) |
| Forecast plots | :white_check_mark: | :white_check_mark: | :white_check_mark: (forecast) |
| Filter plots | :white_check_mark: | :x: | :white_check_mark: (mFilter) |
| Spillover network | :white_check_mark: | :x: | :x: |
| Diagnostic dashboard | :white_check_mark: | :x: | :white_check_mark: (forecast) |
| Themeable plots | :white_check_mark: | :x: | :white_check_mark: (ggplot2) |

---

## API and Usability

| Feature | ChronoBox | statsmodels | R packages |
|---------|:---------:|:-----------:|:----------:|
| Consistent API across models | :white_check_mark: | Partial | :x:^9^ |
| Built-in datasets | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Report generation (HTML/LaTeX) | :white_check_mark: | :x: | :x: |
| CLI interface | :white_check_mark: | :x: | :x: |
| Experiment framework | :white_check_mark: | :x: | :x: |
| Type hints | :white_check_mark: | Partial | — |
| NumPy-style docstrings | :white_check_mark: | :white_check_mark: | — |

<small>
^9^ R time series ecosystem is fragmented across many packages with different APIs.
</small>

---

## Performance Summary

| Category | Fastest | Notes |
|----------|---------|-------|
| **ARIMA estimation** | **ChronoBox** | 2–3× faster than statsmodels |
| **auto_arima** | **ChronoBox** | 1.5× faster than R, 2.5× faster than statsmodels |
| **VAR estimation** | **R vars** | C/Fortran backend advantage |
| **ETS estimation** | **R forecast** | C++ backend, but ChronoBox is close |
| **auto_ets** | **R forecast** | Slight edge; ChronoBox is 3.5× faster than statsmodels |
| **IRF computation** | **R vars** | Bootstrap is faster in R |
| **BVAR** | **ChronoBox** | Only comprehensive Python option |
| **SVAR** | ChronoBox / R | Both available; statsmodels lacks SVAR |

---

## Conclusions and Recommendations

!!! tip "When to use ChronoBox"

    - **Python-first workflow**: ChronoBox integrates naturally with pandas, NumPy, and the Python data science stack
    - **Unified API**: one consistent interface for ARIMA, ETS, VAR, SVAR, BVAR, VECM — no need for multiple packages
    - **Advanced models**: FAVAR, TVP-VAR, GVAR, and spillover analysis in a single library
    - **Reporting**: built-in HTML/LaTeX report generation and CLI
    - **Speed**: fastest Python option for ARIMA; competitive with R across the board

!!! info "When to consider alternatives"

    - **R forecast/vars**: if you need the absolute fastest ETS/VAR estimation, or if your workflow is already in R
    - **statsmodels**: if you need a well-established, widely-tested library and only require basic ARIMA/VAR functionality
    - **Specialized R packages**: for niche features like `svars` (narrative sign restrictions) or `bvarsv` (stochastic volatility BVAR)

---

## See Also

- [ARIMA Benchmark](arima.md) — detailed ARIMA comparison
- [VAR Benchmark](var.md) — detailed VAR comparison
- [ETS Benchmark](ets.md) — detailed ETS comparison
