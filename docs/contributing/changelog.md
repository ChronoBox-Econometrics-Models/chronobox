---
title: "Changelog"
description: "ChronoBox version history — all releases with key changes, migration notes, and breaking changes."
---

# Changelog

All notable changes to ChronoBox are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and ChronoBox adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Sections**: Added, Changed, Fixed, Deprecated, Removed, Security, Performance.

---

## [Unreleased]

### Added

- Comprehensive documentation site (MkDocs Material)
- Tutorials for all model families
- Benchmark comparisons against R and Stata

---

## [0.1.0] — 2026-04

### Summary

**Initial Release — Complete Time Series Econometrics Suite**

ChronoBox v0.1.0 provides a unified, Pythonic interface for univariate and multivariate time series econometrics, with extensive diagnostics, visualization, and reporting.

### Added

#### ARIMA Family

- **ARIMA** — Box-Jenkins ARIMA(p,d,q) with MLE estimation
- **SARIMA** — Seasonal ARIMA(p,d,q)(P,D,Q)[s] with seasonal differencing
- **ARFIMA** — Fractionally integrated ARIMA for long memory processes
- **Auto-ARIMA** — Automatic order selection via stepwise AIC/BIC minimization

#### ETS (Exponential Smoothing)

- **30 ETS models** — Full error/trend/seasonal taxonomy (A/M/N × A/Ad/M/Md/N × A/M/N)
- **Auto-ETS** — Automatic model selection across all valid combinations
- **Theta method** — Assimakopoulos & Nikolopoulos (2000) decomposition-based forecasting

#### VAR & VECM

- **VAR** — Vector Autoregression with OLS estimation and lag selection (AIC, BIC, HQIC, FPE)
- **VECM** — Vector Error Correction Model with Johansen cointegration rank selection
- **IRF** — Impulse Response Functions (orthogonalized and generalized) with bootstrap confidence bands
- **FEVD** — Forecast Error Variance Decomposition (Cholesky and generalized)
- **Granger Causality** — Pairwise and multivariate Granger causality tests

#### SVAR & Advanced Multivariate Models

- **SVAR** — Structural VAR with short-run and long-run restrictions (Blanchard-Quah)
- **BVAR** — Bayesian VAR with Minnesota, Normal-Wishart, and SSVS priors
- **FAVAR** — Factor-Augmented VAR with principal component extraction
- **TVP-VAR** — Time-Varying Parameter VAR via Kalman filter (kalmanbox integration)
- **GVAR** — Global VAR for multi-country/multi-region analysis

#### Economic Filters

- **Hodrick-Prescott** — Trend-cycle decomposition with customizable smoothing parameter
- **Baxter-King** — Symmetric band-pass filter for business cycle extraction
- **Christiano-Fitzgerald** — Asymmetric band-pass filter (fixed-length and full-sample)
- **Hamilton Filter** — Regression-based alternative to HP filter (Hamilton 2018)
- **Beveridge-Nelson** — Permanent-transitory decomposition for I(1) series

#### ARDL & Error Correction

- **ARDL** — Autoregressive Distributed Lag model with automatic lag selection
- **ECM** — Error Correction Model derived from ARDL specification
- **Bounds test** — Pesaran, Shin & Smith (2001) cointegration bounds testing

#### Spillover Analysis

- **Diebold-Yilmaz** — Static and dynamic spillover indices
- **Spillover table** — Directional (from/to) spillover decomposition
- **Rolling spillover** — Time-varying connectedness with customizable window
- **Net spillover** — Net pairwise directional connectedness

#### Statistical Tests

- **Unit root**: ADF, Phillips-Perron, KPSS, Zivot-Andrews, Lee-Strazicich
- **Cointegration**: Johansen trace/max eigenvalue, Engle-Granger, bounds test
- **Structural breaks**: CUSUM, Chow, Bai-Perron
- **Specification**: Ljung-Box, Breusch-Godfrey, ARCH-LM, Jarque-Bera
- **VAR stability**: Eigenvalue stability condition, Portmanteau test
- **Lag selection**: AIC, BIC, HQIC, FPE with sequential testing

#### Visualization

- **Time series plots** — Line plots with trend overlays and recession shading
- **ACF/PACF** — Autocorrelation and partial autocorrelation with confidence bands
- **IRF plots** — Impulse response with confidence intervals (single and grid)
- **FEVD plots** — Stacked area and bar chart variance decomposition
- **Forecast plots** — Point forecasts with fan charts and prediction intervals
- **Filter plots** — Trend-cycle decomposition with original series overlay
- **Spillover plots** — Network graphs and heatmaps for connectedness
- **Diagnostic plots** — Residual analysis, QQ plots, stability plots
- **Themes** — Professional, academic, and presentation themes

#### Reports & CLI

- **HTML reports** — Comprehensive model reports with interactive elements
- **Summary tables** — LaTeX-ready coefficient and diagnostic tables
- **CLI** — Command-line interface for quick model fitting and diagnostics
- **Experiment workflow** — Unified fit → diagnose → compare → report pipeline

---

## Versioning Policy

ChronoBox uses [Semantic Versioning](https://semver.org/):

| Component | When incremented |
|---|---|
| **Major** (X.0.0) | Incompatible API changes |
| **Minor** (0.X.0) | New features, backward compatible |
| **Patch** (0.0.X) | Bug fixes, backward compatible |

---

## See Also

- [Contributing Guide](contributing.md) — How to contribute
- [Roadmap](roadmap.md) — Planned features
- [API Reference](../api/index.md) — Full API documentation
