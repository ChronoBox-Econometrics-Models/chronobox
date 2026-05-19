---
title: "API Reference"
description: "Complete API reference for the chronobox time series econometrics library"
---

# API Reference

!!! info "Module"
    **Import**: `import chronobox`
    **Source**: `chronobox/`
    **Version**: `chronobox.__version__`

## Overview

ChronoBox provides a unified API for time series econometrics. All models follow a consistent
**construct → fit → results** pattern:

```python
import chronobox as cb

model = cb.ARIMA(order=(1, 1, 1))       # 1. Construct
results = model.fit(y)                    # 2. Fit
results.summary()                         # 3. Inspect results
forecasts = results.forecast(steps=12)    # 4. Use results
```

## API Conventions

### The `model.fit()` Pattern

Every model in ChronoBox follows the same interface:

| Step | Method | Returns |
|------|--------|---------|
| Construct | `Model(params...)` | Model instance |
| Fit | `model.fit(endog, ...)` | Results object |
| Summary | `results.summary()` | Formatted string |
| Forecast | `results.forecast(steps)` | Array or dict |

### Results Pattern

All results objects provide:

- **Parameter access**: `results.params`, `results.se`, `results.pvalues`
- **Information criteria**: `results.aic`, `results.bic`, `results.aicc`
- **Diagnostics**: `results.resid`, `results.fittedvalues`
- **Summary**: `results.summary()` for formatted output
- **Forecasting**: `results.forecast(steps=h)` for out-of-sample predictions

## Module Organization

### Core

Foundation classes for data handling, results, and configuration.

| Class | Description |
|-------|-------------|
| [`TimeSeriesData`](core.md#timeseries-data) | Time series container with frequency metadata |
| [`TimeSeriesResults`](core.md#timeseries-results) | Base result class for all models |
| [`ChronoBoxConfig`](core.md#configuration) | Global configuration settings |
| [`LagPolynomial`](core.md#lag-polynomial) | Lag polynomial algebra |

### Models

#### ARIMA Family

| Class / Function | Description |
|------------------|-------------|
| [`ARIMA`](arima.md#arima) | ARIMA(p,d,q) and SARIMA(p,d,q)(P,D,Q)[s] |
| [`ARFIMA`](arima.md#arfima) | Fractionally integrated ARIMA |
| [`auto_arima`](arima.md#auto-arima) | Automatic ARIMA order selection |

#### ETS (Exponential Smoothing)

| Class / Function | Description |
|------------------|-------------|
| [`ETS`](ets.md#ets) | All 30 ETS state-space models |
| [`HoltWinters`](ets.md#holt-winters) | Classical Holt-Winters method |
| [`ThetaMethod`](ets.md#theta-method) | Theta method (Assimakopoulos & Nikolopoulos) |
| [`auto_ets`](ets.md#auto-ets) | Automatic ETS model selection |

#### Decomposition

| Class | Description |
|-------|-------------|
| [`STL`](decomposition.md#stl) | Seasonal-Trend decomposition using LOESS |
| [`X13Wrapper`](decomposition.md#x-13-arima-seats) | X-13 ARIMA-SEATS wrapper |
| [`ClassicalDecomposition`](decomposition.md#classical-decomposition) | Classical additive/multiplicative decomposition |

#### VAR & Multivariate

| Class / Function | Description |
|------------------|-------------|
| [`VAR`](var.md#var) | Vector Autoregression VAR(p) |
| [`VECM`](var.md#vecm) | Vector Error Correction Model |
| [`IRF`](var.md#impulse-response-functions) | Impulse Response Functions |
| [`FEVD`](var.md#forecast-error-variance-decomposition) | Forecast Error Variance Decomposition |
| [`granger_causality`](var.md#granger-causality) | Granger causality test |

#### SVAR & Advanced Models

| Class | Description |
|-------|-------------|
| [`SVAR`](svar.md) | Structural VAR |
| [`BayesianVAR`](svar.md) | Bayesian VAR with Minnesota prior |
| [`FAVAR`](svar.md) | Factor-Augmented VAR |
| [`TVPVAR`](svar.md) | Time-Varying Parameter VAR |
| [`GVAR`](svar.md) | Global VAR |

#### Other Models

| Class | Description |
|-------|-------------|
| [`ARDL`](ardl.md) | Autoregressive Distributed Lag |
| [`ECM`](ardl.md) | Error Correction Model |

### Analysis

| Class | Description |
|-------|-------------|
| [`SpilloverIndex`](spillover.md) | Diebold-Yilmaz spillover index |
| [`HistoricalDecomposition`](svar.md) | Historical shock decomposition |
| [`Counterfactual`](svar.md) | Counterfactual analysis |

### Diagnostics & Tests

| Module | Description |
|--------|-------------|
| [`diagnostics`](diagnostics.md) | Unit root, cointegration, specification tests |

### Supporting Modules

| Module | Description |
|--------|-------------|
| [`visualization`](visualization.md) | Plot functions and themes |
| [`reports`](reports.md) | Report generation (HTML, LaTeX, Markdown) |
| [`experiment`](experiment.md) | ChronoExperiment workflow |
| [`datasets`](datasets.md) | Built-in datasets |
| [`filters`](filters.md) | Economic filters (HP, BK, CF, Hamilton, BN) |
| [`cli`](cli.md) | Command-line interface |

## Quick Import Reference

```python
# Models
from chronobox import ARIMA, ARFIMA, ETS, HoltWinters, ThetaMethod
from chronobox import VAR, VECM, SVAR, BayesianVAR, FAVAR, TVPVAR, GVAR
from chronobox import STL, ClassicalDecomposition

# Automatic selection
from chronobox import auto_arima, auto_ets

# Analysis
from chronobox.analysis.irf import IRF
from chronobox.analysis.fevd import FEVD
from chronobox.analysis.granger import granger_causality
from chronobox.analysis.spillover import SpilloverIndex

# Core
from chronobox.core import TimeSeriesData, TimeSeriesResults, LagPolynomial

# Datasets
from chronobox.datasets import load_dataset, list_datasets
```
