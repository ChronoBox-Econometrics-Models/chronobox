---
title: Benchmarks
description: "Performance benchmarks comparing ChronoBox with statsmodels and R packages across ARIMA, VAR, and ETS models."
---

# Benchmarks

Systematic performance comparison of ChronoBox against established alternatives — **statsmodels** (Python) and **forecast/vars** (R).

!!! info "Benchmark environment"
    All benchmarks were run on the same machine with fixed random seeds for reproducibility.

    | Component | Specification |
    |-----------|--------------|
    | **CPU** | AMD Ryzen 9 5900X (12 cores, 24 threads) |
    | **RAM** | 64 GB DDR4-3600 |
    | **OS** | Ubuntu 22.04 LTS |
    | **Python** | 3.11.7 |
    | **R** | 4.3.2 |
    | **chronobox** | 0.3.0 |
    | **statsmodels** | 0.14.1 |
    | **R forecast** | 8.21.1 |
    | **R vars** | 1.6-1 |

---

## Methodology

### Datasets

All benchmarks use publicly available datasets to ensure reproducibility:

| Dataset | Type | $T$ | $k$ | Frequency | Source |
|---------|------|-----|-----|-----------|--------|
| **Airline** | Univariate | 144 | 1 | Monthly | Box & Jenkins (1976) |
| **Nile** | Univariate | 100 | 1 | Annual | Cobb (1978) |
| **M3 subset** | Univariate | varies | 1 | Mixed | M3 Competition (Makridakis & Hibon, 2000) |
| **M4 subset** | Univariate | varies | 1 | Mixed | M4 Competition (Makridakis et al., 2020) |
| **Canada macro** | Multivariate | 84 | 4 | Quarterly | Lütkepohl (2005) |

### Metrics

| Metric | Description | Used for |
|--------|-------------|----------|
| **Estimation time** | Wall-clock time (median of 10 runs) | Speed comparison |
| **AIC** | Akaike Information Criterion | In-sample fit |
| **BIC** | Bayesian Information Criterion | Model complexity penalty |
| **RMSE** | Root Mean Squared Error (out-of-sample) | Forecast accuracy |
| **MAE** | Mean Absolute Error (out-of-sample) | Forecast accuracy |

!!! note "Out-of-sample evaluation"
    Forecast accuracy uses a rolling origin evaluation: the last 20% of each series is held out for testing, and forecasts are generated at each origin with a fixed horizon ($h = 12$ for monthly, $h = 4$ for quarterly, $h = 1$ for annual).

### Statistical significance

- Timing comparisons report median and interquartile range over 10 repetitions
- Forecast accuracy differences are tested with the Diebold-Mariano test ($\alpha = 0.05$)
- AIC/BIC are compared directly (no test needed — same data, same likelihood)

---

## Summary Results

### Estimation Speed

| Model | ChronoBox | statsmodels | R | Fastest |
|-------|-----------|-------------|---|---------|
| ARIMA(1,1,1) — Airline | 15 ms | 42 ms | 28 ms | **ChronoBox** |
| SARIMA(1,1,1)(1,1,1,12) | 35 ms | 95 ms | 55 ms | **ChronoBox** |
| auto_arima — M3 (avg) | 0.8 s | 2.1 s | 1.2 s | **ChronoBox** |
| VAR(2) — Canada | 8 ms | 12 ms | 5 ms | **R** |
| ETS(A,A,A) — Airline | 22 ms | 68 ms | 18 ms | **R** |
| ETS auto — M3 (avg) | 0.5 s | 1.8 s | 0.4 s | **R** |

### Forecast Accuracy (RMSE, out-of-sample)

| Model / Dataset | ChronoBox | statsmodels | R | Best |
|-----------------|-----------|-------------|---|------|
| ARIMA — Airline | 14.2 | 14.2 | 14.1 | R (ns) |
| ARIMA — Nile | 108.5 | 109.1 | 108.3 | R (ns) |
| auto_arima — M3 avg | 1.42 | 1.51 | 1.38 | **R** |
| VAR(2) — Canada | 2.31 | 2.35 | 2.29 | R (ns) |
| ETS — Airline | 15.1 | 16.8 | 14.9 | **R** |
| auto_ets — M3 avg | 1.48 | 1.62 | 1.45 | R (ns) |

!!! tip "Interpreting the results"
    - **(ns)** means the difference is **not statistically significant** at $\alpha = 0.05$ (Diebold-Mariano test)
    - ChronoBox achieves comparable accuracy to R forecast in most cases
    - ChronoBox is consistently faster than statsmodels for ARIMA estimation
    - R's `forecast::ets` has a slight edge on ETS models due to its mature C++ backend

---

## Detailed Benchmarks

Drill into model-specific comparisons:

<div class="grid cards" markdown>

-   :material-chart-line: **ARIMA Benchmark**

    ---

    ChronoBox vs statsmodels vs R forecast on ARIMA/SARIMA models

    [:octicons-arrow-right-24: ARIMA details](arima.md)

-   :material-chart-multiple: **VAR Benchmark**

    ---

    Multivariate models: estimation, IRF, and forecast comparison

    [:octicons-arrow-right-24: VAR details](var.md)

-   :material-trending-up: **ETS Benchmark**

    ---

    Exponential smoothing: auto-selection and forecasting performance

    [:octicons-arrow-right-24: ETS details](ets.md)

-   :material-table: **Feature Comparison**

    ---

    Full feature matrix across ChronoBox, statsmodels, and R packages

    [:octicons-arrow-right-24: Feature comparison](comparison.md)

</div>
