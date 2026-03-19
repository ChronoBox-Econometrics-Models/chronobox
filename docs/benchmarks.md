# Benchmarks

## Overview

Performance benchmarks for ChronoBox operations on standard datasets.
All benchmarks run on a typical development machine (8-core, 16GB RAM).

## ARIMA Estimation

| Dataset | T | Order | Time (ms) | Time with Numba (ms) |
|---------|---|-------|-----------|---------------------|
| airline (144) | 144 | (0,1,1)(0,1,1,12) | ~50 | ~15 |
| co2 (468) | 468 | (1,1,1) | ~30 | ~10 |
| Simulated (1000) | 1000 | (2,1,2) | ~80 | ~25 |
| Simulated (5000) | 5000 | (2,1,2) | ~300 | ~90 |
| Simulated (10000) | 10000 | (2,1,2) | ~600 | ~180 |

## Auto-ARIMA

| Dataset | T | Stepwise Time (s) | Grid Search Time (s) |
|---------|---|-------------------|---------------------|
| airline | 144 | ~1.5 | ~15 |
| co2 | 468 | ~3.0 | ~30 |
| Simulated (1000) | 1000 | ~5.0 | ~50 |

## VAR Estimation

| K (variables) | T | Lags | Time (ms) |
|---------------|---|------|-----------|
| 3 | 200 | 2 | ~5 |
| 3 | 200 | 8 | ~10 |
| 5 | 500 | 4 | ~20 |
| 10 | 1000 | 4 | ~100 |

## Statistical Tests

| Test | T=200 (ms) | T=1000 (ms) | T=5000 (ms) |
|------|-----------|-------------|-------------|
| ADF | ~2 | ~5 | ~15 |
| PP | ~3 | ~8 | ~20 |
| KPSS | ~2 | ~5 | ~12 |
| Ljung-Box | ~1 | ~2 | ~5 |
| Jarque-Bera | <1 | <1 | ~1 |

## Filters

| Filter | T=200 (ms) | T=1000 (ms) | T=5000 (ms) |
|--------|-----------|-------------|-------------|
| HP | ~1 | ~2 | ~10 |
| BK | ~2 | ~5 | ~15 |
| CF | ~3 | ~10 | ~50 |
| Hamilton | ~5 | ~10 | ~30 |

## Scalability

### ARIMA: Linear in T

The Kalman filter runs in $O(T \cdot r^3)$ where $r$ is the state dimension.
For a fixed ARIMA order, estimation time scales linearly with sample size.

### VAR: Cubic in K

VAR estimation involves $O(K^3)$ matrix operations. Doubling the number
of variables increases estimation time by ~8x.

### Memory Usage

| Operation | T=1000, K=3 | T=1000, K=10 | T=10000, K=3 |
|-----------|------------|-------------|-------------|
| VAR fit | ~1 MB | ~10 MB | ~10 MB |
| IRF (40 periods) | ~0.5 MB | ~5 MB | ~0.5 MB |
| FEVD | ~0.5 MB | ~5 MB | ~0.5 MB |

## Numba Acceleration

When numba is installed, performance-critical loops are JIT-compiled:

| Operation | Without Numba | With Numba | Speedup |
|-----------|:---:|:---:|:---:|
| Kalman filter | 1x | ~5x | 5x |
| ARIMA forecast | 1x | ~3x | 3x |
| BK filter | 1x | ~2x | 2x |
| Lag matrix construction | 1x | ~4x | 4x |

## Comparison with statsmodels

| Operation | ChronoBox | statsmodels | Notes |
|-----------|:---------:|:-----------:|-------|
| ARIMA(1,1,1) fit | ~30ms | ~50ms | Exact MLE via Kalman |
| Auto-ARIMA | ~2s | ~5s | Stepwise algorithm |
| VAR(2) fit (K=3) | ~5ms | ~8ms | OLS estimation |
| ADF test | ~2ms | ~3ms | MacKinnon p-values |

!!! note
    Benchmarks are approximate and depend on hardware, Python version,
    and data characteristics. Run your own benchmarks for precise comparisons.
