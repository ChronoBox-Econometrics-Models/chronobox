# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-17

### Added
- ARIMA/SARIMA model with MLE via Kalman filter (kalmanbox)
- Auto-ARIMA with stepwise algorithm and information criteria
- VAR model with lag selection, Granger causality, IRF, FEVD
- SVAR model with Cholesky, short-run, and long-run identification
- VECM model with Johansen cointegration test
- ARDL model with bounds testing and ECM
- Unit root tests: ADF, Phillips-Perron, KPSS, ERS/DF-GLS
- Cointegration tests: Johansen trace and max eigenvalue
- Diagnostic tests: Ljung-Box, Breusch-Godfrey, ARCH-LM
- Filters: Hodrick-Prescott, Baxter-King, Christiano-Fitzgerald, Hamilton
- Time series decomposition (STL, classical)
- Visualization module with diagnostic plots
- Report generation (HTML, professional themes)
- ChronoExperiment pattern for systematic model comparison
- CLI with 5 commands: estimate, test, forecast, decompose, filter
- ~30 built-in datasets (classic, macro, finance, simulated)
- Download scripts for Brazilian macro data (BCB SGS, IBGE SIDRA)
- Numba optimization for critical loops (@optional_jit)
- MkDocs documentation with ~40 pages
- CI/CD with GitHub Actions
- 10 quality assurance phases

### Dependencies
- kalmanbox (Kalman filter and MLE)
- NumPy >= 1.24
- SciPy >= 1.10
- pandas >= 2.0
- matplotlib >= 3.7
