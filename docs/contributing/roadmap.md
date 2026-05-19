---
title: "Roadmap"
description: "ChronoBox development roadmap — planned features, priorities, and release schedule."
---

# Roadmap

ChronoBox aims to be the most comprehensive Python library for time series econometrics. This roadmap outlines planned features, organized by release version.

!!! info "Current Status"
    ChronoBox v0.1.0 is the current release with ARIMA, ETS, VAR/SVAR/VECM, ARDL, filters, spillover analysis, statistical tests, and visualization.

---

## v0.2.0 — Volatility & Regime Switching

Planned features for the next minor release.

### GARCH Models

- **GARCH(p,q)** — Generalized Autoregressive Conditional Heteroskedasticity (Bollerslev 1986)
- **EGARCH** — Exponential GARCH for asymmetric volatility (Nelson 1991)
- **GJR-GARCH** — Threshold GARCH capturing leverage effects (Glosten, Jagannathan & Runkle 1993)
- **TGARCH** — Threshold GARCH with indicator-based asymmetry
- **IGARCH** — Integrated GARCH for persistent volatility
- **FIGARCH** — Fractionally Integrated GARCH for long memory in volatility
- **DCC-GARCH** — Dynamic Conditional Correlation for multivariate volatility (Engle 2002)
- **BEKK-GARCH** — Full multivariate GARCH with cross-volatility spillovers

### Regime Switching

- **Markov Switching AR** — Hamilton (1989) regime-switching autoregressive model
- **Markov Switching VAR** — MS-VAR for multivariate regime detection
- **Threshold AR (TAR)** — Self-exciting threshold autoregression (Tong 1990)
- **Smooth Transition AR (STAR)** — Gradual regime changes (Terasvirta 1994)

---

## v0.3.0 — Machine Learning Integration

Hybrid approaches combining econometric rigor with ML flexibility.

### Deep Learning Models

- **LSTM** — Long Short-Term Memory networks for sequence forecasting
- **Temporal Fusion Transformer** — Attention-based multi-horizon forecasting (Lim et al. 2021)
- **N-BEATS** — Neural Basis Expansion for interpretable forecasting (Oreshkin et al. 2020)
- **DeepAR** — Autoregressive RNN with probabilistic output (Salinas et al. 2020)

### Statistical ML Models

- **Prophet** — Facebook/Meta's decomposable time series model
- **TBATS** — Trigonometric Box-Cox ARMA Trend Seasonal
- **Elastic Net AR** — Regularized autoregression for high-dimensional time series
- **Random Forest forecasting** — Tree-based ensemble methods with lagged features

### Model Combination

- **Forecast combination** — Optimal forecast pooling (Bates & Granger 1969)
- **Stacking** — Meta-learner combining econometric and ML forecasts
- **Model confidence sets** — Hansen, Lunde & Nason (2011) MCS procedure

---

## v0.4.0 — Panel Time Series

Cross-sectional time series methods, bridging ChronoBox and panelbox.

### Panel Unit Root Tests

- **LLC** — Levin, Lin & Chu (2002)
- **IPS** — Im, Pesaran & Shin (2003)
- **Fisher** — Maddala & Wu (1999) combined p-value test
- **CIPS** — Pesaran (2007) cross-sectionally augmented IPS

### Panel Cointegration

- **Pedroni** — Residual-based panel cointegration tests
- **Westerlund** — ECM-based panel cointegration with bootstrap
- **Panel FMOLS/DOLS** — Fully modified and dynamic OLS for cointegrated panels

### Panel VAR

- **Panel VAR** — Fixed effects VAR for heterogeneous panels
- **Panel IRF/FEVD** — Impulse responses and variance decomposition for panel data
- **Panel Granger** — Dumitrescu-Hurlin (2012) heterogeneous Granger causality

---

## v1.0.0 — Stable API

Production-ready release with guaranteed API stability.

### Stability Guarantees

- **Frozen public API** — No breaking changes within major version
- **Full test coverage** — 95%+ line coverage, validation against R and Stata
- **Performance benchmarks** — Automated regression testing for speed
- **Comprehensive documentation** — Complete API reference, tutorials, and theory

### Quality Metrics

| Metric | Target |
|---|---|
| Test coverage | > 95% |
| Docstring coverage | 100% public API |
| R validation | All models within 1e-6 tolerance |
| Stata validation | All models within 1e-4 tolerance |
| Build time | < 5 minutes CI |

---

## Ecosystem Integration

### kalmanbox Integration

ChronoBox already uses [kalmanbox](https://github.com/NodesEcon/kalmanbox) as the state-space backend for:

- TVP-VAR estimation via Kalman filter/smoother
- ETS state-space representation
- Unobserved components models

**Planned deepening**:

- Unified state-space interface for custom models
- Particle filter support for nonlinear/non-Gaussian models
- EM algorithm for maximum likelihood in state-space models

### panelbox Integration

[panelbox](https://github.com/NodesEcon/panelbox) handles cross-sectional panel data econometrics. Planned integration:

- Shared data containers for panel time series
- Cross-package diagnostic tests
- Unified experiment/report workflow
- Panel VAR bridging both libraries

### Future Ecosystem

```
┌─────────────────────────────────────────────┐
│              NodesEcon Ecosystem             │
├───────────┬───────────┬─────────────────────┤
│ chronobox │ panelbox  │     kalmanbox       │
│ (time     │ (panel    │  (state-space       │
│  series)  │  data)    │   engine)           │
├───────────┴───────────┴─────────────────────┤
│         Shared: datasets, reports,          │
│         visualization, experiment           │
└─────────────────────────────────────────────┘
```

---

## Long-Term Vision

Exploratory features for future major releases.

### GPU Acceleration

- **CuPy / JAX backends** — GPU-accelerated estimation for large-scale VAR and GARCH
- **Automatic backend selection** — Transparent CPU/GPU switching
- **Batch forecasting** — Parallel model fitting across thousands of series

### Real-Time Econometrics

- **Streaming data** — Online model updating with new observations
- **Nowcasting** — Mixed-frequency data combination (MIDAS, bridge equations)
- **Real-time vintage data** — Support for data revisions and real-time datasets

### Interactive Dashboard

- **Streamlit app** — Point-and-click time series analysis
- **Model builder** — Visual specification of ARIMA/VAR/ETS models
- **Real-time diagnostics** — Interactive test interpretation
- **Forecast monitor** — Track forecast accuracy over time

---

## How to Influence the Roadmap

### Feature Requests

Open a [GitHub Issue](https://github.com/NodesEcon/chronobox/issues) with the `[Feature]` label. Include:

1. **Use case**: What problem does it solve?
2. **Description**: What should the feature do?
3. **References**: Academic papers, existing implementations (R, Stata)
4. **Priority justification**: Why is this important for time series analysis?

### Community Voting

React with a :thumbsup: on existing feature request issues to signal demand. Features with more community interest are prioritized higher.

### Contributions

The fastest way to get a feature is to implement it yourself! See the [Contributing Guide](contributing.md) for templates and process.

---

## Release Schedule

### Versioning

ChronoBox follows [Semantic Versioning](https://semver.org/):

| Version | Cadence | Content |
|---|---|---|
| **Major** (X.0.0) | As needed | Breaking API changes |
| **Minor** (0.X.0) | Every 2–3 months | New features, backward compatible |
| **Patch** (0.0.X) | As needed | Bug fixes, documentation updates |

### Release Process

1. Feature freeze 1 week before release
2. Release candidate published for testing
3. Final release after validation against R/Stata
4. Changelog and migration notes published

---

## See Also

- [Contributing Guide](contributing.md) — How to contribute code and documentation
- [Changelog](changelog.md) — Version history
- [Code of Conduct](code-of-conduct.md) — Community standards
- [API Reference](../api/index.md) — Full API documentation
