# ChronoBox

**Time series analysis library for Python**

ChronoBox provides a comprehensive toolkit for time series analysis,
including ARIMA/SARIMA, VAR, SVAR, VECM, ARDL, unit root tests,
cointegration tests, filters, and automatic model selection.

## Features

- **ARIMA/SARIMA**: Full Box-Jenkins methodology with MLE via Kalman filter
- **Auto-ARIMA**: Automatic model selection with stepwise algorithm
- **VAR/SVAR**: Vector Autoregression with Granger causality, IRF, FEVD
- **VECM**: Vector Error Correction Model with Johansen cointegration
- **ARDL**: Autoregressive Distributed Lag with bounds testing
- **Unit Root Tests**: ADF, PP, KPSS, ERS/DF-GLS
- **Filters**: HP, BK, CF, Hamilton
- **Experiment Pattern**: Systematic model comparison and validation
- **~30 Built-in Datasets**: Classic, macro, finance, simulated
- **CLI**: Command-line interface for quick analysis
- **Reports**: Professional HTML reports

## Quick Example

```python
from chronobox import ARIMA, auto_arima
from chronobox.datasets import load_dataset

# Load data
airline = load_dataset('airline')

# Fit ARIMA
model = ARIMA(order=(0,1,1), seasonal_order=(0,1,1,12))
results = model.fit(airline)
print(results.summary())

# Forecast
forecast = results.forecast(steps=12)
results.plot_diagnostics()

# Auto-ARIMA
best = auto_arima(airline, seasonal=True, m=12)
print(best.summary())
```

## Installation

```bash
pip install chronobox
```

## License

MIT License - see [LICENSE](https://github.com/nodesecon/chronobox/blob/main/LICENSE) for details.
