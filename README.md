# chronobox

Time series analysis library with ARIMA, state-space models, and automatic model selection.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from chronobox import ARIMA
from chronobox.datasets import load_dataset

airline = load_dataset('airline')
model = ARIMA(order=(0,1,1), seasonal_order=(0,1,1,12))
results = model.fit(airline['passengers'])
print(results.summary())
forecast = results.forecast(steps=12)
```

## Auto-ARIMA

```python
from chronobox import auto_arima

best = auto_arima(airline['passengers'], seasonal=True, m=12)
print(best.summary())
```

## License

MIT
