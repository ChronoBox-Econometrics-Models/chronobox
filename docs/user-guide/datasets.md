# Datasets

## Overview

ChronoBox includes ~30 built-in datasets for examples, testing,
and benchmarking.

## Loading Datasets

```python
from chronobox.datasets import load_dataset, list_datasets

# List all datasets
print(list_datasets())

# List by category
print(list_datasets(category='classic'))

# Load a dataset
airline = load_dataset('airline')
```

## Available Datasets

### Classic

| Name | Description | Frequency |
|------|-------------|-----------|
| `airline` | Airline passengers (1949-1960) | Monthly |
| `nile` | Nile river flow (1871-1970) | Annual |
| `sunspot` | Sunspot numbers | Annual |
| `lynx` | Canadian lynx trappings | Annual |
| `co2` | Mauna Loa CO2 concentration | Monthly |
| `uspop` | US population | Decennial |

### Macroeconomic

| Name | Description | Frequency |
|------|-------------|-----------|
| `us_gdp` | US GDP | Quarterly |
| `us_macro_quarterly` | US macro variables | Quarterly |
| `us_consumption` | US consumption | Quarterly |
| `canada` | Canadian macro data | Quarterly |
| `uk_drivers` | UK road casualties | Monthly |
| `uk_gas` | UK gas consumption | Quarterly |
| `denmark` | Danish macro data | Quarterly |

### Brazil

| Name | Description | Frequency |
|------|-------------|-----------|
| `ipca` | Brazilian inflation (IPCA) | Monthly |
| `pib_brasil` | Brazilian GDP | Quarterly |
| `selic` | Selic interest rate | Monthly |
| `cambio` | BRL/USD exchange rate | Monthly |
| `desemprego` | Unemployment rate | Monthly |
| `producao_industrial` | Industrial production | Monthly |
| `macro_brasil` | Brazilian macro variables | Monthly |

### Finance

| Name | Description | Frequency |
|------|-------------|-----------|
| `sp500_returns` | S&P 500 returns | Daily |
| `ibovespa_returns` | Ibovespa returns | Daily |
| `exchange_rates` | Exchange rates | Daily |
| `vix` | VIX volatility index | Daily |

### Simulated

| Name | Description |
|------|-------------|
| `arma11` | ARMA(1,1) process |
| `var2` | VAR(2) process |
| `cointegrated` | Cointegrated system |
| `structural_break` | Series with structural break |
| `long_memory` | Long memory process |
