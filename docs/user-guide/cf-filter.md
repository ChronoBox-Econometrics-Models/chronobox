# CF Filter (Christiano-Fitzgerald)

## Overview

The Christiano-Fitzgerald filter is an asymmetric band-pass filter that
uses the full sample for estimation. Unlike the BK filter, it does not
lose observations at the endpoints.

## Usage

```python
from chronobox.filters import cf_filter
from chronobox.datasets import load_dataset

gdp = load_dataset('us_gdp')
cycle, trend = cf_filter(gdp, low=6, high=32)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `y` | required | Time series data |
| `low` | `6` | Minimum period |
| `high` | `32` | Maximum period |

## Advantages over BK Filter

- **No data loss**: Uses full sample
- **Asymmetric weights**: Better endpoint behavior
- **Optimal**: Under random walk assumption

## Mathematical Formulation

The CF filter uses time-varying weights:

$$
y_t^c = B_0 y_t + B_1 y_{t+1} + \cdots + B_{T-1-t} y_{T-1} + \tilde{B}_{T-t} y_T
+ B_1 y_{t-1} + \cdots + B_{t-2} y_2 + \tilde{B}_{t-1} y_1
$$

where weights are derived from the spectral representation.

## Comparison

```python
from chronobox.filters import hp_filter, bk_filter, cf_filter

trend_hp, cycle_hp = hp_filter(gdp, lamb=1600)
cycle_bk = bk_filter(gdp, low=6, high=32, K=12)
cycle_cf, trend_cf = cf_filter(gdp, low=6, high=32)
```
