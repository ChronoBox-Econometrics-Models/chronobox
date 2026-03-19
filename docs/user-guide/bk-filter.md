# BK Filter (Baxter-King)

## Overview

The Baxter-King band-pass filter isolates cyclical components within a
specified frequency band, removing both high-frequency noise and
low-frequency trends.

## Usage

```python
from chronobox.filters import bk_filter
from chronobox.datasets import load_dataset

gdp = load_dataset('us_gdp')
cycle = bk_filter(gdp, low=6, high=32, K=12)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `y` | required | Time series data |
| `low` | `6` | Minimum period (quarters) |
| `high` | `32` | Maximum period (quarters) |
| `K` | `12` | Lead-lag truncation length |

## Standard Business Cycle Frequencies

For quarterly data, the standard NBER business cycle range is 6-32 quarters
(1.5 to 8 years).

## Mathematical Formulation

The ideal band-pass filter is approximated by a symmetric moving average:

$$
y_t^c = \sum_{j=-K}^{K} a_j y_{t-j}
$$

where the weights $a_j$ are chosen to approximate the ideal spectral window.

!!! note
    The BK filter loses $K$ observations at each end of the series.
    Choose $K$ based on available data length.

## Visualization

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(cycle)
plt.axhline(0, color='black', linewidth=0.5)
plt.title("Baxter-King Cyclical Component")
plt.show()
```
