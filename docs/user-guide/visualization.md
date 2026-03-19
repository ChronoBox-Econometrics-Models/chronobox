# Visualization

## Overview

ChronoBox provides publication-quality plotting functions for time series
analysis with customizable themes.

## Time Series Plot

```python
from chronobox.visualization import plot_series

plot_series(data, title="Time Series")
```

## Diagnostic Plots

Four-panel residual diagnostics (residuals, histogram, ACF, QQ-plot):

```python
from chronobox.visualization import plot_diagnostics

plot_diagnostics(results.residuals)
```

## Forecast Plot

Fan chart with confidence intervals:

```python
from chronobox.visualization import plot_forecast

plot_forecast(results, steps=24)
```

## IRF Plot

Impulse response functions grid:

```python
from chronobox.visualization import plot_irf

irf = results.irf(periods=20)
plot_irf(irf)
```

## FEVD Plot

Forecast error variance decomposition:

```python
from chronobox.visualization import plot_fevd

fevd = results.fevd(periods=20)
plot_fevd(fevd)
```

## Historical Decomposition Plot

```python
from chronobox.visualization import plot_hd

plot_hd(hd_result)
```

## Spillover Network

```python
from chronobox.visualization import plot_network, plot_heatmap

plot_network(spillover_result)
plot_heatmap(spillover_result)
```

## Themes

```python
from chronobox.visualization import set_theme, list_themes

# Available themes
print(list_themes())  # ['professional', 'academic', 'presentation', 'bcb']

# Set theme
set_theme('academic')
```

| Theme | Description |
|-------|-------------|
| `professional` | Clean, minimal style |
| `academic` | Publication-ready |
| `presentation` | Large fonts, high contrast |
| `bcb` | Banco Central do Brasil style |
