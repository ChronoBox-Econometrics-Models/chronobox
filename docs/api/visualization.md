---
title: "Visualization API"
description: "API reference for plotting functions — time series, diagnostics, IRF, FEVD, forecasts, spillover, and themes"
---

# Visualization API Reference

!!! info "Module"
    **Import**: `from chronobox.visualization import plot_series, plot_diagnostics, set_theme, ...`
    **Source**: `chronobox/visualization/`

## Overview

| Function | Category | Description |
|----------|----------|-------------|
| `plot_series` | Time Series | Multiple time series with annotations and recessions |
| `plot_diagnostics` | Diagnostics | 2×2 residual diagnostics panel (residuals, histogram, ACF, Q-Q) |
| `plot_forecast` | Forecast | Fan chart with confidence intervals |
| `plot_decomposition` | Decomposition | Vertical panels: trend, seasonal, cycle, remainder |
| `plot_irf` | VAR | Impulse response functions ($K \times K$ grid) |
| `plot_fevd` | VAR | Forecast error variance decomposition |
| `plot_hd` | VAR | Historical decomposition |
| `plot_network` | Spillover | Network graph of spillover connectedness |
| `plot_heatmap` | Spillover | Heatmap of spillover table |
| `plot_rolling` | Spillover | Time-varying rolling spillover |
| `plot_tvp_coefs` | TVP-VAR | Time-varying parameter coefficients |
| `plot_cusum` | Tests | CUSUM test with confidence bands |
| `plot_bai_perron` | Tests | Bai-Perron structural breaks |
| `plot_zivot_andrews` | Tests | Zivot-Andrews break visualization |
| `plot_recursive_coefs` | Tests | Recursive coefficient stability |
| `set_theme` | Themes | Set global plot theme |
| `get_theme` | Themes | Get current theme configuration |
| `list_themes` | Themes | List available themes |
| `save_figure` | Export | Save figure to PNG, SVG, PDF, HTML |
| `figure_to_html` | Export | Convert figure to HTML string |
| `figure_to_base64` | Export | Convert figure to base64-encoded string |

---

## Time Series Plots

### plot_series

Plot one or more time series with optional secondary y-axis, annotations, and
recession shading.

```python
plot_series(
    data: pd.Series | pd.DataFrame | ndarray | list,
    labels: list[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    secondary_y: list[str] | None = None,
    secondary_ylabel: str | None = None,
    annotations: list[dict] | None = None,
    recessions: list[tuple] | None = None,
    recession_color: str = "#cccccc",
    recession_alpha: float = 0.3,
    figsize: tuple[float, float] | None = None,
    legend: bool = True,
    grid: bool = True,
    ax: Axes | None = None,
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `Series \| DataFrame \| ndarray` | *required* | Time series data |
| `labels` | `list[str] \| None` | `None` | Series labels for legend |
| `title` | `str \| None` | `None` | Plot title |
| `secondary_y` | `list[str] \| None` | `None` | Column names to plot on secondary y-axis |
| `annotations` | `list[dict] \| None` | `None` | Annotation dicts with `x`, `y`, `text` keys |
| `recessions` | `list[tuple] \| None` | `None` | List of `(start, end)` for recession shading |
| `figsize` | `tuple \| None` | `None` | Figure size `(width, height)` |
| `ax` | `Axes \| None` | `None` | Existing axes to plot on |

**Returns**: `matplotlib.figure.Figure`

### Example

```python
import pandas as pd
from chronobox.datasets import load_dataset
from chronobox.visualization import plot_series

gdp = load_dataset("us_gdp")
fig = plot_series(
    gdp,
    title="US GDP",
    ylabel="Billions USD",
    recessions=[(pd.Timestamp("2007-12"), pd.Timestamp("2009-06"))],
)
```

::: chronobox.visualization.ts_plot.plot_series
    options:
      show_root_heading: false
      show_source: true

---

### plot_diagnostics

2×2 residual diagnostics panel: time series of residuals, histogram with
KDE overlay, ACF, and Q-Q plot.

```python
plot_diagnostics(
    results: Any | None = None,
    residuals: ndarray | None = None,
    lags: int = 20,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `Any \| None` | `None` | Model results object (extracts `.resid` automatically) |
| `residuals` | `ndarray \| None` | `None` | Raw residuals (alternative to `results`) |
| `lags` | `int` | `20` | Number of lags for ACF |
| `figsize` | `tuple \| None` | `None` | Figure size |
| `title` | `str \| None` | `None` | Plot title |

**Returns**: `matplotlib.figure.Figure`

### Example

```python
from chronobox import ARIMA
from chronobox.visualization import plot_diagnostics
import numpy as np

y = np.cumsum(np.random.default_rng(42).normal(size=200))
results = ARIMA(order=(1, 1, 1)).fit(y)

fig = plot_diagnostics(results, lags=30, title="ARIMA(1,1,1) Residuals")
```

::: chronobox.visualization.diagnostics_plot.plot_diagnostics
    options:
      show_root_heading: false
      show_source: true

---

### plot_forecast

Fan chart showing point forecast with confidence interval bands.

```python
plot_forecast(
    results: Any | None = None,
    steps: int = 12,
    alpha: float = 0.05,
    history: ndarray | None = None,
    forecast_mean: ndarray | None = None,
    forecast_se: ndarray | None = None,
    ci_levels: list[float] | None = None,
    history_periods: int = 50,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `Any \| None` | `None` | Model results (auto-generates forecast) |
| `steps` | `int` | `12` | Forecast horizon |
| `alpha` | `float` | `0.05` | Significance level for CI |
| `history` | `ndarray \| None` | `None` | Historical data to show before forecast |
| `forecast_mean` | `ndarray \| None` | `None` | Pre-computed forecast mean |
| `forecast_se` | `ndarray \| None` | `None` | Pre-computed forecast standard errors |
| `ci_levels` | `list[float] \| None` | `None` | Multiple CI levels for fan chart (e.g., `[0.50, 0.80, 0.95]`) |
| `history_periods` | `int` | `50` | Number of historical periods to display |
| `figsize` | `tuple \| None` | `None` | Figure size |

**Returns**: `matplotlib.figure.Figure`

::: chronobox.visualization.forecast_plot.plot_forecast
    options:
      show_root_heading: false
      show_source: true

---

### plot_decomposition

Vertical panels for decomposition components: observed, trend, seasonal,
and remainder.

```python
plot_decomposition(
    result: Any,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `Any` | *required* | Decomposition result (from STL, Classical, or X-13) |
| `figsize` | `tuple \| None` | `None` | Figure size |
| `title` | `str \| None` | `None` | Plot title |

**Returns**: `matplotlib.figure.Figure`

::: chronobox.visualization.decomposition_plot.plot_decomposition
    options:
      show_root_heading: false
      show_source: true

---

## VAR / Multivariate Plots

### plot_irf

Impulse response function plot. Supports individual shock-response pairs
or full $K \times K$ grid.

```python
plot_irf(
    irf_results: Any | None = None,
    irf_array: ndarray | None = None,
    irf_lower: ndarray | None = None,
    irf_upper: ndarray | None = None,
    var_names: list[str] | None = None,
    impulse: str | int | None = None,
    response: str | int | None = None,
    cumulative: bool = False,
    sigs: float = 0.95,
    periods: int | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `irf_results` | `Any \| None` | `None` | IRF result object from VAR/SVAR |
| `irf_array` | `ndarray \| None` | `None` | Pre-computed IRF array `(periods, K, K)` |
| `irf_lower` | `ndarray \| None` | `None` | Lower confidence bound |
| `irf_upper` | `ndarray \| None` | `None` | Upper confidence bound |
| `var_names` | `list[str] \| None` | `None` | Variable names |
| `impulse` | `str \| int \| None` | `None` | Specific impulse variable (None = all) |
| `response` | `str \| int \| None` | `None` | Specific response variable (None = all) |
| `cumulative` | `bool` | `False` | Plot cumulative IRFs |
| `sigs` | `float` | `0.95` | Confidence level |
| `periods` | `int \| None` | `None` | Number of periods to show |

**Returns**: `matplotlib.figure.Figure`

::: chronobox.visualization.irf_plot.plot_irf
    options:
      show_root_heading: false
      show_source: true

---

### plot_fevd

Forecast error variance decomposition. Stacked area or bar chart showing
the proportion of forecast variance attributable to each shock.

```python
plot_fevd(
    fevd_results: Any,
    var_names: list[str] | None = None,
    periods: int | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fevd_results` | `Any` | *required* | FEVD result object |
| `var_names` | `list[str] \| None` | `None` | Variable names |
| `periods` | `int \| None` | `None` | Number of periods |
| `figsize` | `tuple \| None` | `None` | Figure size |

**Returns**: `matplotlib.figure.Figure`

::: chronobox.visualization.fevd_plot.plot_fevd
    options:
      show_root_heading: false
      show_source: true

---

### plot_hd

Historical decomposition plot. Shows the contribution of each structural
shock to a variable over time.

```python
plot_hd(
    hd_results: Any,
    variable: str | int | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hd_results` | `Any` | *required* | Historical decomposition result |
| `variable` | `str \| int \| None` | `None` | Variable to plot |
| `figsize` | `tuple \| None` | `None` | Figure size |

**Returns**: `matplotlib.figure.Figure`

::: chronobox.visualization.hd_plot.plot_hd
    options:
      show_root_heading: false
      show_source: true

---

## Spillover Plots

### plot_network

Network graph visualization of spillover connectedness. Edge widths reflect
pairwise net spillover magnitudes.

```python
plot_network(
    spillover_result: Any,
    var_names: list[str] | None = None,
    threshold: float = 0.0,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spillover_result` | `Any` | *required* | SpilloverResult object |
| `var_names` | `list[str] \| None` | `None` | Variable names |
| `threshold` | `float` | `0.0` | Minimum spillover to show edge |
| `figsize` | `tuple \| None` | `None` | Figure size |

**Returns**: `matplotlib.figure.Figure`

::: chronobox.visualization.spillover_plot.plot_network
    options:
      show_root_heading: false
      show_source: true

---

### plot_heatmap

Heatmap of the spillover table with FROM/TO aggregates.

```python
plot_heatmap(
    spillover_result: Any,
    var_names: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> Figure
```

**Returns**: `matplotlib.figure.Figure`

::: chronobox.visualization.spillover_plot.plot_heatmap
    options:
      show_root_heading: false
      show_source: true

---

### plot_rolling

Time-varying rolling spillover visualization.

```python
plot_rolling(
    rolling_result: Any,
    var_names: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> Figure
```

**Returns**: `matplotlib.figure.Figure`

::: chronobox.visualization.spillover_plot.plot_rolling
    options:
      show_root_heading: false
      show_source: true

---

## Test Visualization

### plot_cusum

CUSUM test visualization with 5% significance bands.

```python
plot_cusum(
    result: Any,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `Any` | *required* | CUSUM test result |
| `figsize` | `tuple \| None` | `None` | Figure size |
| `title` | `str \| None` | `None` | Plot title |

**Returns**: `matplotlib.figure.Figure`

::: chronobox.visualization.test_plot.plot_cusum
    options:
      show_root_heading: false
      show_source: true

---

### plot_bai_perron

Bai-Perron structural break visualization with estimated break dates.

```python
plot_bai_perron(
    result: Any,
    y: ndarray | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> Figure
```

::: chronobox.visualization.test_plot.plot_bai_perron
    options:
      show_root_heading: false
      show_source: true

---

### plot_zivot_andrews

Zivot-Andrews unit root test with structural break visualization.

```python
plot_zivot_andrews(
    result: Any,
    y: ndarray | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> Figure
```

::: chronobox.visualization.test_plot.plot_zivot_andrews
    options:
      show_root_heading: false
      show_source: true

---

### plot_recursive_coefs

Recursive coefficient estimates over time for stability analysis.

```python
plot_recursive_coefs(
    result: Any,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> Figure
```

::: chronobox.visualization.test_plot.plot_recursive_coefs
    options:
      show_root_heading: false
      show_source: true

---

### plot_tvp_coefs

Time-varying parameter coefficient plots for TVP-VAR models.

```python
plot_tvp_coefs(
    result: Any,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> Figure
```

::: chronobox.visualization.coef_plot.plot_tvp_coefs
    options:
      show_root_heading: false
      show_source: true

---

## Themes

### set_theme

Set the global plot theme for all subsequent chronobox plots.

```python
set_theme(name: str) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Theme name: `'professional'`, `'academic'`, `'presentation'`, `'bcb'` |

Available themes:

| Theme | Description | Best For |
|-------|-------------|----------|
| `professional` | Blue/gray, Helvetica, clean lines | Reports, publications |
| `academic` | B&W, Computer Modern | Academic papers, journals |
| `presentation` | Vibrant colors, large fonts | Slides, talks |
| `bcb` | Banco Central do Brasil institutional style | BCB reports |

### Example

```python
from chronobox.visualization import set_theme, list_themes, plot_series

# List available themes
print(list_themes())  # ['professional', 'academic', 'presentation', 'bcb']

# Set theme globally
set_theme("academic")

# All subsequent plots use the academic theme
fig = plot_series(data, title="GDP Growth")
```

::: chronobox.visualization.themes.set_theme
    options:
      show_root_heading: false
      show_source: true

::: chronobox.visualization.themes.get_theme
    options:
      show_root_heading: false
      show_source: true

::: chronobox.visualization.themes.list_themes
    options:
      show_root_heading: false
      show_source: true

---

## Export

### save_figure

Save a matplotlib or plotly figure to file.

```python
save_figure(
    fig: Figure | Any,
    path: str | Path,
    fmt: str | None = None,
    dpi: int | None = None,
    transparent: bool = False,
    bbox_inches: str = "tight",
    **kwargs,
) -> Path
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fig` | `Figure` | *required* | Matplotlib or plotly figure |
| `path` | `str \| Path` | *required* | Output file path |
| `fmt` | `str \| None` | `None` | Format: `'png'`, `'svg'`, `'pdf'`, `'html'`. Inferred from extension if None |
| `dpi` | `int \| None` | `None` | Resolution in DPI |
| `transparent` | `bool` | `False` | Transparent background |

**Returns**: `Path` — path to the saved file

### figure_to_html

Convert a figure to an HTML string (static PNG for matplotlib, interactive for plotly).

```python
figure_to_html(
    fig: Figure | Any,
    include_plotlyjs: bool = True,
    full_html: bool = True,
    div_id: str | None = None,
) -> str
```

### figure_to_base64

Convert a matplotlib figure to a base64-encoded string.

```python
figure_to_base64(
    fig: Figure,
    fmt: str = "png",
    dpi: int = 150,
) -> str
```

### figures_to_html_gallery

Generate an HTML gallery page from multiple figures.

```python
figures_to_html_gallery(
    figures: list[tuple[str, Figure]],
    title: str = "chronobox Figures",
) -> str
```

### Example

```python
from chronobox.visualization import plot_series, set_theme
from chronobox.visualization.export import save_figure, figure_to_html

set_theme("professional")
fig = plot_series(data, title="GDP Growth")

# Save to multiple formats
save_figure(fig, "gdp.png", dpi=300)
save_figure(fig, "gdp.svg")
save_figure(fig, "gdp.pdf")

# Embed in HTML
html = figure_to_html(fig, full_html=False)
```

::: chronobox.visualization.export.save_figure
    options:
      show_root_heading: false
      show_source: true

::: chronobox.visualization.export.figure_to_html
    options:
      show_root_heading: false
      show_source: true

::: chronobox.visualization.export.figure_to_base64
    options:
      show_root_heading: false
      show_source: true

::: chronobox.visualization.export.figures_to_html_gallery
    options:
      show_root_heading: false
      show_source: true

---

## See Also

- [Diagnostics API](diagnostics.md) -- Statistical tests
- [Spillover API](spillover.md) -- Spillover analysis and connectedness
- [VAR API](var.md) -- VAR, IRF, FEVD
- [Visualization Guide](../visualization/index.md) -- Step-by-step plotting guide
