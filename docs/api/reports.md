---
title: "Reports API"
description: "API reference for automated report generation — HTML, LaTeX, and Markdown from model results"
---

# Reports API Reference

!!! info "Module"
    **Import**: `from chronobox.reports import ReportManager`
    **Source**: `chronobox/reports/`

## Overview

| Class | Description | Use Case |
|-------|-------------|----------|
| `ReportManager` | Main orchestrator for report generation | Generate HTML, LaTeX, or Markdown reports from any model |
| `GeneratedReport` | Container for rendered report content | Save, display, or embed reports |

The reports system follows a pipeline architecture:

```
results → Transformer → template context → TemplateManager → rendered → file
```

Each model type has a dedicated transformer that extracts coefficients,
diagnostics, and statistics into a standardized template context.

---

## ReportManager

Main orchestrator that auto-detects the model type, applies the correct
transformer, and renders the report in the requested format and theme.

```python
ReportManager(
    template_dir: str | Path | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `template_dir` | `str \| Path \| None` | `None` | Custom template directory. If None, uses built-in templates |

### `.generate()` Method

Generate a complete report from model results.

```python
ReportManager.generate(
    results: Any,
    report_type: str = "auto",
    fmt: str = "html",
    theme: str = "professional",
    title: str | None = None,
    include_plots: bool = True,
    **kwargs,
) -> GeneratedReport
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `Any` | *required* | Model results object (ARIMAResults, VARResults, TestResult, etc.) |
| `report_type` | `str` | `"auto"` | `'auto'`, `'arima'`, `'ets'`, `'var'`, `'svar'`, `'bvar'`, `'ardl'`, `'tests'`, `'spillover'` |
| `fmt` | `str` | `"html"` | Output format: `'html'`, `'latex'`, `'markdown'` |
| `theme` | `str` | `"professional"` | Visual theme: `'professional'`, `'academic'`, `'presentation'`, `'bcb'` |
| `title` | `str \| None` | `None` | Custom report title |
| `include_plots` | `bool` | `True` | Whether to include inline plots |

**Returns**: [`GeneratedReport`](#generatedreport)

### `.detect_report_type()` Method

Auto-detect the report type from the results class.

```python
ReportManager.detect_report_type(results: Any) -> str
```

The detection maps result class names to report types:

| Results Class | Report Type |
|---------------|-------------|
| `ARIMAResults`, `SARIMAResults`, `TimeSeriesResults` | `arima` |
| `ETSResults` | `ets` |
| `VARResults`, `VECMResults` | `var` |
| `SVARResults` | `svar` |
| `BVARResults` | `bvar` |
| `ARDLResults` | `ardl` |
| `SpilloverResults` | `spillover` |
| `TestResult` | `tests` |

### `.register_transformer()` Method

Register a custom transformer for new model types.

```python
ReportManager.register_transformer(
    report_type: str,
    transformer: Any,
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `report_type` | `str` | *required* | Report type name |
| `transformer` | `Any` | *required* | Transformer instance with `transform(results) -> dict` method |

### Output Formats

=== "HTML"

    Full-featured HTML reports with CSS styling, interactive tables, and
    embedded plots.

    ```python
    rm = ReportManager()
    report = rm.generate(results, fmt="html", theme="professional")
    report.save("analysis.html")
    ```

=== "LaTeX"

    Publication-ready LaTeX documents with `booktabs` tables, proper
    mathematical notation, and figure floats.

    ```python
    rm = ReportManager()
    report = rm.generate(results, fmt="latex", theme="academic")
    report.save("analysis.tex")
    ```

=== "Markdown"

    Clean Markdown output suitable for documentation, notebooks, or
    further processing.

    ```python
    rm = ReportManager()
    report = rm.generate(results, fmt="markdown")
    report.save("analysis.md")
    ```

### Example

```python
import numpy as np
from chronobox import ARIMA
from chronobox.reports import ReportManager

# Fit model
rng = np.random.default_rng(42)
y = np.cumsum(rng.normal(size=200))
model = ARIMA(order=(1, 1, 1))
results = model.fit(y)

# Generate HTML report (auto-detects ARIMA)
rm = ReportManager()
report = rm.generate(results, fmt="html", theme="professional")
report.save("arima_report.html")
print(f"Report saved: {report}")

# Generate LaTeX for academic paper
report_tex = rm.generate(
    results,
    fmt="latex",
    theme="academic",
    title="ARIMA(1,1,1) Estimation Results",
)
report_tex.save("arima_results.tex")

# Generate Markdown for documentation
report_md = rm.generate(results, fmt="markdown")
print(report_md.content[:200])
```

::: chronobox.reports.report_manager.ReportManager
    options:
      show_root_heading: false
      show_source: true
      members:
        - generate
        - detect_report_type
        - register_transformer

---

## GeneratedReport

Container for a generated report with content, format metadata, and
save functionality.

```python
@dataclass
class GeneratedReport:
    content: str
    fmt: str
    report_type: str
    theme: str
    metadata: dict[str, Any] = {}
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `content` | `str` | Rendered report content (HTML, LaTeX, or Markdown) |
| `fmt` | `str` | Output format (`'html'`, `'latex'`, `'markdown'`) |
| `report_type` | `str` | Report type (e.g., `'arima'`, `'var'`) |
| `theme` | `str` | Theme used for rendering |
| `metadata` | `dict` | Additional metadata (results class, include_plots flag) |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `save(path)` | `Path` | Save report to file |
| `__str__()` | `str` | Return report content as string |

### Example

```python
from chronobox.reports import ReportManager

rm = ReportManager()
report = rm.generate(results, fmt="html")

# Save to file
path = report.save("output/report.html")
print(f"Saved to {path}")

# Access content directly
print(report.content[:500])

# Inspect metadata
print(report.fmt)          # 'html'
print(report.report_type)  # 'arima'
print(report.theme)        # 'professional'
```

::: chronobox.reports.report_manager.GeneratedReport
    options:
      show_root_heading: true
      show_source: false
      members:
        - save

---

## Supported Report Types

| Report Type | Models | Key Sections |
|-------------|--------|--------------|
| `arima` | ARIMA, SARIMA | Coefficients, information criteria, residual diagnostics |
| `ets` | ETS, Holt-Winters | Smoothing parameters, components, forecast accuracy |
| `var` | VAR, VECM | Coefficient matrices, IRF, FEVD, Granger causality |
| `svar` | SVAR | Structural identification, structural IRF |
| `bvar` | Bayesian VAR | Prior specification, posterior distributions |
| `ardl` | ARDL, ECM | Short-run and long-run coefficients, bounds test |
| `spillover` | SpilloverIndex | Spillover table, directional measures, network |
| `tests` | TestResult | Test statistics, critical values, decision |

---

## Transformers

Each report type has a dedicated transformer that converts model results into
a standardized template context.

| Transformer | Module | Source |
|-------------|--------|--------|
| `ARIMATransformer` | `arima` | `chronobox/reports/transformers/arima_transformer.py` |
| `ETSTransformer` | `ets` | `chronobox/reports/transformers/ets_transformer.py` |
| `VARTransformer` | `var` | `chronobox/reports/transformers/var_transformer.py` |
| `SVARTransformer` | `svar` | `chronobox/reports/transformers/svar_transformer.py` |
| `BVARTransformer` | `bvar` | `chronobox/reports/transformers/bvar_transformer.py` |
| `ARDLTransformer` | `ardl` | `chronobox/reports/transformers/ardl_transformer.py` |
| `TestsTransformer` | `tests` | `chronobox/reports/transformers/tests_transformer.py` |
| `SpilloverTransformer` | `spillover` | `chronobox/reports/transformers/spillover_transformer.py` |

### Custom Transformer

```python
from chronobox.reports import ReportManager

class MyTransformer:
    def transform(self, results, **kwargs):
        return {
            "title": "My Custom Report",
            "sections": [
                {"title": "Summary", "content": str(results)},
            ],
        }

rm = ReportManager()
rm.register_transformer("custom", MyTransformer())
report = rm.generate(results, report_type="custom", fmt="markdown")
```

---

## Complete Workflow Example

```python
import numpy as np
from chronobox import ARIMA, auto_arima
from chronobox.tests_stat import adf_test, ljung_box_test
from chronobox.reports import ReportManager

# Prepare data
rng = np.random.default_rng(42)
y = np.cumsum(rng.normal(size=300))

# Run diagnostics
adf = adf_test(y)

# Fit models
results_111 = ARIMA(order=(1, 1, 1)).fit(y)
results_auto = auto_arima(y, seasonal=False)

# Generate reports
rm = ReportManager()

# Model report
report = rm.generate(results_111, fmt="html", theme="professional",
                     title="ARIMA(1,1,1) Analysis")
report.save("arima_report.html")

# Test report
test_report = rm.generate(adf, fmt="html", report_type="tests")
test_report.save("test_report.html")

# LaTeX for paper
tex = rm.generate(results_111, fmt="latex", theme="academic")
tex.save("results_table.tex")
```

---

## See Also

- [Core API](core.md) -- `TimeSeriesResults` attributes used by transformers
- [Experiment API](experiment.md) -- `ChronoExperiment.save_master_report()` for automated reports
- [Visualization API](visualization.md) -- Plots embedded in reports
