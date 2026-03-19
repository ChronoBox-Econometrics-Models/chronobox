# Reports

## Overview

ChronoBox generates professional HTML, Markdown, and LaTeX reports
from model results with embedded diagnostics, tables, and plots.

## Basic Usage

```python
from chronobox.reports import ReportManager

rm = ReportManager()
report = rm.generate(results, format='html')
report.save('arima_report.html')
```

## Supported Formats

| Format | Exporter | Description |
|--------|----------|-------------|
| HTML | `HTMLExporter` | Interactive web report |
| Markdown | `MarkdownExporter` | Plain text report |
| LaTeX | `LaTeXExporter` | Publication-quality PDF |

## Model-specific Reports

Reports are automatically customized by model type:

```python
# ARIMA report
report = rm.generate(arima_results, format='html')

# VAR report (includes IRF, FEVD, Granger causality)
report = rm.generate(var_results, format='html')

# SVAR report (includes structural IRF, HD)
report = rm.generate(svar_results, format='html')
```

## Available Transformers

| Transformer | Model |
|-------------|-------|
| `ARIMATransformer` | ARIMA/SARIMA |
| `ETSTransformer` | ETS |
| `VARTransformer` | VAR |
| `SVARTransformer` | SVAR |
| `BVARTransformer` | Bayesian VAR |
| `ARDLTransformer` | ARDL |
| `SpilloverTransformer` | Spillover analysis |
| `TestsTransformer` | Statistical tests |

## Report Content

Reports typically include:

- Model specification and parameters
- Coefficient estimates with standard errors
- Information criteria (AIC, BIC, AICc)
- Residual diagnostics
- Plots (forecast, diagnostics, IRF/FEVD)
- Test results
