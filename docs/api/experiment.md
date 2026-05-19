---
title: "Experiment API"
description: "API reference for ChronoExperiment — systematic model comparison, validation, and cross-validation"
---

# Experiment API Reference

!!! info "Module"
    **Import**: `from chronobox.experiment import ChronoExperiment, ComparisonResult, ValidationResult, CVResult`
    **Source**: `chronobox/experiment/experiment.py`

## Overview

| Class | Description | Use Case |
|-------|-------------|----------|
| `ChronoExperiment` | Systematic model comparison framework | Fit, compare, and validate multiple models |
| `ComparisonResult` | Result of model comparison | Rank models by AIC, BIC, RMSE, etc. |
| `ValidationResult` | Result of train/test validation | Out-of-sample forecast accuracy |
| `CVResult` | Result of time series cross-validation | Robust performance estimation |

---

## ChronoExperiment

Automated workflow for fitting, comparing, and validating time series models.

```python
ChronoExperiment(
    data: pd.Series | pd.DataFrame,
    name: str = "Experiment",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.Series \| pd.DataFrame` | *required* | Time series data |
| `name` | `str` | `"Experiment"` | Name for the experiment |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `fitted_models` | `dict[str, Any]` | Dictionary of model name → fitted result |

### `.fit_model()` Method

Fit a single model by name and configuration.

```python
ChronoExperiment.fit_model(
    name: str,
    config: dict[str, Any],
) -> Any
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Name for the model |
| `config` | `dict[str, Any]` | *required* | Model configuration (see table below) |

**Configuration keys:**

| Key | Type | Description |
|-----|------|-------------|
| `model_type` | `str` | `'arima'`, `'auto_arima'`, `'var'`, `'vecm'`, `'ardl'` |
| `order` | `tuple` | ARIMA order `(p, d, q)` |
| `seasonal_order` | `tuple` | Seasonal order `(P, D, Q, s)` |
| `maxlags` | `int` | Max lags for VAR |
| `k_ar_diff` | `int` | Differencing lags for VECM |
| `coint_rank` | `int` | Cointegration rank for VECM |
| `seasonal` | `bool` | Enable seasonal for auto_arima |
| `m` | `int` | Seasonal period for auto_arima |

**Returns**: Fitted model result

### `.fit_all_models()` Method

Fit multiple models in sequence.

```python
ChronoExperiment.fit_all_models(
    model_specs: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_specs` | `list[tuple[str, dict]]` | *required* | List of `(name, config)` tuples |

**Returns**: `dict[str, Any]` — model name → fitted result

!!! tip "Error handling"
    Models that fail to fit are skipped with a warning. The experiment
    continues with the remaining models.

### `.compare_models()` Method

Compare all fitted models by information criteria or forecast accuracy.

```python
ChronoExperiment.compare_models(
    criteria: list[str] | None = None,
) -> ComparisonResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `criteria` | `list[str] \| None` | `None` | Criteria for comparison. Default: `['aic', 'bic']`. Options: `'aic'`, `'bic'`, `'aicc'`, `'hqic'`, `'rmse'`, `'loglike'` |

**Returns**: [`ComparisonResult`](#comparisonresult)

### `.validate_model()` Method

Validate a model using train/test split.

```python
ChronoExperiment.validate_model(
    model_name: str,
    test_size: int = 24,
    horizon: int = 12,
) -> ValidationResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | *required* | Name of the model to validate |
| `test_size` | `int` | `24` | Observations held out for testing |
| `horizon` | `int` | `12` | Forecast horizon |

**Returns**: [`ValidationResult`](#validationresult)

### `.time_series_cv()` Method

Expanding-window time series cross-validation.

```python
ChronoExperiment.time_series_cv(
    model_name: str,
    n_splits: int = 5,
    horizon: int = 12,
    min_train_size: int | None = None,
) -> CVResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | *required* | Name of the model to validate |
| `n_splits` | `int` | `5` | Number of CV folds |
| `horizon` | `int` | `12` | Forecast horizon per fold |
| `min_train_size` | `int \| None` | `None` | Minimum training size. Auto-computed if None |

**Returns**: [`CVResult`](#cvresult)

### `.save_master_report()` Method

Save a comprehensive HTML report with data summary, fitted models, and
comparison rankings.

```python
ChronoExperiment.save_master_report(
    filepath: str,
    theme: Literal["professional", "minimal", "dark"] = "professional",
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str` | *required* | Path to save the HTML report |
| `theme` | `str` | `"professional"` | Visual theme: `'professional'`, `'minimal'`, `'dark'` |

### Complete Example

```python
import numpy as np
from chronobox.experiment import ChronoExperiment
from chronobox.datasets import load_dataset

# Load data
data = load_dataset("airline")

# Create experiment
exp = ChronoExperiment(data, name="Airline Passengers")

# Fit multiple models
exp.fit_all_models([
    ("ARIMA(0,1,1)", {"order": (0, 1, 1)}),
    ("ARIMA(1,1,1)", {"order": (1, 1, 1)}),
    ("SARIMA(1,1,1)(1,1,1)12", {
        "order": (1, 1, 1),
        "seasonal_order": (1, 1, 1, 12),
    }),
    ("Auto-ARIMA", {
        "model_type": "auto_arima",
        "seasonal": True,
        "m": 12,
    }),
])

# Compare by AIC, BIC, and in-sample RMSE
comparison = exp.compare_models(criteria=["aic", "bic", "rmse"])
print(comparison.ranking("aic"))
print(f"Best model: {comparison.best_model('aic')}")

# Validate best model out-of-sample
val = exp.validate_model("SARIMA(1,1,1)(1,1,1)12", test_size=24, horizon=12)
print(f"RMSE: {val.rmse():.4f}")
print(f"MAE:  {val.mae():.4f}")
print(f"MAPE: {val.mape():.2f}%")

# Cross-validation
cv = exp.time_series_cv("SARIMA(1,1,1)(1,1,1)12", n_splits=5, horizon=12)
print(f"CV Mean RMSE: {cv.mean_scores()['rmse']:.4f}")
print(f"CV Std RMSE:  {cv.std_scores()['rmse']:.4f}")

# Save master report
exp.save_master_report("airline_experiment.html", theme="professional")
```

::: chronobox.experiment.experiment.ChronoExperiment
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit_model
        - fit_all_models
        - compare_models
        - validate_model
        - time_series_cv
        - save_master_report

---

## ComparisonResult

Result of comparing multiple fitted models across selected criteria.

```python
@dataclass
class ComparisonResult:
    models: dict[str, Any]
    criteria: list[str]
    scores: pd.DataFrame
    fit_times: dict[str, float] = {}
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `models` | `dict[str, Any]` | Fitted model results keyed by name |
| `criteria` | `list[str]` | Criteria used (`'aic'`, `'bic'`, etc.) |
| `scores` | `pd.DataFrame` | Models as rows, criteria as columns |
| `fit_times` | `dict[str, float]` | Fit time per model (seconds) |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `ranking(criterion)` | `pd.DataFrame` | Rank models by criterion (lowest = best) |
| `best_model(criterion)` | `str` | Name of the best model |
| `to_dataframe()` | `pd.DataFrame` | Scores as DataFrame |
| `plot_comparison(criterion)` | `Axes` | Bar chart comparing models |

### Example

```python
comparison = exp.compare_models(criteria=["aic", "bic", "rmse"])

# Ranking table
print(comparison.ranking("aic"))
#                               rank       aic       bic      rmse
# SARIMA(1,1,1)(1,1,1)12           1   1020.34   1035.67     15.23
# Auto-ARIMA                       2   1022.11   1037.44     15.89
# ARIMA(1,1,1)                     3   1150.23   1160.12     22.45
# ARIMA(0,1,1)                     4   1155.67   1162.34     23.01

# Best model
print(comparison.best_model("bic"))  # "SARIMA(1,1,1)(1,1,1)12"

# Visualization
comparison.plot_comparison(criterion="aic")
```

::: chronobox.experiment.experiment.ComparisonResult
    options:
      show_root_heading: true
      show_source: false
      members:
        - ranking
        - best_model
        - to_dataframe
        - plot_comparison

---

## ValidationResult

Result of out-of-sample validation using train/test split.

```python
@dataclass
class ValidationResult:
    model_name: str
    train_size: int
    test_size: int
    horizon: int
    actuals: np.ndarray
    forecasts: np.ndarray
    confidence_intervals: np.ndarray | None = None
    fitted_model: Any = None
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model_name` | `str` | Name of the validated model |
| `train_size` | `int` | Training set size |
| `test_size` | `int` | Test set size |
| `horizon` | `int` | Forecast horizon |
| `actuals` | `ndarray` | Actual test values |
| `forecasts` | `ndarray` | Forecasted values |
| `confidence_intervals` | `ndarray \| None` | CI array `(n, 2)` with lower/upper bounds |
| `fitted_model` | `Any` | The fitted model result |

### Metrics

| Method | Returns | Description |
|--------|---------|-------------|
| `rmse()` | `float` | Root Mean Squared Error: $\sqrt{\frac{1}{n}\sum(y_t - \hat{y}_t)^2}$ |
| `mae()` | `float` | Mean Absolute Error: $\frac{1}{n}\sum\|y_t - \hat{y}_t\|$ |
| `mape()` | `float` | Mean Absolute Percentage Error (%) |
| `coverage(alpha)` | `float` | Empirical coverage of confidence intervals |
| `plot_forecast_vs_actual()` | `Axes` | Plot forecast vs actual with CI and metrics |

### Example

```python
val = exp.validate_model("ARIMA(1,1,1)", test_size=24, horizon=12)

print(f"RMSE: {val.rmse():.4f}")
print(f"MAE:  {val.mae():.4f}")
print(f"MAPE: {val.mape():.2f}%")
print(f"Coverage (95%): {val.coverage():.2%}")

# Visual comparison
val.plot_forecast_vs_actual()
```

::: chronobox.experiment.experiment.ValidationResult
    options:
      show_root_heading: true
      show_source: false
      members:
        - rmse
        - mae
        - mape
        - coverage
        - plot_forecast_vs_actual

---

## CVResult

Result of expanding-window time series cross-validation.

```python
@dataclass
class CVResult:
    model_name: str
    n_splits: int
    horizon: int
    fold_scores: list[dict[str, float]]
    fold_forecasts: list[np.ndarray]
    fold_actuals: list[np.ndarray]
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model_name` | `str` | Name of the model |
| `n_splits` | `int` | Number of CV folds |
| `horizon` | `int` | Forecast horizon per fold |
| `fold_scores` | `list[dict]` | Per-fold score dicts (keys: `rmse`, `mae`, `mape`) |
| `fold_forecasts` | `list[ndarray]` | Forecasts for each fold |
| `fold_actuals` | `list[ndarray]` | Actuals for each fold |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `scores_df` | `pd.DataFrame` | Per-fold scores as DataFrame |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `mean_scores()` | `dict[str, float]` | Mean of each metric across folds |
| `std_scores()` | `dict[str, float]` | Standard deviation across folds |
| `plot_cv_errors(metric)` | `Axes` | Bar chart of per-fold errors with mean/std bands |

### Example

```python
cv = exp.time_series_cv("ARIMA(1,1,1)", n_splits=5, horizon=12)

# Summary statistics
print(cv.scores_df)
#          rmse      mae     mape
# Fold 1  15.23    12.45    8.34
# Fold 2  14.89    11.98    7.92
# ...

print(f"Mean RMSE: {cv.mean_scores()['rmse']:.4f}")
print(f"Std RMSE:  {cv.std_scores()['rmse']:.4f}")

# Visualize stability across folds
cv.plot_cv_errors(metric="rmse")
```

::: chronobox.experiment.experiment.CVResult
    options:
      show_root_heading: true
      show_source: false
      members:
        - mean_scores
        - std_scores
        - plot_cv_errors

---

## See Also

- [Core API](core.md) -- `TimeSeriesResults` attributes used by experiments
- [ARIMA API](arima.md) -- ARIMA model configuration
- [Reports API](reports.md) -- `ReportManager` for detailed model reports
- [Experiment Guide](../user-guide/experiment.md) -- Step-by-step experiment guide
