"""ChronoBox Experiment Pattern - Core implementation.

Provides a systematic workflow for comparing time series models,
performing time series cross-validation, and generating comprehensive reports.

Example
-------
>>> from chronobox.experiment import ChronoExperiment
>>> from chronobox.datasets import load_dataset
>>> data = load_dataset('airline')
>>> exp = ChronoExperiment(data)
>>> exp.fit_all_models([('ARIMA(1,1,1)', {'order': (1,1,1)})])
>>> comparison = exp.compare_models(criteria=['aic', 'bic'])
>>> print(comparison.ranking())
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# ComparisonResult
# ──────────────────────────────────────────────

@dataclass
class ComparisonResult:
    """Result of comparing multiple fitted models.

    Attributes
    ----------
    models : dict[str, Any]
        Fitted model results keyed by model name.
    criteria : list[str]
        Criteria used for comparison (e.g., 'aic', 'bic', 'aicc', 'rmse').
    scores : pd.DataFrame
        DataFrame with models as rows and criteria as columns.
    fit_times : dict[str, float]
        Time taken to fit each model in seconds.
    """

    models: dict[str, Any]
    criteria: list[str]
    scores: pd.DataFrame
    fit_times: dict[str, float] = field(default_factory=dict)

    def ranking(self, criterion: str | None = None) -> pd.DataFrame:
        """Rank models by the specified criterion or overall.

        Parameters
        ----------
        criterion : str | None
            Criterion to rank by. If None, ranks by the first criterion.

        Returns
        -------
        pd.DataFrame
            Ranked DataFrame with rank column added.
        """
        if criterion is None:
            criterion = self.criteria[0]
        if criterion not in self.scores.columns:
            raise ValueError(f"Criterion '{criterion}' not found. Available: {list(self.scores.columns)}")

        ranked = self.scores.sort_values(criterion)
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        return ranked

    def best_model(self, criterion: str | None = None) -> str:
        """Return the name of the best model.

        Parameters
        ----------
        criterion : str | None
            Criterion to use. If None, uses the first criterion.

        Returns
        -------
        str
            Name of the best model.
        """
        ranking = self.ranking(criterion)
        return str(ranking.index[0])

    def to_dataframe(self) -> pd.DataFrame:
        """Return scores as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Scores DataFrame.
        """
        return self.scores.copy()

    def plot_comparison(
        self,
        criterion: str | None = None,
        figsize: tuple[int, int] = (10, 6),
        ax: Any = None,
    ) -> Any:
        """Plot bar chart comparing models.

        Parameters
        ----------
        criterion : str | None
            Criterion to plot. If None, plots all criteria.
        figsize : tuple[int, int]
            Figure size.
        ax : matplotlib.axes.Axes | None
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if criterion is not None:
            self.scores[criterion].plot(kind="bar", ax=ax)
            ax.set_title(f"Model Comparison - {criterion.upper()}")
            ax.set_ylabel(criterion.upper())
        else:
            self.scores.plot(kind="bar", ax=ax)
            ax.set_title("Model Comparison")
            ax.legend(title="Criteria")

        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        return ax


# ──────────────────────────────────────────────
# ValidationResult
# ──────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of train/test validation for a single model.

    Attributes
    ----------
    model_name : str
        Name of the validated model.
    train_size : int
        Number of observations used for training.
    test_size : int
        Number of observations used for testing.
    horizon : int
        Forecast horizon.
    actuals : np.ndarray
        Actual test values.
    forecasts : np.ndarray
        Forecasted values.
    confidence_intervals : np.ndarray | None
        Confidence intervals (n, 2) array with lower and upper bounds.
    fitted_model : Any
        The fitted model result object.
    """

    model_name: str
    train_size: int
    test_size: int
    horizon: int
    actuals: np.ndarray
    forecasts: np.ndarray
    confidence_intervals: np.ndarray | None = None
    fitted_model: Any = None

    def rmse(self) -> float:
        """Root Mean Squared Error.

        Returns
        -------
        float
            RMSE value.
        """
        n = min(len(self.actuals), len(self.forecasts))
        errors = self.actuals[:n] - self.forecasts[:n]
        return float(np.sqrt(np.mean(errors ** 2)))

    def mae(self) -> float:
        """Mean Absolute Error.

        Returns
        -------
        float
            MAE value.
        """
        n = min(len(self.actuals), len(self.forecasts))
        errors = self.actuals[:n] - self.forecasts[:n]
        return float(np.mean(np.abs(errors)))

    def mape(self) -> float:
        """Mean Absolute Percentage Error.

        Returns
        -------
        float
            MAPE value as percentage.
        """
        n = min(len(self.actuals), len(self.forecasts))
        actuals = self.actuals[:n]
        forecasts = self.forecasts[:n]
        mask = actuals != 0
        if not np.any(mask):
            return float("inf")
        return float(np.mean(np.abs((actuals[mask] - forecasts[mask]) / actuals[mask])) * 100)

    def coverage(self, alpha: float = 0.05) -> float:
        """Empirical coverage of confidence intervals.

        Parameters
        ----------
        alpha : float
            Significance level. Default is 0.05 (95% CI).

        Returns
        -------
        float
            Proportion of actuals within the confidence interval.
        """
        if self.confidence_intervals is None:
            warnings.warn("No confidence intervals available.", stacklevel=2)
            return float("nan")

        n = min(len(self.actuals), len(self.confidence_intervals))
        lower = self.confidence_intervals[:n, 0]
        upper = self.confidence_intervals[:n, 1]
        actuals = self.actuals[:n]
        within = np.sum((actuals >= lower) & (actuals <= upper))
        return float(within / n)

    def plot_forecast_vs_actual(
        self,
        figsize: tuple[int, int] = (12, 6),
        ax: Any = None,
    ) -> Any:
        """Plot forecast vs actual values.

        Parameters
        ----------
        figsize : tuple[int, int]
            Figure size.
        ax : matplotlib.axes.Axes | None
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        n = min(len(self.actuals), len(self.forecasts))
        x = np.arange(n)

        ax.plot(x, self.actuals[:n], "b-o", label="Actual", markersize=4)
        ax.plot(x, self.forecasts[:n], "r--s", label="Forecast", markersize=4)

        if self.confidence_intervals is not None:
            ci_n = min(n, len(self.confidence_intervals))
            ax.fill_between(
                x[:ci_n],
                self.confidence_intervals[:ci_n, 0],
                self.confidence_intervals[:ci_n, 1],
                alpha=0.2,
                color="red",
                label="95% CI",
            )

        ax.set_title(f"Forecast vs Actual - {self.model_name}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend()
        metrics_text = f"RMSE: {self.rmse():.4f}\nMAE: {self.mae():.4f}\nMAPE: {self.mape():.2f}%"
        ax.text(
            0.02, 0.98, metrics_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        plt.tight_layout()
        return ax


# ──────────────────────────────────────────────
# CVResult
# ──────────────────────────────────────────────

@dataclass
class CVResult:
    """Result of time series cross-validation.

    Attributes
    ----------
    model_name : str
        Name of the model.
    n_splits : int
        Number of CV folds.
    horizon : int
        Forecast horizon per fold.
    fold_scores : list[dict[str, float]]
        List of score dictionaries per fold (keys: rmse, mae, mape).
    fold_forecasts : list[np.ndarray]
        Forecasts for each fold.
    fold_actuals : list[np.ndarray]
        Actuals for each fold.
    """

    model_name: str
    n_splits: int
    horizon: int
    fold_scores: list[dict[str, float]]
    fold_forecasts: list[np.ndarray]
    fold_actuals: list[np.ndarray]

    @property
    def scores_df(self) -> pd.DataFrame:
        """Scores per fold as DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with folds as rows and metrics as columns.
        """
        return pd.DataFrame(self.fold_scores, index=[f"Fold {i+1}" for i in range(self.n_splits)])

    def mean_scores(self) -> dict[str, float]:
        """Mean scores across folds.

        Returns
        -------
        dict[str, float]
            Mean of each metric across all folds.
        """
        df = self.scores_df
        return {col: float(df[col].mean()) for col in df.columns}

    def std_scores(self) -> dict[str, float]:
        """Standard deviation of scores across folds.

        Returns
        -------
        dict[str, float]
            Std of each metric across all folds.
        """
        df = self.scores_df
        return {col: float(df[col].std()) for col in df.columns}

    def plot_cv_errors(
        self,
        metric: str = "rmse",
        figsize: tuple[int, int] = (10, 6),
        ax: Any = None,
    ) -> Any:
        """Plot CV errors across folds.

        Parameters
        ----------
        metric : str
            Metric to plot (rmse, mae, mape).
        figsize : tuple[int, int]
            Figure size.
        ax : matplotlib.axes.Axes | None
            Axes to plot on.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        df = self.scores_df
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found. Available: {list(df.columns)}")

        folds = range(1, self.n_splits + 1)
        values = df[metric].values
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))

        ax.bar(folds, values, alpha=0.7, color="steelblue", edgecolor="black")
        ax.axhline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.4f}")
        ax.fill_between(
            [0.5, self.n_splits + 0.5],
            mean_val - std_val,
            mean_val + std_val,
            alpha=0.1,
            color="red",
            label=f"Std: {std_val:.4f}",
        )

        ax.set_title(f"CV {metric.upper()} - {self.model_name}")
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric.upper())
        ax.set_xticks(list(folds))
        ax.legend()
        plt.tight_layout()
        return ax


# ──────────────────────────────────────────────
# ChronoExperiment
# ──────────────────────────────────────────────

class ChronoExperiment:
    """Systematic model comparison and validation framework.

    Parameters
    ----------
    data : pd.Series | pd.DataFrame
        Time series data for analysis.
    name : str
        Name for the experiment.

    Example
    -------
    >>> exp = ChronoExperiment(data, name="Airline Analysis")
    >>> exp.fit_all_models([
    ...     ('ARIMA(0,1,1)', {'order': (0,1,1)}),
    ...     ('ARIMA(1,1,1)', {'order': (1,1,1)}),
    ... ])
    >>> comparison = exp.compare_models(criteria=['aic', 'bic'])
    >>> print(comparison.best_model())
    """

    def __init__(
        self,
        data: pd.Series | pd.DataFrame,
        name: str = "Experiment",
    ) -> None:
        self.data = data
        self.name = name
        self._fitted_models: dict[str, Any] = {}
        self._fit_times: dict[str, float] = {}
        self._model_configs: dict[str, dict[str, Any]] = {}

    @property
    def fitted_models(self) -> dict[str, Any]:
        """Access fitted model results.

        Returns
        -------
        dict[str, Any]
            Dictionary of model name -> fitted result.
        """
        return self._fitted_models.copy()

    def fit_model(
        self,
        name: str,
        config: dict[str, Any],
    ) -> Any:
        """Fit a single model.

        Parameters
        ----------
        name : str
            Name for the model.
        config : dict[str, Any]
            Model configuration. Must include 'model_type' or infer from 'order'.
            Keys: model_type, order, seasonal_order, maxlags, k_ar_diff, coint_rank, etc.

        Returns
        -------
        Any
            Fitted model result.
        """
        model_type = config.get("model_type", "arima")
        start_time = time.perf_counter()

        if model_type in ("arima", "sarima") or "order" in config:
            from chronobox import ARIMA

            order = config.get("order", (1, 0, 0))
            seasonal_order = config.get("seasonal_order")

            if seasonal_order:
                model = ARIMA(order=order, seasonal_order=seasonal_order)
            else:
                model = ARIMA(order=order)

            result = model.fit(self.data)

        elif model_type == "auto_arima":
            from chronobox import auto_arima

            seasonal = config.get("seasonal", False)
            m = config.get("m", 1)
            result = auto_arima(self.data, seasonal=seasonal, m=m)

        elif model_type == "var":
            from chronobox.models.var import VAR

            maxlags = config.get("maxlags", 8)
            model = VAR(self.data)
            result = model.fit(maxlags=maxlags)

        elif model_type == "vecm":
            from chronobox.models.vecm import VECM

            k_ar_diff = config.get("k_ar_diff", 1)
            coint_rank = config.get("coint_rank", 1)
            model = VECM(self.data, k_ar_diff=k_ar_diff, coint_rank=coint_rank)
            result = model.fit()

        elif model_type == "ardl":
            from chronobox.models.ardl import ARDL

            lags = config.get("lags")
            model = ARDL(self.data, lags=lags)
            result = model.fit()

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        elapsed = time.perf_counter() - start_time
        self._fitted_models[name] = result
        self._fit_times[name] = elapsed
        self._model_configs[name] = config
        return result

    def fit_all_models(
        self,
        model_specs: list[tuple[str, dict[str, Any]]],
    ) -> dict[str, Any]:
        """Fit multiple models.

        Parameters
        ----------
        model_specs : list[tuple[str, dict[str, Any]]]
            List of (name, config) tuples.

        Returns
        -------
        dict[str, Any]
            Dictionary of model name -> fitted result.
        """
        results: dict[str, Any] = {}
        for name, config in model_specs:
            try:
                result = self.fit_model(name, config)
                results[name] = result
            except Exception as e:
                warnings.warn(f"Failed to fit model '{name}': {e}", stacklevel=2)
        return results

    def compare_models(
        self,
        criteria: list[str] | None = None,
    ) -> ComparisonResult:
        """Compare all fitted models.

        Parameters
        ----------
        criteria : list[str] | None
            Criteria to use for comparison. Default: ['aic', 'bic'].
            Options: 'aic', 'bic', 'aicc', 'hqic', 'rmse', 'loglike'.

        Returns
        -------
        ComparisonResult
            Comparison result with rankings.

        Raises
        ------
        ValueError
            If no models have been fitted.
        """
        if not self._fitted_models:
            raise ValueError("No models fitted. Call fit_all_models() first.")

        if criteria is None:
            criteria = ["aic", "bic"]

        scores_dict: dict[str, dict[str, float]] = {}

        for name, result in self._fitted_models.items():
            model_scores: dict[str, float] = {}
            for criterion in criteria:
                if criterion == "rmse" and hasattr(result, "resid"):
                    resid = np.asarray(result.resid)
                    model_scores["rmse"] = float(np.sqrt(np.mean(resid ** 2)))
                elif criterion == "loglike" and hasattr(result, "loglike"):
                    model_scores["loglike"] = float(-result.loglike)  # negate for ranking
                elif hasattr(result, criterion):
                    val = getattr(result, criterion)
                    model_scores[criterion] = float(val) if val is not None else float("inf")
                else:
                    model_scores[criterion] = float("inf")
            scores_dict[name] = model_scores

        scores_df = pd.DataFrame(scores_dict).T
        scores_df.index.name = "model"

        return ComparisonResult(
            models=self._fitted_models.copy(),
            criteria=criteria,
            scores=scores_df,
            fit_times=self._fit_times.copy(),
        )

    def validate_model(
        self,
        model_name: str,
        test_size: int = 24,
        horizon: int = 12,
    ) -> ValidationResult:
        """Validate a model using train/test split.

        Parameters
        ----------
        model_name : str
            Name of the model (must match a config from fit_all_models).
        test_size : int
            Number of observations to hold out for testing.
        horizon : int
            Forecast horizon.

        Returns
        -------
        ValidationResult
            Validation result with metrics.
        """
        if model_name not in self._model_configs:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self._model_configs.keys())}")

        config = self._model_configs[model_name]
        data = self.data

        if isinstance(data, pd.Series):
            train = data.iloc[:-test_size]
            test = data.iloc[-test_size:]
        else:
            train = data.iloc[:-test_size]
            test = data.iloc[-test_size:]

        # Re-fit on training data
        model_type = config.get("model_type", "arima")

        if model_type in ("arima", "sarima") or "order" in config:
            from chronobox import ARIMA

            order = config.get("order", (1, 0, 0))
            seasonal_order = config.get("seasonal_order", None)

            if seasonal_order:
                model = ARIMA(order=order, seasonal_order=seasonal_order)
            else:
                model = ARIMA(order=order)

            result = model.fit(train)

        elif model_type == "auto_arima":
            from chronobox import auto_arima

            seasonal = config.get("seasonal", False)
            m = config.get("m", 1)
            result = auto_arima(train, seasonal=seasonal, m=m)

        else:
            raise ValueError(f"Validation not supported for model type: {model_type}")

        forecast_result = result.forecast(steps=min(horizon, test_size))
        forecasts = np.asarray(forecast_result)

        if isinstance(test, pd.Series):
            actuals = test.values[:len(forecasts)]
        else:
            actuals = test.values[:len(forecasts)]

        # Try to get confidence intervals
        ci = None
        if hasattr(result, "forecast_ci"):
            try:
                ci_result = result.forecast_ci(steps=min(horizon, test_size))
                ci = np.asarray(ci_result)
            except Exception:
                pass

        return ValidationResult(
            model_name=model_name,
            train_size=len(train),
            test_size=len(test),
            horizon=horizon,
            actuals=actuals,
            forecasts=forecasts,
            confidence_intervals=ci,
            fitted_model=result,
        )

    def time_series_cv(
        self,
        model_name: str,
        n_splits: int = 5,
        horizon: int = 12,
        min_train_size: int | None = None,
    ) -> CVResult:
        """Perform time series cross-validation.

        Uses expanding window cross-validation where each fold adds more
        training data and tests on the next `horizon` observations.

        Parameters
        ----------
        model_name : str
            Name of the model to validate.
        n_splits : int
            Number of CV folds.
        horizon : int
            Forecast horizon per fold.
        min_train_size : int | None
            Minimum training size. If None, computed automatically.

        Returns
        -------
        CVResult
            Cross-validation result with per-fold scores.
        """
        if model_name not in self._model_configs:
            raise ValueError(f"Model '{model_name}' not found.")

        config = self._model_configs[model_name]
        data = self.data

        data_series = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data

        n = len(data_series)
        total_test = n_splits * horizon

        if min_train_size is None:
            min_train_size = max(n - total_test, int(n * 0.5))

        fold_scores: list[dict[str, float]] = []
        fold_forecasts: list[np.ndarray] = []
        fold_actuals: list[np.ndarray] = []

        for fold in range(n_splits):
            test_end = n - (n_splits - fold - 1) * horizon
            test_start = test_end - horizon
            train_end = test_start

            if train_end < min_train_size:
                warnings.warn(
                    f"Fold {fold + 1}: training size {train_end} < min {min_train_size}. Skipping.",
                    stacklevel=2,
                )
                continue

            train_data = data_series.iloc[:train_end]
            test_data = data_series.iloc[test_start:test_end]

            # Fit model on training data
            model_type = config.get("model_type", "arima")

            if model_type in ("arima", "sarima") or "order" in config:
                from chronobox import ARIMA

                order = config.get("order", (1, 0, 0))
                seasonal_order = config.get("seasonal_order", None)

                if seasonal_order:
                    model = ARIMA(order=order, seasonal_order=seasonal_order)
                else:
                    model = ARIMA(order=order)

                result = model.fit(train_data)

            elif model_type == "auto_arima":
                from chronobox import auto_arima

                seasonal = config.get("seasonal", False)
                m = config.get("m", 1)
                result = auto_arima(train_data, seasonal=seasonal, m=m)

            else:
                raise ValueError(f"CV not supported for model type: {model_type}")

            forecasts = np.asarray(result.forecast(steps=horizon))
            actuals = test_data.values

            n_compare = min(len(actuals), len(forecasts))
            errors = actuals[:n_compare] - forecasts[:n_compare]
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            mae = float(np.mean(np.abs(errors)))

            mask = actuals[:n_compare] != 0
            if np.any(mask):
                mape = float(np.mean(np.abs(errors[mask] / actuals[:n_compare][mask])) * 100)
            else:
                mape = float("inf")

            fold_scores.append({"rmse": rmse, "mae": mae, "mape": mape})
            fold_forecasts.append(forecasts)
            fold_actuals.append(actuals)

        return CVResult(
            model_name=model_name,
            n_splits=len(fold_scores),
            horizon=horizon,
            fold_scores=fold_scores,
            fold_forecasts=fold_forecasts,
            fold_actuals=fold_actuals,
        )

    def save_master_report(
        self,
        filepath: str,
        theme: Literal["professional", "minimal", "dark"] = "professional",
    ) -> None:
        """Save a comprehensive HTML report.

        Parameters
        ----------
        filepath : str
            Path to save the HTML report.
        theme : str
            Visual theme: 'professional', 'minimal', or 'dark'.
        """
        from pathlib import Path

        themes = {
            "professional": {
                "bg": "#ffffff",
                "text": "#333333",
                "accent": "#2c3e50",
                "header_bg": "#2c3e50",
                "header_text": "#ffffff",
                "table_stripe": "#f8f9fa",
                "border": "#dee2e6",
            },
            "minimal": {
                "bg": "#ffffff",
                "text": "#000000",
                "accent": "#000000",
                "header_bg": "#f0f0f0",
                "header_text": "#000000",
                "table_stripe": "#fafafa",
                "border": "#dddddd",
            },
            "dark": {
                "bg": "#1a1a2e",
                "text": "#e0e0e0",
                "accent": "#0f3460",
                "header_bg": "#16213e",
                "header_text": "#e94560",
                "table_stripe": "#1a1a2e",
                "border": "#333366",
            },
        }

        t = themes.get(theme, themes["professional"])

        html_parts: list[str] = []
        html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChronoBox Report - {self.name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: {t['bg']};
            color: {t['text']};
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            background-color: {t['header_bg']};
            color: {t['header_text']};
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: {t['accent']};
            border-bottom: 2px solid {t['accent']};
            padding-bottom: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background-color: {t['header_bg']};
            color: {t['header_text']};
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid {t['border']};
        }}
        tr:nth-child(even) {{
            background-color: {t['table_stripe']};
        }}
        .metric-card {{
            display: inline-block;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 8px;
            border: 1px solid {t['border']};
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: {t['accent']};
        }}
        .metric-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid {t['border']};
            font-size: 12px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <h1>ChronoBox Master Report - {self.name}</h1>
""")

        # Data summary
        html_parts.append("<h2>Data Summary</h2>")
        if isinstance(self.data, pd.Series):
            html_parts.append(f"<p>Series length: {len(self.data)}</p>")
            html_parts.append(f"<p>Start: {self.data.index[0]}, End: {self.data.index[-1]}</p>")
        else:
            html_parts.append(f"<p>Shape: {self.data.shape[0]} observations x {self.data.shape[1]} variables</p>")
            html_parts.append(f"<p>Columns: {', '.join(self.data.columns)}</p>")

        # Fitted models
        if self._fitted_models:
            html_parts.append("<h2>Fitted Models</h2>")
            html_parts.append("<table><tr><th>Model</th><th>Fit Time (s)</th><th>Config</th></tr>")
            for name in self._fitted_models:
                t_val = self._fit_times.get(name, 0.0)
                cfg = str(self._model_configs.get(name, {}))
                html_parts.append(f"<tr><td>{name}</td><td>{t_val:.4f}</td><td>{cfg}</td></tr>")
            html_parts.append("</table>")

        # Comparison
        if self._fitted_models:
            try:
                comparison = self.compare_models()
                html_parts.append("<h2>Model Comparison</h2>")
                html_parts.append(comparison.scores.to_html(classes="comparison-table"))
                best = comparison.best_model()
                html_parts.append(f"<p><strong>Best model:</strong> {best}</p>")
            except Exception:
                pass

        # Footer
        html_parts.append("""
    <div class="footer">
        <p>Generated by ChronoBox v0.1.0</p>
    </div>
</body>
</html>""")

        Path(filepath).write_text("\n".join(html_parts), encoding="utf-8")
