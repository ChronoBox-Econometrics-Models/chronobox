"""Tests for ChronoExperiment pattern."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from chronobox.experiment import (
    ChronoExperiment,
    ComparisonResult,
    CVResult,
    ValidationResult,
)


class TestComparisonResult:
    """Tests for ComparisonResult."""

    def test_ranking(self) -> None:
        scores = pd.DataFrame(
            {"aic": [100, 90, 110], "bic": [105, 95, 115]},
            index=["A", "B", "C"],
        )
        result = ComparisonResult(models={}, criteria=["aic", "bic"], scores=scores)
        ranked = result.ranking("aic")
        assert ranked.index[0] == "B"

    def test_best_model(self) -> None:
        scores = pd.DataFrame(
            {"aic": [100, 90, 110]},
            index=["A", "B", "C"],
        )
        result = ComparisonResult(models={}, criteria=["aic"], scores=scores)
        assert result.best_model() == "B"

    def test_to_dataframe(self) -> None:
        scores = pd.DataFrame({"aic": [100]}, index=["A"])
        result = ComparisonResult(models={}, criteria=["aic"], scores=scores)
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_rmse(self) -> None:
        actuals = np.array([1.0, 2.0, 3.0])
        forecasts = np.array([1.1, 2.2, 2.8])
        result = ValidationResult(
            model_name="test", train_size=100, test_size=3,
            horizon=3, actuals=actuals, forecasts=forecasts,
        )
        assert result.rmse() > 0

    def test_mae(self) -> None:
        actuals = np.array([1.0, 2.0, 3.0])
        forecasts = np.array([1.0, 2.0, 3.0])
        result = ValidationResult(
            model_name="test", train_size=100, test_size=3,
            horizon=3, actuals=actuals, forecasts=forecasts,
        )
        assert result.mae() == 0.0

    def test_mape(self) -> None:
        actuals = np.array([100.0, 200.0])
        forecasts = np.array([110.0, 190.0])
        result = ValidationResult(
            model_name="test", train_size=100, test_size=2,
            horizon=2, actuals=actuals, forecasts=forecasts,
        )
        assert 0 < result.mape() < 100

    def test_coverage_no_ci(self) -> None:
        result = ValidationResult(
            model_name="test", train_size=100, test_size=3,
            horizon=3, actuals=np.array([1, 2, 3]),
            forecasts=np.array([1, 2, 3]),
        )
        with pytest.warns(UserWarning):
            cov = result.coverage()
        assert np.isnan(cov)


class TestCVResult:
    """Tests for CVResult."""

    def test_scores_df(self) -> None:
        result = CVResult(
            model_name="test", n_splits=2, horizon=3,
            fold_scores=[{"rmse": 0.1, "mae": 0.05}, {"rmse": 0.2, "mae": 0.1}],
            fold_forecasts=[], fold_actuals=[],
        )
        df = result.scores_df
        assert df.shape == (2, 2)

    def test_mean_scores(self) -> None:
        result = CVResult(
            model_name="test", n_splits=2, horizon=3,
            fold_scores=[{"rmse": 0.1}, {"rmse": 0.3}],
            fold_forecasts=[], fold_actuals=[],
        )
        means = result.mean_scores()
        assert abs(means["rmse"] - 0.2) < 1e-10

    def test_std_scores(self) -> None:
        result = CVResult(
            model_name="test", n_splits=2, horizon=3,
            fold_scores=[{"rmse": 0.1}, {"rmse": 0.3}],
            fold_forecasts=[], fold_actuals=[],
        )
        stds = result.std_scores()
        assert stds["rmse"] > 0


class TestChronoExperiment:
    """Tests for ChronoExperiment."""

    def test_init(self) -> None:
        data = pd.Series(np.random.randn(100))
        exp = ChronoExperiment(data, name="Test")
        assert exp.name == "Test"
        assert len(exp.fitted_models) == 0

    def test_fit_model_arima(self) -> None:
        np.random.seed(42)
        data = pd.Series(np.cumsum(np.random.randn(100)))
        exp = ChronoExperiment(data)
        result = exp.fit_model("ARIMA(1,1,0)", {"order": (1, 1, 0)})
        assert result is not None
        assert "ARIMA(1,1,0)" in exp.fitted_models

    def test_compare_models_no_fits_raises(self) -> None:
        data = pd.Series(np.random.randn(100))
        exp = ChronoExperiment(data)
        with pytest.raises(ValueError, match="No models fitted"):
            exp.compare_models()

    def test_save_master_report(self, tmp_path: Any) -> None:
        np.random.seed(42)
        data = pd.Series(np.cumsum(np.random.randn(100)))
        exp = ChronoExperiment(data, name="Test Report")
        exp.fit_model("ARIMA(1,0,0)", {"order": (1, 0, 0)})
        report_path = str(tmp_path / "report.html")
        exp.save_master_report(report_path)
        assert (tmp_path / "report.html").exists()
        content = (tmp_path / "report.html").read_text()
        assert "ChronoBox" in content
