"""Tests for automatic ETS model selection."""

from __future__ import annotations

import numpy as np
import pytest

from chronobox.selection.auto_ets import AutoETSResult, auto_ets


@pytest.fixture
def airline_passengers() -> np.ndarray:
    """Airline passengers data."""
    try:
        from chronobox.datasets import load_dataset

        df = load_dataset("airline")
        return df["passengers"].to_numpy(dtype=np.float64)
    except Exception:
        rng = np.random.default_rng(42)
        t = np.arange(144)
        trend = 100 + 2 * t
        seasonal = 20 * np.sin(2 * np.pi * t / 12)
        return trend + seasonal + rng.normal(0, 5, 144)


@pytest.fixture
def random_walk_series() -> np.ndarray:
    """Random walk (non-seasonal) series."""
    rng = np.random.default_rng(42)
    eps = rng.normal(0, 1, 200)
    y = np.cumsum(eps) + 100
    return y


@pytest.fixture
def simple_positive() -> np.ndarray:
    """Simple positive non-seasonal series."""
    rng = np.random.default_rng(42)
    return 100 + rng.normal(0, 5, 100)


class TestAutoETSBasic:
    """Basic auto_ets tests."""

    def test_auto_ets_returns_result(self, airline_passengers: np.ndarray) -> None:
        """auto_ets should return AutoETSResult."""
        result = auto_ets(airline_passengers, seasonal_period=12)
        assert isinstance(result, AutoETSResult)

    def test_auto_ets_attributes(self, airline_passengers: np.ndarray) -> None:
        """Result should have all expected attributes."""
        result = auto_ets(airline_passengers, seasonal_period=12)
        assert result.best_model is not None
        assert result.best_spec is not None
        assert len(result.best_spec) == 3
        assert result.n_models_tried > 0
        assert len(result.all_results) > 0

    def test_auto_ets_summary(self, airline_passengers: np.ndarray) -> None:
        """summary() should return formatted string."""
        result = auto_ets(airline_passengers, seasonal_period=12)
        s = result.summary()
        assert "Auto-ETS" in s
        assert "Best Model" in s
        assert "ETS" in s


class TestAutoETSSelection:
    """Tests for model selection behavior."""

    def test_auto_ets_selects_seasonal_for_airline(
        self, airline_passengers: np.ndarray
    ) -> None:
        """auto_ets should select a seasonal model for airline data."""
        result = auto_ets(airline_passengers, seasonal_period=12)
        e, t, s = result.best_spec
        assert s != "N", f"Expected seasonal model, got ETS({e},{t},{s})"

    def test_auto_ets_nonseasonal_for_random_walk(
        self, random_walk_series: np.ndarray
    ) -> None:
        """auto_ets should select non-seasonal model for random walk."""
        result = auto_ets(random_walk_series, seasonal_period=None)
        e, t, s = result.best_spec
        assert s == "N", f"Expected non-seasonal model, got ETS({e},{t},{s})"

    def test_auto_ets_tries_multiple_models(
        self, airline_passengers: np.ndarray
    ) -> None:
        """auto_ets should try multiple models."""
        result = auto_ets(airline_passengers, seasonal_period=12)
        assert result.n_models_tried >= 10, (
            f"Only tried {result.n_models_tried} models"
        )

    def test_auto_ets_aicc_sorted(self, airline_passengers: np.ndarray) -> None:
        """Results should be sorted by AICc."""
        result = auto_ets(airline_passengers, seasonal_period=12)
        aiccs = [aicc for _, aicc in result.all_results]
        assert aiccs == sorted(aiccs), "Results not sorted by AICc"

    def test_auto_ets_best_has_lowest_aicc(
        self, airline_passengers: np.ndarray
    ) -> None:
        """Best model should have the lowest AICc."""
        result = auto_ets(airline_passengers, seasonal_period=12)
        best_aicc = result.best_model.aicc
        all_aiccs = [aicc for _, aicc in result.all_results]
        assert best_aicc == pytest.approx(min(all_aiccs), abs=0.01)


class TestAutoETSForecast:
    """Tests for forecasting from auto_ets."""

    def test_forecast_from_best(self, airline_passengers: np.ndarray) -> None:
        """Should be able to forecast from the best model."""
        result = auto_ets(airline_passengers, seasonal_period=12)
        fc = result.forecast(steps=12)
        assert len(fc) == 12
        assert np.all(np.isfinite(fc))

    def test_forecast_positive(self, airline_passengers: np.ndarray) -> None:
        """Forecasts for positive data should be positive."""
        result = auto_ets(airline_passengers, seasonal_period=12)
        fc = result.forecast(steps=24)
        # Most forecasts should be positive (allow some tolerance for additive models)
        assert np.sum(fc > 0) >= len(fc) * 0.9


class TestAutoETSOptions:
    """Tests for auto_ets options."""

    def test_restrict_error_types(self, simple_positive: np.ndarray) -> None:
        """Restricting to additive error should only try A models."""
        result = auto_ets(simple_positive, error_types=["A"])
        e, t, s = result.best_spec
        assert e == "A"

    def test_restrict_trend_types(self, simple_positive: np.ndarray) -> None:
        """Restricting trend types should limit candidates."""
        result = auto_ets(simple_positive, trend_types=["N", "A"])
        e, t, s = result.best_spec
        assert t in ("N", "A")

    def test_damped_only(self, simple_positive: np.ndarray) -> None:
        """damped=True should only try damped trends."""
        result = auto_ets(simple_positive, damped=True)
        e, t, s = result.best_spec
        assert t in ("N", "Ad", "Md"), f"Expected damped, got {t}"

    def test_aic_criterion(self, airline_passengers: np.ndarray) -> None:
        """Should work with AIC criterion."""
        result = auto_ets(
            airline_passengers, seasonal_period=12, information_criterion="aic"
        )
        assert isinstance(result, AutoETSResult)

    def test_bic_criterion(self, airline_passengers: np.ndarray) -> None:
        """Should work with BIC criterion."""
        result = auto_ets(
            airline_passengers, seasonal_period=12, information_criterion="bic"
        )
        assert isinstance(result, AutoETSResult)


class TestAutoETSEdgeCases:
    """Tests for edge cases."""

    def test_negative_data_no_multiplicative(self) -> None:
        """Negative data should exclude multiplicative models."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 5, 100)  # can be negative
        result = auto_ets(y)
        e, t, s = result.best_spec
        assert e == "A", f"Expected additive error for data with negatives, got {e}"

    def test_short_series(self) -> None:
        """Should work with short (non-seasonal) series."""
        rng = np.random.default_rng(42)
        y = 100 + rng.normal(0, 2, 30)
        result = auto_ets(y)
        assert isinstance(result, AutoETSResult)

    def test_verbose_mode(
        self,
        simple_positive: np.ndarray,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Verbose mode should print progress."""
        auto_ets(simple_positive, verbose=True)
        captured = capsys.readouterr()
        assert "ETS" in captured.out
