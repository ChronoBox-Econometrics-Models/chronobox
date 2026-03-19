"""Tests for Holt-Winters Exponential Smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from chronobox.models.holtwinters import HoltWinters, HoltWintersResults


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
def seasonal_series() -> np.ndarray:
    """Synthetic seasonal series."""
    rng = np.random.default_rng(42)
    t = np.arange(120)
    trend = 100 + 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    return trend + seasonal + rng.normal(0, 2, 120)


class TestHoltWintersAdditive:
    """Tests for additive Holt-Winters."""

    def test_hw_additive_airline(self, airline_passengers: np.ndarray) -> None:
        """Holt-Winters additive should fit airline data and capture seasonality."""
        model = HoltWinters(seasonal="additive", seasonal_period=12)
        results = model.fit(airline_passengers)
        assert isinstance(results, HoltWintersResults)
        assert results.method == "additive"
        assert 0 < results.alpha < 1
        assert 0 < results.beta < 1
        assert 0 < results.gamma < 1
        assert results.rmse > 0

    def test_hw_additive_fit_reduces_sse(self, seasonal_series: np.ndarray) -> None:
        """Optimized SSE should be less than unoptimized."""
        model = HoltWinters(seasonal="additive", seasonal_period=12)
        res_opt = model.fit(seasonal_series, optimized=True)
        res_unopt = model.fit(seasonal_series, optimized=False)
        assert res_opt.sse <= res_unopt.sse + 1.0

    def test_hw_additive_summary(self, seasonal_series: np.ndarray) -> None:
        """summary() should return a formatted string."""
        model = HoltWinters(seasonal="additive", seasonal_period=12)
        results = model.fit(seasonal_series)
        s = results.summary()
        assert "Holt-Winters" in s
        assert "alpha" in s


class TestHoltWintersMultiplicative:
    """Tests for multiplicative Holt-Winters."""

    def test_hw_multiplicative_airline(self, airline_passengers: np.ndarray) -> None:
        """Holt-Winters multiplicative should fit airline data."""
        model = HoltWinters(seasonal="multiplicative", seasonal_period=12)
        results = model.fit(airline_passengers)
        assert isinstance(results, HoltWintersResults)
        assert results.method == "multiplicative"

    def test_hw_multiplicative_better_rmse_airline(
        self, airline_passengers: np.ndarray
    ) -> None:
        """Multiplicative should have lower RMSE than additive on airline data."""
        hw_add = HoltWinters(seasonal="additive", seasonal_period=12)
        hw_mul = HoltWinters(seasonal="multiplicative", seasonal_period=12)
        res_add = hw_add.fit(airline_passengers)
        res_mul = hw_mul.fit(airline_passengers)
        # Multiplicative should generally be better for airline data
        # (increasing variance with level)
        assert res_mul.rmse <= res_add.rmse * 1.1, (
            f"Mult RMSE={res_mul.rmse}, Add RMSE={res_add.rmse}"
        )


class TestHoltWintersForecast:
    """Tests for Holt-Winters forecasting."""

    def test_forecast_length(self, seasonal_series: np.ndarray) -> None:
        """forecast() should return correct number of steps."""
        model = HoltWinters(seasonal="additive", seasonal_period=12)
        results = model.fit(seasonal_series)
        for steps in [1, 6, 12, 24]:
            fc = results.forecast(steps=steps)
            assert len(fc) == steps

    def test_forecast_finite(self, airline_passengers: np.ndarray) -> None:
        """Forecasts should be finite."""
        model = HoltWinters(seasonal="multiplicative", seasonal_period=12)
        results = model.fit(airline_passengers)
        fc = results.forecast(steps=24)
        assert np.all(np.isfinite(fc))

    def test_forecast_seasonal_pattern(self, airline_passengers: np.ndarray) -> None:
        """Forecast 12 months should repeat seasonal pattern."""
        model = HoltWinters(seasonal="multiplicative", seasonal_period=12)
        results = model.fit(airline_passengers)
        fc = results.forecast(steps=24)
        # Summer months (Jul=7, Aug=8) should be higher than winter (Nov=11, Dec=12)
        # In forecast indices: 6,7 vs 10,11 (first year)
        summer = fc[6:8]
        winter = fc[10:12]
        assert np.mean(summer) > np.mean(winter), (
            f"Summer mean={np.mean(summer)}, Winter mean={np.mean(winter)}"
        )

    def test_forecast_positive_multiplicative(
        self, airline_passengers: np.ndarray
    ) -> None:
        """Multiplicative forecast should be positive for positive data."""
        model = HoltWinters(seasonal="multiplicative", seasonal_period=12)
        results = model.fit(airline_passengers)
        fc = results.forecast(steps=24)
        assert np.all(fc > 0)


class TestHoltWintersInvalidInputs:
    """Tests for invalid input handling."""

    def test_invalid_seasonal(self) -> None:
        with pytest.raises(ValueError, match="seasonal"):
            HoltWinters(seasonal="invalid")  # type: ignore[arg-type]

    def test_small_seasonal_period(self) -> None:
        with pytest.raises(ValueError, match="seasonal_period"):
            HoltWinters(seasonal="additive", seasonal_period=1)

    def test_too_few_observations(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model = HoltWinters(seasonal="additive", seasonal_period=12)
        with pytest.raises(ValueError, match="seasonal cycles"):
            model.fit(y)

    def test_negative_data_multiplicative(self) -> None:
        y = np.concatenate([np.arange(-12, 12, dtype=float)] * 3)
        model = HoltWinters(seasonal="multiplicative", seasonal_period=12)
        with pytest.raises(ValueError, match="positive"):
            model.fit(y)
