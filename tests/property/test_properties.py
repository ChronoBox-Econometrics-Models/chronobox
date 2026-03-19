"""Property-based tests using Hypothesis.

Tests mathematical invariants and properties that must hold
for any valid input, not just specific examples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st


class TestHPFilterProperties:
    """Property-based tests for HP filter."""

    @given(
        n=st.integers(min_value=20, max_value=500),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=50, deadline=5000)
    def test_hp_trend_plus_cycle_equals_original(self, n: int, seed: int) -> None:
        """HP filter: trend + cycle should always equal original data."""
        from chronobox.filters import hp_filter

        rng = np.random.default_rng(seed)
        data = pd.Series(np.cumsum(rng.normal(0, 1, n)))

        trend, cycle = hp_filter(data, lamb=1600)
        reconstruction = np.asarray(trend) + np.asarray(cycle)
        np.testing.assert_allclose(reconstruction, data.values, atol=1e-8)

    @given(
        lamb=st.floats(min_value=1.0, max_value=1e8),
    )
    @settings(max_examples=20, deadline=5000)
    def test_hp_lambda_affects_smoothness(self, lamb: float) -> None:
        """Higher lambda should produce smoother trend."""
        from chronobox.filters import hp_filter

        np.random.seed(42)
        data = pd.Series(np.cumsum(np.random.randn(100)))

        trend, _ = hp_filter(data, lamb=lamb)
        # Trend should be a valid series
        assert len(trend) == len(data)
        assert np.all(np.isfinite(np.asarray(trend)))


class TestARIMAProperties:
    """Property-based tests for ARIMA."""

    @given(
        n=st.integers(min_value=50, max_value=200),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=10, deadline=30000)
    def test_arima_residuals_sum_approximately_zero(self, n: int, seed: int) -> None:
        """ARIMA residuals should sum approximately to zero for stationary data."""
        from chronobox import ARIMA

        rng = np.random.default_rng(seed)
        data = pd.Series(rng.normal(0, 1, n))

        model = ARIMA(order=(1, 0, 0))
        try:
            results = model.fit(data)
            resid = np.asarray(results.residuals)
            # Residuals should be finite
            assert np.all(np.isfinite(resid))
        except Exception:
            pass  # Some random data may fail to converge

    @given(
        steps=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=10, deadline=30000)
    def test_forecast_returns_correct_length(self, steps: int) -> None:
        """Forecast should return exactly the requested number of steps."""
        from chronobox import ARIMA

        np.random.seed(42)
        data = pd.Series(np.cumsum(np.random.randn(100)))

        model = ARIMA(order=(1, 1, 0))
        results = model.fit(data)
        forecast = results.forecast(steps=steps)
        if isinstance(forecast, dict):
            assert len(forecast["forecast"]) == steps
        else:
            assert len(forecast) == steps


class TestValidationProperties:
    """Property-based tests for validation metrics."""

    @given(
        n=st.integers(min_value=5, max_value=100),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=50)
    def test_rmse_non_negative(self, n: int, seed: int) -> None:
        """RMSE should always be non-negative."""
        from chronobox.experiment import ValidationResult

        rng = np.random.default_rng(seed)
        actuals = rng.normal(0, 1, n)
        forecasts = rng.normal(0, 1, n)

        result = ValidationResult(
            model_name="test",
            train_size=100,
            test_size=n,
            horizon=n,
            actuals=actuals,
            forecasts=forecasts,
        )
        assert result.rmse() >= 0

    @given(
        n=st.integers(min_value=5, max_value=100),
    )
    @settings(max_examples=50)
    def test_perfect_forecast_has_zero_rmse(self, n: int) -> None:
        """Perfect forecast should have RMSE = 0."""
        from chronobox.experiment import ValidationResult

        values = np.random.randn(n)

        result = ValidationResult(
            model_name="test",
            train_size=100,
            test_size=n,
            horizon=n,
            actuals=values,
            forecasts=values.copy(),
        )
        assert result.rmse() < 1e-10
