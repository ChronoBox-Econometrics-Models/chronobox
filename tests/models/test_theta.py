"""Tests for Theta Method."""

from __future__ import annotations

import numpy as np
import pytest

from chronobox.models.theta import ThetaMethod, ThetaResults, _ses_fit


@pytest.fixture
def trending_series() -> np.ndarray:
    """Series with trend and noise."""
    rng = np.random.default_rng(42)
    t = np.arange(100)
    return 50 + 0.5 * t + rng.normal(0, 3, 100)


@pytest.fixture
def positive_series() -> np.ndarray:
    """Positive series."""
    rng = np.random.default_rng(42)
    return 100 + rng.normal(0, 5, 100)


class TestThetaFit:
    """Tests for Theta method fitting."""

    def test_theta_fit_returns_results(self, trending_series: np.ndarray) -> None:
        """fit() should return ThetaResults."""
        model = ThetaMethod()
        results = model.fit(trending_series)
        assert isinstance(results, ThetaResults)

    def test_theta_attributes(self, trending_series: np.ndarray) -> None:
        """Results should have all expected attributes."""
        model = ThetaMethod()
        results = model.fit(trending_series)
        assert hasattr(results, "ses_level")
        assert hasattr(results, "drift")
        assert hasattr(results, "alpha")
        assert hasattr(results, "slope")
        assert hasattr(results, "intercept")
        assert results.nobs == len(trending_series)

    def test_theta_summary(self, trending_series: np.ndarray) -> None:
        """summary() should return formatted string."""
        model = ThetaMethod()
        results = model.fit(trending_series)
        s = results.summary()
        assert "Theta" in s
        assert "alpha" in s
        assert "Drift" in s


class TestThetaVsSESDrift:
    """Tests for Theta vs SES+drift equivalence."""

    def test_theta_equivalence_ses_drift(self, trending_series: np.ndarray) -> None:
        """Theta forecast should be equivalent to SES + drift.

        Hyndman & Billah (2003): The Theta method with theta=2 is
        equivalent to SES with drift = slope/2.
        """
        model = ThetaMethod(theta=2.0)
        results = model.fit(trending_series)

        # SES with drift forecast
        fc_theta = results.forecast(steps=12)

        # Manual SES + drift
        alpha_opt, l_t, _ = _ses_fit(trending_series)
        slope = results.slope
        drift = slope / 2.0

        fc_manual = np.array([l_t + h * drift for h in range(1, 13)])

        # They should be approximately equal
        # (may differ slightly due to optimization paths)
        np.testing.assert_allclose(fc_theta, fc_manual, rtol=0.01)


class TestThetaForecast:
    """Tests for Theta method forecasting."""

    def test_forecast_length(self, trending_series: np.ndarray) -> None:
        """forecast() should return correct length."""
        model = ThetaMethod()
        results = model.fit(trending_series)
        for steps in [1, 5, 12, 24]:
            fc = results.forecast(steps=steps)
            assert len(fc) == steps

    def test_forecast_finite(self, trending_series: np.ndarray) -> None:
        """Forecasts should be finite."""
        model = ThetaMethod()
        results = model.fit(trending_series)
        fc = results.forecast(steps=24)
        assert np.all(np.isfinite(fc))

    def test_forecast_positive_for_positive_data(
        self, positive_series: np.ndarray
    ) -> None:
        """Forecast should be positive for positive data (at reasonable horizons)."""
        model = ThetaMethod()
        results = model.fit(positive_series)
        fc = results.forecast(steps=12)
        assert np.all(fc > 0), f"Forecast contains non-positive values: {fc}"

    def test_forecast_trending(self, trending_series: np.ndarray) -> None:
        """Forecast of trending series should continue the trend."""
        model = ThetaMethod()
        results = model.fit(trending_series)
        fc = results.forecast(steps=12)
        # Forecast should be monotonically increasing (positive drift)
        assert results.drift > 0
        diffs = np.diff(fc)
        assert np.all(diffs > 0), "Forecast should be increasing for positive drift"


class TestSESFit:
    """Tests for the internal SES fitting."""

    def test_ses_alpha_range(self) -> None:
        """Optimized alpha should be in (0, 1)."""
        rng = np.random.default_rng(42)
        y = rng.normal(100, 5, 100)
        alpha, _l_t, _fitted = _ses_fit(y)
        assert 0 < alpha < 1

    def test_ses_fitted_length(self) -> None:
        """Fitted values should match input length."""
        rng = np.random.default_rng(42)
        y = rng.normal(100, 5, 50)
        _, _, fitted = _ses_fit(y)
        assert len(fitted) == len(y)

    def test_ses_fixed_alpha(self) -> None:
        """SES with fixed alpha should use that alpha."""
        rng = np.random.default_rng(42)
        y = rng.normal(100, 5, 50)
        alpha, _, _ = _ses_fit(y, alpha=0.5)
        assert alpha == 0.5


class TestThetaInvalidInputs:
    """Tests for invalid inputs."""

    def test_negative_theta(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ThetaMethod(theta=-1.0)

    def test_empty_series(self) -> None:
        model = ThetaMethod()
        with pytest.raises(ValueError):
            model.fit(np.array([]))
