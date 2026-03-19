"""Tests for STL decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from chronobox.decomposition.stl import STL, DecompositionResult


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
def synthetic_additive() -> np.ndarray:
    """Synthetic additive series with known components."""
    rng = np.random.default_rng(42)
    t_len = 120
    t = np.arange(t_len)
    trend = 100 + 0.5 * t
    seasonal_pattern = np.array(
        [10, 8, 4, -2, -6, -10, -10, -6, -2, 4, 8, 10], dtype=np.float64
    )
    seasonal = np.tile(seasonal_pattern, t_len // 12 + 1)[:t_len]
    noise = rng.normal(0, 1, t_len)
    return trend + seasonal + noise


class TestSTLBasic:
    """Basic STL tests."""

    def test_stl_fit_returns_result(self, synthetic_additive: np.ndarray) -> None:
        """fit() should return a DecompositionResult."""
        stl = STL(period=12)
        result = stl.fit(synthetic_additive)
        assert isinstance(result, DecompositionResult)
        assert result.model == "additive"
        assert result.period == 12

    def test_stl_components_sum_to_original(
        self, synthetic_additive: np.ndarray
    ) -> None:
        """trend + seasonal + remainder should equal original."""
        stl = STL(period=12)
        result = stl.fit(synthetic_additive)
        reconstructed = result.trend + result.seasonal + result.remainder
        np.testing.assert_allclose(reconstructed, result.observed, atol=1e-10)

    def test_stl_component_lengths(self, synthetic_additive: np.ndarray) -> None:
        """All components should have same length as original."""
        stl = STL(period=12)
        result = stl.fit(synthetic_additive)
        t_len = len(synthetic_additive)
        assert len(result.trend) == t_len
        assert len(result.seasonal) == t_len
        assert len(result.remainder) == t_len


class TestSTLAirline:
    """STL tests on airline data."""

    def test_stl_airline(self, airline_passengers: np.ndarray) -> None:
        """STL on airline with period=12 should decompose successfully."""
        stl = STL(period=12, seasonal=7)
        result = stl.fit(airline_passengers)
        # trend + seasonal + remainder == original
        reconstructed = result.trend + result.seasonal + result.remainder
        np.testing.assert_allclose(reconstructed, result.observed, atol=1e-10)

    def test_stl_airline_trend_increasing(
        self, airline_passengers: np.ndarray
    ) -> None:
        """Trend of airline data should be generally increasing."""
        stl = STL(period=12)
        result = stl.fit(airline_passengers)
        # Compare first quarter mean to last quarter mean
        first_q = np.mean(result.trend[:12])
        last_q = np.mean(result.trend[-12:])
        assert last_q > first_q, "Trend should be increasing for airline data"

    def test_stl_seasonal_pattern(self, airline_passengers: np.ndarray) -> None:
        """Seasonal component should show the expected pattern."""
        stl = STL(period=12)
        result = stl.fit(airline_passengers)
        # Summer months (Jul=6, Aug=7) should have higher seasonal than
        # winter months (Nov=10, Dec=11) - using 0-based indices
        seasonal = result.seasonal
        # Take middle year to avoid edge effects
        mid = 72  # start of 7th year
        summer = seasonal[mid + 6 : mid + 8]
        winter = seasonal[mid + 10 : mid + 12]
        assert np.mean(summer) > np.mean(winter)


class TestSTLSeasonalSum:
    """Tests for seasonal sum property."""

    def test_seasonal_sum_near_zero(self, synthetic_additive: np.ndarray) -> None:
        """Sum of seasonal component over each period should be ~0 (additive)."""
        stl = STL(period=12)
        result = stl.fit(synthetic_additive)
        t_len = len(synthetic_additive)
        m = 12
        # Check sum for each complete year (avoiding edges)
        for start in range(12, t_len - 12, m):
            s_sum = np.sum(result.seasonal[start : start + m])
            assert abs(s_sum) < 5.0, (
                f"Seasonal sum at t={start} is {s_sum}, expected ~0"
            )


class TestSTLRobust:
    """Tests for robust STL."""

    def test_stl_robust_outlier_handling(self) -> None:
        """Robust STL should not be heavily affected by a single outlier."""
        rng = np.random.default_rng(42)
        t_len = 120
        t = np.arange(t_len)
        trend = 100 + 0.5 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        y_clean = trend + seasonal + rng.normal(0, 1, t_len)

        # Add outlier
        y_outlier = y_clean.copy()
        y_outlier[60] += 100  # massive outlier

        stl_clean = STL(period=12, robust=True, n_outer=5)
        stl_outlier = STL(period=12, robust=True, n_outer=5)

        result_clean = stl_clean.fit(y_clean)
        result_outlier = stl_outlier.fit(y_outlier)

        # Trends should be similar except near the outlier
        mask = np.ones(t_len, dtype=bool)
        mask[55:66] = False  # exclude region near outlier
        trend_diff = np.abs(result_clean.trend[mask] - result_outlier.trend[mask])
        assert np.mean(trend_diff) < 5.0, (
            f"Robust STL trends differ too much: mean diff={np.mean(trend_diff)}"
        )

    def test_stl_robust_returns_weights(self) -> None:
        """Robust STL should return robustness weights."""
        rng = np.random.default_rng(42)
        t_len = 60
        t = np.arange(t_len)
        y = 100 + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, t_len)
        stl = STL(period=12, robust=True, n_outer=5)
        result = stl.fit(y)
        assert result.weights is not None
        assert len(result.weights) == t_len


class TestSTLSummary:
    """Tests for summary output."""

    def test_summary_string(self, synthetic_additive: np.ndarray) -> None:
        """summary() should return a formatted string."""
        stl = STL(period=12)
        result = stl.fit(synthetic_additive)
        s = result.summary()
        assert "Decomposition" in s
        assert "Trend" in s
        assert "Seasonal" in s


class TestSTLInvalidInputs:
    """Tests for invalid inputs."""

    def test_period_too_small(self) -> None:
        with pytest.raises(ValueError, match="period"):
            STL(period=1)

    def test_series_too_short(self) -> None:
        stl = STL(period=12)
        y = np.ones(10)
        with pytest.raises(ValueError, match="seasonal periods"):
            stl.fit(y)
