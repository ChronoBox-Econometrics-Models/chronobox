"""Tests for Classical Decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from chronobox.decomposition.classical import ClassicalDecomposition
from chronobox.decomposition.stl import DecompositionResult


@pytest.fixture
def additive_series() -> np.ndarray:
    """Series with known additive components."""
    t_len = 120
    m = 12
    t = np.arange(t_len)
    trend = 100 + 0.5 * t
    seasonal_pattern = np.array(
        [10, 8, 4, -2, -6, -10, -10, -6, -2, 4, 8, 10], dtype=np.float64
    )
    seasonal_pattern -= np.mean(seasonal_pattern)  # ensure sum=0
    seasonal = np.tile(seasonal_pattern, t_len // m + 1)[:t_len]
    return trend + seasonal  # no remainder for exact test


@pytest.fixture
def multiplicative_series() -> np.ndarray:
    """Series with known multiplicative components."""
    t_len = 120
    m = 12
    t = np.arange(t_len)
    trend = 100 + 0.5 * t
    seasonal_pattern = np.array(
        [1.2, 1.15, 1.05, 0.95, 0.85, 0.8, 0.8, 0.85, 0.95, 1.05, 1.15, 1.2],
        dtype=np.float64,
    )
    seasonal_pattern *= m / np.sum(seasonal_pattern)  # normalize to mean=1
    seasonal = np.tile(seasonal_pattern, t_len // m + 1)[:t_len]
    return trend * seasonal  # no remainder for exact test


class TestClassicalAdditive:
    """Tests for additive classical decomposition."""

    def test_additive_components_sum(self, additive_series: np.ndarray) -> None:
        """T + S + R should equal y for additive decomposition."""
        cd = ClassicalDecomposition(period=12, model="additive")
        result = cd.fit(additive_series)

        # Where trend is available (not NaN at edges)
        valid = ~np.isnan(result.trend)
        reconstructed = (
            result.trend[valid] + result.seasonal[valid] + result.remainder[valid]
        )
        np.testing.assert_allclose(
            reconstructed,
            result.observed[valid],
            atol=1e-10,
            err_msg="T + S + R != y for additive decomposition",
        )

    def test_additive_returns_result(self, additive_series: np.ndarray) -> None:
        """fit() should return DecompositionResult."""
        cd = ClassicalDecomposition(period=12, model="additive")
        result = cd.fit(additive_series)
        assert isinstance(result, DecompositionResult)
        assert result.model == "additive"

    def test_additive_seasonal_sum_zero(self, additive_series: np.ndarray) -> None:
        """Seasonal component should sum to ~0 over one period."""
        cd = ClassicalDecomposition(period=12, model="additive")
        result = cd.fit(additive_series)
        s_period = result.seasonal[:12]
        assert abs(np.sum(s_period)) < 1e-10, (
            f"Seasonal sum = {np.sum(s_period)}, expected ~0"
        )


class TestClassicalMultiplicative:
    """Tests for multiplicative classical decomposition."""

    def test_multiplicative_components_product(
        self, multiplicative_series: np.ndarray
    ) -> None:
        """T * S * R should equal y for multiplicative decomposition."""
        cd = ClassicalDecomposition(period=12, model="multiplicative")
        result = cd.fit(multiplicative_series)

        valid = ~np.isnan(result.trend) & ~np.isnan(result.remainder)
        reconstructed = (
            result.trend[valid] * result.seasonal[valid] * result.remainder[valid]
        )
        np.testing.assert_allclose(
            reconstructed,
            result.observed[valid],
            atol=1e-10,
            err_msg="T * S * R != y for multiplicative decomposition",
        )

    def test_multiplicative_returns_result(
        self, multiplicative_series: np.ndarray
    ) -> None:
        cd = ClassicalDecomposition(period=12, model="multiplicative")
        result = cd.fit(multiplicative_series)
        assert isinstance(result, DecompositionResult)
        assert result.model == "multiplicative"

    def test_multiplicative_seasonal_mean_one(
        self, multiplicative_series: np.ndarray
    ) -> None:
        """Seasonal indices should average to ~1 for multiplicative."""
        cd = ClassicalDecomposition(period=12, model="multiplicative")
        result = cd.fit(multiplicative_series)
        s_period = result.seasonal[:12]
        assert abs(np.mean(s_period) - 1.0) < 0.01, (
            f"Seasonal mean = {np.mean(s_period)}, expected ~1"
        )


class TestClassicalEdgeCases:
    """Tests for edge cases."""

    def test_trend_nan_at_edges(self) -> None:
        """Trend should have NaN at edges (cannot compute MA)."""
        rng = np.random.default_rng(42)
        y = 100 + rng.normal(0, 1, 48)
        cd = ClassicalDecomposition(period=12, model="additive")
        result = cd.fit(y)
        # First and last m/2 values should be NaN
        assert np.isnan(result.trend[0])
        assert np.isnan(result.trend[-1])

    def test_invalid_model(self) -> None:
        with pytest.raises(ValueError, match="model"):
            ClassicalDecomposition(period=12, model="invalid")  # type: ignore[arg-type]

    def test_period_too_small(self) -> None:
        with pytest.raises(ValueError, match="period"):
            ClassicalDecomposition(period=1)

    def test_too_few_observations(self) -> None:
        cd = ClassicalDecomposition(period=12)
        with pytest.raises(ValueError, match="seasonal periods"):
            cd.fit(np.ones(10))

    def test_negative_data_multiplicative(self) -> None:
        cd = ClassicalDecomposition(period=4, model="multiplicative")
        y = np.array([-1, 2, 3, 4, 5, 6, 7, 8])
        with pytest.raises(ValueError, match="positive"):
            cd.fit(y)
