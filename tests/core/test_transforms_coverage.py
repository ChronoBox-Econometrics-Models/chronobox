"""Additional transforms tests for coverage improvement."""

from __future__ import annotations

import numpy as np

from chronobox.core.transforms import (
    boxcox,
    detrend,
    exp_transform,
    inv_boxcox,
    log_transform,
    seasonal_difference,
    seasonal_undifference,
    standardize,
    undifference,
    unstandardize,
)


class TestTransformsCoverage:
    def test_seasonal_undifference(self) -> None:
        y = np.array([10.0, 20.0, 30.0, 40.0, 15.0, 25.0, 35.0, 45.0])
        dy = seasonal_difference(y, s=4)
        reconstructed = seasonal_undifference(dy, y[:4])
        np.testing.assert_allclose(reconstructed, y)

    def test_log_exp_roundtrip(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 10.0])
        ly = log_transform(y)
        recovered = exp_transform(ly)
        np.testing.assert_allclose(recovered, y)

    def test_boxcox_auto_lambda(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        transformed, lam = boxcox(y)
        assert len(transformed) == len(y)
        assert isinstance(lam, float)

    def test_boxcox_lambda_zero(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        transformed, lam = boxcox(y, lam=0.0)
        np.testing.assert_allclose(transformed, np.log(y))
        assert lam == 0.0

    def test_boxcox_specific_lambda(self) -> None:
        y = np.array([1.0, 4.0, 9.0])
        transformed, lam = boxcox(y, lam=0.5)
        assert lam == 0.5

    def test_inv_boxcox_lambda_zero(self) -> None:
        z = np.array([0.0, 0.693, 1.099])
        result = inv_boxcox(z, lam=0.0)
        np.testing.assert_allclose(result, np.exp(z))

    def test_inv_boxcox_roundtrip(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        lam = 0.5
        transformed, _ = boxcox(y, lam=lam)
        recovered = inv_boxcox(transformed, lam)
        np.testing.assert_allclose(recovered, y, atol=1e-10)

    def test_standardize_constant_series(self) -> None:
        y = np.array([5.0, 5.0, 5.0, 5.0])
        z, mean, std = standardize(y)
        assert mean == 5.0
        assert std == 1.0  # fallback for zero std
        np.testing.assert_allclose(z, np.zeros(4))

    def test_unstandardize(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        z, mean, std = standardize(y)
        recovered = unstandardize(z, mean, std)
        np.testing.assert_allclose(recovered, y)

    def test_undifference(self) -> None:
        y = np.array([10.0, 12.0, 15.0, 20.0])
        dy = np.diff(y)
        recovered = undifference(dy, y[:1])
        np.testing.assert_allclose(recovered, y)

    def test_detrend(self) -> None:
        t = np.arange(50, dtype=np.float64)
        y = 3.0 * t + 10.0 + np.random.default_rng(42).normal(0, 0.1, 50)
        detrended = detrend(y, order=1)
        assert abs(np.mean(detrended)) < 1.0

    def test_detrend_quadratic(self) -> None:
        t = np.arange(50, dtype=np.float64)
        y = 0.1 * t**2 + t + 5.0
        detrended = detrend(y, order=2)
        assert np.std(detrended) < 1e-10
