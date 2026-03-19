"""Tests for time series transforms."""

from __future__ import annotations

import numpy as np

from chronobox.core.transforms import (
    boxcox,
    difference,
    inv_boxcox,
    standardize,
    undifference,
)


class TestDiffUndiff:
    """Test difference/undifference roundtrip."""

    def test_roundtrip_d1(self) -> None:
        y = np.array([10.0, 13.0, 17.0, 22.0, 28.0])
        dy = difference(y, d=1)
        y_rec = undifference(dy, y[:1])
        np.testing.assert_array_almost_equal(y_rec, y)


class TestBoxCox:
    """Test Box-Cox transform."""

    def test_roundtrip(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        for lam in [0.0, 0.5, 1.0, 2.0]:
            z, _ = boxcox(y, lam=lam)
            y_rec = inv_boxcox(z, lam)
            np.testing.assert_array_almost_equal(y_rec, y, decimal=10)


class TestStandardize:
    """Test standardize."""

    def test_mean_zero(self) -> None:
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        z, mean, std = standardize(y)
        np.testing.assert_almost_equal(np.mean(z), 0.0, decimal=10)

    def test_unit_variance(self) -> None:
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        z, mean, std = standardize(y)
        np.testing.assert_almost_equal(np.std(z, ddof=1), 1.0, decimal=10)
