"""Tests for LagPolynomial."""

from __future__ import annotations

import numpy as np

from chronobox.core.lag_polynomial import LagPolynomial


class TestRoots:
    """Test polynomial root computation."""

    def test_ar1_root(self) -> None:
        """LagPolynomial([1, -0.5]) has root at 2.0."""
        poly = LagPolynomial([1, -0.5])
        roots = poly.roots()
        assert len(roots) == 1
        np.testing.assert_almost_equal(np.abs(roots[0]), 2.0)

    def test_order_zero(self) -> None:
        poly = LagPolynomial([1.0])
        assert poly.order == 0
        assert len(poly.roots()) == 0


class TestStationary:
    """Test stationarity checks."""

    def test_stationary_ar1(self) -> None:
        """AR(1) with phi=0.5 is stationary."""
        poly = LagPolynomial([1, -0.5])
        assert poly.is_stationary()

    def test_nonstationary_ar1(self) -> None:
        """AR(1) with phi=1.2 is not stationary."""
        poly = LagPolynomial([1, -1.2])
        assert not poly.is_stationary()

    def test_unit_root(self) -> None:
        """AR(1) with phi=1.0 is not stationary (unit root)."""
        poly = LagPolynomial([1, -1.0])
        assert not poly.is_stationary()


class TestMultiply:
    """Test polynomial multiplication."""

    def test_ar1_times_ma1(self) -> None:
        """(1 - 0.5L)(1 + 0.3L) = 1 - 0.2L - 0.15L^2."""
        ar = LagPolynomial([1, -0.5])
        ma = LagPolynomial([1, 0.3])
        product = ar.multiply(ma)
        expected = np.convolve([1, -0.5], [1, 0.3])
        np.testing.assert_array_almost_equal(product.coeffs, expected)


class TestCompanion:
    """Test companion matrix."""

    def test_ar2_companion(self) -> None:
        """Companion matrix of AR(2) with phi=[0.5, 0.3]."""
        poly = LagPolynomial([1, -0.5, -0.3])
        result = poly.to_companion()
        expected = np.array([[0.5, 0.3], [1.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ar1_companion(self) -> None:
        """Companion matrix of AR(1) with phi=0.8."""
        poly = LagPolynomial([1, -0.8])
        result = poly.to_companion()
        expected = np.array([[0.8]])
        np.testing.assert_array_almost_equal(result, expected)
