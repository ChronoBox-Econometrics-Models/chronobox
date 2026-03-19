"""Tests for chronobox utility modules."""

from __future__ import annotations

import numpy as np
import pytest

from chronobox.utils.array_ops import (
    convolve_polynomials,
    expand_seasonal_polynomial,
    lag_matrix,
)
from chronobox.utils.validation import (
    validate_endog,
    validate_order,
    validate_seasonal_order,
)


class TestLagMatrix:
    def test_basic(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lag_matrix(y, 2)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result[:, 0], [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(result[:, 1], [1.0, 2.0, 3.0])

    def test_single_lag(self) -> None:
        y = np.array([10.0, 20.0, 30.0])
        result = lag_matrix(y, 1)
        assert result.shape == (2, 1)
        np.testing.assert_array_equal(result[:, 0], [10.0, 20.0])


class TestConvolvePolynomials:
    def test_basic(self) -> None:
        a = np.array([1.0, 0.5])
        b = np.array([1.0, -0.3])
        result = convolve_polynomials(a, b)
        expected = np.convolve(a, b)
        np.testing.assert_allclose(result, expected)

    def test_identity(self) -> None:
        a = np.array([1.0, 0.5, -0.2])
        b = np.array([1.0])
        result = convolve_polynomials(a, b)
        np.testing.assert_allclose(result, a)


class TestExpandSeasonalPolynomial:
    def test_basic(self) -> None:
        coeffs = np.array([1.0, -0.5])
        result = expand_seasonal_polynomial(coeffs, s=4, max_lag=10)
        assert result[0] == 1.0
        assert result[4] == -0.5
        assert np.sum(np.abs(result)) == pytest.approx(1.5)

    def test_truncation(self) -> None:
        coeffs = np.array([1.0, -0.5, 0.3])
        result = expand_seasonal_polynomial(coeffs, s=12, max_lag=20)
        assert result[0] == 1.0
        assert result[12] == -0.5
        assert len(result) == 20


class TestValidateEndog:
    def test_valid_array(self) -> None:
        y = [1.0, 2.0, 3.0]
        result = validate_endog(y)
        assert result.dtype == np.float64
        assert result.ndim == 1

    def test_2d_raises(self) -> None:
        with pytest.raises(ValueError, match=r"^endog must be 1-D"):
            validate_endog(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match=r"^endog must not be empty$"):
            validate_endog(np.array([]))

    def test_all_nan_raises(self) -> None:
        with pytest.raises(ValueError, match=r"^endog contains only NaN"):
            validate_endog(np.array([np.nan, np.nan]))


class TestValidateOrder:
    def test_valid(self) -> None:
        assert validate_order((1, 1, 1)) == (1, 1, 1)

    def test_wrong_length(self) -> None:
        with pytest.raises(ValueError, match=r"^order must have 3 elements"):
            validate_order((1, 2))  # type: ignore[arg-type]

    def test_negative(self) -> None:
        with pytest.raises(ValueError, match=r"^order components must be non-negative"):
            validate_order((1, -1, 0))


class TestValidateSeasonalOrder:
    def test_valid(self) -> None:
        assert validate_seasonal_order((1, 1, 1, 12)) == (1, 1, 1, 12)

    def test_wrong_length(self) -> None:
        with pytest.raises(ValueError, match=r"^seasonal_order must have 4 elements"):
            validate_seasonal_order((1, 2, 3))  # type: ignore[arg-type]

    def test_negative(self) -> None:
        with pytest.raises(ValueError, match=r"^seasonal_order components must be non-negative"):
            validate_seasonal_order((-1, 0, 0, 12))

    def test_small_s(self) -> None:
        with pytest.raises(ValueError, match=r"^seasonal period s must be >= 2"):
            validate_seasonal_order((1, 0, 0, 1))

    def test_zero_seasonal_with_s_zero(self) -> None:
        result = validate_seasonal_order((0, 0, 0, 0))
        assert result == (0, 0, 0, 0)


class TestLogging:
    def test_configure_logging_stdlib(self) -> None:
        from chronobox.utils.logging import configure_logging, get_logger

        configure_logging(level="DEBUG")
        logger = get_logger("test_chronobox")
        assert logger is not None

    def test_get_logger_returns_logger(self) -> None:
        from chronobox.utils.logging import get_logger

        logger = get_logger("chronobox.test")
        assert logger is not None

    def test_configure_logging_json(self) -> None:
        from chronobox.utils.logging import configure_logging

        configure_logging(level="WARNING", json_output=True)
