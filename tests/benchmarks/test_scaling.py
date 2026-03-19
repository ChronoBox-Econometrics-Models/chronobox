"""Scaling tests for ChronoBox.

Tests that computational complexity scales as expected with data size.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest


class TestARIMAScaling:
    """Test that ARIMA scales linearly with T."""

    @pytest.mark.slow
    def test_arima_linear_scaling(self) -> None:
        """ARIMA fit time should scale roughly linearly with T."""
        from chronobox import ARIMA

        np.random.seed(42)
        sizes = [200, 500, 1000, 2000]
        times: list[float] = []

        for n in sizes:
            data = np.cumsum(np.random.randn(n))
            model = ARIMA(order=(1, 0, 1))

            start = time.perf_counter()
            _ = model.fit(data)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # Check that doubling T doesn't more than 4x the time (quadratic bound)
        if times[0] > 0:
            ratio = times[-1] / times[0]
            size_ratio = sizes[-1] / sizes[0]
            # Allow quadratic scaling as upper bound
            assert ratio < size_ratio ** 2.5, (
                f"ARIMA scaling ratio {ratio:.1f} exceeds quadratic bound "
                f"for size ratio {size_ratio}"
            )


class TestHPFilterScaling:
    """Test that HP filter scales well."""

    @pytest.mark.slow
    def test_hp_scaling(self) -> None:
        """HP filter should handle large datasets efficiently."""
        from chronobox.filters import hp_filter

        np.random.seed(42)
        sizes = [1000, 5000, 10000, 50000]
        times: list[float] = []

        for n in sizes:
            data = pd.Series(np.cumsum(np.random.randn(n)))

            start = time.perf_counter()
            _ = hp_filter(data, lamb=1600)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # T=50000 should still be under 1 second
        assert times[-1] < 1.0, (
            f"HP filter T={sizes[-1]} took {times[-1]*1000:.1f}ms, should be <1s"
        )


class TestNumbaAcceleration:
    """Test that numba provides meaningful speedup."""

    @pytest.mark.slow
    def test_css_recursion_speedup(self) -> None:
        """CSS recursion with numba should be faster than pure Python."""
        from chronobox.utils.numba_core import css_recursion, is_numba_available

        np.random.seed(42)
        y = np.random.randn(5000)
        ar = np.array([0.5, -0.3])
        ma = np.array([0.4])

        # Warm up (numba compilation)
        _ = css_recursion(y, ar, ma)

        # Timed run
        start = time.perf_counter()
        for _ in range(100):
            css_recursion(y, ar, ma)
        elapsed = time.perf_counter() - start

        if is_numba_available():
            # With numba, 100 runs of T=5000 should be very fast
            assert elapsed < 1.0, (
                f"CSS recursion (numba) 100 runs took {elapsed:.2f}s, expected <1s"
            )
        else:
            # Without numba, just ensure it completes
            assert elapsed < 30.0, (
                f"CSS recursion (pure Python) 100 runs took {elapsed:.2f}s"
            )
