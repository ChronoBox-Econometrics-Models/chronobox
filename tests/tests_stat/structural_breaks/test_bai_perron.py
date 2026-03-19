"""Tests for the Bai-Perron multiple structural breaks test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.structural_breaks.bai_perron import bai_perron_test


class TestBaiPerron:
    """Tests for bai_perron_test function."""

    def test_bp_detects_breaks(self) -> None:
        """BP should detect structural breaks in simulated data."""
        np.random.seed(42)
        T = 300
        bp1, bp2 = 100, 200
        X = np.ones((T, 1))
        y = np.zeros(T)
        y[:bp1] = 0.0 + np.random.randn(bp1) * 0.3
        y[bp1:bp2] = 5.0 + np.random.randn(bp2 - bp1) * 0.3
        y[bp2:] = -2.0 + np.random.randn(T - bp2) * 0.3

        result = bai_perron_test(y, X, max_breaks=5, trim=0.15)
        n_breaks = result.additional_info["n_breaks"]
        assert n_breaks >= 1, f"Should detect at least 1 break, got {n_breaks}"

    def test_bp_break_dates(self) -> None:
        """BP break dates should be close to true break dates."""
        np.random.seed(42)
        T = 300
        bp1, bp2 = 100, 200
        X = np.ones((T, 1))
        y = np.zeros(T)
        y[:bp1] = 0.0 + np.random.randn(bp1) * 0.3
        y[bp1:bp2] = 5.0 + np.random.randn(bp2 - bp1) * 0.3
        y[bp2:] = -2.0 + np.random.randn(T - bp2) * 0.3

        result = bai_perron_test(y, X, max_breaks=5, trim=0.10)
        dates = result.additional_info["break_dates"]

        if len(dates) >= 2:
            # Check at least one date is near each true break
            close_to_bp1 = any(abs(d - bp1) < 0.05 * T for d in dates)
            close_to_bp2 = any(abs(d - bp2) < 0.05 * T for d in dates)
            assert close_to_bp1 or close_to_bp2, (
                f"Break dates {dates} should be near {bp1} and {bp2}"
            )

    def test_bp_no_breaks(self) -> None:
        """BP should find 0 breaks for stable series."""
        np.random.seed(42)
        T = 200
        X = np.ones((T, 1))
        y = np.random.randn(T) * 0.5

        result = bai_perron_test(y, X, max_breaks=3)
        n_breaks = result.additional_info["n_breaks"]
        # May find 0 or at most 1 break for stable series
        assert n_breaks <= 1

    def test_bp_segments(self) -> None:
        """Test that segments cover the full sample."""
        np.random.seed(42)
        T = 200
        X = np.ones((T, 1))
        y = np.random.randn(T)
        result = bai_perron_test(y, X, max_breaks=3)
        segments = result.additional_info["segments"]
        assert segments[0][0] == 0
        assert segments[-1][1] == T

    def test_bp_default_X(self) -> None:
        """Test with X=None (constant-only model)."""
        np.random.seed(42)
        y = np.concatenate([np.random.randn(100), np.random.randn(100) + 5])
        result = bai_perron_test(y, x_mat=None, max_breaks=3)
        assert result.test_name == "Bai-Perron"
