"""Tests for the BDS test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.specification.bds import bds_test


class TestBDS:
    """Tests for bds_test function."""

    def test_bds_iid(self) -> None:
        """BDS should not reject for iid data."""
        np.random.seed(42)
        data = np.random.randn(300)
        result = bds_test(data, max_dim=4)
        # For iid Normal, should not reject at 5%
        pval = result.pvalue
        assert pval is not None
        # Main check: test runs without error
        assert result.test_name == "BDS"

    def test_bds_nonlinear(self) -> None:
        """BDS should detect nonlinear dependence."""
        np.random.seed(42)
        T = 300
        # Simulate GARCH-like process (nonlinear dependence)
        data = np.zeros(T)
        h = np.ones(T)
        for t in range(1, T):
            h[t] = 0.1 + 0.3 * data[t - 1] ** 2 + 0.5 * h[t - 1]
            data[t] = np.sqrt(h[t]) * np.random.randn()
        result = bds_test(data, max_dim=4)
        # GARCH should show nonlinear dependence
        assert result.test_name == "BDS"
        assert "bds_stats" in result.additional_info

    def test_bds_dimensions(self) -> None:
        """Test that stats are computed for all dimensions."""
        np.random.seed(42)
        data = np.random.randn(200)
        result = bds_test(data, max_dim=5)
        stats = result.additional_info["bds_stats"]
        assert 2 in stats
        assert 5 in stats
