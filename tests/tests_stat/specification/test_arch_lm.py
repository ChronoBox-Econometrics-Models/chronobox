"""Tests for the ARCH-LM test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.specification.arch_lm import arch_lm_test


class TestARCHLM:
    """Tests for arch_lm_test function."""

    def test_arch_iid(self) -> None:
        """ARCH-LM should not reject for iid residuals."""
        np.random.seed(42)
        resid = np.random.randn(500)
        result = arch_lm_test(resid, nlags=5)
        assert not result.reject_at_5pct

    def test_arch_garch_resid(self) -> None:
        """ARCH-LM should reject for GARCH-like residuals."""
        np.random.seed(42)
        T = 500
        resid = np.zeros(T)
        h = np.ones(T)
        for t in range(1, T):
            h[t] = 0.05 + 0.3 * resid[t - 1] ** 2 + 0.6 * h[t - 1]
            resid[t] = np.sqrt(h[t]) * np.random.randn()
        result = arch_lm_test(resid, nlags=5)
        assert result.reject_at_5pct

    def test_arch_pvalue_range(self) -> None:
        """Test that p-value is in [0, 1]."""
        np.random.seed(42)
        resid = np.random.randn(200)
        result = arch_lm_test(resid, nlags=3)
        assert 0.0 <= result.pvalue <= 1.0
