"""Tests for the Gregory-Hansen cointegration test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.cointegration.gregory_hansen import gregory_hansen_test


class TestGregoryHansen:
    """Tests for gregory_hansen_test function."""

    def test_gh_break_detected(self) -> None:
        """GH should detect cointegration with break."""
        np.random.seed(42)
        T = 300
        bp = T // 2
        x = np.cumsum(np.random.randn(T))
        # Cointegrating relationship with regime shift at bp
        y = np.zeros(T)
        y[:bp] = 1.0 * x[:bp] + np.random.randn(bp) * 0.3
        y[bp:] = 2.0 * x[bp:] + 3.0 + np.random.randn(T - bp) * 0.3
        result = gregory_hansen_test(y, x, model="c")
        # Break should be detected near bp
        break_date = result.additional_info["break_date"]
        assert abs(break_date - bp) < 30, (
            f"Break detected at {break_date}, expected near {bp}"
        )

    def test_gh_no_cointegration(self) -> None:
        """GH should not reject for independent random walks."""
        np.random.seed(42)
        T = 200
        x = np.cumsum(np.random.randn(T))
        y = np.cumsum(np.random.randn(T))
        result = gregory_hansen_test(y, x, model="c")
        assert not result.reject_at_5pct

    def test_gh_model_c(self) -> None:
        """Test model C (level shift)."""
        np.random.seed(42)
        T = 200
        x = np.cumsum(np.random.randn(T))
        y = x + np.random.randn(T) * 0.5
        result = gregory_hansen_test(y, x, model="c")
        assert result.additional_info["model"] == "c"

    def test_gh_model_ct(self) -> None:
        """Test model C/T (level shift + trend)."""
        np.random.seed(42)
        T = 200
        x = np.cumsum(np.random.randn(T))
        y = x + np.random.randn(T) * 0.5
        result = gregory_hansen_test(y, x, model="c/t")
        assert result.additional_info["model"] == "c/t"

    def test_gh_model_cs(self) -> None:
        """Test model C/S (regime shift)."""
        np.random.seed(42)
        T = 200
        x = np.cumsum(np.random.randn(T))
        y = x + np.random.randn(T) * 0.5
        result = gregory_hansen_test(y, x, model="c/s")
        assert result.additional_info["model"] == "c/s"

    def test_gh_critical_values(self) -> None:
        """Test that critical values match GH tables."""
        np.random.seed(42)
        T = 200
        x = np.cumsum(np.random.randn(T))
        y = x + np.random.randn(T)
        result = gregory_hansen_test(y, x, model="c")
        assert abs(result.critical_values["5%"] - (-4.61)) < 0.01
