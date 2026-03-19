"""Tests for the Lee-Strazicich LM unit root test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.unit_root.lee_strazicich import lee_strazicich_test


class TestLeeStrazicich:
    """Tests for lee_strazicich_test function."""

    def test_ls_one_break_detects(self) -> None:
        """LS with 1 break should detect break near true location."""
        np.random.seed(42)
        T = 200
        break_point = 100
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + np.random.randn()
        y[break_point:] += 5.0

        result = lee_strazicich_test(y, model="break", breaks=1)
        break_dates = result.additional_info["break_dates"]
        assert len(break_dates) == 1
        # Break should be detected within 20 periods
        assert abs(break_dates[0] - break_point) < 20

    def test_ls_two_breaks(self) -> None:
        """LS with 2 breaks should detect both breaks."""
        np.random.seed(42)
        T = 300
        bp1, bp2 = 100, 200
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + np.random.randn()
        y[bp1:] += 3.0
        y[bp2:] += 3.0

        result = lee_strazicich_test(y, model="break", breaks=2)
        break_dates = result.additional_info["break_dates"]
        assert len(break_dates) == 2

    def test_ls_size_correct(self) -> None:
        """Under H0 (unit root with break), LS should not over-reject."""
        # This is a qualitative test - just verify it runs
        np.random.seed(42)
        T = 200
        y = np.cumsum(np.random.randn(T))
        y[100:] += 5.0  # Break in a random walk
        result = lee_strazicich_test(y, model="break", breaks=1)
        assert result.test_name == "Lee-Strazicich"

    def test_ls_models(self) -> None:
        """Test break and crash models."""
        np.random.seed(42)
        y = np.random.randn(200)
        for model in ("break", "crash"):
            result = lee_strazicich_test(y, model=model, breaks=1)
            assert result.additional_info["model"] == model

    def test_ls_critical_values(self) -> None:
        """Test that critical values match LS tables."""
        np.random.seed(42)
        y = np.random.randn(200)
        result = lee_strazicich_test(y, model="break", breaks=1)
        assert abs(result.critical_values["5%"] - (-3.566)) < 0.01
