"""Tests for the Phillips-Perron unit root test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.unit_root.pp import pp_test


class TestPP:
    """Tests for pp_test function."""

    def test_pp_random_walk_no_reject(self) -> None:
        """PP should not reject H0 for a random walk."""
        np.random.seed(42)
        T = 500
        y = np.cumsum(np.random.randn(T))
        result = pp_test(y, regression="c")
        assert not result.reject_at_5pct, (
            f"Should not reject H0 for random walk, stat={result.statistic:.4f}"
        )

    def test_pp_stationary_rejects(self) -> None:
        """PP should reject H0 for a stationary AR(1) process."""
        np.random.seed(42)
        T = 500
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + np.random.randn()
        result = pp_test(y, regression="c")
        assert result.reject_at_5pct, (
            f"Should reject H0 for stationary AR(1), stat={result.statistic:.4f}"
        )

    def test_pp_nile(self) -> None:
        """PP on Nile dataset - should be consistent with ADF."""
        nile = np.array([
            1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140,
            995, 935, 1110, 994, 1020, 960, 1180, 799, 958, 1140,
            1100, 1210, 1150, 1250, 1260, 1220, 1030, 1100, 774, 840,
            874, 694, 940, 833, 701, 916, 692, 1020, 1050, 969,
            831, 726, 456, 824, 702, 1120, 1100, 832, 764, 821,
            768, 845, 864, 862, 698, 845, 744, 796, 1040, 759,
            781, 865, 845, 944, 984, 897, 822, 1010, 771, 676,
            649, 846, 812, 742, 801, 1040, 860, 874, 848, 890,
            744, 749, 838, 1050, 918, 986, 797, 923, 975, 815,
            1020, 906, 901, 1170, 912, 746, 919, 718, 714, 740,
        ], dtype=float)
        result = pp_test(nile, regression="c")
        assert result.statistic < -2.0

    def test_pp_bandwidth_short(self) -> None:
        """Test with short bandwidth."""
        np.random.seed(42)
        y = np.cumsum(np.random.randn(200))
        result = pp_test(y, lags="short")
        assert result.lags_used is not None
        assert result.lags_used > 0

    def test_pp_bandwidth_long(self) -> None:
        """Test with long bandwidth."""
        np.random.seed(42)
        y = np.cumsum(np.random.randn(200))
        result = pp_test(y, lags="long")
        assert result.lags_used is not None

    def test_pp_bandwidth_int(self) -> None:
        """Test with integer bandwidth."""
        np.random.seed(42)
        y = np.cumsum(np.random.randn(200))
        result = pp_test(y, lags=5)
        assert result.lags_used == 5

    def test_pp_models(self) -> None:
        """Test constant and constant+trend models."""
        np.random.seed(42)
        y = np.cumsum(np.random.randn(200))
        for model in ("c", "ct"):
            result = pp_test(y, regression=model)
            assert result.additional_info["regression"] == model

    def test_pp_critical_values_ordered(self) -> None:
        """Test that critical values are properly ordered."""
        np.random.seed(42)
        y = np.cumsum(np.random.randn(200))
        result = pp_test(y, regression="c")
        cv = result.critical_values
        assert cv["1%"] < cv["5%"] < cv["10%"]

    def test_pp_additional_info(self) -> None:
        """Test that additional_info contains expected keys."""
        np.random.seed(42)
        y = np.cumsum(np.random.randn(200))
        result = pp_test(y)
        assert "Z_alpha" in result.additional_info
        assert "bandwidth" in result.additional_info
        assert "rho_hat" in result.additional_info
