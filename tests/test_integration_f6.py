"""
Integration tests for FASE 6: Filters, ARDL/ECM, Spillover.

Tests cross-component consistency and end-to-end pipelines.
"""

import numpy as np
import pytest


class TestFilterConsistency:
    """Cross-filter consistency tests."""

    def setup_method(self) -> None:
        """Generate test series with known components."""
        np.random.seed(42)
        T = 400
        t = np.arange(T)

        # Trend: slow-moving
        self.trend_true = 0.05 * t + 0.0001 * t**2

        # Cycle: business cycle frequency (period ~20 quarters)
        self.cycle_true = 2.0 * np.sin(2 * np.pi * t / 20)

        # Noise: high frequency
        self.noise = 0.3 * np.random.randn(T)

        # Combined series
        self.y = self.trend_true + self.cycle_true + self.noise

    def test_hp_trend_plus_cycle_equals_original(self) -> None:
        """HP: trend + cycle = original."""
        from chronobox.filters import hp_filter

        trend, cycle = hp_filter(self.y, lamb=1600)
        np.testing.assert_allclose(trend + cycle, self.y, atol=1e-10)

    def test_bn_trend_plus_cycle_equals_original(self) -> None:
        """BN: trend + cycle = original."""
        from chronobox.filters import bn_decomposition

        # BN works on I(1), so use cumulated series
        y_i1 = self.y.cumsum()
        trend, cycle = bn_decomposition(y_i1, p=2)
        np.testing.assert_allclose(trend + cycle, y_i1, atol=1e-10)

    def test_cf_full_sample(self) -> None:
        """CF should not lose observations."""
        from chronobox.filters import cf_filter

        cycle = cf_filter(self.y, low=6, high=32)
        assert len(cycle) == len(self.y)

    def test_cf_similar_to_bk_interior(self) -> None:
        """CF and BK should agree in the interior."""
        from chronobox.filters import bk_filter, cf_filter

        K = 12
        cycle_cf = cf_filter(self.y, low=6, high=32)
        cycle_bk = bk_filter(self.y, low=6, high=32, trunc=K)

        cf_interior = cycle_cf[K : len(self.y) - K]
        corr = np.corrcoef(cf_interior, cycle_bk)[0, 1]
        assert corr > 0.90, f"CF-BK interior correlation {corr} < 0.90"

    def test_hp_cf_correlation(self) -> None:
        """HP and CF cycles should be positively correlated."""
        from chronobox.filters import cf_filter, hp_filter

        _, cycle_hp = hp_filter(self.y, lamb=1600)
        cycle_cf = cf_filter(self.y, low=6, high=32)

        corr = np.corrcoef(cycle_hp, cycle_cf)[0, 1]
        assert corr > 0.5, f"HP-CF correlation {corr} < 0.5"

    def test_hamilton_vs_hp_correlation(self) -> None:
        """Hamilton and HP cycles should be positively correlated."""
        from chronobox.filters import hamilton_filter, hp_filter

        _, cycle_hp = hp_filter(self.y, lamb=1600)
        _, cycle_ham = hamilton_filter(self.y, h=8, p=4)

        valid = ~np.isnan(cycle_ham)
        corr = np.corrcoef(cycle_hp[valid], cycle_ham[valid])[0, 1]
        assert corr > 0.3, f"Hamilton-HP correlation {corr} too low"

    def test_bk_weights_sum_zero(self) -> None:
        """BK filter weights must sum to zero."""
        from chronobox.filters.bk import bk_filter_weights

        weights = bk_filter_weights(low=6, high=32, trunc=12)
        np.testing.assert_allclose(np.sum(weights), 0.0, atol=1e-14)

    def test_all_filters_on_constant_series(self) -> None:
        """All filters should return zero cycle for a constant series."""
        from chronobox.filters import bk_filter, cf_filter, hp_filter

        y_const = np.ones(100) * 5.0

        _, cycle_hp = hp_filter(y_const, lamb=1600)
        np.testing.assert_allclose(cycle_hp, 0.0, atol=1e-10)

        cycle_bk = bk_filter(y_const, low=6, high=32, trunc=12)
        np.testing.assert_allclose(cycle_bk, 0.0, atol=1e-12)

        cycle_cf = cf_filter(y_const, low=6, high=32)
        np.testing.assert_allclose(cycle_cf, 0.0, atol=1e-10)


class TestARDLECMPipeline:
    """Integration tests for ARDL -> ECM pipeline."""

    def setup_method(self) -> None:
        """Generate cointegrated data."""
        np.random.seed(42)
        T = 400

        # x follows random walks
        self.x = np.random.randn(T, 2).cumsum(axis=0)

        # y is cointegrated with x
        eq_error = np.random.randn(T) * 0.5
        self.y = 2.0 * self.x[:, 0] + 0.5 * self.x[:, 1] + eq_error
        for t in range(1, T):
            self.y[t] += 0.7 * (
                self.y[t - 1] - 2.0 * self.x[t - 1, 0] - 0.5 * self.x[t - 1, 1]
            )

    def test_ardl_fit_and_bounds(self) -> None:
        """ARDL auto-select + bounds test pipeline."""
        from chronobox.models.ardl import ARDL

        ardl = ARDL(max_p=3, max_q=3, criterion="aic")
        result = ardl.fit(self.y, self.x)

        assert result.nobs > 0
        assert result.r_squared >= 0

        bt = result.bounds_test()
        assert "f_statistic" in bt
        assert float(bt["f_statistic"]) >= 0

    def test_ardl_to_ecm(self) -> None:
        """ARDL -> to_ecm() pipeline."""
        from chronobox.models.ardl import ARDL

        ardl = ARDL(max_p=2, max_q=2)
        ardl_result = ardl.fit(self.y, self.x, p=2, x_lags=[1, 1])

        ecm_result = ardl_result.to_ecm()
        assert ecm_result.speed_of_adjustment is not None
        assert len(ecm_result.long_run_coefficients) == 2

    def test_ecm_direct(self) -> None:
        """Direct ECM estimation."""
        from chronobox.models.ecm import ECM

        ecm = ECM(lags=2)
        result = ecm.fit(self.y, self.x)

        assert result.speed_of_adjustment < 0, (
            f"Speed of adjustment should be negative, got {result.speed_of_adjustment}"
        )
        assert len(result.long_run_coefficients) == 2
        assert result.nobs > 0

    def test_ecm_summary(self) -> None:
        """ECM summary should work."""
        from chronobox.models.ecm import ECM

        ecm = ECM(lags=2)
        result = ecm.fit(self.y, self.x)
        s = result.summary()
        assert len(s) > 50


class TestSpilloverIntegration:
    """Integration tests for spillover index."""

    def setup_method(self) -> None:
        """Generate VAR data."""
        np.random.seed(42)
        T = 600
        K = 3
        A = np.array([
            [0.5, 0.2, 0.0],
            [0.1, 0.4, 0.1],
            [0.0, 0.1, 0.6],
        ])
        data = np.zeros((T, K))
        for t in range(1, T):
            data[t] = A @ data[t - 1] + np.random.randn(K)
        self.data = data

    def test_full_pipeline(self) -> None:
        """Full spillover pipeline: fit + summary."""
        from chronobox.analysis.spillover import SpilloverIndex

        sp = SpilloverIndex(lags=2, horizon=10)
        result = sp.fit(self.data)

        assert 0 <= result.total_spillover <= 100
        np.testing.assert_allclose(np.sum(result.net_spillover), 0.0, atol=1e-10)
        np.testing.assert_allclose(result.fevd_table.sum(axis=1), 1.0, atol=1e-10)

        s = result.summary()
        assert "Total Spillover" in s

    def test_rolling_pipeline(self) -> None:
        """Rolling spillover pipeline."""
        from chronobox.analysis.spillover import SpilloverIndex

        sp = SpilloverIndex(lags=1, horizon=5)
        rolling = sp.rolling(self.data, window=150)

        assert len(rolling.total_spillover) == len(self.data) - 150 + 1
        # Should have variation
        valid = rolling.total_spillover[~np.isnan(rolling.total_spillover)]
        assert len(valid) > 0
        assert np.std(valid) > 0


class TestEdgeCases:
    """Edge case tests across all F6 components."""

    def test_short_series_hp(self) -> None:
        """HP filter should work with minimum length."""
        from chronobox.filters import hp_filter

        y = np.array([1.0, 2.0, 3.0, 4.0])  # minimum T=4
        trend, cycle = hp_filter(y, lamb=1600)
        np.testing.assert_allclose(trend + cycle, y, atol=1e-10)

    def test_short_series_errors(self) -> None:
        """Filters should raise on too-short series."""
        from chronobox.filters import bk_filter, cf_filter, hp_filter

        with pytest.raises(ValueError):
            hp_filter(np.array([1.0, 2.0, 3.0]))  # T=3 < 4
        with pytest.raises(ValueError):
            bk_filter(np.ones(20), trunc=12)  # T=20 <= 2*trunc=24
        with pytest.raises(ValueError):
            cf_filter(np.array([1.0, 2.0, 3.0]))  # T=3 < 4

    def test_all_imports(self) -> None:
        """All F6 public imports should work."""
        from chronobox.analysis.spillover import SpilloverIndex
        from chronobox.filters import (
            bk_filter,
            bn_decomposition,
            cf_filter,
            hamilton_filter,
            hp_filter,
        )
        from chronobox.models.ardl import ARDL
        from chronobox.models.ecm import ECM

        assert callable(hp_filter)
        assert callable(bk_filter)
        assert callable(cf_filter)
        assert callable(hamilton_filter)
        assert callable(bn_decomposition)
        assert callable(ARDL)
        assert callable(ECM)
        assert callable(SpilloverIndex)
