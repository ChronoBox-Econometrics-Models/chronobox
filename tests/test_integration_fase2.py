"""Integration tests for FASE 2 - Modelos Univariados Completos.

These tests validate that all FASE 2 components work together correctly
and that the public API is consistent.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def airline_passengers() -> np.ndarray:
    """Load airline passengers data."""
    try:
        from chronobox.datasets import load_dataset
        df = load_dataset("airline")
        return df["passengers"].to_numpy(dtype=np.float64)
    except Exception:
        rng = np.random.default_rng(42)
        t = np.arange(144)
        trend = 100 + 2 * t
        seasonal = 20 * np.sin(2 * np.pi * t / 12)
        return trend + seasonal + rng.normal(0, 5, 144)


@pytest.fixture
def nile_volume() -> np.ndarray:
    """Load Nile volume data."""
    try:
        from chronobox.datasets import load_dataset
        df = load_dataset("nile")
        return df["volume"].to_numpy(dtype=np.float64)
    except Exception:
        rng = np.random.default_rng(42)
        return 1000 + rng.normal(0, 100, 100)


class TestImports:
    """Test that all FASE 2 modules are importable."""

    def test_import_arfima(self) -> None:
        from chronobox.models.arfima import ARFIMA
        assert ARFIMA is not None

    def test_import_ets(self) -> None:
        from chronobox.models.ets import ETS
        assert ETS is not None

    def test_import_holtwinters(self) -> None:
        from chronobox.models.holtwinters import HoltWinters
        assert HoltWinters is not None

    def test_import_theta(self) -> None:
        from chronobox.models.theta import ThetaMethod
        assert ThetaMethod is not None

    def test_import_stl(self) -> None:
        from chronobox.decomposition.stl import STL
        assert STL is not None

    def test_import_classical(self) -> None:
        from chronobox.decomposition.classical import ClassicalDecomposition
        assert ClassicalDecomposition is not None

    def test_import_x13(self) -> None:
        from chronobox.decomposition.x13_wrapper import X13Wrapper
        assert X13Wrapper is not None

    def test_import_auto_ets(self) -> None:
        from chronobox.selection.auto_ets import auto_ets
        assert auto_ets is not None

    def test_import_from_package(self) -> None:
        """Test imports from the main package."""
        from chronobox.decomposition import STL, ClassicalDecomposition
        from chronobox.models.arfima import ARFIMA
        from chronobox.models.ets import ETS
        from chronobox.models.holtwinters import HoltWinters
        from chronobox.models.theta import ThetaMethod
        from chronobox.selection.auto_ets import auto_ets
        assert all([ARFIMA, ETS, HoltWinters, ThetaMethod, STL, ClassicalDecomposition, auto_ets])


class TestETSEndToEnd:
    """End-to-end ETS tests."""

    def test_ets_mam_airline_full_pipeline(self, airline_passengers: np.ndarray) -> None:
        """Full ETS(M,A,M) pipeline on airline data."""
        from chronobox.models.ets import ETS

        model = ETS(error="M", trend="A", seasonal="M", seasonal_period=12)
        results = model.fit(airline_passengers)

        # Check results
        assert results.nobs == 144
        assert 0 < results.alpha < 1
        assert results.beta is not None and results.beta > 0
        assert results.gamma is not None and results.gamma > 0
        assert results.s0 is not None and len(results.s0) == 12
        assert np.isfinite(results.loglik)
        assert np.isfinite(results.aicc)

        # Forecast
        fc = results.forecast(steps=12)
        assert len(fc) == 12
        assert np.all(fc > 0)
        assert np.all(np.isfinite(fc))

        # Summary
        s = results.summary()
        assert "ETS(M,A,M)" in s

    def test_ets_ann_matches_ses(self, airline_passengers: np.ndarray) -> None:
        """ETS(A,N,N) should be equivalent to SES."""
        from chronobox.models.ets import ETS

        model = ETS(error="A", trend="N", seasonal="N")
        results = model.fit(airline_passengers)

        # Should have only alpha, l0, sigma2 as parameters
        assert results.beta is None
        assert results.gamma is None
        assert results.phi is None
        assert results.s0 is None


class TestAutoETSEndToEnd:
    """End-to-end Auto-ETS tests."""

    def test_auto_ets_airline(self, airline_passengers: np.ndarray) -> None:
        """Auto-ETS on airline should select seasonal model."""
        from chronobox.selection.auto_ets import auto_ets

        result = auto_ets(airline_passengers, seasonal_period=12)
        e, t, s = result.best_spec
        assert s != "N", f"Expected seasonal model, got ETS({e},{t},{s})"

        # Should be able to forecast
        fc = result.forecast(steps=12)
        assert len(fc) == 12
        assert np.all(np.isfinite(fc))

        # Summary should work
        summary = result.summary()
        assert "Auto-ETS" in summary

    def test_auto_ets_nile(self, nile_volume: np.ndarray) -> None:
        """Auto-ETS on Nile should select non-seasonal model."""
        from chronobox.selection.auto_ets import auto_ets

        result = auto_ets(nile_volume, seasonal_period=None)
        e, t, s = result.best_spec
        assert s == "N", f"Expected non-seasonal for Nile, got ETS({e},{t},{s})"


class TestHoltWintersEndToEnd:
    """End-to-end Holt-Winters tests."""

    def test_hw_airline_pipeline(self, airline_passengers: np.ndarray) -> None:
        """Full HW pipeline on airline data."""
        from chronobox.models.holtwinters import HoltWinters

        # Additive
        hw_add = HoltWinters(seasonal="additive", seasonal_period=12)
        res_add = hw_add.fit(airline_passengers)
        fc_add = res_add.forecast(steps=12)
        assert len(fc_add) == 12

        # Multiplicative
        hw_mul = HoltWinters(seasonal="multiplicative", seasonal_period=12)
        res_mul = hw_mul.fit(airline_passengers)
        fc_mul = res_mul.forecast(steps=12)
        assert len(fc_mul) == 12
        assert np.all(fc_mul > 0)

        # Summary
        s = res_mul.summary()
        assert "Holt-Winters" in s


class TestThetaEndToEnd:
    """End-to-end Theta method tests."""

    def test_theta_pipeline(self, airline_passengers: np.ndarray) -> None:
        """Full Theta pipeline."""
        from chronobox.models.theta import ThetaMethod

        model = ThetaMethod()
        results = model.fit(airline_passengers)
        fc = results.forecast(steps=12)
        assert len(fc) == 12
        assert np.all(np.isfinite(fc))
        assert results.alpha > 0
        assert results.slope != 0  # airline has trend

        s = results.summary()
        assert "Theta" in s


class TestDecompositionEndToEnd:
    """End-to-end decomposition tests."""

    def test_stl_airline(self, airline_passengers: np.ndarray) -> None:
        """STL decomposition of airline data."""
        from chronobox.decomposition.stl import STL

        stl = STL(period=12, seasonal=7)
        result = stl.fit(airline_passengers)

        # Exact reconstruction
        reconstructed = result.trend + result.seasonal + result.remainder
        np.testing.assert_allclose(
            reconstructed, result.observed, atol=1e-10,
            err_msg="STL: trend + seasonal + remainder != original",
        )

        # Summary
        s = result.summary()
        assert "Decomposition" in s

    def test_classical_additive(self, airline_passengers: np.ndarray) -> None:
        """Classical additive decomposition."""
        from chronobox.decomposition.classical import ClassicalDecomposition

        cd = ClassicalDecomposition(period=12, model="additive")
        result = cd.fit(airline_passengers)

        # Check T + S + R = y (where trend available)
        valid = ~np.isnan(result.trend)
        reconstructed = result.trend[valid] + result.seasonal[valid] + result.remainder[valid]
        np.testing.assert_allclose(
            reconstructed, result.observed[valid], atol=1e-10,
        )

    def test_classical_multiplicative(self, airline_passengers: np.ndarray) -> None:
        """Classical multiplicative decomposition."""
        from chronobox.decomposition.classical import ClassicalDecomposition

        cd = ClassicalDecomposition(period=12, model="multiplicative")
        result = cd.fit(airline_passengers)

        valid = ~np.isnan(result.trend) & ~np.isnan(result.remainder)
        reconstructed = result.trend[valid] * result.seasonal[valid] * result.remainder[valid]
        np.testing.assert_allclose(
            reconstructed, result.observed[valid], atol=1e-10,
        )


class TestARFIMAEndToEnd:
    """End-to-end ARFIMA tests."""

    def test_arfima_pipeline(self) -> None:
        """Full ARFIMA pipeline."""
        from chronobox.models.arfima import ARFIMA, simulate_arfima

        # Simulate
        rng = np.random.default_rng(42)
        y = simulate_arfima(n=500, d=0.3, rng=rng)

        # Fit
        model = ARFIMA(order=(1, 0.3, 0))
        results = model.fit(y)
        assert np.isfinite(results.loglik)

        # Forecast
        fc = results.forecast(steps=10)
        assert len(fc) == 10
        assert np.all(np.isfinite(fc))

        # Estimate d
        d_hat = model.estimate_d(y, method="gph")
        assert -0.5 < d_hat < 0.5

        # Summary
        s = results.summary()
        assert "ARFIMA" in s

    def test_arfima_d_zero_consistency(self) -> None:
        """ARFIMA with d=0 should behave like ARMA."""
        from chronobox.models.arfima import ARFIMA

        rng = np.random.default_rng(42)
        phi = 0.5
        n = 300
        eps = rng.normal(size=n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + eps[t]

        model = ARFIMA(order=(1, 0.0, 0))
        results = model.fit(y)
        assert results.d == 0.0
        assert abs(results.ar_params[0] - phi) < 0.15


class TestCrossModelConsistency:
    """Tests comparing different models for consistency."""

    def test_all_models_forecast_same_length(self, airline_passengers: np.ndarray) -> None:
        """All models should produce forecasts of the requested length."""
        from chronobox.models.ets import ETS
        from chronobox.models.holtwinters import HoltWinters
        from chronobox.models.theta import ThetaMethod

        steps = 12

        ets = ETS(error="A", trend="A", seasonal="A", seasonal_period=12)
        hw = HoltWinters(seasonal="additive", seasonal_period=12)
        theta = ThetaMethod()

        fc_ets = ets.fit(airline_passengers).forecast(steps=steps)
        fc_hw = hw.fit(airline_passengers).forecast(steps=steps)
        fc_theta = theta.fit(airline_passengers).forecast(steps=steps)

        assert len(fc_ets) == steps
        assert len(fc_hw) == steps
        assert len(fc_theta) == steps

    def test_decomposition_methods_agree_on_trend_direction(
        self, airline_passengers: np.ndarray
    ) -> None:
        """Both STL and Classical should show increasing trend for airline."""
        from chronobox.decomposition.classical import ClassicalDecomposition
        from chronobox.decomposition.stl import STL

        stl = STL(period=12)
        cd = ClassicalDecomposition(period=12, model="additive")

        stl_result = stl.fit(airline_passengers)
        cd_result = cd.fit(airline_passengers)

        # Both should show increasing trend
        stl_trend_diff = stl_result.trend[-12:].mean() - stl_result.trend[:12].mean()
        valid = ~np.isnan(cd_result.trend)
        cd_valid_trend = cd_result.trend[valid]
        cd_trend_diff = cd_valid_trend[-12:].mean() - cd_valid_trend[:12].mean()

        assert stl_trend_diff > 0, "STL trend should be increasing"
        assert cd_trend_diff > 0, "Classical trend should be increasing"


class TestFullPipeline:
    """Test the complete workflow as described in the spec."""

    def test_complete_workflow(self, airline_passengers: np.ndarray) -> None:
        """Test the complete workflow from the FASE 2 overview."""
        from chronobox.decomposition.classical import ClassicalDecomposition
        from chronobox.decomposition.stl import STL
        from chronobox.models.arfima import ARFIMA
        from chronobox.models.ets import ETS
        from chronobox.models.holtwinters import HoltWinters
        from chronobox.models.theta import ThetaMethod
        from chronobox.selection.auto_ets import auto_ets

        y = airline_passengers

        # 1. ETS
        ets = ETS(error="M", trend="A", seasonal="M", seasonal_period=12)
        ets_res = ets.fit(y)
        ets_fc = ets_res.forecast(steps=12)
        assert len(ets_fc) == 12
        assert ets_res.summary() is not None

        # 2. Auto-ETS
        best = auto_ets(y, seasonal_period=12)
        best_fc = best.forecast(steps=12)
        assert len(best_fc) == 12
        assert best.summary() is not None

        # 3. Holt-Winters
        hw = HoltWinters(seasonal="multiplicative", seasonal_period=12)
        hw_res = hw.fit(y)
        hw_fc = hw_res.forecast(steps=12)
        assert len(hw_fc) == 12

        # 4. Theta
        theta = ThetaMethod()
        theta_res = theta.fit(y)
        theta_fc = theta_res.forecast(steps=12)
        assert len(theta_fc) == 12

        # 5. STL
        stl = STL(period=12)
        stl_res = stl.fit(y)
        reconstructed = stl_res.trend + stl_res.seasonal + stl_res.remainder
        np.testing.assert_allclose(reconstructed, y, atol=1e-10)

        # 6. Classical Decomposition
        cd = ClassicalDecomposition(period=12, model="additive")
        cd_res = cd.fit(y)
        assert cd_res.model == "additive"

        # 7. ARFIMA
        arfima = ARFIMA(order=(1, 0.0, 0))
        arfima_res = arfima.fit(y)
        arfima_fc = arfima_res.forecast(steps=12)
        assert len(arfima_fc) == 12
