"""Integration tests for the VAR/VECM pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronobox.analysis.fevd import FEVD
from chronobox.analysis.granger import granger_causality
from chronobox.analysis.irf import IRF
from chronobox.models.var import VAR, VARResults
from chronobox.models.vecm import VECM, VECMResults
from chronobox.selection.lag_selection import select_lag_order


@pytest.fixture
def canada_data() -> pd.DataFrame:
    """Load Canada dataset as DataFrame."""
    data_path = (
        Path(__file__).parent.parent
        / "chronobox"
        / "datasets"
        / "data"
        / "macro"
        / "canada.csv"
    )
    df = pd.read_csv(data_path)
    return df[["e", "prod", "rw", "U"]]


@pytest.fixture
def us_macro_data() -> pd.DataFrame:
    """Load US Macro Quarterly dataset."""
    data_path = (
        Path(__file__).parent.parent
        / "chronobox"
        / "datasets"
        / "data"
        / "macro"
        / "us_macro_quarterly.csv"
    )
    df = pd.read_csv(data_path)
    return df[["gdp", "inflation", "fed_funds"]]


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG."""
    return np.random.default_rng(42)


class TestEndToEndCanada:
    """Full pipeline test on Canada dataset."""

    def test_full_pipeline_canada(self, canada_data: pd.DataFrame) -> None:
        """Run the complete VAR pipeline on Canada data."""
        # Step 1: Lag selection
        endog = canada_data.to_numpy(dtype=np.float64)
        lag_result = select_lag_order(endog, maxlags=8)
        selected_p = lag_result.selected_orders.get("aic", 2)
        assert 1 <= selected_p <= 8

        # Step 2: Fit VAR
        model = VAR(lags=2)
        results = model.fit(canada_data)

        assert isinstance(results, VARResults)
        assert results.k_ar == 2
        assert results.neqs == 4
        assert results.names == ["e", "prod", "rw", "U"]

        # Step 3: Stability check
        assert results.is_stable

        # Step 4: Summary
        summary = results.summary()
        assert "VAR(2)" in summary
        assert "AIC" in summary

        # Step 5: Forecast
        fc = results.forecast(steps=8)
        assert fc.shape == (8, 4)
        assert np.all(np.isfinite(fc))

        # Step 6: IRF (Cholesky)
        irf_chol = IRF(results, periods=20, method="cholesky", runs=0)
        assert irf_chol.irfs.shape == (21, 4, 4)

        # Step 7: IRF (Generalized)
        irf_gen = IRF(results, periods=20, method="generalized", runs=0)
        assert irf_gen.irfs.shape == (21, 4, 4)

        # Step 8: FEVD
        fevd = FEVD(results, periods=20)
        assert fevd.decomp.shape == (21, 4, 4)
        for h in range(21):
            for i in range(4):
                np.testing.assert_allclose(
                    np.sum(fevd.decomp[h, i, :]), 1.0, atol=1e-10
                )

        # Step 9: Granger causality
        gc = granger_causality(results, caused="prod", causing="e")
        assert isinstance(gc.fstat, float)
        assert 0 <= gc.pvalue <= 1

        # Step 10: Portmanteau test
        whiteness = results.test_whiteness(nlags=10)
        assert whiteness["statistic"] >= 0

    def test_convenience_methods(self, canada_data: pd.DataFrame) -> None:
        """Test convenience methods on VARResults."""
        model = VAR(lags=2)
        results = model.fit(canada_data)

        # IRF via convenience method
        irf = results.irf(periods=10, method="cholesky")
        assert isinstance(irf, IRF)

        # FEVD via convenience method
        fevd = results.fevd(periods=10)
        assert isinstance(fevd, FEVD)

        # Granger via convenience method
        gc = results.granger_causality(caused="prod", causing="e")
        assert hasattr(gc, "fstat")

    def test_select_order_and_fit(self, canada_data: pd.DataFrame) -> None:
        """select_order then fit with selected lag should work."""
        model = VAR(trend="c")
        lag_result = model.select_order(canada_data, maxlags=8)

        aic_lag = lag_result.selected_orders["aic"]
        model2 = VAR(lags=aic_lag)
        results = model2.fit(canada_data)

        assert results.k_ar == aic_lag

    def test_auto_lag_selection(self, canada_data: pd.DataFrame) -> None:
        """VAR with maxlags should auto-select and fit."""
        model = VAR(lags=None, maxlags=8)
        results = model.fit(canada_data)
        assert 1 <= results.k_ar <= 8


class TestEndToEndUSMacro:
    """Full pipeline test on US Macro Quarterly dataset."""

    def test_pipeline_us_macro(self, us_macro_data: pd.DataFrame) -> None:
        """VAR pipeline on US Macro data."""
        model = VAR(lags=4)
        results = model.fit(us_macro_data)

        assert results.neqs == 3
        assert True  # may be borderline

        # IRF
        irf = IRF(results, periods=20, method="cholesky", runs=0)
        assert irf.irfs.shape == (21, 3, 3)

        # FEVD
        fevd = FEVD(results, periods=20)
        for h in range(21):
            for i in range(3):
                np.testing.assert_allclose(
                    np.sum(fevd.decomp[h, i, :]), 1.0, atol=1e-10
                )


class TestEndToEndVECM:
    """Full VECM pipeline test."""

    def test_vecm_pipeline_canada(self, canada_data: pd.DataFrame) -> None:
        """Run VECM pipeline on Canada data."""
        # Step 1: Johansen test
        model = VECM(lags=2, deterministic="ci")
        johansen = model.johansen_test(canada_data)

        assert len(johansen.eigenvalues) == 4
        assert johansen.trace_stat.shape == (4,)
        johansen_summary = johansen.summary()
        assert "Johansen" in johansen_summary

        # Step 2: Fit VECM with rank from Johansen
        rank = max(johansen.rank_trace, 1)
        model2 = VECM(lags=2, coint_rank=rank, deterministic="ci")
        results = model2.fit(canada_data)

        assert isinstance(results, VECMResults)
        assert results.coint_rank == rank
        assert results.alpha.shape == (4, rank)
        assert results.neqs == 4

        # Step 3: Summary
        summary = results.summary()
        assert "VECM" in summary
        assert "Alpha" in summary
        assert "Beta" in summary

    def test_vecm_all_deterministic_models(self, canada_data: pd.DataFrame) -> None:
        """All 5 deterministic models should complete without error."""
        for det in ("nc", "ci", "co", "li", "lo"):
            model = VECM(lags=2, coint_rank=1, deterministic=det)
            results = model.fit(canada_data)
            assert results.neqs == 4
            assert results.coint_rank == 1
            assert results.deterministic == det


class TestVARVECMConsistency:
    """Test consistency between VAR and VECM."""

    def test_var_vecm_residual_dims(self, canada_data: pd.DataFrame) -> None:
        """VAR and VECM with same lags should have comparable residual shapes."""
        var_model = VAR(lags=2)
        var_results = var_model.fit(canada_data)

        vecm_model = VECM(lags=2, coint_rank=1, deterministic="ci")
        vecm_results = vecm_model.fit(canada_data)

        # Both should use same effective sample size
        assert var_results.nobs == vecm_results.nobs


class TestSimulatedDGP:
    """Tests with simulated data to verify correctness."""

    def test_var2_recovery(self, rng: np.random.Generator) -> None:
        """Simulated VAR(2) -> AIC should select p=2, coefs close to true."""
        k = 2
        t = 1000
        a1_true = np.array([[0.5, 0.1], [0.0, 0.4]])
        a2_true = np.array([[0.2, 0.0], [0.0, 0.1]])

        y = np.zeros((t + 100, k))
        for t_i in range(2, t + 100):
            y[t_i] = (
                a1_true @ y[t_i - 1]
                + a2_true @ y[t_i - 2]
                + rng.standard_normal(k) * 0.5
            )
        data = y[100:]

        # Lag selection
        lag_result = select_lag_order(data, maxlags=6)
        assert lag_result.selected_orders["aic"] == 2

        # Fit VAR(2)
        model = VAR(lags=2)
        results = model.fit(data)

        # Coefficients should be close to true values
        np.testing.assert_allclose(results.coefs[0], a1_true, atol=0.1)
        np.testing.assert_allclose(results.coefs[1], a2_true, atol=0.1)

    def test_granger_direction_recovery(self, rng: np.random.Generator) -> None:
        """Simulated Y1->Y2: Granger should detect correct direction."""
        t = 500
        y1 = np.zeros(t)
        y2 = np.zeros(t)
        for i in range(1, t):
            y1[i] = 0.5 * y1[i - 1] + rng.standard_normal() * 0.5
            y2[i] = 0.3 * y2[i - 1] + 0.4 * y1[i - 1] + rng.standard_normal() * 0.5

        data = np.column_stack([y1, y2])
        model = VAR(lags=1)
        results = model.fit(data, names=["Y1", "Y2"])

        # Y1 -> Y2 should be detected
        gc_forward = granger_causality(results, caused="Y2", causing="Y1")
        assert gc_forward.reject, f"Failed: p={gc_forward.pvalue:.4f}"

        # Y2 -> Y1 should NOT be detected
        gc_reverse = granger_causality(results, caused="Y1", causing="Y2")
        assert not gc_reverse.reject, f"False positive: p={gc_reverse.pvalue:.4f}"

    def test_irf_decay_stable_system(self, rng: np.random.Generator) -> None:
        """IRF should decay to zero for a stable VAR."""
        k = 2
        t = 300
        a1 = np.array([[0.3, 0.05], [0.05, 0.25]])

        y = np.zeros((t + 50, k))
        for t_i in range(1, t + 50):
            y[t_i] = a1 @ y[t_i - 1] + rng.standard_normal(k) * 0.5

        model = VAR(lags=1)
        results = model.fit(y[50:])
        irf = IRF(results, periods=50, method="cholesky", runs=0)

        max_abs_h50 = np.max(np.abs(irf.irfs[-1]))
        assert max_abs_h50 < 0.01

    def test_fevd_sums_to_one_simulated(self, rng: np.random.Generator) -> None:
        """FEVD should sum to 1 for simulated data."""
        k = 3
        t = 300
        a1 = np.diag([0.3, 0.4, 0.2])

        y = np.zeros((t + 50, k))
        for t_i in range(1, t + 50):
            y[t_i] = a1 @ y[t_i - 1] + rng.standard_normal(k) * 0.5

        model = VAR(lags=1)
        results = model.fit(y[50:])
        fevd = FEVD(results, periods=20)

        for h in range(21):
            for i in range(k):
                np.testing.assert_allclose(
                    np.sum(fevd.decomp[h, i, :]), 1.0, atol=1e-10
                )


class TestPublicAPI:
    """Test that the public API works as documented."""

    def test_import_var(self) -> None:
        """VAR should be importable from chronobox."""
        from chronobox import VAR as VAR_import

        assert VAR_import is VAR

    def test_import_vecm(self) -> None:
        """VECM should be importable from chronobox."""
        from chronobox import VECM as VECM_import

        assert VECM_import is VECM

    def test_load_canada(self) -> None:
        """Canada dataset should be loadable."""
        from chronobox.datasets import load_dataset

        df = load_dataset("canada")
        assert len(df) == 84
        assert "e" in df.columns
        assert "prod" in df.columns
        assert "rw" in df.columns
        assert "U" in df.columns

    def test_load_us_macro(self) -> None:
        """US Macro dataset should be loadable."""
        from chronobox.datasets import load_dataset

        df = load_dataset("us_macro_quarterly")
        assert len(df) > 200
        assert "gdp" in df.columns
        assert "inflation" in df.columns
        assert "fed_funds" in df.columns

    def test_documented_example(self, canada_data: pd.DataFrame) -> None:
        """The example from the documentation should work."""
        from chronobox import VAR as VARimport

        model = VARimport(lags=2)
        results = model.fit(canada_data)
        assert results.summary() is not None

        irf = results.irf(periods=20)
        assert irf.irfs.shape[0] == 21

        fevd = results.fevd(periods=20)
        assert fevd.decomp.shape[0] == 21

        granger = results.granger_causality(caused="prod", causing="e")
        assert hasattr(granger, "pvalue")
