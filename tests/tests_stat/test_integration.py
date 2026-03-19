"""Integration tests for all statistical tests in chronobox.tests_stat.

Validates:
1. All tests return TestResult with mandatory fields
2. Consistency across test categories for the same data
3. Full workflows (unit root -> cointegration -> specification)
"""

from __future__ import annotations

import numpy as np
import pytest

from chronobox.tests_stat import (
    TestResult,
    adf_test,
    arch_lm_test,
    bai_perron_test,
    bds_test,
    bounds_test,
    breusch_godfrey_test,
    chow_test,
    cusum_test,
    cusumsq_test,
    durbin_watson_test,
    engle_granger_test,
    ers_test,
    gregory_hansen_test,
    hegy_test,
    jarque_bera_test,
    kpss_test,
    lee_strazicich_test,
    ljung_box_test,
    phillips_ouliaris_test,
    pp_test,
    qlr_test,
    reset_test,
    white_test,
    zivot_andrews_test,
)


class TestTestResultStandard:
    """All tests must return TestResult with mandatory fields."""

    @pytest.fixture
    def stationary_data(self) -> np.ndarray:
        np.random.seed(42)
        T = 300
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + np.random.randn()
        return y

    @pytest.fixture
    def random_walk(self) -> np.ndarray:
        np.random.seed(42)
        return np.cumsum(np.random.randn(300))

    def _check_test_result(self, result: TestResult) -> None:
        """Verify mandatory fields of TestResult."""
        assert isinstance(result.test_name, str)
        assert len(result.test_name) > 0
        assert isinstance(result.statistic, (int, float))
        assert result.pvalue is None or isinstance(result.pvalue, float)
        assert isinstance(result.critical_values, dict)
        assert isinstance(result.null_hypothesis, str)
        assert isinstance(result.alternative_hypothesis, str)
        assert isinstance(result.reject_at_5pct, bool)
        assert result.lags_used is None or isinstance(result.lags_used, int)
        assert isinstance(result.additional_info, dict)

        # Summary and repr should not raise
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

        repr_str = repr(result)
        assert isinstance(repr_str, str)

    def test_adf_returns_valid_result(self, stationary_data: np.ndarray) -> None:
        result = adf_test(stationary_data)
        self._check_test_result(result)

    def test_pp_returns_valid_result(self, stationary_data: np.ndarray) -> None:
        result = pp_test(stationary_data)
        self._check_test_result(result)

    def test_kpss_returns_valid_result(self, stationary_data: np.ndarray) -> None:
        result = kpss_test(stationary_data)
        self._check_test_result(result)

    def test_ers_returns_valid_result(self, stationary_data: np.ndarray) -> None:
        result = ers_test(stationary_data)
        self._check_test_result(result)

    def test_za_returns_valid_result(self, stationary_data: np.ndarray) -> None:
        result = zivot_andrews_test(stationary_data, model="a")
        self._check_test_result(result)

    def test_ls_returns_valid_result(self, stationary_data: np.ndarray) -> None:
        result = lee_strazicich_test(stationary_data, breaks=1)
        self._check_test_result(result)

    def test_hegy_returns_valid_result(self) -> None:
        np.random.seed(42)
        y = np.random.randn(100)
        result = hegy_test(y, period=4)
        self._check_test_result(result)

    def test_eg_returns_valid_result(self) -> None:
        np.random.seed(42)
        x = np.cumsum(np.random.randn(200))
        y = 2 * x + np.random.randn(200) * 0.5
        result = engle_granger_test(y, x)
        self._check_test_result(result)

    def test_gh_returns_valid_result(self) -> None:
        np.random.seed(42)
        x = np.cumsum(np.random.randn(200))
        y = x + np.random.randn(200)
        result = gregory_hansen_test(y, x, model="c")
        self._check_test_result(result)

    def test_bounds_returns_valid_result(self) -> None:
        np.random.seed(42)
        x = np.cumsum(np.random.randn(200))
        y = x + np.random.randn(200) * 0.5
        result = bounds_test(y, x)
        self._check_test_result(result)

    def test_po_returns_valid_result(self) -> None:
        np.random.seed(42)
        x = np.cumsum(np.random.randn(200))
        y = x + np.random.randn(200) * 0.5
        result = phillips_ouliaris_test(y, x)
        self._check_test_result(result)

    def test_chow_returns_valid_result(self) -> None:
        np.random.seed(42)
        T = 200
        X = np.ones((T, 1))
        y = np.random.randn(T)
        result = chow_test(y, X, break_point=100)
        self._check_test_result(result)

    def test_bp_returns_valid_result(self) -> None:
        np.random.seed(42)
        T = 200
        X = np.ones((T, 1))
        y = np.concatenate([np.random.randn(100), np.random.randn(100) + 5])
        result = bai_perron_test(y, X, max_breaks=3)
        self._check_test_result(result)

    def test_cusum_returns_valid_result(self) -> None:
        np.random.seed(42)
        T = 200
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(T) * 0.5
        result = cusum_test(y, X)
        self._check_test_result(result)

    def test_cusumsq_returns_valid_result(self) -> None:
        np.random.seed(42)
        T = 200
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(T) * 0.5
        result = cusumsq_test(y, X)
        self._check_test_result(result)

    def test_qlr_returns_valid_result(self) -> None:
        np.random.seed(42)
        T = 200
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(T)
        result = qlr_test(y, X)
        self._check_test_result(result)

    def test_lb_returns_valid_result(self) -> None:
        result = ljung_box_test(np.random.randn(200), lags=10)
        self._check_test_result(result)

    def test_bg_returns_valid_result(self) -> None:
        np.random.seed(42)
        T = 200
        X = np.ones((T, 1))
        resid = np.random.randn(T)
        result = breusch_godfrey_test(resid, X, nlags=4)
        self._check_test_result(result)

    def test_dw_returns_valid_result(self) -> None:
        result = durbin_watson_test(np.random.randn(200))
        self._check_test_result(result)

    def test_bds_returns_valid_result(self) -> None:
        result = bds_test(np.random.randn(200), max_dim=3)
        self._check_test_result(result)

    def test_arch_returns_valid_result(self) -> None:
        result = arch_lm_test(np.random.randn(200), nlags=5)
        self._check_test_result(result)

    def test_white_returns_valid_result(self) -> None:
        np.random.seed(42)
        T = 200
        X = np.random.randn(T, 2)
        resid = np.random.randn(T)
        result = white_test(resid, X)
        self._check_test_result(result)

    def test_jb_returns_valid_result(self) -> None:
        result = jarque_bera_test(np.random.randn(200))
        self._check_test_result(result)

    def test_reset_returns_valid_result(self) -> None:
        np.random.seed(42)
        T = 200
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(T)
        result = reset_test(y, X)
        self._check_test_result(result)


class TestUnitRootWorkflow:
    """Integration: ADF, PP, KPSS should give consistent conclusions."""

    def test_stationary_series(self) -> None:
        """ADF/PP reject, KPSS does not reject for stationary AR(1)."""
        np.random.seed(42)
        T = 500
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + np.random.randn()

        adf = adf_test(y, regression="c")
        pp = pp_test(y, regression="c")
        kpss = kpss_test(y, regression="c")

        assert adf.reject_at_5pct, "ADF should reject for stationary"
        assert pp.reject_at_5pct, "PP should reject for stationary"
        assert not kpss.reject_at_5pct, "KPSS should not reject for stationary"

    def test_random_walk(self) -> None:
        """ADF/PP do not reject, KPSS rejects for random walk."""
        np.random.seed(42)
        y = np.cumsum(np.random.randn(500))

        adf = adf_test(y, regression="c")
        pp = pp_test(y, regression="c")
        kpss = kpss_test(y, regression="c")

        assert not adf.reject_at_5pct, "ADF should not reject for RW"
        assert not pp.reject_at_5pct, "PP should not reject for RW"
        assert kpss.reject_at_5pct, "KPSS should reject for RW"


class TestCointegrationWorkflow:
    """Integration: cointegration tests should agree on simulated DGP."""

    def test_cointegrated_pair(self) -> None:
        """All tests should detect cointegration in simulated DGP."""
        np.random.seed(42)
        T = 300
        x = np.cumsum(np.random.randn(T))
        y = 2.0 * x + np.random.randn(T) * 0.3

        eg = engle_granger_test(y, x)
        po = phillips_ouliaris_test(y, x)
        pss = bounds_test(y, x)

        assert eg.reject_at_5pct, "EG should detect cointegration"
        assert po.reject_at_5pct, "PO should detect cointegration"
        assert pss.reject_at_5pct, "PSS should detect cointegration"


class TestSpecificationWorkflow:
    """Integration: well-specified model should pass all diagnostic tests."""

    def test_well_specified_model(self) -> None:
        """All specification tests should not reject for well-specified model."""
        np.random.seed(42)
        T = 500
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(T) * 0.5

        # OLS
        beta = np.linalg.solve(X.T @ X, X.T @ y)
        resid = y - X @ beta

        # Diagnostic tests
        lb = ljung_box_test(resid, lags=10)
        jb = jarque_bera_test(resid)
        dw = durbin_watson_test(resid)
        arch = arch_lm_test(resid, nlags=5)
        w = white_test(resid, X[:, 1:])  # exclude constant
        rst = reset_test(y, X, power=3)

        # Well-specified model should pass all tests
        assert not lb.reject_at_5pct, "LB should not reject"
        assert not jb.reject_at_5pct, "JB should not reject"
        assert 1.5 < dw.statistic < 2.5, "DW should be near 2"
        assert not arch.reject_at_5pct, "ARCH should not reject"
        assert not w.reject_at_5pct, "White should not reject"
        assert not rst.reject_at_5pct, "RESET should not reject"


class TestStructuralBreakWorkflow:
    """Integration: break tests should detect breaks in simulated DGP."""

    def test_break_detection(self) -> None:
        """Chow, QLR should detect break in simulated DGP."""
        np.random.seed(42)
        T = 200
        bp = 100
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        y = np.zeros(T)
        y[:bp] = X[:bp] @ np.array([1.0, 2.0]) + np.random.randn(bp) * 0.5
        y[bp:] = X[bp:] @ np.array([5.0, -1.0]) + np.random.randn(T - bp) * 0.5

        chow_res = chow_test(y, X, break_point=bp)
        qlr_res = qlr_test(y, X, trim=0.15)

        assert chow_res.reject_at_5pct, "Chow should detect break"
        assert qlr_res.reject_at_5pct, "QLR should detect break"
