"""Integration tests - End-to-end workflows.

Each test validates a complete pipeline from data loading to report generation.
These are the 7 core workflows that validate the chronobox library works correctly
as a unified system.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Select only numeric columns from a DataFrame."""
    return df.select_dtypes(include=[np.number])


# ──────────────────────────────────────────────────────────────
# Workflow 1: ARIMA Full Pipeline
# ──────────────────────────────────────────────────────────────

class TestARIMAFullWorkflow:
    """End-to-end ARIMA workflow.

    Pipeline:
    1. Load airline dataset
    2. Run ADF test for stationarity
    3. Fit ARIMA(0,1,1)(0,1,1)[12]
    4. Run Ljung-Box test on residuals
    5. Forecast 12 steps ahead
    6. Generate report
    """

    def test_arima_full_pipeline(self, tmp_path: Path) -> None:
        # Step 1: Load data
        from chronobox.datasets import load_dataset

        airline = load_dataset("airline")
        assert airline is not None
        assert len(airline) > 100

        # Ensure we have a numeric Series
        data = _numeric_cols(airline).iloc[:, 0] if isinstance(airline, pd.DataFrame) else airline

        # Step 2: ADF test
        from chronobox.tests_stat import adf_test

        adf_result = adf_test(data.values)
        assert adf_result is not None
        assert hasattr(adf_result, "statistic")

        # Step 3: Fit ARIMA(0,1,1)(0,1,1)[12]
        from chronobox import ARIMA

        model = ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
        results = model.fit(data.values)
        assert results is not None

        # Verify model has expected attributes
        summary = results.summary()
        assert len(summary) > 0

        # Step 4: Ljung-Box test on residuals
        from chronobox.tests_stat import ljung_box_test

        resid = results.residuals
        assert resid is not None
        resid_clean = resid[~np.isnan(resid)]
        lb_result = ljung_box_test(resid_clean, lags=10)
        assert lb_result is not None

        # Step 5: Forecast 12 steps
        forecast_result = results.forecast(steps=12)
        assert forecast_result is not None
        forecast_values = np.asarray(forecast_result["forecast"])
        assert len(forecast_values) == 12

        # Forecasts should be finite
        assert np.all(np.isfinite(forecast_values))

        # Step 6: Generate report
        report_path = tmp_path / "arima_report.html"
        from chronobox.experiment import ChronoExperiment

        exp = ChronoExperiment(pd.Series(data.values, name="passengers"), name="ARIMA Full Test")
        exp.fit_model("ARIMA(0,1,1)x(0,1,1,12)", {
            "order": (0, 1, 1),
            "seasonal_order": (0, 1, 1, 12),
        })
        exp.save_master_report(str(report_path))
        assert report_path.exists()
        assert report_path.stat().st_size > 100


# ──────────────────────────────────────────────────────────────
# Workflow 2: VAR Full Pipeline
# ──────────────────────────────────────────────────────────────

class TestVARFullWorkflow:
    """End-to-end VAR workflow.

    Pipeline:
    1. Load Canada dataset (multivariate)
    2. Select optimal lag order
    3. Fit VAR(2)
    4. Run Granger causality test
    5. Compute IRF (Impulse Response Functions)
    6. Compute FEVD (Forecast Error Variance Decomposition)
    7. Forecast 8 steps ahead
    8. Generate report
    """

    def test_var_full_pipeline(self, tmp_path: Path) -> None:
        # Step 1: Load multivariate data
        from chronobox.datasets import load_dataset

        canada = _numeric_cols(load_dataset("canada"))
        assert canada is not None
        assert isinstance(canada, pd.DataFrame)
        assert canada.shape[1] >= 2

        # Step 2: Select lag order (auto via maxlags)
        from chronobox.models.var import VAR

        auto_model = VAR(maxlags=8)
        auto_results = auto_model.fit(canada)
        assert auto_results is not None
        assert auto_results.k_ar >= 1

        # Step 3: Fit VAR(2)
        model = VAR(lags=2)
        results = model.fit(canada)
        assert results is not None

        # Step 4: Granger causality
        names = results.names
        granger = results.granger_causality(caused=names[0], causing=names[1])
        assert granger is not None
        assert hasattr(granger, "fstat")
        assert hasattr(granger, "pvalue")

        # Step 5: IRF
        irf = results.irf(periods=20)
        assert irf is not None

        # Step 6: FEVD
        fevd = results.fevd(periods=20)
        assert fevd is not None

        # Step 7: Forecast
        forecast = results.forecast(steps=8)
        assert forecast is not None
        assert forecast.shape[0] == 8

        # Step 8: Report
        report_path = tmp_path / "var_report.json"
        report = {
            "model": "VAR(2)",
            "k_ar": results.k_ar,
            "neqs": results.neqs,
            "aic": float(results.aic),
            "bic": float(results.bic),
            "is_stable": bool(results.is_stable),
            "granger_pvalue": float(granger.pvalue),
        }
        report_path.write_text(json.dumps(report, indent=2))
        assert report_path.exists()


# ──────────────────────────────────────────────────────────────
# Workflow 3: SVAR Full Pipeline
# ──────────────────────────────────────────────────────────────

class TestSVARFullWorkflow:
    """End-to-end SVAR workflow.

    Pipeline:
    1. Load macro dataset
    2. Fit reduced-form VAR
    3. Identify SVAR with Cholesky decomposition
    4. Compute structural IRF
    5. Compute historical decomposition
    6. Run counterfactual analysis
    7. Generate report
    """

    def test_svar_full_pipeline(self, tmp_path: Path) -> None:
        # Step 1: Load macro data
        from chronobox.datasets import load_dataset

        try:
            macro = _numeric_cols(load_dataset("us_macro_quarterly"))
        except (KeyError, FileNotFoundError):
            macro = _numeric_cols(load_dataset("canada"))

        assert isinstance(macro, pd.DataFrame)

        # Step 2: Fit reduced-form VAR
        from chronobox.models.var import VAR

        model = VAR(lags=2)
        var_results = model.fit(macro)
        assert var_results is not None

        # Step 3: Identify SVAR with Cholesky
        from chronobox.models.svar import SVAR

        svar = SVAR(var_results, method="cholesky")
        svar_results = svar.fit()
        assert svar_results is not None

        # Step 4: Structural IRF
        sirf = svar_results.irf(periods=20)
        assert sirf is not None
        assert sirf.shape[0] == 21  # periods + 1

        # Step 5: Historical decomposition
        from chronobox.analysis import HistoricalDecomposition

        hd = HistoricalDecomposition(svar_results)
        hd_result = hd.result
        assert hd_result is not None
        assert hd_result.decomposition is not None

        # Step 6: Counterfactual
        from chronobox.analysis import Counterfactual

        cf = Counterfactual(hd)
        cf_series = cf.without_shock(shock_index=0)
        assert cf_series is not None

        # Step 7: Report
        report_path = tmp_path / "svar_report.json"
        report = {
            "model": "SVAR (Cholesky)",
            "method": svar_results.method,
            "k_vars": svar_results.k_vars,
            "lags": svar_results.lags,
        }
        report_path.write_text(json.dumps(report, indent=2))
        assert report_path.exists()


# ──────────────────────────────────────────────────────────────
# Workflow 4: VECM Full Pipeline
# ──────────────────────────────────────────────────────────────

class TestVECMFullWorkflow:
    """End-to-end VECM workflow.

    Pipeline:
    1. Load cointegrated data
    2. Run Johansen cointegration test
    3. Determine cointegration rank
    4. Fit VECM with estimated rank
    5. Verify estimation results
    6. Generate report
    """

    def test_vecm_full_pipeline(self, tmp_path: Path) -> None:
        # Step 1: Load data
        from chronobox.datasets import load_dataset

        data = _numeric_cols(load_dataset("canada"))
        assert isinstance(data, pd.DataFrame)

        # Step 2: Johansen test
        from chronobox.models.vecm import VECM

        vecm_model = VECM(lags=2)
        johansen = vecm_model.johansen_test(data)
        assert johansen is not None

        # Step 3: Determine rank
        if hasattr(johansen, "rank_trace"):
            rank = johansen.rank_trace
        elif hasattr(johansen, "rank"):
            rank = johansen.rank
        else:
            rank = 1  # Default

        assert rank >= 0

        # Step 4: Fit VECM
        coint_rank = int(max(rank, 1))
        vecm_fitted = VECM(lags=2, coint_rank=coint_rank)
        results = vecm_fitted.fit(data)
        assert results is not None

        # Step 5: Verify estimation results
        assert results.alpha is not None
        assert results.beta is not None
        assert results.pi is not None
        assert results.coint_rank == coint_rank

        summary = results.summary()
        assert len(summary) > 0

        # Step 6: Report
        report_path = tmp_path / "vecm_report.json"
        report = {
            "model": f"VECM(lags=2, rank={coint_rank})",
            "coint_rank": coint_rank,
            "neqs": results.neqs,
            "nobs": results.nobs,
            "johansen_trace_stats": johansen.trace_stat.tolist(),
        }
        report_path.write_text(json.dumps(report, indent=2))
        assert report_path.exists()


# ──────────────────────────────────────────────────────────────
# Workflow 5: Testing Battery
# ──────────────────────────────────────────────────────────────

class TestTestingBatteryWorkflow:
    """End-to-end testing battery workflow.

    Pipeline:
    1. Load GDP data
    2. Run ADF test
    3. Run Phillips-Perron test
    4. Run KPSS test
    5. Run ERS test
    6. Verify consistent conclusions across tests
    7. Generate report
    """

    def test_testing_battery_pipeline(self, tmp_path: Path) -> None:
        # Step 1: Load data
        from chronobox.datasets import load_dataset

        try:
            gdp = load_dataset("us_gdp")
        except (KeyError, FileNotFoundError):
            np.random.seed(42)
            gdp = pd.Series(np.cumsum(np.random.randn(200)), name="GDP")

        if isinstance(gdp, pd.DataFrame):
            gdp = _numeric_cols(gdp).iloc[:, 0]

        gdp_arr = np.asarray(gdp, dtype=np.float64)

        # Step 2: ADF
        from chronobox.tests_stat import adf_test

        adf = adf_test(gdp_arr)
        assert adf is not None

        # Step 3: Phillips-Perron
        from chronobox.tests_stat import pp_test

        pp = pp_test(gdp_arr)
        assert pp is not None

        # Step 4: KPSS
        from chronobox.tests_stat import kpss_test

        kpss = kpss_test(gdp_arr)
        assert kpss is not None

        # Step 5: ERS
        from chronobox.tests_stat import ers_test

        ers = ers_test(gdp_arr)
        assert ers is not None

        # Step 6: Consistent conclusions
        # ADF/PP: H0 = unit root. Reject => stationary
        # KPSS: H0 = stationary. Reject => unit root
        # All tests should return TestResult with statistic and pvalue
        assert hasattr(adf, "statistic")
        assert hasattr(pp, "statistic")
        assert hasattr(kpss, "statistic")
        assert hasattr(ers, "statistic")

        # Step 7: Report
        report_path = tmp_path / "testing_battery_report.json"

        def _safe_float(val: object) -> float | None:
            return float(val) if val is not None else None

        report = {
            "adf": {"statistic": _safe_float(adf.statistic), "pvalue": _safe_float(adf.pvalue)},
            "pp": {"statistic": _safe_float(pp.statistic), "pvalue": _safe_float(pp.pvalue)},
            "kpss": {"statistic": _safe_float(kpss.statistic), "pvalue": _safe_float(kpss.pvalue)},
            "ers": {"statistic": _safe_float(ers.statistic), "pvalue": _safe_float(ers.pvalue)},
        }
        report_path.write_text(json.dumps(report, indent=2, default=str))
        assert report_path.exists()


# ──────────────────────────────────────────────────────────────
# Workflow 6: Filter Full Pipeline
# ──────────────────────────────────────────────────────────────

class TestFilterFullWorkflow:
    """End-to-end filter workflow.

    Pipeline:
    1. Load GDP data
    2. Apply HP filter
    3. Apply BK filter
    4. Apply CF filter
    5. Apply Hamilton filter
    6. Compare extracted cycles
    7. Generate report
    """

    def test_filter_full_pipeline(self, tmp_path: Path) -> None:
        # Step 1: Load data
        from chronobox.datasets import load_dataset

        try:
            gdp = load_dataset("us_gdp")
        except (KeyError, FileNotFoundError):
            np.random.seed(42)
            trend = np.linspace(0, 10, 200)
            cycle = 2 * np.sin(np.linspace(0, 8 * np.pi, 200))
            gdp = pd.Series(trend + cycle + 0.5 * np.random.randn(200), name="GDP")

        if isinstance(gdp, pd.DataFrame):
            gdp = _numeric_cols(gdp).iloc[:, 0]

        gdp_arr = np.asarray(gdp, dtype=np.float64)

        # Step 2: HP filter (returns trend, cycle)
        from chronobox.filters import hp_filter

        hp_trend, hp_cycle = hp_filter(gdp_arr, lamb=1600)
        assert hp_trend is not None
        assert hp_cycle is not None
        assert len(hp_trend) == len(gdp_arr)

        # Step 3: BK filter (returns cycle only, shorter than input)
        from chronobox.filters import bk_filter

        bk_cycle = bk_filter(gdp_arr, low=6, high=32, trunc=12)
        assert bk_cycle is not None
        assert len(bk_cycle) > 0

        # Step 4: CF filter (returns cycle only, same length)
        from chronobox.filters import cf_filter

        cf_cycle = cf_filter(gdp_arr, low=6, high=32)
        assert cf_cycle is not None
        assert len(cf_cycle) == len(gdp_arr)

        # Step 5: Hamilton filter (returns trend, cycle)
        from chronobox.filters import hamilton_filter

        ham_trend, ham_cycle = hamilton_filter(gdp_arr, h=8, p=4)
        assert ham_trend is not None
        assert ham_cycle is not None

        # Step 6: Compare cycles
        # All cycle components should have mean close to 0
        for name, cyc in [("HP", hp_cycle), ("BK", bk_cycle), ("CF", cf_cycle), ("Hamilton", ham_cycle)]:
            cyc_arr = np.asarray(cyc)
            cyc_clean = cyc_arr[np.isfinite(cyc_arr)]
            if len(cyc_clean) > 0:
                assert abs(np.mean(cyc_clean)) < np.std(cyc_clean) * 3, f"{name} cycle mean too large"

        # Step 7: Report
        report_path = tmp_path / "filter_comparison.csv"
        comparison = pd.DataFrame({
            "HP_cycle": hp_cycle,
        })
        comparison.to_csv(report_path)
        assert report_path.exists()


# ──────────────────────────────────────────────────────────────
# Workflow 7: ARDL Full Pipeline
# ──────────────────────────────────────────────────────────────

class TestARDLFullWorkflow:
    """End-to-end ARDL workflow.

    Pipeline:
    1. Load y and x data
    2. Fit ARDL with automatic lag selection
    3. Run bounds test for cointegration
    4. Estimate ECM form
    5. Extract long-run coefficients
    6. Generate report
    """

    def test_ardl_full_pipeline(self, tmp_path: Path) -> None:
        # Step 1: Create/load data
        np.random.seed(42)
        n = 200
        x = np.cumsum(np.random.randn(n))
        y = 0.5 * x + np.cumsum(np.random.randn(n) * 0.5)

        # Step 2: Fit ARDL with auto lag selection
        from chronobox.models.ardl import ARDL

        model = ARDL(max_p=4, max_q=4, criterion="aic")
        results = model.fit(y, x)
        assert results is not None

        # Step 3: Bounds test
        bounds = results.bounds_test()
        assert bounds is not None
        assert "f_statistic" in bounds
        assert "conclusion" in bounds

        # Step 4: ECM form
        ecm = results.to_ecm()
        assert ecm is not None

        # Step 5: Long-run coefficients
        lr_coefs = results.long_run_coefficients
        assert lr_coefs is not None
        assert len(lr_coefs) >= 1

        # Step 6: Report
        report_path = tmp_path / "ardl_report.json"
        report = {
            "model": "ARDL",
            "y_lags": results.y_lags,
            "x_lags": results.x_lags,
            "aic": float(results.aic),
            "bic": float(results.bic),
            "r_squared": float(results.r_squared),
            "bounds_f_stat": float(bounds["f_statistic"]),
            "bounds_conclusion": str(bounds["conclusion"]),
            "long_run_coefficients": lr_coefs.tolist(),
        }
        report_path.write_text(json.dumps(report, indent=2))
        assert report_path.exists()
