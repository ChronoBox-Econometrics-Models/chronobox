"""Tests for TestResult dataclass."""

from __future__ import annotations

import pytest

from chronobox.tests_stat.base import TestResult


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_creation(self) -> None:
        """Test basic creation of TestResult."""
        result = TestResult(
            test_name="ADF",
            statistic=-3.5,
            pvalue=0.01,
            critical_values={"1%": -3.43, "5%": -2.86, "10%": -2.57},
            null_hypothesis="Unit root",
            alternative_hypothesis="Stationary",
            reject_at_5pct=True,
        )
        assert result.test_name == "ADF"
        assert result.statistic == -3.5
        assert result.pvalue == 0.01
        assert result.reject_at_5pct is True
        assert result.lags_used is None
        assert result.additional_info == {}

    def test_creation_with_optionals(self) -> None:
        """Test TestResult with optional fields."""
        result = TestResult(
            test_name="ADF",
            statistic=-3.5,
            pvalue=0.01,
            critical_values={"1%": -3.43, "5%": -2.86, "10%": -2.57},
            null_hypothesis="Unit root",
            alternative_hypothesis="Stationary",
            reject_at_5pct=True,
            lags_used=4,
            additional_info={"regression": "c", "nobs": 100},
        )
        assert result.lags_used == 4
        assert result.additional_info["regression"] == "c"

    def test_summary_basic(self) -> None:
        """Test summary output."""
        result = TestResult(
            test_name="ADF",
            statistic=-3.5,
            pvalue=0.01,
            critical_values={"1%": -3.43, "5%": -2.86, "10%": -2.57},
            null_hypothesis="Unit root",
            alternative_hypothesis="Stationary",
            reject_at_5pct=True,
            lags_used=2,
        )
        summary = result.summary()
        assert "ADF Test" in summary
        assert "-3.5" in summary
        assert "Reject H0" in summary
        assert "Lags used" in summary

    def test_summary_no_pvalue(self) -> None:
        """Test summary when pvalue is None."""
        result = TestResult(
            test_name="KPSS",
            statistic=0.5,
            pvalue=None,
            critical_values={"1%": 0.739, "5%": 0.463, "10%": 0.347},
            null_hypothesis="Stationarity",
            alternative_hypothesis="Unit root",
            reject_at_5pct=True,
        )
        summary = result.summary()
        assert "N/A" in summary

    def test_repr(self) -> None:
        """Test string representation."""
        result = TestResult(
            test_name="ADF",
            statistic=-3.5,
            pvalue=0.01,
            critical_values={"1%": -3.43, "5%": -2.86, "10%": -2.57},
            null_hypothesis="Unit root",
            alternative_hypothesis="Stationary",
            reject_at_5pct=True,
        )
        r = repr(result)
        assert "TestResult" in r
        assert "ADF" in r
        assert "Reject H0" in r

    def test_repr_no_reject(self) -> None:
        """Test string representation when not rejecting."""
        result = TestResult(
            test_name="ADF",
            statistic=-1.5,
            pvalue=0.50,
            critical_values={"1%": -3.43, "5%": -2.86, "10%": -2.57},
            null_hypothesis="Unit root",
            alternative_hypothesis="Stationary",
            reject_at_5pct=False,
        )
        r = repr(result)
        assert "Fail to reject H0" in r

    def test_additional_info_in_summary(self) -> None:
        """Test that additional_info appears in summary."""
        result = TestResult(
            test_name="ZA",
            statistic=-4.5,
            pvalue=0.01,
            critical_values={"1%": -5.57, "5%": -5.08, "10%": -4.82},
            null_hypothesis="Unit root",
            alternative_hypothesis="Stationary with break",
            reject_at_5pct=True,
            additional_info={"break_date": 50, "model": "c"},
        )
        summary = result.summary()
        assert "break_date" in summary
        assert "50" in summary


class TestMacKinnonCriticalValues:
    """Tests for MacKinnon critical values module."""

    def test_adf_constant(self) -> None:
        """Test ADF critical values with constant."""
        from chronobox.tests_stat.critical_values.mackinnon import mackinnon_crit

        cv = mackinnon_crit(nobs=100, regression="c", test_type="adf")
        assert "1%" in cv
        assert "5%" in cv
        assert "10%" in cv
        # Standard ADF critical values for nobs=100 with constant
        assert cv["1%"] < cv["5%"] < cv["10%"] < 0  # all negative, ordered

    def test_adf_trend(self) -> None:
        """Test ADF critical values with constant+trend."""
        from chronobox.tests_stat.critical_values.mackinnon import mackinnon_crit

        cv = mackinnon_crit(nobs=200, regression="ct", test_type="adf")
        assert cv["1%"] < cv["5%"] < cv["10%"] < 0
        # With trend, critical values are more negative
        cv_c = mackinnon_crit(nobs=200, regression="c", test_type="adf")
        assert cv["5%"] < cv_c["5%"]

    def test_adf_no_constant(self) -> None:
        """Test ADF critical values without constant."""
        from chronobox.tests_stat.critical_values.mackinnon import mackinnon_crit

        cv = mackinnon_crit(nobs=100, regression="n", test_type="adf")
        assert cv["1%"] < cv["5%"] < cv["10%"] < 0

    def test_engle_granger(self) -> None:
        """Test Engle-Granger critical values."""
        from chronobox.tests_stat.critical_values.mackinnon import mackinnon_crit

        cv = mackinnon_crit(nobs=200, n_vars=2, test_type="eg")
        assert cv["1%"] < cv["5%"] < cv["10%"] < 0
        # EG critical values are more negative than ADF
        cv_adf = mackinnon_crit(nobs=200, regression="c", test_type="adf")
        assert cv["5%"] < cv_adf["5%"]

    def test_invalid_regression(self) -> None:
        """Test invalid regression model raises error."""
        from chronobox.tests_stat.critical_values.mackinnon import mackinnon_crit

        with pytest.raises(ValueError, match="Unsupported ADF model"):
            mackinnon_crit(nobs=100, regression="invalid", test_type="adf")

    def test_sample_size_effect(self) -> None:
        """Test that larger T gives values closer to asymptotic."""
        from chronobox.tests_stat.critical_values.mackinnon import mackinnon_crit

        cv_large = mackinnon_crit(nobs=10000, regression="c", test_type="adf")
        # For ADF constant, asymptotic 5% ~ -2.86
        assert abs(cv_large["5%"] - (-2.8621)) < 0.01

    def test_mackinnon_pvalue(self) -> None:
        """Test approximate p-value computation."""
        from chronobox.tests_stat.critical_values.mackinnon import mackinnon_pvalue

        # Very negative stat -> small p-value
        p1 = mackinnon_pvalue(-5.0, regression="c", nobs=200)
        assert p1 < 0.01

        # Stat near zero -> large p-value
        p2 = mackinnon_pvalue(-1.0, regression="c", nobs=200)
        assert p2 > 0.10


class TestOsterwaldLenum:
    """Tests for Osterwald-Lenum critical values."""

    def test_trace_cv(self) -> None:
        """Test trace critical values."""
        from chronobox.tests_stat.critical_values.osterwald_lenum import (
            johansen_trace_cv,
        )

        cv = johansen_trace_cv(n_vars_minus_rank=2, model="li")
        assert cv["10%"] < cv["5%"] < cv["1%"]

    def test_max_eigen_cv(self) -> None:
        """Test max-eigenvalue critical values."""
        from chronobox.tests_stat.critical_values.osterwald_lenum import (
            johansen_max_eigen_cv,
        )

        cv = johansen_max_eigen_cv(n_vars_minus_rank=2, model="li")
        assert cv["10%"] < cv["5%"] < cv["1%"]

    def test_invalid_rank(self) -> None:
        """Test invalid rank raises error."""
        from chronobox.tests_stat.critical_values.osterwald_lenum import (
            johansen_trace_cv,
        )

        with pytest.raises(ValueError):
            johansen_trace_cv(n_vars_minus_rank=20, model="li")


class TestPSSBounds:
    """Tests for PSS bounds critical values."""

    def test_f_bounds(self) -> None:
        """Test F-test bounds."""
        from chronobox.tests_stat.critical_values.pss_bounds import pss_f_bounds

        bounds = pss_f_bounds(k=1, case=3)
        assert "5%" in bounds
        lower, upper = bounds["5%"]
        assert lower < upper

    def test_t_bounds(self) -> None:
        """Test t-test bounds."""
        from chronobox.tests_stat.critical_values.pss_bounds import pss_t_bounds

        bounds = pss_t_bounds(k=1, case=3)
        lower, upper = bounds["5%"]
        assert lower > upper  # t-bounds are negative, lower is less negative

    def test_invalid_k(self) -> None:
        """Test invalid k raises error."""
        from chronobox.tests_stat.critical_values.pss_bounds import pss_f_bounds

        with pytest.raises(ValueError):
            pss_f_bounds(k=20, case=3)
