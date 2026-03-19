"""Tests for TimeSeriesResults."""

from __future__ import annotations

import numpy as np
import pytest

from chronobox.core.results import TimeSeriesResults


@pytest.fixture
def mock_results() -> TimeSeriesResults:
    """Create a mock TimeSeriesResults for testing."""
    params = np.array([0.5, -0.3, 100.0])
    param_names = ["phi_1", "theta_1", "sigma2"]
    se = np.array([0.1, 0.08, 15.0])
    residuals = np.random.default_rng(42).normal(0, 10, 100)
    fitted = np.cumsum(np.random.default_rng(42).normal(0, 1, 100))
    return TimeSeriesResults(
        params=params,
        param_names=param_names,
        se=se,
        loglike=-500.0,
        nobs=100,
        nobs_effective=98,
        residuals=residuals,
        fitted_values=fitted,
        model_name="ARIMA(1,1,1)",
    )


class TestInformationCriteria:
    """Test AIC, BIC, AICc, HQIC computation."""

    def test_aic(self, mock_results: TimeSeriesResults) -> None:
        # AIC = -2 * loglike + 2 * k = -2 * (-500) + 2 * 3 = 1006
        np.testing.assert_almost_equal(mock_results.aic, 1006.0)

    def test_bic(self, mock_results: TimeSeriesResults) -> None:
        # BIC = -2 * loglike + k * log(n) = 1000 + 3 * log(98)
        expected = 1000.0 + 3 * np.log(98)
        np.testing.assert_almost_equal(mock_results.bic, expected)

    def test_aicc(self, mock_results: TimeSeriesResults) -> None:
        # AICc = AIC + 2k(k+1)/(n-k-1) = 1006 + 2*3*4/(98-3-1)
        expected = 1006.0 + 24.0 / 94.0
        np.testing.assert_almost_equal(mock_results.aicc, expected)

    def test_hqic(self, mock_results: TimeSeriesResults) -> None:
        # HQIC = -2*loglike + 2*k*log(log(n)) = 1000 + 6*log(log(98))
        expected = 1000.0 + 6.0 * np.log(np.log(98))
        np.testing.assert_almost_equal(mock_results.hqic, expected)


class TestTvaluesPvalues:
    """Test t-values and p-values computation."""

    def test_tvalues(self, mock_results: TimeSeriesResults) -> None:
        expected = np.array([0.5 / 0.1, -0.3 / 0.08, 100.0 / 15.0])
        np.testing.assert_array_almost_equal(mock_results.tvalues, expected)

    def test_pvalues_within_range(self, mock_results: TimeSeriesResults) -> None:
        assert all(0 <= p <= 1 for p in mock_results.pvalues)

    def test_zero_se_gives_zero_tvalue(self) -> None:
        res = TimeSeriesResults(
            params=np.array([1.0]),
            param_names=["x"],
            se=np.array([0.0]),
            loglike=-10.0,
            nobs=50,
            nobs_effective=50,
            residuals=np.zeros(50),
            fitted_values=np.zeros(50),
            model_name="test",
        )
        assert res.tvalues[0] == 0.0
        assert res.pvalues[0] == 1.0


class TestSummary:
    """Test summary generation."""

    def test_summary_string(self, mock_results: TimeSeriesResults) -> None:
        s = mock_results.summary()
        assert isinstance(s, str)
        assert "ARIMA(1,1,1)" in s
        assert "phi_1" in s
        assert "theta_1" in s
        assert "Log-Likelihood" in s
        assert "AIC" in s

    def test_summary_no_error(self, mock_results: TimeSeriesResults) -> None:
        """summary() should not raise."""
        mock_results.summary()


class TestToDataframe:
    """Test DataFrame conversion."""

    def test_to_dataframe(self, mock_results: TimeSeriesResults) -> None:
        df = mock_results.to_dataframe()
        assert list(df.index) == ["phi_1", "theta_1", "sigma2"]
        assert "estimate" in df.columns
        assert "std_err" in df.columns
        assert "t_value" in df.columns
        assert "p_value" in df.columns


class TestPlotDiagnostics:
    """Test plot diagnostics."""

    def test_plot_no_error(self, mock_results: TimeSeriesResults) -> None:
        """plot_diagnostics() should not raise."""
        import matplotlib

        matplotlib.use("Agg")
        fig = mock_results.plot_diagnostics()
        assert fig is not None


class TestSaveLoad:
    """Test save/load roundtrip."""

    def test_roundtrip(self, mock_results: TimeSeriesResults, tmp_path: object) -> None:
        import pathlib

        path = pathlib.Path(str(tmp_path)) / "results.pkl"
        mock_results.save(str(path))
        loaded = TimeSeriesResults.load(str(path))
        assert loaded.model_name == mock_results.model_name
        np.testing.assert_array_equal(loaded.params, mock_results.params)
        np.testing.assert_almost_equal(loaded.aic, mock_results.aic)
        np.testing.assert_almost_equal(loaded.bic, mock_results.bic)
