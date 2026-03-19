"""Tests for the ARDL model."""

import numpy as np
import pytest

from chronobox.models.ardl import ARDL, ARDLResult


class TestARDL:
    """Tests for ARDL model."""

    def setup_method(self) -> None:
        """Set up test data."""
        np.random.seed(42)
        self.T = 300
        # Cointegrated system
        self.x = np.random.randn(self.T, 2).cumsum(axis=0)
        self.y = (
            0.5 * self.x[:, 0]
            + 0.3 * self.x[:, 1]
            + np.random.randn(self.T) * 0.5
        )
        self.y = self.y.cumsum()

    def test_ols_correct(self) -> None:
        """OLS estimation should produce valid results."""
        ardl = ARDL(max_p=2, max_q=2)
        result = ardl.fit(self.y, self.x, p=2, x_lags=[1, 1])
        assert isinstance(result, ARDLResult)
        assert result.nobs > 0
        assert result.r_squared >= 0
        assert result.r_squared <= 1.0
        assert len(result.coefficients) == result.k_params
        assert len(result.residuals) == result.nobs

    def test_fitted_plus_residual(self) -> None:
        """Fitted values + residuals should reconstruct dependent variable."""
        ardl = ARDL(max_p=2, max_q=2)
        result = ardl.fit(self.y, self.x, p=2, x_lags=[1, 1])
        # Residuals should have zero mean (OLS property)
        np.testing.assert_allclose(np.mean(result.residuals), 0.0, atol=1e-10)

    def test_auto_select(self) -> None:
        """Automatic lag selection should choose a model."""
        ardl = ARDL(max_p=3, max_q=3, criterion="aic")
        result = ardl.fit(self.y, self.x)
        assert result.y_lags >= 1
        assert result.y_lags <= 3
        for q in result.x_lags:
            assert q >= 0
            assert q <= 3

    def test_auto_select_bic(self) -> None:
        """BIC selection should also work."""
        ardl = ARDL(max_p=3, max_q=3, criterion="bic")
        result = ardl.fit(self.y, self.x)
        assert result.y_lags >= 1

    def test_bounds_test(self) -> None:
        """Bounds test should return valid F-statistic."""
        ardl = ARDL(max_p=2, max_q=2)
        result = ardl.fit(self.y, self.x, p=2, x_lags=[1, 1])
        bt = result.bounds_test()
        assert "f_statistic" in bt
        assert float(bt["f_statistic"]) >= 0
        assert bt["conclusion"] in (
            "reject_h0",
            "fail_to_reject",
            "inconclusive",
        )

    def test_to_ecm(self) -> None:
        """Conversion to ECM should work."""
        ardl = ARDL(max_p=2, max_q=2)
        result = ardl.fit(self.y, self.x, p=2, x_lags=[1, 1])
        ecm_result = result.to_ecm()
        assert ecm_result.speed_of_adjustment is not None

    def test_long_run_coefficients(self) -> None:
        """Long-run coefficients should be computed."""
        ardl = ARDL(max_p=2, max_q=2)
        result = ardl.fit(self.y, self.x, p=2, x_lags=[1, 1])
        lr = result.long_run_coefficients
        assert len(lr) == 2  # two x variables

    def test_summary(self) -> None:
        """Summary should return a non-empty string."""
        ardl = ARDL(max_p=2, max_q=2)
        result = ardl.fit(self.y, self.x, p=2, x_lags=[1, 1])
        s = result.summary()
        assert len(s) > 100
        assert "ARDL" in s
        assert "R-squared" in s

    def test_1d_x(self) -> None:
        """Should handle 1-D x input."""
        ardl = ARDL(max_p=2, max_q=2)
        result = ardl.fit(self.y, self.x[:, 0], p=2, x_lags=[1])
        assert result.nobs > 0

    def test_invalid_inputs(self) -> None:
        """Should raise for invalid inputs."""
        with pytest.raises(ValueError):
            ARDL(max_p=0)
        with pytest.raises(ValueError):
            ARDL(max_q=-1)
        with pytest.raises(ValueError):
            ARDL(criterion="invalid")  # type: ignore[arg-type]

    def test_standard_errors(self) -> None:
        """Standard errors should be positive."""
        ardl = ARDL(max_p=2, max_q=2)
        result = ardl.fit(self.y, self.x, p=2, x_lags=[1, 1])
        assert np.all(result.se > 0)
        assert np.all(np.isfinite(result.t_stats))
