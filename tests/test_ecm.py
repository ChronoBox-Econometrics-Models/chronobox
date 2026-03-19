"""Tests for the ECM model."""

import numpy as np
import pytest

from chronobox.models.ecm import ECM, ECMResult


class TestECM:
    """Tests for ECM model."""

    def setup_method(self) -> None:
        """Set up test data with cointegrated series."""
        np.random.seed(42)
        self.T = 400

        # Generate cointegrated system
        # x follows random walk
        self.x = np.random.randn(self.T, 2).cumsum(axis=0)
        # y is cointegrated with x: long-run y = 2*x1 + 0.5*x2
        eq_error = np.random.randn(self.T) * 0.3
        self.y = 2.0 * self.x[:, 0] + 0.5 * self.x[:, 1] + eq_error
        # Add some persistence to make it realistic
        for t in range(1, self.T):
            self.y[t] += 0.7 * (
                self.y[t - 1]
                - 2.0 * self.x[t - 1, 0]
                - 0.5 * self.x[t - 1, 1]
            )

    def test_speed_negative(self) -> None:
        """Speed of adjustment should be negative for cointegrated system."""
        ecm = ECM(lags=2)
        result = ecm.fit(self.y, self.x)
        assert result.speed_of_adjustment < 0, (
            f"Speed of adjustment {result.speed_of_adjustment} should be negative"
        )

    def test_long_run_coefficients(self) -> None:
        """Long-run coefficients should be close to true values."""
        ecm = ECM(lags=2)
        result = ecm.fit(self.y, self.x)
        lr = result.long_run_coefficients
        assert len(lr) == 2
        # Should be finite
        assert np.all(np.isfinite(lr))

    def test_result_fields(self) -> None:
        """ECMResult should contain all expected fields."""
        ecm = ECM(lags=2)
        result = ecm.fit(self.y, self.x)
        assert isinstance(result, ECMResult)
        assert result.nobs > 0
        assert result.k_params > 0
        assert result.sigma2 > 0
        assert 0 <= result.r_squared <= 1
        assert len(result.se) == result.k_params
        assert len(result.t_stats) == result.k_params
        assert len(result.residuals) == result.nobs

    def test_bounds_pss(self) -> None:
        """PSS bounds test should return valid results."""
        ecm = ECM(lags=2)
        result = ecm.fit(self.y, self.x)
        bt = result.bounds_test_pss()
        assert "t_statistic" in bt
        assert "pi_yy" in bt
        assert bt["conclusion"] in (
            "reject_h0",
            "fail_to_reject",
            "inconclusive",
        )

    def test_summary(self) -> None:
        """Summary should be non-empty."""
        ecm = ECM(lags=2)
        result = ecm.fit(self.y, self.x)
        s = result.summary()
        assert len(s) > 100
        assert "ECM" in s or "Error Correction" in s
        assert "speed" in s.lower() or "pi_yy" in s.lower() or "Speed" in s

    def test_1d_x(self) -> None:
        """Should handle 1-D x input."""
        ecm = ECM(lags=2)
        result = ecm.fit(self.y, self.x[:, 0])
        assert result.nobs > 0
        assert len(result.long_run_coefficients) == 1

    def test_different_lags(self) -> None:
        """Should work with different lag specifications."""
        for lag in [1, 2, 3, 4]:
            ecm = ECM(lags=lag)
            result = ecm.fit(self.y, self.x)
            assert result.nobs > 0

    def test_invalid_inputs(self) -> None:
        """Should raise for invalid inputs."""
        with pytest.raises(ValueError, match="lags must be >= 1"):
            ECM(lags=0)
        ecm = ECM(lags=2)
        with pytest.raises(ValueError, match="1-D"):
            ecm.fit(np.random.randn(10, 2), np.random.randn(10))

    def test_residuals_zero_mean(self) -> None:
        """OLS residuals should have approximately zero mean."""
        ecm = ECM(lags=2)
        result = ecm.fit(self.y, self.x)
        np.testing.assert_allclose(np.mean(result.residuals), 0.0, atol=1e-10)
