"""Tests for ARFIMA model."""

from __future__ import annotations

import numpy as np
import pytest

from chronobox.models.arfima import (
    ARFIMA,
    ARFIMAResults,
    estimate_d_gph,
    estimate_d_local_whittle,
    fractional_diff,
    fractional_diff_coefficients,
    simulate_arfima,
)


class TestFractionalDiffCoefficients:
    """Tests for fractional differencing coefficients."""

    def test_d_zero_identity(self) -> None:
        """When d=0, pi_0=1 and all other pi_k=0."""
        pi = fractional_diff_coefficients(0.0, 10)
        assert pi[0] == 1.0
        np.testing.assert_allclose(pi[1:], 0.0, atol=1e-15)

    def test_d_one_first_diff(self) -> None:
        """When d=1, pi = [1, -1, 0, 0, ...]."""
        pi = fractional_diff_coefficients(1.0, 5)
        expected = np.array([1.0, -1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(pi, expected, atol=1e-15)

    def test_d_positive_decay(self) -> None:
        """For d=0.4, coefficients decay hyperbolically."""
        pi = fractional_diff_coefficients(0.4, 100)
        assert pi[0] == 1.0
        # Coefficients should alternate in sign and decay
        # |pi_k| should decay as k^{-d-1} = k^{-1.4}
        for k in range(10, 100):
            assert abs(pi[k]) < abs(pi[k - 1]) or abs(pi[k]) < 0.1

    def test_d_half_known_values(self) -> None:
        """Verify some known coefficient values for d=0.5."""
        pi = fractional_diff_coefficients(0.5, 5)
        # pi_0 = 1
        # pi_1 = (0 - 0.5) / 1 = -0.5
        # pi_2 = -0.5 * (1 - 0.5) / 2 = -0.5 * 0.25 = -0.125
        assert pi[0] == pytest.approx(1.0)
        assert pi[1] == pytest.approx(-0.5)
        assert pi[2] == pytest.approx(-0.125)

    def test_coefficients_length(self) -> None:
        """Output length matches requested n."""
        for n in [1, 5, 50, 200]:
            pi = fractional_diff_coefficients(0.3, n)
            assert len(pi) == n


class TestFractionalDiff:
    """Tests for fractional differencing of a series."""

    def test_d_zero_no_change(self) -> None:
        """Fractional diff with d=0 returns the original series."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = fractional_diff(y, 0.0)
        np.testing.assert_allclose(w, y, atol=1e-12)

    def test_d_one_is_first_diff(self) -> None:
        """Fractional diff with d=1 equals standard first difference."""
        y = np.array([10.0, 12.0, 15.0, 11.0, 14.0, 20.0])
        w = fractional_diff(y, 1.0)
        expected = np.diff(y, prepend=0)
        expected[0] = y[0]  # first value is just y[0]
        np.testing.assert_allclose(w, expected, atol=1e-10)

    def test_output_length(self) -> None:
        """Output has same length as input."""
        y = np.random.default_rng(42).normal(size=100)
        w = fractional_diff(y, 0.3)
        assert len(w) == len(y)


class TestEstimateDGPH:
    """Tests for GPH d estimation."""

    def test_gph_d_estimation(self) -> None:
        """GPH should estimate d close to true value for simulated ARFIMA(0,d,0)."""
        rng = np.random.default_rng(42)
        d_true = 0.4
        y = simulate_arfima(n=2000, d=d_true, rng=rng)
        d_hat, se = estimate_d_gph(y, bandwidth_exp=0.5)
        assert abs(d_hat - d_true) < 0.15, f"GPH d_hat={d_hat}, expected ~{d_true}"

    def test_gph_d_zero(self) -> None:
        """For white noise (d=0), GPH should estimate d near 0."""
        rng = np.random.default_rng(123)
        y = rng.normal(size=2000)
        d_hat, se = estimate_d_gph(y, bandwidth_exp=0.5)
        assert abs(d_hat) < 0.15, f"GPH d_hat={d_hat}, expected ~0"

    def test_gph_returns_se(self) -> None:
        """GPH should return a positive standard error."""
        rng = np.random.default_rng(42)
        y = rng.normal(size=500)
        d_hat, se = estimate_d_gph(y)
        assert se > 0


class TestEstimateDLocalWhittle:
    """Tests for Local Whittle d estimation."""

    def test_whittle_d_estimation(self) -> None:
        """Local Whittle should estimate d close to true value."""
        rng = np.random.default_rng(42)
        d_true = 0.3
        y = simulate_arfima(n=2000, d=d_true, rng=rng)
        d_hat, se = estimate_d_local_whittle(y, bandwidth_exp=0.65)
        assert abs(d_hat - d_true) < 0.15, f"Whittle d_hat={d_hat}, expected ~{d_true}"

    def test_whittle_returns_positive_se(self) -> None:
        """Local Whittle should return a positive standard error."""
        rng = np.random.default_rng(42)
        y = rng.normal(size=500)
        d_hat, se = estimate_d_local_whittle(y)
        assert se > 0


class TestARFIMA:
    """Tests for the ARFIMA model class."""

    def test_arfima_d_zero_is_arma(self) -> None:
        """ARFIMA(1,0,0) with d=0 should be equivalent to ARIMA(1,0,0)."""
        rng = np.random.default_rng(42)
        # Simulate AR(1) process
        n = 500
        phi = 0.6
        eps = rng.normal(size=n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + eps[t]

        model = ARFIMA(order=(1, 0.0, 0))
        results = model.fit(y, method="css")

        # d should remain 0
        assert results.d == 0.0
        # AR parameter should be close to true phi
        assert abs(results.ar_params[0] - phi) < 0.1, (
            f"AR param={results.ar_params[0]}, expected ~{phi}"
        )

    def test_arfima_fit_returns_results(self) -> None:
        """fit() should return an ARFIMAResults instance."""
        rng = np.random.default_rng(42)
        y = simulate_arfima(n=300, d=0.3, rng=rng)
        model = ARFIMA(order=(0, 0.3, 0))
        results = model.fit(y)
        assert isinstance(results, ARFIMAResults)

    def test_arfima_results_attributes(self) -> None:
        """Results should have all expected attributes."""
        rng = np.random.default_rng(42)
        y = simulate_arfima(n=200, d=0.2, rng=rng)
        model = ARFIMA(order=(1, 0.2, 1))
        results = model.fit(y)

        assert hasattr(results, "d")
        assert hasattr(results, "ar_params")
        assert hasattr(results, "ma_params")
        assert hasattr(results, "sigma2")
        assert hasattr(results, "resid")
        assert hasattr(results, "loglik")
        assert hasattr(results, "aic")
        assert hasattr(results, "bic")
        assert hasattr(results, "aicc")
        assert len(results.ar_params) == 1
        assert len(results.ma_params) == 1
        assert results.sigma2 > 0

    def test_arfima_summary(self) -> None:
        """summary() should return a non-empty string."""
        rng = np.random.default_rng(42)
        y = rng.normal(size=200)
        model = ARFIMA(order=(1, 0.0, 0))
        results = model.fit(y)
        s = results.summary()
        assert isinstance(s, str)
        assert "ARFIMA" in s
        assert "Log-Likelihood" in s

    def test_arfima_estimate_d_method(self) -> None:
        """estimate_d() should return a float."""
        rng = np.random.default_rng(42)
        y = simulate_arfima(n=1000, d=0.35, rng=rng)
        model = ARFIMA(order=(0, 0.0, 0))
        d_hat = model.estimate_d(y, method="gph")
        assert isinstance(d_hat, float)
        assert -0.5 < d_hat < 0.5

    def test_arfima_forecast(self) -> None:
        """forecast() should return array of correct length."""
        rng = np.random.default_rng(42)
        y = simulate_arfima(n=200, d=0.2, rng=rng)
        model = ARFIMA(order=(1, 0.2, 0))
        results = model.fit(y)
        fc = results.forecast(steps=10)
        assert len(fc) == 10
        assert np.all(np.isfinite(fc))

    def test_arfima_fit_with_estimate_d(self) -> None:
        """fit() with estimate_d=True should estimate d."""
        rng = np.random.default_rng(42)
        d_true = 0.3
        y = simulate_arfima(n=500, d=d_true, rng=rng)
        model = ARFIMA(order=(0, 0.0, 0))
        results = model.fit(y, estimate_d=True)
        # d should be estimated close to true value
        assert abs(results.d - d_true) < 0.2, (
            f"Estimated d={results.d}, expected ~{d_true}"
        )

    def test_arfima_invalid_order(self) -> None:
        """Invalid order should raise ValueError."""
        with pytest.raises(ValueError, match="order must have 3 elements"):
            ARFIMA(order=(1, 0.5))  # type: ignore[arg-type]

    def test_arfima_negative_p_q(self) -> None:
        """Negative p or q should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            ARFIMA(order=(-1, 0.0, 0))


class TestSimulateARFIMA:
    """Tests for ARFIMA simulation."""

    def test_simulate_length(self) -> None:
        """Simulated series should have correct length."""
        y = simulate_arfima(n=100, d=0.3, rng=np.random.default_rng(42))
        assert len(y) == 100

    def test_simulate_d_zero_is_wn(self) -> None:
        """ARFIMA(0,0,0) should produce white noise."""
        rng = np.random.default_rng(42)
        y = simulate_arfima(n=5000, d=0.0, rng=rng)
        # ACF at lag 1 should be near 0
        acf_1 = np.corrcoef(y[:-1], y[1:])[0, 1]
        assert abs(acf_1) < 0.05

    def test_simulate_long_memory_acf(self) -> None:
        """ARFIMA(0,d,0) with d>0 should have slowly decaying ACF."""
        rng = np.random.default_rng(42)
        y = simulate_arfima(n=5000, d=0.3, rng=rng)
        # ACF at lag 1 should be positive and significant
        acf_1 = np.corrcoef(y[:-1], y[1:])[0, 1]
        assert acf_1 > 0.1
        # ACF at large lag should still be positive (long memory)
        lag = 50
        acf_lag = np.corrcoef(y[:-lag], y[lag:])[0, 1]
        assert acf_lag > 0.0

    def test_simulate_with_ar(self) -> None:
        """ARFIMA(1,d,0) should produce valid series."""
        rng = np.random.default_rng(42)
        ar = np.array([0.5])
        y = simulate_arfima(n=200, d=0.2, ar=ar, rng=rng)
        assert len(y) == 200
        assert np.all(np.isfinite(y))

    def test_simulate_reproducible(self) -> None:
        """Same seed should produce same series."""
        y1 = simulate_arfima(n=100, d=0.3, rng=np.random.default_rng(42))
        y2 = simulate_arfima(n=100, d=0.3, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(y1, y2)
