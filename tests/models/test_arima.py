"""Tests for ARIMA model."""

from __future__ import annotations

import numpy as np

from chronobox.models.arima import ARIMA


class TestAirlineSARIMA:
    """Test SARIMA(0,1,1)(0,1,1)[12] on airline dataset.

    Reference values from statsmodels SARIMAX (Python standard).
    """

    def test_airline_sarima_params(self, airline_passengers: np.ndarray) -> None:
        """Validate MA parameters."""
        model = ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
        results = model.fit(airline_passengers, method="css-mle")

        # theta_1 ~ -0.3087 (tol=0.05)
        theta_idx = results.param_names.index("ma.L1")
        assert abs(results.params[theta_idx] - (-0.3087)) < 0.05, (
            f"theta_1 = {results.params[theta_idx]}, expected ~ -0.3087"
        )

        # Theta_1 ~ -0.1075 (tol=0.05)
        stheta_idx = results.param_names.index("ma.S.L12")
        assert abs(results.params[stheta_idx] - (-0.1075)) < 0.05, (
            f"Theta_1 = {results.params[stheta_idx]}, expected ~ -0.1075"
        )

    def test_airline_sarima_sigma2(self, airline_passengers: np.ndarray) -> None:
        """sigma2 ~ 135.4 (tol=5%)."""
        model = ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
        results = model.fit(airline_passengers, method="css-mle")

        sigma2_idx = results.param_names.index("sigma2")
        sigma2 = results.params[sigma2_idx]
        assert abs(sigma2 - 135.4) / 135.4 < 0.05, (
            f"sigma2 = {sigma2}, expected ~ 135.4"
        )

    def test_airline_sarima_loglike(self, airline_passengers: np.ndarray) -> None:
        """Log-likelihood ~ -507.50 (tol=0.5)."""
        model = ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
        results = model.fit(airline_passengers, method="css-mle")
        assert abs(results.loglike - (-507.50)) < 0.5, (
            f"loglike = {results.loglike}, expected ~ -507.50"
        )

    def test_airline_sarima_aic(self, airline_passengers: np.ndarray) -> None:
        """AIC ~ 1021.00 (tol=1.0)."""
        model = ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
        results = model.fit(airline_passengers, method="css-mle")
        assert abs(results.aic - 1021.00) < 1.0, (
            f"AIC = {results.aic}, expected ~ 1021.00"
        )


class TestARIMA111Nile:
    """Test ARIMA(1,1,1) on Nile dataset."""

    def test_nile_params(self, nile_volume: np.ndarray) -> None:
        """phi_1 ~ 0.254, theta_1 ~ -0.874 (tol=0.1)."""
        model = ARIMA(order=(1, 1, 1))
        results = model.fit(nile_volume, method="css-mle")

        ar_idx = results.param_names.index("ar.L1")
        ma_idx = results.param_names.index("ma.L1")

        assert abs(results.params[ar_idx] - 0.254) < 0.1, (
            f"phi_1 = {results.params[ar_idx]}, expected ~ 0.254"
        )
        assert abs(results.params[ma_idx] - (-0.874)) < 0.1, (
            f"theta_1 = {results.params[ma_idx]}, expected ~ -0.874"
        )


class TestForecast:
    """Test forecasting."""

    def test_forecast_12_airline(self, airline_passengers: np.ndarray) -> None:
        """Forecast 12 steps should capture seasonality."""
        model = ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
        results = model.fit(airline_passengers)
        fc = results.forecast(steps=12)

        assert "forecast" in fc
        assert "lower" in fc
        assert "upper" in fc
        assert len(fc["forecast"]) == 12
        # Forecasts should have confidence intervals
        assert np.all(fc["lower"] < fc["forecast"])
        assert np.all(fc["upper"] > fc["forecast"])


class TestCSSvsMLE:
    """Test CSS and MLE produce consistent results."""

    def test_css_vs_mle(self, nile_volume: np.ndarray) -> None:
        """CSS and MLE should both produce valid fits."""
        model_css = ARIMA(order=(1, 1, 1))
        results_css = model_css.fit(nile_volume, method="css")

        model_mle = ARIMA(order=(1, 1, 1))
        results_mle = model_mle.fit(nile_volume, method="css-mle")

        # Both should produce finite loglikelihood
        assert np.isfinite(results_css.loglike)
        assert np.isfinite(results_mle.loglike)
        # MLE should have better (higher) loglikelihood than CSS
        # or at least comparable
        assert results_mle.loglike >= results_css.loglike - 5.0


class TestSummary:
    """Test summary output."""

    def test_summary_no_error(self, nile_volume: np.ndarray) -> None:
        model = ARIMA(order=(1, 0, 1))
        results = model.fit(nile_volume)
        summary = results.summary()
        assert isinstance(summary, str)
        assert "ARIMA" in summary


class TestResidualDiagnostics:
    """Test residual diagnostics."""

    def test_residuals_white_noise(self, airline_passengers: np.ndarray) -> None:
        """Ljung-Box on residuals should have p > 0.05."""
        from scipy import stats as sp_stats

        model = ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
        results = model.fit(airline_passengers)

        resid = results.residuals[~np.isnan(results.residuals)]
        n = len(resid)
        max_lag = min(10, n // 5)

        # Compute Ljung-Box
        y_c = resid - np.mean(resid)
        c0 = np.dot(y_c, y_c) / n
        if c0 > 0:
            acf_vals = np.array(
                [np.dot(y_c[: n - k], y_c[k:]) / n / c0 for k in range(1, max_lag + 1)]
            )
            lb = n * (n + 2) * np.sum(acf_vals**2 / (n - np.arange(1, max_lag + 1)))
            p_value = 1.0 - sp_stats.chi2.cdf(lb, df=max_lag)
            assert p_value > 0.05, f"Ljung-Box p-value = {p_value}, expected > 0.05"


class TestStationarityInvertibility:
    """Test stationarity and invertibility checks."""

    def test_ar_roots(self, nile_volume: np.ndarray) -> None:
        """AR roots should be outside unit circle."""
        from chronobox.core.lag_polynomial import LagPolynomial

        model = ARIMA(order=(1, 1, 1))
        results = model.fit(nile_volume)
        ar_idx = results.param_names.index("ar.L1")
        ar_coeff = results.params[ar_idx]
        poly = LagPolynomial([1, -ar_coeff])
        assert poly.is_stationary()

    def test_ma_roots(self, nile_volume: np.ndarray) -> None:
        """MA roots should be outside unit circle."""
        from chronobox.core.lag_polynomial import LagPolynomial

        model = ARIMA(order=(1, 1, 1))
        results = model.fit(nile_volume)
        ma_idx = results.param_names.index("ma.L1")
        ma_coeff = results.params[ma_idx]
        poly = LagPolynomial([1, ma_coeff])
        assert poly.is_invertible()
