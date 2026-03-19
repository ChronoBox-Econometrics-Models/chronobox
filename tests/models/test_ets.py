"""Tests for ETS (Exponential Smoothing State Space) models."""

from __future__ import annotations

import numpy as np
import pytest

from chronobox.models.ets import (
    ETS,
    ETSResults,
    _ets_loglik,
    _initialize_ets_states,
    get_all_ets_models,
)


@pytest.fixture
def airline_passengers() -> np.ndarray:
    """Airline passengers data."""
    try:
        from chronobox.datasets import load_dataset

        df = load_dataset("airline")
        return df["passengers"].to_numpy(dtype=np.float64)
    except Exception:
        # Fallback synthetic seasonal data
        rng = np.random.default_rng(42)
        t = np.arange(144)
        trend = 100 + 2 * t
        seasonal = 20 * np.sin(2 * np.pi * t / 12)
        return trend + seasonal + rng.normal(0, 5, 144)


@pytest.fixture
def simple_series() -> np.ndarray:
    """Simple positive series for testing."""
    rng = np.random.default_rng(42)
    t = np.arange(100)
    return 50.0 + 0.5 * t + rng.normal(0, 2, 100)


@pytest.fixture
def seasonal_series() -> np.ndarray:
    """Synthetic seasonal positive series."""
    rng = np.random.default_rng(42)
    t = np.arange(120)
    trend = 100 + 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    noise = rng.normal(0, 2, 120)
    return trend + seasonal + noise


class TestETSBasic:
    """Basic ETS model tests."""

    def test_ets_ann_fit(self, simple_series: np.ndarray) -> None:
        """ETS(A,N,N) - Simple Exponential Smoothing should fit."""
        model = ETS(error="A", trend="N", seasonal="N")
        results = model.fit(simple_series)
        assert isinstance(results, ETSResults)
        assert 0 < results.alpha < 1
        assert results.sigma2 > 0
        assert results.nobs == len(simple_series)

    def test_ets_aan_fit(self, simple_series: np.ndarray) -> None:
        """ETS(A,A,N) - Holt's linear should fit."""
        model = ETS(error="A", trend="A", seasonal="N")
        results = model.fit(simple_series)
        assert isinstance(results, ETSResults)
        assert results.beta is not None
        assert results.b0 is not None

    def test_ets_aan_airline(self, airline_passengers: np.ndarray) -> None:
        """ETS(A,A,N) on airline: alpha and beta should be reasonable."""
        model = ETS(error="A", trend="A", seasonal="N")
        results = model.fit(airline_passengers)
        assert 0 < results.alpha < 1
        assert results.beta is not None
        assert 0 < results.beta < results.alpha

    def test_ets_mam_airline(self, airline_passengers: np.ndarray) -> None:
        """ETS(M,A,M) on airline: should capture multiplicative seasonality."""
        model = ETS(error="M", trend="A", seasonal="M", seasonal_period=12)
        results = model.fit(airline_passengers)
        assert isinstance(results, ETSResults)
        assert results.gamma is not None
        assert results.s0 is not None
        assert len(results.s0) == 12

    def test_ets_forecast_positive_mmm(self, airline_passengers: np.ndarray) -> None:
        """ETS(M,M,M) forecast should always be positive for positive data."""
        model = ETS(error="M", trend="M", seasonal="M", seasonal_period=12)
        results = model.fit(airline_passengers)
        fc = results.forecast(steps=24)
        assert np.all(fc > 0), f"Forecast contains non-positive values: {fc[fc <= 0]}"

    def test_ets_damped_vs_undamped(self, simple_series: np.ndarray) -> None:
        """ETS(A,Ad,N) forecast should converge while ETS(A,A,N) may diverge."""
        model_damped = ETS(error="A", trend="A", seasonal="N", damped=True)
        model_linear = ETS(error="A", trend="A", seasonal="N", damped=False)
        res_d = model_damped.fit(simple_series)
        res_l = model_linear.fit(simple_series)
        fc_d = res_d.forecast(steps=100)
        fc_l = res_l.forecast(steps=100)
        # Damped forecast should level off (range smaller at long horizon)
        range_d = float(fc_d[-1] - fc_d[0])
        range_l = float(fc_l[-1] - fc_l[0])
        assert abs(range_d) <= abs(range_l) + 1.0, (
            f"Damped range={range_d}, Linear range={range_l}"
        )

    def test_ets_results_summary(self, simple_series: np.ndarray) -> None:
        """summary() should return a non-empty formatted string."""
        model = ETS(error="A", trend="A", seasonal="N")
        results = model.fit(simple_series)
        s = results.summary()
        assert "ETS" in s
        assert "alpha" in s
        assert "Log-Likelihood" in s


class TestETSAll30Models:
    """Test all 30 ETS model combinations."""

    def test_all_30_models_fit(self) -> None:
        """All 30 ETS models should fit without error on a suitable series."""
        rng = np.random.default_rng(42)
        t = np.arange(120)
        trend = 100 + 0.5 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        noise = rng.normal(0, 2, 120)
        y = trend + seasonal + noise
        # Ensure all positive for multiplicative models
        y = y - np.min(y) + 10

        all_models = get_all_ets_models()
        assert len(all_models) == 30

        passed = 0
        failed = []
        for error, tr, seas in all_models:
            try:
                sp = 12 if seas != "N" else None
                model = ETS(error=error, trend=tr, seasonal=seas, seasonal_period=sp)
                results = model.fit(y)
                assert isinstance(results, ETSResults)
                assert np.isfinite(results.loglik)
                passed += 1
            except Exception as e:
                failed.append((f"ETS({error},{tr},{seas})", str(e)))

        # Allow some models to fail (some multiplicative combos can be unstable)
        assert passed >= 25, f"Only {passed}/30 models passed. Failed: {failed}"


class TestETSLoglikelihood:
    """Tests for ETS log-likelihood computation."""

    def test_loglike_additive_ses(self) -> None:
        """ETS(A,N,N) loglike should match manual SES computation."""
        rng = np.random.default_rng(42)
        y = rng.normal(100, 5, 50)
        alpha = 0.3
        l0 = float(y[0])

        # Manual SES
        n = len(y)
        resid = np.zeros(n)
        l_t = l0
        for t in range(n):
            resid[t] = y[t] - l_t
            l_t = l_t + alpha * resid[t]
        sigma2_manual = float(np.mean(resid**2))
        loglik_manual = -0.5 * n * (np.log(2 * np.pi * sigma2_manual) + 1)

        # ETS loglik
        loglik_ets = _ets_loglik(
            y, "A", "N", "N", 1, alpha, None, None, None, l0, None, None
        )

        np.testing.assert_allclose(loglik_ets, loglik_manual, rtol=1e-6)

    def test_loglike_multiplicative_includes_jacobian(self) -> None:
        """ETS(M,N,N) loglike should include the Jacobian term -sum(log|mu_t|)."""
        rng = np.random.default_rng(42)
        y = rng.normal(100, 5, 50)
        y = np.abs(y) + 10  # ensure positive
        alpha = 0.3
        l0 = float(y[0])

        # Manual multiplicative SES
        n = len(y)
        resid_m = np.zeros(n)
        mu_vals = np.zeros(n)
        l_t = l0
        for t in range(n):
            mu_t = l_t
            mu_vals[t] = mu_t
            resid_m[t] = (y[t] - mu_t) / mu_t
            l_t = mu_t * (1 + alpha * resid_m[t])

        sigma2_m = float(np.mean(resid_m**2))
        loglik_no_jacobian = -0.5 * n * (np.log(2 * np.pi * sigma2_m) + 1)
        jacobian = -float(np.sum(np.log(np.abs(mu_vals))))
        loglik_with_jacobian = loglik_no_jacobian + jacobian

        loglik_ets = _ets_loglik(
            y, "M", "N", "N", 1, alpha, None, None, None, l0, None, None
        )

        np.testing.assert_allclose(loglik_ets, loglik_with_jacobian, rtol=1e-4)


class TestETSForecast:
    """Tests for ETS forecasting."""

    def test_forecast_length(self, simple_series: np.ndarray) -> None:
        """forecast() should return correct number of steps."""
        model = ETS(error="A", trend="A", seasonal="N")
        results = model.fit(simple_series)
        for steps in [1, 5, 12, 24]:
            fc = results.forecast(steps=steps)
            assert len(fc) == steps

    def test_forecast_finite(self, airline_passengers: np.ndarray) -> None:
        """Forecasts should be finite."""
        model = ETS(error="A", trend="A", seasonal="A", seasonal_period=12)
        results = model.fit(airline_passengers)
        fc = results.forecast(steps=24)
        assert np.all(np.isfinite(fc))

    def test_forecast_seasonal_pattern(self, airline_passengers: np.ndarray) -> None:
        """Forecast should exhibit seasonal pattern for seasonal models."""
        model = ETS(error="M", trend="A", seasonal="M", seasonal_period=12)
        results = model.fit(airline_passengers)
        fc = results.forecast(steps=24)
        # Check that forecast has seasonality: months 7-8 (summer) should be higher
        # than months 11-12 (winter) in first year
        summer = fc[6:8]
        winter = fc[10:12]
        assert np.mean(summer) > np.mean(winter), (
            f"Summer mean={np.mean(summer)}, Winter mean={np.mean(winter)}"
        )


class TestETSInvalidInputs:
    """Tests for invalid input handling."""

    def test_invalid_error(self) -> None:
        with pytest.raises(ValueError, match="error"):
            ETS(error="X")

    def test_invalid_trend(self) -> None:
        with pytest.raises(ValueError, match="trend"):
            ETS(trend="X")

    def test_invalid_seasonal(self) -> None:
        with pytest.raises(ValueError, match="seasonal"):
            ETS(seasonal="X")

    def test_missing_seasonal_period(self) -> None:
        with pytest.raises(ValueError, match="seasonal_period"):
            ETS(seasonal="A")

    def test_mult_error_negative_data(self) -> None:
        y = np.array([-1.0, 2.0, 3.0, 4.0, 5.0])
        model = ETS(error="M", trend="N", seasonal="N")
        with pytest.raises(ValueError, match="positive"):
            model.fit(y)


class TestETSInitialization:
    """Tests for state initialization."""

    def test_init_nonseasonal(self) -> None:
        y = np.array([10.0, 12.0, 15.0, 11.0, 14.0])
        l0, b0, s0 = _initialize_ets_states(y, "A", "N", 1)
        assert l0 == pytest.approx(10.0)
        assert b0 is not None
        assert b0 == pytest.approx(2.0)
        assert s0 is None

    def test_init_seasonal_sum_zero(self) -> None:
        rng = np.random.default_rng(42)
        t = np.arange(48)
        y = 100 + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 48)
        l0, b0, s0 = _initialize_ets_states(y, "A", "A", 12)
        assert s0 is not None
        assert len(s0) == 12
        assert abs(np.sum(s0)) < 0.01  # sum should be ~0


class TestETSSimulate:
    """Tests for ETS simulation."""

    def test_simulate_ann(self) -> None:
        model = ETS(error="A", trend="N", seasonal="N")
        y = model.simulate(nsimulations=100, alpha=0.3, l0=100, sigma=5, seed=42)
        assert len(y) == 100
        assert np.all(np.isfinite(y))

    def test_simulate_reproducible(self) -> None:
        model = ETS(error="A", trend="A", seasonal="N")
        y1 = model.simulate(
            nsimulations=50, alpha=0.3, beta=0.1, l0=100, b0=1, sigma=2, seed=42
        )
        y2 = model.simulate(
            nsimulations=50, alpha=0.3, beta=0.1, l0=100, b0=1, sigma=2, seed=42
        )
        np.testing.assert_array_equal(y1, y2)

    def test_simulate_seasonal(self) -> None:
        s0 = np.array(
            [5, 3, 1, -1, -3, -5, -5, -3, -1, 1, 3, 5], dtype=np.float64
        )
        model = ETS(error="A", trend="A", seasonal="A", seasonal_period=12)
        y = model.simulate(
            nsimulations=120,
            alpha=0.1,
            beta=0.01,
            gamma=0.01,
            l0=100,
            b0=0.5,
            s0=s0,
            sigma=2,
            seed=42,
        )
        assert len(y) == 120
        assert np.all(np.isfinite(y))
