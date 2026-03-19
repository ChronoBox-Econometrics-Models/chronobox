"""Tests for TVP-VAR (Time-Varying Parameter VAR)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray


def _make_tvpvar_data(
    K: int = 2,
    T: int = 300,
    p: int = 1,
    time_varying: bool = False,
    seed: int = 42,
) -> NDArray:
    """Generate synthetic data for TVP-VAR testing.

    Parameters
    ----------
    time_varying : bool
        If True, coefficients change over time.
    """
    rng = np.random.default_rng(seed)

    Y = np.zeros((T, K))
    Sigma = np.eye(K) * 0.1

    if time_varying:
        # Coefficients evolve smoothly over time
        for t in range(p, T):
            # Time-varying coefficient: smooth transition
            alpha = t / T  # goes from 0 to 1
            coef = 0.3 * (1 - alpha) + 0.7 * alpha  # increases over time
            for s in range(p):
                A_s = np.eye(K) * coef * 0.5 / (s + 1)
                Y[t] += Y[t - s - 1] @ A_s.T
            Y[t] += rng.multivariate_normal(np.zeros(K), Sigma)
    else:
        # Constant coefficients
        A = np.eye(K) * 0.3
        for t in range(p, T):
            for s in range(p):
                Y[t] += Y[t - s - 1] @ (A / (s + 1)).T
            Y[t] += rng.multivariate_normal(np.zeros(K), Sigma)

    return Y


class TestTVPVARConstantParams:
    """Tests for TVP-VAR with constant parameters (Q ~ 0)."""

    def test_constant_params_close_to_ols(self) -> None:
        """With Q ~ 0, TVP-VAR coefficients should be approximately constant."""
        from chronobox.models.tvpvar import TVPVAR

        data = _make_tvpvar_data(K=2, T=200, p=1, time_varying=False, seed=42)
        tvpvar = TVPVAR(lags=1, Q_scale=1e-6)
        result = tvpvar.fit(data)

        # Check that coefficients are approximately constant over time
        coef_path_00 = result.coefficient_path(lag=0, i=0, j=0)

        # Standard deviation should be small relative to mean
        coef_std = np.std(coef_path_00)
        coef_mean = np.mean(np.abs(coef_path_00))
        if coef_mean > 0.01:
            cv = coef_std / coef_mean
            assert cv < 0.5, (
                f"Coefficient variation {cv:.2f} too high for constant params"
            )

    def test_shapes(self) -> None:
        """TVP-VAR results should have correct shapes."""
        from chronobox.models.tvpvar import TVPVAR

        K = 2
        T_total = 200
        p = 1
        data = _make_tvpvar_data(K=K, T=T_total, p=p)
        tvpvar = TVPVAR(lags=p)
        result = tvpvar.fit(data)

        T = T_total - p
        d = K * (K * p + 1)

        assert result.beta_filtered.shape == (T, d)
        assert result.coefs_time.shape == (T, p, K, K)
        assert result.intercept_time.shape == (T, K)
        assert result.sigma.shape == (K, K)
        assert result.n_obs == T
        assert result.k_vars == K
        assert result.lags == p

    def test_smoothed_available(self) -> None:
        """Smoothed state should be available."""
        from chronobox.models.tvpvar import TVPVAR

        data = _make_tvpvar_data(K=2, T=200, p=1)
        tvpvar = TVPVAR(lags=1)
        result = tvpvar.fit(data)

        assert result.beta_smoothed is not None
        assert result.beta_smoothed.shape == result.beta_filtered.shape


class TestTVPVARTimeVarying:
    """Tests for TVP-VAR with time-varying parameters."""

    def test_params_evolve(self) -> None:
        """With Q > 0, coefficients should vary over time."""
        from chronobox.models.tvpvar import TVPVAR

        data = _make_tvpvar_data(K=2, T=300, p=1, time_varying=True, seed=42)
        tvpvar = TVPVAR(lags=1, Q_scale=0.01)
        result = tvpvar.fit(data)

        # Check that coefficients vary over time
        coef_path = result.coefficient_path(lag=0, i=0, j=0)
        coef_std = np.std(coef_path)
        assert coef_std > 1e-6, "Coefficients should vary over time when Q > 0"

    def test_time_varying_irf_differs(self) -> None:
        """IRF at different time points should differ."""
        from chronobox.models.tvpvar import TVPVAR

        data = _make_tvpvar_data(K=2, T=300, p=1, time_varying=True, seed=42)
        tvpvar = TVPVAR(lags=1, Q_scale=0.01)
        result = tvpvar.fit(data)

        irf_early = result.time_varying_irf(t=50, periods=10)
        irf_late = result.time_varying_irf(t=200, periods=10)

        assert irf_early.shape == (11, 2, 2)
        assert irf_late.shape == (11, 2, 2)

        # They should differ (at least slightly)
        diff = np.max(np.abs(irf_early - irf_late))
        assert diff > 1e-6, "IRFs at different times should differ"

    def test_irf_shape(self) -> None:
        """Time-varying IRF should have correct shape."""
        from chronobox.models.tvpvar import TVPVAR

        data = _make_tvpvar_data(K=3, T=200, p=2)
        tvpvar = TVPVAR(lags=2)
        result = tvpvar.fit(data)

        irf = result.time_varying_irf(t=50, periods=20)
        assert irf.shape == (21, 3, 3)

    def test_irf_invalid_t_raises(self) -> None:
        """Invalid time index should raise ValueError."""
        from chronobox.models.tvpvar import TVPVAR

        data = _make_tvpvar_data(K=2, T=100, p=1)
        tvpvar = TVPVAR(lags=1)
        result = tvpvar.fit(data)

        with pytest.raises(ValueError, match="out of range"):
            result.time_varying_irf(t=-1, periods=10)

        with pytest.raises(ValueError, match="out of range"):
            result.time_varying_irf(t=result.n_obs, periods=10)


class TestTVPVARCustomQ:
    """Tests for TVP-VAR with custom Q matrix."""

    def test_custom_q_zero(self) -> None:
        """With Q = 0, coefficients should be constant after initial period."""
        from chronobox.models.tvpvar import TVPVAR

        K = 2
        p = 1
        data = _make_tvpvar_data(K=K, T=200, p=p)
        tvpvar = TVPVAR(lags=p)
        d = K * (K * p + 1)
        Q_zero = np.zeros((d, d))
        result = tvpvar.fit(data, Q=Q_zero)

        # With Q=0, the Kalman filter should converge to constant coefficients
        # after a few observations (the initial prior gets overwhelmed by data)
        coef_path = result.coefficient_path(lag=0, i=0, j=0)
        # Late coefficients should be very similar
        late_std = np.std(coef_path[-50:])
        early_std = np.std(coef_path[:50])
        # Late coefficients should be much more stable than early ones
        # (Kalman gain shrinks as P_t converges with Q=0)
        assert late_std < early_std or late_std < 0.05, (
            f"With Q=0, late coefficients should be stable: std={late_std}"
        )


class TestTVPVAREdgeCases:
    """Edge case tests."""

    def test_insufficient_data(self) -> None:
        """Too few observations should raise ValueError."""
        from chronobox.models.tvpvar import TVPVAR

        data = np.random.randn(3, 2)
        tvpvar = TVPVAR(lags=5)
        with pytest.raises(ValueError, match="Not enough observations"):
            tvpvar.fit(data)

    def test_1d_data(self) -> None:
        """1D data should be auto-reshaped."""
        from chronobox.models.tvpvar import TVPVAR

        rng = np.random.default_rng(42)
        data = np.cumsum(rng.normal(0, 1, 200))
        tvpvar = TVPVAR(lags=1)
        result = tvpvar.fit(data)

        assert result.k_vars == 1
        assert result.coefs_time.shape[2] == 1
        assert result.coefs_time.shape[3] == 1

    def test_log_likelihood_finite(self) -> None:
        """Log-likelihood should be finite."""
        from chronobox.models.tvpvar import TVPVAR

        data = _make_tvpvar_data(K=2, T=200, p=1)
        tvpvar = TVPVAR(lags=1)
        result = tvpvar.fit(data)

        assert np.isfinite(result.log_likelihood)

    def test_coefficient_path(self) -> None:
        """coefficient_path should return correct slice."""
        from chronobox.models.tvpvar import TVPVAR

        data = _make_tvpvar_data(K=2, T=100, p=1)
        tvpvar = TVPVAR(lags=1)
        result = tvpvar.fit(data)

        path = result.coefficient_path(lag=0, i=0, j=0)
        assert path.shape == (result.n_obs,)
        # Should match the direct extraction
        np.testing.assert_array_equal(path, result.coefs_time[:, 0, 0, 0])

    def test_multiple_lags(self) -> None:
        """TVP-VAR should work with multiple lags."""
        from chronobox.models.tvpvar import TVPVAR

        data = _make_tvpvar_data(K=2, T=200, p=3)
        tvpvar = TVPVAR(lags=3)
        result = tvpvar.fit(data)

        assert result.lags == 3
        assert result.coefs_time.shape[1] == 3
