"""Tests for Bayesian VAR (BVAR) models."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import linalg as la

from chronobox.models.bvar import BayesianVAR


def _make_var_data(K: int = 3, T: int = 200, p: int = 2, seed: int = 42) -> NDArray:
    """Generate synthetic stable VAR(p) data."""
    rng = np.random.default_rng(seed)

    coefs = np.zeros((p, K, K))
    for s in range(p):
        coefs[s] = rng.normal(0, 0.05 / (s + 1), (K, K))

    Sigma = np.eye(K) * 0.1

    Y = np.zeros((T + 100, K))
    for t in range(p, T + 100):
        for s in range(p):
            Y[t] += Y[t - s - 1] @ coefs[s].T
        Y[t] += rng.multivariate_normal(np.zeros(K), Sigma)

    return Y[100:]


class TestMinnesotaPrior:
    """Tests for Minnesota prior."""

    def test_minnesota_flat_equals_ols(self) -> None:
        """BVAR with very large lambda_1 should approximate OLS."""
        data = _make_var_data(K=2, T=200, p=1, seed=42)

        bvar = BayesianVAR(lags=1, prior="minnesota", lambda_1=100.0, lambda_2=1.0)
        bvar_res = bvar.fit(data, n_draws=2000, burnin=500, seed=42)

        T = len(data) - 1
        K = 2
        Y = data[1:]
        Z = np.column_stack([data[:-1], np.ones(T)])
        B_ols = la.lstsq(Z, Y)[0]
        coefs_ols = B_ols[:K].T

        np.testing.assert_allclose(
            bvar_res.coefs_mean[0],
            coefs_ols,
            atol=0.05,
            err_msg="With flat prior (large lambda_1), BVAR should approximate OLS",
        )

    def test_minnesota_shrinkage(self) -> None:
        """BVAR with small lambda_1 should shrink toward prior mean."""
        data = _make_var_data(K=2, T=100, p=1, seed=42)

        bvar_tight = BayesianVAR(lags=1, prior="minnesota", lambda_1=0.01, delta=1.0)
        res_tight = bvar_tight.fit(data, n_draws=2000, burnin=500, seed=42)

        prior_mean = np.eye(2)
        diff = np.abs(res_tight.coefs_mean[0] - prior_mean)
        assert np.mean(diff) < 0.5, "Tight prior should shrink coefficients toward prior mean"

    def test_minnesota_forecast(self) -> None:
        """BVAR Minnesota forecast should produce credibility bands."""
        data = _make_var_data(K=2, T=200, p=1, seed=42)

        bvar = BayesianVAR(lags=1, prior="minnesota", lambda_1=0.1)
        res = bvar.fit(data, n_draws=1000, burnin=200, seed=42)
        fc = res.forecast(steps=12, n_draws=500)

        assert fc["mean"].shape == (12, 2)
        assert fc["median"].shape == (12, 2)
        assert fc["lower_68"].shape == (12, 2)
        assert fc["upper_68"].shape == (12, 2)
        assert fc["lower_95"].shape == (12, 2)
        assert fc["upper_95"].shape == (12, 2)
        assert fc["draws"].shape[0] == 500
        assert fc["draws"].shape[1] == 12
        assert fc["draws"].shape[2] == 2

        width_68 = fc["upper_68"] - fc["lower_68"]
        width_95 = fc["upper_95"] - fc["lower_95"]
        assert np.all(width_68 <= width_95 + 1e-10)


class TestNormalWishartPrior:
    """Tests for Normal-Wishart prior."""

    def test_normal_wishart_posterior(self) -> None:
        """Posterior mean should be between prior mean and OLS."""
        data = _make_var_data(K=2, T=200, p=1, seed=42)

        bvar = BayesianVAR(
            lags=1,
            prior="normal_wishart",
            V_0=np.eye(3) * 1.0,
            v_0=4.0,
            S_0=np.eye(2) * 0.01,
        )
        res = bvar.fit(data, n_draws=2000, burnin=500, seed=42)

        assert res.coefs_mean.shape == (1, 2, 2)
        assert res.sigma_mean.shape == (2, 2)

        eigvals = np.linalg.eigvalsh(res.sigma_mean)
        assert np.all(eigvals > 0), "Posterior mean of Sigma should be positive definite"

    def test_normal_wishart_draws_shape(self) -> None:
        """Posterior draws should have correct shapes."""
        data = _make_var_data(K=2, T=100, p=1, seed=42)

        bvar = BayesianVAR(lags=1, prior="normal_wishart")
        res = bvar.fit(data, n_draws=500, burnin=100, seed=42)

        assert res.coefs_draws.shape == (500, 1, 2, 2)
        assert res.sigma_draws.shape == (500, 2, 2)
        assert res.intercept_draws.shape == (500, 2)

    def test_log_marginal_likelihood(self) -> None:
        """Marginal likelihood should be a finite number."""
        data = _make_var_data(K=2, T=200, p=1, seed=42)

        bvar = BayesianVAR(lags=1, prior="normal_wishart")
        res = bvar.fit(data, n_draws=500, burnin=100, seed=42)

        assert np.isfinite(res.log_marginal_likelihood)

    def test_marginal_likelihood_comparison(self) -> None:
        """Different priors should yield different marginal likelihoods."""
        data = _make_var_data(K=2, T=200, p=1, seed=42)

        bvar1 = BayesianVAR(lags=1, prior="normal_wishart", V_0=np.eye(3) * 1.0)
        res1 = bvar1.fit(data, n_draws=500, burnin=100, seed=42)

        bvar2 = BayesianVAR(lags=1, prior="normal_wishart", V_0=np.eye(3) * 100.0)
        res2 = bvar2.fit(data, n_draws=500, burnin=100, seed=42)

        assert res1.log_marginal_likelihood != res2.log_marginal_likelihood


class TestSimsZhaPrior:
    """Tests for Sims-Zha prior."""

    def test_sims_zha_basic(self) -> None:
        """Sims-Zha prior should produce valid results."""
        data = _make_var_data(K=2, T=200, p=1, seed=42)

        bvar = BayesianVAR(
            lags=1, prior="sims_zha", lambda_1=0.1, mu_5=1.0, mu_6=1.0
        )
        res = bvar.fit(data, n_draws=1000, burnin=200, seed=42)

        assert res.coefs_mean.shape == (1, 2, 2)
        assert res.prior == "sims_zha"
        assert np.isfinite(res.log_marginal_likelihood)


class TestFlatPrior:
    """Tests for flat (diffuse) prior."""

    def test_flat_equals_ols(self) -> None:
        """Flat prior posterior mean should approximate OLS."""
        data = _make_var_data(K=2, T=300, p=1, seed=42)

        bvar = BayesianVAR(lags=1, prior="flat")
        res = bvar.fit(data, n_draws=3000, burnin=1000, seed=42)

        T = len(data) - 1
        K = 2
        Y = data[1:]
        Z = np.column_stack([data[:-1], np.ones(T)])
        B_ols = la.lstsq(Z, Y)[0]
        coefs_ols = B_ols[:K].T

        np.testing.assert_allclose(
            res.coefs_mean[0],
            coefs_ols,
            atol=0.05,
            err_msg="Flat prior posterior mean should be close to OLS",
        )


class TestBVARIRF:
    """Tests for BVAR IRF computation."""

    def test_irf_shape(self) -> None:
        """IRF should have correct shape."""
        data = _make_var_data(K=2, T=200, p=1, seed=42)

        bvar = BayesianVAR(lags=1, prior="minnesota", lambda_1=0.1)
        res = bvar.fit(data, n_draws=500, burnin=100, seed=42)
        irf = res.irf(periods=20)

        assert irf.shape == (21, 2, 2)

    def test_irf_draws_shape(self) -> None:
        """IRF draws should have correct shape."""
        data = _make_var_data(K=2, T=200, p=1, seed=42)

        bvar = BayesianVAR(lags=1, prior="minnesota", lambda_1=0.1)
        res = bvar.fit(data, n_draws=100, burnin=50, seed=42)
        irf_draws = res.irf_draws_compute(periods=10)

        assert irf_draws.shape == (100, 11, 2, 2)


class TestBVAREdgeCases:
    """Edge case tests."""

    def test_unknown_prior_raises(self) -> None:
        """Unknown prior should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown prior"):
            BayesianVAR(lags=1, prior="invalid")

    def test_insufficient_data_raises(self) -> None:
        """Too few observations should raise ValueError."""
        data = np.random.default_rng(42).normal(size=(3, 2))
        bvar = BayesianVAR(lags=5, prior="flat")
        with pytest.raises(ValueError, match="Not enough observations"):
            bvar.fit(data)

    def test_single_variable(self) -> None:
        """BVAR should work with K=1."""
        rng = np.random.default_rng(42)
        data = np.cumsum(rng.normal(0, 1, (200, 1)), axis=0)

        bvar = BayesianVAR(lags=1, prior="minnesota", lambda_1=0.1)
        res = bvar.fit(data, n_draws=500, burnin=100, seed=42)

        assert res.coefs_mean.shape == (1, 1, 1)
