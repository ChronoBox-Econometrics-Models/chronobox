"""
Integration tests for FASE 4 - SVAR and Advanced Models.

Tests the complete pipeline: SVAR -> HD -> Counterfactual,
BVAR forecasting, FAVAR factor extraction, TVP-VAR time-varying
coefficients, GVAR global system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la

# ============================================================
# Helper: Mock VARResults
# ============================================================


@dataclass
class MockVARResults:
    """Mock VARResults for integration testing."""

    coefs: NDArray[np.floating[Any]]
    sigma_u: NDArray[np.floating[Any]]
    resid: NDArray[np.floating[Any]]
    n_obs: int
    k_vars: int
    lags: int
    intercept: NDArray[np.floating[Any]] | None = None
    endog: NDArray[np.floating[Any]] | None = None

    @property
    def resid_cov(self) -> NDArray[np.floating[Any]]:
        return self.sigma_u


def _generate_var_data(
    K: int = 3,
    T: int = 500,
    p: int = 2,
    seed: int = 42,
) -> tuple[NDArray, MockVARResults]:
    """Generate synthetic VAR(p) data and mock results."""
    rng = np.random.default_rng(seed)

    coefs = np.zeros((p, K, K))
    for s in range(p):
        coefs[s] = rng.normal(0, 0.08 / (s + 1), (K, K))

    L_true = np.eye(K)
    L_true[np.tril_indices(K, -1)] = rng.normal(0, 0.2, K * (K - 1) // 2)
    Sigma = L_true @ L_true.T

    Y = np.zeros((T + p + 200, K))
    for t in range(p, T + p + 200):
        for s in range(p):
            Y[t] += Y[t - s - 1] @ coefs[s].T
        Y[t] += rng.multivariate_normal(np.zeros(K), Sigma)
    Y = Y[200:]

    T_eff = T - p
    resid = np.zeros((T_eff, K))
    for t in range(p, T):
        pred = np.zeros(K)
        for s in range(p):
            pred += Y[t - s - 1] @ coefs[s].T
        resid[t - p] = Y[t] - pred

    sigma_hat = (resid.T @ resid) / T_eff

    var_res = MockVARResults(
        coefs=coefs,
        sigma_u=sigma_hat,
        resid=resid,
        n_obs=T_eff,
        k_vars=K,
        lags=p,
        intercept=np.zeros(K),
        endog=Y,
    )

    return Y, var_res


# ============================================================
# Integration Test: SVAR -> HD -> Counterfactual
# ============================================================


class TestSVARHDCounterfactualPipeline:
    """End-to-end pipeline: SVAR -> Historical Decomposition -> Counterfactual."""

    def test_full_pipeline_cholesky(self) -> None:
        """Full pipeline with Cholesky identification."""
        from chronobox.analysis.counterfactual import Counterfactual
        from chronobox.analysis.hd import HistoricalDecomposition
        from chronobox.models.svar import SVAR

        _, var_res = _generate_var_data(K=3, T=300, p=2)

        # Step 1: SVAR
        svar = SVAR(var_res, method="cholesky")
        svar_res = svar.fit()
        assert svar_res.A0_inv.shape == (3, 3)

        # Step 2: Historical Decomposition
        hd = HistoricalDecomposition(svar_res)
        assert hd.decomposition.shape == (var_res.n_obs, 3, 3)
        assert hd.result.verify_decomposition(atol=1e-8)

        # Step 3: Counterfactual
        cf = Counterfactual(hd)
        y_cf = cf.without_shock(0)
        assert y_cf.shape == (var_res.n_obs, 3)

        # Removing all shocks = base
        y_base = cf.without_all_shocks()
        np.testing.assert_allclose(y_base, hd.base, atol=1e-10)

    def test_pipeline_blanchard_quah(self) -> None:
        """Pipeline with Blanchard-Quah identification."""
        from chronobox.analysis.hd import HistoricalDecomposition
        from chronobox.models.svar import SVAR

        _, var_res = _generate_var_data(K=2, T=500, p=2, seed=55)

        svar = SVAR(var_res, method="long_run")
        svar_res = svar.fit()

        hd = HistoricalDecomposition(svar_res)
        assert hd.result.verify_decomposition(atol=1e-7)

        # Long-run check: C(1) lower triangular
        A_sum = np.zeros((2, 2))
        for s in range(svar_res.lags):
            A_sum += svar_res.coefs[s]
        Theta1 = la.inv(np.eye(2) - A_sum)
        C1 = Theta1 @ svar_res.A0_inv
        assert abs(C1[0, 1]) < 1e-7, "Long-run restriction should hold"

    def test_pipeline_sign_restrictions(self) -> None:
        """Pipeline with sign restrictions."""
        from chronobox.analysis.counterfactual import Counterfactual
        from chronobox.analysis.hd import HistoricalDecomposition
        from chronobox.models.svar import SVAR

        _, var_res = _generate_var_data(K=2, T=300, p=2, seed=77)

        sign_restr = {(0, 0, range(0, 3)): "+"}
        svar = SVAR(var_res, method="sign", sign_restrictions=sign_restr)
        svar_res = svar.fit(n_draws=100, max_draws=20000)

        hd = HistoricalDecomposition(svar_res)
        assert hd.decomposition.shape[1] == 2  # 2 shocks

        cf = Counterfactual(hd)
        y_cf = cf.without_shock(0)
        assert y_cf.shape == (var_res.n_obs, 2)


# ============================================================
# Integration Test: BVAR Forecast Pipeline
# ============================================================


class TestBVARForecastPipeline:
    """End-to-end BVAR estimation and forecasting."""

    def test_minnesota_forecast_pipeline(self) -> None:
        """Minnesota prior -> fit -> forecast."""
        from chronobox.models.bvar import BayesianVAR

        Y, _ = _generate_var_data(K=2, T=200, p=1)

        bvar = BayesianVAR(lags=1, prior="minnesota", lambda_1=0.1, lambda_2=0.5)
        res = bvar.fit(Y, n_draws=1000, burnin=200, seed=42)

        fc = res.forecast(steps=12, n_draws=500)

        assert fc["mean"].shape == (12, 2)
        assert fc["draws"].shape == (500, 12, 2)

        # 95% bands should contain the mean
        assert np.all(fc["lower_95"] <= fc["mean"] + 1e-10)
        assert np.all(fc["mean"] <= fc["upper_95"] + 1e-10)

    def test_bvar_prior_comparison(self) -> None:
        """Compare marginal likelihoods across priors."""
        from chronobox.models.bvar import BayesianVAR

        Y, _ = _generate_var_data(K=2, T=200, p=1)

        results = {}
        for prior_name in ["minnesota", "flat", "normal_wishart"]:
            bvar = BayesianVAR(lags=1, prior=prior_name)
            res = bvar.fit(Y, n_draws=500, burnin=100, seed=42)
            results[prior_name] = res.log_marginal_likelihood

        # All should be finite
        for prior_name, ml in results.items():
            assert np.isfinite(ml), f"{prior_name} marginal likelihood is not finite"

    def test_bvar_irf(self) -> None:
        """BVAR IRF computation."""
        from chronobox.models.bvar import BayesianVAR

        Y, _ = _generate_var_data(K=2, T=200, p=1)

        bvar = BayesianVAR(lags=1, prior="minnesota", lambda_1=0.1)
        res = bvar.fit(Y, n_draws=500, burnin=100, seed=42)
        irf = res.irf(periods=20)

        assert irf.shape == (21, 2, 2)


# ============================================================
# Integration Test: FAVAR Pipeline
# ============================================================


class TestFAVARPipeline:
    """End-to-end FAVAR estimation and analysis."""

    def test_favar_full_pipeline(self) -> None:
        """FAVAR: extract factors -> VAR -> IRF."""
        from chronobox.models.favar import FAVAR

        rng = np.random.default_rng(42)
        T = 300
        K = 3
        N = 40
        M = 1

        # True factors
        F = np.zeros((T, K))
        for t in range(1, T):
            F[t] = 0.5 * F[t - 1] + rng.normal(0, 1, K)

        Y = np.zeros((T, M))
        for t in range(1, T):
            Y[t] = 0.3 * Y[t - 1] + 0.2 * F[t, 0] + rng.normal(0, 0.5, M)

        Lambda = rng.normal(0, 1, (N, K))
        X = F @ Lambda.T + rng.normal(0, 0.5, (T, N))

        favar = FAVAR(n_factors=K, lags=2)
        result = favar.fit(X, Y)

        # Factors explain meaningful variance
        assert result.total_explained_variance > 0.2

        # IRF
        irf = result.irf(periods=20)
        assert irf.shape == (21, K + M, K + M)

        # Panel IRF
        irf_panel = result.irf_panel(periods=20)
        assert irf_panel.shape == (21, N, K + M)


# ============================================================
# Integration Test: TVP-VAR Pipeline
# ============================================================


class TestTVPVARPipeline:
    """End-to-end TVP-VAR estimation."""

    def test_tvpvar_constant_vs_ols(self) -> None:
        """TVP-VAR with Q~0 should approximate constant VAR."""
        from chronobox.models.tvpvar import TVPVAR

        Y, var_res = _generate_var_data(K=2, T=200, p=1)

        tvpvar = TVPVAR(lags=1, Q_scale=1e-6)
        result = tvpvar.fit(Y)

        # Late coefficients should be close to each other
        coef_late = result.coefs_time[-50:, 0, :, :]
        coef_std = np.std(coef_late, axis=0)
        assert np.max(coef_std) < 0.05, "With Q~0, late coefficients should be stable"

    def test_tvpvar_time_varying_irf(self) -> None:
        """IRFs at different times should exist and have correct shapes."""
        from chronobox.models.tvpvar import TVPVAR

        rng = np.random.default_rng(42)
        K = 2
        T = 300
        Y = np.zeros((T, K))
        for t in range(1, T):
            alpha = t / T
            A = np.eye(K) * (0.3 + 0.4 * alpha)
            Y[t] = A @ Y[t - 1] + rng.normal(0, 0.3, K)

        tvpvar = TVPVAR(lags=1, Q_scale=0.005)
        result = tvpvar.fit(Y)

        irf_50 = result.time_varying_irf(t=50, periods=10)
        irf_200 = result.time_varying_irf(t=200, periods=10)

        assert irf_50.shape == (11, K, K)
        assert irf_200.shape == (11, K, K)


# ============================================================
# Integration Test: GVAR Pipeline
# ============================================================


class TestGVARPipeline:
    """End-to-end GVAR estimation."""

    def test_gvar_full_pipeline(self) -> None:
        """GVAR: fit -> GIRF -> cross-country analysis."""
        from chronobox.models.gvar import GVAR

        rng = np.random.default_rng(42)
        N = 3
        K_per = 2
        T = 300
        K_total = N * K_per

        # Trade weights
        W = np.array([[0, 0.6, 0.4], [0.5, 0, 0.5], [0.3, 0.7, 0]])

        # Generate connected data
        Y = np.zeros((T, K_total))
        A = np.eye(K_total) * 0.3
        # Add cross-country effects
        for i in range(N):
            for j in range(N):
                if i != j:
                    A[i * K_per, j * K_per] = 0.05 * W[i, j]

        Sigma = np.eye(K_total) * 0.1
        for t in range(1, T):
            Y[t] = A @ Y[t - 1] + rng.multivariate_normal(np.zeros(K_total), Sigma)

        data_dict = {
            f"country_{i}": Y[:, i * K_per : (i + 1) * K_per] for i in range(N)
        }

        gvar = GVAR(trade_weights=W, domestic_lags=1, foreign_lags=1)
        result = gvar.fit(data_dict)

        assert result.n_countries == 3
        assert result.k_total == 6

        # GIRF
        girf = result.girf(shock_country="country_0", shock_var=0, periods=20)
        assert girf.shape == (21, 6)

        # Cross-country IRF
        irf_c1 = result.irf_country(
            shock_country="country_0",
            shock_var=0,
            response_country="country_1",
            periods=20,
        )
        assert irf_c1.shape == (21, 2)


# ============================================================
# Cross-Model Consistency Tests
# ============================================================


class TestCrossModelConsistency:
    """Tests that verify consistency across different models."""

    def test_svar_cholesky_irf_correct(self) -> None:
        """SVAR Cholesky IRF at h=0 should equal the Cholesky factor."""
        from chronobox.models.svar import SVAR

        _, var_res = _generate_var_data(K=3, T=500, p=2)

        svar = SVAR(var_res, method="cholesky")
        svar_res = svar.fit()
        irf = svar_res.irf(periods=0)

        # At h=0, IRF should equal A0_inv = Cholesky(Sigma_u)
        P = la.cholesky(var_res.sigma_u, lower=True)
        np.testing.assert_allclose(irf[0], P, atol=1e-10)

    def test_fevd_sums_to_one(self) -> None:
        """FEVD from SVAR should sum to 1 at each horizon."""
        from chronobox.models.svar import SVAR

        _, var_res = _generate_var_data(K=3, T=500, p=2)

        svar = SVAR(var_res, method="cholesky")
        svar_res = svar.fit()
        fevd = svar_res.fevd(periods=20)

        for h in range(21):
            for i in range(3):
                np.testing.assert_allclose(
                    fevd[h, i, :].sum(),
                    1.0,
                    atol=1e-10,
                    err_msg=f"FEVD does not sum to 1 at h={h}, var={i}",
                )

    def test_hd_exact_decomposition(self) -> None:
        """Historical decomposition must sum exactly to observed (tol=1e-8)."""
        from chronobox.analysis.hd import HistoricalDecomposition
        from chronobox.models.svar import SVAR

        _, var_res = _generate_var_data(K=3, T=300, p=2, seed=123)

        svar = SVAR(var_res, method="cholesky")
        svar_res = svar.fit()

        hd = HistoricalDecomposition(svar_res)
        assert hd.result.verify_decomposition(atol=1e-8), (
            "HD does not sum to observed values within tolerance"
        )


# ============================================================
# Validation: Code Quality
# ============================================================


class TestCodeQuality:
    """Tests that validate code quality requirements."""

    def test_all_models_importable(self) -> None:
        """All FASE 4 modules should be importable."""
        # These imports should not raise
        from chronobox.analysis.counterfactual import Counterfactual
        from chronobox.analysis.hd import (
            HistoricalDecomposition,
        )
        from chronobox.models.bvar import BayesianVAR
        from chronobox.models.favar import FAVAR
        from chronobox.models.gvar import GVAR
        from chronobox.models.svar import SVAR
        from chronobox.models.tvpvar import TVPVAR

        assert SVAR is not None
        assert BayesianVAR is not None
        assert FAVAR is not None
        assert TVPVAR is not None
        assert GVAR is not None
        assert HistoricalDecomposition is not None
        assert Counterfactual is not None

    def test_results_have_required_attributes(self) -> None:
        """Result objects should have all documented attributes."""
        from chronobox.models.svar import SVAR

        _, var_res = _generate_var_data(K=2, T=200, p=1)
        svar = SVAR(var_res, method="cholesky")
        result = svar.fit()

        # SVARResults required attributes
        assert hasattr(result, "method")
        assert hasattr(result, "A0_inv")
        assert hasattr(result, "structural_shocks")
        assert hasattr(result, "sigma_u")
        assert hasattr(result, "n_obs")
        assert hasattr(result, "k_vars")
        assert hasattr(result, "lags")
        assert hasattr(result, "coefs")
        assert hasattr(result, "resid")

        # Methods
        assert callable(result.irf)
        assert callable(result.fevd)
        assert callable(result.irf_with_bands)
