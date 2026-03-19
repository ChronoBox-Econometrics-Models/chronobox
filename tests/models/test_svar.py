"""Tests for SVAR (Structural VAR) identification schemes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray


@dataclass
class MockVARResults:
    """Mock VARResults for testing SVAR."""

    coefs: NDArray[np.floating[Any]]
    sigma_u: NDArray[np.floating[Any]]
    resid: NDArray[np.floating[Any]]
    n_obs: int
    k_vars: int
    lags: int
    intercept: NDArray[np.floating[Any]] | None = None


def _make_var_data(
    K: int = 3,
    T: int = 500,
    p: int = 2,
    seed: int = 42,
) -> MockVARResults:
    """Generate synthetic VAR(p) data and return mock results."""
    rng = np.random.default_rng(seed)

    # Generate stable VAR coefficients
    coefs = np.zeros((p, K, K))
    for s in range(p):
        coefs[s] = rng.normal(0, 0.1 / (s + 1), (K, K))

    # True structural impact matrix (lower triangular for Cholesky ground truth)
    L_true = np.eye(K)
    L_true[np.tril_indices(K, -1)] = rng.normal(0, 0.3, K * (K - 1) // 2)
    Sigma_u = L_true @ L_true.T

    # Simulate VAR
    Y = np.zeros((T + p + 100, K))
    for t in range(p, T + p + 100):
        for s in range(p):
            Y[t] += Y[t - s - 1] @ coefs[s].T
        Y[t] += rng.multivariate_normal(np.zeros(K), Sigma_u)

    # Drop burn-in
    Y = Y[100 + p :]

    # Compute residuals (for mock)
    resid = np.zeros((T - p, K))
    for t in range(p, T):
        pred = np.zeros(K)
        for s in range(p):
            pred += Y[t - s - 1] @ coefs[s].T
        resid[t - p] = Y[t] - pred

    sigma_hat = (resid.T @ resid) / len(resid)

    return MockVARResults(
        coefs=coefs,
        sigma_u=sigma_hat,
        resid=resid,
        n_obs=len(resid),
        k_vars=K,
        lags=p,
        intercept=np.zeros(K),
    )


class TestCholeskyIdentification:
    """Tests for Cholesky identification scheme."""

    def test_cholesky_basic(self) -> None:
        """SVAR Cholesky produces valid results."""
        from chronobox.models.svar import SVAR

        var_res = _make_var_data(K=3, T=500, p=2)
        svar = SVAR(var_res, method="cholesky")
        result = svar.fit()

        assert result.method == "cholesky"
        assert result.A0_inv.shape == (3, 3)
        assert result.structural_shocks.shape == (var_res.n_obs, 3)

    def test_cholesky_reconstructs_sigma(self) -> None:
        """A0_inv @ A0_inv.T should equal Sigma_u."""
        from chronobox.models.svar import SVAR

        var_res = _make_var_data(K=3, T=500, p=2)
        svar = SVAR(var_res, method="cholesky")
        result = svar.fit()

        reconstructed = result.A0_inv @ result.A0_inv.T
        np.testing.assert_allclose(reconstructed, var_res.sigma_u, atol=1e-10)

    def test_cholesky_is_lower_triangular(self) -> None:
        """Cholesky impact matrix should be lower triangular."""
        from chronobox.models.svar import SVAR

        var_res = _make_var_data(K=3, T=500, p=2)
        svar = SVAR(var_res, method="cholesky")
        result = svar.fit()

        upper = np.triu(result.A0_inv, k=1)
        np.testing.assert_allclose(upper, 0.0, atol=1e-12)

    def test_cholesky_irf_shape(self) -> None:
        """IRF should have correct shape (H+1, K, K)."""
        from chronobox.models.svar import SVAR

        var_res = _make_var_data(K=3, T=500, p=2)
        svar = SVAR(var_res, method="cholesky")
        result = svar.fit()
        irf = result.irf(periods=20)

        assert irf.shape == (21, 3, 3)

    def test_cholesky_irf_impact(self) -> None:
        """IRF at horizon 0 should equal the impact matrix."""
        from chronobox.models.svar import SVAR

        var_res = _make_var_data(K=3, T=500, p=2)
        svar = SVAR(var_res, method="cholesky")
        result = svar.fit()
        irf = result.irf(periods=20)

        np.testing.assert_allclose(irf[0], result.A0_inv, atol=1e-12)

    def test_cholesky_shocks_uncorrelated(self) -> None:
        """Structural shocks should be approximately uncorrelated with unit variance."""
        from chronobox.models.svar import SVAR

        var_res = _make_var_data(K=3, T=2000, p=2, seed=123)
        svar = SVAR(var_res, method="cholesky")
        result = svar.fit()

        eps = result.structural_shocks
        cov = (eps.T @ eps) / len(eps)
        np.testing.assert_allclose(cov, np.eye(3), atol=0.1)

    def test_ordering_matters_cholesky(self) -> None:
        """Changing variable ordering should produce different Cholesky IRFs."""
        from chronobox.models.svar import SVAR

        var_res1 = _make_var_data(K=3, T=500, p=2, seed=42)
        svar1 = SVAR(var_res1, method="cholesky")
        result1 = svar1.fit()
        irf1 = result1.irf(periods=10)

        K = 3
        perm = list(reversed(range(K)))
        coefs2 = var_res1.coefs[:, perm, :][:, :, perm]
        sigma2 = var_res1.sigma_u[perm, :][:, perm]
        resid2 = var_res1.resid[:, perm]

        var_res2 = MockVARResults(
            coefs=coefs2,
            sigma_u=sigma2,
            resid=resid2,
            n_obs=var_res1.n_obs,
            k_vars=K,
            lags=var_res1.lags,
            intercept=np.zeros(K),
        )
        svar2 = SVAR(var_res2, method="cholesky")
        result2 = svar2.fit()
        irf2 = result2.irf(periods=10)

        assert not np.allclose(irf1, irf2, atol=1e-6)

    def test_cholesky_fevd(self) -> None:
        """FEVD should sum to 1 across shocks at each horizon."""
        from chronobox.models.svar import SVAR

        var_res = _make_var_data(K=3, T=500, p=2)
        svar = SVAR(var_res, method="cholesky")
        result = svar.fit()
        fevd = result.fevd(periods=20)

        assert fevd.shape == (21, 3, 3)
        for h in range(21):
            for i in range(3):
                np.testing.assert_allclose(fevd[h, i, :].sum(), 1.0, atol=1e-10)


class TestABModel:
    """Tests for AB-model identification."""

    def test_ab_just_identified(self) -> None:
        """AB-model with just-identified restrictions converges."""
        from chronobox.models.svar import SVAR

        K = 2
        var_res = _make_var_data(K=K, T=500, p=1, seed=99)

        A = np.array([[1.0, 0.0], [np.nan, 1.0]])
        B = np.array([[np.nan, 0.0], [0.0, np.nan]])

        svar = SVAR(var_res, method="ab", a_mat=A, b_mat=B)
        result = svar.fit()

        assert result.method == "ab"
        assert result.A0 is not None
        assert result.B is not None
        assert result.A0.shape == (K, K)

    def test_ab_reconstructs_sigma(self) -> None:
        """A^{-1}*B*B'*(A^{-1})' should approximate Sigma_u."""
        from scipy import linalg as la

        from chronobox.models.svar import SVAR

        K = 2
        var_res = _make_var_data(K=K, T=1000, p=1, seed=99)

        A = np.array([[1.0, 0.0], [np.nan, 1.0]])
        B = np.array([[np.nan, 0.0], [0.0, np.nan]])

        svar = SVAR(var_res, method="ab", a_mat=A, b_mat=B)
        result = svar.fit()

        A_inv = la.inv(result.A0)
        reconstructed = A_inv @ result.B @ result.B.T @ A_inv.T
        np.testing.assert_allclose(reconstructed, var_res.sigma_u, atol=0.05)


class TestBlanchardQuah:
    """Tests for Blanchard-Quah long-run restrictions."""

    def test_long_run_lower_triangular(self) -> None:
        """Long-run impact matrix C(1) = Theta(1)*P should be lower triangular."""
        from scipy import linalg as la

        from chronobox.models.svar import SVAR

        K = 2
        var_res = _make_var_data(K=K, T=1000, p=2, seed=55)
        svar = SVAR(var_res, method="long_run")
        result = svar.fit()

        A_sum = np.zeros((K, K))
        for s in range(result.lags):
            A_sum += result.coefs[s]
        Theta1 = la.inv(np.eye(K) - A_sum)
        C1 = Theta1 @ result.A0_inv

        for i in range(K):
            for j in range(i + 1, K):
                assert abs(C1[i, j]) < 1e-8, f"C(1)[{i},{j}] = {C1[i, j]} is not zero"

    def test_blanchard_quah_demand_no_longrun_effect(self) -> None:
        """In BQ identification, demand shock should have zero long-run effect on output."""
        from scipy import linalg as la

        from chronobox.models.svar import SVAR

        K = 2
        var_res = _make_var_data(K=K, T=2000, p=2, seed=55)
        svar = SVAR(var_res, method="long_run")
        result = svar.fit()

        A_sum = np.zeros((K, K))
        for s in range(result.lags):
            A_sum += result.coefs[s]
        Theta1 = la.inv(np.eye(K) - A_sum)
        C1 = Theta1 @ result.A0_inv

        assert abs(C1[0, 1]) < 1e-8


class TestSignRestrictions:
    """Tests for sign restrictions identification."""

    def test_sign_restrictions_valid_draws(self) -> None:
        """All accepted draws should satisfy sign restrictions."""
        from chronobox.models.svar import SVAR, _compute_irf

        K = 2
        var_res = _make_var_data(K=K, T=500, p=2, seed=77)

        sign_restr = {
            (0, 0, range(0, 4)): "+",
        }
        svar = SVAR(var_res, method="sign", sign_restrictions=sign_restr)
        result = svar.fit(n_draws=100, max_draws=10000)

        assert result.accepted_draws is not None
        assert len(result.accepted_draws) > 0

        for impact in result.accepted_draws:
            irf = _compute_irf(var_res.coefs, impact, 3)
            for h in range(4):
                assert irf[h, 0, 0] >= 0, f"Sign restriction violated at h={h}"

    def test_sign_restrictions_multiple(self) -> None:
        """Multiple sign restrictions should all be satisfied."""
        from chronobox.models.svar import SVAR, _compute_irf

        K = 2
        var_res = _make_var_data(K=K, T=500, p=2, seed=77)

        # Use only impact period (h=0) for both restrictions to increase acceptance
        sign_restr = {
            (0, 0, range(0, 1)): "+",
            (1, 1, range(0, 1)): "+",
        }
        svar = SVAR(var_res, method="sign", sign_restrictions=sign_restr)
        result = svar.fit(n_draws=50, max_draws=50000)

        assert result.accepted_draws is not None
        for impact in result.accepted_draws:
            irf = _compute_irf(var_res.coefs, impact, 0)
            assert irf[0, 0, 0] >= 0
            assert irf[0, 1, 1] >= 0

    def test_sign_irf_with_bands(self) -> None:
        """IRF with bands should return three arrays."""
        from chronobox.models.svar import SVAR

        K = 2
        var_res = _make_var_data(K=K, T=500, p=2, seed=77)

        sign_restr = {(0, 0, range(0, 3)): "+"}
        svar = SVAR(var_res, method="sign", sign_restrictions=sign_restr)
        result = svar.fit(n_draws=50, max_draws=10000)

        median, lower, upper = result.irf_with_bands(periods=10)
        assert median.shape == (11, K, K)
        assert lower.shape == (11, K, K)
        assert upper.shape == (11, K, K)
        assert np.all(lower <= median + 1e-10)
        assert np.all(median <= upper + 1e-10)

    def test_sign_no_restrictions_raises(self) -> None:
        """Sign restrictions with empty dict should raise ValueError."""
        from chronobox.models.svar import SVAR

        var_res = _make_var_data(K=2, T=500, p=2)
        svar = SVAR(var_res, method="sign", sign_restrictions={})
        with pytest.raises(ValueError, match="sign_restrictions must be provided"):
            svar.fit()


class TestSVAREdgeCases:
    """Edge case tests."""

    def test_unknown_method_raises(self) -> None:
        """Unknown method should raise ValueError."""
        from chronobox.models.svar import SVAR

        var_res = _make_var_data(K=2, T=100, p=1)
        with pytest.raises(ValueError, match="Unknown method"):
            SVAR(var_res, method="unknown")

    def test_single_variable(self) -> None:
        """SVAR should work with K=1 (trivial case)."""
        from chronobox.models.svar import SVAR

        var_res = _make_var_data(K=1, T=500, p=1, seed=10)
        svar = SVAR(var_res, method="cholesky")
        result = svar.fit()

        assert result.A0_inv.shape == (1, 1)
        irf = result.irf(periods=10)
        assert irf.shape == (11, 1, 1)

    def test_large_k(self) -> None:
        """SVAR should work with larger K."""
        from chronobox.models.svar import SVAR

        var_res = _make_var_data(K=5, T=1000, p=1, seed=20)
        svar = SVAR(var_res, method="cholesky")
        result = svar.fit()

        assert result.A0_inv.shape == (5, 5)
        irf = result.irf(periods=10)
        assert irf.shape == (11, 5, 5)
