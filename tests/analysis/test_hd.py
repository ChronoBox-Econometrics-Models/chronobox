"""Tests for Historical Decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class MockVARResults:
    """Mock VARResults."""

    coefs: NDArray[np.floating[Any]]
    sigma_u: NDArray[np.floating[Any]]
    resid: NDArray[np.floating[Any]]
    n_obs: int
    k_vars: int
    lags: int
    intercept: NDArray[np.floating[Any]] | None = None
    endog: NDArray[np.floating[Any]] | None = None


def _make_svar_results(
    K: int = 3,
    T: int = 200,
    p: int = 2,
    seed: int = 42,
) -> Any:
    """Create mock SVAR results for testing HD."""
    from chronobox.models.svar import SVAR

    rng = np.random.default_rng(seed)

    # Generate stable VAR coefficients
    coefs = np.zeros((p, K, K))
    for s in range(p):
        coefs[s] = rng.normal(0, 0.05 / (s + 1), (K, K))

    # Generate data
    Sigma = np.eye(K) * 0.1
    Y = np.zeros((T + p + 100, K))
    for t in range(p, T + p + 100):
        for s in range(p):
            Y[t] += Y[t - s - 1] @ coefs[s].T
        Y[t] += rng.multivariate_normal(np.zeros(K), Sigma)
    Y = Y[100:]

    # Compute residuals
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

    svar = SVAR(var_res, method="cholesky")
    return svar.fit()


class TestHistoricalDecomposition:
    """Tests for Historical Decomposition."""

    def test_hd_sums_to_observed(self) -> None:
        """base + sum(HD_k) should equal observed values (tol=1e-8)."""
        from chronobox.analysis.hd import HistoricalDecomposition

        svar_res = _make_svar_results(K=3, T=200, p=2)
        hd = HistoricalDecomposition(svar_res)

        assert hd.result.verify_decomposition(atol=1e-8), (
            "HD does not sum to observed values"
        )

    def test_hd_shape(self) -> None:
        """HD should have shape (T, K_shocks, K_vars)."""
        from chronobox.analysis.hd import HistoricalDecomposition

        svar_res = _make_svar_results(K=3, T=200, p=2)
        hd = HistoricalDecomposition(svar_res)

        T = svar_res.n_obs
        K = svar_res.k_vars
        assert hd.decomposition.shape == (T, K, K)

    def test_hd_base_shape(self) -> None:
        """Base should have shape (T, K)."""
        from chronobox.analysis.hd import HistoricalDecomposition

        svar_res = _make_svar_results(K=3, T=200, p=2)
        hd = HistoricalDecomposition(svar_res)

        assert hd.base.shape == (svar_res.n_obs, svar_res.k_vars)

    def test_contribution_of_shock(self) -> None:
        """contribution_of_shock should return correct slice."""
        from chronobox.analysis.hd import HistoricalDecomposition

        svar_res = _make_svar_results(K=3, T=200, p=2)
        hd = HistoricalDecomposition(svar_res)

        for k in range(3):
            contrib = hd.result.contribution_of_shock(k)
            assert contrib.shape == (svar_res.n_obs, 3)
            np.testing.assert_array_equal(contrib, hd.decomposition[:, k, :])

    def test_to_dataframe(self) -> None:
        """to_dataframe should return a DataFrame with correct columns."""
        from chronobox.analysis.hd import HistoricalDecomposition

        svar_res = _make_svar_results(K=2, T=100, p=1)
        hd = HistoricalDecomposition(svar_res, shock_names=["Supply", "Demand"])

        df = hd.to_dataframe(variable=0)
        assert "Supply" in df.columns
        assert "Demand" in df.columns
        assert "Base" in df.columns
        assert "Observed" in df.columns

    def test_hd_plot_no_error(self) -> None:
        """plot() should execute without error."""
        import matplotlib

        matplotlib.use("Agg")
        from chronobox.analysis.hd import HistoricalDecomposition

        svar_res = _make_svar_results(K=2, T=100, p=1)
        hd = HistoricalDecomposition(svar_res)

        fig = hd.plot(variable=0, stacked=True)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_hd_plot_unstacked(self) -> None:
        """plot(stacked=False) should execute without error."""
        import matplotlib

        matplotlib.use("Agg")
        from chronobox.analysis.hd import HistoricalDecomposition

        svar_res = _make_svar_results(K=2, T=100, p=1)
        hd = HistoricalDecomposition(svar_res)

        fig = hd.plot(variable=0, stacked=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_hd_different_k(self) -> None:
        """HD should work for different K values."""
        from chronobox.analysis.hd import HistoricalDecomposition

        for K in [1, 2, 4]:
            svar_res = _make_svar_results(K=K, T=100, p=1, seed=K * 10)
            hd = HistoricalDecomposition(svar_res)
            assert hd.result.verify_decomposition(atol=1e-7)
