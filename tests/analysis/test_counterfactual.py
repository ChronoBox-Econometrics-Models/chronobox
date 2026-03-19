"""Tests for Counterfactual analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
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


def _make_hd(K: int = 3, T: int = 200, p: int = 2, seed: int = 42) -> Any:
    """Create HistoricalDecomposition for testing."""
    from chronobox.analysis.hd import HistoricalDecomposition
    from chronobox.models.svar import SVAR

    rng = np.random.default_rng(seed)
    coefs = np.zeros((p, K, K))
    for s in range(p):
        coefs[s] = rng.normal(0, 0.05 / (s + 1), (K, K))

    Sigma = np.eye(K) * 0.1
    Y = np.zeros((T + p + 100, K))
    for t in range(p, T + p + 100):
        for s in range(p):
            Y[t] += Y[t - s - 1] @ coefs[s].T
        Y[t] += rng.multivariate_normal(np.zeros(K), Sigma)
    Y = Y[100:]

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
    svar_res = svar.fit()
    return HistoricalDecomposition(svar_res)


class TestCounterfactual:
    """Tests for Counterfactual analysis."""

    def test_without_shock_exact(self) -> None:
        """Y_cf = Y - HD_k should hold exactly."""
        from chronobox.analysis.counterfactual import Counterfactual

        hd = _make_hd(K=3, T=200, p=2)
        cf = Counterfactual(hd)

        for k in range(3):
            y_cf = cf.without_shock(k)
            expected = hd.result.observed - hd.decomposition[:, k, :]
            np.testing.assert_allclose(y_cf, expected, atol=1e-12)

    def test_all_shocks_removed_equals_base(self) -> None:
        """Removing all shocks should yield the base forecast."""
        from chronobox.analysis.counterfactual import Counterfactual

        hd = _make_hd(K=3, T=200, p=2)
        cf = Counterfactual(hd)

        y_base = cf.without_all_shocks()
        np.testing.assert_allclose(y_base, hd.base, atol=1e-10)

    def test_modified_shock_scale_0(self) -> None:
        """with_modified_shock(scale=0) should equal without_shock."""
        from chronobox.analysis.counterfactual import Counterfactual

        hd = _make_hd(K=2, T=100, p=1)
        cf = Counterfactual(hd)

        for k in range(2):
            y_cf_remove = cf.without_shock(k)
            y_cf_scale0 = cf.with_modified_shock(k, scale=0.0)
            np.testing.assert_allclose(y_cf_remove, y_cf_scale0, atol=1e-12)

    def test_modified_shock_scale_1(self) -> None:
        """with_modified_shock(scale=1) should equal observed."""
        from chronobox.analysis.counterfactual import Counterfactual

        hd = _make_hd(K=2, T=100, p=1)
        cf = Counterfactual(hd)

        y_cf = cf.with_modified_shock(0, scale=1.0)
        np.testing.assert_allclose(y_cf, hd.result.observed, atol=1e-12)

    def test_modified_shock_scale_half(self) -> None:
        """with_modified_shock(scale=0.5) should be halfway."""
        from chronobox.analysis.counterfactual import Counterfactual

        hd = _make_hd(K=2, T=100, p=1)
        cf = Counterfactual(hd)

        y_cf = cf.with_modified_shock(0, scale=0.5)
        y_remove = cf.without_shock(0)
        y_obs = hd.result.observed

        # y_cf should be midpoint between observed and without_shock
        expected = (y_obs + y_remove) / 2.0
        np.testing.assert_allclose(y_cf, expected, atol=1e-12)

    def test_remove_multiple_shocks(self) -> None:
        """Removing multiple shocks should work correctly."""
        from chronobox.analysis.counterfactual import Counterfactual

        hd = _make_hd(K=3, T=100, p=1)
        cf = Counterfactual(hd)

        y_cf = cf.without_shock([0, 1])
        expected = (
            hd.result.observed
            - hd.decomposition[:, 0, :]
            - hd.decomposition[:, 1, :]
        )
        np.testing.assert_allclose(y_cf, expected, atol=1e-12)

    def test_compute_result(self) -> None:
        """compute() should return CounterfactualResult."""
        from chronobox.analysis.counterfactual import Counterfactual

        hd = _make_hd(K=2, T=100, p=1)
        cf = Counterfactual(hd)

        result = cf.compute(shock_index=0, scale=0.0)
        assert result.counterfactual.shape == (hd.result.n_obs, 2)
        assert result.observed.shape == (hd.result.n_obs, 2)
        assert result.scale == 0.0

    def test_plot_no_error(self) -> None:
        """plot() should execute without error."""
        import matplotlib

        matplotlib.use("Agg")
        from chronobox.analysis.counterfactual import Counterfactual

        hd = _make_hd(K=2, T=100, p=1)
        cf = Counterfactual(hd)

        fig = cf.plot(variable=0, shock_index=0)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_accepts_hd_result_directly(self) -> None:
        """Counterfactual should accept HistoricalDecompositionResult directly."""
        from chronobox.analysis.counterfactual import Counterfactual

        hd = _make_hd(K=2, T=100, p=1)
        cf = Counterfactual(hd.result)  # Pass result, not HD object

        y_cf = cf.without_shock(0)
        assert y_cf.shape == (hd.result.n_obs, 2)

    def test_invalid_hd_raises(self) -> None:
        """Invalid input should raise TypeError."""
        from chronobox.analysis.counterfactual import Counterfactual

        with pytest.raises(TypeError):
            Counterfactual("not a valid input")
