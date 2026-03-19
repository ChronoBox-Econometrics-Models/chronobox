"""Tests for FAVAR (Factor-Augmented VAR)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray


def _make_favar_data(
    N: int = 50,
    K: int = 3,
    M: int = 1,
    T: int = 300,
    seed: int = 42,
) -> tuple[NDArray, NDArray, NDArray]:
    """Generate synthetic FAVAR data.

    Returns
    -------
    X : ndarray (T, N) - panel data
    Y : ndarray (T, M) - policy variables
    F_true : ndarray (T, K) - true factors (for validation)
    """
    rng = np.random.default_rng(seed)

    # True factors (AR(1) process)
    F_true = np.zeros((T, K))
    for t in range(1, T):
        F_true[t] = 0.5 * F_true[t - 1] + rng.normal(0, 1, K)

    # Policy variable (AR(1) with factor influence)
    Y = np.zeros((T, M))
    for t in range(1, T):
        Y[t] = 0.3 * Y[t - 1] + 0.2 * F_true[t, 0] + rng.normal(0, 0.5, M)

    # Panel data: X = Lambda * F + Lambda_y * Y + noise
    Lambda_f = rng.normal(0, 1, (N, K))
    Lambda_y = rng.normal(0, 0.5, (N, M))

    X = F_true @ Lambda_f.T + Y @ Lambda_y.T + rng.normal(0, 0.5, (T, N))

    return X, Y, F_true


class TestFAVARTwoStep:
    """Tests for two-step FAVAR estimation."""

    def test_two_step_basic(self) -> None:
        """Two-step FAVAR should produce valid results."""
        from chronobox.models.favar import FAVAR

        X, Y, _ = _make_favar_data()
        favar = FAVAR(n_factors=3, lags=2, method="two_step")
        result = favar.fit(X, Y)

        assert result.method == "two_step"
        assert result.factors.shape[1] == 3
        assert result.n_factors == 3
        assert result.n_policy == 1
        assert result.n_panel == 50

    def test_factors_shape(self) -> None:
        """Extracted factors should have correct shape."""
        from chronobox.models.favar import FAVAR

        X, Y, _ = _make_favar_data(N=30, K=2, T=200)
        favar = FAVAR(n_factors=2, lags=1)
        result = favar.fit(X, Y)

        assert result.factors.shape == (200, 2)
        assert result.loadings.shape == (30, 2)

    def test_factors_explain_variance(self) -> None:
        """Factors should explain a meaningful fraction of variance."""
        from chronobox.models.favar import FAVAR

        X, Y, _ = _make_favar_data(N=50, K=3, T=500, seed=42)
        favar = FAVAR(n_factors=3, lags=2)
        result = favar.fit(X, Y)

        # With 3 true factors, extracted factors should explain > 30% variance
        assert result.total_explained_variance > 0.3, (
            f"Factors explain only {result.total_explained_variance:.1%} of variance"
        )

    def test_var_coefs_shape(self) -> None:
        """VAR coefficient matrices should have correct shape."""
        from chronobox.models.favar import FAVAR

        X, Y, _ = _make_favar_data(N=20, K=2, M=1, T=200)
        favar = FAVAR(n_factors=2, lags=3)
        result = favar.fit(X, Y)

        KM = 2 + 1  # K + M
        assert result.var_coefs.shape == (3, KM, KM)
        assert result.var_sigma.shape == (KM, KM)

    def test_irf_shape(self) -> None:
        """IRF should have correct shape."""
        from chronobox.models.favar import FAVAR

        X, Y, _ = _make_favar_data(N=20, K=2, M=1, T=200)
        favar = FAVAR(n_factors=2, lags=2)
        result = favar.fit(X, Y)

        irf = result.irf(periods=20)
        KM = 2 + 1
        assert irf.shape == (21, KM, KM)

    def test_irf_panel_shape(self) -> None:
        """Panel IRF should have correct shape."""
        from chronobox.models.favar import FAVAR

        N = 30
        X, Y, _ = _make_favar_data(N=N, K=2, M=1, T=200)
        favar = FAVAR(n_factors=2, lags=2)
        result = favar.fit(X, Y)

        irf_panel = result.irf_panel(periods=20)
        KM = 2 + 1
        assert irf_panel.shape == (21, N, KM)

    def test_irf_impact_nonzero(self) -> None:
        """At impact (h=0), IRF should be non-zero."""
        from chronobox.models.favar import FAVAR

        X, Y, _ = _make_favar_data(N=20, K=2, M=1, T=200)
        favar = FAVAR(n_factors=2, lags=2)
        result = favar.fit(X, Y)

        irf = result.irf(periods=10)
        assert np.any(np.abs(irf[0]) > 1e-10)

    def test_loadings_not_zero(self) -> None:
        """Factor loadings should be non-trivial."""
        from chronobox.models.favar import FAVAR

        X, Y, _ = _make_favar_data(N=50, K=3, T=300)
        favar = FAVAR(n_factors=3, lags=2)
        result = favar.fit(X, Y)

        assert np.std(result.loadings) > 0.01
        assert np.std(result.loadings_y) > 0.01

    def test_no_remove_y(self) -> None:
        """FAVAR without removing Y from factors should work."""
        from chronobox.models.favar import FAVAR

        X, Y, _ = _make_favar_data(N=20, K=2, T=200)
        favar = FAVAR(n_factors=2, lags=1, remove_y_from_factors=False)
        result = favar.fit(X, Y)

        assert result.factors.shape == (200, 2)


class TestFAVAREdgeCases:
    """Edge case tests."""

    def test_unknown_method_raises(self) -> None:
        """Unknown method should raise ValueError."""
        from chronobox.models.favar import FAVAR

        with pytest.raises(ValueError, match="Unknown method"):
            FAVAR(method="invalid")

    def test_mismatched_T_raises(self) -> None:
        """X and Y with different T should raise ValueError."""
        from chronobox.models.favar import FAVAR

        X = np.random.randn(100, 20)
        Y = np.random.randn(50, 1)
        favar = FAVAR(n_factors=2, lags=1)
        with pytest.raises(ValueError, match="same number of rows"):
            favar.fit(X, Y)

    def test_insufficient_data_raises(self) -> None:
        """Too few observations should raise ValueError."""
        from chronobox.models.favar import FAVAR

        X = np.random.randn(5, 20)
        Y = np.random.randn(5, 1)
        favar = FAVAR(n_factors=2, lags=10)
        with pytest.raises(ValueError, match="Not enough observations"):
            favar.fit(X, Y)

    def test_1d_policy_variable(self) -> None:
        """1D policy variable (T,) should be auto-reshaped."""
        from chronobox.models.favar import FAVAR

        X = np.random.randn(200, 20)
        Y = np.random.randn(200)  # 1D
        favar = FAVAR(n_factors=2, lags=1)
        result = favar.fit(X, Y)
        assert result.n_policy == 1

    def test_multiple_policy_variables(self) -> None:
        """FAVAR should work with multiple policy variables."""
        from chronobox.models.favar import FAVAR

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 30))
        Y = rng.normal(0, 1, (200, 3))
        favar = FAVAR(n_factors=2, lags=1)
        result = favar.fit(X, Y)

        assert result.n_policy == 3
        KM = 2 + 3
        assert result.var_coefs.shape == (1, KM, KM)
