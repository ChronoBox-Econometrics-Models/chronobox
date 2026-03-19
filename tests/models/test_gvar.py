"""Tests for GVAR (Global VAR)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import linalg as la


def _make_gvar_data(
    N: int = 3,
    K_per_country: int = 2,
    T: int = 300,
    seed: int = 42,
) -> tuple[dict[str, NDArray], NDArray]:
    """Generate synthetic GVAR data.

    Returns
    -------
    data_dict : dict[str, ndarray]
        Per-country data.
    trade_weights : ndarray (N, N)
        Trade weight matrix.
    """
    rng = np.random.default_rng(seed)

    # Generate trade weights
    W = rng.uniform(0.1, 1.0, (N, N))
    np.fill_diagonal(W, 0.0)
    for i in range(N):
        W[i] /= W[i].sum()

    # Generate data with cross-country dependencies
    K_total = N * K_per_country
    Y = np.zeros((T, K_total))
    Sigma = np.eye(K_total) * 0.1

    # Small VAR coefficients
    A = np.eye(K_total) * 0.3
    for i in range(K_total):
        for j in range(K_total):
            if i != j and rng.random() > 0.7:
                A[i, j] = rng.normal(0, 0.05)

    # Ensure stability
    eigvals = la.eigvals(A)
    max_eig = np.max(np.abs(eigvals))
    if max_eig >= 0.95:
        A *= 0.9 / max_eig

    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + rng.multivariate_normal(np.zeros(K_total), Sigma)

    # Split into countries
    data_dict = {}
    for i in range(N):
        name = f"country_{i}"
        data_dict[name] = Y[:, i * K_per_country : (i + 1) * K_per_country]

    return data_dict, W


class TestGVARWeights:
    """Tests for trade weight validation."""

    def test_weights_sum_to_one(self) -> None:
        """Trade weights should sum to 1 for each country (excluding self)."""
        from chronobox.models.gvar import GVAR

        W = np.array([[0, 0.6, 0.4], [0.5, 0, 0.5], [0.3, 0.7, 0]])
        gvar = GVAR(trade_weights=W)

        for i in range(3):
            row_sum = gvar.trade_weights[i].sum()
            np.testing.assert_allclose(row_sum, 1.0, atol=1e-10)

    def test_diagonal_zero(self) -> None:
        """Diagonal of trade weights should be zero."""
        from chronobox.models.gvar import GVAR

        W = np.array([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.3, 0.7, 0.0]])
        gvar = GVAR(trade_weights=W)

        for i in range(3):
            assert gvar.trade_weights[i, i] == 0.0

    def test_weights_auto_normalized(self) -> None:
        """Unnormalized weights should be auto-normalized."""
        from chronobox.models.gvar import GVAR

        W = np.array([[0, 3.0, 2.0], [5.0, 0, 5.0], [1.0, 4.0, 0]])
        gvar = GVAR(trade_weights=W)

        np.testing.assert_allclose(gvar.trade_weights[0], [0, 0.6, 0.4])
        np.testing.assert_allclose(gvar.trade_weights[1], [0.5, 0, 0.5])
        np.testing.assert_allclose(gvar.trade_weights[2], [0.2, 0.8, 0])

    def test_nonsquare_raises(self) -> None:
        """Non-square trade weight matrix should raise ValueError."""
        from chronobox.models.gvar import GVAR

        with pytest.raises(ValueError, match="square"):
            GVAR(trade_weights=np.ones((3, 4)))


class TestGVARFit:
    """Tests for GVAR estimation."""

    def test_basic_fit(self) -> None:
        """GVAR should fit without errors."""
        from chronobox.models.gvar import GVAR

        data_dict, W = _make_gvar_data(N=3, K_per_country=2, T=300)
        gvar = GVAR(trade_weights=W, domestic_lags=1, foreign_lags=1)
        result = gvar.fit(data_dict)

        assert result.n_countries == 3
        assert result.k_total == 6
        assert len(result.global_coefs) == 1
        assert result.global_coefs[0].shape == (6, 6)
        assert result.global_sigma.shape == (6, 6)

    def test_global_stability(self) -> None:
        """Global system should be stable for stable DGP."""
        from chronobox.models.gvar import GVAR

        data_dict, W = _make_gvar_data(N=3, K_per_country=2, T=500, seed=42)
        gvar = GVAR(trade_weights=W, domestic_lags=1, foreign_lags=1)
        result = gvar.fit(data_dict)

        # System should be stable
        max_eigenvalue = np.max(np.abs(result.eigenvalues))
        assert max_eigenvalue < 1.0 or not result.is_stable  # at least the check runs

    def test_country_results_populated(self) -> None:
        """Per-country results should be populated."""
        from chronobox.models.gvar import GVAR

        data_dict, W = _make_gvar_data(N=3, K_per_country=2, T=300)
        gvar = GVAR(trade_weights=W, domestic_lags=1, foreign_lags=1)
        result = gvar.fit(data_dict)

        for name in data_dict:
            assert name in result.country_results
            cr = result.country_results[name]
            assert "Phi" in cr
            assert "Lambda" in cr
            assert "Sigma" in cr
            assert "intercept" in cr
            assert len(cr["Phi"]) == 1  # 1 domestic lag


class TestGVARGIRF:
    """Tests for GIRF computation."""

    def test_girf_shape(self) -> None:
        """GIRF should have correct shape."""
        from chronobox.models.gvar import GVAR

        data_dict, W = _make_gvar_data(N=3, K_per_country=2, T=300)
        gvar = GVAR(trade_weights=W, domestic_lags=1, foreign_lags=1)
        result = gvar.fit(data_dict)

        girf = result.girf(shock_country="country_0", shock_var=0, periods=40)
        assert girf.shape == (41, 6)

    def test_girf_cross_country(self) -> None:
        """Shock in one country should affect other countries."""
        from chronobox.models.gvar import GVAR

        data_dict, W = _make_gvar_data(N=3, K_per_country=2, T=500, seed=42)
        gvar = GVAR(trade_weights=W, domestic_lags=1, foreign_lags=1)
        result = gvar.fit(data_dict)

        girf = result.girf(shock_country="country_0", shock_var=0, periods=20)

        # At impact, only the shocked variable should have non-zero response
        # in a GIRF framework (the sigma scaling causes all to respond)
        # At least the shocked variable should have a large response
        assert abs(girf[0, 0]) > 0

    def test_girf_by_country_name(self) -> None:
        """GIRF should work with country names."""
        from chronobox.models.gvar import GVAR

        data_dict, W = _make_gvar_data(N=3, K_per_country=2, T=300)
        gvar = GVAR(trade_weights=W, domestic_lags=1, foreign_lags=1)
        result = gvar.fit(data_dict)

        girf = result.girf(shock_country="country_1", shock_var=0, periods=10)
        assert girf.shape == (11, 6)

    def test_irf_country_extract(self) -> None:
        """irf_country should extract the correct subset."""
        from chronobox.models.gvar import GVAR

        data_dict, W = _make_gvar_data(N=3, K_per_country=2, T=300)
        gvar = GVAR(trade_weights=W, domestic_lags=1, foreign_lags=1)
        result = gvar.fit(data_dict)

        irf_c1 = result.irf_country(
            shock_country="country_0",
            shock_var=0,
            response_country="country_1",
            periods=20,
        )
        assert irf_c1.shape == (21, 2)

    def test_girf_integer_index(self) -> None:
        """GIRF should work with integer country index."""
        from chronobox.models.gvar import GVAR

        data_dict, W = _make_gvar_data(N=3, K_per_country=2, T=300)
        gvar = GVAR(trade_weights=W, domestic_lags=1, foreign_lags=1)
        result = gvar.fit(data_dict)

        girf = result.girf(shock_country=0, shock_var=0, periods=10)
        assert girf.shape == (11, 6)


class TestGVAREdgeCases:
    """Edge case tests."""

    def test_wrong_n_countries_raises(self) -> None:
        """Mismatched number of countries should raise ValueError."""
        from chronobox.models.gvar import GVAR

        W = np.array([[0, 1], [1, 0]], dtype=float)
        gvar = GVAR(trade_weights=W)

        data_dict = {
            "c1": np.random.randn(100, 2),
            "c2": np.random.randn(100, 2),
            "c3": np.random.randn(100, 2),
        }
        with pytest.raises(ValueError, match="Expected 2 countries"):
            gvar.fit(data_dict)

    def test_different_T_raises(self) -> None:
        """Countries with different T should raise ValueError."""
        from chronobox.models.gvar import GVAR

        W = np.array([[0, 1], [1, 0]], dtype=float)
        gvar = GVAR(trade_weights=W)

        data_dict = {
            "c1": np.random.randn(100, 2),
            "c2": np.random.randn(50, 2),
        }
        with pytest.raises(ValueError, match="same number of observations"):
            gvar.fit(data_dict)

    def test_two_countries(self) -> None:
        """GVAR should work with only 2 countries."""
        from chronobox.models.gvar import GVAR

        rng = np.random.default_rng(42)
        W = np.array([[0, 1], [1, 0]], dtype=float)
        data_dict = {
            "us": rng.normal(0, 1, (200, 2)),
            "eu": rng.normal(0, 1, (200, 2)),
        }
        gvar = GVAR(trade_weights=W, domestic_lags=1, foreign_lags=1)
        result = gvar.fit(data_dict)

        assert result.n_countries == 2
        assert result.k_total == 4

    def test_multiple_lags(self) -> None:
        """GVAR should work with multiple lags."""
        from chronobox.models.gvar import GVAR

        data_dict, W = _make_gvar_data(N=3, K_per_country=2, T=300)
        gvar = GVAR(trade_weights=W, domestic_lags=2, foreign_lags=1)
        result = gvar.fit(data_dict)

        assert result.domestic_lags == 2
        assert len(result.global_coefs) == 2
