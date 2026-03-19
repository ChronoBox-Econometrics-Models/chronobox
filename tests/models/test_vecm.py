"""Tests for VECM model and Johansen procedure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronobox.models.vecm import VECM, JohansenResults, VECMResults


@pytest.fixture
def canada_data() -> pd.DataFrame:
    """Load Canada dataset as DataFrame."""
    data_path = (
        Path(__file__).parent.parent.parent
        / "chronobox"
        / "datasets"
        / "data"
        / "macro"
        / "canada.csv"
    )
    df = pd.read_csv(data_path)
    return df[["e", "prod", "rw", "U"]]


@pytest.fixture
def canada_array(canada_data: pd.DataFrame) -> np.ndarray:
    """Canada dataset as numpy array."""
    return canada_data.to_numpy(dtype=np.float64)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG."""
    return np.random.default_rng(42)


@pytest.fixture
def cointegrated_data(rng: np.random.Generator) -> np.ndarray:
    """Simulate a cointegrated system with known rank=1.

    DGP:
        x1_t = x1_{t-1} + e1_t  (random walk)
        x2_t = x1_t + e2_t      (cointegrated with x1, beta = [1, -1])
        x3_t = x3_{t-1} + e3_t  (independent random walk)
    """
    t = 300
    e1 = rng.standard_normal(t) * 0.5
    rng.standard_normal(t) * 0.3
    e3 = rng.standard_normal(t) * 0.5

    x1 = np.cumsum(e1)
    x3 = np.cumsum(e3)
    x2 = x1 + rng.standard_normal(t) * 0.2  # x2 - x1 is stationary

    return np.column_stack([x1, x2, x3])


class TestJohansen:
    """Tests for the Johansen cointegration test."""

    def test_johansen_trace_canada(self, canada_data: pd.DataFrame) -> None:
        """Johansen trace test on Canada dataset."""
        model = VECM(lags=2, deterministic="ci")
        johansen = model.johansen_test(canada_data)

        assert isinstance(johansen, JohansenResults)
        assert len(johansen.eigenvalues) == 4
        assert johansen.trace_stat.shape == (4,)

        # Eigenvalues should be in [0, 1) and sorted descending
        assert np.all(johansen.eigenvalues >= 0)
        assert np.all(johansen.eigenvalues < 1)
        assert np.all(johansen.eigenvalues[:-1] >= johansen.eigenvalues[1:] - 1e-10)

        # Trace stats should be non-negative and decreasing
        assert np.all(johansen.trace_stat >= -1e-10)

        # Rank should be determined (>= 0)
        assert johansen.rank_trace >= 0
        assert johansen.rank_trace <= 4

    def test_johansen_maxeig(self, canada_data: pd.DataFrame) -> None:
        """Max-eigenvalue test should be consistent with trace."""
        model = VECM(lags=2, deterministic="ci")
        johansen = model.johansen_test(canada_data)

        assert johansen.max_eig_stat.shape == (4,)
        assert np.all(johansen.max_eig_stat >= -1e-10)
        assert johansen.rank_maxeig >= 0
        assert johansen.rank_maxeig <= 4

    def test_johansen_eigenvalue_range(self, canada_data: pd.DataFrame) -> None:
        """Eigenvalues should be in [0, 1)."""
        model = VECM(lags=2, deterministic="co")
        johansen = model.johansen_test(canada_data)

        for ev in johansen.eigenvalues:
            assert 0 <= ev < 1, f"Eigenvalue {ev} outside [0, 1)"

    def test_johansen_critical_values(self, canada_data: pd.DataFrame) -> None:
        """Critical values should be present and ordered."""
        model = VECM(lags=2, deterministic="ci")
        johansen = model.johansen_test(canada_data)

        assert johansen.trace_crit.shape == (4, 3)
        assert johansen.max_eig_crit.shape == (4, 3)

        # 90% < 95% < 99%
        for r in range(4):
            assert johansen.trace_crit[r, 0] < johansen.trace_crit[r, 1]
            assert johansen.trace_crit[r, 1] < johansen.trace_crit[r, 2]

    def test_johansen_summary(self, canada_data: pd.DataFrame) -> None:
        """Johansen summary should produce formatted output."""
        model = VECM(lags=2, deterministic="ci")
        johansen = model.johansen_test(canada_data)
        summary = johansen.summary()

        assert isinstance(summary, str)
        assert "Johansen" in summary
        assert "Trace" in summary
        assert len(summary) > 200

    def test_johansen_cointegrated_data(self, cointegrated_data: np.ndarray) -> None:
        """Johansen should detect cointegration in simulated data."""
        model = VECM(lags=2, deterministic="co")
        johansen = model.johansen_test(cointegrated_data)

        # Should detect at least 1 cointegrating relationship
        assert johansen.rank_trace >= 1, (
            f"Trace test failed to detect cointegration: rank={johansen.rank_trace}"
        )


class TestVECM:
    """Tests for VECM estimation."""

    def test_vecm_alpha_beta_dims(self, canada_data: pd.DataFrame) -> None:
        """alpha and beta should have correct dimensions (K x r)."""
        model = VECM(lags=2, coint_rank=1, deterministic="ci")
        results = model.fit(canada_data)

        assert isinstance(results, VECMResults)
        assert results.alpha.shape == (4, 1)
        # beta may be (K+1, r) for 'ci' model (includes restricted constant)
        assert results.beta.shape[0] >= 4
        assert results.beta.shape[1] == 1

    def test_vecm_beta_normalization(self, canada_data: pd.DataFrame) -> None:
        """First element of beta should be normalized to 1."""
        model = VECM(lags=2, coint_rank=1, deterministic="ci")
        results = model.fit(canada_data)

        # beta[0, 0] should be 1 (normalized)
        np.testing.assert_allclose(
            results.beta[0, 0],
            1.0,
            atol=1e-10,
            err_msg="Beta not normalized: beta[0,0] != 1",
        )

    def test_vecm_gamma_count(self, canada_data: pd.DataFrame) -> None:
        """Number of gamma matrices should be p-1."""
        model = VECM(lags=2, coint_rank=1, deterministic="ci")
        results = model.fit(canada_data)

        # lags=2 -> p-1=1 gamma matrix
        assert len(results.gamma) == 1
        assert results.gamma[0].shape == (4, 4)

    def test_vecm_gamma_count_lags3(self, canada_data: pd.DataFrame) -> None:
        """VECM with lags=3 should have 2 gamma matrices."""
        model = VECM(lags=3, coint_rank=1, deterministic="ci")
        results = model.fit(canada_data)

        assert len(results.gamma) == 2
        for g in results.gamma:
            assert g.shape == (4, 4)

    def test_vecm_residual_cov(self, canada_data: pd.DataFrame) -> None:
        """Sigma_u should be symmetric and positive definite."""
        model = VECM(lags=2, coint_rank=1, deterministic="ci")
        results = model.fit(canada_data)

        np.testing.assert_allclose(results.sigma_u, results.sigma_u.T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(results.sigma_u)
        assert np.all(eigvals > 0)

    def test_vecm_pi_matrix(self, canada_data: pd.DataFrame) -> None:
        """Pi = alpha @ beta' should have correct shape and rank."""
        model = VECM(lags=2, coint_rank=1, deterministic="ci")
        results = model.fit(canada_data)

        assert results.pi.shape == (4, 4)
        # Pi should have rank <= r
        rank_pi = np.linalg.matrix_rank(results.pi, tol=1e-8)
        assert rank_pi <= 1, f"Pi has rank {rank_pi}, expected <= 1"

    def test_vecm_summary(self, canada_data: pd.DataFrame) -> None:
        """VECM summary should produce formatted output without errors."""
        model = VECM(lags=2, coint_rank=1, deterministic="ci")
        results = model.fit(canada_data)
        summary = results.summary()

        assert isinstance(summary, str)
        assert "VECM" in summary
        assert "Alpha" in summary
        assert "Beta" in summary
        assert len(summary) > 300

    def test_vecm_rank2(self, canada_data: pd.DataFrame) -> None:
        """VECM with coint_rank=2 should work."""
        model = VECM(lags=2, coint_rank=2, deterministic="ci")
        results = model.fit(canada_data)

        assert results.coint_rank == 2
        assert results.alpha.shape == (4, 2)
        assert results.beta.shape[1] == 2

    def test_5_deterministic_models(self, canada_data: pd.DataFrame) -> None:
        """All 5 deterministic models should run without error."""
        for det in ("nc", "ci", "co", "li", "lo"):
            model = VECM(lags=2, coint_rank=1, deterministic=det)
            results = model.fit(canada_data)

            assert isinstance(results, VECMResults), (
                f"Deterministic model '{det}' failed"
            )
            assert results.alpha.shape[1] == 1
            assert results.neqs == 4

    def test_vecm_dataframe_names(self, canada_data: pd.DataFrame) -> None:
        """VECM should preserve variable names from DataFrame."""
        model = VECM(lags=2, coint_rank=1, deterministic="ci")
        results = model.fit(canada_data)

        assert results.names == ["e", "prod", "rw", "U"]

    def test_vecm_eigenvalues(self, canada_data: pd.DataFrame) -> None:
        """Eigenvalues should be in [0, 1) and sorted descending."""
        model = VECM(lags=2, coint_rank=1, deterministic="ci")
        results = model.fit(canada_data)

        assert len(results.eigenvalues) == 4
        assert np.all(results.eigenvalues >= 0)
        assert np.all(results.eigenvalues < 1)

    def test_vecm_auto_rank(self, cointegrated_data: np.ndarray) -> None:
        """VECM with coint_rank=None should auto-detect rank."""
        model = VECM(lags=2, coint_rank=None, deterministic="co")
        results = model.fit(cointegrated_data)

        assert results.coint_rank >= 1

    def test_vecm_invalid_deterministic(self) -> None:
        """Invalid deterministic spec should raise ValueError."""
        with pytest.raises(ValueError, match="deterministic"):
            VECM(lags=2, deterministic="invalid")

    def test_vecm_residuals_shape(self, canada_data: pd.DataFrame) -> None:
        """Residuals should have correct shape."""
        model = VECM(lags=2, coint_rank=1, deterministic="ci")
        results = model.fit(canada_data)

        # T_eff = T - lags = 84 - 2 = 82
        assert results.resid.shape == (82, 4)
