"""Tests for the Diebold-Yilmaz Spillover Index."""

import numpy as np
import pytest

from chronobox.analysis.spillover import (
    RollingSpilloverResult,
    SpilloverIndex,
    _fit_var_ols,  # pyright: ignore[reportPrivateUsage]
)


class TestSpilloverIndex:
    """Tests for SpilloverIndex."""

    def setup_method(self) -> None:
        """Set up test data."""
        np.random.seed(42)
        self.T = 500
        self.K = 4
        # Generate VAR(1) data with some cross-variable dependence
        A = np.array([
            [0.5, 0.1, 0.0, 0.0],
            [0.1, 0.4, 0.1, 0.0],
            [0.0, 0.1, 0.6, 0.1],
            [0.0, 0.0, 0.1, 0.5],
        ])
        data = np.zeros((self.T, self.K))
        for t in range(1, self.T):
            data[t] = A @ data[t - 1] + np.random.randn(self.K)
        self.data = data

    def test_total_range_0_100(self) -> None:
        """Total spillover should be in [0, 100]."""
        sp = SpilloverIndex(lags=2, horizon=10)
        result = sp.fit(self.data)
        assert 0 <= result.total_spillover <= 100, (
            f"Total spillover {result.total_spillover} out of range [0, 100]"
        )

    def test_net_sums_to_zero(self) -> None:
        """Net spillover should sum to approximately zero."""
        sp = SpilloverIndex(lags=2, horizon=10)
        result = sp.fit(self.data)
        np.testing.assert_allclose(
            np.sum(result.net_spillover),
            0.0,
            atol=1e-10,
            err_msg="Net spillover should sum to zero",
        )

    def test_table_rows_sum_to_one(self) -> None:
        """Each row of the normalized FEVD table should sum to 1."""
        sp = SpilloverIndex(lags=2, horizon=10)
        result = sp.fit(self.data)
        row_sums = result.fevd_table.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_diagonal_own_variance(self) -> None:
        """Diagonal should represent own variance shares (typically largest)."""
        sp = SpilloverIndex(lags=2, horizon=10)
        result = sp.fit(self.data)
        diag = np.diag(result.fevd_table)
        # Own variance should be positive
        assert np.all(diag > 0), "Diagonal elements should be positive"
        # For well-behaved VAR, own contribution should be substantial
        assert np.all(diag > 0.1), "Own variance share seems too small"

    def test_pairwise_antisymmetric(self) -> None:
        """Pairwise spillover should be antisymmetric: S_ij = -S_ji."""
        sp = SpilloverIndex(lags=2, horizon=10)
        result = sp.fit(self.data)
        pw = result.pairwise_spillover
        np.testing.assert_allclose(pw, -pw.T, atol=1e-10)

    def test_output_shapes(self) -> None:
        """Output shapes should be correct."""
        sp = SpilloverIndex(lags=2, horizon=10)
        result = sp.fit(self.data)
        assert result.fevd_table.shape == (self.K, self.K)
        assert result.directional_from.shape == (self.K,)
        assert result.directional_to.shape == (self.K,)
        assert result.net_spillover.shape == (self.K,)
        assert result.pairwise_spillover.shape == (self.K, self.K)

    def test_rolling_varies(self) -> None:
        """Rolling spillover should vary over time."""
        sp = SpilloverIndex(lags=1, horizon=5)
        result = sp.rolling(self.data, window=100)
        assert isinstance(result, RollingSpilloverResult)
        assert len(result.total_spillover) == self.T - 100 + 1
        # Should have some variation
        assert np.std(result.total_spillover[~np.isnan(result.total_spillover)]) > 0

    def test_rolling_shapes(self) -> None:
        """Rolling spillover output shapes should be correct."""
        window = 150
        sp = SpilloverIndex(lags=1, horizon=5)
        result = sp.rolling(self.data, window=window)
        n_windows = self.T - window + 1
        assert result.total_spillover.shape == (n_windows,)
        assert result.directional_from.shape == (n_windows, self.K)
        assert result.directional_to.shape == (n_windows, self.K)
        assert result.net_spillover.shape == (n_windows, self.K)

    def test_summary(self) -> None:
        """Summary should produce non-empty output."""
        sp = SpilloverIndex(lags=2, horizon=10)
        result = sp.fit(self.data)
        s = result.summary()
        assert len(s) > 50
        assert "Total Spillover" in s

    def test_independent_variables_low_spillover(self) -> None:
        """Independent variables should have low spillover."""
        np.random.seed(123)
        indep_data = np.random.randn(500, 4)  # independent columns
        sp = SpilloverIndex(lags=1, horizon=10)
        result = sp.fit(indep_data)
        # Total spillover should be relatively low for independent data
        # (not exactly zero due to estimation noise)
        assert result.total_spillover < 50, (
            f"Total spillover {result.total_spillover} too high for independent data"
        )

    def test_highly_correlated_high_spillover(self) -> None:
        """Highly correlated variables should have higher spillover."""
        np.random.seed(123)
        base = np.random.randn(500)
        corr_data = np.column_stack([
            base + 0.1 * np.random.randn(500),
            base + 0.1 * np.random.randn(500),
            base + 0.1 * np.random.randn(500),
        ])
        sp = SpilloverIndex(lags=1, horizon=10)
        result = sp.fit(corr_data)
        assert result.total_spillover > 20, (
            f"Expected higher spillover for correlated data, got {result.total_spillover}"
        )

    def test_invalid_inputs(self) -> None:
        """Should raise for invalid inputs."""
        sp = SpilloverIndex(lags=2, horizon=10)
        with pytest.raises(ValueError, match="2-D"):
            sp.fit(np.random.randn(100))
        with pytest.raises(ValueError, match="at least 2"):
            sp.fit(np.random.randn(100, 1))
        with pytest.raises(ValueError, match="Not enough"):
            sp.fit(np.random.randn(5, 3))

    def test_var_ols(self) -> None:
        """Internal VAR OLS should produce valid results."""
        A_list, resid, sigma = _fit_var_ols(self.data, lags=2)
        assert A_list.shape == (2, self.K, self.K)
        assert resid.shape == (self.T - 2, self.K)
        assert sigma.shape == (self.K, self.K)
        # Sigma should be positive semi-definite
        eigvals = np.linalg.eigvalsh(sigma)
        assert np.all(eigvals >= -1e-10)

    def test_constructor_validation(self) -> None:
        """Constructor should validate parameters."""
        with pytest.raises(ValueError):
            SpilloverIndex(lags=0)
        with pytest.raises(ValueError):
            SpilloverIndex(horizon=0)
