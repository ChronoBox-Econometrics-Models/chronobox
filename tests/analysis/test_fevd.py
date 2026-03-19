"""Tests for Forecast Error Variance Decomposition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronobox.analysis.fevd import FEVD
from chronobox.models.var import VAR, VARResults


@pytest.fixture
def canada_data() -> pd.DataFrame:
    """Load Canada dataset."""
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
def var_results(canada_data: pd.DataFrame) -> VARResults:
    """Fit VAR(2) on Canada data."""
    model = VAR(lags=2)
    return model.fit(canada_data)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG."""
    return np.random.default_rng(42)


class TestFEVD:
    """Tests for FEVD."""

    def test_fevd_shape(self, var_results: VARResults) -> None:
        """FEVD decomp should have shape (periods+1, K, K)."""
        fevd = FEVD(var_results, periods=20)
        assert fevd.decomp.shape == (21, 4, 4)

    def test_fevd_sums_to_one(self, var_results: VARResults) -> None:
        """FEVD should sum to 1 across shocks for each variable and horizon.

        sum_{k=0}^{K-1} FEVD_{ik}(h) = 1 for all i, h
        """
        fevd = FEVD(var_results, periods=20)

        for h in range(21):
            for i in range(4):
                total = np.sum(fevd.decomp[h, i, :])
                np.testing.assert_allclose(
                    total, 1.0, atol=1e-10,
                    err_msg=f"FEVD does not sum to 1 at h={h}, i={i}: sum={total}"
                )

    def test_fevd_own_shock_dominant(self, var_results: VARResults) -> None:
        """At h=0, the own shock should be the largest contributor.

        FEVD_{ii}(0) should be the maximum across k for each i.
        """
        fevd = FEVD(var_results, periods=10)

        for i in range(4):
            own_share = fevd.decomp[0, i, i]
            max_share = np.max(fevd.decomp[0, i, :])
            assert own_share == pytest.approx(max_share, abs=1e-10), (
                f"Variable {i}: own shock share {own_share:.4f} "
                f"< max share {max_share:.4f} at h=0"
            )

    def test_fevd_nonnegative(self, var_results: VARResults) -> None:
        """All FEVD values should be non-negative."""
        fevd = FEVD(var_results, periods=20)
        assert np.all(fevd.decomp >= -1e-12), "FEVD has negative values"

    def test_fevd_bounded(self, var_results: VARResults) -> None:
        """All FEVD values should be in [0, 1]."""
        fevd = FEVD(var_results, periods=20)
        assert np.all(fevd.decomp >= -1e-12)
        assert np.all(fevd.decomp <= 1.0 + 1e-12)

    def test_fevd_summary(self, var_results: VARResults) -> None:
        """summary() should produce formatted output."""
        fevd = FEVD(var_results, periods=20)
        summary = fevd.summary()

        assert isinstance(summary, str)
        assert "Forecast Error Variance Decomposition" in summary
        assert len(summary) > 200

    def test_fevd_to_dataframe(self, var_results: VARResults) -> None:
        """to_dataframe should return a long-form DataFrame."""
        fevd = FEVD(var_results, periods=5)
        df = fevd.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "horizon" in df.columns
        assert "response" in df.columns
        assert "shock" in df.columns
        assert "fevd" in df.columns
        # 6 horizons * 4 responses * 4 shocks = 96
        assert len(df) == 6 * 4 * 4

    def test_fevd_plot_no_error(self, var_results: VARResults) -> None:
        """plot() should run without error."""
        import matplotlib
        matplotlib.use("Agg")

        fevd = FEVD(var_results, periods=10)
        fig = fevd.plot()
        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_fevd_plot_single_variable(self, var_results: VARResults) -> None:
        """plot() with specific variable should work."""
        import matplotlib
        matplotlib.use("Agg")

        fevd = FEVD(var_results, periods=10)
        fig = fevd.plot(variable="e")
        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_fevd_long_horizon(self, var_results: VARResults) -> None:
        """FEVD at long horizons should still sum to 1."""
        fevd = FEVD(var_results, periods=100)

        for i in range(4):
            total = np.sum(fevd.decomp[100, i, :])
            np.testing.assert_allclose(total, 1.0, atol=1e-10)

    def test_fevd_convenience_method(self, var_results: VARResults) -> None:
        """VARResults.fevd() convenience method should work."""
        fevd = var_results.fevd(periods=10)
        assert isinstance(fevd, FEVD)
        assert fevd.decomp.shape == (11, 4, 4)
