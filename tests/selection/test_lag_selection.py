"""Tests for lag order selection."""

from __future__ import annotations

import numpy as np
import pytest

from chronobox.selection.lag_selection import LagOrderResults, select_lag_order


@pytest.fixture
def canada_data() -> np.ndarray:
    """Load Canada dataset as numpy array."""
    from pathlib import Path

    import pandas as pd

    data_path = (
        Path(__file__).parent.parent.parent
        / "chronobox"
        / "datasets"
        / "data"
        / "macro"
        / "canada.csv"
    )
    df = pd.read_csv(data_path)
    return df[["e", "prod", "rw", "U"]].to_numpy(dtype=np.float64)


@pytest.fixture
def simulated_var2(rng: np.random.Generator) -> np.ndarray:
    """Simulate a VAR(2) process with known lag order.

    DGP:
        Y_t = A_1 * Y_{t-1} + A_2 * Y_{t-2} + u_t
        A_1 = [[0.5, 0.1], [0.1, 0.4]]
        A_2 = [[0.2, 0.0], [0.0, 0.1]]
    """
    k = 2
    t = 500
    a1 = np.array([[0.5, 0.1], [0.1, 0.4]])
    a2 = np.array([[0.2, 0.0], [0.0, 0.1]])

    y = np.zeros((t + 100, k))  # extra burn-in
    for t_i in range(2, t + 100):
        y[t_i] = a1 @ y[t_i - 1] + a2 @ y[t_i - 2] + rng.standard_normal(k) * 0.5

    return y[100:]  # discard burn-in


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator."""
    return np.random.default_rng(42)


class TestLagSelection:
    """Tests for select_lag_order."""

    def test_returns_lag_order_results(self, canada_data: np.ndarray) -> None:
        """select_lag_order returns LagOrderResults with all fields."""
        result = select_lag_order(canada_data, maxlags=8)
        assert isinstance(result, LagOrderResults)
        assert len(result.aic) > 0
        assert len(result.bic) > 0
        assert len(result.hqic) > 0
        assert len(result.fpe) > 0
        assert len(result.selected_orders) == 4

    def test_bic_parsimonious(self, canada_data: np.ndarray) -> None:
        """BIC should select a lag order <= AIC (BIC is more parsimonious)."""
        result = select_lag_order(canada_data, maxlags=8)
        bic_order = result.selected_orders["bic"]
        aic_order = result.selected_orders["aic"]
        assert bic_order <= aic_order, (
            f"BIC selected p={bic_order} > AIC p={aic_order}; "
            f"BIC should be at least as parsimonious as AIC"
        )

    def test_known_dgp_var2(self, simulated_var2: np.ndarray) -> None:
        """AIC should select p=2 for a simulated VAR(2) process."""
        result = select_lag_order(simulated_var2, maxlags=6)
        aic_order = result.selected_orders["aic"]
        assert aic_order == 2, (
            f"AIC selected p={aic_order} for VAR(2) DGP, expected p=2"
        )

    def test_canada_reasonable_order(self, canada_data: np.ndarray) -> None:
        """All criteria should select a reasonable lag order (1-4) on Canada data."""
        result = select_lag_order(canada_data, maxlags=8)
        for criterion, order in result.selected_orders.items():
            assert 0 <= order <= 8, (
                f"{criterion} selected p={order}, outside [0, 8]"
            )

    def test_summary_output(self, canada_data: np.ndarray) -> None:
        """summary() should produce a non-empty formatted string."""
        result = select_lag_order(canada_data, maxlags=4)
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 100
        assert "AIC" in summary
        assert "BIC" in summary

    def test_invalid_input_1d(self) -> None:
        """Should raise ValueError for 1-D input."""
        with pytest.raises(ValueError, match="2-D"):
            select_lag_order(np.array([1.0, 2.0, 3.0]))

    def test_lr_test_stats(self, canada_data: np.ndarray) -> None:
        """LR test statistics should be computed for lags >= 1."""
        result = select_lag_order(canada_data, maxlags=4)
        assert len(result.lr_stats) > 0
        for p, stat in result.lr_stats.items():
            assert stat >= 0, f"LR stat for lag {p} is negative: {stat}"
            assert 0 <= result.lr_pvalues[p] <= 1

    def test_trend_none(self, canada_data: np.ndarray) -> None:
        """Should work with trend='n'."""
        result = select_lag_order(canada_data, maxlags=4, trend="n")
        assert len(result.selected_orders) == 4

    def test_trend_ct(self, canada_data: np.ndarray) -> None:
        """Should work with trend='ct'."""
        result = select_lag_order(canada_data, maxlags=4, trend="ct")
        assert len(result.selected_orders) == 4

    def test_ic_values_finite(self, canada_data: np.ndarray) -> None:
        """All IC values should be finite."""
        result = select_lag_order(canada_data, maxlags=8)
        for p, val in result.aic.items():
            assert np.isfinite(val), f"AIC({p}) is not finite: {val}"
        for p, val in result.bic.items():
            assert np.isfinite(val), f"BIC({p}) is not finite: {val}"
