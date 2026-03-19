"""Tests for Granger causality."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronobox.analysis.granger import granger_causality
from chronobox.models.var import VAR, GrangerResult, VARResults


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG."""
    return np.random.default_rng(42)


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
def var_results_canada(canada_data: pd.DataFrame) -> VARResults:
    """Fit VAR(2) on Canada data."""
    model = VAR(lags=2)
    return model.fit(canada_data)


@pytest.fixture
def unidirectional_dgp(rng: np.random.Generator) -> np.ndarray:
    """Simulate a unidirectional Granger causality DGP.

    Y1 -> Y2 (Y1 Granger-causes Y2, but not vice versa):
        Y1_t = 0.5 * Y1_{t-1} + e1_t
        Y2_t = 0.3 * Y2_{t-1} + 0.4 * Y1_{t-1} + e2_t
    """
    t = 500
    y1 = np.zeros(t)
    y2 = np.zeros(t)

    for i in range(1, t):
        y1[i] = 0.5 * y1[i - 1] + rng.standard_normal() * 0.5
        y2[i] = 0.3 * y2[i - 1] + 0.4 * y1[i - 1] + rng.standard_normal() * 0.5

    return np.column_stack([y1, y2])


@pytest.fixture
def independent_dgp(rng: np.random.Generator) -> np.ndarray:
    """Simulate two independent AR(1) processes (no Granger causality).

        Y1_t = 0.5 * Y1_{t-1} + e1_t
        Y2_t = 0.4 * Y2_{t-1} + e2_t
    """
    t = 500
    y1 = np.zeros(t)
    y2 = np.zeros(t)

    for i in range(1, t):
        y1[i] = 0.5 * y1[i - 1] + rng.standard_normal() * 0.5
        y2[i] = 0.4 * y2[i - 1] + rng.standard_normal() * 0.5

    return np.column_stack([y1, y2])


@pytest.fixture
def bilateral_dgp(rng: np.random.Generator) -> np.ndarray:
    """Simulate bilateral (feedback) Granger causality.

        Y1_t = 0.3 * Y1_{t-1} + 0.3 * Y2_{t-1} + e1_t
        Y2_t = 0.3 * Y2_{t-1} + 0.3 * Y1_{t-1} + e2_t
    """
    t = 500
    y1 = np.zeros(t)
    y2 = np.zeros(t)

    for i in range(1, t):
        y1[i] = 0.3 * y1[i - 1] + 0.3 * y2[i - 1] + rng.standard_normal() * 0.5
        y2[i] = 0.3 * y2[i - 1] + 0.3 * y1[i - 1] + rng.standard_normal() * 0.5

    return np.column_stack([y1, y2])


class TestGrangerKnownDGP:
    """Tests for Granger causality with known DGPs."""

    def test_granger_detects_direction(self, unidirectional_dgp: np.ndarray) -> None:
        """Granger test should detect Y1 -> Y2 (reject H0) in unidirectional DGP."""
        model = VAR(lags=1)
        results = model.fit(unidirectional_dgp, names=["Y1", "Y2"])

        # Y1 -> Y2: should reject
        gc = granger_causality(results, caused="Y2", causing="Y1")
        assert gc.reject, (
            f"Failed to detect Y1 -> Y2: F={gc.fstat:.4f}, p={gc.pvalue:.4f}"
        )

    def test_granger_no_reverse(self, unidirectional_dgp: np.ndarray) -> None:
        """Granger test should NOT detect Y2 -> Y1 in unidirectional DGP."""
        model = VAR(lags=1)
        results = model.fit(unidirectional_dgp, names=["Y1", "Y2"])

        # Y2 -> Y1: should NOT reject
        gc = granger_causality(results, caused="Y1", causing="Y2")
        assert not gc.reject, (
            f"Incorrectly detected Y2 -> Y1: F={gc.fstat:.4f}, p={gc.pvalue:.4f}"
        )

    def test_granger_no_causality(self, independent_dgp: np.ndarray) -> None:
        """Independent processes: Granger test should not reject in either direction."""
        model = VAR(lags=1)
        results = model.fit(independent_dgp, names=["Y1", "Y2"])

        gc_12 = granger_causality(results, caused="Y2", causing="Y1")
        gc_21 = granger_causality(results, caused="Y1", causing="Y2")

        assert gc_12.pvalue > 0.05, (
            f"False positive Y1 -> Y2: p={gc_12.pvalue:.4f}"
        )
        assert gc_21.pvalue > 0.05, (
            f"False positive Y2 -> Y1: p={gc_21.pvalue:.4f}"
        )

    def test_granger_bilateral(self, bilateral_dgp: np.ndarray) -> None:
        """Bilateral DGP: both directions should be detected."""
        model = VAR(lags=1)
        results = model.fit(bilateral_dgp, names=["Y1", "Y2"])

        gc_12 = granger_causality(results, caused="Y2", causing="Y1")
        gc_21 = granger_causality(results, caused="Y1", causing="Y2")

        assert gc_12.reject, (
            f"Failed to detect Y1 -> Y2 in bilateral DGP: p={gc_12.pvalue:.4f}"
        )
        assert gc_21.reject, (
            f"Failed to detect Y2 -> Y1 in bilateral DGP: p={gc_21.pvalue:.4f}"
        )


class TestGrangerOutput:
    """Tests for Granger causality output format."""

    def test_granger_result_type(self, var_results_canada: VARResults) -> None:
        """granger_causality should return GrangerResult."""
        gc = granger_causality(var_results_canada, caused="prod", causing="e")
        assert isinstance(gc, GrangerResult)

    def test_granger_result_attributes(self, var_results_canada: VARResults) -> None:
        """GrangerResult should have all required attributes."""
        gc = granger_causality(var_results_canada, caused="prod", causing="e")

        assert isinstance(gc.fstat, float)
        assert isinstance(gc.pvalue, float)
        assert isinstance(gc.df, tuple)
        assert len(gc.df) == 2
        assert isinstance(gc.reject, bool)
        assert isinstance(gc.wald_stat, float)
        assert isinstance(gc.wald_pvalue, float)
        assert gc.caused == "prod"
        assert gc.causing == "e"

    def test_granger_fstat_nonneg(self, var_results_canada: VARResults) -> None:
        """F-statistic should be non-negative."""
        gc = granger_causality(var_results_canada, caused="prod", causing="e")
        assert gc.fstat >= 0

    def test_granger_pvalue_range(self, var_results_canada: VARResults) -> None:
        """P-value should be in [0, 1]."""
        gc = granger_causality(var_results_canada, caused="prod", causing="e")
        assert 0 <= gc.pvalue <= 1
        assert 0 <= gc.wald_pvalue <= 1

    def test_granger_repr(self, var_results_canada: VARResults) -> None:
        """GrangerResult repr should be informative."""
        gc = granger_causality(var_results_canada, caused="prod", causing="e")
        rep = repr(gc)
        assert "GrangerResult" in rep
        assert "e -> prod" in rep

    def test_granger_convenience_method(self, var_results_canada: VARResults) -> None:
        """VARResults.granger_causality() convenience method should work."""
        gc = var_results_canada.granger_causality(caused="prod", causing="e")
        assert isinstance(gc, GrangerResult)

    def test_granger_by_index(self, var_results_canada: VARResults) -> None:
        """Granger test should accept integer indices."""
        gc = granger_causality(var_results_canada, caused=1, causing=0)
        assert gc.caused == "prod"
        assert gc.causing == "e"

    def test_granger_invalid_variable(self, var_results_canada: VARResults) -> None:
        """Invalid variable name should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            granger_causality(var_results_canada, caused="nonexistent", causing="e")

    def test_granger_wald_consistent_with_f(
        self, var_results_canada: VARResults
    ) -> None:
        """Wald and F tests should give consistent conclusions."""
        gc = granger_causality(var_results_canada, caused="prod", causing="e")
        # Both should reject or both should not reject (at similar significance)
        f_reject = gc.pvalue < 0.05
        wald_reject = gc.wald_pvalue < 0.05
        # Allow for minor disagreements due to asymptotic approximation
        # but they should generally agree
        assert f_reject == wald_reject or abs(gc.pvalue - 0.05) < 0.02
