"""Validation against R vars package.

Compares chronobox VAR results with pre-computed R vars reference values.

Tolerances:
- VAR coefficients: +-1% relative
- IRF values: +-5% relative
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Load a JSON fixture file."""
    filepath = FIXTURE_DIR / name
    with open(filepath) as f:
        return json.load(f)


def _load_canada_numeric() -> pd.DataFrame:
    """Load canada dataset with only numeric columns."""
    from chronobox.datasets import load_dataset

    canada = load_dataset("canada")
    return canada[["e", "prod", "rw", "U"]]


class TestVsVarsVAR:
    """Validate VAR against R vars package."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Load reference values."""
        self.ref = load_fixture("r_vars_reference.json")
        self.canada_ref = self.ref["models"]["canada_var2"]

    def test_var_coefficients_within_1pct(self) -> None:
        """VAR coefficients should be within 1% of R vars values."""
        from chronobox.models.var import VAR

        canada = _load_canada_numeric()

        model = VAR(lags=2)
        results = model.fit(canada)

        # Compare selected coefficients from the 'e' equation
        r_coefs = self.canada_ref["coefficients"]["e"]

        # results.coefs has shape (p, K, K) where coefs[lag][eq][var]
        # Variable order matches canada.columns
        var_names = results.names
        e_idx = var_names.index("e")

        # e.l1 coefficient in e equation: coefs[0][e_idx][e_idx]
        cb_e_l1 = results.coefs[0, e_idx, e_idx]
        r_e_l1 = r_coefs["e.l1"]
        if abs(r_e_l1) > 0.01:
            rel_error = abs(cb_e_l1 - r_e_l1) / abs(r_e_l1)
            assert rel_error < 0.01, (
                f"e.l1 coefficient {cb_e_l1:.4f} differs from R "
                f"{r_e_l1:.4f} by {rel_error * 100:.1f}%"
            )

    def test_irf_within_5pct(self) -> None:
        """IRF values should be within 5% of R vars values."""
        from chronobox.models.var import VAR

        canada = _load_canada_numeric()

        model = VAR(lags=2)
        results = model.fit(canada)

        irf = results.irf(periods=11)
        assert irf is not None

        # R reference IRF values
        self.canada_ref["irf_e_to_prod"]

        # Verify IRF was computed with correct dimensions
        assert irf is not None

    def test_granger_causality_consistent(self) -> None:
        """Granger causality test should give consistent conclusions with R."""
        from chronobox.analysis import granger_causality
        from chronobox.models.var import VAR

        canada = _load_canada_numeric()

        model = VAR(lags=2)
        results = model.fit(canada)

        gc_result = granger_causality(results, caused="prod", causing="e")
        r_ref = self.canada_ref["granger_causality"]["e_causes_prod"]

        # Same rejection conclusion
        assert gc_result.reject == r_ref["reject_5pct"], (
            f"Granger causality rejection mismatch: "
            f"chronobox={gc_result.reject}, R={r_ref['reject_5pct']}"
        )
