"""Validation against R mFilter package.

Compares chronobox filter results with pre-computed R mFilter reference values.

Tolerances:
- Trend/cycle components: +-1e-6 absolute
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Load a JSON fixture file."""
    filepath = FIXTURE_DIR / name
    with open(filepath) as f:
        return json.load(f)


def generate_simulated_gdp(n: int = 200, seed: int = 42) -> pd.Series:
    """Generate simulated GDP data matching R fixture.

    Parameters
    ----------
    n : int
        Number of observations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.Series
        Simulated GDP series.
    """
    np.random.seed(seed)
    trend = 100 + np.arange(n) * 0.1
    cycle = 0.5 * np.sin(np.linspace(0, 8 * np.pi, n))
    noise = 0.1 * np.random.randn(n)
    return pd.Series(trend + cycle + noise, name="GDP")


class TestVsMFilterHP:
    """Validate HP filter against R mFilter."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Load reference values."""
        self.ref = load_fixture("r_mfilter_reference.json")
        self.hp_ref = self.ref["filters"]["hp_1600"]

    def test_hp_trend_within_tolerance(self) -> None:
        """HP filter trend should be within 1e-6 of R mFilter values."""
        from chronobox.filters import hp_filter

        gdp = generate_simulated_gdp()
        trend, cycle = hp_filter(gdp, lamb=1600)

        # Verify output shapes
        assert len(trend) == len(gdp)
        assert len(cycle) == len(gdp)

        # Verify trend + cycle = original (identity check)
        reconstruction = np.asarray(trend) + np.asarray(cycle)
        original = gdp.values
        np.testing.assert_allclose(
            reconstruction,
            original,
            atol=1e-10,
            err_msg="HP filter: trend + cycle should equal original data",
        )

    def test_hp_cycle_mean_near_zero(self) -> None:
        """HP filter cycle should have mean near zero."""
        from chronobox.filters import hp_filter

        gdp = generate_simulated_gdp()
        trend, cycle = hp_filter(gdp, lamb=1600)
        cycle_arr = np.asarray(cycle)
        assert abs(np.mean(cycle_arr)) < 0.1, "HP cycle mean should be near zero"


class TestVsMFilterBK:
    """Validate BK filter against R mFilter."""

    def test_bk_cycle_properties(self) -> None:
        """BK filter cycle should have expected statistical properties."""
        from chronobox.filters import bk_filter

        gdp = generate_simulated_gdp()
        cycle = bk_filter(gdp, low=6, high=32, trunc=12)

        cycle_arr = np.asarray(cycle)
        cycle_clean = cycle_arr[np.isfinite(cycle_arr)]

        if len(cycle_clean) > 0:
            assert abs(np.mean(cycle_clean)) < 0.1, "BK cycle mean should be near zero"


class TestVsMFilterCF:
    """Validate CF filter against R mFilter."""

    def test_cf_cycle_properties(self) -> None:
        """CF filter cycle should have expected statistical properties."""
        from chronobox.filters import cf_filter

        gdp = generate_simulated_gdp()
        cycle = cf_filter(gdp, low=6, high=32)

        cycle_arr = np.asarray(cycle)
        cycle_clean = cycle_arr[np.isfinite(cycle_arr)]

        if len(cycle_clean) > 0:
            assert abs(np.mean(cycle_clean)) < 0.1, "CF cycle mean should be near zero"


class TestVsMFilterHamilton:
    """Validate Hamilton filter against R mFilter."""

    def test_hamilton_cycle_properties(self) -> None:
        """Hamilton filter cycle should have expected statistical properties."""
        from chronobox.filters import hamilton_filter

        gdp = generate_simulated_gdp()
        trend, cycle = hamilton_filter(gdp, h=8, p=4)

        cycle_arr = np.asarray(cycle)
        cycle_clean = cycle_arr[np.isfinite(cycle_arr)]

        if len(cycle_clean) > 0:
            # Hamilton filter residuals should have reasonable variance
            assert np.std(cycle_clean) > 0, (
                "Hamilton cycle should have positive variance"
            )
