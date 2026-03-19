"""Validation against R urca package.

Compares chronobox unit root and cointegration test results with R urca values.

Tolerances:
- Test statistics: +-0.1 absolute
- Same rejection conclusion at 5% level
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


def _load_airline_passengers() -> np.ndarray:
    """Load airline passengers as a float numpy array."""
    from chronobox.datasets import load_dataset

    airline = load_dataset("airline")
    return airline["passengers"].values.astype(float)


def _load_canada_numeric() -> pd.DataFrame:
    """Load canada dataset with only numeric columns."""
    from chronobox.datasets import load_dataset

    canada = load_dataset("canada")
    return canada[["e", "prod", "rw", "U"]]


class TestVsUrcaUnitRoot:
    """Validate unit root tests against R urca package."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Load reference values."""
        self.ref = load_fixture("r_urca_reference.json")

    def test_adf_statistic_within_tolerance(self) -> None:
        """ADF test statistic should be within 0.1 of R urca value."""
        from chronobox.tests_stat import adf_test

        data = _load_airline_passengers()
        result = adf_test(data)
        r_ref = self.ref["tests"]["airline_adf"]
        r_stat = r_ref["statistic"]

        cb_stat = result.statistic
        assert abs(cb_stat - r_stat) < 0.1, (
            f"ADF statistic {cb_stat:.3f} differs from R {r_stat:.3f} "
            f"by {abs(cb_stat - r_stat):.3f}"
        )

    def test_adf_same_rejection_conclusion(self) -> None:
        """ADF test should give same rejection conclusion as R urca."""
        from chronobox.tests_stat import adf_test

        data = _load_airline_passengers()
        result = adf_test(data)
        r_ref = self.ref["tests"]["airline_adf"]
        r_reject = r_ref["reject_5pct"]

        cb_reject = result.reject_at_5pct

        assert cb_reject == r_reject, (
            f"ADF rejection mismatch: chronobox={cb_reject}, R={r_reject}"
        )

    def test_pp_statistic_within_tolerance(self) -> None:
        """Phillips-Perron test statistic should be within 0.1 of R urca value."""
        from chronobox.tests_stat import pp_test

        data = _load_airline_passengers()
        result = pp_test(data)
        r_ref = self.ref["tests"]["airline_pp"]
        r_stat = r_ref["statistic"]

        cb_stat = result.statistic
        assert abs(cb_stat - r_stat) < 0.1, (
            f"PP statistic {cb_stat:.3f} differs from R {r_stat:.3f}"
        )

    def test_kpss_same_rejection_conclusion(self) -> None:
        """KPSS test should give same rejection conclusion as R urca."""
        from chronobox.tests_stat import kpss_test

        data = _load_airline_passengers()
        result = kpss_test(data)
        r_ref = self.ref["tests"]["airline_kpss"]
        r_reject = r_ref["reject_5pct"]

        cb_reject = result.reject_at_5pct

        assert cb_reject == r_reject, (
            f"KPSS rejection mismatch: chronobox={cb_reject}, R={r_reject}"
        )


class TestVsUrcaJohansen:
    """Validate Johansen cointegration test against R urca."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Load reference values."""
        self.ref = load_fixture("r_urca_reference.json")

    def test_johansen_rank_consistent(self) -> None:
        """Johansen rank should be consistent with R urca."""
        from chronobox.models.vecm import VECM

        canada = _load_canada_numeric()
        r_ref = self.ref["tests"]["canada_johansen"]
        r_rank = r_ref["rank"]

        vecm = VECM(lags=2)
        joh_result = vecm.johansen_test(canada)

        cb_rank = joh_result.rank_trace

        assert cb_rank == r_rank, (
            f"Johansen rank mismatch: chronobox={cb_rank}, R={r_rank}"
        )
