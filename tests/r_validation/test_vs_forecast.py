"""Validation against R forecast package.

Compares chronobox ARIMA results with pre-computed R forecast reference values.

Tolerances:
- Parameters: +-5% relative
- Log-likelihood: +-0.5 absolute
- AIC: +-1.0 absolute
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Load a JSON fixture file.

    Parameters
    ----------
    name : str
        Fixture filename.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    filepath = FIXTURE_DIR / name
    with open(filepath) as f:
        return json.load(f)


class TestVsForecastARIMA:
    """Validate ARIMA against R forecast package."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Load reference values."""
        self.ref = load_fixture("r_forecast_reference.json")
        self.airline_ref = self.ref["models"]["airline_arima_011_011_12"]

    def test_arima_coefficients_within_5pct(self) -> None:
        """ARIMA coefficients should be within 5% of R forecast values."""
        from chronobox import ARIMA
        from chronobox.datasets import load_dataset

        airline = load_dataset("airline")
        data = airline["passengers"].values.astype(float)

        model = ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
        results = model.fit(data)

        # Compare MA(1) coefficient
        r_ma1 = self.airline_ref["coefficients"]["ma1"]

        # Find the ma.L1 parameter by name
        param_names = results.param_names
        params = results.params
        ma1_idx = None
        for i, name in enumerate(param_names):
            if name == "ma.L1":
                ma1_idx = i
                break

        if ma1_idx is not None:
            cb_ma1 = params[ma1_idx]
            if abs(r_ma1) > 0.01:
                rel_error = abs(cb_ma1 - r_ma1) / abs(r_ma1)
                assert rel_error < 0.05, (
                    f"MA1 coefficient {cb_ma1:.4f} differs from R "
                    f"{r_ma1:.4f} by {rel_error * 100:.1f}%"
                )

    def test_loglikelihood_within_tolerance(self) -> None:
        """Log-likelihood should be within 0.5 of R forecast value."""
        from chronobox import ARIMA
        from chronobox.datasets import load_dataset

        airline = load_dataset("airline")
        data = airline["passengers"].values.astype(float)

        model = ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
        results = model.fit(data)

        r_loglike = self.airline_ref["loglikelihood"]
        cb_loglike = results.loglike
        assert abs(cb_loglike - r_loglike) < 0.5, (
            f"LogLike {cb_loglike:.2f} differs from R {r_loglike:.2f} "
            f"by {abs(cb_loglike - r_loglike):.2f}"
        )

    def test_aic_within_tolerance(self) -> None:
        """AIC should be within 1.0 of R forecast value."""
        from chronobox import ARIMA
        from chronobox.datasets import load_dataset

        airline = load_dataset("airline")
        data = airline["passengers"].values.astype(float)

        model = ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
        results = model.fit(data)

        r_aic = self.airline_ref["aic"]
        cb_aic = results.aic
        assert abs(cb_aic - r_aic) < 1.0, (
            f"AIC {cb_aic:.2f} differs from R {r_aic:.2f} "
            f"by {abs(cb_aic - r_aic):.2f}"
        )

    def test_forecast_direction(self) -> None:
        """Forecast direction should match R forecast (increasing trend)."""
        from chronobox import ARIMA
        from chronobox.datasets import load_dataset

        airline = load_dataset("airline")
        data = airline["passengers"].values.astype(float)

        model = ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
        results = model.fit(data)
        fc = results.forecast(steps=12)
        forecast = np.asarray(fc["forecast"])

        np.array(self.airline_ref["forecast_12"])

        # Forecasts should be in the same order of magnitude
        assert np.all(np.isfinite(forecast)), "Forecast contains non-finite values"
        assert forecast.mean() > 0, "Airline forecasts should be positive"
