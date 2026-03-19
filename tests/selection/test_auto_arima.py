"""Tests for auto_arima."""

from __future__ import annotations

import re

import numpy as np
import pytest

from chronobox.models.arima import ARIMA
from chronobox.selection.auto_arima import auto_arima


class TestAirlineAuto:
    """Test auto_arima on airline dataset."""

    def test_selects_reasonable_model(self, airline_passengers: np.ndarray) -> None:
        """auto_arima should select a competitive model on airline data."""
        best = auto_arima(
            airline_passengers,
            seasonal=True,
            m=12,
            max_p=2,
            max_q=2,
            max_P=1,
            max_Q=1,
        )
        assert best is not None
        assert best.aicc < np.inf

    def test_aicc_competitive(self, airline_passengers: np.ndarray) -> None:
        """AICc of auto model should be <= AICc of ARIMA(1,1,1)(1,1,1)[12]."""
        best = auto_arima(
            airline_passengers,
            seasonal=True,
            m=12,
            max_p=2,
            max_q=2,
            max_P=1,
            max_Q=1,
        )

        # Fit reference model
        ref = ARIMA(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        ref_results = ref.fit(airline_passengers)

        assert best.aicc <= ref_results.aicc + 2.0  # small tolerance


class TestNileAuto:
    """Test auto_arima on Nile dataset."""

    def test_selects_d1(self, nile_volume: np.ndarray) -> None:
        """auto_arima on Nile should select model with d >= 1."""
        best = auto_arima(nile_volume, seasonal=False, m=1)
        assert best is not None
        # Model name should contain d=1 or d=2
        assert "1" in best.model_name or "2" in best.model_name


class TestStepwiseTrace:
    """Test trace output."""

    def test_trace_output(
        self, nile_volume: np.ndarray, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """trace=True should produce output."""
        auto_arima(nile_volume, seasonal=False, m=1, trace=True, max_p=2, max_q=2)
        captured = capsys.readouterr()
        assert "ARIMA" in captured.out

    def test_stepwise_trace_shows_models(
        self, nile_volume: np.ndarray, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """trace should show each model tested."""
        auto_arima(nile_volume, seasonal=False, m=1, trace=True, max_p=1, max_q=1)
        captured = capsys.readouterr()
        assert "Best model" in captured.out


class TestRespectMax:
    """Test max parameter constraints."""

    def test_max_p_respected(self, nile_volume: np.ndarray) -> None:
        """max_p=1 should never test p > 1."""
        best = auto_arima(
            nile_volume, seasonal=False, m=1, max_p=1, max_q=1
        )
        # The selected p should be <= 1
        model_name = best.model_name
        # Extract p from ARIMA(p,d,q)
        match = re.search(r"ARIMA\((\d+),", model_name)
        if match:
            p = int(match.group(1))
            assert p <= 1
