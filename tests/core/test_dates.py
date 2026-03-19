"""Tests for date utilities."""

from __future__ import annotations

import pandas as pd

from chronobox.core.dates import (
    create_forecast_index,
    detect_frequency,
    infer_periods_per_year,
    validate_index,
)


class TestDetectFrequency:
    """Test frequency detection."""

    def test_monthly(self) -> None:
        idx = pd.date_range("2020-01", periods=24, freq="MS")
        assert detect_frequency(idx) is not None

    def test_quarterly(self) -> None:
        idx = pd.date_range("2020-01", periods=20, freq="QS")
        freq = detect_frequency(idx)
        assert freq is not None


class TestPeriodsPerYear:
    """Test periods per year inference."""

    def test_monthly(self) -> None:
        assert infer_periods_per_year("MS") == 12
        assert infer_periods_per_year("M") == 12

    def test_quarterly(self) -> None:
        assert infer_periods_per_year("Q") == 4
        assert infer_periods_per_year("QS") == 4

    def test_annual(self) -> None:
        assert infer_periods_per_year("A") == 1
        assert infer_periods_per_year("Y") == 1

    def test_weekly(self) -> None:
        assert infer_periods_per_year("W") == 52


class TestForecastIndex:
    """Test forecast index creation."""

    def test_monthly_forecast(self) -> None:
        last = pd.Timestamp("2024-12-01")
        idx = create_forecast_index(last, 3, "MS")
        assert len(idx) == 3
        assert idx[0] == pd.Timestamp("2025-01-01")


class TestValidateIndex:
    """Test index validation."""

    def test_regular_monthly(self) -> None:
        idx = pd.date_range("2020-01", periods=24, freq="MS")
        assert validate_index(idx)
