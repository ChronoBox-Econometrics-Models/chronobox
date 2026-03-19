"""Additional tsdata and dates tests for coverage improvement."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chronobox.core.dates import (
    create_forecast_index,
    detect_frequency,
    infer_periods_per_year,
    validate_index,
)
from chronobox.core.tsdata import TimeSeriesData


class TestDatesCoverage:
    def test_detect_frequency_non_datetime(self) -> None:
        result = detect_frequency(pd.RangeIndex(10))  # type: ignore[arg-type]
        assert result is None

    def test_infer_periods_per_year_monthly(self) -> None:
        assert infer_periods_per_year("MS") == 12
        assert infer_periods_per_year("ME") == 12

    def test_infer_periods_per_year_quarterly(self) -> None:
        assert infer_periods_per_year("QS") == 4
        assert infer_periods_per_year("QE") == 4

    def test_infer_periods_per_year_annual(self) -> None:
        assert infer_periods_per_year("YE") == 1

    def test_infer_periods_per_year_daily(self) -> None:
        assert infer_periods_per_year("D") == 365

    def test_infer_periods_per_year_none(self) -> None:
        assert infer_periods_per_year(None) == 1

    def test_infer_periods_per_year_unknown(self) -> None:
        assert infer_periods_per_year("UNKNOWN") == 1

    def test_infer_periods_per_year_prefix_match(self) -> None:
        assert infer_periods_per_year("MS-JAN") == 12

    def test_create_forecast_index(self) -> None:
        last = pd.Timestamp("2024-12-01")
        idx = create_forecast_index(last, 3, "ME")
        assert len(idx) == 3

    def test_create_forecast_index_none_freq(self) -> None:
        last = pd.Timestamp("2024-12-01")
        idx = create_forecast_index(last, 3, None)
        assert len(idx) == 3

    def test_validate_index_regular(self) -> None:
        idx = pd.date_range("2020-01-01", periods=50, freq="ME")
        assert validate_index(idx) is True

    def test_validate_index_non_datetime(self) -> None:
        assert validate_index(pd.RangeIndex(10)) is False  # type: ignore[arg-type]


class TestTSDataCoverage:
    def test_from_dataframe(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        ts = TimeSeriesData(df)
        assert ts.nobs == 3
        assert ts.name == "x"

    def test_from_dataframe_no_numeric_raises(self) -> None:
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        with pytest.raises(ValueError, match="no numeric"):
            TimeSeriesData(df)

    def test_from_list(self) -> None:
        ts = TimeSeriesData([1.0, 2.0, 3.0])
        assert ts.nobs == 3
        assert isinstance(ts.index, pd.RangeIndex)

    def test_from_series_with_name(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0], name="test_series")
        ts = TimeSeriesData(s)
        assert ts.name == "test_series"

    def test_from_series_no_name(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        ts = TimeSeriesData(s)
        assert ts.name == "y"

    def test_invalid_index_fallback(self) -> None:
        ts = TimeSeriesData(
            [1.0, 2.0, 3.0],
            index=["not", "a", "date"],  # type: ignore[arg-type]
        )
        assert isinstance(ts.index, (pd.RangeIndex, pd.DatetimeIndex))

    def test_log_positive(self) -> None:
        ts = TimeSeriesData([1.0, 2.0, 3.0])
        logged = ts.log()
        np.testing.assert_allclose(logged.values, np.log([1.0, 2.0, 3.0]))

    def test_log_non_positive_raises(self) -> None:
        ts = TimeSeriesData([-1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="positive"):
            ts.log()

    def test_boxcox_auto(self) -> None:
        ts = TimeSeriesData(np.arange(1.0, 20.0))
        transformed, lam = ts.boxcox()
        assert len(transformed) == 19
        assert isinstance(lam, float)

    def test_boxcox_non_positive_raises(self) -> None:
        ts = TimeSeriesData([-1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="positive"):
            ts.boxcox()

    def test_split_int(self) -> None:
        ts = TimeSeriesData(np.arange(100.0))
        train, test = ts.split(20)
        assert len(train) == 80
        assert len(test) == 20

    def test_split_float(self) -> None:
        ts = TimeSeriesData(np.arange(100.0))
        train, test = ts.split(0.2)
        assert len(train) == 80
        assert len(test) == 20

    def test_plot(self) -> None:
        ts = TimeSeriesData(np.random.randn(50))
        ax = ts.plot()
        assert ax is not None
        plt.close("all")

    def test_describe(self) -> None:
        ts = TimeSeriesData([1.0, 2.0, 3.0, 4.0, 5.0])
        desc = ts.describe()
        assert desc["nobs"] == 5
        assert desc["mean"] == 3.0

    def test_to_pandas(self) -> None:
        ts = TimeSeriesData([1.0, 2.0, 3.0], name="test")
        s = ts.to_pandas()
        assert isinstance(s, pd.Series)
        assert s.name == "test"

    def test_from_pandas_classmethod(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0], name="test")
        ts = TimeSeriesData.from_pandas(s)
        assert ts.nobs == 3

    def test_len(self) -> None:
        ts = TimeSeriesData([1.0, 2.0, 3.0])
        assert len(ts) == 3

    def test_frequency_detection(self) -> None:
        idx = pd.date_range("2020-01-01", periods=50, freq="ME")
        ts = TimeSeriesData(np.random.randn(50), index=idx)
        assert ts.frequency is not None

    def test_frequency_override(self) -> None:
        ts = TimeSeriesData([1.0, 2.0, 3.0], frequency="M")
        assert ts.frequency == "M"
