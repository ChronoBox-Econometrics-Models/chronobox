"""Tests for TimeSeriesData."""

from __future__ import annotations

import numpy as np
import pandas as pd

from chronobox.core.tsdata import TimeSeriesData


class TestFromPandas:
    """Test creation from pandas objects."""

    def test_from_series(self) -> None:
        idx = pd.date_range("1949-01", periods=12, freq="MS")
        s = pd.Series(range(12), index=idx, name="test", dtype=float)
        ts = TimeSeriesData.from_pandas(s)
        assert ts.nobs == 12
        assert ts.name == "test"
        assert ts.frequency is not None

    def test_from_dataframe(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        ts = TimeSeriesData(df)
        assert ts.nobs == 3
        assert ts.name == "a"  # first numeric column

    def test_from_array(self) -> None:
        ts = TimeSeriesData(np.array([1.0, 2.0, 3.0]))
        assert ts.nobs == 3
        assert isinstance(ts.index, pd.RangeIndex)


class TestDiff:
    """Test differencing."""

    def test_diff_1(self) -> None:
        ts = TimeSeriesData([1.0, 3.0, 6.0, 10.0])
        d = ts.diff(1)
        assert d.nobs == 3
        np.testing.assert_array_equal(d.values, [2.0, 3.0, 4.0])

    def test_diff_2(self) -> None:
        ts = TimeSeriesData([1.0, 3.0, 6.0, 10.0, 15.0])
        d = ts.diff(2)
        assert d.nobs == 3
        np.testing.assert_array_almost_equal(d.values, [1.0, 1.0, 1.0])


class TestSeasonalDiff:
    """Test seasonal differencing."""

    def test_seasonal_diff_12(self, airline_passengers: np.ndarray) -> None:
        ts = TimeSeriesData(airline_passengers)
        sd = ts.seasonal_diff(12)
        assert sd.nobs == ts.nobs - 12


class TestBoxCox:
    """Test Box-Cox transformation."""

    def test_boxcox_lambda_zero(self) -> None:
        ts = TimeSeriesData([1.0, 2.0, 3.0, 4.0, 5.0])
        bc, lam = ts.boxcox(lam=0)
        np.testing.assert_array_almost_equal(bc.values, np.log(ts.values))
        assert lam == 0

    def test_boxcox_auto(self) -> None:
        ts = TimeSeriesData(np.exp(np.arange(1.0, 6.0)))
        bc, lam = ts.boxcox()
        assert isinstance(lam, float)
        assert bc.nobs == ts.nobs


class TestSplit:
    """Test train/test split."""

    def test_split_int(self) -> None:
        ts = TimeSeriesData(np.arange(100.0))
        train, test = ts.split(20)
        assert train.nobs == 80
        assert test.nobs == 20
        np.testing.assert_array_equal(
            np.concatenate([train.values, test.values]), ts.values
        )

    def test_split_float(self) -> None:
        ts = TimeSeriesData(np.arange(100.0))
        train, test = ts.split(0.2)
        assert train.nobs == 80
        assert test.nobs == 20
