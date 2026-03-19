"""TimeSeriesData - Core container for time series data."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


class TimeSeriesData:
    """Container for time series data with frequency metadata.

    Parameters
    ----------
    data : ndarray, pd.Series, pd.DataFrame, or list
        Input data. If DataFrame, uses the first numeric column.
    index : DatetimeIndex, RangeIndex, or None
        Temporal index. Inferred from data if possible.
    frequency : str or None
        Frequency string ('M', 'Q', 'A', 'W', 'D'). Auto-detected if None.
    name : str or None
        Name of the series.

    Attributes
    ----------
    values : ndarray
        Numeric values of the series.
    index : DatetimeIndex or RangeIndex
        Temporal index.
    frequency : str or None
        Detected or provided frequency.
    name : str
        Series name.
    nobs : int
        Number of observations.
    missing_mask : ndarray[bool]
        Boolean mask of missing (NaN) values.
    """

    def __init__(
        self,
        data: NDArray[np.float64] | pd.Series | pd.DataFrame | list[float],
        index: pd.DatetimeIndex | pd.RangeIndex | None = None,
        frequency: str | None = None,
        name: str | None = None,
    ) -> None:
        if isinstance(data, pd.DataFrame):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                msg = "DataFrame has no numeric columns"
                raise ValueError(msg)
            series = data[numeric_cols[0]]
            self.values = series.to_numpy(dtype=np.float64)
            if index is None:
                index = series.index
            if name is None:
                name = str(numeric_cols[0])
        elif isinstance(data, pd.Series):
            self.values = data.to_numpy(dtype=np.float64)
            if index is None:
                index = data.index
            if name is None:
                name = str(data.name) if data.name is not None else "y"
        else:
            self.values = np.asarray(data, dtype=np.float64)

        self.nobs = len(self.values)
        self.missing_mask = np.isnan(self.values)
        self.name = name or "y"

        # Handle index
        if index is None:
            self.index: pd.DatetimeIndex | pd.RangeIndex = pd.RangeIndex(self.nobs)
        elif isinstance(index, (pd.DatetimeIndex, pd.RangeIndex)):
            self.index = index
        else:
            try:
                self.index = pd.DatetimeIndex(index)
            except (ValueError, TypeError):
                self.index = pd.RangeIndex(self.nobs)

        # Detect frequency
        if frequency is not None:
            self.frequency: str | None = frequency
        elif isinstance(self.index, pd.DatetimeIndex):
            inferred = pd.infer_freq(self.index)
            self.frequency = inferred
        else:
            self.frequency = None

    def diff(self, d: int = 1) -> TimeSeriesData:
        """Regular differencing.

        Parameters
        ----------
        d : int
            Order of differencing.

        Returns
        -------
        TimeSeriesData
            Differenced series with d fewer observations.
        """
        result = self.values.copy()
        for _ in range(d):
            result = np.diff(result)
        new_index = self.index[d:]
        return TimeSeriesData(
            data=result,
            index=new_index,
            frequency=self.frequency,
            name=f"diff({self.name}, {d})",
        )

    def seasonal_diff(self, s: int) -> TimeSeriesData:
        """Seasonal differencing: y_t - y_{t-s}.

        Parameters
        ----------
        s : int
            Seasonal period.

        Returns
        -------
        TimeSeriesData
            Seasonally differenced series with s fewer observations.
        """
        result = self.values[s:] - self.values[:-s]
        new_index = self.index[s:]
        return TimeSeriesData(
            data=result,
            index=new_index,
            frequency=self.frequency,
            name=f"sdiff({self.name}, {s})",
        )

    def log(self) -> TimeSeriesData:
        """Natural log transformation.

        Returns
        -------
        TimeSeriesData
            Log-transformed series.

        Raises
        ------
        ValueError
            If series contains non-positive values.
        """
        if np.any(self.values[~self.missing_mask] <= 0):
            msg = "log transform requires all positive values"
            raise ValueError(msg)
        return TimeSeriesData(
            data=np.log(self.values),
            index=self.index,
            frequency=self.frequency,
            name=f"log({self.name})",
        )

    def boxcox(self, lam: float | None = None) -> tuple[TimeSeriesData, float]:
        """Box-Cox transformation.

        Parameters
        ----------
        lam : float or None
            Box-Cox parameter. If None, optimal lambda is estimated.

        Returns
        -------
        tuple of (TimeSeriesData, float)
            Transformed series and the lambda used.
        """
        valid = self.values[~self.missing_mask]
        if np.any(valid <= 0):
            msg = "Box-Cox transform requires all positive values"
            raise ValueError(msg)
        if lam is None:
            _, lam = stats.boxcox(valid)
        transformed = np.log(self.values) if lam == 0 else (self.values**lam - 1.0) / lam
        ts = TimeSeriesData(
            data=transformed,
            index=self.index,
            frequency=self.frequency,
            name=f"boxcox({self.name}, {lam:.4f})",
        )
        return ts, float(lam)

    def split(
        self, test_size: int | float
    ) -> tuple[TimeSeriesData, TimeSeriesData]:
        """Split into train and test sets preserving temporal order.

        Parameters
        ----------
        test_size : int or float
            If int, number of test observations. If float, fraction of total.

        Returns
        -------
        tuple of (TimeSeriesData, TimeSeriesData)
            Train and test sets.
        """
        n_test = int(self.nobs * test_size) if isinstance(test_size, float) else test_size
        n_train = self.nobs - n_test
        train = TimeSeriesData(
            data=self.values[:n_train],
            index=self.index[:n_train],
            frequency=self.frequency,
            name=f"{self.name}_train",
        )
        test = TimeSeriesData(
            data=self.values[n_train:],
            index=self.index[n_train:],
            frequency=self.frequency,
            name=f"{self.name}_test",
        )
        return train, test

    def plot(self, **kwargs: object) -> object:
        """Plot the time series.

        Returns
        -------
        matplotlib Axes
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 4)))  # type: ignore[arg-type]
        ax.plot(self.index, self.values, **kwargs)
        ax.set_title(self.name)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        return ax

    def describe(self) -> pd.Series:
        """Descriptive statistics.

        Returns
        -------
        pd.Series
            Statistics including mean, std, min, max, etc.
        """
        valid = self.values[~self.missing_mask]
        return pd.Series(
            {
                "nobs": self.nobs,
                "missing": int(self.missing_mask.sum()),
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid, ddof=1)),
                "min": float(np.min(valid)),
                "25%": float(np.percentile(valid, 25)),
                "50%": float(np.percentile(valid, 50)),
                "75%": float(np.percentile(valid, 75)),
                "max": float(np.max(valid)),
            },
            name=self.name,
        )

    def to_pandas(self) -> pd.Series:
        """Convert to pandas Series.

        Returns
        -------
        pd.Series
        """
        return pd.Series(self.values, index=self.index, name=self.name)

    @classmethod
    def from_pandas(cls, series: pd.Series, frequency: str | None = None) -> Self:
        """Create from pandas Series.

        Parameters
        ----------
        series : pd.Series
            Input pandas Series.
        frequency : str or None
            Override frequency detection.

        Returns
        -------
        TimeSeriesData
        """
        return cls(
            data=series,
            frequency=frequency,
        )

    def __len__(self) -> int:
        return self.nobs

    def __repr__(self) -> str:
        return (
            f"TimeSeriesData(name='{self.name}', nobs={self.nobs}, "
            f"frequency='{self.frequency}')"
        )
