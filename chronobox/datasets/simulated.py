"""Simulated time series datasets for testing and examples.

Generates synthetic time series with known DGP (Data Generating Process)
for testing estimation accuracy and demonstrating library features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_arma11(
    n: int = 500,
    ar: float = 0.7,
    ma: float = 0.4,
    sigma: float = 1.0,
    seed: int = 42,
) -> pd.Series:
    """Generate ARMA(1,1) process.

    y_t = ar * y_{t-1} + e_t + ma * e_{t-1}
    where e_t ~ N(0, sigma^2)

    Parameters
    ----------
    n : int
        Number of observations.
    ar : float
        AR(1) coefficient. Must satisfy |ar| < 1.
    ma : float
        MA(1) coefficient.
    sigma : float
        Innovation standard deviation.
    seed : int
        Random seed.

    Returns
    -------
    pd.Series
        Simulated ARMA(1,1) series.
    """
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, n + 100)
    y = np.zeros(n + 100)

    for t in range(1, n + 100):
        y[t] = ar * y[t - 1] + e[t] + ma * e[t - 1]

    # Discard burn-in
    series = y[100:]
    index = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.Series(series, index=index, name="arma11")


def generate_var2(
    n: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate bivariate VAR(2) process.

    [y1_t]   [0.5  0.1] [y1_{t-1}]   [-0.2  0.0] [y1_{t-2}]   [e1_t]
    [y2_t] = [0.2  0.6] [y2_{t-1}] + [0.0  -0.1] [y2_{t-2}] + [e2_t]

    Parameters
    ----------
    n : int
        Number of observations.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Bivariate VAR(2) series with columns ['y1', 'y2'].
    """
    rng = np.random.default_rng(seed)
    K = 2
    burn = 100
    total = n + burn

    A1 = np.array([[0.5, 0.1], [0.2, 0.6]])
    A2 = np.array([[-0.2, 0.0], [0.0, -0.1]])

    e = rng.multivariate_normal(np.zeros(K), np.eye(K) * 0.5, total)
    y = np.zeros((total, K))

    for t in range(2, total):
        y[t] = A1 @ y[t - 1] + A2 @ y[t - 2] + e[t]

    data = y[burn:]
    index = pd.date_range("2000-01-01", periods=n, freq="QS")
    return pd.DataFrame(data, index=index, columns=["y1", "y2"])


def generate_cointegrated(
    n: int = 500,
    beta: float = 0.8,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate cointegrated pair.

    x_t = x_{t-1} + e1_t  (random walk)
    y_t = beta * x_t + u_t  where u_t is stationary AR(1)

    Parameters
    ----------
    n : int
        Number of observations.
    beta : float
        Cointegrating coefficient.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Cointegrated pair with columns ['y', 'x'].
    """
    rng = np.random.default_rng(seed)

    # Common stochastic trend
    e1 = rng.normal(0, 1, n)
    x = np.cumsum(e1)

    # Stationary error
    e2 = rng.normal(0, 0.5, n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.3 * u[t - 1] + e2[t]

    y = beta * x + u

    index = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.DataFrame({"y": y, "x": x}, index=index)


def generate_structural_break(
    n: int = 500,
    break_point: float = 0.5,
    seed: int = 42,
) -> pd.Series:
    """Generate series with a structural break in mean.

    y_t = mu_1 + e_t  for t < break_point * n
    y_t = mu_2 + e_t  for t >= break_point * n

    Parameters
    ----------
    n : int
        Number of observations.
    break_point : float
        Relative position of break (0 to 1).
    seed : int
        Random seed.

    Returns
    -------
    pd.Series
        Series with structural break. Attribute `break_index` is set.
    """
    rng = np.random.default_rng(seed)
    bp = int(n * break_point)

    mu1, mu2 = 0.0, 3.0
    e = rng.normal(0, 1, n)

    y = np.zeros(n)
    y[:bp] = mu1 + e[:bp]
    y[bp:] = mu2 + e[bp:]

    index = pd.date_range("2000-01-01", periods=n, freq="MS")
    series = pd.Series(y, index=index, name="structural_break")
    series.attrs["break_index"] = bp
    series.attrs["break_date"] = index[bp]
    return series


def generate_long_memory(
    n: int = 500,
    d: float = 0.35,
    seed: int = 42,
) -> pd.Series:
    """Generate fractionally integrated (ARFIMA) process.

    Uses the Cholesky method to generate a process with long memory
    parameter d (0 < d < 0.5).

    Parameters
    ----------
    n : int
        Number of observations.
    d : float
        Fractional differencing parameter (0 < d < 0.5).
    seed : int
        Random seed.

    Returns
    -------
    pd.Series
        Long memory process.
    """
    rng = np.random.default_rng(seed)

    # Generate fractional differencing weights
    weights = np.zeros(n)
    weights[0] = 1.0
    for k in range(1, n):
        weights[k] = weights[k - 1] * (d + k - 1) / k

    # Convolve with white noise
    e = rng.normal(0, 1, 2 * n)
    y = np.convolve(weights, e[:n])[:n]

    index = pd.date_range("2000-01-01", periods=n, freq="MS")
    series = pd.Series(y, index=index, name="long_memory")
    series.attrs["d"] = d
    return series
