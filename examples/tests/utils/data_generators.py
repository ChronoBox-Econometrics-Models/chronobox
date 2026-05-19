"""
Data generators for statistical tests examples.

Generates synthetic time series with known properties for testing:
- Unit root processes: I(0), I(1), I(2)
- Cointegrated pairs
- Series with structural breaks
- Trend-stationary vs difference-stationary processes
"""

import numpy as np
import pandas as pd


def generate_unit_root_process(
    n: int = 200,
    phi: float = 1.0,
    seed: int = 42,
    order: int = 1,
    drift: float = 0.0,
    sigma: float = 1.0,
) -> pd.Series:
    """
    Generate an AR(1) process that is stationary or has a unit root.

    Parameters
    ----------
    n : int
        Number of observations.
    phi : float
        AR(1) coefficient. phi=1.0 gives a unit root (I(1)),
        |phi|<1 gives a stationary I(0) process.
    seed : int
        Random seed for reproducibility.
    order : int
        Integration order. 1 gives the raw AR(1) process.
        2 cumulates once more to produce an I(2) process (only when phi=1.0).
    drift : float
        Constant drift term added at each step.
    sigma : float
        Standard deviation of the innovation term.

    Returns
    -------
    pd.Series
        Generated time series.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, n)

    y = np.zeros(n)
    y[0] = eps[0]
    for t in range(1, n):
        y[t] = drift + phi * y[t - 1] + eps[t]

    # For I(2): cumulate the I(1) process once more
    if order == 2 and phi == 1.0:
        y = np.cumsum(y)

    return pd.Series(y, name="y")


def generate_cointegrated_pair(
    n: int = 200,
    beta: float = 1.5,
    seed: int = 42,
    sigma_eq: float = 0.5,
    sigma_x: float = 1.0,
) -> pd.DataFrame:
    """
    Generate a cointegrated pair (y, x) where y = beta * x + stationary error.

    Both y and x are I(1) individually, but the linear combination
    y - beta * x is I(0).

    Parameters
    ----------
    n : int
        Number of observations.
    beta : float
        Cointegrating coefficient.
    seed : int
        Random seed for reproducibility.
    sigma_eq : float
        Std dev of the equilibrium error (stationary component).
    sigma_x : float
        Std dev of innovations driving x.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'y', 'x', and 'equilibrium_error'.
    """
    rng = np.random.default_rng(seed)

    # x is a random walk (I(1))
    eps_x = rng.normal(0, sigma_x, n)
    x = np.cumsum(eps_x)

    # Equilibrium error is stationary AR(1) with phi < 1
    eps_eq = rng.normal(0, sigma_eq, n)
    eq_error = np.zeros(n)
    eq_error[0] = eps_eq[0]
    for t in range(1, n):
        eq_error[t] = 0.3 * eq_error[t - 1] + eps_eq[t]

    # y = beta * x + stationary error => y and x are cointegrated
    y = beta * x + eq_error

    return pd.DataFrame({
        "y": y,
        "x": x,
        "equilibrium_error": eq_error,
    })


def generate_structural_break(
    n: int = 200,
    break_point: float = 0.5,
    shift: float = 3.0,
    seed: int = 42,
    sigma: float = 1.0,
    trend_before: float = 0.0,
    trend_after: float = 0.0,
) -> pd.DataFrame:
    """
    Generate a time series with a structural break in level and/or trend.

    Parameters
    ----------
    n : int
        Number of observations.
    break_point : float
        Fraction of sample where the break occurs (0 < break_point < 1).
    shift : float
        Level shift at the break point.
    seed : int
        Random seed for reproducibility.
    sigma : float
        Std dev of the noise.
    trend_before : float
        Linear trend coefficient before the break.
    trend_after : float
        Linear trend coefficient after the break.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'y', 'break_dummy', and 'true_break_index'.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, n)

    bp = int(n * break_point)
    t = np.arange(n)

    # Construct deterministic component
    mean = np.zeros(n)
    mean[:bp] = trend_before * t[:bp]
    mean[bp:] = trend_before * bp + shift + trend_after * (t[bp:] - bp)

    y = mean + eps

    break_dummy = np.zeros(n, dtype=int)
    break_dummy[bp:] = 1

    return pd.DataFrame({
        "y": y,
        "break_dummy": break_dummy,
        "true_break_index": bp,
    })


def generate_trend_stationary(
    n: int = 200,
    trend_coef: float = 0.05,
    seed: int = 42,
    sigma: float = 1.0,
    phi: float = 0.5,
) -> pd.DataFrame:
    """
    Generate trend-stationary and difference-stationary series for comparison.

    A trend-stationary process: y_t = a + b*t + u_t (stationary around trend).
    A difference-stationary process: y_t = y_{t-1} + drift + e_t (random walk with drift).

    Parameters
    ----------
    n : int
        Number of observations.
    trend_coef : float
        Slope of the deterministic trend.
    seed : int
        Random seed for reproducibility.
    sigma : float
        Std dev of innovations.
    phi : float
        AR(1) coefficient for the stationary error in the trend-stationary process.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'trend_stationary' and 'difference_stationary'.
    """
    rng = np.random.default_rng(seed)

    t = np.arange(n)
    eps1 = rng.normal(0, sigma, n)
    eps2 = rng.normal(0, sigma, n)

    # Trend-stationary: y = a + b*t + AR(1) error
    u = np.zeros(n)
    u[0] = eps1[0]
    for i in range(1, n):
        u[i] = phi * u[i - 1] + eps1[i]
    ts_series = 10.0 + trend_coef * t + u

    # Difference-stationary: random walk with drift
    ds_series = np.zeros(n)
    ds_series[0] = 10.0
    for i in range(1, n):
        ds_series[i] = ds_series[i - 1] + trend_coef + eps2[i]

    return pd.DataFrame({
        "trend_stationary": ts_series,
        "difference_stationary": ds_series,
    })
