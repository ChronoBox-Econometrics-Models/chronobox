"""
Data generators for ETS examples.

Provides functions to generate synthetic ETS, multiplicative seasonal,
and damped trend processes with fixed seeds for reproducibility.
"""

import numpy as np


def generate_ets_process(
    n: int = 200,
    alpha: float = 0.3,
    beta: float = 0.1,
    gamma: float = 0.15,
    seasonal_periods: int = 12,
    seed: int = 42,
    sigma: float = 1.0,
    level_start: float = 100.0,
    trend_start: float = 0.5,
) -> np.ndarray:
    """Generate an ETS(A,A,A) process — additive error, trend, and seasonality.

    Implements the state-space form:
        y_t = l_{t-1} + b_{t-1} + s_{t-m} + e_t
        l_t = l_{t-1} + b_{t-1} + alpha * e_t
        b_t = b_{t-1} + beta * e_t
        s_t = s_{t-m} + gamma * e_t

    Parameters
    ----------
    n : int
        Number of observations to generate.
    alpha : float
        Level smoothing parameter (0 < alpha < 1).
    beta : float
        Trend smoothing parameter (0 < beta < 1).
    gamma : float
        Seasonal smoothing parameter (0 < gamma < 1).
    seasonal_periods : int
        Number of periods in one seasonal cycle (e.g., 12 for monthly data).
    seed : int
        Random seed for reproducibility.
    sigma : float
        Standard deviation of the innovation process.
    level_start : float
        Initial level.
    trend_start : float
        Initial trend (slope).

    Returns
    -------
    np.ndarray
        Generated ETS(A,A,A) process of length n.
    """
    rng = np.random.default_rng(seed)
    m = seasonal_periods

    # Initialize seasonal indices (sum to zero for additive)
    seasonal_init = rng.normal(0, sigma * 2, m)
    seasonal_init -= seasonal_init.mean()

    # State vectors
    level = np.zeros(n + 1)
    trend = np.zeros(n + 1)
    seasonal = np.zeros(n + m)

    level[0] = level_start
    trend[0] = trend_start
    seasonal[:m] = seasonal_init

    # Innovations
    errors = rng.normal(0, sigma, n)

    y = np.zeros(n)

    for t in range(n):
        y[t] = level[t] + trend[t] + seasonal[t] + errors[t]
        level[t + 1] = level[t] + trend[t] + alpha * errors[t]
        trend[t + 1] = trend[t] + beta * errors[t]
        seasonal[t + m] = seasonal[t] + gamma * errors[t]

    return y


def generate_multiplicative_seasonal(
    n: int = 200,
    level: float = 100.0,
    trend: float = 0.5,
    seasonal_amp: float = 0.15,
    s: int = 12,
    seed: int = 42,
    sigma: float = 2.0,
    alpha: float = 0.3,
    beta: float = 0.05,
    gamma: float = 0.1,
) -> np.ndarray:
    """Generate an ETS(M,A,M) process — multiplicative error and seasonality, additive trend.

    Implements the state-space form:
        y_t = (l_{t-1} + b_{t-1}) * s_{t-m} * (1 + e_t)
        l_t = (l_{t-1} + b_{t-1}) * (1 + alpha * e_t)
        b_t = b_{t-1} + beta * (l_{t-1} + b_{t-1}) * e_t  (approximation)
        s_t = s_{t-m} * (1 + gamma * e_t)

    Parameters
    ----------
    n : int
        Number of observations to generate.
    level : float
        Initial level.
    trend : float
        Initial trend (additive increment per period).
    seasonal_amp : float
        Amplitude of seasonal factors around 1.0. Seasonal indices are
        initialized as 1 +/- seasonal_amp * pattern.
    s : int
        Seasonal period (e.g., 12 for monthly data).
    seed : int
        Random seed for reproducibility.
    sigma : float
        Scale of the relative error (percentage of level).
    alpha : float
        Level smoothing parameter.
    beta : float
        Trend smoothing parameter.
    gamma : float
        Seasonal smoothing parameter.

    Returns
    -------
    np.ndarray
        Generated ETS(M,A,M) process of length n.
    """
    rng = np.random.default_rng(seed)

    # Initialize seasonal indices (multiplicative: centered on 1.0)
    # Typical monthly pattern: peaks mid-year
    pattern = np.sin(2 * np.pi * np.arange(s) / s)
    seasonal_init = 1.0 + seasonal_amp * pattern
    # Normalize so product = 1 (geometric mean = 1)
    seasonal_init /= seasonal_init.prod() ** (1.0 / s)

    # State vectors
    l = np.zeros(n + 1)
    b = np.zeros(n + 1)
    sea = np.zeros(n + s)

    l[0] = level
    b[0] = trend
    sea[:s] = seasonal_init

    # Relative errors (small, centered on 0)
    errors = rng.normal(0, sigma / level, n)

    y = np.zeros(n)

    for t in range(n):
        y[t] = (l[t] + b[t]) * sea[t] * (1 + errors[t])
        l[t + 1] = (l[t] + b[t]) * (1 + alpha * errors[t])
        b[t + 1] = b[t] + beta * (l[t] + b[t]) * errors[t]
        sea[t + s] = sea[t] * (1 + gamma * errors[t])

    return y


def generate_damped_trend(
    n: int = 200,
    alpha: float = 0.3,
    beta: float = 0.1,
    phi: float = 0.9,
    seed: int = 42,
    sigma: float = 1.0,
    level_start: float = 100.0,
    trend_start: float = 2.0,
) -> np.ndarray:
    """Generate an ETS(A,Ad,N) process — additive error, damped trend, no seasonality.

    Implements the state-space form:
        y_t = l_{t-1} + phi * b_{t-1} + e_t
        l_t = l_{t-1} + phi * b_{t-1} + alpha * e_t
        b_t = phi * b_{t-1} + beta * e_t

    The damping parameter phi (0 < phi < 1) causes the trend to
    gradually flatten, producing forecasts that converge to a constant.

    Parameters
    ----------
    n : int
        Number of observations to generate.
    alpha : float
        Level smoothing parameter (0 < alpha < 1).
    beta : float
        Trend smoothing parameter (0 < beta < 1).
    phi : float
        Damping parameter (0 < phi < 1). Values near 1 give near-linear
        trend; values near 0 give rapid damping.
    seed : int
        Random seed for reproducibility.
    sigma : float
        Standard deviation of the innovation process.
    level_start : float
        Initial level.
    trend_start : float
        Initial trend.

    Returns
    -------
    np.ndarray
        Generated ETS(A,Ad,N) process of length n.
    """
    rng = np.random.default_rng(seed)

    # State vectors
    level = np.zeros(n + 1)
    trend_vec = np.zeros(n + 1)

    level[0] = level_start
    trend_vec[0] = trend_start

    # Innovations
    errors = rng.normal(0, sigma, n)

    y = np.zeros(n)

    for t in range(n):
        y[t] = level[t] + phi * trend_vec[t] + errors[t]
        level[t + 1] = level[t] + phi * trend_vec[t] + alpha * errors[t]
        trend_vec[t + 1] = phi * trend_vec[t] + beta * errors[t]

    return y
