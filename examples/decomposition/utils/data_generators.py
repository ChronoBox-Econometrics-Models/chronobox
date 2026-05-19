"""
Data generators for decomposition examples.

Provides functions to generate synthetic time series with known components
(trend, seasonal, residual) for testing decomposition methods.
All functions use fixed seeds for reproducibility.
"""

import numpy as np


def generate_additive_seasonal(
    n: int = 120,
    trend_slope: float = 0.5,
    seasonal_amp: float = 10.0,
    noise_std: float = 2.0,
    s: int = 12,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a time series with additive decomposition: Y = T + S + R.

    Parameters
    ----------
    n : int
        Number of observations.
    trend_slope : float
        Linear trend slope per observation.
    seasonal_amp : float
        Amplitude of the seasonal component.
    noise_std : float
        Standard deviation of the residual (noise) component.
    s : int
        Seasonal period (e.g., 12 for monthly data).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of np.ndarray
        (series, trend, seasonal, residual) each of length n.
    """
    rng = np.random.default_rng(seed)

    t = np.arange(n)

    # Trend: linear with slight curvature
    trend = 100.0 + trend_slope * t + 0.001 * t**2

    # Seasonal: sinusoidal pattern
    seasonal_pattern = seasonal_amp * np.sin(2 * np.pi * np.arange(s) / s)
    # Ensure seasonal pattern sums to zero (additive constraint)
    seasonal_pattern -= seasonal_pattern.mean()
    seasonal = np.tile(seasonal_pattern, n // s + 1)[:n]

    # Residual: white noise
    residual = rng.normal(0, noise_std, n)

    series = trend + seasonal + residual

    return series, trend, seasonal, residual


def generate_multiplicative_seasonal(
    n: int = 144,
    base_level: float = 100.0,
    trend_rate: float = 0.003,
    seasonal_factor: float = 0.15,
    s: int = 12,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a time series with multiplicative decomposition: Y = T * S * R.

    Parameters
    ----------
    n : int
        Number of observations.
    base_level : float
        Base level of the series.
    trend_rate : float
        Exponential growth rate per observation.
    seasonal_factor : float
        Amplitude of the seasonal factors (as proportion of level).
    s : int
        Seasonal period.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of np.ndarray
        (series, trend, seasonal, residual) each of length n.
        - trend: the trend component
        - seasonal: multiplicative seasonal factors (centered around 1)
        - residual: multiplicative residual factors (centered around 1)
    """
    rng = np.random.default_rng(seed)

    t = np.arange(n)

    # Trend: exponential growth
    trend = base_level * np.exp(trend_rate * t)

    # Seasonal: multiplicative factors centered around 1
    # Classic pattern: higher in summer, lower in winter
    angles = 2 * np.pi * np.arange(s) / s
    seasonal_pattern = 1.0 + seasonal_factor * np.sin(angles + np.pi / 6)
    # Normalize so product over one cycle is ~1
    seasonal_pattern = seasonal_pattern / seasonal_pattern.mean()
    seasonal = np.tile(seasonal_pattern, n // s + 1)[:n]

    # Residual: multiplicative noise centered around 1
    residual = 1.0 + rng.normal(0, 0.03, n)

    series = trend * seasonal * residual

    return series, trend, seasonal, residual


def generate_complex_seasonal(
    n: int = 240,
    s1: int = 12,
    s2: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a time series with multiple seasonalities (additive).

    Useful for testing STL with multiple seasonal periods or MSTL.
    Y = T + S1 + S2 + R

    Parameters
    ----------
    n : int
        Number of observations.
    s1 : int
        Primary seasonal period (e.g., 12 for annual seasonality in monthly data).
    s2 : int
        Secondary seasonal period (e.g., 4 for quarterly pattern).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of np.ndarray
        (series, trend, seasonal1, seasonal2, residual) each of length n.
    """
    rng = np.random.default_rng(seed)

    t = np.arange(n)

    # Trend: piecewise linear with a change point
    trend = np.where(
        t < n // 2,
        50.0 + 0.3 * t,
        50.0 + 0.3 * (n // 2) + 0.1 * (t - n // 2),
    )

    # Primary seasonal (annual): strong sinusoidal
    s1_pattern = 8.0 * np.sin(2 * np.pi * np.arange(s1) / s1)
    s1_pattern -= s1_pattern.mean()
    seasonal1 = np.tile(s1_pattern, n // s1 + 1)[:n]

    # Secondary seasonal (quarterly): weaker pattern
    s2_pattern = 3.0 * np.cos(2 * np.pi * np.arange(s2) / s2)
    s2_pattern -= s2_pattern.mean()
    seasonal2 = np.tile(s2_pattern, n // s2 + 1)[:n]

    # Residual
    residual = rng.normal(0, 1.5, n)

    series = trend + seasonal1 + seasonal2 + residual

    return series, trend, seasonal1, seasonal2, residual
