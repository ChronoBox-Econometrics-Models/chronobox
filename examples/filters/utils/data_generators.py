"""
Data generators for filters examples.

Generates synthetic macroeconomic-style time series with known trend and
cyclical components, useful for validating filter decompositions.
"""

import numpy as np
import pandas as pd


def generate_trend_cycle(
    n: int = 200,
    trend_type: str = "linear",
    cycle_period: int = 32,
    cycle_amp: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a time series with a deterministic trend and cyclical component.

    Parameters
    ----------
    n : int
        Number of observations.
    trend_type : str
        Type of trend: 'linear', 'quadratic', or 'log'.
    cycle_period : int
        Period of the cyclical component in observations.
    cycle_amp : float
        Amplitude of the cyclical component.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, observed, trend, cycle, noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1, dtype=float)

    if trend_type == "linear":
        trend = 100 + 0.5 * t
    elif trend_type == "quadratic":
        trend = 100 + 0.5 * t + 0.001 * t**2
    elif trend_type == "log":
        trend = 100 * np.log(t + 1)
    else:
        raise ValueError(f"Unknown trend_type: {trend_type}")

    cycle = cycle_amp * np.sin(2 * np.pi * t / cycle_period)
    noise = rng.normal(0, 0.3, size=n)
    observed = trend + cycle + noise

    dates = pd.date_range(start="1970-01-01", periods=n, freq="QS")

    return pd.DataFrame(
        {
            "date": dates,
            "observed": observed,
            "trend": trend,
            "cycle": cycle,
            "noise": noise,
        }
    )


def generate_business_cycle(
    n: int = 200,
    expansion_length: int = 24,
    recession_depth: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate a business-cycle series with asymmetric expansions/recessions.

    The series features a log-linear trend with regime-switching cyclical
    dynamics: long expansions followed by sharp, shorter recessions.

    Parameters
    ----------
    n : int
        Number of observations.
    expansion_length : int
        Average length of an expansion phase in observations.
    recession_depth : float
        Depth scaling factor for recession troughs.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, gdp, trend, cycle.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1, dtype=float)

    trend = np.log(100) + 0.005 * t

    cycle = np.zeros(n)
    phase = "expansion"
    phase_counter = 0
    current_cycle = 0.0

    for i in range(n):
        if phase == "expansion":
            current_cycle += rng.uniform(0.01, 0.04)
            phase_counter += 1
            if phase_counter >= expansion_length and rng.random() > 0.7:
                phase = "recession"
                phase_counter = 0
        else:
            current_cycle -= rng.uniform(0.05, 0.15) * recession_depth
            phase_counter += 1
            if phase_counter >= 6 and rng.random() > 0.5:
                phase = "expansion"
                phase_counter = 0
                current_cycle = max(current_cycle, -0.5)

        cycle[i] = current_cycle

    cycle = cycle - np.mean(cycle)
    noise = rng.normal(0, 0.01, size=n)
    gdp = np.exp(trend + cycle + noise) * 1000

    dates = pd.date_range(start="1970-01-01", periods=n, freq="QS")

    return pd.DataFrame(
        {
            "date": dates,
            "gdp": gdp,
            "trend": np.exp(trend) * 1000,
            "cycle": cycle,
        }
    )


def generate_multivariate_cycle(
    n: int = 200,
    k: int = 4,
    common_factor_weight: float = 0.6,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate multivariate series with a common cyclical factor.

    Creates *k* series that share a common business-cycle factor plus
    idiosyncratic components. Useful for testing spillover/connectedness
    measures.

    Parameters
    ----------
    n : int
        Number of observations.
    k : int
        Number of variables.
    common_factor_weight : float
        Weight of the common factor (0–1). Higher values mean more
        co-movement across series.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with a 'date' column and k variable columns (var_1 … var_k).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1, dtype=float)

    common_factor = np.sin(2 * np.pi * t / 32) + 0.5 * np.sin(2 * np.pi * t / 16)
    common_factor += np.cumsum(rng.normal(0, 0.05, size=n))

    dates = pd.date_range(start="1970-01-01", periods=n, freq="QS")
    data: dict[str, object] = {"date": dates}

    for j in range(1, k + 1):
        idio = np.cumsum(rng.normal(0, 0.1, size=n))
        idio += 0.8 * np.sin(2 * np.pi * t / (20 + 5 * j))
        series = common_factor_weight * common_factor + (1 - common_factor_weight) * idio
        trend_j = 100 + 0.3 * t
        data[f"var_{j}"] = trend_j + series + rng.normal(0, 0.2, size=n)

    return pd.DataFrame(data)
