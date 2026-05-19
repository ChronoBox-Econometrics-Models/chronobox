"""
Generate synthetic datasets for statistical tests examples.

Produces:
- us_gdp_quarterly.csv: Synthetic US real GDP (200 quarterly obs, seed=42)
- brazil_gdp.csv: Synthetic Brazil GDP (120 quarterly obs, seed=42)

Both datasets mimic realistic GDP dynamics with trend growth, persistence,
and potential structural features useful for unit root and cointegration testing.
"""

import os
import numpy as np
import pandas as pd



def generate_us_gdp_quarterly(seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic US real GDP quarterly data.

    Mimics US GDP characteristics:
    - 200 quarterly observations (1974Q1 to 2023Q4)
    - Trend growth ~0.7% per quarter (~2.8% annual)
    - High persistence (unit root in levels, stationary in growth rates)
    - Log-GDP follows approximately a random walk with drift

    Parameters
    ----------
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(seed)
    n = 200

    # Start with log(GDP) as random walk with drift
    drift = 0.007  # ~2.8% annual growth
    sigma = 0.008  # quarterly volatility
    eps = rng.normal(0, sigma, n)

    log_gdp = np.zeros(n)
    log_gdp[0] = np.log(5000)  # starting level in billions
    for t in range(1, n):
        log_gdp[t] = log_gdp[0] + drift * t + np.sum(eps[1:t + 1])

    gdp = np.exp(log_gdp)

    # Create date index
    dates = pd.date_range(start="1974-01-01", periods=n, freq="QS")

    df = pd.DataFrame({
        "date": dates,
        "gdp_real": np.round(gdp, 2),
        "log_gdp": np.round(log_gdp, 6),
        "gdp_growth": np.round(np.concatenate([[np.nan], np.diff(log_gdp) * 100]), 4),
    })

    return df


def generate_brazil_gdp(seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic Brazil GDP quarterly data.

    Mimics Brazil GDP characteristics:
    - 120 quarterly observations (1994Q1 to 2023Q4)
    - Higher volatility than US
    - Structural break around 2003 (policy shift)
    - Trend growth ~0.5% per quarter (~2% annual)

    Parameters
    ----------
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(seed)
    n = 120

    # Log GDP with drift and higher volatility
    drift_early = 0.004  # slower growth pre-2003
    drift_late = 0.008   # faster growth post-2003
    sigma = 0.012
    break_q = 36  # ~2003Q1

    eps = rng.normal(0, sigma, n)

    log_gdp = np.zeros(n)
    log_gdp[0] = np.log(800)  # starting level in BRL billions

    for t in range(1, n):
        d = drift_early if t < break_q else drift_late
        log_gdp[t] = log_gdp[t - 1] + d + eps[t]

    gdp = np.exp(log_gdp)

    # Create date index
    dates = pd.date_range(start="1994-01-01", periods=n, freq="QS")

    df = pd.DataFrame({
        "date": dates,
        "gdp_real": np.round(gdp, 2),
        "log_gdp": np.round(log_gdp, 6),
        "gdp_growth": np.round(np.concatenate([[np.nan], np.diff(log_gdp) * 100]), 4),
    })

    return df


if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.abspath(__file__))

    print("Generating US GDP quarterly dataset...")
    us_gdp = generate_us_gdp_quarterly(seed=42)
    us_path = os.path.join(data_dir, "us_gdp_quarterly.csv")
    us_gdp.to_csv(us_path, index=False)
    print(f"  Saved {len(us_gdp)} observations to {us_path}")
    print(f"  Date range: {us_gdp['date'].iloc[0]} to {us_gdp['date'].iloc[-1]}")
    print(f"  GDP range: {us_gdp['gdp_real'].min():.2f} to {us_gdp['gdp_real'].max():.2f}")

    print("\nGenerating Brazil GDP dataset...")
    br_gdp = generate_brazil_gdp(seed=42)
    br_path = os.path.join(data_dir, "brazil_gdp.csv")
    br_gdp.to_csv(br_path, index=False)
    print(f"  Saved {len(br_gdp)} observations to {br_path}")
    print(f"  Date range: {br_gdp['date'].iloc[0]} to {br_gdp['date'].iloc[-1]}")
    print(f"  GDP range: {br_gdp['gdp_real'].min():.2f} to {br_gdp['gdp_real'].max():.2f}")

    print("\nDone! All datasets generated successfully.")
