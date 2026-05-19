"""
Generate synthetic datasets for filters examples.

Creates:
- us_gdp_quarterly.csv  : 200 quarterly observations mimicking US real GDP
- brazil_gdp.csv        : 120 quarterly observations mimicking Brazil GDP

All datasets use seed=42 for full reproducibility.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the utils package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_generators import generate_business_cycle, generate_trend_cycle

DATA_DIR = Path(__file__).resolve().parent


def create_us_gdp_quarterly(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic US GDP quarterly dataset (200 obs)."""
    rng = np.random.default_rng(seed)

    bc = generate_business_cycle(n=200, expansion_length=24, recession_depth=2.0, seed=seed)

    base_gdp = 2000 + np.cumsum(rng.normal(15, 8, size=200))
    base_gdp = base_gdp + bc["cycle"].values * 50

    base_gdp = np.maximum(base_gdp, 500)

    dates = pd.date_range(start="1970-01-01", periods=200, freq="QS")

    df = pd.DataFrame(
        {
            "date": dates,
            "gdp_real": np.round(base_gdp, 2),
            "gdp_log": np.round(np.log(base_gdp), 6),
        }
    )
    return df


def create_brazil_gdp(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic Brazil GDP quarterly dataset (120 obs)."""
    rng = np.random.default_rng(seed)

    tc = generate_trend_cycle(
        n=120, trend_type="quadratic", cycle_period=28, cycle_amp=3.0, seed=seed
    )

    base_level = 500
    gdp = base_level + tc["trend"].values - tc["trend"].values[0] + tc["cycle"].values * 20
    gdp += np.cumsum(rng.normal(2, 5, size=120))
    gdp = np.maximum(gdp, 100)

    dates = pd.date_range(start="2000-01-01", periods=120, freq="QS")

    df = pd.DataFrame(
        {
            "date": dates,
            "gdp_real": np.round(gdp, 2),
            "gdp_log": np.round(np.log(gdp), 6),
        }
    )
    return df


if __name__ == "__main__":
    print("Generating US GDP quarterly dataset (200 obs)...")
    us_gdp = create_us_gdp_quarterly()
    us_gdp.to_csv(DATA_DIR / "us_gdp_quarterly.csv", index=False)
    print(f"  Saved: {DATA_DIR / 'us_gdp_quarterly.csv'}")
    print(f"  Shape: {us_gdp.shape}")
    print(f"  Date range: {us_gdp['date'].iloc[0]} to {us_gdp['date'].iloc[-1]}")

    print("\nGenerating Brazil GDP dataset (120 obs)...")
    br_gdp = create_brazil_gdp()
    br_gdp.to_csv(DATA_DIR / "brazil_gdp.csv", index=False)
    print(f"  Saved: {DATA_DIR / 'brazil_gdp.csv'}")
    print(f"  Shape: {br_gdp.shape}")
    print(f"  Date range: {br_gdp['date'].iloc[0]} to {br_gdp['date'].iloc[-1]}")

    print("\nDone.")
