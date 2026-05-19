#!/usr/bin/env python3
"""
Generate all datasets for the ETS examples.

Datasets:
  - airline.csv: Classic Air Passengers (144 monthly obs, 1949-1960) — shared with ARIMA
  - brazil_ipca.csv: Synthetic Brazilian IPCA monthly inflation (240 obs, seed=42) — shared with ARIMA
  - ets_synthetic.csv: Synthetic series with trend and multiplicative seasonality (180 obs, seed=42)

All datasets use fixed seeds for reproducibility and ISO 8601 date format.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing from utils/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_generators import generate_multiplicative_seasonal


def generate_ets_synthetic_csv(output_path: Path) -> None:
    """Generate a synthetic series with trend and multiplicative seasonality.

    180 monthly observations (2009-01 to 2023-12) with seed=42.
    Uses ETS(M,A,M) model: multiplicative errors and seasonality, additive trend.
    Suitable for Holt-Winters multiplicative and ETS model fitting.
    """
    n = 180
    dates = pd.date_range(start="2009-01-01", periods=n, freq="MS")

    y = generate_multiplicative_seasonal(
        n=n,
        level=200.0,
        trend=0.8,
        seasonal_amp=0.20,
        s=12,
        seed=42,
        sigma=3.0,
        alpha=0.25,
        beta=0.04,
        gamma=0.12,
    )

    # Round to 2 decimals for clean CSV output
    y = np.round(y, 2)

    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "value": y})
    df.to_csv(output_path, index=False)
    print(f"  ets_synthetic.csv: {len(df)} rows written")


def main():
    data_dir = Path(__file__).resolve().parent
    print("Generating ETS example datasets...")

    # Check that shared datasets exist
    for shared in ["airline.csv", "brazil_ipca.csv"]:
        path = data_dir / shared
        if path.exists():
            n_rows = sum(1 for _ in open(path)) - 1
            print(f"  {shared}: {n_rows} rows (shared with ARIMA)")
        else:
            print(f"  WARNING: {shared} not found — copy from examples/arima/data/")

    generate_ets_synthetic_csv(data_dir / "ets_synthetic.csv")
    print("Done.")


if __name__ == "__main__":
    main()
