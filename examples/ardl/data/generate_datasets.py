#!/usr/bin/env python3
"""
Generate all datasets for the ARDL examples.

Datasets:
  - us_macro_quarterly.csv: Synthetic US macro (200 quarterly obs, 4 variables)
  - ardl_synthetic.csv: Synthetic ARDL data with long-run relationship and
    mixed integration orders (200 obs, 4 variables)

All datasets use fixed seeds for reproducibility and ISO 8601 date format.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Parent path available for importing utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_generators import generate_mixed_integration


def generate_us_macro_quarterly_csv(output_path: Path) -> None:
    """Copy/generate the US macro quarterly dataset for ARDL examples.

    200 quarterly observations with 4 variables:
      - gdp: Real GDP growth rate (annualized %)
      - inflation: CPI inflation rate (annualized %)
      - fed_funds: Federal funds rate (%)
      - unemployment: Unemployment rate (%)
    """
    rng = np.random.default_rng(42)
    n = 200

    # VAR(2) coefficient matrices for [gdp, inflation, fed_funds, unemployment]
    A1 = np.array([
        [0.40, -0.10, -0.05, -0.05],
        [0.08, 0.50, 0.02, -0.05],
        [0.10, 0.15, 0.70, 0.05],
        [-0.08, 0.02, 0.03, 0.75],
    ])

    A2 = np.array([
        [0.10, 0.05, -0.02, 0.02],
        [0.03, 0.15, -0.01, 0.02],
        [0.05, 0.05, 0.10, -0.02],
        [-0.03, 0.01, 0.02, 0.10],
    ])

    sigma = np.array([
        [0.50, 0.05, 0.02, -0.10],
        [0.05, 0.20, 0.03, 0.02],
        [0.02, 0.03, 0.15, -0.01],
        [-0.10, 0.02, -0.01, 0.10],
    ])

    const = np.array([0.8, 0.3, 0.2, 0.5])

    k = 4
    p = 2
    burn = 200
    total = n + burn

    innovations = rng.multivariate_normal(np.zeros(k), sigma, total)
    A_matrices = [A1, A2]

    y = np.zeros((total, k))
    for t in range(total):
        y[t] = const + innovations[t]
        for lag_idx, A in enumerate(A_matrices):
            if t - lag_idx - 1 >= 0:
                y[t] += A @ y[t - lag_idx - 1]

    raw = y[burn:]

    gdp = np.clip(2.5 + raw[:, 0], -4.0, 10.0)
    inflation = np.clip(2.5 + raw[:, 1], -1.0, 12.0)
    fed_funds = np.clip(3.0 + raw[:, 2], 0.0, 15.0)
    unemployment = np.clip(5.5 + raw[:, 3], 2.0, 14.0)

    dates = pd.date_range(start="1975-01-01", periods=n, freq="QS")

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "gdp": np.round(gdp, 4),
        "inflation": np.round(inflation, 4),
        "fed_funds": np.round(fed_funds, 4),
        "unemployment": np.round(unemployment, 4),
    })
    df.to_csv(output_path, index=False)
    print(f"  us_macro_quarterly.csv: {len(df)} rows, {len(df.columns) - 1} variables written")


def generate_ardl_synthetic_csv(output_path: Path) -> None:
    """Generate synthetic ARDL dataset with mixed integration orders.

    200 observations with 4 variables of different integration orders:
      - y:  I(1), dependent variable cointegrated with x1
      - x1: I(1), cointegrated with y (long-run: y = 1.5 + 0.6*x1)
      - x2: I(0), stationary regressor
      - x3: I(1), independent (not cointegrated with y)

    This is the ideal scenario for Pesaran-Shin-Smith bounds testing.
    """
    raw = generate_mixed_integration(n=200, seed=42)

    dates = pd.date_range(start="1970-01-01", periods=200, freq="QS")

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "y": np.round(raw[:, 0], 4),
        "x1": np.round(raw[:, 1], 4),
        "x2": np.round(raw[:, 2], 4),
        "x3": np.round(raw[:, 3], 4),
    })
    df.to_csv(output_path, index=False)
    print(f"  ardl_synthetic.csv: {len(df)} rows, {len(df.columns) - 1} variables written")


def main():
    data_dir = Path(__file__).resolve().parent
    print("Generating ARDL example datasets...")
    generate_us_macro_quarterly_csv(data_dir / "us_macro_quarterly.csv")
    generate_ardl_synthetic_csv(data_dir / "ardl_synthetic.csv")
    print("Done.")


if __name__ == "__main__":
    main()
