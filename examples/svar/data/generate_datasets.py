#!/usr/bin/env python3
"""
Generate all datasets for the SVAR/BVAR examples.

Datasets:
  - us_macro_quarterly.csv: Synthetic US macro (200 quarterly obs, 4 variables)
  - demand_supply.csv: Synthetic demand-supply system (200 obs, 2 variables)
  - monetary_policy.csv: Synthetic monetary policy system (200 obs, 3 variables)
  - blanchard_quah.csv: Synthetic BQ-style system (300 obs, 2 variables)
  - sign_restriction.csv: Synthetic sign-restriction DGP (200 obs, 3 variables)

All datasets use fixed seeds for reproducibility and ISO 8601 date format.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Parent path available for importing utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_generators import (
    generate_demand_supply,
    generate_monetary_policy,
    generate_blanchard_quah,
    generate_sign_restriction_dgp,
)


def generate_demand_supply_csv(output_path: Path) -> None:
    """Generate demand-supply dataset for short-run identification examples."""
    data, params = generate_demand_supply(n=200, seed=42)

    dates = pd.date_range(start="1975-01-01", periods=len(data), freq="QS")

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "price": np.round(data[:, 0], 4),
        "quantity": np.round(data[:, 1], 4),
    })
    df.to_csv(output_path, index=False)
    print(f"  demand_supply.csv: {len(df)} rows, {len(df.columns) - 1} variables written")


def generate_monetary_policy_csv(output_path: Path) -> None:
    """Generate monetary policy dataset for Cholesky identification examples."""
    data, params = generate_monetary_policy(n=200, seed=42)

    dates = pd.date_range(start="1975-01-01", periods=len(data), freq="QS")

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "output_gap": np.round(data[:, 0], 4),
        "inflation": np.round(data[:, 1], 4),
        "interest_rate": np.round(data[:, 2], 4),
    })
    df.to_csv(output_path, index=False)
    print(f"  monetary_policy.csv: {len(df)} rows, {len(df.columns) - 1} variables written")


def generate_blanchard_quah_csv(output_path: Path) -> None:
    """Generate Blanchard-Quah dataset for long-run identification examples."""
    data, params = generate_blanchard_quah(n=300, seed=42)

    dates = pd.date_range(start="1950-01-01", periods=len(data), freq="QS")

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "output_growth": np.round(data[:, 0], 4),
        "unemployment": np.round(data[:, 1], 4),
    })
    df.to_csv(output_path, index=False)
    print(f"  blanchard_quah.csv: {len(df)} rows, {len(df.columns) - 1} variables written")


def generate_sign_restriction_csv(output_path: Path) -> None:
    """Generate sign-restriction dataset for set-identified SVAR examples."""
    data, params = generate_sign_restriction_dgp(n=200, seed=42)

    dates = pd.date_range(start="1975-01-01", periods=len(data), freq="QS")

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "output": np.round(data[:, 0], 4),
        "inflation": np.round(data[:, 1], 4),
        "rate": np.round(data[:, 2], 4),
    })
    df.to_csv(output_path, index=False)
    print(f"  sign_restriction.csv: {len(df)} rows, {len(df.columns) - 1} variables written")


def main():
    data_dir = Path(__file__).resolve().parent
    print("Generating SVAR/BVAR example datasets...")
    generate_demand_supply_csv(data_dir / "demand_supply.csv")
    generate_monetary_policy_csv(data_dir / "monetary_policy.csv")
    generate_blanchard_quah_csv(data_dir / "blanchard_quah.csv")
    generate_sign_restriction_csv(data_dir / "sign_restriction.csv")
    print("Done.")


if __name__ == "__main__":
    main()
