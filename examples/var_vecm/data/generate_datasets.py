#!/usr/bin/env python3
"""
Generate all datasets for the VAR/VECM examples.

Datasets:
  - canada_macro.csv: Synthetic Canada macro (84 quarterly obs, 4 variables)
  - us_macro_quarterly.csv: Synthetic US macro (200 quarterly obs, 4 variables)

All datasets use fixed seeds for reproducibility and ISO 8601 date format.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Parent path available for importing utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_generators import generate_var_process, generate_vecm_process


def generate_canada_macro_csv(output_path: Path) -> None:
    """Generate synthetic Canada macro dataset.

    Inspired by the 'Canada' dataset from the R 'vars' package.
    4 quarterly variables (1980Q1 to 2000Q4, 84 observations):
      - e: employment
      - prod: productivity (labor)
      - rw: real wage
      - U: unemployment rate

    The variables exhibit cointegration, mimicking the original dataset
    where employment, productivity, and wages share common stochastic trends.
    """
    n = 84

    # Generate a VECM-like process with 2 cointegrating relations among 4 variables
    # alpha: loading matrix (4 x 2)
    alpha = np.array([
        [-0.05, 0.02],
        [0.03, -0.04],
        [-0.02, 0.03],
        [0.04, -0.06],
    ])

    # beta: cointegrating vectors (4 x 2)
    beta = np.array([
        [1.0, 0.0],
        [-0.8, 1.0],
        [0.0, -0.7],
        [-0.3, -0.2],
    ])

    # Short-run dynamics
    Gamma1 = np.array([
        [0.2, 0.05, 0.0, -0.1],
        [0.0, 0.15, 0.05, 0.0],
        [0.05, 0.0, 0.1, 0.0],
        [-0.1, 0.0, 0.0, 0.3],
    ])

    sigma = np.array([
        [0.20, 0.05, 0.03, -0.02],
        [0.05, 0.15, 0.02, 0.01],
        [0.03, 0.02, 0.10, -0.01],
        [-0.02, 0.01, -0.01, 0.25],
    ])

    raw = generate_vecm_process(
        n=n, alpha=alpha, beta=beta, Gamma=[Gamma1],
        sigma=sigma, seed=42,
    )

    # Transform to realistic scales matching the original Canada dataset
    # e (employment): ~920-990 range
    e = 950 + 15 * raw[:, 0]
    # prod (productivity): ~400-470 range
    prod = 430 + 12 * raw[:, 1]
    # rw (real wage): ~-1 to 4 range
    rw = 1.5 + 1.2 * raw[:, 2]
    # U (unemployment): ~7-12 range
    U = 9.0 + 1.5 * raw[:, 3]
    U = np.clip(U, 4.0, 16.0)

    dates = pd.date_range(start="1980-01-01", periods=n, freq="QS")

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "e": np.round(e, 4),
        "prod": np.round(prod, 4),
        "rw": np.round(rw, 4),
        "U": np.round(U, 4),
    })
    df.to_csv(output_path, index=False)
    print(f"  canada_macro.csv: {len(df)} rows, {len(df.columns) - 1} variables written")


def generate_us_macro_quarterly_csv(output_path: Path) -> None:
    """Generate synthetic US macro quarterly dataset.

    200 quarterly observations with 4 variables:
      - gdp: Real GDP growth rate (annualized %)
      - inflation: CPI inflation rate (annualized %)
      - fed_funds: Federal funds rate (%)
      - unemployment: Unemployment rate (%)

    Generated as a VAR(2) process with realistic cross-variable dynamics:
    - Fed responds to inflation and output gap
    - Unemployment responds negatively to GDP growth (Okun's law)
    - Inflation responds to output and monetary policy
    """
    n = 200

    # VAR(2) coefficient matrices for [gdp, inflation, fed_funds, unemployment]
    A1 = np.array([
        [0.40, -0.10, -0.05, -0.05],   # GDP persistence, negative response to inflation/rates
        [0.08, 0.50, 0.02, -0.05],      # Inflation responds to output, own persistence
        [0.10, 0.15, 0.70, 0.05],       # Fed responds to GDP, inflation; strong persistence
        [-0.08, 0.02, 0.03, 0.75],      # Unemployment: Okun's law, persistence
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

    # Constant terms (unconditional means contribution)
    const = np.array([0.8, 0.3, 0.2, 0.5])

    raw = generate_var_process(
        n=n, A_matrices=[A1, A2], sigma=sigma,
        seed=42, const=const,
    )

    # Transform to realistic scales
    gdp = 2.5 + raw[:, 0]           # ~0-5% range
    inflation = 2.5 + raw[:, 1]      # ~1-5% range
    fed_funds = 3.0 + raw[:, 2]      # ~0-8% range
    unemployment = 5.5 + raw[:, 3]   # ~3-10% range

    # Clip to reasonable ranges
    gdp = np.clip(gdp, -4.0, 10.0)
    inflation = np.clip(inflation, -1.0, 12.0)
    fed_funds = np.clip(fed_funds, 0.0, 15.0)
    unemployment = np.clip(unemployment, 2.0, 14.0)

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


def main():
    data_dir = Path(__file__).resolve().parent
    print("Generating VAR/VECM example datasets...")
    generate_canada_macro_csv(data_dir / "canada_macro.csv")
    generate_us_macro_quarterly_csv(data_dir / "us_macro_quarterly.csv")
    print("Done.")


if __name__ == "__main__":
    main()
