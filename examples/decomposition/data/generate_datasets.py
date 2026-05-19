#!/usr/bin/env python3
"""
Generate all datasets for the decomposition examples.

Datasets:
  - airline.csv: Classic Air Passengers (144 monthly obs, 1949-1960)
  - co2.csv: Mauna Loa CO2 monthly averages (468 obs, 1959-01 to 1997-12)
  - brazil_ipca.csv: Synthetic Brazilian IPCA monthly inflation (240 obs, seed=42)

All datasets use fixed seeds for reproducibility and ISO 8601 date format.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_airline_csv(output_path: Path) -> None:
    """Generate the classic airline passengers dataset.

    Monthly totals of international airline passengers (in thousands),
    January 1949 through December 1960. This is the well-known Box-Jenkins
    airline dataset with trend and multiplicative seasonality.
    """
    passengers = [
        112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
        115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
        145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
        171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
        196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
        204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
        242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
        284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
        315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
        340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
        360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
        417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
    ]

    dates = pd.date_range(start="1949-01-01", periods=144, freq="MS")
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "passengers": passengers})
    df.to_csv(output_path, index=False)
    print(f"  airline.csv: {len(df)} rows written")


def generate_co2_csv(output_path: Path) -> None:
    """Generate the Mauna Loa CO2 dataset.

    Monthly average atmospheric CO2 concentrations (ppm) at Mauna Loa
    Observatory, Hawaii. 468 observations from 1959-01 to 1997-12.

    This synthetic version replicates the key statistical properties:
    - Upward trend (~1.3 ppm/year in 1960s, accelerating to ~1.8 ppm/year in 1990s)
    - Strong annual seasonal cycle (~6 ppm peak-to-trough)
    - Seasonal amplitude that slightly increases over time
    """
    rng = np.random.default_rng(42)

    n = 468  # 39 years * 12 months (1959-01 to 1997-12)
    dates = pd.date_range(start="1959-01-01", periods=n, freq="MS")

    t = np.arange(n)
    years = t / 12.0

    # Trend: quadratic to capture acceleration
    # CO2 was ~315 ppm in 1959, ~365 ppm in 1997
    trend = 315.0 + 1.3 * years + 0.01 * years**2

    # Seasonal cycle: peaks in May, trough in September-October
    # Pattern based on Northern Hemisphere vegetation cycle
    month = np.array([d.month for d in dates])
    seasonal_pattern = {
        1: -0.5, 2: 0.2, 3: 1.2, 4: 2.5, 5: 3.0, 6: 2.0,
        7: 0.0, 8: -2.0, 9: -3.2, 10: -3.0, 11: -1.8, 12: -0.8,
    }
    seasonal = np.array([seasonal_pattern[m] for m in month])
    # Slight increase in seasonal amplitude over time
    seasonal = seasonal * (1.0 + 0.003 * years)

    # Residual: small autocorrelated noise
    residual = np.zeros(n)
    residual[0] = rng.normal(0, 0.3)
    for i in range(1, n):
        residual[i] = 0.4 * residual[i - 1] + rng.normal(0, 0.3)

    co2 = trend + seasonal + residual
    co2 = np.round(co2, 2)

    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "co2_ppm": co2})
    df.to_csv(output_path, index=False)
    print(f"  co2.csv: {len(df)} rows written")


def generate_brazil_ipca_csv(output_path: Path) -> None:
    """Generate synthetic Brazilian IPCA monthly inflation data.

    240 monthly observations (2004-01 to 2023-12) with seed=42.
    Simulates IPCA-like behavior: seasonal pattern, persistent level,
    occasional shocks.
    """
    rng = np.random.default_rng(42)

    n = 240
    dates = pd.date_range(start="2004-01-01", periods=n, freq="MS")

    # Base monthly inflation around 0.4% with AR(1) persistence
    base = np.zeros(n)
    base[0] = 0.4
    for t in range(1, n):
        base[t] = 0.1 + 0.7 * base[t - 1] + rng.normal(0, 0.15)

    # Seasonal component (higher in Jan/Feb, lower mid-year)
    seasonal = np.array([0.15, 0.10, 0.05, 0.02, -0.02, -0.05,
                         -0.03, -0.01, 0.02, 0.03, 0.05, 0.08])
    seasonal_component = np.tile(seasonal, n // 12)

    # Combine
    ipca = base + seasonal_component

    # Clip to reasonable range
    ipca = np.clip(ipca, -0.2, 2.5)
    ipca = np.round(ipca, 2)

    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "ipca": ipca})
    df.to_csv(output_path, index=False)
    print(f"  brazil_ipca.csv: {len(df)} rows written")


def main():
    data_dir = Path(__file__).resolve().parent
    print("Generating decomposition example datasets...")
    generate_airline_csv(data_dir / "airline.csv")
    generate_co2_csv(data_dir / "co2.csv")
    generate_brazil_ipca_csv(data_dir / "brazil_ipca.csv")
    print("Done.")


if __name__ == "__main__":
    main()
