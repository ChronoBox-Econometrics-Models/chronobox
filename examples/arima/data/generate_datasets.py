#!/usr/bin/env python3
"""
Generate all datasets for the ARIMA examples.

Datasets:
  - airline.csv: Classic Air Passengers (144 monthly obs, 1949-1960)
  - nile.csv: Nile River annual flow (100 obs, 1871-1970)
  - brazil_ipca.csv: Synthetic Brazilian IPCA monthly inflation (240 obs, seed=42)

All datasets use fixed seeds for reproducibility and ISO 8601 date format.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Parent path available for importing utils if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_airline_csv(output_path: Path) -> None:
    """Generate the classic airline passengers dataset.

    Monthly totals of international airline passengers (in thousands),
    January 1949 through December 1960. This is the well-known Box-Jenkins
    airline dataset with trend and multiplicative seasonality.
    """
    # Classic airline passengers data (Box & Jenkins)
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


def generate_nile_csv(output_path: Path) -> None:
    """Generate the Nile river flow dataset.

    Annual flow of the Nile River at Aswan, 1871-1970 (100 observations).
    Classic dataset for change-point analysis and ARIMA modeling.
    """
    # Nile river flow data (Cobb, 1978; Balke, 1993)
    flow = [
        1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140,
        995, 935, 1110, 994, 1020, 960, 1180, 799, 958, 1140,
        1100, 1210, 1150, 1250, 1260, 1220, 1030, 1100, 774, 840,
        874, 694, 940, 833, 701, 916, 692, 1020, 1050, 969,
        831, 726, 456, 824, 702, 1120, 1100, 832, 764, 821,
        768, 845, 864, 862, 698, 845, 744, 796, 1040, 759,
        781, 865, 845, 944, 984, 897, 822, 1010, 771, 676,
        649, 846, 812, 742, 801, 1040, 860, 874, 848, 890,
        744, 749, 838, 1050, 918, 986, 797, 923, 975, 815,
        1020, 906, 901, 1170, 912, 746, 919, 718, 714, 740,
    ]

    dates = pd.date_range(start="1871-01-01", periods=100, freq="YS")
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "flow": flow})
    df.to_csv(output_path, index=False)
    print(f"  nile.csv: {len(df)} rows written")


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
    print("Generating ARIMA example datasets...")
    generate_airline_csv(data_dir / "airline.csv")
    generate_nile_csv(data_dir / "nile.csv")
    generate_brazil_ipca_csv(data_dir / "brazil_ipca.csv")
    print("Done.")


if __name__ == "__main__":
    main()
