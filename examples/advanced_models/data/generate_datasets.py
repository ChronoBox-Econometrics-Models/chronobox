"""
Generate datasets for advanced model examples.

This script creates:
- us_macro_panel.csv: Multi-country quarterly panel (5 countries, 4 variables, 80 obs)

us_macro_quarterly.csv is copied from the VAR/VECM examples.

Usage:
    python generate_datasets.py
"""

import sys
from pathlib import Path

# Add parent to path for utils import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_generators import generate_gvar_data


def main():
    output_dir = Path(__file__).resolve().parent

    # Generate multi-country panel dataset
    print("Generating us_macro_panel.csv...")
    df_panel = generate_gvar_data(n=80, n_countries=5, k=4, seed=42)
    df_panel.to_csv(output_dir / "us_macro_panel.csv", index=False)
    print(f"  Shape: {df_panel.shape}")
    print(f"  Countries: {df_panel['country'].unique().tolist()}")
    print(f"  Columns: {df_panel.columns.tolist()}")
    print(f"  Date range: {df_panel['date'].min()} to {df_panel['date'].max()}")

    print("\nDatasets generated successfully!")


if __name__ == "__main__":
    main()
