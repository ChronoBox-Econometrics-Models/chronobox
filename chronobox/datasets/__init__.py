"""ChronoBox Datasets - Built-in time series datasets.

Provides ~30 datasets across 5 categories:
- Classic: airline, nile, sunspot, lynx, co2, uspop
- Macro International: us_gdp, us_macro_quarterly, us_consumption, canada,
  uk_drivers, uk_gas, denmark
- Macro Brazil: ipca, pib_brasil, selic, cambio, desemprego,
  producao_industrial, macro_brasil
- Finance: sp500_returns, ibovespa_returns, exchange_rates, vix
- Simulated: arma11, var2, cointegrated, structural_break, long_memory
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# Dataset catalog with metadata
DATASET_CATALOG: dict[str, dict[str, Any]] = {
    # Classic
    "airline": {
        "path": "classic/airline.csv",
        "category": "classic",
        "description": "Monthly airline passengers (1949-1960)",
    },
    "nile": {
        "path": "classic/nile.csv",
        "category": "classic",
        "description": "Annual Nile river flow (1871-1970)",
    },
    "sunspot": {
        "path": "classic/sunspot.csv",
        "category": "classic",
        "description": "Annual sunspot numbers (1700-2008)",
    },
    "lynx": {
        "path": "classic/lynx.csv",
        "category": "classic",
        "description": "Annual Canadian lynx trappings (1821-1934)",
    },
    "co2": {
        "path": "classic/co2.csv",
        "category": "classic",
        "description": "Monthly CO2 Mauna Loa (1959-1997)",
    },
    "uspop": {
        "path": "classic/uspop.csv",
        "category": "classic",
        "description": "US population decennial (1790-2000)",
    },
    # Macro International
    "us_gdp": {
        "path": "macro/us_gdp.csv",
        "category": "macro",
        "description": "US quarterly GDP (1947-2023)",
    },
    "us_macro_quarterly": {
        "path": "macro/us_macro_quarterly.csv",
        "category": "macro",
        "description": "US macro quarterly panel",
    },
    "us_consumption": {
        "path": "macro/us_consumption.csv",
        "category": "macro",
        "description": "US personal consumption monthly",
    },
    "canada": {
        "path": "macro/canada.csv",
        "category": "macro",
        "description": "Canada dataset (e, prod, rw, U)",
    },
    "uk_drivers": {
        "path": "macro/uk_drivers.csv",
        "category": "macro",
        "description": "UK road casualties (1969-1984)",
    },
    "uk_gas": {
        "path": "macro/uk_gas.csv",
        "category": "macro",
        "description": "UK quarterly gas consumption",
    },
    "denmark": {
        "path": "macro/denmark.csv",
        "category": "macro",
        "description": "Denmark money, income, prices, interest",
    },
    # Macro Brazil
    "ipca": {
        "path": "brazil/ipca.csv",
        "category": "brazil",
        "description": "IPCA monthly inflation",
    },
    "pib_brasil": {
        "path": "brazil/pib_brasil.csv",
        "category": "brazil",
        "description": "Brazil quarterly GDP",
    },
    "selic": {
        "path": "brazil/selic.csv",
        "category": "brazil",
        "description": "SELIC target rate",
    },
    "cambio": {
        "path": "brazil/cambio.csv",
        "category": "brazil",
        "description": "BRL/USD exchange rate",
    },
    "desemprego": {
        "path": "brazil/desemprego.csv",
        "category": "brazil",
        "description": "Unemployment rate (PNAD)",
    },
    "producao_industrial": {
        "path": "brazil/producao_industrial.csv",
        "category": "brazil",
        "description": "Industrial production index",
    },
    "macro_brasil": {
        "path": "brazil/macro_brasil.csv",
        "category": "brazil",
        "description": "Brazil macro panel",
    },
    # Finance
    "sp500_returns": {
        "path": "finance/sp500_returns.csv",
        "category": "finance",
        "description": "S&P 500 daily returns",
    },
    "ibovespa_returns": {
        "path": "finance/ibovespa_returns.csv",
        "category": "finance",
        "description": "Ibovespa daily returns",
    },
    "exchange_rates": {
        "path": "finance/exchange_rates.csv",
        "category": "finance",
        "description": "Multiple exchange rates",
    },
    "vix": {
        "path": "finance/vix.csv",
        "category": "finance",
        "description": "VIX volatility index",
    },
    # Simulated
    "arma11": {
        "path": None,
        "category": "simulated",
        "description": "Simulated ARMA(1,1) process",
    },
    "var2": {
        "path": None,
        "category": "simulated",
        "description": "Simulated bivariate VAR(2)",
    },
    "cointegrated": {
        "path": None,
        "category": "simulated",
        "description": "Simulated cointegrated pair",
    },
    "structural_break": {
        "path": None,
        "category": "simulated",
        "description": "Simulated structural break",
    },
    "long_memory": {
        "path": None,
        "category": "simulated",
        "description": "Simulated long memory (ARFIMA)",
    },
}


def load_dataset(name: str) -> pd.Series | pd.DataFrame:
    """Load a built-in dataset by name.

    Parameters
    ----------
    name : str
        Dataset name (see list_datasets() for available names).

    Returns
    -------
    pd.Series or pd.DataFrame
        Loaded dataset.

    Raises
    ------
    KeyError
        If dataset name is not found.
    FileNotFoundError
        If CSV file is missing (for non-simulated datasets).
    """
    if name not in DATASET_CATALOG:
        available = ", ".join(sorted(DATASET_CATALOG.keys()))
        raise KeyError(f"Dataset '{name}' not found. Available: {available}")

    info = DATASET_CATALOG[name]

    # Simulated datasets
    if info["path"] is None:
        return _load_simulated(name)

    # CSV datasets
    filepath = DATA_DIR / info["path"]
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {filepath}. "
            f"Run download scripts or check installation."
        )

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    if df.shape[1] == 1:
        return df.iloc[:, 0]
    return df


def _load_simulated(name: str) -> pd.Series | pd.DataFrame:
    """Load a simulated dataset."""
    from chronobox.datasets.simulated import (
        generate_arma11,
        generate_cointegrated,
        generate_long_memory,
        generate_structural_break,
        generate_var2,
    )

    generators: dict[str, Any] = {
        "arma11": generate_arma11,
        "var2": generate_var2,
        "cointegrated": generate_cointegrated,
        "structural_break": generate_structural_break,
        "long_memory": generate_long_memory,
    }

    return generators[name]()


def list_datasets(category: str | None = None) -> pd.DataFrame:
    """List all available datasets.

    Parameters
    ----------
    category : str | None
        Filter by category (classic, macro, brazil, finance, simulated).
        If None, returns all.

    Returns
    -------
    pd.DataFrame
        DataFrame with dataset metadata.
    """
    rows: list[dict[str, Any]] = []
    for name, info in DATASET_CATALOG.items():
        if category is None or info["category"] == category:
            rows.append(
                {
                    "name": name,
                    "category": info["category"],
                    "description": info["description"],
                    "bundled": info["path"] is not None,
                }
            )
    return pd.DataFrame(rows)


__all__ = ["DATASET_CATALOG", "list_datasets", "load_dataset"]
