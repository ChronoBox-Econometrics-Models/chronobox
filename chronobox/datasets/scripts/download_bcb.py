"""Download macroeconomic data from BCB SGS (Banco Central do Brasil).

SGS - Sistema Gerenciador de Series Temporais
API: https://dadosabertos.bcb.gov.br/

Series codes:
- 433: IPCA (monthly inflation)
- 4380: PIB mensal (GDP proxy)
- 432: SELIC target rate
- 1: USD/BRL exchange rate
- 24369: Unemployment rate (PNAD)
- 21859: Industrial production
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

BCB_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json"

SERIES_CATALOG = {
    "ipca": {
        "code": 433,
        "description": "IPCA - Indice Nacional de Precos ao Consumidor Amplo (monthly)",
        "frequency": "monthly",
        "unit": "% monthly change",
    },
    "selic": {
        "code": 432,
        "description": "Taxa SELIC meta (annualized)",
        "frequency": "daily",
        "unit": "% per year",
    },
    "cambio": {
        "code": 1,
        "description": "USD/BRL exchange rate (buy)",
        "frequency": "daily",
        "unit": "BRL per USD",
    },
    "desemprego": {
        "code": 24369,
        "description": "Taxa de desemprego PNAD Continua",
        "frequency": "monthly",
        "unit": "%",
    },
    "producao_industrial": {
        "code": 21859,
        "description": "Producao industrial - Industria geral",
        "frequency": "monthly",
        "unit": "index",
    },
}


def download_bcb_series(
    code: int,
    start_date: str = "2000-01-01",
    end_date: str | None = None,
) -> pd.Series:
    """Download a time series from BCB SGS.

    Parameters
    ----------
    code : int
        SGS series code.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str | None
        End date. If None, uses today.

    Returns
    -------
    pd.Series
        Downloaded time series.

    Raises
    ------
    ConnectionError
        If the download fails.
    """
    import urllib.request

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    # Format dates for BCB API (dd/mm/yyyy)
    start_fmt = datetime.strptime(start_date, "%Y-%m-%d").strftime("%d/%m/%Y")
    end_fmt = datetime.strptime(end_date, "%Y-%m-%d").strftime("%d/%m/%Y")

    url = BCB_SGS_URL.format(code=code)
    url += f"&dataInicial={start_fmt}&dataFinal={end_fmt}"

    try:
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except Exception as e:
        raise ConnectionError(f"Failed to download BCB series {code}: {e}") from e

    data = json.loads(raw)

    dates: list[datetime] = []
    values: list[float] = []
    for entry in data:
        date_str = entry["data"]
        value_str = entry["valor"]
        try:
            date = datetime.strptime(date_str, "%d/%m/%Y")
            value = float(value_str)
            dates.append(date)
            values.append(value)
        except (ValueError, TypeError):
            continue

    series = pd.Series(values, index=pd.DatetimeIndex(dates), name=f"bcb_{code}")
    series = series.sort_index()
    return series


def download_all_brazil_data(
    output_dir: str | Path = ".",
    start_date: str = "2000-01-01",
) -> dict[str, pd.Series]:
    """Download all Brazilian macro series and save as CSV.

    Parameters
    ----------
    output_dir : str | Path
        Directory to save CSV files.
    start_date : str
        Start date for all series.

    Returns
    -------
    dict[str, pd.Series]
        Dictionary of downloaded series.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.Series] = {}

    for name, info in SERIES_CATALOG.items():
        try:
            series = download_bcb_series(int(info["code"]), start_date=start_date)
            series.name = name
            series.to_csv(output_path / f"{name}.csv")
            results[name] = series
            print(f"  Downloaded {name}: {len(series)} observations")
        except Exception as e:
            print(f"  Failed to download {name}: {e}")

    return results


def download_pib_trimestral(
    output_dir: str | Path = ".",
) -> pd.Series:
    """Download quarterly GDP from IBGE SIDRA.

    Uses the IBGE SIDRA API to download quarterly GDP at market prices.

    Parameters
    ----------
    output_dir : str | Path
        Directory to save CSV file.

    Returns
    -------
    pd.Series
        Quarterly GDP series.
    """
    import urllib.request

    # SIDRA table 1620 - GDP quarterly
    url = (
        "https://apisidra.ibge.gov.br/values/t/1620/n1/all/v/all"
        "/p/all/c11255/90687/d/v583%202"
    )

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")

        data = json.loads(raw)

        # Parse SIDRA response (first row is header)
        values: list[float] = []
        dates: list[pd.Timestamp] = []
        for row in data[1:]:
            try:
                period = row.get("D3C", "")
                value = float(row.get("V", "0"))
                # Period format: YYYYQQ (e.g., 200001 = 2000 Q1)
                year = int(period[:4])
                quarter = int(period[4:6])
                date = pd.Timestamp(year=year, month=quarter * 3, day=1)
                dates.append(date)
                values.append(value)
            except (ValueError, TypeError, KeyError):
                continue

        series = pd.Series(values, index=pd.DatetimeIndex(dates), name="pib_brasil")
        series = series.sort_index()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        series.to_csv(output_path / "pib_brasil.csv")

        return series

    except Exception as e:
        raise ConnectionError(f"Failed to download GDP from IBGE: {e}") from e


if __name__ == "__main__":
    import sys

    output = sys.argv[1] if len(sys.argv) > 1 else "."
    print("Downloading Brazilian macro data from BCB SGS...")
    download_all_brazil_data(output)
    print("\nDownloading quarterly GDP from IBGE SIDRA...")
    try:
        download_pib_trimestral(output)
    except Exception as e:
        print(f"Failed: {e}")
    print("\nDone!")
