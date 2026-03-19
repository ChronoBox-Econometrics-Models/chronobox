"""ChronoBox CLI - Main entry point.

Usage:
    chronobox estimate arima --order 0,1,1 --data airline.csv
    chronobox test adf --data gdp.csv
    chronobox forecast --model arima --steps 12 --data airline.csv
    chronobox decompose --method stl --data co2.csv
    chronobox filter hp --data gdp.csv --lamb 1600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_data(data_arg: str, column: str | None = None) -> pd.Series | pd.DataFrame:
    """Load data from CSV file or built-in dataset name.

    Parameters
    ----------
    data_arg : str
        Path to CSV file or name of built-in dataset.
    column : str | None
        Column name to select from DataFrame. If None and DataFrame has
        multiple columns, returns DataFrame for multivariate models.

    Returns
    -------
    pd.Series or pd.DataFrame
        Loaded time series data.
    """
    path = Path(data_arg)
    if path.exists() and path.suffix == ".csv":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if column is not None:
            return df[column]
        if df.shape[1] == 1:
            return df.iloc[:, 0]
        return df
    else:
        from chronobox.datasets import load_dataset

        dataset = load_dataset(data_arg)
        if isinstance(dataset, pd.Series):
            return dataset
        if column is not None:
            return dataset[column]
        if dataset.shape[1] == 1:
            return dataset.iloc[:, 0]
        return dataset


def _parse_order(order_str: str) -> tuple[int, ...]:
    """Parse comma-separated order string into tuple.

    Parameters
    ----------
    order_str : str
        Comma-separated integers, e.g. '0,1,1' or '0,1,1,12'.

    Returns
    -------
    tuple[int, ...]
        Parsed order tuple.
    """
    return tuple(int(x.strip()) for x in order_str.split(","))


def _format_output(result: Any, output_format: str = "text") -> str:
    """Format result for output.

    Parameters
    ----------
    result : Any
        Result object to format.
    output_format : str
        Output format: 'text', 'json', or 'csv'.

    Returns
    -------
    str
        Formatted output string.
    """
    if output_format == "json":
        if hasattr(result, "to_dict"):
            return json.dumps(result.to_dict(), indent=2, default=str)
        elif isinstance(result, (pd.DataFrame, pd.Series)):
            return result.to_json(indent=2)
        else:
            return json.dumps(str(result), default=str)
    elif output_format == "csv":
        if isinstance(result, (pd.DataFrame, pd.Series)):
            return result.to_csv()
        else:
            return str(result)
    else:
        if hasattr(result, "summary"):
            return result.summary()
        return str(result)


# ──────────────────────────────────────────────
# Command: estimate
# ──────────────────────────────────────────────

def cmd_estimate(args: argparse.Namespace) -> None:
    """Execute the 'estimate' command.

    Fits a time series model to the provided data and prints summary.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments containing model type, order, data path, etc.
    """
    data = _load_data(args.data, getattr(args, "column", None))

    model_type = args.model_type.lower()

    if model_type == "arima":
        from chronobox import ARIMA

        order = _parse_order(args.order) if args.order else (1, 0, 0)
        seasonal_order: tuple[int, ...] | None = None
        if args.seasonal:
            seasonal_order = _parse_order(args.seasonal)

        if seasonal_order and len(seasonal_order) == 4:
            model = ARIMA(
                order=(order[0], order[1], order[2]),
                seasonal_order=(
                    seasonal_order[0],
                    seasonal_order[1],
                    seasonal_order[2],
                    seasonal_order[3],
                ),
            )
        else:
            model = ARIMA(order=(order[0], order[1], order[2]))

        y = np.asarray(data, dtype=np.float64)
        results = model.fit(y)
        print(_format_output(results, args.format))

    elif model_type == "auto_arima":
        from chronobox import auto_arima

        seasonal = getattr(args, "seasonal", None)
        m = int(args.m) if args.m else 1

        y = np.asarray(data, dtype=np.float64)
        results = auto_arima(
            y,
            seasonal=seasonal is not None,
            m=m,
        )
        print(_format_output(results, args.format))

    elif model_type == "var":
        from chronobox.models.var import VAR

        if isinstance(data, pd.Series):
            print("ERROR: VAR requires multivariate data (DataFrame).", file=sys.stderr)
            sys.exit(1)

        maxlags = int(args.maxlags) if args.maxlags else 8
        model = VAR(maxlags=maxlags)
        results = model.fit(data)
        print(_format_output(results, args.format))

    elif model_type == "vecm":
        from chronobox.models.vecm import VECM

        if isinstance(data, pd.Series):
            print("ERROR: VECM requires multivariate data (DataFrame).", file=sys.stderr)
            sys.exit(1)

        k_ar_diff = int(args.lags) if args.lags else 1
        coint_rank = int(args.rank) if args.rank else 1
        model = VECM(lags=k_ar_diff, coint_rank=coint_rank)
        results = model.fit(data)
        print(_format_output(results, args.format))

    elif model_type == "ardl":
        from chronobox.models.ardl import ARDL

        if isinstance(data, pd.Series):
            print("ERROR: ARDL requires multivariate data (DataFrame).", file=sys.stderr)
            sys.exit(1)

        max_p = int(args.lags) if args.lags else 4
        y_col = data.iloc[:, 0].values
        x_cols = data.iloc[:, 1:].values
        model = ARDL(max_p=max_p)
        results = model.fit(y_col, x_cols)
        print(_format_output(results, args.format))

    else:
        print(f"ERROR: Unknown model type '{model_type}'.", file=sys.stderr)
        print("Available models: arima, auto_arima, var, vecm, ardl", file=sys.stderr)
        sys.exit(1)


# ──────────────────────────────────────────────
# Command: test
# ──────────────────────────────────────────────

def cmd_test(args: argparse.Namespace) -> None:
    """Execute the 'test' command.

    Runs a statistical test on the provided data.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments containing test type, data path, etc.
    """
    data = _load_data(args.data, getattr(args, "column", None))

    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            data = data.iloc[:, 0]
        else:
            column = getattr(args, "column", None)
            if column is None:
                print("ERROR: Multivariate data requires --column.", file=sys.stderr)
                sys.exit(1)
            data = data[column]

    test_type = args.test_type.lower()
    y: np.ndarray[Any, np.dtype[np.float64]] = np.asarray(data, dtype=np.float64)

    if test_type == "adf":
        from chronobox.tests_stat import adf_test

        result = adf_test(y)
        print(_format_output(result, args.format))

    elif test_type == "pp":
        from chronobox.tests_stat import pp_test

        result = pp_test(y)
        print(_format_output(result, args.format))

    elif test_type == "kpss":
        from chronobox.tests_stat import kpss_test

        result = kpss_test(y)
        print(_format_output(result, args.format))

    elif test_type == "ers":
        from chronobox.tests_stat import ers_test

        result = ers_test(y)
        print(_format_output(result, args.format))

    elif test_type == "ljungbox":
        from chronobox.tests_stat import ljung_box_test

        lags = int(args.lags) if getattr(args, "lags", None) else 10
        result = ljung_box_test(y, lags=lags)
        print(_format_output(result, args.format))

    elif test_type == "engle_granger":
        from chronobox.tests_stat import engle_granger_test

        multidata = _load_data(args.data)
        if isinstance(multidata, pd.Series):
            print("ERROR: Engle-Granger test requires multivariate data.", file=sys.stderr)
            sys.exit(1)
        vals = multidata.values.astype(np.float64)
        result = engle_granger_test(vals[:, 0], vals[:, 1:])
        print(_format_output(result, args.format))

    else:
        print(f"ERROR: Unknown test type '{test_type}'.", file=sys.stderr)
        print(
            "Available tests: adf, pp, kpss, ers, ljungbox, engle_granger",
            file=sys.stderr,
        )
        sys.exit(1)


# ──────────────────────────────────────────────
# Command: forecast
# ──────────────────────────────────────────────

def cmd_forecast(args: argparse.Namespace) -> None:
    """Execute the 'forecast' command.

    Fits a model and produces forecasts.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments containing model type, steps, data path, etc.
    """
    data = _load_data(args.data, getattr(args, "column", None))
    steps = int(args.steps) if args.steps else 12
    model_type = args.model.lower()

    if model_type == "arima":
        from chronobox import ARIMA

        order = _parse_order(args.order) if args.order else (1, 0, 0)
        seasonal_order_val: tuple[int, ...] | None = None
        if args.seasonal:
            seasonal_order_val = _parse_order(args.seasonal)

        if seasonal_order_val and len(seasonal_order_val) == 4:
            model = ARIMA(
                order=(order[0], order[1], order[2]),
                seasonal_order=(
                    seasonal_order_val[0],
                    seasonal_order_val[1],
                    seasonal_order_val[2],
                    seasonal_order_val[3],
                ),
            )
        else:
            model = ARIMA(order=(order[0], order[1], order[2]))

        y = np.asarray(data, dtype=np.float64)
        results = model.fit(y)

    elif model_type == "auto_arima":
        from chronobox import auto_arima

        m = int(args.m) if getattr(args, "m", None) else 1
        y = np.asarray(data, dtype=np.float64)
        results = auto_arima(y, seasonal=m > 1, m=m)

    elif model_type == "var":
        from chronobox.models.var import VAR

        multidata = _load_data(args.data)
        if isinstance(multidata, pd.Series):
            print("ERROR: VAR requires multivariate data.", file=sys.stderr)
            sys.exit(1)
        maxlags = int(args.maxlags) if getattr(args, "maxlags", None) else 8
        model = VAR(maxlags=maxlags)
        results = model.fit(multidata)

    else:
        print(f"ERROR: Unknown model '{model_type}'.", file=sys.stderr)
        sys.exit(1)

    forecast_result = results.forecast(steps=steps)

    # ARIMA forecast returns dict with 'forecast', 'lower', 'upper'
    if isinstance(forecast_result, dict):
        forecast_vals = forecast_result["forecast"]
        forecast_series = pd.Series(forecast_vals, name="forecast")
    elif isinstance(forecast_result, (pd.DataFrame, pd.Series)):
        forecast_series = forecast_result
    else:
        forecast_series = pd.Series(
            forecast_result,
            name="forecast",
        )

    if args.format == "json":
        print(forecast_series.to_json(indent=2))
    elif args.format == "csv":
        print(forecast_series.to_csv())
    else:
        print(f"Forecast ({steps} steps):")
        print(forecast_series.to_string())

    if args.output:
        output_path = Path(args.output)
        forecast_series.to_csv(output_path)
        print(f"\nForecast saved to {output_path}", file=sys.stderr)


# ──────────────────────────────────────────────
# Command: decompose
# ──────────────────────────────────────────────

def cmd_decompose(args: argparse.Namespace) -> None:
    """Execute the 'decompose' command.

    Decomposes a time series into trend, seasonal, and residual components.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments containing method, data path, etc.
    """
    data = _load_data(args.data, getattr(args, "column", None))

    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            data = data.iloc[:, 0]
        else:
            print("ERROR: Decompose requires univariate data. Use --column.", file=sys.stderr)
            sys.exit(1)

    method = args.method.lower()
    period = int(args.period) if getattr(args, "period", None) else None

    y: np.ndarray[Any, np.dtype[np.float64]] = np.asarray(data, dtype=np.float64)

    if method == "stl":
        from chronobox.decomposition import STL

        if period is None:
            print("ERROR: STL requires --period.", file=sys.stderr)
            sys.exit(1)
        decomp = STL(period=period)
        result = decomp.fit(y)

    elif method == "classical":
        from chronobox.decomposition import ClassicalDecomposition

        model_type = getattr(args, "type", "additive")
        if period is None:
            print("ERROR: Classical decomposition requires --period.", file=sys.stderr)
            sys.exit(1)
        decomp_cls = ClassicalDecomposition(period=period, model=model_type)
        result = decomp_cls.fit(y)

    else:
        print(f"ERROR: Unknown decomposition method '{method}'.", file=sys.stderr)
        print("Available methods: stl, classical", file=sys.stderr)
        sys.exit(1)

    print(_format_output(result, args.format))

    if getattr(args, "plot", False) and hasattr(result, "plot"):
        result.plot()  # type: ignore[union-attr]

    if args.output:
        output_path = Path(args.output)
        df_out = pd.DataFrame({
            "trend": result.trend,
            "seasonal": result.seasonal,
            "remainder": result.remainder,
        })
        df_out.to_csv(output_path)
        print(f"\nDecomposition saved to {output_path}", file=sys.stderr)


# ──────────────────────────────────────────────
# Command: filter
# ──────────────────────────────────────────────

def cmd_filter(args: argparse.Namespace) -> None:
    """Execute the 'filter' command.

    Applies a time series filter to extract trend and cycle components.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments containing filter type, data path, lambda, etc.
    """
    data = _load_data(args.data, getattr(args, "column", None))

    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            data = data.iloc[:, 0]
        else:
            print("ERROR: Filter requires univariate data. Use --column.", file=sys.stderr)
            sys.exit(1)

    y: np.ndarray[Any, np.dtype[np.float64]] = np.asarray(data, dtype=np.float64)
    filter_type = args.filter_type.lower()

    if filter_type == "hp":
        from chronobox.filters import hp_filter

        lamb = float(args.lamb) if getattr(args, "lamb", None) else 1600.0
        trend, cycle = hp_filter(y, lamb=lamb)

    elif filter_type == "bk":
        from chronobox.filters import bk_filter

        low = int(args.low) if getattr(args, "low", None) else 6
        high = int(args.high) if getattr(args, "high", None) else 32
        trunc = int(args.K) if getattr(args, "K", None) else 12
        cycle = bk_filter(y, low=low, high=high, trunc=trunc)
        # BK filter trims observations; trend = observed - cycle for the trimmed range
        trimmed = y[trunc : len(y) - trunc]
        trend = trimmed - cycle

    elif filter_type == "cf":
        from chronobox.filters import cf_filter

        low = int(args.low) if getattr(args, "low", None) else 6
        high = int(args.high) if getattr(args, "high", None) else 32
        cycle = cf_filter(y, low=low, high=high)
        trend = y - cycle

    elif filter_type == "hamilton":
        from chronobox.filters import hamilton_filter

        h = int(args.h) if getattr(args, "h", None) else 8
        p = int(args.p) if getattr(args, "p", None) else 4
        trend, cycle = hamilton_filter(y, h=h, p=p)

    else:
        print(f"ERROR: Unknown filter type '{filter_type}'.", file=sys.stderr)
        print("Available filters: hp, bk, cf, hamilton", file=sys.stderr)
        sys.exit(1)

    output_df = pd.DataFrame({"trend": trend, "cycle": cycle})
    print(_format_output(output_df, args.format))

    if args.output:
        output_path = Path(args.output)
        output_df.to_csv(output_path)
        print(f"\nFilter output saved to {output_path}", file=sys.stderr)


# ──────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ChronoBox CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="chronobox",
        description="ChronoBox - Time Series Analysis CLI",
        epilog="Use 'chronobox <command> --help' for more information on a command.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── estimate ──
    est_parser = subparsers.add_parser("estimate", help="Estimate a time series model")
    est_parser.add_argument("model_type", help="Model type: arima, auto_arima, var, vecm, ardl")
    est_parser.add_argument("--data", required=True, help="CSV file path or built-in dataset name")
    est_parser.add_argument("--column", help="Column name for univariate extraction")
    est_parser.add_argument("--order", help="Model order, e.g. '0,1,1'")
    est_parser.add_argument("--seasonal", help="Seasonal order, e.g. '0,1,1,12'")
    est_parser.add_argument("--m", help="Seasonal period for auto_arima")
    est_parser.add_argument("--lags", help="Number of lags for VAR/VECM/ARDL")
    est_parser.add_argument("--maxlags", help="Maximum lags for VAR lag selection")
    est_parser.add_argument("--rank", help="Cointegration rank for VECM")
    est_parser.add_argument(
        "--format", default="text", choices=["text", "json", "csv"], help="Output format"
    )
    est_parser.set_defaults(func=cmd_estimate)

    # ── test ──
    test_parser = subparsers.add_parser("test", help="Run statistical tests")
    test_parser.add_argument(
        "test_type", help="Test type: adf, pp, kpss, ers, ljungbox, engle_granger"
    )
    test_parser.add_argument("--data", required=True, help="CSV file path or built-in dataset name")
    test_parser.add_argument("--column", help="Column name for univariate extraction")
    test_parser.add_argument("--lags", help="Number of lags for test")
    test_parser.add_argument(
        "--format", default="text", choices=["text", "json", "csv"], help="Output format"
    )
    test_parser.set_defaults(func=cmd_test)

    # ── forecast ──
    fc_parser = subparsers.add_parser("forecast", help="Generate forecasts")
    fc_parser.add_argument("--model", required=True, help="Model type: arima, auto_arima, var")
    fc_parser.add_argument("--data", required=True, help="CSV file path or built-in dataset name")
    fc_parser.add_argument("--column", help="Column name for univariate extraction")
    fc_parser.add_argument("--steps", default="12", help="Forecast horizon")
    fc_parser.add_argument("--order", help="ARIMA order, e.g. '0,1,1'")
    fc_parser.add_argument("--seasonal", help="Seasonal order, e.g. '0,1,1,12'")
    fc_parser.add_argument("--m", help="Seasonal period for auto_arima")
    fc_parser.add_argument("--maxlags", help="Maximum lags for VAR")
    fc_parser.add_argument("--output", help="Output file path")
    fc_parser.add_argument(
        "--format", default="text", choices=["text", "json", "csv"], help="Output format"
    )
    fc_parser.set_defaults(func=cmd_forecast)

    # ── decompose ──
    dec_parser = subparsers.add_parser("decompose", help="Decompose a time series")
    dec_parser.add_argument("--method", required=True, help="Method: stl, classical")
    dec_parser.add_argument("--data", required=True, help="CSV file path or built-in dataset name")
    dec_parser.add_argument("--column", help="Column name for univariate extraction")
    dec_parser.add_argument("--period", help="Seasonal period")
    dec_parser.add_argument(
        "--type",
        default="additive",
        choices=["additive", "multiplicative"],
        help="Decomposition type (classical only)",
    )
    dec_parser.add_argument("--plot", action="store_true", help="Show plot")
    dec_parser.add_argument("--output", help="Output file path")
    dec_parser.add_argument(
        "--format", default="text", choices=["text", "json", "csv"], help="Output format"
    )
    dec_parser.set_defaults(func=cmd_decompose)

    # ── filter ──
    flt_parser = subparsers.add_parser("filter", help="Apply time series filters")
    flt_parser.add_argument("filter_type", help="Filter type: hp, bk, cf, hamilton")
    flt_parser.add_argument("--data", required=True, help="CSV file path or built-in dataset name")
    flt_parser.add_argument("--column", help="Column name for univariate extraction")
    flt_parser.add_argument("--lamb", help="Lambda for HP filter (default: 1600)")
    flt_parser.add_argument("--low", help="Lower bound for BK/CF (default: 6)")
    flt_parser.add_argument("--high", help="Upper bound for BK/CF (default: 32)")
    flt_parser.add_argument("--K", help="K parameter for BK filter (default: 12)")
    flt_parser.add_argument("--h", help="Horizon for Hamilton filter (default: 8)")
    flt_parser.add_argument("--p", help="Lags for Hamilton filter (default: 4)")
    flt_parser.add_argument("--output", help="Output file path")
    flt_parser.add_argument(
        "--format", default="text", choices=["text", "json", "csv"], help="Output format"
    )
    flt_parser.set_defaults(func=cmd_filter)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the ChronoBox CLI.

    Parameters
    ----------
    argv : list[str] | None
        Command line arguments. If None, uses sys.argv.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"ERROR: Column or dataset not found: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"ERROR: Required module not available: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
