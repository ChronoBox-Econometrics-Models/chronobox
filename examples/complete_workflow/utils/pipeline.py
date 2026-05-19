"""
Pipeline utilities for complete workflow examples.

Provides end-to-end pipelines for univariate and multivariate time series
analysis using chronobox, plus cross-validation reporting against R and Stata.
"""

import numpy as np
import pandas as pd
from typing import Any

import chronobox
from chronobox import tests_stat, filters


# ---------------------------------------------------------------------------
# Univariate pipeline
# ---------------------------------------------------------------------------

def univariate_pipeline(
    series: pd.Series,
    tests: list[str] | None = None,
    models: list[str] | None = None,
    forecast_horizon: int = 12,
    seasonal_period: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run a complete univariate time series analysis pipeline.

    Steps
    -----
    1. Stationarity / unit-root tests
    2. Decomposition (STL when seasonal, classical otherwise)
    3. Filters (HP, CF)
    4. Model estimation (ARIMA, ETS, Theta, ...)
    5. Out-of-sample forecasts

    Parameters
    ----------
    series : pd.Series
        Time series with a DatetimeIndex.
    tests : list[str], optional
        Statistical tests to run.  Defaults to ``["adf", "kpss", "pp"]``.
    models : list[str], optional
        Models to estimate.  Defaults to ``["auto_arima", "ets", "theta"]``.
    forecast_horizon : int
        Number of periods to forecast.
    seasonal_period : int, optional
        Seasonal period (e.g. 12 for monthly).  Auto-detected if *None*.
    verbose : bool
        If *True* print progress messages.

    Returns
    -------
    dict
        Keys: ``"tests"``, ``"decomposition"``, ``"filters"``,
        ``"models"``, ``"forecasts"``, ``"best_model"``.
    """
    if tests is None:
        tests = ["adf", "kpss", "pp"]
    if models is None:
        models = ["auto_arima", "ets", "theta"]
    if seasonal_period is None:
        seasonal_period = _detect_seasonal_period(series)

    results: dict[str, Any] = {}

    # --- 1. Statistical tests -------------------------------------------------
    if verbose:
        print("Step 1/5: Running stationarity tests...")
    results["tests"] = _run_unit_root_tests(series, tests)

    # --- 2. Decomposition -----------------------------------------------------
    if verbose:
        print("Step 2/5: Decomposing series...")
    results["decomposition"] = _run_decomposition(series, seasonal_period)

    # --- 3. Filters -----------------------------------------------------------
    if verbose:
        print("Step 3/5: Applying filters (HP, CF)...")
    results["filters"] = _run_filters(series)

    # --- 4. Model estimation --------------------------------------------------
    if verbose:
        print("Step 4/5: Estimating models...")
    train = series.iloc[:-forecast_horizon]
    test = series.iloc[-forecast_horizon:]
    results["models"] = _estimate_models(train, models, seasonal_period)

    # --- 5. Forecasts ---------------------------------------------------------
    if verbose:
        print("Step 5/5: Generating forecasts...")
    results["forecasts"] = _generate_forecasts(
        results["models"], forecast_horizon, test
    )
    results["best_model"] = _select_best_model(results["forecasts"])

    if verbose:
        print(f"Pipeline complete. Best model: {results['best_model']}")

    return results


# ---------------------------------------------------------------------------
# Multivariate pipeline
# ---------------------------------------------------------------------------

def multivariate_pipeline(
    data: pd.DataFrame,
    tests: list[str] | None = None,
    var_spec: dict[str, Any] | None = None,
    svar_spec: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run a complete multivariate time series analysis pipeline.

    Steps
    -----
    1. Unit-root tests per variable
    2. Cointegration tests (Engle-Granger, Johansen proxy via VECM)
    3. VAR / VECM estimation
    4. SVAR identification (optional)
    5. IRF, FEVD, forecasts

    Parameters
    ----------
    data : pd.DataFrame
        Multivariate time series with DatetimeIndex.
    tests : list[str], optional
        Tests to run per variable.  Defaults to ``["adf", "kpss"]``.
    var_spec : dict, optional
        VAR specification: ``{"maxlags": 8, "ic": "aic"}``.
    svar_spec : dict, optional
        SVAR specification: ``{"identification": "cholesky"}``.
        If *None*, SVAR step is skipped.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        Keys: ``"tests"``, ``"cointegration"``, ``"var"``,
        ``"vecm"``, ``"svar"``, ``"irf"``, ``"fevd"``, ``"forecasts"``.
    """
    if tests is None:
        tests = ["adf", "kpss"]
    if var_spec is None:
        var_spec = {"maxlags": 8, "ic": "aic"}

    results: dict[str, Any] = {}

    # --- 1. Unit-root tests ---------------------------------------------------
    if verbose:
        print("Step 1/5: Running unit-root tests per variable...")
    results["tests"] = {}
    for col in data.columns:
        results["tests"][col] = _run_unit_root_tests(data[col], tests)

    # --- 2. Cointegration tests -----------------------------------------------
    if verbose:
        print("Step 2/5: Testing for cointegration...")
    results["cointegration"] = _run_cointegration_tests(data)

    # --- 3. VAR / VECM estimation ---------------------------------------------
    if verbose:
        print("Step 3/5: Estimating VAR model...")
    results["var"] = _estimate_var(data, var_spec)

    coint_detected = results["cointegration"].get("engle_granger_reject", False)
    if coint_detected:
        if verbose:
            print("         Cointegration detected — also estimating VECM...")
        results["vecm"] = _estimate_vecm(data, var_spec)
    else:
        results["vecm"] = None

    # --- 4. SVAR (optional) ---------------------------------------------------
    if svar_spec is not None:
        if verbose:
            print("Step 4/5: Identifying SVAR...")
        results["svar"] = _estimate_svar(data, svar_spec)
    else:
        results["svar"] = None

    # --- 5. IRF, FEVD, forecasts ----------------------------------------------
    if verbose:
        print("Step 5/5: Computing IRF, FEVD, and forecasts...")
    var_model = results["var"]["model"]
    results["irf"] = _compute_irf(var_model)
    results["fevd"] = _compute_fevd(var_model)
    results["forecasts"] = _compute_var_forecasts(var_model, steps=12)

    if verbose:
        print("Multivariate pipeline complete.")

    return results


# ---------------------------------------------------------------------------
# Cross-validation report
# ---------------------------------------------------------------------------

def cross_validation_report(
    python_results: dict[str, Any],
    r_results: dict[str, Any] | None = None,
    stata_results: dict[str, Any] | None = None,
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """Compare results from Python (chronobox), R, and Stata.

    Parameters
    ----------
    python_results : dict
        Results dictionary with metric names as keys and numeric values.
        Example: ``{"aic": -345.2, "rmse": 0.032, "coef_ar1": 0.85}``.
    r_results : dict, optional
        Corresponding results from R.
    stata_results : dict, optional
        Corresponding results from Stata.
    tolerance : float
        Relative tolerance for flagging discrepancies (default 5 %).

    Returns
    -------
    pd.DataFrame
        Comparison table with columns ``metric``, ``python``, ``r``,
        ``stata``, ``max_rel_diff``, ``match``.
    """
    metrics = sorted(set(
        list(python_results.keys())
        + list((r_results or {}).keys())
        + list((stata_results or {}).keys())
    ))

    rows = []
    for m in metrics:
        py_val = python_results.get(m)
        r_val = (r_results or {}).get(m)
        st_val = (stata_results or {}).get(m)

        vals = [v for v in (py_val, r_val, st_val) if v is not None]
        if len(vals) >= 2:
            ref = np.mean(vals)
            if ref != 0:
                max_rel = max(abs(v - ref) / abs(ref) for v in vals)
            else:
                max_rel = max(abs(v) for v in vals)
            match = max_rel <= tolerance
        else:
            max_rel = np.nan
            match = True  # nothing to compare

        rows.append({
            "metric": m,
            "python": py_val,
            "r": r_val,
            "stata": st_val,
            "max_rel_diff": round(max_rel, 6) if not np.isnan(max_rel) else np.nan,
            "match": match,
        })

    return pd.DataFrame(rows)


# ===========================================================================
# Internal helpers
# ===========================================================================

_TEST_FUNCS = {
    "adf": tests_stat.adf_test,
    "kpss": tests_stat.kpss_test,
    "pp": tests_stat.pp_test,
    "ers": tests_stat.ers_test,
    "hegy": tests_stat.hegy_test,
}


def _detect_seasonal_period(series: pd.Series) -> int | None:
    """Infer seasonal period from DatetimeIndex frequency."""
    if not hasattr(series.index, "freqstr"):
        return None
    freq = getattr(series.index, "freqstr", "") or ""
    freq = freq.upper()
    if freq.startswith("M") or freq.startswith("MS"):
        return 12
    if freq.startswith("Q") or freq.startswith("QS"):
        return 4
    if freq.startswith("W"):
        return 52
    return None


def _run_unit_root_tests(
    series: pd.Series, test_names: list[str]
) -> dict[str, dict]:
    out = {}
    for name in test_names:
        func = _TEST_FUNCS.get(name)
        if func is None:
            out[name] = {"error": f"Unknown test: {name}"}
            continue
        try:
            result = func(series.dropna().values)
            out[name] = {
                "statistic": float(result.statistic),
                "p_value": float(result.p_value) if result.p_value is not None else None,
                "reject_h0": bool(result.reject_h0) if result.reject_h0 is not None else None,
            }
        except Exception as e:
            out[name] = {"error": str(e)}
    return out


def _run_decomposition(
    series: pd.Series, seasonal_period: int | None
) -> dict[str, Any]:
    try:
        if seasonal_period and seasonal_period > 1:
            dec = chronobox.STL(period=seasonal_period)
            res = dec.fit(series.values)
        else:
            dec = chronobox.ClassicalDecomposition(period=seasonal_period or 1)
            res = dec.fit(series.values)
        return {
            "method": "STL" if (seasonal_period and seasonal_period > 1) else "Classical",
            "trend": res.trend.tolist() if hasattr(res, "trend") else None,
            "seasonal": res.seasonal.tolist() if hasattr(res, "seasonal") else None,
            "residual": res.resid.tolist() if hasattr(res, "resid") else None,
        }
    except Exception as e:
        return {"error": str(e)}


def _run_filters(series: pd.Series) -> dict[str, Any]:
    out = {}
    y = series.dropna().values
    try:
        cycle, trend = filters.hp_filter(y)
        out["hp"] = {"cycle": cycle.tolist(), "trend": trend.tolist()}
    except Exception as e:
        out["hp"] = {"error": str(e)}
    try:
        cycle, trend = filters.cf_filter(y)
        out["cf"] = {"cycle": cycle.tolist(), "trend": trend.tolist()}
    except Exception as e:
        out["cf"] = {"error": str(e)}
    return out


def _estimate_models(
    train: pd.Series,
    model_names: list[str],
    seasonal_period: int | None,
) -> dict[str, Any]:
    fitted = {}
    for name in model_names:
        try:
            if name == "auto_arima":
                model = chronobox.auto_arima(
                    train.values,
                    seasonal=bool(seasonal_period),
                    m=seasonal_period or 1,
                )
                fitted[name] = {"model": model, "aic": float(model.aic())}
            elif name == "ets":
                model = chronobox.auto_ets(train.values, seasonal_period=seasonal_period or 1)
                fitted[name] = {"model": model, "aic": float(model.aic())}
            elif name == "theta":
                model = chronobox.ThetaMethod()
                model.fit(train.values)
                fitted[name] = {"model": model, "aic": None}
            elif name == "arima":
                model = chronobox.ARIMA(order=(1, 1, 1))
                model.fit(train.values)
                fitted[name] = {"model": model, "aic": float(model.aic())}
            else:
                fitted[name] = {"error": f"Unknown model: {name}"}
        except Exception as e:
            fitted[name] = {"error": str(e)}
    return fitted


def _generate_forecasts(
    models_dict: dict[str, Any],
    horizon: int,
    actual: pd.Series,
) -> dict[str, dict]:
    forecasts = {}
    for name, info in models_dict.items():
        if "error" in info:
            forecasts[name] = {"error": info["error"]}
            continue
        try:
            model = info["model"]
            fc = model.forecast(horizon)
            fc_vals = np.asarray(fc).flatten()[:horizon]
            actual_vals = actual.values[:horizon]
            rmse = float(np.sqrt(np.mean((fc_vals - actual_vals) ** 2)))
            mae = float(np.mean(np.abs(fc_vals - actual_vals)))
            forecasts[name] = {
                "values": fc_vals.tolist(),
                "rmse": rmse,
                "mae": mae,
            }
        except Exception as e:
            forecasts[name] = {"error": str(e)}
    return forecasts


def _select_best_model(forecasts: dict[str, dict]) -> str:
    best_name, best_rmse = None, np.inf
    for name, info in forecasts.items():
        if "rmse" in info and info["rmse"] < best_rmse:
            best_rmse = info["rmse"]
            best_name = name
    return best_name or "none"


def _run_cointegration_tests(data: pd.DataFrame) -> dict[str, Any]:
    out = {}
    cols = list(data.columns)
    if len(cols) >= 2:
        try:
            res = tests_stat.engle_granger_test(
                data[cols[0]].dropna().values,
                data[cols[1]].dropna().values,
            )
            out["engle_granger"] = {
                "statistic": float(res.statistic),
                "p_value": float(res.p_value) if res.p_value is not None else None,
                "reject_h0": bool(res.reject_h0) if res.reject_h0 is not None else None,
            }
            out["engle_granger_reject"] = bool(res.reject_h0) if res.reject_h0 is not None else False
        except Exception as e:
            out["engle_granger"] = {"error": str(e)}
            out["engle_granger_reject"] = False
    return out


def _estimate_var(data: pd.DataFrame, spec: dict) -> dict[str, Any]:
    try:
        maxlags = spec.get("maxlags", 8)
        ic = spec.get("ic", "aic")
        model = chronobox.VAR(data.values, maxlags=maxlags, ic=ic)
        model.fit()
        return {
            "model": model,
            "lags": model.k if hasattr(model, "k") else maxlags,
            "aic": float(model.aic()) if hasattr(model, "aic") else None,
        }
    except Exception as e:
        return {"error": str(e)}


def _estimate_vecm(data: pd.DataFrame, spec: dict) -> dict[str, Any]:
    try:
        maxlags = spec.get("maxlags", 8)
        model = chronobox.VECM(data.values, k_ar_diff=min(maxlags, 4), coint_rank=1)
        model.fit()
        return {"model": model}
    except Exception as e:
        return {"error": str(e)}


def _estimate_svar(data: pd.DataFrame, spec: dict) -> dict[str, Any]:
    try:
        ident = spec.get("identification", "cholesky")
        model = chronobox.SVAR(data.values, identification=ident)
        model.fit()
        return {"model": model, "identification": ident}
    except Exception as e:
        return {"error": str(e)}


def _compute_irf(var_model: Any, periods: int = 20) -> dict[str, Any]:
    try:
        irf = var_model.irf(periods=periods)
        return {"periods": periods, "irf": irf}
    except Exception as e:
        return {"error": str(e)}


def _compute_fevd(var_model: Any, periods: int = 20) -> dict[str, Any]:
    try:
        fevd = var_model.fevd(periods=periods)
        return {"periods": periods, "fevd": fevd}
    except Exception as e:
        return {"error": str(e)}


def _compute_var_forecasts(
    var_model: Any, steps: int = 12
) -> dict[str, Any]:
    try:
        fc = var_model.forecast(steps=steps)
        return {"steps": steps, "values": np.asarray(fc).tolist()}
    except Exception as e:
        return {"error": str(e)}
