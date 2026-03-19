"""Granger causality test for VAR models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

if TYPE_CHECKING:
    from chronobox.models.var import GrangerResult, VARResults


def granger_causality(
    var_results: VARResults,
    caused: str | int,
    causing: str | int,
    signif: float = 0.05,
) -> GrangerResult:
    """Test Granger causality in a fitted VAR model.

    Tests whether the variable 'causing' Granger-causes the variable 'caused'
    by testing whether all coefficients of the causing variable in the
    equation of the caused variable are jointly zero.

    H0: causing does NOT Granger-cause caused
    H1: causing Granger-causes caused

    Parameters
    ----------
    var_results : VARResults
        Fitted VAR model results.
    caused : str or int
        Name or index of the caused (dependent) variable.
    causing : str or int
        Name or index of the causing (independent) variable.
    signif : float, default 0.05
        Significance level for the test.

    Returns
    -------
    GrangerResult
        Test result with F-statistic, p-value, Wald statistic, and decision.

    Notes
    -----
    The test is performed using the F-test approach:

    1. From the VAR, extract the equation for 'caused'.
    2. The unrestricted model includes all K variables at lags 1..p.
    3. The restricted model excludes all lags of 'causing'.
    4. F = ((SSR_r - SSR_u) / p) / (SSR_u / (T' - Kp - d))

    The Wald test statistic is: W = T' * (SSR_r - SSR_u) / SSR_u ~ chi2(p)
    """
    from chronobox.models.var import GrangerResult

    # Resolve variable indices
    caused_idx = _resolve_var_index(var_results, caused)
    causing_idx = _resolve_var_index(var_results, causing)

    caused_name = var_results.names[caused_idx]
    causing_name = var_results.names[causing_idx]

    k = var_results.neqs
    p = var_results.k_ar
    t_eff = var_results.nobs
    t_total = var_results.nobs_total
    endog = var_results.endog

    # Determine deterministic term count
    d = 0
    if var_results.trend in ("c", "ct", "ctt"):
        d += 1
    if var_results.trend in ("ct", "ctt"):
        d += 1
    if var_results.trend == "ctt":
        d += 1

    # Build the dependent variable for the caused equation
    y_dep = endog[p:, caused_idx]  # (T', )

    # Build the full regressor matrix (same as VAR estimation)
    z_parts_full: list[NDArray[np.float64]] = []
    for lag_i in range(1, p + 1):
        z_parts_full.append(endog[p - lag_i : t_total - lag_i])  # (T', K)

    # Deterministic terms
    if var_results.trend in ("c", "ct", "ctt"):
        z_parts_full.append(np.ones((t_eff, 1), dtype=np.float64))
    if var_results.trend in ("ct", "ctt"):
        z_parts_full.append(
            np.arange(1, t_eff + 1, dtype=np.float64).reshape(-1, 1)
        )
    if var_results.trend == "ctt":
        z_parts_full.append(
            (np.arange(1, t_eff + 1, dtype=np.float64) ** 2).reshape(-1, 1)
        )

    z_full = np.column_stack(z_parts_full)  # (T', Kp + d)

    # Unrestricted OLS
    b_u, _, _, _ = np.linalg.lstsq(z_full, y_dep, rcond=None)
    resid_u = y_dep - z_full @ b_u
    ssr_u = float(resid_u @ resid_u)

    # Restricted: remove columns corresponding to the causing variable at all lags
    # In the Z matrix, lag_i contributes columns [(lag_i-1)*K : lag_i*K]
    # The causing variable is at column (lag_i-1)*K + causing_idx within each lag block
    cols_to_remove: list[int] = []
    for lag_i in range(p):
        col_idx = lag_i * k + causing_idx
        cols_to_remove.append(col_idx)

    cols_to_keep = [i for i in range(z_full.shape[1]) if i not in cols_to_remove]
    z_restricted = z_full[:, cols_to_keep]  # (T', Kp + d - p)

    # Restricted OLS
    b_r, _, _, _ = np.linalg.lstsq(z_restricted, y_dep, rcond=None)
    resid_r = y_dep - z_restricted @ b_r
    ssr_r = float(resid_r @ resid_r)

    # F-test
    df1 = p  # number of restrictions
    df2 = t_eff - k * p - d  # residual degrees of freedom

    if ssr_u > 0 and df2 > 0:
        f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
        f_pvalue = 1.0 - scipy_stats.f.cdf(f_stat, df1, df2)
    else:
        f_stat = 0.0
        f_pvalue = 1.0

    # Wald test
    if ssr_u > 0:
        wald_stat = t_eff * (ssr_r - ssr_u) / ssr_u
        wald_pvalue = 1.0 - scipy_stats.chi2.cdf(wald_stat, df1)
    else:
        wald_stat = 0.0
        wald_pvalue = 1.0

    reject = bool(f_pvalue < signif)

    return GrangerResult(
        fstat=float(f_stat),
        pvalue=float(f_pvalue),
        df=(df1, df2),
        reject=reject,
        wald_stat=float(wald_stat),
        wald_pvalue=float(wald_pvalue),
        caused=caused_name,
        causing=causing_name,
        signif=signif,
    )


def _resolve_var_index(var_results: VARResults, var: str | int) -> int:
    """Resolve a variable name or index to an integer index.

    Parameters
    ----------
    var_results : VARResults
        VAR results with names attribute.
    var : str or int
        Variable name or index.

    Returns
    -------
    int
        Variable index.
    """
    if isinstance(var, int):
        if var < 0 or var >= var_results.neqs:
            msg = f"Variable index {var} out of range [0, {var_results.neqs})"
            raise ValueError(msg)
        return var
    if isinstance(var, str):
        if var in var_results.names:
            return var_results.names.index(var)
        msg = f"Variable '{var}' not found. Available: {var_results.names}"
        raise ValueError(msg)
    msg = f"var must be str or int, got {type(var)}"
    raise TypeError(msg)
