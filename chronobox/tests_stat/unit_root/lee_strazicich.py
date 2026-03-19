"""
Lee-Strazicich minimum LM unit root test with structural breaks.

LM-based unit root test that allows for 1 or 2 endogenous structural breaks.
Unlike the Zivot-Andrews test, this test is size-correct under the null
hypothesis (does not confuse breaks with unit roots).

    H0: unit root with breaks
    H1: trend-stationary with breaks

References
----------
- Lee, J. & Strazicich, M.C. (2003). Minimum Lagrange multiplier unit root
  test with two structural breaks. Review of Economics and Statistics, 85(4),
  1082-1089.
- Lee, J. & Strazicich, M.C. (2004). Minimum LM unit root test with one
  structural break. Manuscript, Appalachian State University.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from chronobox.tests_stat.base import TestResult

# Lee-Strazicich critical values
_LS_CV: dict[tuple[int, str], dict[str, float]] = {
    # (n_breaks, model)
    (1, "break"): {"1%": -4.239, "5%": -3.566, "10%": -3.211},
    (1, "crash"): {"1%": -5.11, "5%": -4.50, "10%": -4.21},
    (2, "break"): {"1%": -4.545, "5%": -3.842, "10%": -3.504},
    (2, "crash"): {"1%": -5.823, "5%": -5.286, "10%": -4.989},
}


def _ls_regression(
    y: NDArray[np.floating[Any]],
    break_dates: list[int],
    model: str,
    lags: int,
) -> tuple[float, float]:
    """Estimate Lee-Strazicich LM regression for given break dates.

    Parameters
    ----------
    y : ndarray, shape (T,)
        Time series.
    break_dates : list[int]
        Break date indices.
    model : str
        'break' (intercept shift) or 'crash' (intercept + trend shift).
    lags : int
        Number of lagged differences.

    Returns
    -------
    tau_lm : float
        LM t-statistic for phi.
    ssr : float
        Sum of squared residuals.
    """
    nobs = len(y)

    # Build Z_t (deterministic components under H1)
    z_cols: list[NDArray[np.floating[Any]]] = [
        np.ones(nobs),
        np.arange(1, nobs + 1, dtype=float),
    ]
    for tb in break_dates:
        du = np.zeros(nobs)
        du[tb:] = 1.0
        z_cols.append(du)
        if model == "crash":
            dt_dummy = np.zeros(nobs)
            dt_dummy[tb:] = np.arange(1, nobs - tb + 1, dtype=float)
            z_cols.append(dt_dummy)

    z_mat = np.column_stack(z_cols)

    # Estimate delta under H0 (unit root): use first differences
    # Delta y_t = delta' * Delta Z_t + error
    # Note: Delta Z has rank-deficient columns (constant diffs to 0),
    # so use lstsq for robust estimation
    dy = np.diff(y)
    dz = np.diff(z_mat, axis=0)
    delta_hat, _, _, _ = np.linalg.lstsq(dz, dy, rcond=None)

    # Compute S_t (detrended series under H0)
    # S_t = y_t - psi_x - delta' * Z_t, where psi_x = y_1 - delta' * Z_1
    psi_x = y[0] - float(z_mat[0] @ delta_hat)
    s_tilde = y - psi_x - z_mat @ delta_hat

    # LM regression: Delta y_t = delta' * Delta Z_t + phi * S_{t-1}
    #                + sum(psi_j * Delta S_{t-j}) + eps_t
    # Effective sample
    start = max(1, lags)

    # Build regression matrices
    dy_dep = dy[start:]
    n = len(dy_dep)

    regressors: list[NDArray[np.floating[Any]]] = []

    # Delta Z_t (only non-zero variance columns)
    dz_part = dz[start:]
    for j in range(dz_part.shape[1]):
        col = dz_part[:, j]
        if np.any(col != 0):
            regressors.append(col)

    # S_{t-1}
    s_lag = s_tilde[start : nobs - 1]
    phi_idx = len(regressors)
    regressors.append(s_lag)

    # Lagged Delta S
    ds = np.diff(s_tilde)
    for j in range(1, lags + 1):
        idx_start = start - j
        idx_end = nobs - 1 - j
        if idx_start >= 0 and idx_end >= start:
            lag_ds = ds[idx_start:idx_end]
            if len(lag_ds) == n:
                regressors.append(lag_ds)

    x_mat = np.column_stack(regressors)
    k = x_mat.shape[1]

    # OLS via lstsq for robustness
    beta, _, _, _ = np.linalg.lstsq(x_mat, dy_dep, rcond=None)
    resid = dy_dep - x_mat @ beta
    ssr = float(np.sum(resid**2))
    sigma2 = ssr / (n - k)

    # Compute standard errors via (X'X)^{-1}
    xtx = x_mat.T @ x_mat
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)
    var_beta = sigma2 * xtx_inv
    se = np.sqrt(np.abs(np.diag(var_beta)))

    phi = float(beta[phi_idx])
    tau_lm = phi / float(se[phi_idx]) if se[phi_idx] > 0 else np.inf

    return tau_lm, ssr


def lee_strazicich_test(
    y: NDArray[np.floating[Any]],
    model: str = "break",
    breaks: int = 1,
    trim: float = 0.15,
    maxlag: int | None = None,
) -> TestResult:
    """Lee-Strazicich minimum LM unit root test with structural breaks.

    Size-correct under H0 (does not confuse breaks with unit roots).

    Parameters
    ----------
    y : array_like, shape (T,)
        Time series data.
    model : str, default 'break'
        Break specification:
        - 'break': break in intercept only (Model A)
        - 'crash': break in intercept and trend (Model C)
    breaks : int, default 1
        Number of breaks: 1 or 2.
    trim : float, default 0.15
        Fraction to trim from each end.
    maxlag : int or None, default None
        Maximum lags. If None, uses int(12*(T/100)^{1/4}).

    Returns
    -------
    TestResult
        With additional_info containing 'break_dates' (list of break indices).
    """
    y = np.asarray(y, dtype=float).ravel()
    nobs = len(y)

    if nobs < 30:
        raise ValueError(
            f"Series too short: T={nobs}, need at least 30 observations."
        )

    if model not in ("break", "crash"):
        raise ValueError(
            f"Invalid model '{model}'. Choose from 'break', 'crash'."
        )

    if breaks not in (1, 2):
        raise ValueError(f"Invalid breaks={breaks}. Choose 1 or 2.")

    if maxlag is None:
        maxlag = int(12.0 * (nobs / 100.0) ** 0.25)
    maxlag = min(maxlag, nobs // 4)

    # Use a fixed lag for simplicity (select via AIC later)
    lag = min(maxlag, int(4.0 * (nobs / 100.0) ** 0.25))

    start_idx = int(np.ceil(trim * nobs))
    end_idx = int(np.floor((1.0 - trim) * nobs))

    min_tau = np.inf
    best_breaks: list[int] = [start_idx]

    if breaks == 1:
        for tb1 in range(start_idx, end_idx + 1):
            try:
                tau, _ = _ls_regression(y, [tb1], model, lag)
                if tau < min_tau:
                    min_tau = tau
                    best_breaks = [tb1]
            except (np.linalg.LinAlgError, ValueError):
                continue
    else:  # breaks == 2
        step = max(1, (end_idx - start_idx) // 50)  # Coarse grid for speed
        min_gap = int(trim * nobs)
        for tb1 in range(start_idx, end_idx - min_gap, step):
            for tb2 in range(tb1 + min_gap, end_idx + 1, step):
                try:
                    tau, _ = _ls_regression(y, [tb1, tb2], model, lag)
                    if tau < min_tau:
                        min_tau = tau
                        best_breaks = [tb1, tb2]
                except (np.linalg.LinAlgError, ValueError):
                    continue

        # Refine around best breaks
        if len(best_breaks) == 2:
            tb1_range = range(
                max(start_idx, best_breaks[0] - step),
                min(end_idx, best_breaks[0] + step) + 1,
            )
            tb2_range = range(
                max(start_idx, best_breaks[1] - step),
                min(end_idx, best_breaks[1] + step) + 1,
            )
            for tb1 in tb1_range:
                for tb2 in tb2_range:
                    if tb2 <= tb1 + min_gap:
                        continue
                    try:
                        tau, _ = _ls_regression(y, [tb1, tb2], model, lag)
                        if tau < min_tau:
                            min_tau = tau
                            best_breaks = [tb1, tb2]
                    except (np.linalg.LinAlgError, ValueError):
                        continue

    cv_key = (breaks, model)
    cv = _LS_CV.get(cv_key, {"1%": -5.0, "5%": -4.5, "10%": -4.0})

    reject = min_tau < cv["5%"]

    return TestResult(
        test_name="Lee-Strazicich",
        statistic=min_tau,
        pvalue=None,
        critical_values=cv,
        null_hypothesis=f"Unit root with {breaks} break(s)",
        alternative_hypothesis=f"Trend-stationary with {breaks} break(s)",
        reject_at_5pct=reject,
        lags_used=lag,
        additional_info={
            "break_dates": best_breaks,
            "break_fractions": [b / nobs for b in best_breaks],
            "model": model,
            "n_breaks": breaks,
            "nobs": nobs,
        },
    )
