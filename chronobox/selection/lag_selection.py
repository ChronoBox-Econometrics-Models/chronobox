"""Lag order selection for VAR models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats


@dataclass
class LagOrderResults:
    """Results from VAR lag order selection.

    Attributes
    ----------
    aic : dict[int, float]
        AIC values for each lag order.
    bic : dict[int, float]
        BIC values for each lag order.
    hqic : dict[int, float]
        HQIC values for each lag order.
    fpe : dict[int, float]
        FPE values for each lag order.
    lr_stats : dict[int, float]
        Sequential LR test statistics (lag p vs p-1).
    lr_pvalues : dict[int, float]
        P-values for sequential LR tests.
    selected_orders : dict[str, int]
        Selected lag order by each criterion.
    """

    aic: dict[int, float] = field(default_factory=dict)
    bic: dict[int, float] = field(default_factory=dict)
    hqic: dict[int, float] = field(default_factory=dict)
    fpe: dict[int, float] = field(default_factory=dict)
    lr_stats: dict[int, float] = field(default_factory=dict)
    lr_pvalues: dict[int, float] = field(default_factory=dict)
    selected_orders: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """Return formatted summary table of lag selection criteria.

        Returns
        -------
        str
            Formatted table with all criteria values and selected orders.
        """
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("VAR Lag Order Selection")
        lines.append("=" * 72)
        lines.append(
            f"{'Lag':>4s}  {'AIC':>12s}  {'BIC':>12s}  {'HQIC':>12s}  {'FPE':>14s}"
        )
        lines.append("-" * 72)

        all_lags = sorted(self.aic.keys())
        for lag in all_lags:
            aic_val = self.aic.get(lag, float("nan"))
            bic_val = self.bic.get(lag, float("nan"))
            hqic_val = self.hqic.get(lag, float("nan"))
            fpe_val = self.fpe.get(lag, float("nan"))

            # Mark selected orders with asterisk
            aic_mark = "*" if self.selected_orders.get("aic") == lag else " "
            bic_mark = "*" if self.selected_orders.get("bic") == lag else " "
            hqic_mark = "*" if self.selected_orders.get("hqic") == lag else " "
            fpe_mark = "*" if self.selected_orders.get("fpe") == lag else " "

            lines.append(
                f"{lag:4d}  {aic_val:11.4f}{aic_mark} {bic_val:11.4f}{bic_mark}"
                f" {hqic_val:11.4f}{hqic_mark} {fpe_val:13.6e}{fpe_mark}"
            )

        lines.append("-" * 72)
        lines.append("* indicates lag order selected by the criterion")
        lines.append("")

        for criterion, order in sorted(self.selected_orders.items()):
            lines.append(f"  {criterion.upper():>5s} selects lag order {order}")

        lines.append("=" * 72)
        return "\n".join(lines)


def _estimate_var_ols(
    endog: NDArray[np.float64],
    lags: int,
    trend: str = "c",
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Estimate VAR(p) by OLS and return residual covariance.

    Parameters
    ----------
    endog : ndarray of shape (T, K)
        Multivariate time series data.
    lags : int
        Number of lags.
    trend : str
        Trend specification: 'n' (none), 'c' (constant), 'ct' (constant+trend).

    Returns
    -------
    sigma_u : ndarray of shape (K, K)
        Residual covariance matrix.
    resid : ndarray of shape (T', K)
        Residual matrix.
    nobs_eff : int
        Effective number of observations (T - lags).
    """
    t_total, k = endog.shape
    t_eff = t_total - lags

    # Build Y matrix: (T', K) -> each row is Y_t for t = lags, ..., T-1
    y_mat = endog[lags:]  # (T', K)

    # Build Z matrix: lagged values + deterministic terms
    z_parts: list[NDArray[np.float64]] = []

    # Lagged endogenous variables
    for lag_i in range(1, lags + 1):
        z_parts.append(endog[lags - lag_i : t_total - lag_i])  # (T', K)

    # Deterministic terms
    if trend == "c" or trend == "ct":
        z_parts.append(np.ones((t_eff, 1), dtype=np.float64))
    if trend == "ct":
        z_parts.append(
            np.arange(1, t_eff + 1, dtype=np.float64).reshape(-1, 1)
        )

    z_mat = np.column_stack(z_parts)  # (T', Kp + d)

    # OLS: B_hat = (Z'Z)^{-1} Z'Y
    # b_hat shape: (Kp + d, K)
    b_hat, _, _, _ = np.linalg.lstsq(z_mat, y_mat, rcond=None)

    # Residuals
    resid = y_mat - z_mat @ b_hat  # (T', K)

    # Degrees of freedom adjustment
    d = 0
    if trend == "c":
        d = 1
    elif trend == "ct":
        d = 2

    df = t_eff - k * lags - d
    if df <= 0:
        df = 1  # avoid division by zero for overfitted models

    sigma_u = (resid.T @ resid) / df  # (K, K)

    return sigma_u, resid, t_eff


def select_lag_order(
    endog: NDArray[np.float64],
    maxlags: int = 15,
    trend: str = "c",
) -> LagOrderResults:
    """Select VAR lag order using information criteria.

    Estimates VAR(p) for p = 0, 1, ..., maxlags and computes AIC, BIC,
    HQIC, FPE, and sequential LR test statistics.

    Parameters
    ----------
    endog : ndarray of shape (T, K)
        Multivariate time series data. Each column is a variable.
    maxlags : int, default 15
        Maximum number of lags to consider.
    trend : str, default 'c'
        Trend specification: 'n' (none), 'c' (constant), 'ct' (constant+trend).

    Returns
    -------
    LagOrderResults
        Object containing all criteria values and selected orders.

    Raises
    ------
    ValueError
        If endog is not 2-D or has insufficient observations.

    Notes
    -----
    Information criteria for VAR(p) with K variables and T effective obs:

    - AIC(p)  = log|Sigma_p| + 2*p*K^2/T
    - BIC(p)  = log|Sigma_p| + p*K^2*log(T)/T
    - HQIC(p) = log|Sigma_p| + 2*p*K^2*log(log(T))/T
    - FPE(p)  = |Sigma_p| * ((T + K*p + d) / (T - K*p - d))^K

    Sequential LR test:
    - LR(p vs p-1) = (T - K*p - 1.5) * (log|Sigma_{p-1}| - log|Sigma_p|)
    - LR ~ chi2(K^2)
    """
    endog = np.asarray(endog, dtype=np.float64)
    if endog.ndim != 2:
        msg = f"endog must be 2-D array, got {endog.ndim}-D"
        raise ValueError(msg)

    t_total, k = endog.shape

    if t_total <= maxlags + k:
        msg = (
            f"Insufficient observations: T={t_total}, maxlags={maxlags}, K={k}. "
            f"Need T > maxlags + K."
        )
        raise ValueError(msg)

    # Deterministic term count
    d = 0
    if trend == "c":
        d = 1
    elif trend == "ct":
        d = 2

    results = LagOrderResults()
    prev_log_det: float | None = None

    for p in range(0, maxlags + 1):
        if p == 0:
            # VAR(0): just the mean
            if trend == "n":
                resid = endog.copy()
            else:
                z_parts: list[NDArray[np.float64]] = []
                if trend in ("c", "ct"):
                    z_parts.append(np.ones((t_total, 1), dtype=np.float64))
                if trend == "ct":
                    z_parts.append(
                        np.arange(1, t_total + 1, dtype=np.float64).reshape(-1, 1)
                    )
                z_mat = np.column_stack(z_parts)
                b_hat, _, _, _ = np.linalg.lstsq(z_mat, endog, rcond=None)
                resid = endog - z_mat @ b_hat

            t_eff = t_total
            sigma_u = (resid.T @ resid) / (t_eff - d)
        else:
            sigma_u, resid, t_eff = _estimate_var_ols(endog, p, trend)

        # Use ML-type sigma (divide by T, not T-Kp-d) for IC computation
        sigma_ml = (resid.T @ resid) / t_eff

        det_sigma = np.linalg.det(sigma_ml)
        if det_sigma <= 0:
            # If determinant is non-positive, skip this lag
            continue

        log_det = np.log(det_sigma)

        # Information criteria
        results.aic[p] = log_det + 2.0 * p * k * k / t_eff
        results.bic[p] = log_det + p * k * k * np.log(t_eff) / t_eff
        if np.log(t_eff) > 0:
            results.hqic[p] = (
                log_det + 2.0 * p * k * k * np.log(np.log(t_eff)) / t_eff
            )
        else:
            results.hqic[p] = log_det

        # FPE
        numer = (t_eff + k * p + d) ** k
        denom = max((t_eff - k * p - d), 1) ** k
        results.fpe[p] = det_sigma * (numer / denom)

        # Sequential LR test (p vs p-1)
        if prev_log_det is not None and p >= 1:
            lr_stat = (t_eff - k * p - 1.5) * (prev_log_det - log_det)
            lr_pval = 1.0 - scipy_stats.chi2.cdf(lr_stat, k * k)
            results.lr_stats[p] = lr_stat
            results.lr_pvalues[p] = lr_pval

        prev_log_det = log_det

    # Select best order for each criterion
    if results.aic:
        results.selected_orders["aic"] = min(results.aic, key=results.aic.get)  # type: ignore[arg-type]
    if results.bic:
        results.selected_orders["bic"] = min(results.bic, key=results.bic.get)  # type: ignore[arg-type]
    if results.hqic:
        results.selected_orders["hqic"] = min(results.hqic, key=results.hqic.get)  # type: ignore[arg-type]
    if results.fpe:
        results.selected_orders["fpe"] = min(results.fpe, key=results.fpe.get)  # type: ignore[arg-type]

    return results
