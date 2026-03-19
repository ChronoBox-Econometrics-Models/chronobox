"""VECM - Vector Error Correction Model via Johansen procedure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# ============================================================================
# Critical values for Johansen trace and max-eigenvalue tests
# From Osterwald-Lenum (1992), Tables 1 and 1*
# Indexed by (n_vars - rank) = number of common trends under H0
# Columns: 90%, 95%, 99%
# ============================================================================

# Trace test critical values for model with restricted constant (ci, model 2)
_TRACE_CRIT_CI: dict[int, NDArray[np.float64]] = {
    1: np.array([7.52, 9.24, 12.97]),
    2: np.array([17.85, 19.96, 24.60]),
    3: np.array([32.00, 34.91, 41.07]),
    4: np.array([49.65, 53.12, 60.16]),
    5: np.array([71.86, 76.07, 84.45]),
    6: np.array([97.18, 102.14, 111.01]),
    7: np.array([126.58, 131.70, 143.09]),
    8: np.array([159.48, 165.58, 177.20]),
    9: np.array([196.37, 202.92, 215.74]),
    10: np.array([236.54, 244.15, 257.68]),
}

# Max-eigenvalue critical values for model with restricted constant (ci, model 2)
_MAXEIG_CRIT_CI: dict[int, NDArray[np.float64]] = {
    1: np.array([7.52, 9.24, 12.97]),
    2: np.array([13.75, 15.67, 20.20]),
    3: np.array([19.77, 22.00, 26.81]),
    4: np.array([25.56, 28.14, 33.24]),
    5: np.array([31.66, 34.40, 39.79]),
    6: np.array([37.45, 40.30, 46.82]),
    7: np.array([43.25, 46.45, 52.31]),
    8: np.array([48.91, 52.00, 57.95]),
    9: np.array([54.35, 57.42, 63.71]),
    10: np.array([60.25, 63.57, 69.94]),
}

# Trace test critical values for model with unrestricted constant (co, model 3)
_TRACE_CRIT_CO: dict[int, NDArray[np.float64]] = {
    1: np.array([6.50, 8.18, 11.65]),
    2: np.array([15.66, 17.95, 23.52]),
    3: np.array([28.71, 31.52, 37.22]),
    4: np.array([45.23, 48.28, 55.43]),
    5: np.array([66.49, 70.60, 78.87]),
    6: np.array([91.11, 95.75, 104.96]),
    7: np.array([119.80, 124.24, 136.06]),
    8: np.array([152.32, 157.11, 168.92]),
    9: np.array([188.30, 193.54, 206.34]),
    10: np.array([228.29, 234.24, 247.18]),
}

# Max-eigenvalue critical values for model with unrestricted constant (co, model 3)
_MAXEIG_CRIT_CO: dict[int, NDArray[np.float64]] = {
    1: np.array([6.50, 8.18, 11.65]),
    2: np.array([12.91, 14.90, 19.19]),
    3: np.array([18.90, 21.07, 25.75]),
    4: np.array([24.78, 27.14, 32.14]),
    5: np.array([30.84, 33.32, 38.78]),
    6: np.array([36.25, 39.43, 44.59]),
    7: np.array([42.06, 44.91, 51.30]),
    8: np.array([48.43, 51.07, 57.07]),
    9: np.array([53.98, 57.00, 63.37]),
    10: np.array([59.62, 62.81, 68.83]),
}

# Trace test critical values for no constant, no trend (nc, model 1)
_TRACE_CRIT_NC: dict[int, NDArray[np.float64]] = {
    1: np.array([2.69, 3.76, 6.65]),
    2: np.array([13.33, 15.41, 20.04]),
    3: np.array([26.79, 29.68, 35.65]),
    4: np.array([43.95, 47.21, 54.46]),
    5: np.array([65.06, 68.52, 76.07]),
    6: np.array([89.48, 94.15, 103.18]),
    7: np.array([118.50, 124.24, 133.57]),
    8: np.array([151.38, 156.00, 168.36]),
    9: np.array([186.54, 192.89, 204.95]),
    10: np.array([226.34, 233.13, 246.27]),
}

_MAXEIG_CRIT_NC: dict[int, NDArray[np.float64]] = {
    1: np.array([2.69, 3.76, 6.65]),
    2: np.array([12.07, 14.07, 18.63]),
    3: np.array([18.63, 20.97, 25.52]),
    4: np.array([24.16, 27.07, 32.24]),
    5: np.array([30.69, 33.46, 38.77]),
    6: np.array([36.58, 39.37, 45.10]),
    7: np.array([42.45, 45.28, 51.57]),
    8: np.array([48.45, 51.42, 57.69]),
    9: np.array([53.89, 56.88, 63.18]),
    10: np.array([60.05, 62.52, 68.61]),
}

# Trace critical values for linear trend inside ECM (li, model 4)
_TRACE_CRIT_LI: dict[int, NDArray[np.float64]] = {
    1: np.array([10.49, 12.53, 16.31]),
    2: np.array([22.76, 25.32, 30.45]),
    3: np.array([39.06, 42.44, 48.45]),
    4: np.array([59.14, 62.99, 70.05]),
    5: np.array([83.20, 87.31, 96.58]),
    6: np.array([110.42, 114.90, 124.75]),
    7: np.array([141.01, 146.76, 155.36]),
    8: np.array([176.67, 182.51, 192.89]),
    9: np.array([215.17, 222.46, 233.14]),
    10: np.array([257.68, 264.22, 277.71]),
}

_MAXEIG_CRIT_LI: dict[int, NDArray[np.float64]] = {
    1: np.array([10.49, 12.53, 16.31]),
    2: np.array([16.85, 18.96, 23.65]),
    3: np.array([23.11, 25.54, 30.34]),
    4: np.array([28.83, 31.46, 36.65]),
    5: np.array([34.75, 37.52, 42.36]),
    6: np.array([40.91, 43.97, 48.94]),
    7: np.array([46.32, 49.42, 54.71]),
    8: np.array([52.16, 55.50, 61.57]),
    9: np.array([57.87, 61.29, 67.91]),
    10: np.array([63.18, 66.23, 72.82]),
}

# Trace critical values for unrestricted linear trend (lo, model 5)
_TRACE_CRIT_LO: dict[int, NDArray[np.float64]] = {
    1: np.array([9.52, 11.65, 15.69]),
    2: np.array([21.58, 24.08, 29.19]),
    3: np.array([37.03, 40.17, 46.60]),
    4: np.array([56.28, 60.06, 67.64]),
    5: np.array([79.32, 83.93, 92.71]),
    6: np.array([105.94, 111.01, 120.37]),
    7: np.array([136.61, 141.20, 153.63]),
    8: np.array([170.80, 175.77, 187.82]),
    9: np.array([208.97, 215.41, 228.22]),
    10: np.array([250.84, 257.95, 271.78]),
}

_MAXEIG_CRIT_LO: dict[int, NDArray[np.float64]] = {
    1: np.array([9.52, 11.65, 15.69]),
    2: np.array([15.59, 17.68, 22.26]),
    3: np.array([21.58, 23.78, 28.83]),
    4: np.array([27.62, 30.33, 34.80]),
    5: np.array([33.18, 36.41, 41.58]),
    6: np.array([39.08, 42.48, 47.54]),
    7: np.array([44.59, 47.46, 53.59]),
    8: np.array([50.74, 53.19, 59.24]),
    9: np.array([56.35, 59.33, 65.40]),
    10: np.array([61.76, 64.84, 71.01]),
}


def _get_crit_tables(
    deterministic: str,
) -> tuple[dict[int, NDArray[np.float64]], dict[int, NDArray[np.float64]]]:
    """Get the appropriate critical value tables for the deterministic model.

    Parameters
    ----------
    deterministic : str
        One of 'nc', 'ci', 'co', 'li', 'lo'.

    Returns
    -------
    tuple of (trace_crit_table, maxeig_crit_table)
    """
    tables = {
        "nc": (_TRACE_CRIT_NC, _MAXEIG_CRIT_NC),
        "ci": (_TRACE_CRIT_CI, _MAXEIG_CRIT_CI),
        "co": (_TRACE_CRIT_CO, _MAXEIG_CRIT_CO),
        "li": (_TRACE_CRIT_LI, _MAXEIG_CRIT_LI),
        "lo": (_TRACE_CRIT_LO, _MAXEIG_CRIT_LO),
    }
    if deterministic not in tables:
        msg = f"deterministic must be one of {list(tables.keys())}, got '{deterministic}'"
        raise ValueError(msg)
    return tables[deterministic]


@dataclass
class JohansenResults:
    """Results from the Johansen cointegration test.

    Attributes
    ----------
    eigenvalues : ndarray of shape (K,)
        Eigenvalues sorted in descending order.
    eigenvectors : ndarray of shape (K, K)
        Corresponding eigenvectors (columns are cointegrating vectors).
    trace_stat : ndarray of shape (K,)
        Trace test statistics for r = 0, 1, ..., K-1.
    max_eig_stat : ndarray of shape (K,)
        Max-eigenvalue statistics for r = 0, 1, ..., K-1.
    trace_crit : ndarray of shape (K, 3)
        Critical values (90%, 95%, 99%) for trace test.
    max_eig_crit : ndarray of shape (K, 3)
        Critical values (90%, 95%, 99%) for max-eigenvalue test.
    rank_trace : int
        Cointegration rank selected by trace test at 5%.
    rank_maxeig : int
        Cointegration rank selected by max-eigenvalue test at 5%.
    s00 : ndarray of shape (K, K)
        Moment matrix S_00.
    s01 : ndarray of shape (K, K)
        Moment matrix S_01.
    s10 : ndarray of shape (K, K)
        Moment matrix S_10.
    s11 : ndarray of shape (K, K)
        Moment matrix S_11.
    nobs : int
        Effective number of observations.
    deterministic : str
        Deterministic specification used.
    """

    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    trace_stat: NDArray[np.float64]
    max_eig_stat: NDArray[np.float64]
    trace_crit: NDArray[np.float64]
    max_eig_crit: NDArray[np.float64]
    rank_trace: int
    rank_maxeig: int
    s00: NDArray[np.float64]
    s01: NDArray[np.float64]
    s10: NDArray[np.float64]
    s11: NDArray[np.float64]
    nobs: int
    deterministic: str

    def summary(self) -> str:
        """Formatted summary of the Johansen test results."""
        k = len(self.eigenvalues)
        lines: list[str] = []
        lines.append("=" * 78)
        lines.append("  Johansen Cointegration Test")
        lines.append(f"  Deterministic: {self.deterministic}")
        lines.append(f"  Observations: {self.nobs}")
        lines.append("=" * 78)
        lines.append("")

        # Trace test
        lines.append("  Trace Test")
        lines.append("-" * 78)
        lines.append(
            f"  {'H0: r<=':<10s} {'Eigenvalue':>12s} {'Trace Stat':>12s}"
            f" {'90% CV':>10s} {'95% CV':>10s} {'99% CV':>10s}"
        )
        lines.append("  " + "-" * 74)
        for r in range(k):
            sig_mark = " **" if self.trace_stat[r] > self.trace_crit[r, 1] else ""
            lines.append(
                f"  {r:<10d} {self.eigenvalues[r]:12.4f} {self.trace_stat[r]:12.4f}"
                f" {self.trace_crit[r, 0]:10.2f} {self.trace_crit[r, 1]:10.2f}"
                f" {self.trace_crit[r, 2]:10.2f}{sig_mark}"
            )
        lines.append(f"  Selected rank (trace, 5%): {self.rank_trace}")
        lines.append("")

        # Max-eigenvalue test
        lines.append("  Max-Eigenvalue Test")
        lines.append("-" * 78)
        lines.append(
            f"  {'H0: r=':<10s} {'Eigenvalue':>12s} {'Max-Eig Stat':>12s}"
            f" {'90% CV':>10s} {'95% CV':>10s} {'99% CV':>10s}"
        )
        lines.append("  " + "-" * 74)
        for r in range(k):
            sig_mark = " **" if self.max_eig_stat[r] > self.max_eig_crit[r, 1] else ""
            lines.append(
                f"  {r:<10d} {self.eigenvalues[r]:12.4f} {self.max_eig_stat[r]:12.4f}"
                f" {self.max_eig_crit[r, 0]:10.2f} {self.max_eig_crit[r, 1]:10.2f}"
                f" {self.max_eig_crit[r, 2]:10.2f}{sig_mark}"
            )
        lines.append(f"  Selected rank (max-eig, 5%): {self.rank_maxeig}")
        lines.append("")

        lines.append("  ** denotes rejection at 5% significance level")
        lines.append("=" * 78)
        return "\n".join(lines)


class VECMResults:
    """Results from fitting a VECM model.

    Attributes
    ----------
    alpha : ndarray of shape (K, r)
        Loading matrix (adjustment speeds).
    beta : ndarray of shape (K, r) or (K+d_ci, r)
        Cointegrating vectors (may include restricted deterministic terms).
    gamma : list of ndarray, each (K, K)
        Short-run coefficient matrices Gamma_1, ..., Gamma_{p-1}.
    det_coefs : ndarray or None
        Unrestricted deterministic coefficients.
    sigma_u : ndarray of shape (K, K)
        Residual covariance matrix.
    resid : ndarray of shape (T', K)
        Residuals.
    endog : ndarray of shape (T, K)
        Original endogenous data.
    coint_rank : int
        Cointegration rank r.
    eigenvalues : ndarray of shape (K,)
        Johansen eigenvalues.
    trace_stat : ndarray of shape (K,)
        Trace test statistics.
    max_eig_stat : ndarray of shape (K,)
        Max-eigenvalue statistics.
    trace_crit : ndarray of shape (K, 3)
        Trace critical values.
    nobs : int
        Effective observations.
    k_ar : int
        VAR lag order (in levels).
    neqs : int
        Number of equations K.
    names : list[str]
        Variable names.
    deterministic : str
        Deterministic specification.
    pi : ndarray of shape (K, K)
        Long-run matrix Pi = alpha @ beta'.
    """

    def __init__(
        self,
        alpha: NDArray[np.float64],
        beta: NDArray[np.float64],
        gamma: list[NDArray[np.float64]],
        det_coefs: NDArray[np.float64] | None,
        sigma_u: NDArray[np.float64],
        resid: NDArray[np.float64],
        endog: NDArray[np.float64],
        coint_rank: int,
        eigenvalues: NDArray[np.float64],
        trace_stat: NDArray[np.float64],
        max_eig_stat: NDArray[np.float64],
        trace_crit: NDArray[np.float64],
        nobs: int,
        k_ar: int,
        neqs: int,
        names: list[str],
        deterministic: str,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.det_coefs = det_coefs
        self.sigma_u = sigma_u
        self.resid = resid
        self.endog = endog
        self.coint_rank = coint_rank
        self.eigenvalues = eigenvalues
        self.trace_stat = trace_stat
        self.max_eig_stat = max_eig_stat
        self.trace_crit = trace_crit
        self.nobs = nobs
        self.k_ar = k_ar
        self.neqs = neqs
        self.names = names
        self.deterministic = deterministic

        # Compute Pi = alpha @ beta' (using first K rows of beta if extended)
        beta_k = self.beta[: self.neqs, :]
        self.pi = self.alpha @ beta_k.T

    def summary(self) -> str:
        """Formatted summary of the VECM results."""
        lines: list[str] = []
        lines.append("=" * 78)
        lines.append("  VECM Estimation Results (Johansen)")
        lines.append("=" * 78)
        lines.append(f"  No. of equations:      {self.neqs}")
        lines.append(f"  Lags in levels (VAR):  {self.k_ar}")
        lines.append(f"  Lags in differences:   {self.k_ar - 1}")
        lines.append(f"  Cointegration rank:    {self.coint_rank}")
        lines.append(f"  Deterministic:         {self.deterministic}")
        lines.append(f"  Observations (used):   {self.nobs}")
        lines.append("")

        # Alpha (loading matrix)
        lines.append("-" * 78)
        lines.append("  Alpha (Loading Matrix)")
        lines.append("-" * 78)
        header = "  " + "".join(
            f"{'EC' + str(j + 1):>12s}" for j in range(self.coint_rank)
        )
        lines.append(header)
        for i in range(self.neqs):
            row = f"  {self.names[i]:<10s}" + "".join(
                f"{self.alpha[i, j]:12.6f}" for j in range(self.coint_rank)
            )
            lines.append(row)
        lines.append("")

        # Beta (cointegrating vectors)
        lines.append("-" * 78)
        lines.append("  Beta (Cointegrating Vectors)")
        lines.append("-" * 78)
        header = "  " + "".join(
            f"{'CV' + str(j + 1):>12s}" for j in range(self.coint_rank)
        )
        lines.append(header)
        for i in range(self.beta.shape[0]):
            label = self.names[i] if i < self.neqs else f"det_{i - self.neqs + 1}"
            row = f"  {label:<10s}" + "".join(
                f"{self.beta[i, j]:12.6f}" for j in range(self.coint_rank)
            )
            lines.append(row)
        lines.append("")

        # Gamma (short-run coefficients)
        for g_idx, g_mat in enumerate(self.gamma):
            lines.append("-" * 78)
            lines.append(
                f"  Gamma_{g_idx + 1} (Short-run, lag {g_idx + 1} of differences)"
            )
            lines.append("-" * 78)
            header = "  " + "".join(f"{'D.' + n:>12s}" for n in self.names)
            lines.append(header)
            for i in range(self.neqs):
                row = f"  {self.names[i]:<10s}" + "".join(
                    f"{g_mat[i, j]:12.6f}" for j in range(self.neqs)
                )
                lines.append(row)
            lines.append("")

        # Pi matrix
        lines.append("-" * 78)
        lines.append("  Pi = alpha @ beta' (Long-run Matrix)")
        lines.append("-" * 78)
        header = "  " + "".join(f"{n:>12s}" for n in self.names)
        lines.append(header)
        for i in range(self.neqs):
            row = f"  {self.names[i]:<10s}" + "".join(
                f"{self.pi[i, j]:12.6f}" for j in range(self.neqs)
            )
            lines.append(row)
        lines.append("")

        # Eigenvalues
        lines.append("-" * 78)
        lines.append("  Johansen Eigenvalues")
        lines.append("-" * 78)
        for i, ev in enumerate(self.eigenvalues):
            lines.append(f"  lambda_{i + 1} = {ev:.6f}")
        lines.append("")

        lines.append("=" * 78)
        return "\n".join(lines)


def _johansen_estimation(
    endog: NDArray[np.float64],
    lags: int,
    deterministic: str = "ci",
) -> tuple[
    NDArray[np.float64],  # eigenvalues
    NDArray[np.float64],  # eigenvectors
    NDArray[np.float64],  # s00
    NDArray[np.float64],  # s01
    NDArray[np.float64],  # s10
    NDArray[np.float64],  # s11
    NDArray[np.float64],  # r0 residuals
    NDArray[np.float64],  # r1 residuals
    int,  # nobs
]:
    """Perform the Johansen reduced rank regression.

    Parameters
    ----------
    endog : ndarray of shape (T, K)
        Multivariate time series in levels.
    lags : int
        Number of lags in levels (p). The VECM uses p-1 lags in differences.
    deterministic : str
        Deterministic specification: 'nc', 'ci', 'co', 'li', 'lo'.

    Returns
    -------
    tuple
        eigenvalues, eigenvectors, S00, S01, S10, S11, R0, R1, nobs
    """
    t_total, k = endog.shape
    p = lags

    # Compute differences
    dy = np.diff(endog, axis=0)  # (T-1, K)

    # Effective sample: t = p, ..., T-1 (in original indexing)
    # After differencing: t = p-1, ..., T-2 (in differenced indexing)
    t_eff = t_total - p

    # Dependent variable: Delta Y_t for t = p, ..., T-1
    dy_dep = dy[p - 1 :]  # (T_eff, K)

    # Y_{t-1} for t = p, ..., T-1
    y_lag1 = endog[p - 1 : t_total - 1]  # (T_eff, K)

    # Build X_t: lagged differences + deterministic terms
    x_parts: list[NDArray[np.float64]] = []

    # Lagged differences: Delta Y_{t-1}, ..., Delta Y_{t-p+1}
    for lag_i in range(1, p):
        x_parts.append(dy[p - 1 - lag_i : t_total - 1 - lag_i])

    # Deterministic terms depend on model
    if deterministic in ("co", "li", "lo"):
        # Unrestricted constant
        x_parts.append(np.ones((t_eff, 1), dtype=np.float64))
    if deterministic == "lo":
        # Unrestricted linear trend
        x_parts.append(np.arange(p, t_total, dtype=np.float64).reshape(-1, 1))

    # If model has restricted constant or trend, augment Y_{t-1}
    if deterministic == "ci":
        # Restricted constant: augment y_lag1 with column of 1s
        y_lag1 = np.column_stack([y_lag1, np.ones(t_eff, dtype=np.float64)])
    elif deterministic == "li":
        # Restricted linear trend: augment y_lag1 with trend
        trend = np.arange(p, t_total, dtype=np.float64)
        y_lag1 = np.column_stack([y_lag1, trend])

    # Regress dy_dep and y_lag1 on X (lagged differences + unrestricted deterministics)
    if x_parts:
        x_mat = np.column_stack(x_parts)  # (T_eff, m)

        # R0 = dy_dep - X * (X'X)^{-1} X' dy_dep
        xtx_inv = np.linalg.inv(x_mat.T @ x_mat)
        proj_x = x_mat @ xtx_inv @ x_mat.T

        r0 = dy_dep - proj_x @ dy_dep  # (T_eff, K)
        r1 = y_lag1 - proj_x @ y_lag1  # (T_eff, K or K+1)
    else:
        # No regressors in X
        r0 = dy_dep.copy()
        r1 = y_lag1.copy()

    # Moment matrices
    s00 = (r0.T @ r0) / t_eff  # (K, K)
    s01 = (r0.T @ r1) / t_eff  # (K, K+d_ci)
    s10 = s01.T  # (K+d_ci, K)
    s11 = (r1.T @ r1) / t_eff  # (K+d_ci, K+d_ci)

    # Solve generalized eigenvalue problem:
    # S_10 * S_00^{-1} * S_01 * v = lambda * S_11 * v
    s00_inv = np.linalg.inv(s00)
    mat = np.linalg.inv(s11) @ s10 @ s00_inv @ s01

    eigenvalues_raw, eigenvectors_raw = np.linalg.eig(mat)

    # Take real parts (eigenvalues should be real for symmetric problems)
    eigenvalues_real = np.real(eigenvalues_raw)
    eigenvectors_real = np.real(eigenvectors_raw)

    # Clamp eigenvalues to [0, 1)
    eigenvalues_real = np.clip(eigenvalues_real, 0.0, 1.0 - 1e-15)

    # Sort by descending eigenvalue
    idx = np.argsort(-eigenvalues_real)
    eigenvalues_sorted = eigenvalues_real[idx]
    eigenvectors_sorted = eigenvectors_real[:, idx]

    # Normalize eigenvectors: beta' * S_11 * beta = I
    for j in range(eigenvectors_sorted.shape[1]):
        v = eigenvectors_sorted[:, j]
        scale = v.T @ s11 @ v
        if scale > 0:
            eigenvectors_sorted[:, j] = v / np.sqrt(scale)

    return (
        eigenvalues_sorted,
        eigenvectors_sorted,
        s00,
        s01,
        s10,
        s11,
        r0,
        r1,
        t_eff,
    )


class VECM:
    """Vector Error Correction Model estimated via Johansen procedure.

    Parameters
    ----------
    lags : int, default 1
        Number of lags in levels (the VECM uses lags-1 in differences).
        Must be >= 1. If lags=1, the VECM has no lagged differences.
    coint_rank : int or None
        Cointegration rank r. If None, determined via the Johansen trace test.
    deterministic : str, default 'ci'
        Deterministic specification:
        - 'nc': no constant, no trend
        - 'ci': constant inside ECM (restricted)
        - 'co': constant outside ECM (unrestricted)
        - 'li': linear trend inside ECM + unrestricted constant
        - 'lo': linear trend outside ECM + unrestricted constant

    Examples
    --------
    >>> import numpy as np
    >>> from chronobox.models.vecm import VECM
    >>> data = np.random.randn(100, 3).cumsum(axis=0)  # I(1) processes
    >>> model = VECM(lags=2, coint_rank=1, deterministic='ci')
    >>> results = model.fit(data)
    >>> print(results.summary())
    """

    def __init__(
        self,
        lags: int = 1,
        coint_rank: int | None = None,
        deterministic: str = "ci",
    ) -> None:
        if lags < 1:
            msg = f"lags must be >= 1, got {lags}"
            raise ValueError(msg)
        if deterministic not in ("nc", "ci", "co", "li", "lo"):
            msg = (
                f"deterministic must be 'nc', 'ci', 'co', 'li', or 'lo',"
                f" got '{deterministic}'"
            )
            raise ValueError(msg)
        if coint_rank is not None and coint_rank < 1:
            msg = f"coint_rank must be >= 1, got {coint_rank}"
            raise ValueError(msg)

        self.lags = lags
        self.coint_rank = coint_rank
        self.deterministic = deterministic

    def johansen_test(
        self,
        endog: NDArray[np.float64] | pd.DataFrame,
    ) -> JohansenResults:
        """Perform the Johansen cointegration test.

        Parameters
        ----------
        endog : ndarray of shape (T, K) or DataFrame
            Multivariate time series in levels.

        Returns
        -------
        JohansenResults
            Test results including eigenvalues, test statistics, and critical values.
        """
        if isinstance(endog, pd.DataFrame):
            endog_arr = endog.to_numpy(dtype=np.float64)
        else:
            endog_arr = np.asarray(endog, dtype=np.float64)

        k_vars = endog_arr.shape[1]

        eigenvalues_all, eigenvectors, s00, s01, s10, s11, _r0, _r1, t_eff = (
            _johansen_estimation(endog_arr, self.lags, self.deterministic)
        )

        # Use only first K eigenvalues for test statistics
        # (augmented models like 'ci', 'li' produce K+1 eigenvalues)
        eigenvalues = eigenvalues_all[:k_vars]

        # Trace and max-eigenvalue statistics
        trace_stat = np.zeros(k_vars, dtype=np.float64)
        max_eig_stat = np.zeros(k_vars, dtype=np.float64)

        for r in range(k_vars):
            # Trace: -T * sum_{i=r+1}^{K} log(1 - lambda_i)
            trace_stat[r] = -t_eff * np.sum(np.log(1.0 - eigenvalues[r:]))

            # Max-eigenvalue: -T * log(1 - lambda_{r+1})
            max_eig_stat[r] = -t_eff * np.log(1.0 - eigenvalues[r])

        # Critical values
        trace_crit_table, maxeig_crit_table = _get_crit_tables(self.deterministic)

        trace_crit = np.zeros((k_vars, 3), dtype=np.float64)
        max_eig_crit = np.zeros((k_vars, 3), dtype=np.float64)

        for r in range(k_vars):
            n_common = k_vars - r  # number of common trends under H0
            if n_common in trace_crit_table:
                trace_crit[r] = trace_crit_table[n_common]
            if n_common in maxeig_crit_table:
                max_eig_crit[r] = maxeig_crit_table[n_common]

        # Select rank using trace test at 5% (column index 1)
        rank_trace = 0
        for r in range(k_vars):
            if trace_stat[r] > trace_crit[r, 1]:
                rank_trace = r + 1
            else:
                break

        # Select rank using max-eigenvalue test at 5%
        rank_maxeig = 0
        for r in range(k_vars):
            if max_eig_stat[r] > max_eig_crit[r, 1]:
                rank_maxeig = r + 1
            else:
                break

        return JohansenResults(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            trace_stat=trace_stat,
            max_eig_stat=max_eig_stat,
            trace_crit=trace_crit,
            max_eig_crit=max_eig_crit,
            rank_trace=rank_trace,
            rank_maxeig=rank_maxeig,
            s00=s00,
            s01=s01,
            s10=s10,
            s11=s11,
            nobs=t_eff,
            deterministic=self.deterministic,
        )

    def fit(
        self,
        endog: NDArray[np.float64] | pd.DataFrame,
        names: list[str] | None = None,
    ) -> VECMResults:
        """Fit the VECM by Johansen procedure.

        Parameters
        ----------
        endog : ndarray of shape (T, K) or DataFrame
            Multivariate time series in levels.
        names : list of str or None
            Variable names. Inferred from DataFrame if available.

        Returns
        -------
        VECMResults
            Fitted VECM results.
        """
        if isinstance(endog, pd.DataFrame):
            if names is None:
                names = list(endog.columns)
            endog_arr = endog.to_numpy(dtype=np.float64)
        else:
            endog_arr = np.asarray(endog, dtype=np.float64)

        t_total, k = endog_arr.shape

        if names is None:
            names = [f"y{i + 1}" for i in range(k)]

        # Run Johansen
        eigenvalues_all, eigenvectors, _s00, s01, _s10, s11, _r0, _r1, _t_eff = (
            _johansen_estimation(endog_arr, self.lags, self.deterministic)
        )
        # Only first K eigenvalues are meaningful
        eigenvalues = eigenvalues_all[:k]

        # Determine rank
        if self.coint_rank is not None:
            r = self.coint_rank
        else:
            # Use trace test at 5%
            johansen = self.johansen_test(endog_arr)
            r = johansen.rank_trace
            if r == 0:
                r = 1  # default to at least 1

        # Extract beta (first r eigenvectors) and alpha
        beta = eigenvectors[:, :r]  # (K+d_ci, r) or (K, r)

        # alpha = S_01 * beta * (beta' * S_11 * beta)^{-1}
        bsb = beta.T @ s11 @ beta  # (r, r)
        bsb_inv = np.linalg.inv(bsb)
        alpha = s01 @ beta @ bsb_inv  # (K, r)

        # Normalize beta: first variable coefficient = 1 for each vector
        for j in range(r):
            if abs(beta[0, j]) > 1e-12:
                scale = beta[0, j]
                beta[:, j] /= scale
                alpha[:, j] *= scale

        # Now estimate the full VECM to get gamma coefficients and residuals
        p = self.lags
        dy = np.diff(endog_arr, axis=0)
        t_eff_full = t_total - p

        # Dependent: Delta Y_t
        dy_dep = dy[p - 1 :]  # (T_eff, K)

        # ECM term: Y_{t-1} @ beta -> (T_eff, r)
        y_lag1 = endog_arr[p - 1 : t_total - 1]  # (T_eff, K)

        if self.deterministic == "ci":
            y_lag1_ext = np.column_stack(
                [y_lag1, np.ones(t_eff_full, dtype=np.float64)]
            )
        elif self.deterministic == "li":
            trend = np.arange(p, t_total, dtype=np.float64)
            y_lag1_ext = np.column_stack([y_lag1, trend])
        else:
            y_lag1_ext = y_lag1

        ecm_term = y_lag1_ext @ beta  # (T_eff, r)

        # Build regressor matrix for full VECM OLS
        z_parts: list[NDArray[np.float64]] = [ecm_term]

        # Lagged differences
        for lag_i in range(1, p):
            z_parts.append(dy[p - 1 - lag_i : t_total - 1 - lag_i])

        # Unrestricted deterministic terms
        if self.deterministic in ("co", "li", "lo"):
            z_parts.append(np.ones((t_eff_full, 1), dtype=np.float64))
        if self.deterministic == "lo":
            z_parts.append(
                np.arange(p, t_total, dtype=np.float64).reshape(-1, 1)
            )

        z_mat = np.column_stack(z_parts)

        # OLS
        b_hat, _, _, _ = np.linalg.lstsq(z_mat, dy_dep, rcond=None)

        # Extract coefficients
        idx = 0

        # Alpha from regression (overwrite Johansen alpha with OLS estimate)
        alpha_ols = b_hat[idx : idx + r].T  # (K, r)
        idx += r

        # Gamma matrices
        gamma: list[NDArray[np.float64]] = []
        for _lag_i in range(1, p):
            g = b_hat[idx : idx + k].T  # (K, K)
            gamma.append(g)
            idx += k

        # Deterministic coefficients
        det_coefs: NDArray[np.float64] | None = None
        remaining = b_hat.shape[0] - idx
        if remaining > 0:
            det_coefs = b_hat[idx:].T  # (K, d)

        # Residuals
        resid = dy_dep - z_mat @ b_hat  # (T_eff, K)
        sigma_u = (resid.T @ resid) / (t_eff_full - z_mat.shape[1])

        # Compute test statistics for the summary
        trace_stat_arr = np.zeros(k, dtype=np.float64)
        max_eig_stat_arr = np.zeros(k, dtype=np.float64)
        for ri in range(k):
            trace_stat_arr[ri] = -t_eff_full * np.sum(
                np.log(1.0 - eigenvalues[ri:])
            )
            max_eig_stat_arr[ri] = -t_eff_full * np.log(1.0 - eigenvalues[ri])

        trace_crit_table, _ = _get_crit_tables(self.deterministic)
        trace_crit = np.zeros((k, 3), dtype=np.float64)
        for ri in range(k):
            n_common = k - ri
            if n_common in trace_crit_table:
                trace_crit[ri] = trace_crit_table[n_common]

        return VECMResults(
            alpha=alpha_ols,
            beta=beta,
            gamma=gamma,
            det_coefs=det_coefs,
            sigma_u=sigma_u,
            resid=resid,
            endog=endog_arr,
            coint_rank=r,
            eigenvalues=eigenvalues,
            trace_stat=trace_stat_arr,
            max_eig_stat=max_eig_stat_arr,
            trace_crit=trace_crit,
            nobs=t_eff_full,
            k_ar=p,
            neqs=k,
            names=names,
            deterministic=self.deterministic,
        )
