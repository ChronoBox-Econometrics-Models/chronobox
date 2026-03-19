"""
Quandt Likelihood Ratio (QLR) / sup-Wald test for structural break.

Tests for a structural break at an unknown date by computing the supremum
of the Wald statistic over all possible break dates in [trim, 1-trim].

    QLR = sup W(tau) over tau in [tau_0, 1-tau_0]

References
----------
- Andrews, D.W.K. (1993). Tests for parameter instability and structural
  change with unknown change point. Econometrica, 61(4), 821-856.
- Quandt, R.E. (1960). Tests of the hypothesis that a linear regression
  system obeys two separate regimes. JASA, 55(290), 324-330.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats  # type: ignore[reportMissingTypeStubs]

from chronobox.tests_stat.base import TestResult

# Andrews (1993) approximate critical values for sup-Wald
# Keys: (k, significance)
# k = number of restrictions being tested
_ANDREWS_CV: dict[tuple[int, str], float] = {
    (1, "10%"): 7.12, (1, "5%"): 8.68, (1, "1%"): 12.16,
    (2, "10%"): 5.00, (2, "5%"): 5.86, (2, "1%"): 7.78,
    (3, "10%"): 4.09, (3, "5%"): 4.71, (3, "1%"): 6.02,
    (4, "10%"): 3.59, (4, "5%"): 4.09, (4, "1%"): 5.12,
    (5, "10%"): 3.26, (5, "5%"): 3.66, (5, "1%"): 4.53,
    (6, "10%"): 3.02, (6, "5%"): 3.37, (6, "1%"): 4.12,
}


def qlr_test(
    y: NDArray[np.floating],
    x_mat: NDArray[np.floating],
    trim: float = 0.15,
) -> TestResult:
    """Quandt Likelihood Ratio (sup-Wald) test for structural break.

    Parameters
    ----------
    y : array_like, shape (T,)
        Dependent variable.
    x_mat : array_like, shape (T, k)
        Regressor matrix (should include constant if desired).
    trim : float, default 0.15
        Fraction to trim from each end when searching for break.

    Returns
    -------
    TestResult
        With additional_info containing:
        - 'break_date': index of the maximum Wald statistic
        - 'wald_sequence': array of Wald statistics for each tested tau
        - 'break_fraction': break date as fraction of sample
    """
    y = np.asarray(y, dtype=float).ravel()
    x_mat = np.asarray(x_mat, dtype=float)
    if x_mat.ndim == 1:
        x_mat = x_mat.reshape(-1, 1)

    n_obs, k = x_mat.shape

    start = int(np.ceil(trim * n_obs))
    end = int(np.floor((1.0 - trim) * n_obs))

    if start >= end:
        raise ValueError(f"Trim too large: no valid break points in [{start}, {end}].")

    # Full sample OLS
    beta_full = np.linalg.solve(x_mat.T @ x_mat, x_mat.T @ y)
    resid_full = y - x_mat @ beta_full
    ssr_full = float(np.sum(resid_full**2))

    wald_seq = np.zeros(n_obs)
    max_wald = -np.inf
    best_break = start

    for tb in range(start, end + 1):
        # Sub-sample 1
        y1, x1 = y[:tb], x_mat[:tb]
        # Sub-sample 2
        y2, x2 = y[tb:], x_mat[tb:]

        n1, n2 = len(y1), len(y2)
        if n1 < k + 1 or n2 < k + 1:
            continue

        try:
            beta1 = np.linalg.solve(x1.T @ x1, x1.T @ y1)
            resid1 = y1 - x1 @ beta1
            ssr1 = float(np.sum(resid1**2))

            beta2 = np.linalg.solve(x2.T @ x2, x2.T @ y2)
            resid2 = y2 - x2 @ beta2
            ssr2 = float(np.sum(resid2**2))

            # Wald statistic
            ssr_sum = ssr1 + ssr2
            if ssr_sum < 1e-15:
                continue

            w_stat = ((ssr_full - ssr_sum) / k) / (ssr_sum / (n_obs - 2 * k))
            wald_seq[tb] = w_stat

            if max_wald < w_stat:
                max_wald = w_stat
                best_break = tb
        except np.linalg.LinAlgError:
            continue

    # Critical values
    cv: dict[str, float] = {}
    for level in ("1%", "5%", "10%"):
        key = (k, level)
        if key in _ANDREWS_CV:
            cv[level] = _ANDREWS_CV[key]
        else:
            # Fallback: use F critical values
            pct = float(level.strip("%")) / 100.0
            cv[level] = float(sp_stats.f.ppf(1.0 - pct, k, n_obs - 2 * k))  # type: ignore[reportUnknownMemberType]

    reject = max_wald > cv.get("5%", np.inf)

    return TestResult(
        test_name="QLR (sup-Wald)",
        statistic=max_wald,
        pvalue=None,
        critical_values=cv,
        null_hypothesis="No structural break (constant coefficients)",
        alternative_hypothesis="Structural break at unknown date",
        reject_at_5pct=reject,
        additional_info={
            "break_date": best_break,
            "break_fraction": best_break / n_obs,
            "wald_sequence": wald_seq,
            "trim": trim,
            "nobs": n_obs,
            "k": k,
        },
    )
