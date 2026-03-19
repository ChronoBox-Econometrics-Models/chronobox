"""
Osterwald-Lenum critical values for Johansen cointegration tests.

Provides trace and max-eigenvalue critical values for the Johansen test
with different numbers of cointegrating relations and deterministic models.

References
----------
- Osterwald-Lenum, M. (1992). A note with quantiles of the asymptotic
  distribution of the maximum likelihood cointegration rank test statistics.
  Oxford Bulletin of Economics and Statistics, 54(3), 461-472.
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration
  vectors in Gaussian vector autoregressive models. Econometrica, 59(6), 1551-1580.
"""

from __future__ import annotations

# -----------------------------------------------------------------------
# Trace test critical values
# Keys: (n_vars - rank, model)
# model: 'ci' (case 2: intercept in CE, no trend), 'li' (case 3: linear trend)
# Values: (cv_10%, cv_5%, cv_1%)
# -----------------------------------------------------------------------
_TRACE_CV: dict[tuple[int, str], tuple[float, float, float]] = {
    # Case 2: restricted constant (intercept in CE only)
    (1, "ci"): (7.52, 9.16, 12.97),
    (2, "ci"): (17.85, 19.96, 24.60),
    (3, "ci"): (32.00, 34.91, 41.07),
    (4, "ci"): (49.65, 53.12, 60.16),
    (5, "ci"): (71.86, 76.07, 84.45),
    (6, "ci"): (97.18, 102.14, 111.01),
    (7, "ci"): (126.58, 131.70, 143.09),
    (8, "ci"): (159.48, 165.58, 177.20),
    (9, "ci"): (196.37, 202.92, 215.74),
    (10, "ci"): (236.21, 244.15, 257.68),
    (11, "ci"): (280.72, 288.77, 303.13),
    (12, "ci"): (329.37, 337.45, 353.27),
    # Case 3: unrestricted constant, no trend
    (1, "li"): (6.50, 8.18, 11.65),
    (2, "li"): (15.66, 17.95, 23.52),
    (3, "li"): (28.71, 31.52, 37.22),
    (4, "li"): (45.23, 48.28, 55.43),
    (5, "li"): (66.49, 70.60, 78.87),
    (6, "li"): (90.45, 95.75, 104.20),
    (7, "li"): (118.50, 124.24, 133.57),
    (8, "li"): (150.53, 156.00, 168.36),
    (9, "li"): (186.39, 192.89, 204.95),
    (10, "li"): (225.85, 233.13, 246.27),
    (11, "li"): (269.96, 277.71, 291.40),
    (12, "li"): (318.14, 326.53, 341.18),
}

# -----------------------------------------------------------------------
# Max-eigenvalue test critical values
# Keys: (n_vars - rank, model)
# Values: (cv_10%, cv_5%, cv_1%)
# -----------------------------------------------------------------------
_MAX_EIGEN_CV: dict[tuple[int, str], tuple[float, float, float]] = {
    # Case 2: restricted constant
    (1, "ci"): (7.52, 9.16, 12.97),
    (2, "ci"): (13.75, 15.67, 19.19),
    (3, "ci"): (19.77, 22.00, 25.75),
    (4, "ci"): (25.56, 28.14, 32.14),
    (5, "ci"): (31.66, 34.40, 38.78),
    (6, "ci"): (37.45, 40.30, 44.59),
    (7, "ci"): (43.25, 46.45, 51.30),
    (8, "ci"): (48.91, 52.00, 57.07),
    (9, "ci"): (54.35, 57.42, 63.37),
    (10, "ci"): (60.25, 63.57, 68.61),
    (11, "ci"): (66.02, 69.74, 74.36),
    (12, "ci"): (71.80, 75.33, 80.65),
    # Case 3: unrestricted constant
    (1, "li"): (6.50, 8.18, 11.65),
    (2, "li"): (12.91, 14.90, 19.19),
    (3, "li"): (18.90, 21.07, 25.75),
    (4, "li"): (24.78, 27.14, 32.14),
    (5, "li"): (30.84, 33.32, 38.78),
    (6, "li"): (36.25, 39.43, 44.59),
    (7, "li"): (42.06, 44.91, 51.30),
    (8, "li"): (47.97, 51.07, 57.07),
    (9, "li"): (53.12, 57.00, 63.37),
    (10, "li"): (59.33, 62.81, 68.61),
    (11, "li"): (65.14, 68.83, 74.36),
    (12, "li"): (70.80, 74.79, 80.65),
}


def johansen_trace_cv(
    n_vars_minus_rank: int,
    model: str = "li",
) -> dict[str, float]:
    """Get critical values for Johansen trace test.

    Parameters
    ----------
    n_vars_minus_rank : int
        Number of variables minus the hypothesized rank (K - r).
    model : str
        Deterministic model: 'ci' (restricted constant) or 'li' (linear intercept).

    Returns
    -------
    dict[str, float]
        Critical values at 1%, 5%, and 10%.

    Raises
    ------
    ValueError
        If the combination is not in the tables.
    """
    key = (n_vars_minus_rank, model)
    if key not in _TRACE_CV:
        raise ValueError(
            f"No trace critical values for n_vars-rank={n_vars_minus_rank}, "
            f"model='{model}'. Supported: 1-12 for models 'ci', 'li'."
        )
    cv_10, cv_5, cv_1 = _TRACE_CV[key]
    return {"1%": cv_1, "5%": cv_5, "10%": cv_10}


def johansen_max_eigen_cv(
    n_vars_minus_rank: int,
    model: str = "li",
) -> dict[str, float]:
    """Get critical values for Johansen max-eigenvalue test.

    Parameters
    ----------
    n_vars_minus_rank : int
        Number of variables minus the hypothesized rank (K - r).
    model : str
        Deterministic model: 'ci' (restricted constant) or 'li' (linear intercept).

    Returns
    -------
    dict[str, float]
        Critical values at 1%, 5%, and 10%.

    Raises
    ------
    ValueError
        If the combination is not in the tables.
    """
    key = (n_vars_minus_rank, model)
    if key not in _MAX_EIGEN_CV:
        raise ValueError(
            f"No max-eigenvalue critical values for n_vars-rank={n_vars_minus_rank}, "
            f"model='{model}'. Supported: 1-12 for models 'ci', 'li'."
        )
    cv_10, cv_5, cv_1 = _MAX_EIGEN_CV[key]
    return {"1%": cv_1, "5%": cv_5, "10%": cv_10}
