"""
PSS Bounds test critical values for ARDL cointegration testing.

Provides F-test and t-test critical values (lower/upper bounds) from
Pesaran, Shin & Smith (2001) for the ARDL bounds testing approach.

References
----------
- Pesaran, M.H., Shin, Y. & Smith, R.J. (2001). Bounds testing approaches
  to the analysis of level relationships. Journal of Applied Econometrics,
  16(3), 289-326.
"""

from __future__ import annotations

# -----------------------------------------------------------------------
# F-test critical values (lower, upper) bounds
# Keys: (k, case, significance_level)
# k = number of regressors (excluding the dependent variable)
# case: 3 = unrestricted intercept, no trend (most common)
# case: 5 = unrestricted intercept and trend
# Values: (lower_bound, upper_bound)
# -----------------------------------------------------------------------
_F_BOUNDS: dict[tuple[int, int, str], tuple[float, float]] = {
    # Case 3: unrestricted intercept, no trend
    # k=1
    (1, 3, "10%"): (3.02, 3.51),
    (1, 3, "5%"): (3.62, 4.16),
    (1, 3, "2.5%"): (4.18, 4.79),
    (1, 3, "1%"): (4.94, 5.58),
    # k=2
    (2, 3, "10%"): (2.63, 3.35),
    (2, 3, "5%"): (3.10, 3.87),
    (2, 3, "2.5%"): (3.55, 4.38),
    (2, 3, "1%"): (4.13, 5.00),
    # k=3
    (3, 3, "10%"): (2.44, 3.28),
    (3, 3, "5%"): (2.87, 3.78),
    (3, 3, "2.5%"): (3.25, 4.26),
    (3, 3, "1%"): (3.77, 4.92),
    # k=4
    (4, 3, "10%"): (2.32, 3.25),
    (4, 3, "5%"): (2.69, 3.73),
    (4, 3, "2.5%"): (3.05, 4.18),
    (4, 3, "1%"): (3.49, 4.73),
    # k=5
    (5, 3, "10%"): (2.22, 3.23),
    (5, 3, "5%"): (2.56, 3.68),
    (5, 3, "2.5%"): (2.89, 4.12),
    (5, 3, "1%"): (3.34, 4.68),
    # k=6
    (6, 3, "10%"): (2.17, 3.21),
    (6, 3, "5%"): (2.45, 3.61),
    (6, 3, "2.5%"): (2.75, 4.06),
    (6, 3, "1%"): (3.15, 4.61),
    # k=7
    (7, 3, "10%"): (2.11, 3.18),
    (7, 3, "5%"): (2.39, 3.60),
    (7, 3, "2.5%"): (2.67, 4.01),
    (7, 3, "1%"): (3.03, 4.54),
    # k=8
    (8, 3, "10%"): (2.03, 3.13),
    (8, 3, "5%"): (2.32, 3.58),
    (8, 3, "2.5%"): (2.59, 3.99),
    (8, 3, "1%"): (2.96, 4.50),
    # k=9
    (9, 3, "10%"): (1.98, 3.09),
    (9, 3, "5%"): (2.26, 3.53),
    (9, 3, "2.5%"): (2.53, 3.96),
    (9, 3, "1%"): (2.88, 4.44),
    # k=10
    (10, 3, "10%"): (1.95, 3.06),
    (10, 3, "5%"): (2.20, 3.48),
    (10, 3, "2.5%"): (2.47, 3.91),
    (10, 3, "1%"): (2.81, 4.44),
    # Case 5: unrestricted intercept and trend
    # k=1
    (1, 5, "10%"): (4.05, 4.49),
    (1, 5, "5%"): (4.68, 5.15),
    (1, 5, "2.5%"): (5.28, 5.84),
    (1, 5, "1%"): (6.10, 6.73),
    # k=2
    (2, 5, "10%"): (3.38, 4.02),
    (2, 5, "5%"): (3.88, 4.61),
    (2, 5, "2.5%"): (4.37, 5.16),
    (2, 5, "1%"): (4.99, 5.85),
    # k=3
    (3, 5, "10%"): (3.05, 3.83),
    (3, 5, "5%"): (3.47, 4.40),
    (3, 5, "2.5%"): (3.89, 4.93),
    (3, 5, "1%"): (4.40, 5.54),
    # k=4
    (4, 5, "10%"): (2.84, 3.72),
    (4, 5, "5%"): (3.23, 4.25),
    (4, 5, "2.5%"): (3.60, 4.75),
    (4, 5, "1%"): (4.06, 5.40),
    # k=5
    (5, 5, "10%"): (2.68, 3.66),
    (5, 5, "5%"): (3.05, 4.15),
    (5, 5, "2.5%"): (3.40, 4.62),
    (5, 5, "1%"): (3.80, 5.23),
}

# -----------------------------------------------------------------------
# t-test critical values (lower, upper) bounds
# Keys: (k, case, significance_level)
# Values: (lower_bound, upper_bound)
# -----------------------------------------------------------------------
_T_BOUNDS: dict[tuple[int, int, str], tuple[float, float]] = {
    # Case 3: unrestricted intercept, no trend
    (1, 3, "10%"): (-2.57, -2.91),
    (1, 3, "5%"): (-2.86, -3.22),
    (1, 3, "2.5%"): (-3.13, -3.50),
    (1, 3, "1%"): (-3.43, -3.82),
    (2, 3, "10%"): (-2.57, -3.21),
    (2, 3, "5%"): (-2.86, -3.53),
    (2, 3, "2.5%"): (-3.13, -3.80),
    (2, 3, "1%"): (-3.43, -4.10),
    (3, 3, "10%"): (-2.57, -3.42),
    (3, 3, "5%"): (-2.86, -3.78),
    (3, 3, "2.5%"): (-3.13, -4.05),
    (3, 3, "1%"): (-3.43, -4.37),
    (4, 3, "10%"): (-2.57, -3.59),
    (4, 3, "5%"): (-2.86, -3.99),
    (4, 3, "2.5%"): (-3.13, -4.26),
    (4, 3, "1%"): (-3.43, -4.60),
    (5, 3, "10%"): (-2.57, -3.74),
    (5, 3, "5%"): (-2.86, -4.16),
    (5, 3, "2.5%"): (-3.13, -4.44),
    (5, 3, "1%"): (-3.43, -4.79),
}


def pss_f_bounds(
    k: int,
    case: int = 3,
) -> dict[str, tuple[float, float]]:
    """Get PSS F-test critical value bounds.

    Parameters
    ----------
    k : int
        Number of regressors (excluding the dependent variable), 1-10.
    case : int
        Deterministic specification case (3 or 5). Default is 3.

    Returns
    -------
    dict[str, tuple[float, float]]
        Critical value bounds at each significance level.
        Keys are '1%', '2.5%', '5%', '10%'.
        Values are (lower_bound, upper_bound).

    Raises
    ------
    ValueError
        If k or case is not in the tables.
    """
    result: dict[str, tuple[float, float]] = {}
    for level in ["1%", "2.5%", "5%", "10%"]:
        key = (k, case, level)
        if key not in _F_BOUNDS:
            raise ValueError(
                f"No F-bounds for k={k}, case={case}. "
                f"Supported: k=1-10 for case 3, k=1-5 for case 5."
            )
        result[level] = _F_BOUNDS[key]
    return result


def pss_t_bounds(
    k: int,
    case: int = 3,
) -> dict[str, tuple[float, float]]:
    """Get PSS t-test critical value bounds.

    Parameters
    ----------
    k : int
        Number of regressors (excluding the dependent variable), 1-5.
    case : int
        Deterministic specification case (3 only currently).

    Returns
    -------
    dict[str, tuple[float, float]]
        Critical value bounds at each significance level.
        Keys are '1%', '2.5%', '5%', '10%'.
        Values are (lower_bound, upper_bound).

    Raises
    ------
    ValueError
        If k or case is not in the tables.
    """
    result: dict[str, tuple[float, float]] = {}
    for level in ["1%", "2.5%", "5%", "10%"]:
        key = (k, case, level)
        if key not in _T_BOUNDS:
            raise ValueError(
                f"No t-bounds for k={k}, case={case}. "
                f"Supported: k=1-5 for case 3."
            )
        result[level] = _T_BOUNDS[key]
    return result
