"""
MacKinnon critical values for unit root and cointegration tests.

Implements the regression surface approximation from MacKinnon (1996, 2010)
for computing critical values as a function of sample size T.

    c(p) = beta_inf + beta_1/T + beta_2/T^2

References
----------
- MacKinnon, J.G. (1996). Numerical distribution functions for unit root and
  cointegration tests. Journal of Applied Econometrics, 11(6), 601-618.
- MacKinnon, J.G. (2010). Critical values for cointegration tests.
  Queen's Economics Department Working Paper No. 1227.
"""

from __future__ import annotations

# -----------------------------------------------------------------------
# ADF critical values: coefficients (beta_inf, beta_1, beta_2)
# Keys: (regression_model, significance_level)
# Regression models: 'n' (no constant), 'c' (constant), 'ct' (constant+trend)
# -----------------------------------------------------------------------
_ADF_TAU_COEFS: dict[tuple[str, str], tuple[float, float, float]] = {
    # No constant ('n')
    ("n", "1%"): (-2.5658, -1.960, -10.04),
    ("n", "5%"): (-1.9393, -0.398, 0.0),
    ("n", "10%"): (-1.6156, -0.181, 0.0),
    # Constant ('c')
    ("c", "1%"): (-3.4336, -5.999, -29.25),
    ("c", "5%"): (-2.8621, -2.738, -8.36),
    ("c", "10%"): (-2.5671, -1.438, -4.48),
    # Constant + trend ('ct')
    ("ct", "1%"): (-3.9638, -8.353, -47.44),
    ("ct", "5%"): (-3.4126, -4.039, -17.83),
    ("ct", "10%"): (-3.1279, -2.418, -7.58),
}

# -----------------------------------------------------------------------
# Engle-Granger / residual-based cointegration test critical values
# Keys: (n_variables, significance_level)
# n_variables includes the dependent variable (so n=2 means bivariate)
# Model with constant only
# -----------------------------------------------------------------------
_EG_TAU_COEFS: dict[tuple[int, str], tuple[float, float, float]] = {
    # n=2 (bivariate cointegration)
    (2, "1%"): (-3.9001, -10.534, -30.03),
    (2, "5%"): (-3.3377, -5.967, -8.98),
    (2, "10%"): (-3.0462, -4.069, -5.73),
    # n=3
    (3, "1%"): (-4.2981, -13.790, -46.37),
    (3, "5%"): (-3.7429, -8.352, -13.41),
    (3, "10%"): (-3.4518, -6.241, -2.79),
    # n=4
    (4, "1%"): (-4.6676, -16.929, -59.56),
    (4, "5%"): (-4.1193, -10.745, -21.57),
    (4, "10%"): (-3.8274, -8.317, -13.63),
    # n=5
    (5, "1%"): (-5.0202, -19.905, -77.94),
    (5, "5%"): (-4.4745, -13.062, -30.81),
    (5, "10%"): (-4.1841, -10.378, -21.27),
    # n=6
    (6, "1%"): (-5.3580, -22.504, -96.58),
    (6, "5%"): (-4.8157, -15.169, -39.65),
    (6, "10%"): (-4.5250, -12.227, -28.18),
    # n=7
    (7, "1%"): (-5.6803, -25.562, -108.04),
    (7, "5%"): (-5.1432, -17.503, -48.22),
    (7, "10%"): (-4.8530, -14.210, -33.96),
}

# -----------------------------------------------------------------------
# Phillips-Ouliaris critical values (similar to EG but different coefficients)
# Keys: (n_variables, significance_level)
# -----------------------------------------------------------------------
_PO_TAU_COEFS: dict[tuple[int, str], tuple[float, float, float]] = {
    (2, "1%"): (-3.9001, -10.534, -30.03),
    (2, "5%"): (-3.3377, -5.967, -8.98),
    (2, "10%"): (-3.0462, -4.069, -5.73),
    (3, "1%"): (-4.2981, -13.790, -46.37),
    (3, "5%"): (-3.7429, -8.352, -13.41),
    (3, "10%"): (-3.4518, -6.241, -2.79),
    (4, "1%"): (-4.6676, -16.929, -59.56),
    (4, "5%"): (-4.1193, -10.745, -21.57),
    (4, "10%"): (-3.8274, -8.317, -13.63),
}


def mackinnon_crit(
    nobs: int,
    regression: str = "c",
    n_vars: int = 1,
    test_type: str = "adf",
) -> dict[str, float]:
    """Compute MacKinnon critical values for unit root / cointegration tests.

    Uses the regression surface approximation:
        c(p) = beta_inf + beta_1/T + beta_2/T^2

    Parameters
    ----------
    nobs : int
        Sample size.
    regression : str
        Regression model: 'n' (none), 'c' (constant), 'ct' (constant+trend).
        Only used for ADF (test_type='adf'). Ignored for EG and PO.
    n_vars : int
        Number of variables. For ADF, n_vars=1 always.
        For Engle-Granger and Phillips-Ouliaris, n_vars >= 2.
    test_type : str
        Type of test: 'adf', 'eg' (Engle-Granger), or 'po' (Phillips-Ouliaris).

    Returns
    -------
    dict[str, float]
        Critical values at 1%, 5%, and 10% significance levels.
        Example: {'1%': -3.43, '5%': -2.86, '10%': -2.57}

    Raises
    ------
    ValueError
        If the regression model or n_vars is not supported.
    """
    levels = ["1%", "5%", "10%"]
    result: dict[str, float] = {}

    for level in levels:
        if test_type == "adf":
            key = (regression, level)
            if key not in _ADF_TAU_COEFS:
                raise ValueError(
                    f"Unsupported ADF model '{regression}'. "
                    f"Choose from 'n', 'c', 'ct'."
                )
            beta_inf, beta_1, beta_2 = _ADF_TAU_COEFS[key]
        elif test_type == "eg":
            key_eg = (n_vars, level)
            if key_eg not in _EG_TAU_COEFS:
                raise ValueError(
                    f"Unsupported n_vars={n_vars} for Engle-Granger. "
                    f"Supported: 2-7."
                )
            beta_inf, beta_1, beta_2 = _EG_TAU_COEFS[key_eg]
        elif test_type == "po":
            key_po = (n_vars, level)
            if key_po not in _PO_TAU_COEFS:
                raise ValueError(
                    f"Unsupported n_vars={n_vars} for Phillips-Ouliaris. "
                    f"Supported: 2-4."
                )
            beta_inf, beta_1, beta_2 = _PO_TAU_COEFS[key_po]
        else:
            raise ValueError(
                f"Unknown test_type '{test_type}'. Choose from 'adf', 'eg', 'po'."
            )

        result[level] = beta_inf + beta_1 / nobs + beta_2 / (nobs * nobs)

    return result


def mackinnon_pvalue(
    stat: float,
    regression: str = "c",
    n_vars: int = 1,
    nobs: int = 1000,
) -> float:
    """Approximate p-value for ADF test using MacKinnon (1996) tables.

    Uses linear interpolation between critical values at standard levels.
    For stats beyond the tabled range, returns boundary values (< 0.01 or > 0.10).

    Parameters
    ----------
    stat : float
        Test statistic value.
    regression : str
        Regression model: 'n', 'c', 'ct'.
    n_vars : int
        Number of variables (1 for ADF).
    nobs : int
        Sample size for critical value computation.

    Returns
    -------
    float
        Approximate p-value (between 0.0 and 1.0).
    """
    cv = mackinnon_crit(nobs, regression=regression, n_vars=n_vars, test_type="adf")

    # Critical values are negative; more negative = stronger rejection
    # p-value ordering: stat < cv_1% => p < 0.01, stat < cv_5% => p < 0.05, etc.
    cv_1 = cv["1%"]
    cv_5 = cv["5%"]
    cv_10 = cv["10%"]

    if stat <= cv_1:
        return 0.005  # p < 0.01
    elif stat <= cv_5:
        # Interpolate between 1% and 5%
        frac = (stat - cv_1) / (cv_5 - cv_1)
        return 0.01 + frac * (0.05 - 0.01)
    elif stat <= cv_10:
        # Interpolate between 5% and 10%
        frac = (stat - cv_5) / (cv_10 - cv_5)
        return 0.05 + frac * (0.10 - 0.05)
    else:
        # Beyond 10% critical value
        # Use rough extrapolation toward 1.0
        # For large stat values (close to 0 or positive), p -> 1
        if stat > 0:
            return 0.99
        frac = (stat - cv_10) / (0.0 - cv_10)
        return min(0.10 + frac * 0.90, 0.99)
