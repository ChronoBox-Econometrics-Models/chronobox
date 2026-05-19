"""
Data generators for ARDL examples.

Provides functions to generate synthetic ARDL processes, ECM data with
long-run equilibrium relationships, and mixed-integration variables
suitable for Pesaran-Shin-Smith bounds testing.
"""

import numpy as np


def generate_ardl_process(
    n: int = 200,
    ar_lags: int = 2,
    x_lags: int = 2,
    seed: int = 42,
    ar_coefs: np.ndarray | None = None,
    x_coefs: np.ndarray | None = None,
    const: float = 1.0,
    sigma_y: float = 0.5,
    sigma_x: float = 1.0,
) -> np.ndarray:
    """Generate an ARDL(p, q) process.

    Generates y_t = const + sum_{i=1}^{p} phi_i * y_{t-i}
                           + sum_{j=0}^{q} theta_j * x_{t-j} + u_t

    where x_t follows an independent AR(1) process.

    Parameters
    ----------
    n : int
        Number of observations to generate.
    ar_lags : int
        Number of AR lags (p) for the dependent variable.
    x_lags : int
        Number of lags of x (q) to include (plus contemporaneous).
    seed : int
        Random seed for reproducibility.
    ar_coefs : np.ndarray or None
        AR coefficients [phi_1, ..., phi_p]. If None, generates stable defaults.
    x_coefs : np.ndarray or None
        Coefficients for x [theta_0, theta_1, ..., theta_q]. If None, generates defaults.
    const : float
        Intercept term.
    sigma_y : float
        Standard deviation of innovations for y.
    sigma_x : float
        Standard deviation of innovations for x.

    Returns
    -------
    np.ndarray
        Array of shape (n, 2) where column 0 is y and column 1 is x.
    """
    rng = np.random.default_rng(seed)

    if ar_coefs is None:
        # Generate stable AR coefficients that decay
        ar_coefs = np.array([0.4 / (i + 1) for i in range(ar_lags)])

    if x_coefs is None:
        # Contemporaneous effect strongest, then decay
        x_coefs = np.array([0.6 / (j + 1) for j in range(x_lags + 1)])

    burn = max(200, 20 * max(ar_lags, x_lags))
    total = n + burn

    eps_y = rng.normal(0, sigma_y, total)
    eps_x = rng.normal(0, sigma_x, total)

    # Generate x as AR(1)
    x = np.zeros(total)
    for t in range(1, total):
        x[t] = 0.7 * x[t - 1] + eps_x[t]

    # Generate y as ARDL(p, q)
    y = np.zeros(total)
    for t in range(max(ar_lags, x_lags), total):
        y[t] = const + eps_y[t]
        for i in range(ar_lags):
            y[t] += ar_coefs[i] * y[t - i - 1]
        for j in range(x_lags + 1):
            y[t] += x_coefs[j] * x[t - j]

    result = np.column_stack([y[burn:], x[burn:]])
    return result


def generate_ecm_data(
    n: int = 200,
    speed_of_adjustment: float = -0.3,
    long_run_coef: float = 0.8,
    seed: int = 42,
    sigma_y: float = 0.5,
    sigma_x: float = 1.0,
) -> np.ndarray:
    """Generate data with a long-run equilibrium relationship for ECM analysis.

    The data-generating process is:
        y_t and x_t are I(1) and cointegrated with long-run relation y = a + b*x.
        The ECM representation:
            Delta y_t = gamma * (y_{t-1} - a - b * x_{t-1})
                        + c_1 * Delta x_t + u_t

    Parameters
    ----------
    n : int
        Number of observations to generate.
    speed_of_adjustment : float
        Error correction speed (gamma). Should be negative for stability.
        Typical range: (-1, 0). More negative = faster adjustment.
    long_run_coef : float
        Long-run coefficient (b) linking y to x.
    seed : int
        Random seed for reproducibility.
    sigma_y : float
        Standard deviation of innovations for y equation.
    sigma_x : float
        Standard deviation of innovations for x (random walk).

    Returns
    -------
    np.ndarray
        Array of shape (n, 2) where column 0 is y and column 1 is x.
    """
    rng = np.random.default_rng(seed)

    burn = 200
    total = n + burn
    long_run_const = 2.0
    short_run_x = 0.4

    eps_y = rng.normal(0, sigma_y, total)
    eps_x = rng.normal(0, sigma_x, total)

    x = np.zeros(total)
    y = np.zeros(total)

    # Initialize x on its long-run path
    x[0] = rng.normal(0, 1)
    y[0] = long_run_const + long_run_coef * x[0] + rng.normal(0, 0.5)

    for t in range(1, total):
        # x follows a random walk with drift
        x[t] = x[t - 1] + 0.05 + eps_x[t]

        # ECM for y
        equilibrium_error = y[t - 1] - long_run_const - long_run_coef * x[t - 1]
        dx = x[t] - x[t - 1]
        dy = speed_of_adjustment * equilibrium_error + short_run_x * dx + eps_y[t]
        y[t] = y[t - 1] + dy

    result = np.column_stack([y[burn:], x[burn:]])
    return result


def generate_mixed_integration(
    n: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Generate variables with mixed orders of integration: I(0) and I(1).

    This is the ideal scenario for ARDL bounds testing, which is valid
    regardless of whether regressors are I(0), I(1), or a mix.

    Generates 4 variables:
        - y:  I(1), dependent variable with long-run relationship to x1
        - x1: I(1), cointegrated with y
        - x2: I(0), stationary regressor
        - x3: I(1), non-cointegrated with y (nuisance I(1) variable)

    Parameters
    ----------
    n : int
        Number of observations to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n, 4) with columns [y, x1, x2, x3].
    """
    rng = np.random.default_rng(seed)

    burn = 200
    total = n + burn

    eps_y = rng.normal(0, 0.5, total)
    eps_x1 = rng.normal(0, 0.8, total)
    eps_x2 = rng.normal(0, 1.0, total)
    eps_x3 = rng.normal(0, 0.6, total)

    x1 = np.zeros(total)  # I(1)
    x2 = np.zeros(total)  # I(0) - stationary AR(1)
    x3 = np.zeros(total)  # I(1) - independent random walk
    y = np.zeros(total)   # I(1) - cointegrated with x1

    # Long-run relationship: y = 1.5 + 0.6 * x1
    long_run_const = 1.5
    long_run_coef = 0.6
    ecm_speed = -0.25

    x1[0] = rng.normal(0, 1)
    x3[0] = rng.normal(0, 1)
    y[0] = long_run_const + long_run_coef * x1[0]

    for t in range(1, total):
        # x1: random walk with small drift
        x1[t] = x1[t - 1] + 0.03 + eps_x1[t]

        # x2: stationary AR(1) — I(0)
        x2[t] = 0.6 * x2[t - 1] + eps_x2[t]

        # x3: independent random walk — I(1) but not cointegrated with y
        x3[t] = x3[t - 1] + 0.02 + eps_x3[t]

        # y: ECM with x1 (cointegrated), short-run effects from x2
        eq_error = y[t - 1] - long_run_const - long_run_coef * x1[t - 1]
        dx1 = x1[t] - x1[t - 1]
        dy = (ecm_speed * eq_error
              + 0.4 * dx1
              + 0.3 * x2[t]
              + eps_y[t])
        y[t] = y[t - 1] + dy

    result = np.column_stack([y[burn:], x1[burn:], x2[burn:], x3[burn:]])
    return result
