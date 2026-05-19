"""
Data generators for VAR/VECM examples.

Provides functions to generate synthetic VAR, VECM, and Granger-causal
processes with fixed seeds for reproducibility.
"""

import numpy as np


def generate_var_process(
    n: int = 200,
    A_matrices: list[np.ndarray] | None = None,
    sigma: np.ndarray | None = None,
    seed: int = 42,
    const: np.ndarray | None = None,
) -> np.ndarray:
    """Generate a VAR(p) process.

    Parameters
    ----------
    n : int
        Number of observations to generate.
    A_matrices : list of np.ndarray or None
        List of coefficient matrices [A_1, A_2, ...] where each A_i is (k x k).
        Convention: y_t = const + A_1 * y_{t-1} + A_2 * y_{t-2} + ... + u_t.
        If None, generates a stationary bivariate VAR(1).
    sigma : np.ndarray or None
        Covariance matrix of innovations (k x k). If None, uses identity.
    seed : int
        Random seed for reproducibility.
    const : np.ndarray or None
        Constant/intercept vector (k,). If None, uses zeros.

    Returns
    -------
    np.ndarray
        Generated VAR process of shape (n, k).
    """
    rng = np.random.default_rng(seed)

    if A_matrices is None:
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.3]])]

    k = A_matrices[0].shape[0]
    p = len(A_matrices)

    if sigma is None:
        sigma = np.eye(k)

    if const is None:
        const = np.zeros(k)

    burn = max(200, 20 * p)
    total = n + burn

    # Generate innovations from multivariate normal
    innovations = rng.multivariate_normal(np.zeros(k), sigma, total)

    y = np.zeros((total, k))

    for t in range(total):
        y[t] = const + innovations[t]
        for lag_idx, A in enumerate(A_matrices):
            if t - lag_idx - 1 >= 0:
                y[t] += A @ y[t - lag_idx - 1]

    return y[burn:]


def generate_vecm_process(
    n: int = 200,
    alpha: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    Gamma: list[np.ndarray] | None = None,
    sigma: np.ndarray | None = None,
    seed: int = 42,
    const: np.ndarray | None = None,
) -> np.ndarray:
    """Generate a VECM process with cointegration.

    The VECM representation:
        Delta y_t = const + alpha * beta' * y_{t-1} + Gamma_1 * Delta y_{t-1} + ... + u_t

    Parameters
    ----------
    n : int
        Number of observations to generate.
    alpha : np.ndarray or None
        Loading/adjustment matrix (k x r) where r is cointegration rank.
        If None, uses default for bivariate system with 1 cointegrating relation.
    beta : np.ndarray or None
        Cointegrating vectors (k x r).
        If None, uses default [1, -1]'.
    Gamma : list of np.ndarray or None
        Short-run dynamics matrices for lagged differences.
        If None, no short-run dynamics beyond the error correction term.
    sigma : np.ndarray or None
        Covariance matrix of innovations (k x k). If None, uses identity.
    seed : int
        Random seed for reproducibility.
    const : np.ndarray or None
        Constant/intercept vector (k,). If None, uses zeros.

    Returns
    -------
    np.ndarray
        Generated VECM process of shape (n, k) in levels.
    """
    rng = np.random.default_rng(seed)

    if alpha is None:
        alpha = np.array([[-0.3], [0.2]])

    if beta is None:
        beta = np.array([[1.0], [-1.0]])

    k = alpha.shape[0]

    if Gamma is None:
        Gamma = []

    if sigma is None:
        sigma = np.eye(k)

    if const is None:
        const = np.zeros(k)

    p_gamma = len(Gamma)
    burn = max(200, 20 * (p_gamma + 1))
    total = n + burn

    innovations = rng.multivariate_normal(np.zeros(k), sigma, total)

    # Generate in levels
    y = np.zeros((total, k))
    # Initialize with small random values
    y[0] = rng.normal(0, 0.1, k)

    for t in range(1, total):
        # Error correction term: alpha * beta' * y_{t-1}
        ec_term = alpha @ (beta.T @ y[t - 1])

        # Short-run dynamics
        sr_term = np.zeros(k)
        for lag_idx, G in enumerate(Gamma):
            if t - lag_idx - 2 >= 0:
                dy_lag = y[t - lag_idx - 1] - y[t - lag_idx - 2]
                sr_term += G @ dy_lag

        # Delta y_t
        dy = const + ec_term + sr_term + innovations[t]

        y[t] = y[t - 1] + dy

    return y[burn:]


def generate_granger_causal(
    n: int = 200,
    lag_effect: int = 2,
    seed: int = 42,
    coeff: float = 0.5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Generate a bivariate process where x Granger-causes y.

    x_t is an independent AR(1) process.
    y_t depends on its own lag and lagged values of x (with specified lag).

    Parameters
    ----------
    n : int
        Number of observations to generate.
    lag_effect : int
        The lag at which x affects y (e.g., 2 means x_{t-2} affects y_t).
    seed : int
        Random seed for reproducibility.
    coeff : float
        Strength of the Granger-causal effect from x to y.
    sigma : float
        Standard deviation of innovations.

    Returns
    -------
    np.ndarray
        Array of shape (n, 2) where column 0 is x and column 1 is y.
    """
    rng = np.random.default_rng(seed)

    burn = max(200, 20 * lag_effect)
    total = n + burn

    x = np.zeros(total)
    y = np.zeros(total)

    eps_x = rng.normal(0, sigma, total)
    eps_y = rng.normal(0, sigma, total)

    for t in range(1, total):
        # x is AR(1) - independent
        x[t] = 0.6 * x[t - 1] + eps_x[t]

        # y depends on own lag and lagged x
        y[t] = 0.4 * y[t - 1] + eps_y[t]
        if t - lag_effect >= 0:
            y[t] += coeff * x[t - lag_effect]

    result = np.column_stack([x[burn:], y[burn:]])
    return result
