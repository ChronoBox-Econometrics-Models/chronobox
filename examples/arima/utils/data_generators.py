"""
Data generators for ARIMA examples.

Provides functions to generate synthetic ARIMA, SARIMA, and ARFIMA processes
with fixed seeds for reproducibility.
"""

import numpy as np


def generate_arima_process(
    n: int = 200,
    ar: list | None = None,
    ma: list | None = None,
    d: int = 0,
    seed: int = 42,
    mean: float = 0.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """Generate an ARIMA(p,d,q) process.

    Parameters
    ----------
    n : int
        Number of observations to generate (after differencing).
    ar : list or None
        AR coefficients [phi_1, phi_2, ...]. Convention: y_t = phi_1*y_{t-1} + ...
    ma : list or None
        MA coefficients [theta_1, theta_2, ...]. Convention: e_t + theta_1*e_{t-1} + ...
    d : int
        Order of integration (number of times to cumulatively sum).
    seed : int
        Random seed for reproducibility.
    mean : float
        Mean of the innovation process.
    sigma : float
        Standard deviation of the innovation process.

    Returns
    -------
    np.ndarray
        Generated ARIMA process of length n.
    """
    rng = np.random.default_rng(seed)
    ar = ar or []
    ma = ma or []

    p = len(ar)
    q = len(ma)

    # Burn-in period to reduce dependence on initial conditions
    burn = max(100, 10 * max(p, q))
    total = n + burn

    innovations = rng.normal(mean, sigma, total)
    y = np.zeros(total)

    for t in range(total):
        # AR component
        ar_term = 0.0
        for i, phi in enumerate(ar):
            if t - i - 1 >= 0:
                ar_term += phi * y[t - i - 1]

        # MA component
        ma_term = 0.0
        for j, theta in enumerate(ma):
            if t - j - 1 >= 0:
                ma_term += theta * innovations[t - j - 1]

        y[t] = ar_term + ma_term + innovations[t]

    # Remove burn-in
    y = y[burn:]

    # Apply integration (cumulative sum d times)
    for _ in range(d):
        y = np.cumsum(y)

    return y


def generate_sarima_process(
    n: int = 200,
    ar: list | None = None,
    ma: list | None = None,
    d: int = 0,
    sar: list | None = None,
    sma: list | None = None,
    D: int = 0,
    s: int = 12,
    seed: int = 42,
    sigma: float = 1.0,
) -> np.ndarray:
    """Generate a SARIMA(p,d,q)(P,D,Q)[s] process.

    Parameters
    ----------
    n : int
        Number of observations to generate.
    ar : list or None
        Non-seasonal AR coefficients.
    ma : list or None
        Non-seasonal MA coefficients.
    d : int
        Non-seasonal differencing order.
    sar : list or None
        Seasonal AR coefficients.
    sma : list or None
        Seasonal MA coefficients.
    D : int
        Seasonal differencing order.
    s : int
        Seasonal period (e.g., 12 for monthly data).
    seed : int
        Random seed for reproducibility.
    sigma : float
        Standard deviation of the innovation process.

    Returns
    -------
    np.ndarray
        Generated SARIMA process of length n.
    """
    rng = np.random.default_rng(seed)
    ar = ar or []
    ma = ma or []
    sar = sar or []
    sma = sma or []

    p = len(ar)
    q = len(ma)
    P = len(sar)
    Q = len(sma)

    # Build the full AR polynomial by convolving non-seasonal and seasonal parts
    # Non-seasonal AR: 1 - phi_1*B - phi_2*B^2 - ...
    ar_poly = np.zeros(p + 1)
    ar_poly[0] = 1.0
    for i, phi in enumerate(ar):
        ar_poly[i + 1] = -phi

    # Seasonal AR: 1 - Phi_1*B^s - Phi_2*B^{2s} - ...
    sar_poly = np.zeros(P * s + 1)
    sar_poly[0] = 1.0
    for i, Phi in enumerate(sar):
        sar_poly[(i + 1) * s] = -Phi

    # Full AR = convolution of non-seasonal and seasonal
    full_ar = np.convolve(ar_poly, sar_poly)

    # Non-seasonal MA: 1 + theta_1*B + theta_2*B^2 + ...
    ma_poly = np.zeros(q + 1)
    ma_poly[0] = 1.0
    for j, theta in enumerate(ma):
        ma_poly[j + 1] = theta

    # Seasonal MA: 1 + Theta_1*B^s + Theta_2*B^{2s} + ...
    sma_poly = np.zeros(Q * s + 1)
    sma_poly[0] = 1.0
    for j, Theta in enumerate(sma):
        sma_poly[(j + 1) * s] = Theta

    # Full MA = convolution of non-seasonal and seasonal
    full_ma = np.convolve(ma_poly, sma_poly)

    max_lag = max(len(full_ar), len(full_ma))
    burn = max(200, 10 * max_lag)
    total = n + burn

    innovations = rng.normal(0, sigma, total)
    y = np.zeros(total)

    for t in range(total):
        # AR component (skip index 0 which is the leading 1)
        ar_term = 0.0
        for i in range(1, len(full_ar)):
            if t - i >= 0:
                ar_term -= full_ar[i] * y[t - i]

        # MA component (skip index 0)
        ma_term = 0.0
        for j in range(1, len(full_ma)):
            if t - j >= 0:
                ma_term += full_ma[j] * innovations[t - j]

        y[t] = ar_term + ma_term + innovations[t]

    y = y[burn:]

    # Apply seasonal differencing D times
    for _ in range(D):
        y = np.cumsum(y)

    # Apply non-seasonal differencing d times
    for _ in range(d):
        y = np.cumsum(y)

    return y


def generate_arfima_process(
    n: int = 500,
    d_frac: float = 0.3,
    ar: list | None = None,
    ma: list | None = None,
    seed: int = 42,
    sigma: float = 1.0,
    truncation: int = 1000,
) -> np.ndarray:
    """Generate an ARFIMA(p,d,q) process with fractional integration.

    Uses the Cholesky method based on the fractional differencing filter
    (truncated binomial expansion of (1-B)^{-d}).

    Parameters
    ----------
    n : int
        Number of observations to generate.
    d_frac : float
        Fractional differencing parameter. Typically in (-0.5, 0.5).
        Values in (0, 0.5) produce long memory.
    ar : list or None
        AR coefficients.
    ma : list or None
        MA coefficients.
    seed : int
        Random seed for reproducibility.
    sigma : float
        Standard deviation of innovations.
    truncation : int
        Truncation lag for the fractional differencing filter.

    Returns
    -------
    np.ndarray
        Generated ARFIMA process of length n.
    """
    rng = np.random.default_rng(seed)
    ar = ar or []
    ma = ma or []

    p = len(ar)
    q = len(ma)

    burn = max(200, 10 * max(p, q, 1))
    total = n + burn

    innovations = rng.normal(0, sigma, total)

    # Step 1: Apply ARMA filter to innovations
    arma = np.zeros(total)
    for t in range(total):
        arma[t] = innovations[t]

        # MA component
        for j, theta in enumerate(ma):
            if t - j - 1 >= 0:
                arma[t] += theta * innovations[t - j - 1]

        # AR component
        for i, phi in enumerate(ar):
            if t - i - 1 >= 0:
                arma[t] += phi * arma[t - i - 1]

    # Step 2: Apply fractional integration filter (1-B)^{-d}
    # Coefficients: pi_0 = 1, pi_k = pi_{k-1} * (k - 1 + d) / k
    trunc = min(truncation, total)
    pi = np.zeros(trunc)
    pi[0] = 1.0
    for k in range(1, trunc):
        pi[k] = pi[k - 1] * (k - 1 + d_frac) / k

    y = np.zeros(total)
    for t in range(total):
        for k in range(min(t + 1, trunc)):
            y[t] += pi[k] * arma[t - k]

    return y[burn:]
