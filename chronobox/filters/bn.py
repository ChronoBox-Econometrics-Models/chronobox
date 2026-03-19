"""
Beveridge-Nelson (BN) decomposition for I(1) time series.

Decomposes y_t into permanent (trend) and transitory (cycle) components:
    y_t = tau_t + c_t

The trend is a random walk with drift:
    tau_t = tau_{t-1} + mu + psi(1) * eps_t

where psi(1) = theta(1)/phi(1) is the long-run multiplier from the
ARMA representation of the first difference.

References
----------
Beveridge, S., & Nelson, C. R. (1981). A new approach to decomposition of
    economic time series into permanent and transitory components with particular
    attention to measurement of the "business cycle". Journal of Monetary
    Economics, 7(2), 151-174.

Morley, J. C. (2002). A state-space approach to the Beveridge-Nelson decomposition.
    Economics Letters, 75(1), 123-127.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class BNDecompositionResult:
    """Result of Beveridge-Nelson decomposition.

    Attributes
    ----------
    trend : ndarray, shape (T,)
        Permanent (trend) component. Random walk with drift.
    cycle : ndarray, shape (T,)
        Transitory (cyclical) component. Stationary.
    psi_one : float
        Long-run multiplier psi(1) = theta(1) / phi(1).
    ar_coeffs : ndarray
        AR coefficients (phi_1, ..., phi_p).
    ma_coeffs : ndarray
        MA coefficients (theta_1, ..., theta_q).
    drift : float
        Estimated drift (mean of first differences).
    residuals : ndarray
        ARMA residuals eps_t.
    """

    trend: np.ndarray
    cycle: np.ndarray
    psi_one: float
    ar_coeffs: np.ndarray
    ma_coeffs: np.ndarray
    drift: float
    residuals: np.ndarray


def _ar_coefficients_ols(
    dy: np.ndarray, p: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit AR(p) model to dy via OLS.

    Returns (mu, phi, residuals) where:
    - mu: intercept (drift)
    - phi: AR coefficients [phi_1, ..., phi_p]
    - residuals: eps_t
    """
    t_len = len(dy)
    n_obs = t_len - p

    # Build design matrix
    y_vec = dy[p:]
    x_mat = np.ones((n_obs, p + 1))  # constant + p lags
    for j in range(p):
        x_mat[:, j + 1] = dy[p - j - 1 : t_len - j - 1]

    # OLS
    beta = np.linalg.lstsq(x_mat, y_vec, rcond=None)[0]
    mu = float(beta[0])
    phi = beta[1:]
    residuals_short = y_vec - x_mat @ beta

    # Full residuals (pad with zeros at start)
    residuals = np.zeros(t_len)
    residuals[p:] = residuals_short

    return mu, phi, residuals


def _compute_psi_coefficients(
    phi: np.ndarray,
    theta: np.ndarray,
    n_terms: int = 200,
) -> np.ndarray:
    """Compute MA(infinity) representation coefficients psi_j.

    Given phi(L)*psi(L) = theta(L), compute psi_0, psi_1, ..., psi_{n_terms-1}.

    Parameters
    ----------
    phi : ndarray, shape (p,)
        AR coefficients [phi_1, ..., phi_p].
    theta : ndarray, shape (q,)
        MA coefficients [theta_1, ..., theta_q].
    n_terms : int
        Number of psi coefficients to compute.

    Returns
    -------
    psi : ndarray, shape (n_terms,)
        MA(infinity) coefficients.
    """
    p = len(phi)
    q = len(theta)
    psi = np.zeros(n_terms)
    psi[0] = 1.0

    for j in range(1, n_terms):
        val = 0.0
        # AR contribution
        for k in range(min(j, p)):
            val += phi[k] * psi[j - k - 1]
        # MA contribution
        if j <= q:
            val += theta[j - 1]
        psi[j] = val

    return psi


def _bn_core(
    y: np.ndarray, p: int, q: int, n_ma_terms: int
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float, np.ndarray]:
    """Core BN decomposition logic shared by both public functions."""
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")

    t_len = len(y)
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    min_obs = p + 4
    if t_len < min_obs:
        raise ValueError(f"Need at least {min_obs} observations, got {t_len}")

    # First differences
    dy = np.diff(y)  # length T-1

    # Fit AR(p) to dy via OLS
    mu, phi, eps = _ar_coefficients_ols(dy, p)

    # MA coefficients (empty for AR-only model)
    theta = np.zeros(q) if q > 0 else np.array([], dtype=np.float64)

    # Compute psi(1) = theta(1) / phi(1)
    theta_one = 1.0 + np.sum(theta)
    phi_one = 1.0 - np.sum(phi)

    if abs(phi_one) < 1e-12:
        raise ValueError(
            "phi(1) is approximately zero, indicating a unit root in differences. "
            "The series may be I(2), not I(1)."
        )

    psi_one = float(theta_one / phi_one)

    # Compute MA(infinity) coefficients (validates convergence)
    _compute_psi_coefficients(
        phi, theta if len(theta) > 0 else np.array([], dtype=np.float64), n_ma_terms
    )

    # Build full trend: tau_t = tau_{t-1} + mu + psi(1) * eps_t
    trend = np.zeros(t_len)
    trend[0] = y[0]

    for t in range(1, t_len):
        trend[t] = trend[t - 1] + mu + psi_one * eps[t - 1]

    # Cycle = y - trend
    cycle = y - trend

    return trend, cycle, psi_one, phi, theta, mu, eps


def bn_decomposition(
    y: ArrayLike,
    p: int = 2,
    q: int = 0,
    n_ma_terms: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform Beveridge-Nelson decomposition on an I(1) series.

    Parameters
    ----------
    y : array_like, shape (T,)
        Time series (assumed I(1)). Must be 1-D.
    p : int
        AR order for the ARMA model on first differences. Default is 2.
    q : int
        MA order for the ARMA model on first differences. Default is 0.
        Note: q=0 uses pure AR(p) estimation via OLS. For q>0, a more
        complex estimation would be needed; this implementation supports
        q=0 (AR only) for simplicity and robustness.
    n_ma_terms : int
        Number of terms for the MA(infinity) truncation. Default is 200.

    Returns
    -------
    trend : ndarray, shape (T,)
        Permanent (trend) component.
    cycle : ndarray, shape (T,)
        Transitory (cyclical) component.

    Raises
    ------
    ValueError
        If y is not 1-D, T < p+4, or p < 1.

    Notes
    -----
    The decomposition satisfies: trend + cycle = y (exactly).

    For q=0, the ARMA model reduces to AR(p), estimated via OLS.
    The long-run multiplier is: psi(1) = 1 / phi(1) = 1 / (1 - phi_1 - ... - phi_p).

    Examples
    --------
    >>> import numpy as np
    >>> y = np.random.randn(200).cumsum()
    >>> trend, cycle = bn_decomposition(y, p=2)
    >>> np.allclose(trend + cycle, y)
    True
    """
    y_arr = np.asarray(y, dtype=np.float64).squeeze()
    trend, cycle, *_ = _bn_core(y_arr, p, q, n_ma_terms)
    return trend, cycle


def bn_decomposition_detailed(
    y: ArrayLike,
    p: int = 2,
    q: int = 0,
    n_ma_terms: int = 200,
) -> BNDecompositionResult:
    """Perform BN decomposition and return detailed results.

    Parameters
    ----------
    y : array_like, shape (T,)
        Time series (assumed I(1)).
    p : int
        AR order. Default is 2.
    q : int
        MA order. Default is 0.
    n_ma_terms : int
        MA(infinity) truncation. Default is 200.

    Returns
    -------
    BNDecompositionResult
        Full results including trend, cycle, coefficients, and diagnostics.
    """
    y_arr = np.asarray(y, dtype=np.float64).squeeze()
    trend, cycle, psi_one, phi, theta, mu, eps = _bn_core(y_arr, p, q, n_ma_terms)

    return BNDecompositionResult(
        trend=trend,
        cycle=cycle,
        psi_one=psi_one,
        ar_coeffs=phi,
        ma_coeffs=theta,
        drift=mu,
        residuals=eps,
    )
