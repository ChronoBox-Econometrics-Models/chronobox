"""ARFIMA(p,d,q) - Fractionally Integrated ARIMA model.

Supports fractional differencing with d in (-0.5, 0.5) for long-memory processes.

References
----------
- Granger, C.W.J. & Joyeux, R. (1980). An introduction to long-memory time series
  models and fractional differencing. Journal of Time Series Analysis, 1(1), 15-29.
- Hosking, J.R.M. (1981). Fractional differencing. Biometrika, 68(1), 165-176.
- Geweke, J. & Porter-Hudak, S. (1983). The estimation and application of long memory
  time series models. Journal of Time Series Analysis, 4(4), 221-238.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from chronobox.utils.validation import validate_endog


@dataclass
class ARFIMAResults:
    """Results from fitting an ARFIMA model.

    Attributes
    ----------
    d : float
        Fractional differencing parameter.
    ar_params : NDArray[np.float64]
        AR coefficients (length p).
    ma_params : NDArray[np.float64]
        MA coefficients (length q).
    sigma2 : float
        Estimated innovation variance.
    resid : NDArray[np.float64]
        Residuals from the fitted model.
    nobs : int
        Number of observations used in fitting.
    loglik : float
        Log-likelihood value.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    aicc : float
        Corrected AIC.
    order : tuple[int, float, int]
        Model order (p, d, q).
    endog : NDArray[np.float64]
        Original series.
    fittedvalues : NDArray[np.float64]
        Fitted values (one-step-ahead predictions).
    """

    d: float
    ar_params: NDArray[np.float64]
    ma_params: NDArray[np.float64]
    sigma2: float
    resid: NDArray[np.float64]
    nobs: int
    loglik: float
    aic: float
    bic: float
    aicc: float
    order: tuple[int, float, int]
    endog: NDArray[np.float64]
    fittedvalues: NDArray[np.float64]

    def summary(self) -> str:
        """Return a formatted summary of the ARFIMA results."""
        p, d, q = self.order
        lines = [
            "=" * 60,
            f"{'ARFIMA Model Results':^60}",
            "=" * 60,
            f"Model:              ARFIMA({p}, {d:.4f}, {q})",
            f"No. Observations:   {self.nobs}",
            f"Log-Likelihood:     {self.loglik:.4f}",
            f"AIC:                {self.aic:.4f}",
            f"BIC:                {self.bic:.4f}",
            f"AICc:               {self.aicc:.4f}",
            f"Sigma2:             {self.sigma2:.6f}",
            "-" * 60,
            f"{'Parameter':<20} {'Estimate':>12}",
            "-" * 60,
            f"{'d':<20} {self.d:>12.6f}",
        ]
        for i, phi in enumerate(self.ar_params):
            lines.append(f"{'ar.' + str(i + 1):<20} {phi:>12.6f}")
        for i, theta in enumerate(self.ma_params):
            lines.append(f"{'ma.' + str(i + 1):<20} {theta:>12.6f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def forecast(self, steps: int = 1) -> NDArray[np.float64]:
        """Forecast future values.

        Uses the fractionally differenced representation to produce
        multi-step-ahead forecasts.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.

        Returns
        -------
        NDArray[np.float64]
            Array of forecasted values.
        """
        p = len(self.ar_params)
        q = len(self.ma_params)
        y = self.endog.copy()
        resid = self.resid.copy()
        nobs = len(y)

        # Compute fractional diff coefficients for the full series + forecast horizon
        n_pi = nobs + steps
        pi = fractional_diff_coefficients(self.d, n_pi)

        forecasts = np.empty(steps, dtype=np.float64)

        # Extend y and resid arrays for forecasting
        y_ext = np.concatenate([y, np.zeros(steps)])
        resid_ext = np.concatenate([resid, np.zeros(steps)])

        for h in range(steps):
            t = nobs + h

            # Compute w_t from AR/MA recursion (future innovations = 0)
            w_t = 0.0
            for i in range(p):
                if t - 1 - i >= 0:
                    # Compute w at past time
                    w_past = _compute_w(y_ext, pi, t - 1 - i)
                    w_t += self.ar_params[i] * w_past
            for j in range(q):
                if t - 1 - j < nobs and t - 1 - j >= 0:
                    w_t += self.ma_params[j] * resid_ext[t - 1 - j]

            # Invert fractional differencing: y_t = w_t - sum_{k=1}^{t} pi_k * y_{t-k}
            y_t = w_t
            for k in range(1, min(t + 1, len(pi))):
                if t - k >= 0:
                    y_t -= pi[k] * y_ext[t - k]
            y_ext[t] = y_t
            forecasts[h] = y_t

        return forecasts


def _compute_w(
    y: NDArray[np.float64], pi: NDArray[np.float64], t: int
) -> float:
    """Compute the fractionally differenced value w_t = sum_k pi_k * y_{t-k}."""
    w = 0.0
    for k in range(min(t + 1, len(pi))):
        w += pi[k] * y[t - k]
    return w


def fractional_diff_coefficients(
    d: float, n: int
) -> NDArray[np.float64]:
    """Compute coefficients pi_k of the fractional differencing operator (1-L)^d.

    The expansion is:
        (1-L)^d = sum_{k=0}^{n-1} pi_k * L^k

    where:
        pi_0 = 1
        pi_k = pi_{k-1} * (k - 1 - d) / k

    Parameters
    ----------
    d : float
        Fractional differencing parameter. Typically d in (-0.5, 0.5).
    n : int
        Number of coefficients to compute.

    Returns
    -------
    NDArray[np.float64]
        Array of coefficients [pi_0, pi_1, ..., pi_{n-1}].
    """
    pi = np.empty(n, dtype=np.float64)
    pi[0] = 1.0
    for k in range(1, n):
        pi[k] = pi[k - 1] * (k - 1 - d) / k
    return pi


def fractional_diff(
    y: NDArray[np.float64], d: float, truncation: int | None = None
) -> NDArray[np.float64]:
    """Apply fractional differencing (1-L)^d to a time series.

    Parameters
    ----------
    y : NDArray[np.float64]
        Input series of length T.
    d : float
        Fractional differencing parameter.
    truncation : int or None
        Maximum number of coefficients. Default: min(len(y), 1000).

    Returns
    -------
    NDArray[np.float64]
        Fractionally differenced series (same length as y, with initial
        values computed using available data).
    """
    n = len(y)
    if truncation is None:
        truncation = min(n, 1000)

    pi = fractional_diff_coefficients(d, truncation)
    w = np.empty(n, dtype=np.float64)

    for t in range(n):
        n_terms = min(t + 1, truncation)
        w[t] = np.dot(pi[:n_terms], y[t::-1][:n_terms]) if n_terms > 0 else 0.0

    return w


def estimate_d_gph(
    y: NDArray[np.float64],
    bandwidth_exp: float = 0.5,
) -> tuple[float, float]:
    """Estimate fractional differencing parameter d via GPH log-periodogram regression.

    Geweke & Porter-Hudak (1983) estimator.

    Parameters
    ----------
    y : NDArray[np.float64]
        Input time series.
    bandwidth_exp : float
        Exponent for bandwidth: g(T) = T^bandwidth_exp. Default 0.5.

    Returns
    -------
    tuple[float, float]
        (d_hat, se_d) - estimated d and its standard error.
    """
    n_obs = len(y)
    g = int(np.floor(n_obs**bandwidth_exp))
    if g < 3:
        msg = f"Bandwidth too small: g={g}. Need at least 3 frequencies."
        raise ValueError(msg)

    # Compute periodogram
    freqs = 2.0 * np.pi * np.arange(1, g + 1) / n_obs
    # Use FFT-based periodogram
    fft_vals = np.fft.fft(y - np.mean(y))
    periodogram = np.abs(fft_vals) ** 2 / (2.0 * np.pi * n_obs)
    pgram = periodogram[1 : g + 1]

    # Regressor: log(4 * sin^2(w_j / 2))
    x = np.log(4.0 * np.sin(freqs / 2.0) ** 2)
    log_pgram = np.log(pgram + 1e-300)  # avoid log(0)

    # OLS regression: log(I(w_j)) = c - d * x_j + u_j
    # d_hat = -slope
    n = len(x)
    x_mean = np.mean(x)
    log_pgram_mean = np.mean(log_pgram)

    ss_xx = np.sum((x - x_mean) ** 2)
    ss_xy = np.sum((x - x_mean) * (log_pgram - log_pgram_mean))

    slope = ss_xy / ss_xx
    d_hat = -slope

    # Standard error
    intercept = log_pgram_mean - slope * x_mean
    residuals = log_pgram - intercept - slope * x
    mse = np.sum(residuals**2) / (n - 2)
    se_d = np.sqrt(mse / ss_xx)

    return float(d_hat), float(se_d)


def estimate_d_local_whittle(
    y: NDArray[np.float64],
    bandwidth_exp: float = 0.65,
) -> tuple[float, float]:
    """Estimate fractional differencing parameter d via Local Whittle estimator.

    Robinson (1995) semi-parametric estimator.

    Parameters
    ----------
    y : NDArray[np.float64]
        Input time series.
    bandwidth_exp : float
        Exponent for bandwidth: m = T^bandwidth_exp. Default 0.65.

    Returns
    -------
    tuple[float, float]
        (d_hat, se_d) - estimated d and its approximate standard error.
    """
    n_obs = len(y)
    m = int(np.floor(n_obs**bandwidth_exp))
    if m < 3:
        msg = f"Bandwidth too small: m={m}. Need at least 3 frequencies."
        raise ValueError(msg)

    # Compute periodogram
    fft_vals = np.fft.fft(y - np.mean(y))
    periodogram = np.abs(fft_vals) ** 2 / (2.0 * np.pi * n_obs)
    freqs = 2.0 * np.pi * np.arange(1, m + 1) / n_obs
    pgram = periodogram[1 : m + 1]

    def objective(d_val: float) -> float:
        """Local Whittle objective function R(d)."""
        lambda_j = freqs ** (-2.0 * d_val)
        g_hat = np.mean(pgram / lambda_j)
        obj = np.log(g_hat) - (2.0 * d_val / m) * np.sum(np.log(freqs))
        return float(obj)

    result = optimize.minimize_scalar(
        objective, bounds=(-0.499, 0.499), method="bounded"
    )
    d_hat = float(result.x)

    # Approximate standard error: se = 1 / (2 * sqrt(m))
    se_d = 1.0 / (2.0 * np.sqrt(m))

    return d_hat, se_d


class ARFIMA:
    """ARFIMA(p,d,q) - Fractionally Integrated ARIMA model.

    Allows fractional differencing parameter d in (-0.5, 0.5) for modeling
    long-memory processes.

    Parameters
    ----------
    order : tuple[int, float, int]
        (p, d, q) where p is the AR order, d is the fractional differencing
        parameter (can be float), and q is the MA order. If d is provided
        as a starting value, it can be estimated during fitting.

    Examples
    --------
    >>> import numpy as np
    >>> from chronobox.models.arfima import ARFIMA
    >>> rng = np.random.default_rng(42)
    >>> y = rng.normal(size=200)
    >>> model = ARFIMA(order=(1, 0.0, 0))
    >>> results = model.fit(y)
    >>> print(results.d)
    """

    def __init__(self, order: tuple[int, float, int] = (0, 0.0, 0)) -> None:
        if len(order) != 3:
            msg = f"order must have 3 elements (p, d, q), got {len(order)}"
            raise ValueError(msg)
        p, d, q = order
        if p < 0 or q < 0:
            msg = f"p and q must be non-negative, got p={p}, q={q}"
            raise ValueError(msg)
        if not (-0.5 < d < 0.5):
            warnings.warn(
                f"d={d} is outside the stationary region (-0.5, 0.5). "
                "Results may be unreliable.",
                stacklevel=2,
            )
        self.p = int(p)
        self.d_init = float(d)
        self.q = int(q)
        self.order = (self.p, float(d), self.q)

    def fit(
        self,
        endog: NDArray[np.float64] | list[float],
        method: Literal["css", "mle"] = "css",
        estimate_d: bool = False,
    ) -> ARFIMAResults:
        """Fit the ARFIMA model.

        Parameters
        ----------
        endog : array-like
            Time series data.
        method : str
            Estimation method. 'css' for Conditional Sum of Squares (default),
            'mle' for approximate Maximum Likelihood.
        estimate_d : bool
            If True, estimate d jointly with AR/MA parameters. If False, use
            the d provided in the constructor.

        Returns
        -------
        ARFIMAResults
            Fitted model results.
        """
        y = validate_endog(endog)

        if estimate_d:
            return self._fit_estimate_d(y, method)
        else:
            return self._fit_fixed_d(y, self.d_init, method)

    def _fit_fixed_d(
        self,
        y: NDArray[np.float64],
        d: float,
        method: str,
    ) -> ARFIMAResults:
        """Fit ARFIMA with fixed d value."""
        n_obs = len(y)

        # Apply fractional differencing
        w = fractional_diff(y, d)

        # Now fit ARMA(p, q) to the fractionally differenced series w
        ar_params, ma_params, sigma2, resid = self._fit_arma(w, self.p, self.q)

        # Compute fitted values
        fittedvalues = y - resid  # approximate

        # Compute log-likelihood (Gaussian CSS)
        n_eff = n_obs - max(self.p, self.q)
        loglik = -0.5 * n_eff * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(
            resid[max(self.p, self.q) :] ** 2
        ) / sigma2

        # Information criteria
        k = self.p + self.q + 1 + 1  # AR + MA + d + sigma2
        aic = -2.0 * loglik + 2.0 * k
        bic = -2.0 * loglik + k * np.log(n_eff)
        aicc = (
            aic + 2.0 * k * (k + 1) / (n_eff - k - 1)
            if n_eff - k - 1 > 0
            else np.inf
        )

        return ARFIMAResults(
            d=d,
            ar_params=ar_params,
            ma_params=ma_params,
            sigma2=sigma2,
            resid=resid,
            nobs=n_obs,
            loglik=loglik,
            aic=aic,
            bic=bic,
            aicc=aicc,
            order=(self.p, d, self.q),
            endog=y,
            fittedvalues=fittedvalues,
        )

    def _fit_estimate_d(
        self,
        y: NDArray[np.float64],
        method: str,
    ) -> ARFIMAResults:
        """Fit ARFIMA with d estimated jointly."""

        def neg_loglik(params: NDArray[np.float64]) -> float:
            """Negative log-likelihood for optimization."""
            d_val = params[0]
            if not (-0.499 < d_val < 0.499):
                return 1e12

            try:
                w = fractional_diff(y, d_val)
                ar_p = params[1 : 1 + self.p]
                ma_p = params[1 + self.p : 1 + self.p + self.q]

                # Compute residuals from ARMA
                resid = self._compute_arma_residuals(w, ar_p, ma_p)
                n_eff = len(resid)
                if n_eff <= 0:
                    return 1e12

                sigma2 = float(np.mean(resid**2))
                if sigma2 <= 0:
                    return 1e12

                ll = -0.5 * n_eff * np.log(2 * np.pi * sigma2) - 0.5 * n_eff
                return float(-ll)
            except Exception:
                return 1e12

        # Initial parameters
        x0 = np.zeros(1 + self.p + self.q)
        x0[0] = self.d_init

        # Bounds
        bounds: list[tuple[float, float]] = [(-0.499, 0.499)]
        bounds.extend([(-0.99, 0.99)] * self.p)  # AR
        bounds.extend([(-0.99, 0.99)] * self.q)  # MA

        result = optimize.minimize(
            neg_loglik,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-8},
        )

        d_hat = float(result.x[0])
        return self._fit_fixed_d(y, d_hat, method)

    def _fit_arma(
        self,
        w: NDArray[np.float64],
        p: int,
        q: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, NDArray[np.float64]]:
        """Fit ARMA(p,q) to a series via CSS.

        Returns (ar_params, ma_params, sigma2, residuals).
        """
        if p == 0 and q == 0:
            sigma2 = float(np.var(w, ddof=0))
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                sigma2,
                w.copy(),
            )

        def css_objective(params: NDArray[np.float64]) -> float:
            ar_p = params[:p]
            ma_p = params[p : p + q]
            resid = self._compute_arma_residuals(w, ar_p, ma_p)
            return float(np.sum(resid**2))

        x0 = np.zeros(p + q)
        bounds: list[tuple[float, float]] = [(-0.99, 0.99)] * (p + q)

        result = optimize.minimize(
            css_objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        ar_params = result.x[:p].copy()
        ma_params = result.x[p : p + q].copy()
        resid = self._compute_arma_residuals(w, ar_params, ma_params)
        sigma2 = float(np.mean(resid**2))

        return ar_params, ma_params, sigma2, resid

    def _compute_arma_residuals(
        self,
        w: NDArray[np.float64],
        ar_params: NDArray[np.float64],
        ma_params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute residuals from ARMA model given parameters.

        Uses recursive computation:
            eps_t = w_t - sum_i ar_i * w_{t-i} - sum_j ma_j * eps_{t-j}
        """
        n_obs = len(w)
        p = len(ar_params)
        q = len(ma_params)
        resid = np.zeros(n_obs, dtype=np.float64)

        for t in range(n_obs):
            eps_t = w[t]
            for i in range(p):
                if t - 1 - i >= 0:
                    eps_t -= ar_params[i] * w[t - 1 - i]
            for j in range(q):
                if t - 1 - j >= 0:
                    eps_t -= ma_params[j] * resid[t - 1 - j]
            resid[t] = eps_t

        return resid

    def estimate_d(
        self,
        endog: NDArray[np.float64] | list[float],
        method: Literal["gph", "whittle"] = "gph",
        bandwidth_exp: float | None = None,
    ) -> float:
        """Estimate the fractional differencing parameter d.

        Parameters
        ----------
        endog : array-like
            Time series data.
        method : str
            'gph' for GPH log-periodogram regression (default),
            'whittle' for Local Whittle estimator.
        bandwidth_exp : float or None
            Bandwidth exponent. Default depends on method.

        Returns
        -------
        float
            Estimated d value.
        """
        y = validate_endog(endog)

        if method == "gph":
            bw = bandwidth_exp if bandwidth_exp is not None else 0.5
            d_hat, _ = estimate_d_gph(y, bandwidth_exp=bw)
        elif method == "whittle":
            bw = bandwidth_exp if bandwidth_exp is not None else 0.65
            d_hat, _ = estimate_d_local_whittle(y, bandwidth_exp=bw)
        else:
            msg = f"Unknown method '{method}'. Use 'gph' or 'whittle'."
            raise ValueError(msg)

        return d_hat


def simulate_arfima(
    n: int,
    d: float,
    ar: NDArray[np.float64] | None = None,
    ma: NDArray[np.float64] | None = None,
    sigma: float = 1.0,
    burnin: int = 500,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Simulate an ARFIMA(p,d,q) process.

    Parameters
    ----------
    n : int
        Length of the output series.
    d : float
        Fractional differencing parameter.
    ar : array-like or None
        AR coefficients [phi_1, ..., phi_p].
    ma : array-like or None
        MA coefficients [theta_1, ..., theta_q].
    sigma : float
        Innovation standard deviation.
    burnin : int
        Number of burn-in observations to discard.
    rng : numpy random Generator or None
        Random number generator. If None, uses default_rng().

    Returns
    -------
    NDArray[np.float64]
        Simulated ARFIMA series of length n.
    """
    if rng is None:
        rng = np.random.default_rng()
    if ar is None:
        ar = np.array([], dtype=np.float64)
    if ma is None:
        ma = np.array([], dtype=np.float64)

    total = n + burnin
    eps = rng.normal(0, sigma, total)

    # Step 1: Generate ARMA innovations
    p = len(ar)
    q = len(ma)
    w = np.zeros(total, dtype=np.float64)
    for t in range(total):
        w[t] = eps[t]
        for i in range(p):
            if t - 1 - i >= 0:
                w[t] += ar[i] * w[t - 1 - i]
        for j in range(q):
            if t - 1 - j >= 0:
                w[t] += ma[j] * eps[t - 1 - j]

    # Step 2: Apply inverse fractional differencing (cumulate with d)
    # (1-L)^{-d} w_t = y_t
    # y_t = sum_{k=0}^{t} psi_k * w_{t-k}
    # where psi_k are coefficients of (1-L)^{-d}
    pi_inv = fractional_diff_coefficients(-d, total)
    y = np.zeros(total, dtype=np.float64)
    for t in range(total):
        n_terms = min(t + 1, total)
        for k in range(n_terms):
            y[t] += pi_inv[k] * w[t - k]

    return y[burnin:]
