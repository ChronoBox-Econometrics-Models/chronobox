"""ETS - Exponential Smoothing State Space Models.

Implements all 30 ETS(Error, Trend, Seasonal) model combinations using the
state-space innovation form.

References
----------
- Hyndman, R.J., Koehler, A.B., Ord, J.K. & Snyder, R.D. (2008).
  Forecasting with Exponential Smoothing: The State Space Approach. Springer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from chronobox.utils.validation import validate_endog

# Valid component options
_VALID_ERRORS = ("A", "M")
_VALID_TRENDS = ("N", "A", "Ad", "M", "Md")
_VALID_SEASONALS = ("N", "A", "M")


def _validate_ets_components(
    error: str, trend: str, seasonal: str, damped: bool
) -> tuple[str, str, str]:
    """Validate and normalize ETS component strings.

    Returns the canonical (error, trend, seasonal) tuple.
    If damped=True, appends 'd' to trend if not already damped.
    """
    error = error.upper()
    if error not in _VALID_ERRORS:
        msg = f"error must be one of {_VALID_ERRORS}, got '{error}'"
        raise ValueError(msg)

    trend = trend.upper() if len(trend) == 1 else trend[0].upper() + trend[1:]
    if damped and trend in ("A", "M"):
        trend = trend + "d"
    if trend not in _VALID_TRENDS:
        msg = f"trend must be one of {_VALID_TRENDS}, got '{trend}'"
        raise ValueError(msg)

    seasonal = seasonal.upper()
    if seasonal not in _VALID_SEASONALS:
        msg = f"seasonal must be one of {_VALID_SEASONALS}, got '{seasonal}'"
        raise ValueError(msg)

    return error, trend, seasonal


@dataclass
class ETSResults:
    """Results from fitting an ETS model.

    Attributes
    ----------
    error : str
        Error type ('A' or 'M').
    trend : str
        Trend type ('N', 'A', 'Ad', 'M', 'Md').
    seasonal : str
        Seasonal type ('N', 'A', 'M').
    seasonal_period : int
        Seasonal period (m). 1 if no seasonality.
    alpha : float
        Level smoothing parameter.
    beta : float | None
        Trend smoothing parameter. None if trend='N'.
    gamma : float | None
        Seasonal smoothing parameter. None if seasonal='N'.
    phi : float | None
        Damping parameter. None if not damped.
    l0 : float
        Initial level.
    b0 : float | None
        Initial trend. None if trend='N'.
    s0 : NDArray[np.float64] | None
        Initial seasonal components. None if seasonal='N'.
    sigma2 : float
        Estimated innovation variance.
    resid : NDArray[np.float64]
        Innovation residuals.
    states : NDArray[np.float64]
        State matrix (T+1 x num_states).
    nobs : int
        Number of observations.
    loglik : float
        Maximized log-likelihood.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    aicc : float
        Corrected AIC.
    nparams : int
        Number of estimated parameters.
    endog : NDArray[np.float64]
        Original series.
    fittedvalues : NDArray[np.float64]
        One-step-ahead fitted values.
    """

    error: str
    trend: str
    seasonal: str
    seasonal_period: int
    alpha: float
    beta: float | None
    gamma: float | None
    phi: float | None
    l0: float
    b0: float | None
    s0: NDArray[np.float64] | None
    sigma2: float
    resid: NDArray[np.float64]
    states: NDArray[np.float64]
    nobs: int
    loglik: float
    aic: float
    bic: float
    aicc: float
    nparams: int
    endog: NDArray[np.float64]
    fittedvalues: NDArray[np.float64]

    def summary(self) -> str:
        """Return a formatted summary of the ETS results."""
        model_str = f"ETS({self.error},{self.trend},{self.seasonal})"
        lines = [
            "=" * 60,
            f"{'ETS Model Results':^60}",
            "=" * 60,
            f"Model:              {model_str}",
            f"Seasonal Period:    {self.seasonal_period}",
            f"No. Observations:   {self.nobs}",
            f"Log-Likelihood:     {self.loglik:.4f}",
            f"AIC:                {self.aic:.4f}",
            f"BIC:                {self.bic:.4f}",
            f"AICc:               {self.aicc:.4f}",
            f"Sigma2:             {self.sigma2:.6f}",
            "-" * 60,
            f"{'Smoothing Parameters':^60}",
            "-" * 60,
            f"  alpha = {self.alpha:.6f}",
        ]
        if self.beta is not None:
            lines.append(f"  beta  = {self.beta:.6f}")
        if self.gamma is not None:
            lines.append(f"  gamma = {self.gamma:.6f}")
        if self.phi is not None:
            lines.append(f"  phi   = {self.phi:.6f}")
        lines.extend([
            "-" * 60,
            f"{'Initial States':^60}",
            "-" * 60,
            f"  l0 = {self.l0:.4f}",
        ])
        if self.b0 is not None:
            lines.append(f"  b0 = {self.b0:.4f}")
        if self.s0 is not None:
            for i, s in enumerate(self.s0):
                lines.append(f"  s[{i}] = {s:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def forecast(self, steps: int = 1) -> NDArray[np.float64]:
        """Forecast future values.

        Parameters
        ----------
        steps : int
            Number of steps ahead.

        Returns
        -------
        NDArray[np.float64]
            Forecasted values.
        """
        return _ets_forecast(
            error=self.error,
            trend=self.trend,
            seasonal=self.seasonal,
            m=self.seasonal_period,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            phi=self.phi,
            states=self.states,
            steps=steps,
        )


def _ets_forecast(
    error: str,
    trend: str,
    seasonal: str,
    m: int,
    alpha: float,
    beta: float | None,
    gamma: float | None,
    phi: float | None,
    states: NDArray[np.float64],
    steps: int,
) -> NDArray[np.float64]:
    """Generate multi-step-ahead forecasts from fitted ETS states."""
    forecasts = np.empty(steps, dtype=np.float64)

    # Get final states
    n_rows = states.shape[0] - 1  # states has T+1 rows
    l_t = states[n_rows, 0]
    b_t = states[n_rows, 1] if trend != "N" else None
    s_vec = states[n_rows - m + 1 : n_rows + 1, -1].copy() if seasonal != "N" else None

    is_damped = trend in ("Ad", "Md")
    phi_val = phi if is_damped and phi is not None else 1.0

    for h in range(1, steps + 1):
        # Compute phi cumulative for damped trend
        phi_h = sum(phi_val**i for i in range(1, h + 1)) if is_damped else float(h)

        # Get seasonal index
        if seasonal != "N" and s_vec is not None:
            s_idx = (h - 1) % m
            s_h = s_vec[s_idx]
        else:
            s_h = 0.0

        # Compute point forecast
        if trend == "N":
            mu = l_t
        elif trend in ("A", "Ad"):
            mu = l_t + phi_h * b_t if b_t is not None else l_t
        elif trend in ("M", "Md"):
            mu = l_t * (b_t**phi_h) if b_t is not None else l_t
        else:
            mu = l_t

        if seasonal == "A":
            mu = mu + s_h
        elif seasonal == "M":
            mu = mu * s_h

        forecasts[h - 1] = mu

    return forecasts


def _update_states(
    l_t: float,
    b_t: float | None,
    s_tm: float,
    eps_t: float,
    alpha: float,
    beta: float | None,
    gamma: float | None,
    phi_val: float,
    error: str,
    trend: str,
    seasonal: str,
    is_mult_error: bool,
    is_mult_trend: bool,
    is_mult_seasonal: bool,
    has_trend: bool,
    has_seasonal: bool,
    mu_t: float,
    level_trend: float,
) -> tuple[float, float | None, float]:
    """Update ETS states for one time step.

    Returns (l_new, b_new, s_new).
    """
    beta_val = beta if beta is not None else 0.0
    gamma_val = gamma if gamma is not None else 0.0

    # --- Additive Error ---
    if not is_mult_error:
        # Level update
        if is_mult_seasonal:
            l_new = level_trend + alpha * eps_t / s_tm if abs(s_tm) > 1e-300 else level_trend
        else:
            l_new = level_trend + alpha * eps_t

        # Trend update
        b_new: float | None = None
        if has_trend and b_t is not None:
            if is_mult_trend:
                b_new = (
                    b_t**phi_val + beta_val * eps_t / l_t
                    if abs(l_t) > 1e-300
                    else b_t**phi_val
                )
            else:
                b_new = phi_val * b_t + beta_val * eps_t

        # Seasonal update
        if has_seasonal:
            if is_mult_seasonal:
                s_new = (
                    s_tm + gamma_val * eps_t / level_trend
                    if abs(level_trend) > 1e-300
                    else s_tm
                )
            else:
                s_new = s_tm + gamma_val * eps_t
        else:
            s_new = 0.0

    # --- Multiplicative Error ---
    else:
        # Level update
        if has_trend and not is_mult_trend and has_seasonal and not is_mult_seasonal:
            # M,A,A or M,Ad,A: additive trend + additive seasonal
            l_new = level_trend + alpha * level_trend * eps_t
        else:
            l_new = level_trend * (1 + alpha * eps_t)

        # Trend update
        b_new = None
        if has_trend and b_t is not None:
            if is_mult_trend:
                b_new = b_t**phi_val * (1 + beta_val * eps_t)
            else:
                b_new = phi_val * b_t + beta_val * level_trend * eps_t

        # Seasonal update
        if has_seasonal:
            if is_mult_seasonal:
                s_new = s_tm * (1 + gamma_val * eps_t)
            else:
                s_new = s_tm + gamma_val * level_trend * eps_t
        else:
            s_new = 0.0

    return l_new, b_new, s_new


def _ets_filter(
    y: NDArray[np.float64],
    error: str,
    trend: str,
    seasonal: str,
    m: int,
    alpha: float,
    beta: float | None,
    gamma: float | None,
    phi: float | None,
    l0: float,
    b0: float | None,
    s0: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Run the ETS state-space filter.

    Returns (resid, fitted, states, mu_values).
    states: (T+1, num_states) where columns are [l, b, s] depending on model.
    """
    n = len(y)
    is_damped = trend in ("Ad", "Md")
    phi_val = phi if is_damped and phi is not None else 1.0

    has_trend = trend != "N"
    has_seasonal = seasonal != "N"
    is_mult_trend = trend in ("M", "Md")
    is_mult_seasonal = seasonal == "M"
    is_mult_error = error == "M"

    # Number of state columns: l, [b], [s]
    n_cols = 1 + (1 if has_trend else 0) + (1 if has_seasonal else 0)

    states = np.zeros((n + 1, n_cols), dtype=np.float64)
    resid = np.zeros(n, dtype=np.float64)
    fitted = np.zeros(n, dtype=np.float64)
    mu_values = np.zeros(n, dtype=np.float64)

    # Initialize states
    states[0, 0] = l0
    col_b = 1 if has_trend else -1
    col_s = (2 if has_trend else 1) if has_seasonal else -1

    if has_trend and b0 is not None:
        states[0, col_b] = b0

    # Seasonal state stored separately for easy cycling
    if has_seasonal and s0 is not None:
        s_state = np.zeros(n + m, dtype=np.float64)
        s_state[:m] = s0
    else:
        s_state = np.zeros(1, dtype=np.float64)

    l_t = l0
    b_t = b0 if has_trend else None

    for t in range(n):
        # Get seasonal component
        s_tm = s_state[t] if has_seasonal else (1.0 if is_mult_seasonal else 0.0)

        # Compute mu_t (one-step-ahead prediction)
        if is_mult_trend:
            level_trend = l_t * (b_t**phi_val) if b_t is not None else l_t
        else:
            level_trend = (l_t + phi_val * b_t) if (has_trend and b_t is not None) else l_t

        if is_mult_seasonal:
            mu_t = level_trend * s_tm
        elif has_seasonal:
            mu_t = level_trend + s_tm
        else:
            mu_t = level_trend

        mu_values[t] = mu_t
        fitted[t] = mu_t

        # Compute innovation
        if is_mult_error:
            eps_t = (y[t] - mu_t) / mu_t if abs(mu_t) > 1e-300 else 0.0
        else:
            eps_t = y[t] - mu_t

        resid[t] = eps_t

        # Update states
        l_new, b_new, s_new = _update_states(
            l_t=l_t,
            b_t=b_t,
            s_tm=s_tm,
            eps_t=eps_t,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            phi_val=phi_val,
            error=error,
            trend=trend,
            seasonal=seasonal,
            is_mult_error=is_mult_error,
            is_mult_trend=is_mult_trend,
            is_mult_seasonal=is_mult_seasonal,
            has_trend=has_trend,
            has_seasonal=has_seasonal,
            mu_t=mu_t,
            level_trend=level_trend,
        )

        l_t = l_new
        if has_trend:
            b_t = b_new

        # Store states
        states[t + 1, 0] = l_t
        if has_trend and b_t is not None:
            states[t + 1, col_b] = b_t
        if has_seasonal:
            s_state[t + m] = s_new
            if col_s >= 0:
                states[t + 1, col_s] = s_new

    return resid, fitted, states, mu_values


def _ets_loglik(
    y: NDArray[np.float64],
    error: str,
    trend: str,
    seasonal: str,
    m: int,
    alpha: float,
    beta: float | None,
    gamma: float | None,
    phi: float | None,
    l0: float,
    b0: float | None,
    s0: NDArray[np.float64] | None,
) -> float:
    """Compute the ETS log-likelihood.

    For additive error:
        logL = -T/2 * log(2*pi*sigma2) - T/2

    For multiplicative error:
        logL = -T/2 * log(2*pi*sigma2) - T/2 - sum(log|mu_t|)
    """
    resid, _fitted, _states, mu_values = _ets_filter(
        y, error, trend, seasonal, m, alpha, beta, gamma, phi, l0, b0, s0
    )

    n = len(y)
    sigma2 = float(np.mean(resid**2))

    if sigma2 <= 1e-300:
        return -1e20

    if error == "A":
        loglik = -0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0)
    else:
        # Multiplicative error: includes Jacobian
        log_mu = np.sum(np.log(np.abs(mu_values) + 1e-300))
        loglik = -0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0) - log_mu

    if not np.isfinite(loglik):
        return -1e20

    return float(loglik)


def _initialize_ets_states(
    y: NDArray[np.float64],
    trend: str,
    seasonal: str,
    m: int,
) -> tuple[float, float | None, NDArray[np.float64] | None]:
    """Heuristic initialization of ETS states.

    Uses the first few periods to estimate initial level, trend, and seasonal.

    Returns (l0, b0, s0).
    """
    n = len(y)
    is_mult_seasonal = seasonal == "M"
    has_trend = trend != "N"
    has_seasonal = seasonal != "N"

    if has_seasonal and m > 1 and n >= 2 * m:
        first_cycle = y[:m]
        second_cycle = y[m : 2 * m]

        l0 = float(np.mean(first_cycle))

        b0 = float(np.mean(second_cycle) - np.mean(first_cycle)) / m if has_trend else None

        if is_mult_seasonal:
            s0 = first_cycle / l0 if abs(l0) > 1e-300 else np.ones(m, dtype=np.float64)
            # Normalize so sum = m (mean = 1)
            s0 = s0 * (m / np.sum(s0))
        else:
            s0 = first_cycle - l0
            # Normalize so sum = 0
            s0 = s0 - np.mean(s0)

    elif has_seasonal and m > 1:
        l0 = float(np.mean(y[:m]))
        b0 = 0.0 if has_trend else None
        s0 = np.zeros(m, dtype=np.float64)
    else:
        l0 = float(y[0])
        b0 = (float(y[1] - y[0]) if n >= 2 else 0.0) if has_trend else None
        s0 = None

    return l0, b0, s0


class ETS:
    """ETS - Exponential Smoothing State Space Model.

    Supports all 30 combinations of Error(A,M) x Trend(N,A,Ad,M,Md) x Seasonal(N,A,M).

    Parameters
    ----------
    error : str
        Error type: 'A' (additive) or 'M' (multiplicative).
    trend : str
        Trend type: 'N' (none), 'A' (additive), 'M' (multiplicative).
    seasonal : str
        Seasonal type: 'N' (none), 'A' (additive), 'M' (multiplicative).
    seasonal_period : int or None
        Seasonal period. Required if seasonal != 'N'.
    damped : bool
        Whether to use damped trend.

    Examples
    --------
    >>> import numpy as np
    >>> from chronobox.models.ets import ETS
    >>> y = np.array([10.0, 12.0, 15.0, 11.0, 14.0, 18.0, 13.0, 16.0])
    >>> model = ETS(error='A', trend='A', seasonal='N')
    >>> results = model.fit(y)
    >>> forecast = results.forecast(steps=4)
    """

    def __init__(
        self,
        error: str = "A",
        trend: str = "A",
        seasonal: str = "N",
        seasonal_period: int | None = None,
        damped: bool = False,
    ) -> None:
        self.error, self.trend, self.seasonal = _validate_ets_components(
            error, trend, seasonal, damped
        )
        self.damped = self.trend in ("Ad", "Md")

        if self.seasonal != "N":
            if seasonal_period is None or seasonal_period < 2:
                msg = "seasonal_period must be >= 2 when seasonal != 'N'"
                raise ValueError(msg)
            self.seasonal_period = seasonal_period
        else:
            self.seasonal_period = 1

    def fit(
        self,
        endog: NDArray[np.float64] | list[float],
        optimized: bool = True,
    ) -> ETSResults:
        """Fit the ETS model to data.

        Parameters
        ----------
        endog : array-like
            Time series data.
        optimized : bool
            If True, optimize parameters via L-BFGS-B. If False, use
            heuristic initial values.

        Returns
        -------
        ETSResults
            Fitted model results.
        """
        y = validate_endog(endog)
        n = len(y)
        m = self.seasonal_period

        if self.error == "M" and np.any(y <= 0):
            msg = "Multiplicative error requires all positive data"
            raise ValueError(msg)
        if self.seasonal == "M" and np.any(y <= 0):
            msg = "Multiplicative seasonality requires all positive data"
            raise ValueError(msg)
        if self.seasonal != "N" and n < 2 * m:
            msg = f"Need at least 2 full seasonal cycles ({2 * m} obs), got {n}"
            raise ValueError(msg)

        has_trend = self.trend != "N"
        has_seasonal = self.seasonal != "N"

        l0_init, b0_init, s0_init = _initialize_ets_states(
            y, self.trend, self.seasonal, m
        )

        if not optimized:
            alpha = 0.1
            beta = 0.01 if has_trend else None
            gamma = 0.01 if has_seasonal else None
            phi = 0.98 if self.damped else None
            return self._build_results(
                y, alpha, beta, gamma, phi, l0_init, b0_init, s0_init
            )

        # --- Optimization ---
        x0, bounds, pack_info = self._pack_params(y, l0_init, b0_init, s0_init)

        def neg_loglik(params: NDArray[np.float64]) -> float:
            alpha_v, beta_v, gamma_v, phi_v, l0_v, b0_v, s0_v = self._unpack_params(
                params, pack_info, m
            )
            try:
                ll = _ets_loglik(
                    y,
                    self.error,
                    self.trend,
                    self.seasonal,
                    m,
                    alpha_v,
                    beta_v,
                    gamma_v,
                    phi_v,
                    l0_v,
                    b0_v,
                    s0_v,
                )
                return -ll
            except Exception:
                return 1e20

        result = optimize.minimize(
            neg_loglik,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        alpha, beta, gamma, phi, l0, b0, s0 = self._unpack_params(
            result.x, pack_info, m
        )

        return self._build_results(y, alpha, beta, gamma, phi, l0, b0, s0)

    def _pack_params(
        self,
        y: NDArray[np.float64],
        l0_init: float,
        b0_init: float | None,
        s0_init: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.float64], list[tuple[float, float]], dict[str, int]]:
        """Pack parameters into a flat array for optimization."""
        has_trend = self.trend != "N"
        has_seasonal = self.seasonal != "N"
        m = self.seasonal_period

        params: list[float] = []
        bounds: list[tuple[float, float]] = []
        pack_info: dict[str, int] = {}

        # alpha
        pack_info["alpha"] = len(params)
        params.append(0.1)
        bounds.append((1e-4, 0.9999))

        # beta
        if has_trend:
            pack_info["beta"] = len(params)
            params.append(0.01)
            bounds.append((1e-4, 0.5))

        # gamma
        if has_seasonal:
            pack_info["gamma"] = len(params)
            params.append(0.01)
            bounds.append((1e-4, 0.5))

        # phi
        if self.damped:
            pack_info["phi"] = len(params)
            params.append(0.98)
            bounds.append((0.8, 0.98))

        # l0
        pack_info["l0"] = len(params)
        params.append(l0_init)
        y_range = float(np.max(y) - np.min(y))
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        bounds.append((y_min - y_range, y_max + y_range))

        # b0
        if has_trend:
            pack_info["b0"] = len(params)
            params.append(b0_init if b0_init is not None else 0.0)
            bounds.append((-y_range, y_range))

        # s0 (m-1 free parameters)
        if has_seasonal and s0_init is not None:
            pack_info["s0_start"] = len(params)
            pack_info["s0_len"] = m - 1
            for i in range(m - 1):
                params.append(float(s0_init[i]))
                if self.seasonal == "M":
                    bounds.append((0.01, 5.0))
                else:
                    bounds.append((-y_range * 2, y_range * 2))

        return np.array(params), bounds, pack_info

    def _unpack_params(
        self,
        params: NDArray[np.float64],
        pack_info: dict[str, int],
        m: int,
    ) -> tuple[
        float,
        float | None,
        float | None,
        float | None,
        float,
        float | None,
        NDArray[np.float64] | None,
    ]:
        """Unpack flat parameter array back to named parameters."""
        alpha = float(params[pack_info["alpha"]])

        beta: float | None = None
        if "beta" in pack_info:
            beta = float(params[pack_info["beta"]])
            beta = min(beta, alpha - 1e-6)

        gamma: float | None = None
        if "gamma" in pack_info:
            gamma = float(params[pack_info["gamma"]])
            gamma = min(gamma, 1.0 - alpha - 1e-6)

        phi: float | None = None
        if "phi" in pack_info:
            phi = float(params[pack_info["phi"]])

        l0 = float(params[pack_info["l0"]])

        b0: float | None = None
        if "b0" in pack_info:
            b0 = float(params[pack_info["b0"]])

        s0: NDArray[np.float64] | None = None
        if "s0_start" in pack_info:
            s0_free = params[pack_info["s0_start"] : pack_info["s0_start"] + pack_info["s0_len"]]
            s0 = np.empty(m, dtype=np.float64)
            s0[: m - 1] = s0_free
            if self.seasonal == "M":
                s0[m - 1] = m - np.sum(s0_free)
                if s0[m - 1] <= 0:
                    s0[m - 1] = 0.01
            else:
                s0[m - 1] = -np.sum(s0_free)

        return alpha, beta, gamma, phi, l0, b0, s0

    def _build_results(
        self,
        y: NDArray[np.float64],
        alpha: float,
        beta: float | None,
        gamma: float | None,
        phi: float | None,
        l0: float,
        b0: float | None,
        s0: NDArray[np.float64] | None,
    ) -> ETSResults:
        """Build ETSResults from fitted parameters."""
        m = self.seasonal_period
        resid, fitted, states, _mu_values = _ets_filter(
            y,
            self.error,
            self.trend,
            self.seasonal,
            m,
            alpha,
            beta,
            gamma,
            phi,
            l0,
            b0,
            s0,
        )

        n = len(y)
        loglik = _ets_loglik(
            y,
            self.error,
            self.trend,
            self.seasonal,
            m,
            alpha,
            beta,
            gamma,
            phi,
            l0,
            b0,
            s0,
        )

        # Count parameters
        nparams = 1  # alpha
        if beta is not None:
            nparams += 1
        if gamma is not None:
            nparams += 1
        if phi is not None:
            nparams += 1
        nparams += 1  # l0
        if b0 is not None:
            nparams += 1
        if s0 is not None:
            nparams += len(s0) - 1  # m-1 free seasonal params
        nparams += 1  # sigma2

        sigma2 = float(np.mean(resid**2))
        aic = -2.0 * loglik + 2.0 * nparams
        bic = -2.0 * loglik + nparams * np.log(n)
        if n - nparams - 1 > 0:
            aicc = aic + 2.0 * nparams * (nparams + 1) / (n - nparams - 1)
        else:
            aicc = np.inf

        return ETSResults(
            error=self.error,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_period=m,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            phi=phi,
            l0=l0,
            b0=b0,
            s0=s0,
            sigma2=sigma2,
            resid=resid,
            states=states,
            nobs=n,
            loglik=loglik,
            aic=aic,
            bic=bic,
            aicc=float(aicc),
            nparams=nparams,
            endog=y,
            fittedvalues=fitted,
        )

    def simulate(
        self,
        nsimulations: int,
        alpha: float = 0.1,
        beta: float | None = None,
        gamma: float | None = None,
        phi: float | None = None,
        l0: float = 100.0,
        b0: float | None = None,
        s0: NDArray[np.float64] | None = None,
        sigma: float = 1.0,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """Simulate from an ETS model with given parameters.

        Parameters
        ----------
        nsimulations : int
            Length of simulated series.
        alpha, beta, gamma, phi : float
            Smoothing parameters.
        l0, b0 : float
            Initial level and trend.
        s0 : array-like or None
            Initial seasonal components.
        sigma : float
            Innovation standard deviation.
        seed : int or None
            Random seed.

        Returns
        -------
        NDArray[np.float64]
            Simulated series.
        """
        rng = np.random.default_rng(seed)
        m = self.seasonal_period
        has_trend = self.trend != "N"
        has_seasonal = self.seasonal != "N"
        is_mult_error = self.error == "M"
        is_mult_trend = self.trend in ("M", "Md")
        is_mult_seasonal = self.seasonal == "M"
        is_damped = self.trend in ("Ad", "Md")
        phi_val = phi if is_damped and phi is not None else 1.0

        y = np.zeros(nsimulations, dtype=np.float64)
        eps = rng.normal(0, sigma, nsimulations)

        l_t = l0
        b_t = b0 if has_trend else None

        if has_seasonal and s0 is not None:
            s_state = np.zeros(nsimulations + m, dtype=np.float64)
            s_state[:m] = s0
        else:
            s_state = np.zeros(1)

        for t in range(nsimulations):
            s_tm = s_state[t] if has_seasonal else (1.0 if is_mult_seasonal else 0.0)

            if is_mult_trend:
                level_trend = l_t * (b_t**phi_val) if b_t is not None else l_t
            else:
                level_trend = (
                    (l_t + phi_val * b_t) if (has_trend and b_t is not None) else l_t
                )

            if is_mult_seasonal:
                mu_t = level_trend * s_tm
            elif has_seasonal:
                mu_t = level_trend + s_tm
            else:
                mu_t = level_trend

            if is_mult_error:
                y[t] = mu_t * (1 + eps[t])
            else:
                y[t] = mu_t + eps[t]

            eps_t = eps[t]

            l_new, b_new, s_new = _update_states(
                l_t,
                b_t,
                s_tm,
                eps_t,
                alpha,
                beta,
                gamma,
                phi_val,
                self.error,
                self.trend,
                self.seasonal,
                is_mult_error,
                is_mult_trend,
                is_mult_seasonal,
                has_trend,
                has_seasonal,
                mu_t,
                level_trend,
            )
            l_t = l_new
            if has_trend:
                b_t = b_new
            if has_seasonal:
                s_state[t + m] = s_new

        return y


def get_all_ets_models() -> list[tuple[str, str, str]]:
    """Return all 30 valid ETS model specifications.

    Returns
    -------
    list of (error, trend, seasonal) tuples
    """
    models = []
    for e in _VALID_ERRORS:
        for t in _VALID_TRENDS:
            for s in _VALID_SEASONALS:
                models.append((e, t, s))
    return models
