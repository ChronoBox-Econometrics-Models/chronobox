"""ARIMA(p,d,q) and SARIMA(p,d,q)(P,D,Q)[s] models."""

# State-space notation uses uppercase variable names (T, Z, R, Q, H, K, P, F)
# following Durbin & Koopman (2012) and Harvey (1989) conventions.

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

from chronobox._logging import get_logger
from chronobox.core.results import TimeSeriesResults
from chronobox.core.transforms import difference, seasonal_difference
from chronobox.utils.array_ops import convolve_polynomials
from chronobox.utils.validation import (
    validate_endog,
    validate_order,
    validate_seasonal_order,
)

logger = get_logger("arima")


def _diffuse_kalman_loglike(
    y: NDArray[np.float64],
    T: NDArray[np.float64],
    Z: NDArray[np.float64],
    R: NDArray[np.float64],
    Q: NDArray[np.float64],
    H: NDArray[np.float64],
    a0: NDArray[np.float64],
    P_star0: NDArray[np.float64],
    P_inf0: NDArray[np.float64],
) -> float:
    """Exact diffuse Kalman filter loglikelihood (Durbin & Koopman 2012).

    Uses the exact diffuse initialization to handle non-stationary states
    in ARIMA models, matching R's approach.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Observed series.
    T, Z, R, Q, H : state-space matrices.
    a0 : ndarray of shape (m,)
        Initial state mean.
    P_star0 : ndarray of shape (m, m)
        Finite part of initial state covariance.
    P_inf0 : ndarray of shape (m, m)
        Diffuse part of initial state covariance.

    Returns
    -------
    float
        Diffuse log-likelihood.
    """
    n = len(y)
    a = a0.copy()
    P_star = P_star0.copy()
    P_inf = P_inf0.copy()
    RQR = R @ Q @ R.T

    loglike = 0.0
    tol = 1e-8

    for t in range(n):
        v_t = float((y[t] - Z @ a)[0])
        F_inf = float((Z @ P_inf @ Z.T)[0, 0])
        F_star = float((Z @ P_star @ Z.T + H)[0, 0])

        if F_inf > tol:
            # Diffuse observation (Durbin & Koopman 2012, Ch. 5)
            loglike += -0.5 * np.log(F_inf)
            # Kalman gains
            K_0 = T @ P_inf @ Z.T / F_inf  # (m, 1)
            K_1 = (T @ P_star @ Z.T / F_inf
                    - T @ P_inf @ Z.T * F_star / (F_inf * F_inf))  # (m, 1)
            # Transition matrices
            L_0 = T - K_0 @ Z  # (m, m)
            L_1 = -K_1 @ Z  # (m, m)
            # State update
            a = T @ a + K_0[:, 0] * v_t
            # Covariance update (D&K eq 5.14)
            P_inf_new = T @ P_inf @ L_0.T
            P_star_new = T @ P_star @ L_0.T + T @ P_inf @ L_1.T + RQR
            P_inf = 0.5 * (P_inf_new + P_inf_new.T)
            P_star = 0.5 * (P_star_new + P_star_new.T)
        else:
            # Non-diffuse observation (standard Kalman filter)
            P_inf[:] = 0.0
            if F_star <= 0:
                return -np.inf
            loglike += -0.5 * (np.log(2 * np.pi) + np.log(F_star)
                               + v_t**2 / F_star)
            K = T @ P_star @ Z.T / F_star  # (m, 1)
            a = T @ a + K[:, 0] * v_t
            P_star = T @ P_star @ T.T + RQR - F_star * (K @ K.T)
            P_star = 0.5 * (P_star + P_star.T)

    return loglike


class ARIMA:
    """ARIMA(p,d,q) with optional seasonal extension SARIMA(p,d,q)(P,D,Q)[s].

    Parameters
    ----------
    order : tuple of (int, int, int)
        (p, d, q) - AR order, differencing order, MA order.
    seasonal_order : tuple of (int, int, int, int)
        (P, D, Q, s) - Seasonal AR, differencing, MA orders, and period.
        Default (0, 0, 0, 0) means no seasonal component.
    trend : str or None
        Trend type: 'n' (none), 'c' (constant), 't' (linear trend),
        'ct' (constant + trend). Default None means 'c' if d+D=0, else 'n'.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 0),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: str | None = None,
    ) -> None:
        self.order = validate_order(order)
        self.seasonal_order = validate_seasonal_order(seasonal_order)
        self.p, self.d, self.q = self.order
        self.P, self.D, self.Q, self.s = self.seasonal_order

        if trend is None:
            self.trend = "n" if (self.d + self.D) > 0 else "c"
        else:
            self.trend = trend

        self._endog: NDArray[np.float64] | None = None
        self._endog_diff: NDArray[np.float64] | None = None
        self._results: TimeSeriesResults | None = None

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        name = f"ARIMA({self.p},{self.d},{self.q})"
        if self.s > 0 and (self.P > 0 or self.D > 0 or self.Q > 0):
            name += f"({self.P},{self.D},{self.Q})[{self.s}]"
        return name

    @property
    def _n_params(self) -> int:
        """Total number of parameters."""
        n = self.p + self.q + self.P + self.Q + 1  # +1 for sigma2
        if self.trend in ("c", "t"):
            n += 1
        elif self.trend == "ct":
            n += 2
        return n

    def _param_names(self) -> list[str]:
        """Generate parameter names."""
        names: list[str] = []
        if "c" in self.trend:
            names.append("const")
        if "t" in self.trend and self.trend != "c":
            names.append("trend")
        for i in range(1, self.p + 1):
            names.append(f"ar.L{i}")
        for i in range(1, self.q + 1):
            names.append(f"ma.L{i}")
        for i in range(1, self.P + 1):
            names.append(f"ar.S.L{i * self.s}")
        for i in range(1, self.Q + 1):
            names.append(f"ma.S.L{i * self.s}")
        names.append("sigma2")
        return names

    def _apply_differencing(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply regular and seasonal differencing."""
        result = y.copy()
        if self.D > 0 and self.s > 0:
            for _ in range(self.D):
                result = seasonal_difference(result, self.s)
        if self.d > 0:
            result = difference(result, self.d)
        return result

    def _unpack_params(
        self, params: NDArray[np.float64]
    ) -> dict[str, Any]:
        """Unpack parameter vector into named components."""
        idx = 0
        result: dict[str, Any] = {}

        # Trend
        if self.trend == "c":
            result["const"] = params[idx]
            idx += 1
        elif self.trend == "t":
            result["trend"] = params[idx]
            idx += 1
        elif self.trend == "ct":
            result["const"] = params[idx]
            result["trend"] = params[idx + 1]
            idx += 2

        # AR coefficients
        result["ar"] = params[idx : idx + self.p]
        idx += self.p

        # MA coefficients
        result["ma"] = params[idx : idx + self.q]
        idx += self.q

        # Seasonal AR
        result["sar"] = params[idx : idx + self.P]
        idx += self.P

        # Seasonal MA
        result["sma"] = params[idx : idx + self.Q]
        idx += self.Q

        # sigma2
        result["sigma2"] = params[idx]
        return result

    def _build_expanded_polynomials(
        self, components: dict[str, Any]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Build expanded AR and MA polynomials from components."""
        ar = components["ar"]
        ma = components["ma"]
        sar = components["sar"]
        sma = components["sma"]

        # AR polynomial
        ar_poly = np.array([1.0], dtype=np.float64)
        if self.p > 0:
            ar_full = np.zeros(self.p + 1, dtype=np.float64)
            ar_full[0] = 1.0
            ar_full[1:] = -ar
            ar_poly = ar_full

        if self.P > 0 and self.s > 0:
            sar_full = np.zeros(self.P * self.s + 1, dtype=np.float64)
            sar_full[0] = 1.0
            for i in range(self.P):
                sar_full[(i + 1) * self.s] = -sar[i]
            ar_poly = convolve_polynomials(ar_poly, sar_full)

        # MA polynomial
        ma_poly = np.array([1.0], dtype=np.float64)
        if self.q > 0:
            ma_full = np.zeros(self.q + 1, dtype=np.float64)
            ma_full[0] = 1.0
            ma_full[1:] = ma
            ma_poly = ma_full

        if self.Q > 0 and self.s > 0:
            sma_full = np.zeros(self.Q * self.s + 1, dtype=np.float64)
            sma_full[0] = 1.0
            for i in range(self.Q):
                sma_full[(i + 1) * self.s] = sma[i]
            ma_poly = convolve_polynomials(ma_poly, sma_full)

        return ar_poly, ma_poly

    def _adjust_trend(
        self, y: NDArray[np.float64], components: dict[str, Any]
    ) -> NDArray[np.float64]:
        """Subtract trend from series."""
        y_adj = y.copy()
        n = len(y)
        if "const" in components:
            y_adj -= components["const"]
        if "trend" in components:
            y_adj -= components["trend"] * np.arange(n, dtype=np.float64)
        return y_adj

    def _css_loglike(
        self, params: NDArray[np.float64], y: NDArray[np.float64]
    ) -> float:
        """Conditional sum of squares log-likelihood.

        Runs on the original (undifferenced) data with the full AR polynomial
        (including differencing), matching R's CSS approach. The first ncond
        residuals are skipped (ncond = max(full_ar_order, ma_order)).
        """
        components = self._unpack_params(params)
        sigma2 = components["sigma2"]

        if sigma2 <= 0:
            return -np.inf

        if self._endog is None:
            return -np.inf

        ar_poly, ma_poly = self._build_expanded_polynomials(components)
        diff_poly = self._build_differencing_polynomial()
        full_ar_poly = convolve_polynomials(ar_poly, diff_poly)

        # Use original data
        y_orig = self._endog.copy()
        n = len(y_orig)

        if "const" in components:
            y_orig = y_orig - components["const"]
        if "trend" in components:
            y_orig = y_orig - components["trend"] * np.arange(n, dtype=np.float64)

        p_eff = len(full_ar_poly) - 1
        q_eff = len(ma_poly) - 1
        ncond = max(p_eff, q_eff)

        if ncond >= n:
            return -np.inf

        eps = np.zeros(n, dtype=np.float64)
        for t in range(n):
            val = y_orig[t]
            for j in range(1, p_eff + 1):
                if t - j >= 0:
                    val -= (-full_ar_poly[j]) * y_orig[t - j]
            for j in range(1, q_eff + 1):
                if t - j >= 0:
                    val -= ma_poly[j] * eps[t - j]
            eps[t] = val

        # Skip first ncond residuals (conditioning period)
        css_resid = eps[ncond:]
        n_eff = len(css_resid)

        loglike = -0.5 * n_eff * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(
            css_resid**2
        ) / sigma2
        return float(loglike)

    def _build_differencing_polynomial(self) -> NDArray[np.float64]:
        """Build the differencing polynomial (1-L)^d * (1-L^s)^D."""
        diff_poly = np.array([1.0], dtype=np.float64)
        # Regular differencing: (1 - L)^d
        delta = np.array([1.0, -1.0], dtype=np.float64)
        for _ in range(self.d):
            diff_poly = convolve_polynomials(diff_poly, delta)
        # Seasonal differencing: (1 - L^s)^D
        if self.D > 0 and self.s > 0:
            sdelta = np.zeros(self.s + 1, dtype=np.float64)
            sdelta[0] = 1.0
            sdelta[self.s] = -1.0
            for _ in range(self.D):
                diff_poly = convolve_polynomials(diff_poly, sdelta)
        return diff_poly

    def _build_state_space(
        self, components: dict[str, Any]
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Build state-space matrices for Kalman filter on original data.

        Incorporates differencing into the AR polynomial so the filter
        runs on the original (undifferenced) data.

        Returns (T, Z, R, Q, H, a0, P0).
        """
        ar_poly, ma_poly = self._build_expanded_polynomials(components)
        sigma2 = components["sigma2"]

        # Multiply AR polynomial by differencing polynomial
        diff_poly = self._build_differencing_polynomial()
        full_ar_poly = convolve_polynomials(ar_poly, diff_poly)

        p_eff = len(full_ar_poly) - 1
        q_eff = len(ma_poly) - 1
        m = max(p_eff, q_eff + 1)
        if m == 0:
            m = 1

        # T: companion matrix
        T = np.zeros((m, m), dtype=np.float64)
        for j in range(min(p_eff, m)):
            T[0, j] = -full_ar_poly[j + 1]
        if m > 1:
            T[1:, :-1] += np.eye(m - 1)

        # Z: observation matrix
        Z = np.zeros((1, m), dtype=np.float64)
        for j in range(min(len(ma_poly), m)):
            Z[0, j] = ma_poly[j]

        # R: selection matrix
        R = np.zeros((m, 1), dtype=np.float64)
        R[0, 0] = 1.0

        # Q: state disturbance covariance
        Q_mat = np.array([[sigma2]], dtype=np.float64)

        # H: observation noise
        H = np.zeros((1, 1), dtype=np.float64)

        # Diffuse initialization
        a0 = np.zeros(m, dtype=np.float64)
        P0 = np.eye(m, dtype=np.float64) * 1e6

        return T, Z, R, Q_mat, H, a0, P0

    def _mle_loglike(
        self, params: NDArray[np.float64], y: NDArray[np.float64]
    ) -> float:
        """Exact MLE log-likelihood via Kalman filter on differenced data.

        Uses the stationary ARMA state-space on differenced data with
        Lyapunov initialization for the state covariance.
        """
        components = self._unpack_params(params)
        sigma2 = components["sigma2"]
        if sigma2 <= 0:
            return -np.inf

        ar_poly, ma_poly = self._build_expanded_polynomials(components)

        p_eff = len(ar_poly) - 1
        q_eff = len(ma_poly) - 1
        m = max(p_eff, q_eff + 1)
        if m == 0:
            m = 1

        # Build state-space for ARMA on differenced data (no differencing)
        T_mat = np.zeros((m, m), dtype=np.float64)
        for j in range(min(p_eff, m)):
            T_mat[0, j] = -ar_poly[j + 1]
        if m > 1:
            T_mat[1:, :-1] += np.eye(m - 1)

        Z_mat = np.zeros((1, m), dtype=np.float64)
        for j in range(min(len(ma_poly), m)):
            Z_mat[0, j] = ma_poly[j]

        R_mat = np.zeros((m, 1), dtype=np.float64)
        R_mat[0, 0] = 1.0
        Q_mat = np.array([[sigma2]], dtype=np.float64)
        H_mat = np.zeros((1, 1), dtype=np.float64)

        # Stationary initialization via Lyapunov equation
        from scipy import linalg as sp_linalg

        RQR = R_mat @ Q_mat @ R_mat.T
        try:
            P0 = sp_linalg.solve_discrete_lyapunov(T_mat, RQR)
        except Exception:
            P0 = np.eye(m, dtype=np.float64) * sigma2

        a0 = np.zeros(m, dtype=np.float64)

        y_adj = self._adjust_trend(y, components)

        n = len(y_adj)
        a = a0.copy()
        P = P0.copy()

        loglike = 0.0
        for t in range(n):
            v_t = float((y_adj[t] - Z_mat @ a)[0])
            F_t = float((Z_mat @ P @ Z_mat.T + H_mat)[0, 0])
            if F_t <= 0:
                return -np.inf
            loglike += -0.5 * (np.log(2 * np.pi) + np.log(F_t) + v_t**2 / F_t)
            K = T_mat @ P @ Z_mat.T / F_t
            a = T_mat @ a + K[:, 0] * v_t
            P = T_mat @ P @ T_mat.T + RQR - F_t * (K @ K.T)
            P = 0.5 * (P + P.T)

        return loglike

    def _get_start_params(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get starting parameter values."""
        params: list[float] = []

        if self.trend == "c":
            params.append(float(np.mean(y)))
        elif self.trend == "t":
            params.append(0.0)
        elif self.trend == "ct":
            params.append(float(np.mean(y)))
            params.append(0.0)

        for _ in range(self.p + self.q + self.P + self.Q):
            params.append(0.0)

        params.append(float(np.var(y, ddof=1)))

        return np.array(params, dtype=np.float64)

    def fit(
        self,
        endog: NDArray[np.float64] | list[float],
        method: str = "css-mle",
        maxiter: int = 500,
    ) -> TimeSeriesResults:
        """Fit the ARIMA model.

        Parameters
        ----------
        endog : array-like
            Endogenous time series.
        method : str
            Estimation method: 'css', 'mle', or 'css-mle'.
        maxiter : int
            Maximum number of optimizer iterations.

        Returns
        -------
        TimeSeriesResults
        """
        self._endog = validate_endog(endog)
        self._endog_diff = self._apply_differencing(self._endog)
        y = self._endog_diff
        n_orig = len(self._endog)
        n_eff = len(y)

        start_params = self._get_start_params(y)

        # Reparameterize: optimize log(sigma2) to handle scale difference
        def _to_internal(params: NDArray[np.float64]) -> NDArray[np.float64]:
            """Transform params to optimization space (log sigma2)."""
            internal = params.copy()
            internal[-1] = np.log(max(params[-1], 1e-10))
            return internal

        def _from_internal(internal: NDArray[np.float64]) -> NDArray[np.float64]:
            """Transform from optimization space back to model params."""
            params = internal.copy()
            params[-1] = np.exp(internal[-1])
            return params

        if method in ("css", "css-mle"):
            def neg_css(p_internal: NDArray[np.float64]) -> float:
                return -self._css_loglike(_from_internal(p_internal), y)

            result_css = optimize.minimize(
                neg_css,
                _to_internal(start_params),
                method="L-BFGS-B",
                options={"maxiter": maxiter},
            )
            best_params = _from_internal(result_css.x)

            if method == "css-mle":
                def neg_mle(p_internal: NDArray[np.float64]) -> float:
                    return -self._mle_loglike(_from_internal(p_internal), y)

                result_mle = optimize.minimize(
                    neg_mle,
                    _to_internal(best_params),
                    method="L-BFGS-B",
                    options={"maxiter": maxiter},
                )
                best_params = _from_internal(result_mle.x)
                final_loglike = -result_mle.fun
            else:
                final_loglike = -result_css.fun

        elif method == "mle":
            def neg_mle(p_internal: NDArray[np.float64]) -> float:
                return -self._mle_loglike(_from_internal(p_internal), y)

            result_mle = optimize.minimize(
                neg_mle,
                _to_internal(start_params),
                method="L-BFGS-B",
                options={"maxiter": maxiter},
            )
            best_params = _from_internal(result_mle.x)
            final_loglike = -result_mle.fun
        else:
            msg = f"Unknown method '{method}'. Use 'css', 'mle', or 'css-mle'."
            raise ValueError(msg)

        # Ensure sigma2 is positive
        best_params[-1] = abs(best_params[-1])

        # Compute standard errors via numerical Hessian
        se = self._compute_se(best_params, y, method)

        # Compute residuals
        residuals = self._compute_residuals(best_params, y)

        # Fitted values
        fitted = y - residuals

        self._results = TimeSeriesResults(
            params=best_params,
            param_names=self._param_names(),
            se=se,
            loglike=final_loglike,
            nobs=n_orig,
            nobs_effective=n_eff,
            residuals=residuals,
            fitted_values=fitted,
            model_name=self.model_name,
            forecast_func=self._forecast,
            model=self,
        )
        return self._results

    def _compute_se(
        self,
        params: NDArray[np.float64],
        y: NDArray[np.float64],
        method: str,
    ) -> NDArray[np.float64]:
        """Compute standard errors via numerical Hessian."""
        loglike_func = self._mle_loglike if "mle" in method else self._css_loglike

        def neg_ll(p: NDArray[np.float64]) -> float:
            return -loglike_func(p, y)

        k = len(params)
        eps_h = 1e-5
        hessian = np.zeros((k, k), dtype=np.float64)

        for i in range(k):
            for j in range(i, k):
                p_pp = params.copy()
                p_pm = params.copy()
                p_mp = params.copy()
                p_mm = params.copy()

                p_pp[i] += eps_h
                p_pp[j] += eps_h
                p_pm[i] += eps_h
                p_pm[j] -= eps_h
                p_mp[i] -= eps_h
                p_mp[j] += eps_h
                p_mm[i] -= eps_h
                p_mm[j] -= eps_h

                hessian[i, j] = (
                    neg_ll(p_pp) - neg_ll(p_pm) - neg_ll(p_mp) + neg_ll(p_mm)
                ) / (4 * eps_h**2)
                hessian[j, i] = hessian[i, j]

        try:
            cov = np.linalg.inv(hessian)
            se = np.sqrt(np.abs(np.diag(cov)))
        except np.linalg.LinAlgError:
            se = np.full(k, np.nan)

        return se

    def _compute_residuals(
        self, params: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute residuals on differenced data via CSS recursion."""
        components = self._unpack_params(params)
        ar_poly, ma_poly = self._build_expanded_polynomials(components)

        n = len(y)
        p_eff = len(ar_poly) - 1
        q_eff = len(ma_poly) - 1

        y_adj = self._adjust_trend(y, components)

        eps = np.zeros(n, dtype=np.float64)
        for t in range(n):
            val = y_adj[t]
            for j in range(1, p_eff + 1):
                if t - j >= 0:
                    val -= (-ar_poly[j]) * y_adj[t - j]
            for j in range(1, q_eff + 1):
                if t - j >= 0:
                    val -= ma_poly[j] * eps[t - j]
            eps[t] = val

        return eps

    def _compute_css_residuals_original(
        self, params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute CSS residuals on original data with full AR poly."""
        if self._endog is None:
            msg = "Must call fit() first"
            raise RuntimeError(msg)

        components = self._unpack_params(params)
        ar_poly, ma_poly = self._build_expanded_polynomials(components)
        diff_poly = self._build_differencing_polynomial()
        full_ar_poly = convolve_polynomials(ar_poly, diff_poly)

        y_orig = self._endog.copy()
        n = len(y_orig)
        if "const" in components:
            y_orig = y_orig - components["const"]
        if "trend" in components:
            y_orig = y_orig - components["trend"] * np.arange(n, dtype=np.float64)

        p_eff = len(full_ar_poly) - 1
        q_eff = len(ma_poly) - 1

        eps = np.zeros(n, dtype=np.float64)
        for t in range(n):
            val = y_orig[t]
            for j in range(1, p_eff + 1):
                if t - j >= 0:
                    val -= (-full_ar_poly[j]) * y_orig[t - j]
            for j in range(1, q_eff + 1):
                if t - j >= 0:
                    val -= ma_poly[j] * eps[t - j]
            eps[t] = val

        return eps

    def _forecast(
        self, steps: int, alpha: float = 0.05
    ) -> dict[str, NDArray[np.float64]]:
        """Forecast h steps ahead using psi-weight representation."""
        if self._results is None or self._endog_diff is None:
            msg = "Model must be fit before forecasting"
            raise RuntimeError(msg)

        params = self._results.params
        components = self._unpack_params(params)
        y = self._endog_diff
        sigma2 = components["sigma2"]

        ar_poly, ma_poly = self._build_expanded_polynomials(components)
        p_eff = len(ar_poly) - 1
        q_eff = len(ma_poly) - 1

        # Compute psi weights (MA(inf) representation)
        psi = np.zeros(steps, dtype=np.float64)
        psi[0] = 1.0
        for j in range(1, steps):
            val = 0.0
            if j <= q_eff:
                val += ma_poly[j]
            for i in range(1, min(j, p_eff) + 1):
                val += (-ar_poly[i]) * (psi[j - i] if j - i >= 0 else 0.0)
            psi[j] = val

        # Forecast via recursion on the differenced series
        n = len(y)
        residuals = self._compute_residuals(params, y)
        residuals = np.nan_to_num(residuals, nan=0.0)

        y_ext = np.concatenate([y, np.zeros(steps)])
        eps_ext = np.concatenate([residuals, np.zeros(steps)])

        y_adj = y_ext.copy()
        y_adj[:n] = self._adjust_trend(y, components)

        for h in range(steps):
            t = n + h
            val = 0.0
            if "const" in components:
                val += components["const"]
            if "trend" in components:
                val += components["trend"] * t

            for j in range(1, p_eff + 1):
                if t - j >= 0:
                    val += (-ar_poly[j]) * y_adj[t - j]
            for j in range(1, q_eff + 1):
                if t - j >= 0:
                    val += ma_poly[j] * eps_ext[t - j]
            y_ext[t] = val
            y_adj[t] = val
            if "const" in components:
                y_adj[t] -= components["const"]
            if "trend" in components:
                y_adj[t] -= components["trend"] * t

        forecasts = y_ext[n:]

        # Forecast variance
        var_forecast = np.zeros(steps, dtype=np.float64)
        for h in range(steps):
            var_forecast[h] = sigma2 * np.sum(psi[: h + 1] ** 2)

        z = stats.norm.ppf(1 - alpha / 2)
        se_forecast = np.sqrt(var_forecast)

        return {
            "forecast": forecasts,
            "lower": forecasts - z * se_forecast,
            "upper": forecasts + z * se_forecast,
        }

    def simulate(
        self, n: int, seed: int | None = None
    ) -> NDArray[np.float64]:
        """Simulate from the fitted model.

        Parameters
        ----------
        n : int
            Number of observations to simulate.
        seed : int or None
            Random seed.

        Returns
        -------
        ndarray
            Simulated series.
        """
        if self._results is None:
            msg = "Model must be fit before simulating"
            raise RuntimeError(msg)

        rng = np.random.default_rng(seed)
        params = self._results.params
        components = self._unpack_params(params)
        sigma2 = components["sigma2"]
        ar = components["ar"]
        ma = components["ma"]

        eps = rng.normal(0, np.sqrt(sigma2), n)
        y = np.zeros(n, dtype=np.float64)

        m = max(self.p, self.q)
        for t in range(m, n):
            val = eps[t]
            for j in range(self.p):
                val += ar[j] * y[t - j - 1]
            for j in range(self.q):
                val += ma[j] * eps[t - j - 1]
            if "const" in components:
                val += components["const"]
            y[t] = val

        return y

    def loglike(self, params: NDArray[np.float64]) -> float:
        """Compute log-likelihood for given parameters.

        Parameters
        ----------
        params : ndarray
            Parameter vector.

        Returns
        -------
        float
            Log-likelihood value.
        """
        if self._endog_diff is None:
            msg = "Must call fit() first or provide data"
            raise RuntimeError(msg)
        return self._mle_loglike(params, self._endog_diff)
