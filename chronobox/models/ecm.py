"""
ECM (Error Correction Model) for cointegrated time series.

Models short-run dynamics and long-run equilibrium adjustment:
    Delta y_t = c + pi_yy * y_{t-1} + pi_yx' * x_{t-1}
                + sum gamma_i * Delta y_{t-i}
                + sum delta_{jl} * Delta x_{j,t-l}
                + eps_t

References
----------
Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction:
    representation, estimation, and testing. Econometrica, 55(2), 251-276.
Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing approaches
    to the analysis of level relationships. Journal of Applied Econometrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class ECMResult:
    """Result from ECM estimation.

    Attributes
    ----------
    coefficients : ndarray
        All estimated coefficients.
    residuals : ndarray
        OLS residuals.
    fitted_values : ndarray
        Fitted values.
    speed_of_adjustment : float
        pi_yy coefficient. Should be negative for stability.
    long_run_coefficients : ndarray
        Long-run coefficients: -(pi_yx / pi_yy) for each x variable.
    pi_yy : float
        Error correction coefficient for y_{t-1}.
    pi_yx : ndarray
        Error correction coefficients for x_{t-1}.
    short_run_y : ndarray
        Coefficients on lagged Delta y.
    short_run_x : ndarray
        Coefficients on lagged Delta x (flattened).
    sigma2 : float
        Residual variance.
    r_squared : float
        R-squared.
    nobs : int
        Number of observations.
    k_params : int
        Number of parameters.
    se : ndarray
        Standard errors.
    t_stats : ndarray
        t-statistics.
    aic : float
        AIC.
    bic : float
        BIC.
    """

    coefficients: np.ndarray
    residuals: np.ndarray
    fitted_values: np.ndarray
    speed_of_adjustment: float
    long_run_coefficients: np.ndarray
    pi_yy: float
    pi_yx: np.ndarray
    short_run_y: np.ndarray
    short_run_x: np.ndarray
    sigma2: float
    r_squared: float
    nobs: int
    k_params: int
    se: np.ndarray
    t_stats: np.ndarray
    aic: float
    bic: float

    def summary(self) -> str:
        """Return text summary of ECM results."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("Error Correction Model (ECM) Results")
        lines.append("=" * 60)
        lines.append(f"Observations:         {self.nobs}")
        lines.append(f"Parameters:           {self.k_params}")
        lines.append(f"R-squared:            {self.r_squared:.6f}")
        lines.append(f"AIC:                  {self.aic:.4f}")
        lines.append(f"BIC:                  {self.bic:.4f}")
        lines.append(f"Sigma^2:              {self.sigma2:.6f}")
        lines.append("-" * 60)
        lines.append(
            f"Speed of adjustment (pi_yy): {self.speed_of_adjustment:.6f}"
        )
        lines.append(
            f"Long-run coefficients:       {self.long_run_coefficients}"
        )
        lines.append("-" * 60)
        lines.append(
            f"{'Param':>12} {'Coef':>12} {'Std Err':>12} {'t-stat':>10}"
        )
        lines.append("-" * 60)
        for i, (c, s, t) in enumerate(
            zip(self.coefficients, self.se, self.t_stats, strict=True)
        ):
            lines.append(
                f"{'beta_' + str(i):>12} {c:>12.6f} {s:>12.6f} {t:>10.4f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def bounds_test_pss(self) -> dict[str, float | str]:
        """Perform PSS bounds test on the ECM.

        Tests H0: pi_yy = 0 and pi_yx = 0 (no long-run relationship).

        Returns
        -------
        dict
            With keys: 't_statistic' (for pi_yy), 'pi_yy', 'conclusion'.
        """
        # t-test on pi_yy (second coefficient, after constant)
        idx_pi_yy = 1
        t_stat = float(self.t_stats[idx_pi_yy])

        # PSS t-test critical values (Case III, approximate)
        k_x = len(self.pi_yx)
        cv: dict[int, dict[str, float]] = {
            1: {"lower_5": -2.86, "upper_5": -3.22},
            2: {"lower_5": -2.86, "upper_5": -3.53},
            3: {"lower_5": -2.86, "upper_5": -3.78},
            4: {"lower_5": -2.86, "upper_5": -3.99},
        }
        crit = cv.get(k_x, cv[4])

        if t_stat < crit["upper_5"]:
            conclusion = "reject_h0"
        elif t_stat > crit["lower_5"]:
            conclusion = "fail_to_reject"
        else:
            conclusion = "inconclusive"

        return {
            "t_statistic": t_stat,
            "pi_yy": self.pi_yy,
            **crit,
            "conclusion": conclusion,
        }


class ECM:
    """Error Correction Model.

    Parameters
    ----------
    lags : int
        Number of lags for short-run dynamics. Default is 2.
    x_lags : int or None
        Number of lags for Delta x. If None, same as lags. Default is None.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.random.randn(200).cumsum()
    >>> x = np.random.randn(200, 2).cumsum(axis=0)
    >>> ecm = ECM(lags=2)
    >>> result = ecm.fit(y, x)
    >>> print(result.speed_of_adjustment)  # should be negative
    >>> print(result.long_run_coefficients)
    """

    def __init__(self, lags: int = 2, x_lags: int | None = None) -> None:
        if lags < 1:
            raise ValueError(f"lags must be >= 1, got {lags}")
        self.lags = lags
        self.x_lags = x_lags if x_lags is not None else lags

    def fit(self, y: ArrayLike, x: ArrayLike) -> ECMResult:
        """Estimate the ECM via OLS.

        Parameters
        ----------
        y : array_like, shape (T,)
            Dependent variable (levels).
        x : array_like, shape (T,) or (T, k)
            Exogenous variables (levels).

        Returns
        -------
        ECMResult
            Estimation results including speed of adjustment
            and long-run coefficients.
        """
        y_arr = np.asarray(y, dtype=np.float64).squeeze()
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y_arr.shape}")
        if len(y_arr) != x_arr.shape[0]:
            raise ValueError(
                f"Length mismatch: y={len(y_arr)}, x={x_arr.shape[0]}"
            )

        t_len = len(y_arr)
        k_x = x_arr.shape[1]
        p = self.lags
        q = self.x_lags

        max_lag = max(p, q) + 1  # +1 for the difference
        if t_len <= max_lag + 2:
            raise ValueError(
                f"Not enough observations: T={t_len}, need > {max_lag + 2}"
            )

        # First differences
        dy = np.diff(y_arr)  # length T-1
        dx = np.diff(x_arr, axis=0)  # (T-1, k_x)

        # Start index (in dy space)
        start = max(p, q)
        n = len(dy) - start

        # Dependent variable: Delta y_t
        dep = dy[start:]

        # Build regressors
        regressors: list[np.ndarray] = []

        # Constant
        regressors.append(np.ones(n))

        # y_{t-1} (levels)
        regressors.append(y_arr[start : start + n])

        # x_{j,t-1} (levels)
        for j in range(k_x):
            regressors.append(x_arr[start : start + n, j])

        # Lagged Delta y_{t-i}
        for i in range(1, p + 1):
            regressors.append(dy[start - i : start - i + n])

        # Lagged Delta x_{j,t-l}
        for j in range(k_x):
            for lag_l in range(q):
                regressors.append(dx[start - lag_l : start - lag_l + n, j])

        x_mat = np.column_stack(regressors)
        k_params = x_mat.shape[1]

        if n <= k_params:
            raise ValueError(f"Not enough obs: n={n}, params={k_params}")

        # OLS
        beta = np.linalg.lstsq(x_mat, dep, rcond=None)[0]
        fitted = x_mat @ beta
        resid = dep - fitted

        # Extract components
        pi_yy = float(beta[1])
        pi_yx = beta[2 : 2 + k_x]
        sr_y = beta[2 + k_x : 2 + k_x + p]
        sr_x = beta[2 + k_x + p :]

        # Long-run coefficients: -(pi_yx / pi_yy)
        lr_coefs = -(pi_yx / pi_yy) if abs(pi_yy) > 1e-12 else np.full(k_x, np.nan)

        # Statistics
        rss = float(np.sum(resid**2))
        tss = float(np.sum((dep - np.mean(dep)) ** 2))
        sigma2 = rss / (n - k_params)
        r_sq = 1.0 - rss / tss if tss > 0 else 0.0

        aic = n * np.log(rss / n) + 2 * k_params
        bic_val = n * np.log(rss / n) + k_params * np.log(n)

        # Standard errors
        try:
            cov = sigma2 * np.linalg.inv(x_mat.T @ x_mat)
            se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            se = np.full(k_params, np.nan)

        t_stats = np.where(se > 0, beta / se, np.nan)

        return ECMResult(
            coefficients=beta,
            residuals=resid,
            fitted_values=fitted,
            speed_of_adjustment=pi_yy,
            long_run_coefficients=lr_coefs,
            pi_yy=pi_yy,
            pi_yx=pi_yx,
            short_run_y=sr_y,
            short_run_x=sr_x,
            sigma2=sigma2,
            r_squared=r_sq,
            nobs=n,
            k_params=k_params,
            se=se,
            t_stats=t_stats,
            aic=float(aic),
            bic=float(bic_val),
        )
