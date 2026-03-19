"""
ARDL (Autoregressive Distributed Lag) model.

Supports automatic lag selection via AIC/BIC and bounds testing
for cointegration (Pesaran, Shin & Smith, 2001).

References
----------
Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing approaches
    to the analysis of level relationships. Journal of Applied Econometrics,
    16(3), 289-326.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from chronobox.models.ecm import ECMResult


@dataclass
class ARDLResult:
    """Result from ARDL model estimation.

    Attributes
    ----------
    coefficients : ndarray
        All estimated OLS coefficients.
    residuals : ndarray
        OLS residuals.
    fitted_values : ndarray
        Fitted values from the model.
    y_lags : int
        Selected number of lags for y (p).
    x_lags : list[int]
        Selected number of lags for each x variable (q_j).
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    sigma2 : float
        Estimated residual variance.
    r_squared : float
        R-squared of the model.
    adj_r_squared : float
        Adjusted R-squared.
    nobs : int
        Number of observations used.
    k_params : int
        Number of estimated parameters.
    se : ndarray
        Standard errors of coefficients.
    t_stats : ndarray
        t-statistics for coefficients.
    _y : np.ndarray
        Original y series (for bounds_test and to_ecm).
    _x : np.ndarray
        Original x matrix (for bounds_test and to_ecm).
    """

    coefficients: np.ndarray
    residuals: np.ndarray
    fitted_values: np.ndarray
    y_lags: int
    x_lags: list[int]
    aic: float
    bic: float
    sigma2: float
    r_squared: float
    adj_r_squared: float
    nobs: int
    k_params: int
    se: np.ndarray
    t_stats: np.ndarray
    _y: np.ndarray = field(repr=False)
    _x: np.ndarray = field(repr=False)

    def summary(self) -> str:
        """Return a text summary of the ARDL model."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(
            f"ARDL({self.y_lags}, {', '.join(map(str, self.x_lags))}) Results"
        )
        lines.append("=" * 60)
        lines.append(f"Observations: {self.nobs}")
        lines.append(f"Parameters:   {self.k_params}")
        lines.append(f"R-squared:    {self.r_squared:.6f}")
        lines.append(f"Adj R-sq:     {self.adj_r_squared:.6f}")
        lines.append(f"AIC:          {self.aic:.4f}")
        lines.append(f"BIC:          {self.bic:.4f}")
        lines.append(f"Sigma^2:      {self.sigma2:.6f}")
        lines.append("-" * 60)
        lines.append(f"{'Param':>10} {'Coef':>12} {'Std Err':>12} {'t-stat':>10}")
        lines.append("-" * 60)
        for i, (c, s, t) in enumerate(
            zip(self.coefficients, self.se, self.t_stats, strict=True)
        ):
            lines.append(
                f"{'beta_' + str(i):>10} {c:>12.6f} {s:>12.6f} {t:>10.4f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def bounds_test(self) -> dict[str, float | int | str]:
        """Perform PSS bounds test for cointegration.

        Returns a dict with F-statistic and approximate critical values.
        H0: no long-run relationship (pi_yy = 0 and pi_yx = 0).

        Returns
        -------
        dict
            Keys: 'f_statistic', 'k' (number of x variables),
            'lower_10', 'upper_10', 'lower_5', 'upper_5',
            'lower_1', 'upper_1', 'conclusion'.
        """
        y = self._y
        x = self._x
        p = self.y_lags
        q_max = max(self.x_lags) if self.x_lags else 0
        k_x = x.shape[1] if x.ndim > 1 else 1

        max_lag = max(p, q_max + 1)
        start = max_lag

        # Dependent variable: Delta y_t
        dy = np.diff(y)
        dep = dy[start - 1 :]
        n = len(dep)

        # Unrestricted model regressors
        regressors_u: list[np.ndarray] = [np.ones(n)]  # constant
        regressors_u.append(y[start - 1 : start - 1 + n])  # y_{t-1}

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        for j in range(k_x):
            regressors_u.append(x[start - 1 : start - 1 + n, j])  # x_{j,t-1}

        # Lagged Delta y
        for i in range(1, p):
            regressors_u.append(dy[start - 1 - i : start - 1 - i + n])

        # Lagged Delta x
        dx = np.diff(x, axis=0)
        for j in range(k_x):
            q_j = self.x_lags[j] if j < len(self.x_lags) else 0
            for lag_l in range(q_j):
                regressors_u.append(
                    dx[start - 1 - lag_l : start - 1 - lag_l + n, j]
                )

        x_u = np.column_stack(regressors_u)
        beta_u = np.linalg.lstsq(x_u, dep, rcond=None)[0]
        rss_u = float(np.sum((dep - x_u @ beta_u) ** 2))

        # Restricted model: exclude y_{t-1} and x_{t-1}
        cols_to_remove = set(range(1, 2 + k_x))
        cols_to_keep = [
            i for i in range(x_u.shape[1]) if i not in cols_to_remove
        ]
        x_r = x_u[:, cols_to_keep]
        beta_r = np.linalg.lstsq(x_r, dep, rcond=None)[0]
        rss_r = float(np.sum((dep - x_r @ beta_r) ** 2))

        # F-statistic
        m = 1 + k_x
        df_u = n - x_u.shape[1]
        f_stat = ((rss_r - rss_u) / m) / (rss_u / df_u)

        # Approximate PSS critical values (Case III: unrestricted constant)
        pss_cv: dict[int, dict[str, float]] = {
            1: {
                "lower_10": 4.04,
                "upper_10": 4.78,
                "lower_5": 4.94,
                "upper_5": 5.73,
                "lower_1": 6.84,
                "upper_1": 7.84,
            },
            2: {
                "lower_10": 3.17,
                "upper_10": 4.14,
                "lower_5": 3.79,
                "upper_5": 4.85,
                "lower_1": 5.15,
                "upper_1": 6.36,
            },
            3: {
                "lower_10": 2.72,
                "upper_10": 3.77,
                "lower_5": 3.23,
                "upper_5": 4.35,
                "lower_1": 4.29,
                "upper_1": 5.61,
            },
            4: {
                "lower_10": 2.45,
                "upper_10": 3.52,
                "lower_5": 2.86,
                "upper_5": 4.01,
                "lower_1": 3.74,
                "upper_1": 5.06,
            },
        }

        cv = pss_cv.get(k_x, pss_cv[4])

        if f_stat > cv["upper_5"]:
            conclusion = "reject_h0"
        elif f_stat < cv["lower_5"]:
            conclusion = "fail_to_reject"
        else:
            conclusion = "inconclusive"

        return {
            "f_statistic": f_stat,
            "k": k_x,
            **cv,
            "conclusion": conclusion,
        }

    def to_ecm(self) -> ECMResult:
        """Convert ARDL results to ECM form."""
        from chronobox.models.ecm import ECM

        ecm = ECM(lags=self.y_lags)
        return ecm.fit(self._y, self._x)

    @property
    def long_run_coefficients(self) -> np.ndarray:
        """Compute long-run coefficients theta_j = sum(beta_jl) / (1 - sum(phi_i)).

        Returns
        -------
        ndarray, shape (k,)
            Long-run coefficient for each x variable.
        """
        p = self.y_lags
        x_lags = self.x_lags
        coef = self.coefficients

        # coef layout: [const, y_{t-1}, ..., y_{t-p}, x1_t, x1_{t-1}, ..., x2_t, ...]
        phi_sum = float(np.sum(coef[1 : 1 + p]))
        denom = 1.0 - phi_sum

        lr_coefs: list[float] = []
        idx = 1 + p  # start of x coefficients
        for q_j in x_lags:
            beta_sum = float(np.sum(coef[idx : idx + q_j + 1]))
            lr_coefs.append(beta_sum / denom)
            idx += q_j + 1

        return np.array(lr_coefs)


def _build_ardl_matrices(
    y: np.ndarray,
    x: np.ndarray,
    p: int,
    x_lags: list[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build OLS design matrix for ARDL(p, q_1, ..., q_k).

    Returns (Y_dep, X_reg, start_idx).
    """
    t_len = len(y)
    k_x = x.shape[1] if x.ndim > 1 else 1
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    q_max = max(x_lags) if x_lags else 0
    start = max(p, q_max)

    n_obs = t_len - start
    if n_obs <= 0:
        raise ValueError(f"Not enough observations: T={t_len}, max_lag={start}")

    # Dependent: y[start:]
    y_dep = y[start:]

    # Regressors: constant + p lags of y + q_j+1 lags of each x_j
    n_params = 1 + p + sum(q_j + 1 for q_j in x_lags)
    x_reg = np.zeros((n_obs, n_params))

    x_reg[:, 0] = 1.0  # constant

    col = 1
    for i in range(1, p + 1):
        x_reg[:, col] = y[start - i : t_len - i]
        col += 1

    for j in range(k_x):
        q_j = x_lags[j] if j < len(x_lags) else 0
        for lag_l in range(q_j + 1):
            x_reg[:, col] = x[start - lag_l : t_len - lag_l, j]
            col += 1

    return y_dep, x_reg, start


class ARDL:
    """ARDL (Autoregressive Distributed Lag) model with automatic lag selection.

    Parameters
    ----------
    max_p : int
        Maximum number of lags for y. Default is 4.
    max_q : int
        Maximum number of lags for each x variable. Default is 4.
    criterion : {"aic", "bic"}
        Information criterion for automatic lag selection. Default is "aic".
    uniform_q : bool
        If True, use the same q for all x variables. Default is True.
        This reduces the search space significantly.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.random.randn(200)
    >>> x = np.random.randn(200, 2)
    >>> ardl = ARDL(max_p=4, max_q=4, criterion='aic')
    >>> result = ardl.fit(y, x)
    >>> print(result.summary())
    """

    def __init__(
        self,
        max_p: int = 4,
        max_q: int = 4,
        criterion: Literal["aic", "bic"] = "aic",
        uniform_q: bool = True,
    ) -> None:
        if max_p < 1:
            raise ValueError(f"max_p must be >= 1, got {max_p}")
        if max_q < 0:
            raise ValueError(f"max_q must be >= 0, got {max_q}")
        if criterion not in ("aic", "bic"):
            raise ValueError(
                f"criterion must be 'aic' or 'bic', got '{criterion}'"
            )

        self.max_p = max_p
        self.max_q = max_q
        self.criterion = criterion
        self.uniform_q = uniform_q

    def fit(
        self,
        y: ArrayLike,
        x: ArrayLike,
        p: int | None = None,
        x_lags: list[int] | None = None,
    ) -> ARDLResult:
        """Estimate ARDL model.

        If p and x_lags are provided, estimate that specific model.
        Otherwise, perform automatic lag selection.

        Parameters
        ----------
        y : array_like, shape (T,)
            Dependent variable.
        x : array_like, shape (T,) or (T, k)
            Exogenous variables.
        p : int or None
            Number of lags for y. If None, auto-select.
        x_lags : list[int] or None
            Number of lags for each x variable. If None, auto-select.

        Returns
        -------
        ARDLResult
            Estimation results.
        """
        y_arr = np.asarray(y, dtype=np.float64).squeeze()
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y_arr.shape}")
        if len(y_arr) != x_arr.shape[0]:
            raise ValueError(
                f"y and x must have same length: {len(y_arr)} vs {x_arr.shape[0]}"
            )

        k_x = x_arr.shape[1]

        if p is not None and x_lags is not None:
            return self._estimate(y_arr, x_arr, p, x_lags)

        # Automatic lag selection
        best_ic = np.inf
        best_p = 1
        best_q_list: list[int] = [0] * k_x

        p_range = range(1, self.max_p + 1)

        if self.uniform_q:
            q_range = range(0, self.max_q + 1)
            for p_try in p_range:
                for q_try in q_range:
                    q_list = [q_try] * k_x
                    try:
                        result = self._estimate(y_arr, x_arr, p_try, q_list)
                        ic = (
                            result.aic
                            if self.criterion == "aic"
                            else result.bic
                        )
                        if ic < best_ic:
                            best_ic = ic
                            best_p = p_try
                            best_q_list = q_list
                    except (np.linalg.LinAlgError, ValueError):
                        continue
        else:
            q_values = list(range(0, self.max_q + 1))
            q_combos = list(product(q_values, repeat=k_x))
            for p_try in p_range:
                for q_combo in q_combos:
                    q_list = list(q_combo)
                    try:
                        result = self._estimate(y_arr, x_arr, p_try, q_list)
                        ic = (
                            result.aic
                            if self.criterion == "aic"
                            else result.bic
                        )
                        if ic < best_ic:
                            best_ic = ic
                            best_p = p_try
                            best_q_list = q_list
                    except (np.linalg.LinAlgError, ValueError):
                        continue

        return self._estimate(y_arr, x_arr, best_p, best_q_list)

    def _estimate(
        self,
        y: np.ndarray,
        x: np.ndarray,
        p: int,
        x_lags: list[int],
    ) -> ARDLResult:
        """Estimate a specific ARDL(p, q_1, ..., q_k) model via OLS."""
        y_dep, x_reg, _start = _build_ardl_matrices(y, x, p, x_lags)
        n = len(y_dep)
        k = x_reg.shape[1]

        if n <= k:
            raise ValueError(f"Not enough observations: n={n}, params={k}")

        # OLS
        beta = np.linalg.lstsq(x_reg, y_dep, rcond=None)[0]
        fitted = x_reg @ beta
        resid = y_dep - fitted

        # Statistics
        rss = float(np.sum(resid**2))
        tss = float(np.sum((y_dep - np.mean(y_dep)) ** 2))
        sigma2 = rss / (n - k)
        r_sq = 1.0 - rss / tss if tss > 0 else 0.0
        adj_r_sq = (
            1.0 - (1.0 - r_sq) * (n - 1) / (n - k) if n > k else 0.0
        )

        # Information criteria
        aic = n * np.log(rss / n) + 2 * k
        bic = n * np.log(rss / n) + k * np.log(n)

        # Standard errors
        try:
            cov = sigma2 * np.linalg.inv(x_reg.T @ x_reg)
            se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            se = np.full(k, np.nan)

        t_stats = np.where(se > 0, beta / se, np.nan)

        return ARDLResult(
            coefficients=beta,
            residuals=resid,
            fitted_values=fitted,
            y_lags=p,
            x_lags=x_lags,
            aic=float(aic),
            bic=float(bic),
            sigma2=sigma2,
            r_squared=r_sq,
            adj_r_squared=adj_r_sq,
            nobs=n,
            k_params=k,
            se=se,
            t_stats=t_stats,
            _y=y,
            _x=x,
        )
