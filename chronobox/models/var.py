"""VAR(p) - Vector Autoregression estimated by OLS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as scipy_stats

if TYPE_CHECKING:
    from chronobox.analysis.fevd import FEVD
    from chronobox.analysis.irf import IRF

from chronobox.selection.lag_selection import LagOrderResults, select_lag_order


@dataclass
class GrangerResult:
    """Result of a Granger causality test.

    Attributes
    ----------
    fstat : float
        F-test statistic.
    pvalue : float
        P-value from F distribution.
    df : tuple[int, int]
        Degrees of freedom (numerator, denominator).
    reject : bool
        Whether to reject H0 (no Granger causality) at the given significance level.
    wald_stat : float
        Wald test statistic.
    wald_pvalue : float
        P-value from chi-squared distribution for Wald test.
    caused : str
        Name of the caused (dependent) variable.
    causing : str
        Name of the causing variable.
    signif : float
        Significance level used for the test.
    """

    fstat: float
    pvalue: float
    df: tuple[int, int]
    reject: bool
    wald_stat: float
    wald_pvalue: float
    caused: str
    causing: str
    signif: float

    def __repr__(self) -> str:
        direction = f"{self.causing} -> {self.caused}"
        result = "REJECT H0" if self.reject else "FAIL TO REJECT H0"
        return (
            f"GrangerResult({direction}: F={self.fstat:.4f}, "
            f"p={self.pvalue:.4f}, {result} at {self.signif:.0%})"
        )


class VARResults:
    """Results from fitting a VAR(p) model.

    Attributes
    ----------
    coefs : ndarray of shape (p, K, K)
        Coefficient matrices A_1, ..., A_p.
    sigma_u : ndarray of shape (K, K)
        Residual covariance matrix (bias-corrected).
    sigma_u_ml : ndarray of shape (K, K)
        Residual covariance matrix (ML, divided by T').
    intercept : ndarray of shape (K,)
        Intercept/constant terms (zeros if trend='n').
    trend_coefs : ndarray of shape (K, d) or None
        All deterministic coefficients (including intercept).
    resid : ndarray of shape (T', K)
        Residual matrix.
    endog : ndarray of shape (T, K)
        Original endogenous data.
    nobs : int
        Effective number of observations T'.
    nobs_total : int
        Total number of observations T.
    k_ar : int
        VAR order p.
    neqs : int
        Number of equations/variables K.
    names : list[str]
        Variable names.
    trend : str
        Trend specification used.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    hqic : float
        Hannan-Quinn Information Criterion.
    fpe : float
        Final Prediction Error.
    is_stable : bool
        Whether the VAR process is stable (all companion eigenvalues inside unit circle).
    roots : ndarray
        Eigenvalues of the companion matrix.
    """

    def __init__(
        self,
        coefs: NDArray[np.float64],
        sigma_u: NDArray[np.float64],
        sigma_u_ml: NDArray[np.float64],
        intercept: NDArray[np.float64],
        trend_coefs: NDArray[np.float64] | None,
        resid: NDArray[np.float64],
        endog: NDArray[np.float64],
        nobs: int,
        nobs_total: int,
        k_ar: int,
        neqs: int,
        names: list[str],
        trend: str,
    ) -> None:
        self.coefs = coefs
        self.sigma_u = sigma_u
        self.sigma_u_ml = sigma_u_ml
        self.intercept = intercept
        self.trend_coefs = trend_coefs
        self.resid = resid
        self.endog = endog
        self.nobs = nobs
        self.nobs_total = nobs_total
        self.k_ar = k_ar
        self.neqs = neqs
        self.names = names
        self.trend = trend

        # Compute information criteria
        self.aic = self._compute_aic()
        self.bic = self._compute_bic()
        self.hqic = self._compute_hqic()
        self.fpe = self._compute_fpe()

        # Compute stability
        self.roots = self._companion_eigenvalues()
        self.is_stable = bool(np.all(np.abs(self.roots) < 1.0))

    def _compute_aic(self) -> float:
        """Compute AIC."""
        log_det = np.log(np.linalg.det(self.sigma_u_ml))
        return float(log_det + 2.0 * self.k_ar * self.neqs**2 / self.nobs)

    def _compute_bic(self) -> float:
        """Compute BIC."""
        log_det = np.log(np.linalg.det(self.sigma_u_ml))
        return float(
            log_det + self.k_ar * self.neqs**2 * np.log(self.nobs) / self.nobs
        )

    def _compute_hqic(self) -> float:
        """Compute HQIC."""
        log_det = np.log(np.linalg.det(self.sigma_u_ml))
        return float(
            log_det
            + 2.0
            * self.k_ar
            * self.neqs**2
            * np.log(np.log(self.nobs))
            / self.nobs
        )

    def _compute_fpe(self) -> float:
        """Compute FPE."""
        det_ml = np.linalg.det(self.sigma_u_ml)
        d = self._det_terms_count()
        numer = (self.nobs + self.neqs * self.k_ar + d) ** self.neqs
        denom = max((self.nobs - self.neqs * self.k_ar - d), 1) ** self.neqs
        return float(det_ml * numer / denom)

    def _det_terms_count(self) -> int:
        """Count deterministic terms."""
        if self.trend == "n":
            return 0
        elif self.trend == "c":
            return 1
        elif self.trend == "ct":
            return 2
        elif self.trend == "ctt":
            return 3
        return 0

    def _companion_matrix(self) -> NDArray[np.float64]:
        """Build the companion matrix A_c (Kp x Kp).

        Returns
        -------
        ndarray of shape (Kp, Kp)
            Companion form matrix.
        """
        k = self.neqs
        p = self.k_ar

        if p == 0:
            return np.zeros((k, k), dtype=np.float64)

        kp = k * p
        companion = np.zeros((kp, kp), dtype=np.float64)

        # First block row: [A_1, A_2, ..., A_p]
        for i in range(p):
            companion[:k, i * k : (i + 1) * k] = self.coefs[i]

        # Identity blocks on the sub-diagonal
        if p > 1:
            companion[k:, : k * (p - 1)] = np.eye(k * (p - 1))

        return companion

    def _companion_eigenvalues(self) -> NDArray[np.complex128]:
        """Compute eigenvalues of the companion matrix.

        Returns
        -------
        ndarray
            Eigenvalues sorted by descending modulus.
        """
        companion = self._companion_matrix()
        eigenvalues = np.linalg.eigvals(companion)
        # Sort by descending modulus
        idx = np.argsort(-np.abs(eigenvalues))
        return eigenvalues[idx]

    def _ma_coefs(self, periods: int) -> NDArray[np.float64]:
        """Compute MA coefficient matrices Phi_0, Phi_1, ..., Phi_periods.

        Phi_0 = I_K
        Phi_h = sum_{j=1}^{min(h,p)} A_j * Phi_{h-j}

        Parameters
        ----------
        periods : int
            Number of periods.

        Returns
        -------
        ndarray of shape (periods+1, K, K)
            MA coefficient matrices.
        """
        k = self.neqs
        p = self.k_ar
        phi = np.zeros((periods + 1, k, k), dtype=np.float64)
        phi[0] = np.eye(k)

        for h in range(1, periods + 1):
            for j in range(1, min(h, p) + 1):
                phi[h] += self.coefs[j - 1] @ phi[h - j]

        return phi

    def forecast(
        self,
        steps: int = 1,
        alpha: float = 0.05,
    ) -> NDArray[np.float64]:
        """Produce h-step ahead forecasts.

        Parameters
        ----------
        steps : int, default 1
            Number of forecast steps.
        alpha : float, default 0.05
            Significance level for confidence intervals.

        Returns
        -------
        ndarray of shape (steps, K)
            Point forecasts.
        """
        k = self.neqs
        p = self.k_ar
        y = self.endog.copy()  # (T, K)
        t = y.shape[0]

        forecasts = np.zeros((steps, k), dtype=np.float64)

        # Extended data array: original + forecasts appended
        y_ext = np.vstack([y, np.zeros((steps, k))])

        for h in range(steps):
            t_h = t + h
            yhat = np.zeros(k, dtype=np.float64)

            # AR part
            for lag_i in range(p):
                yhat += self.coefs[lag_i] @ y_ext[t_h - lag_i - 1]

            # Deterministic part
            if self.trend in ("c", "ct", "ctt"):
                yhat += self.intercept
            if self.trend in ("ct", "ctt") and self.trend_coefs is not None:
                # Add linear trend evaluated at t_h + 1
                trend_val = t_h + 1
                if self.trend_coefs.shape[1] >= 2:
                    yhat += self.trend_coefs[:, 1] * trend_val

            y_ext[t_h] = yhat
            forecasts[h] = yhat

        return forecasts

    def irf(
        self,
        periods: int = 20,
        method: str = "cholesky",
        sigs: float = 0.95,
        runs: int = 1000,
    ) -> IRF:
        """Compute Impulse Response Functions.

        Parameters
        ----------
        periods : int, default 20
            Number of periods for IRF.
        method : str, default 'cholesky'
            Method: 'cholesky' (orthogonalized) or 'generalized' (Pesaran-Shin).
        sigs : float, default 0.95
            Confidence level for bootstrap bands.
        runs : int, default 1000
            Number of bootstrap replications.

        Returns
        -------
        IRF
            Impulse Response Function results.
        """
        from chronobox.analysis.irf import IRF as _IRF

        return _IRF(self, periods=periods, method=method, sigs=sigs, runs=runs)

    def fevd(self, periods: int = 20, method: str = "cholesky") -> FEVD:
        """Compute Forecast Error Variance Decomposition.

        Parameters
        ----------
        periods : int, default 20
            Number of periods for FEVD.
        method : str, default 'cholesky'
            Method for orthogonalization.

        Returns
        -------
        FEVD
            Forecast Error Variance Decomposition results.
        """
        from chronobox.analysis.fevd import FEVD as _FEVD

        return _FEVD(self, periods=periods, method=method)

    def granger_causality(
        self,
        caused: str | int,
        causing: str | int,
        signif: float = 0.05,
    ) -> GrangerResult:
        """Test Granger causality.

        Parameters
        ----------
        caused : str or int
            Name or index of the caused (dependent) variable.
        causing : str or int
            Name or index of the causing variable.
        signif : float, default 0.05
            Significance level for the test.

        Returns
        -------
        GrangerResult
            Granger causality test result.
        """
        from chronobox.analysis.granger import granger_causality

        return granger_causality(self, caused=caused, causing=causing, signif=signif)

    def test_whiteness(self, nlags: int = 10) -> dict[str, Any]:
        """Portmanteau test for residual whiteness (Ljung-Box multivariate).

        Parameters
        ----------
        nlags : int, default 10
            Number of lags to test.

        Returns
        -------
        dict
            Dictionary with 'statistic', 'pvalue', 'df', 'reject' keys.
        """
        k = self.neqs
        t_eff = self.nobs
        resid = self.resid  # (T', K)

        # C_0 = sample covariance at lag 0
        c0 = (resid.T @ resid) / t_eff
        c0_inv = np.linalg.inv(c0)

        q_stat = 0.0
        for i in range(1, nlags + 1):
            # C_i = cross-covariance at lag i
            ci = (resid[i:].T @ resid[:-i]) / t_eff
            q_stat += np.trace(ci.T @ c0_inv @ ci @ c0_inv) / (t_eff - i)

        q_stat *= t_eff * (t_eff + 2)

        df = k * k * (nlags - self.k_ar)
        if df <= 0:
            df = 1

        pvalue = 1.0 - scipy_stats.chi2.cdf(q_stat, df)

        return {
            "statistic": float(q_stat),
            "pvalue": float(pvalue),
            "df": df,
            "reject": pvalue < 0.05,
        }

    def summary(self) -> str:
        """Generate formatted summary of the VAR results.

        Returns
        -------
        str
            Formatted summary table with coefficients by equation.
        """
        lines: list[str] = []
        lines.append("=" * 78)
        lines.append(f"  VAR({self.k_ar}) Estimation Results")
        lines.append("=" * 78)
        lines.append(f"  No. of equations:   {self.neqs}")
        lines.append(f"  No. of lags:        {self.k_ar}")
        lines.append(f"  No. of obs (total): {self.nobs_total}")
        lines.append(f"  No. of obs (used):  {self.nobs}")
        lines.append(f"  Trend:              {self.trend}")
        lines.append(f"  Stable:             {self.is_stable}")
        lines.append(f"  AIC:                {self.aic:.4f}")
        lines.append(f"  BIC:                {self.bic:.4f}")
        lines.append(f"  HQIC:               {self.hqic:.4f}")
        lines.append(f"  FPE:                {self.fpe:.6e}")
        lines.append("")

        # Compute standard errors from OLS
        k = self.neqs
        p = self.k_ar
        d = self._det_terms_count()
        total_regressors = k * p + d

        # Z matrix reconstruction for SE computation
        z_parts: list[NDArray[np.float64]] = []
        for lag_i in range(1, p + 1):
            z_parts.append(self.endog[p - lag_i : self.nobs_total - lag_i])

        if self.trend in ("c", "ct", "ctt"):
            z_parts.append(np.ones((self.nobs, 1)))
        if self.trend in ("ct", "ctt"):
            z_parts.append(
                np.arange(1, self.nobs + 1, dtype=np.float64).reshape(-1, 1)
            )
        if self.trend == "ctt":
            z_parts.append(
                (np.arange(1, self.nobs + 1, dtype=np.float64) ** 2).reshape(-1, 1)
            )

        if z_parts:
            z_mat = np.column_stack(z_parts)
            try:
                ztz_inv = np.linalg.inv(z_mat.T @ z_mat)
            except np.linalg.LinAlgError:
                ztz_inv = np.linalg.pinv(z_mat.T @ z_mat)
        else:
            ztz_inv = np.zeros((1, 1))

        for eq_idx in range(k):
            lines.append("-" * 78)
            lines.append(f"  Equation: {self.names[eq_idx]}")
            lines.append("-" * 78)
            lines.append(
                f"  {'Variable':<20s} {'Coef':>12s} {'Std.Err':>12s} "
                f"{'t-stat':>10s} {'p-value':>10s}"
            )
            lines.append("  " + "-" * 74)

            # SE for this equation: sqrt(sigma_u[eq,eq] * diag(ZtZ_inv))
            se_scale = self.sigma_u[eq_idx, eq_idx]

            coef_idx = 0
            for lag_i in range(p):
                for var_j in range(k):
                    coef_val = self.coefs[lag_i, eq_idx, var_j]
                    if coef_idx < ztz_inv.shape[0]:
                        se_val = np.sqrt(se_scale * ztz_inv[coef_idx, coef_idx])
                    else:
                        se_val = float("nan")
                    t_stat = coef_val / se_val if se_val > 0 else float("nan")
                    p_val = (
                        2.0
                        * (
                            1.0
                            - scipy_stats.t.cdf(
                                abs(t_stat), self.nobs - total_regressors
                            )
                        )
                        if np.isfinite(t_stat)
                        else float("nan")
                    )
                    label = f"L{lag_i + 1}.{self.names[var_j]}"
                    lines.append(
                        f"  {label:<20s} {coef_val:12.6f} {se_val:12.6f} "
                        f"{t_stat:10.4f} {p_val:10.4f}"
                    )
                    coef_idx += 1

            # Deterministic terms
            if self.trend in ("c", "ct", "ctt"):
                coef_val = self.intercept[eq_idx]
                if coef_idx < ztz_inv.shape[0]:
                    se_val = np.sqrt(se_scale * ztz_inv[coef_idx, coef_idx])
                else:
                    se_val = float("nan")
                t_stat = coef_val / se_val if se_val > 0 else float("nan")
                p_val = (
                    2.0
                    * (
                        1.0
                        - scipy_stats.t.cdf(
                            abs(t_stat), self.nobs - total_regressors
                        )
                    )
                    if np.isfinite(t_stat)
                    else float("nan")
                )
                lines.append(
                    f"  {'const':<20s} {coef_val:12.6f} {se_val:12.6f} "
                    f"{t_stat:10.4f} {p_val:10.4f}"
                )

            lines.append("")

        # Residual covariance matrix
        lines.append("-" * 78)
        lines.append("  Residual Covariance Matrix (Sigma_u)")
        lines.append("-" * 78)
        header = "  " + "".join(f"{n:>12s}" for n in self.names)
        lines.append(header)
        for i in range(k):
            row = f"  {self.names[i]:<10s}" + "".join(
                f"{self.sigma_u[i, j]:12.6f}" for j in range(k)
            )
            lines.append(row)

        lines.append("")
        lines.append("=" * 78)
        return "\n".join(lines)

    def plot_forecast(
        self,
        steps: int = 8,
        alpha: float = 0.05,
    ) -> Any:
        """Plot forecasts for each variable.

        Parameters
        ----------
        steps : int, default 8
            Number of forecast steps.
        alpha : float, default 0.05
            Significance level for confidence intervals.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        import matplotlib.pyplot as plt

        forecasts = self.forecast(steps=steps, alpha=alpha)
        k = self.neqs
        n_last = min(20, self.nobs)

        fig, axes = plt.subplots(k, 1, figsize=(10, 3 * k), squeeze=False)

        for i in range(k):
            ax = axes[i, 0]
            # Plot last n_last historical observations
            hist_vals = self.endog[-n_last:, i]
            hist_x = np.arange(-n_last, 0)
            ax.plot(hist_x, hist_vals, "b-", label="Historical")

            # Plot forecast
            fc_x = np.arange(0, steps)
            ax.plot(fc_x, forecasts[:, i], "r--", label="Forecast")

            ax.axvline(x=-0.5, color="gray", linestyle=":", alpha=0.5)
            ax.set_title(self.names[i])
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"VAR({self.k_ar}) Forecast", fontsize=14)
        plt.tight_layout()
        return fig


class VAR:
    """Vector Autoregression model estimated by OLS.

    Parameters
    ----------
    lags : int or None
        Number of lags. If None and maxlags is provided, selects
        automatically via AIC.
    trend : str, default 'c'
        Trend specification:
        - 'n': no deterministic terms
        - 'c': constant only
        - 'ct': constant + linear trend
        - 'ctt': constant + linear + quadratic trend
    maxlags : int or None
        Maximum lags for automatic selection. Only used when lags is None.

    Examples
    --------
    >>> import numpy as np
    >>> from chronobox.models.var import VAR
    >>> data = np.random.randn(100, 3)
    >>> model = VAR(lags=2)
    >>> results = model.fit(data)
    >>> print(results.summary())
    """

    def __init__(
        self,
        lags: int | None = None,
        trend: str = "c",
        maxlags: int | None = None,
    ) -> None:
        self.lags = lags
        self.trend = trend
        self.maxlags = maxlags

        if trend not in ("n", "c", "ct", "ctt"):
            msg = f"trend must be 'n', 'c', 'ct', or 'ctt', got '{trend}'"
            raise ValueError(msg)

    def fit(
        self,
        endog: NDArray[np.float64] | pd.DataFrame,
        names: list[str] | None = None,
    ) -> VARResults:
        """Fit the VAR(p) model by OLS.

        Parameters
        ----------
        endog : ndarray of shape (T, K) or DataFrame
            Multivariate time series data. Each column is a variable.
        names : list of str or None
            Variable names. Inferred from DataFrame columns if available.

        Returns
        -------
        VARResults
            Fitted model results.

        Raises
        ------
        ValueError
            If data has insufficient observations or wrong dimensionality.
        """
        # Convert DataFrame
        if isinstance(endog, pd.DataFrame):
            if names is None:
                names = list(endog.columns)
            endog_arr = endog.to_numpy(dtype=np.float64)
        else:
            endog_arr = np.asarray(endog, dtype=np.float64)

        if endog_arr.ndim != 2:
            msg = f"endog must be 2-D, got {endog_arr.ndim}-D"
            raise ValueError(msg)

        t_total, k = endog_arr.shape

        if names is None:
            names = [f"y{i + 1}" for i in range(k)]

        # Determine lag order
        if self.lags is not None:
            p = self.lags
        elif self.maxlags is not None:
            lag_result = select_lag_order(
                endog_arr, maxlags=self.maxlags, trend=self.trend
            )
            p = lag_result.selected_orders.get("aic", 1)
        else:
            msg = "Either lags or maxlags must be specified"
            raise ValueError(msg)

        if p < 1:
            msg = f"lags must be >= 1, got {p}"
            raise ValueError(msg)

        if t_total <= p + k:
            msg = (
                f"Insufficient observations: T={t_total}, p={p}, K={k}. "
                f"Need T > p + K."
            )
            raise ValueError(msg)

        # Effective sample
        t_eff = t_total - p

        # Build Y matrix: (T', K)
        y_mat = endog_arr[p:]  # (T', K)

        # Build Z matrix: lagged values + deterministic terms
        z_parts: list[NDArray[np.float64]] = []

        for lag_i in range(1, p + 1):
            z_parts.append(endog_arr[p - lag_i : t_total - lag_i])  # (T', K)

        # Deterministic terms
        if self.trend in ("c", "ct", "ctt"):
            z_parts.append(np.ones((t_eff, 1), dtype=np.float64))
        if self.trend in ("ct", "ctt"):
            z_parts.append(
                np.arange(1, t_eff + 1, dtype=np.float64).reshape(-1, 1)
            )
        if self.trend == "ctt":
            z_parts.append(
                (np.arange(1, t_eff + 1, dtype=np.float64) ** 2).reshape(-1, 1)
            )

        z_mat = np.column_stack(z_parts)  # (T', Kp + d)

        # OLS: B = (Z'Z)^{-1} Z'Y
        b_hat, _, _, _ = np.linalg.lstsq(z_mat, y_mat, rcond=None)
        # b_hat shape: (Kp + d, K)

        # Extract coefficients
        coefs = np.zeros((p, k, k), dtype=np.float64)
        for lag_i in range(p):
            coefs[lag_i] = b_hat[lag_i * k : (lag_i + 1) * k].T  # (K, K)

        # Extract deterministic coefficients
        det_start = p * k
        d = 0
        if self.trend in ("c", "ct", "ctt"):
            d += 1
        if self.trend in ("ct", "ctt"):
            d += 1
        if self.trend == "ctt":
            d += 1

        intercept = np.zeros(k, dtype=np.float64)
        trend_coefs: NDArray[np.float64] | None = None

        if d > 0:
            trend_coefs_raw = b_hat[det_start : det_start + d].T  # (K, d)
            trend_coefs = trend_coefs_raw
            if self.trend in ("c", "ct", "ctt"):
                intercept = trend_coefs_raw[:, 0]

        # Residuals
        resid = y_mat - z_mat @ b_hat  # (T', K)

        # Covariance matrices
        df = t_eff - k * p - d
        if df <= 0:
            df = 1
        sigma_u = (resid.T @ resid) / df  # bias-corrected
        sigma_u_ml = (resid.T @ resid) / t_eff  # ML

        return VARResults(
            coefs=coefs,
            sigma_u=sigma_u,
            sigma_u_ml=sigma_u_ml,
            intercept=intercept,
            trend_coefs=trend_coefs,
            resid=resid,
            endog=endog_arr,
            nobs=t_eff,
            nobs_total=t_total,
            k_ar=p,
            neqs=k,
            names=names,
            trend=self.trend,
        )

    def select_order(
        self,
        endog: NDArray[np.float64] | pd.DataFrame,
        maxlags: int = 15,
    ) -> LagOrderResults:
        """Select optimal lag order using information criteria.

        Parameters
        ----------
        endog : ndarray of shape (T, K) or DataFrame
            Multivariate time series data.
        maxlags : int, default 15
            Maximum number of lags.

        Returns
        -------
        LagOrderResults
            Lag order selection results.
        """
        if isinstance(endog, pd.DataFrame):
            endog_arr = endog.to_numpy(dtype=np.float64)
        else:
            endog_arr = np.asarray(endog, dtype=np.float64)

        return select_lag_order(endog_arr, maxlags=maxlags, trend=self.trend)
