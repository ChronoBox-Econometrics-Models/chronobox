"""
Structural Vector Autoregression (SVAR) with multiple identification schemes.

Identification schemes:
- Cholesky decomposition (recursive ordering)
- AB-model (Amisano & Giannini 1997)
- Blanchard-Quah long-run restrictions (1989)
- Sign restrictions (Uhlig 2005)

References
----------
- Kilian, L. & Lutkepohl, H. (2017). Structural Vector Autoregressive Analysis.
- Blanchard, O.J. & Quah, D. (1989). AER, 79(4), 655-673.
- Uhlig, H. (2005). JME, 52(2), 381-419.
- Amisano, G. & Giannini, C. (1997). Topics in Structural VAR Econometrics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la
from scipy.optimize import minimize


@dataclass
class SVARResults:
    """Results from SVAR estimation.

    Attributes
    ----------
    method : str
        Identification method used.
    A0_inv : ndarray of shape (K, K)
        Structural impact matrix (A_0^{-1} * B), maps structural shocks to reduced-form.
    A0 : ndarray of shape (K, K) or None
        A_0 matrix (if applicable).
    B : ndarray of shape (K, K) or None
        B matrix (if applicable).
    var_results : Any
        The underlying VARResults object.
    structural_shocks : ndarray of shape (T, K)
        Estimated structural shocks eps_t = A0_inv^{-1} * u_t.
    sigma_u : ndarray of shape (K, K)
        Reduced-form residual covariance.
    n_obs : int
        Number of observations.
    k_vars : int
        Number of endogenous variables.
    lags : int
        Number of lags in the underlying VAR.
    coefs : ndarray of shape (p, K, K)
        VAR coefficient matrices A_1, ..., A_p.
    intercept : ndarray of shape (K,) or None
        Intercept vector.
    resid : ndarray of shape (T, K)
        Reduced-form residuals u_t.
    accepted_draws : list[ndarray] | None
        List of accepted impact matrices (sign restrictions only).
    irf_draws : ndarray | None
        IRF for each accepted draw (sign restrictions only).
    """

    method: str
    A0_inv: NDArray[np.floating[Any]]
    A0: NDArray[np.floating[Any]] | None
    B: NDArray[np.floating[Any]] | None
    var_results: Any
    structural_shocks: NDArray[np.floating[Any]]
    sigma_u: NDArray[np.floating[Any]]
    n_obs: int
    k_vars: int
    lags: int
    coefs: NDArray[np.floating[Any]]
    intercept: NDArray[np.floating[Any]] | None
    resid: NDArray[np.floating[Any]]
    accepted_draws: list[NDArray[np.floating[Any]]] | None = None
    irf_draws: NDArray[np.floating[Any]] | None = None

    def irf(self, periods: int = 20) -> NDArray[np.floating[Any]]:
        """Compute structural impulse response functions.

        Parameters
        ----------
        periods : int
            Number of periods for the IRF.

        Returns
        -------
        ndarray of shape (periods+1, K, K)
            irf[h, i, j] = response of variable i to structural shock j at horizon h.
        """
        if self.method == "sign" and self.accepted_draws is not None:
            draws = np.array(
                [_compute_irf(self.coefs, imp, periods) for imp in self.accepted_draws]
            )
            return np.median(draws, axis=0)

        return _compute_irf(self.coefs, self.A0_inv, periods)

    def irf_with_bands(
        self, periods: int = 20, alpha: float = 0.16
    ) -> tuple[
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
    ]:
        """Compute IRF with confidence bands (sign restrictions only).

        Parameters
        ----------
        periods : int
            Number of periods.
        alpha : float
            Significance level for bands (default 0.16 for 68% bands).

        Returns
        -------
        median : ndarray of shape (periods+1, K, K)
        lower : ndarray of shape (periods+1, K, K)
        upper : ndarray of shape (periods+1, K, K)
        """
        if self.accepted_draws is None:
            irf_point = self.irf(periods)
            return irf_point, irf_point, irf_point

        draws = np.array(
            [_compute_irf(self.coefs, imp, periods) for imp in self.accepted_draws]
        )
        median = np.median(draws, axis=0)
        lower = np.percentile(draws, alpha * 100, axis=0)
        upper = np.percentile(draws, (1 - alpha) * 100, axis=0)
        return median, lower, upper

    def fevd(self, periods: int = 20) -> NDArray[np.floating[Any]]:
        """Compute forecast error variance decomposition.

        Returns
        -------
        ndarray of shape (periods+1, K, K)
            fevd[h, i, j] = fraction of h-step forecast error variance of
            variable i due to structural shock j.
        """
        irf_vals = self.irf(periods)
        k = self.k_vars
        n_horizons = periods + 1
        result = np.zeros((n_horizons, k, k))

        for h in range(n_horizons):
            cum_sq = np.zeros((k, k))
            for s in range(h + 1):
                cum_sq += irf_vals[s] ** 2
            total = cum_sq.sum(axis=1, keepdims=True)
            total = np.where(total == 0, 1.0, total)
            result[h] = cum_sq / total

        return result


class SVAR:
    """Structural Vector Autoregression.

    Parameters
    ----------
    var_results : VARResults
        Fitted reduced-form VAR results. Must have attributes:
        - coefs: ndarray (p, K, K) coefficient matrices
        - sigma_u or resid_cov: ndarray (K, K) residual covariance
        - resid: ndarray (T, K) residuals
        - n_obs: int
        - k_vars: int
        - lags: int
        - intercept: ndarray (K,) or None
    method : str
        Identification method: 'cholesky', 'ab', 'long_run', 'sign'.
    a_mat : ndarray or None
        A matrix for AB-model. NaN = free parameter, float = restricted.
    b_mat : ndarray or None
        B matrix for AB-model. NaN = free parameter, float = restricted.
    sign_restrictions : dict or None
        For sign restrictions method. Keys are tuples
        (shock_index, var_index, horizons) where horizons is a range or list.
        Values are '+' or '-'.
    """

    def __init__(
        self,
        var_results: Any,
        method: str = "cholesky",
        a_mat: NDArray[np.floating[Any]] | None = None,
        b_mat: NDArray[np.floating[Any]] | None = None,
        sign_restrictions: dict[tuple[Any, ...], str] | None = None,
    ) -> None:
        self.var_results = var_results
        self.method = method.lower()
        self.a_template = a_mat
        self.b_template = b_mat
        self.sign_restrictions = sign_restrictions

        if self.method not in ("cholesky", "ab", "long_run", "sign"):
            msg = (
                f"Unknown method: {self.method}. "
                "Use 'cholesky', 'ab', 'long_run', or 'sign'."
            )
            raise ValueError(msg)

        # Extract VAR attributes
        self.coefs = np.asarray(var_results.coefs)
        self.sigma_u = np.asarray(
            getattr(var_results, "sigma_u", getattr(var_results, "resid_cov", None))
        )
        if self.sigma_u is None:
            resid = np.asarray(var_results.resid)
            self.sigma_u = (resid.T @ resid) / len(resid)
        self.resid = np.asarray(var_results.resid)
        self.n_obs = int(getattr(var_results, "n_obs", getattr(var_results, "nobs", 0)))
        self.k_vars = int(getattr(var_results, "k_vars", getattr(var_results, "neqs", 0)))
        self.lags = int(getattr(var_results, "lags", getattr(var_results, "k_ar", 0)))
        self.intercept = getattr(var_results, "intercept", None)
        if self.intercept is not None:
            self.intercept = np.asarray(self.intercept)

    def fit(self, n_draws: int = 1000, max_draws: int = 50000) -> SVARResults:
        """Estimate the structural model.

        Parameters
        ----------
        n_draws : int
            Number of accepted draws for sign restrictions.
        max_draws : int
            Maximum total draws for sign restrictions.

        Returns
        -------
        SVARResults
        """
        if self.method == "cholesky":
            return self._fit_cholesky()
        if self.method == "ab":
            return self._fit_ab()
        if self.method == "long_run":
            return self._fit_long_run()
        if self.method == "sign":
            return self._fit_sign(n_draws=n_draws, max_draws=max_draws)
        msg = f"Unknown method: {self.method}"
        raise ValueError(msg)

    def _fit_cholesky(self) -> SVARResults:
        """Cholesky identification: P = chol(Sigma_u), A0_inv = P."""
        chol = la.cholesky(self.sigma_u, lower=True)
        a0_inv = chol
        a0 = la.inv(chol)

        structural_shocks = (a0 @ self.resid.T).T

        return SVARResults(
            method="cholesky",
            A0_inv=a0_inv,
            A0=a0,
            B=np.eye(self.k_vars),
            var_results=self.var_results,
            structural_shocks=structural_shocks,
            sigma_u=self.sigma_u,
            n_obs=self.n_obs,
            k_vars=self.k_vars,
            lags=self.lags,
            coefs=self.coefs,
            intercept=self.intercept,
            resid=self.resid,
        )

    def _fit_ab(self) -> SVARResults:
        """AB-model identification via concentrated MLE.

        A * u_t = B * eps_t
        A * Sigma_u * A' = B * B'
        """
        k = self.k_vars
        if self.a_template is None:
            self.a_template = np.eye(k)
        if self.b_template is None:
            self.b_template = np.full((k, k), np.nan)

        a_tmpl = np.array(self.a_template, dtype=float)
        b_tmpl = np.array(self.b_template, dtype=float)

        a_free_mask = np.isnan(a_tmpl)
        b_free_mask = np.isnan(b_tmpl)
        n_a_free = int(a_free_mask.sum())
        n_b_free = int(b_free_mask.sum())

        n_restrictions = k * k - n_a_free + k * k - n_b_free
        n_unique = k * (k + 1) // 2
        if n_restrictions < n_unique:
            msg = (
                f"Under-identified: {n_restrictions} restrictions but need "
                f"at least {n_unique} for {k} variables."
            )
            raise ValueError(msg)

        t_obs = self.n_obs
        sigma_u = self.sigma_u

        def _reconstruct(
            params: NDArray[np.floating[Any]],
        ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
            a_cur = a_tmpl.copy()
            b_cur = b_tmpl.copy()
            a_cur[a_free_mask] = params[:n_a_free]
            b_cur[b_free_mask] = params[n_a_free:]
            return a_cur, b_cur

        def _neg_loglik(params: NDArray[np.floating[Any]]) -> float:
            a_cur, b_cur = _reconstruct(params)
            try:
                la.inv(a_cur)
                b_inv = la.inv(b_cur)
            except la.LinAlgError:
                return 1e10

            log_det_a = np.log(np.abs(la.det(a_cur)) + 1e-300)
            log_det_b = np.log(np.abs(la.det(b_cur)) + 1e-300)

            inner = b_inv @ a_cur @ sigma_u @ a_cur.T @ b_inv.T
            trace_term = np.trace(inner)

            log_lik = (t_obs / 2.0) * (
                2.0 * log_det_a
                - 2.0 * log_det_b
                - k * np.log(2 * np.pi)
                - trace_term
            )
            return float(-log_lik)

        # Initial values from Cholesky
        chol = la.cholesky(sigma_u, lower=True)
        a0_init = la.inv(chol)
        b0_init = np.eye(k)

        x0_a = a0_init[a_free_mask] if n_a_free > 0 else np.array([])
        x0_b = b0_init[b_free_mask] if n_b_free > 0 else np.array([])
        x0 = np.concatenate([x0_a, x0_b])

        result = minimize(
            _neg_loglik,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-10, "fatol": 1e-10},
        )

        a_hat, b_hat = _reconstruct(result.x)
        a0_inv = la.inv(a_hat) @ b_hat

        structural_shocks = (la.inv(a0_inv) @ self.resid.T).T

        return SVARResults(
            method="ab",
            A0_inv=a0_inv,
            A0=a_hat,
            B=b_hat,
            var_results=self.var_results,
            structural_shocks=structural_shocks,
            sigma_u=self.sigma_u,
            n_obs=self.n_obs,
            k_vars=self.k_vars,
            lags=self.lags,
            coefs=self.coefs,
            intercept=self.intercept,
            resid=self.resid,
        )

    def _fit_long_run(self) -> SVARResults:
        """Blanchard-Quah long-run restrictions.

        Theta(1) = (I - A_1 - ... - A_p)^{-1}
        C(1) = Theta(1) * P  must be lower triangular.
        """
        k = self.k_vars
        p = self.lags

        a_sum = np.zeros((k, k))
        for s in range(p):
            a_sum += self.coefs[s]

        eye_k = np.eye(k)
        theta1 = la.inv(eye_k - a_sum)

        long_run_cov = theta1 @ self.sigma_u @ theta1.T
        long_run_cov = (long_run_cov + long_run_cov.T) / 2.0
        q_chol = la.cholesky(long_run_cov, lower=True)

        p_lr = la.inv(theta1) @ q_chol
        a0_inv = p_lr

        # Verify: Theta(1) * P_lr should be lower triangular
        c1 = theta1 @ p_lr
        for i in range(k):
            for j in range(i + 1, k):
                if abs(c1[i, j]) > 1e-6:
                    warnings.warn(
                        f"Long-run matrix C(1)[{i},{j}] = {c1[i, j]:.6f} "
                        "is not zero. Long-run restrictions may not hold exactly.",
                        stacklevel=2,
                    )

        a0 = la.inv(a0_inv)
        structural_shocks = (a0 @ self.resid.T).T

        return SVARResults(
            method="long_run",
            A0_inv=a0_inv,
            A0=a0,
            B=np.eye(k),
            var_results=self.var_results,
            structural_shocks=structural_shocks,
            sigma_u=self.sigma_u,
            n_obs=self.n_obs,
            k_vars=self.k_vars,
            lags=self.lags,
            coefs=self.coefs,
            intercept=self.intercept,
            resid=self.resid,
        )

    def _fit_sign(
        self, n_draws: int = 1000, max_draws: int = 50000
    ) -> SVARResults:
        """Sign restrictions identification (Uhlig 2005).

        Draw random orthogonal matrices Q, check sign constraints on IRF,
        keep valid draws.
        """
        if self.sign_restrictions is None or len(self.sign_restrictions) == 0:
            msg = "sign_restrictions must be provided for method='sign'."
            raise ValueError(msg)

        k = self.k_vars
        chol = la.cholesky(self.sigma_u, lower=True)

        parsed_restrictions: list[tuple[int, int, list[int], str]] = []
        for key, sign in self.sign_restrictions.items():
            shock_idx = (
                key[0] if isinstance(key[0], int) else int(key[0].split("_")[-1])
            )
            var_idx = int(key[1])
            horizons = list(key[2])
            parsed_restrictions.append((shock_idx, var_idx, horizons, sign))

        max_h = 0
        for _, _, horizons, _ in parsed_restrictions:
            max_h = max(max_h, max(horizons))

        accepted_impacts: list[NDArray[np.floating[Any]]] = []
        accepted_irfs: list[NDArray[np.floating[Any]]] = []

        rng = np.random.default_rng()

        total_draws = 0
        while len(accepted_impacts) < n_draws and total_draws < max_draws:
            total_draws += 1

            z = rng.standard_normal((k, k))
            q_orth, r_tri = la.qr(z)
            diag_signs = np.sign(np.diag(r_tri))
            diag_signs[diag_signs == 0] = 1.0
            q_orth = q_orth * diag_signs[np.newaxis, :]

            impact = chol @ q_orth

            irf_candidate = _compute_irf(self.coefs, impact, max_h)

            valid = True
            for shock_idx, var_idx, horizons, sign in parsed_restrictions:
                for h in horizons:
                    val = irf_candidate[h, var_idx, shock_idx]
                    if sign == "+" and val < 0:
                        valid = False
                        break
                    if sign == "-" and val > 0:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                accepted_impacts.append(impact)
                accepted_irfs.append(irf_candidate)

        if len(accepted_impacts) == 0:
            msg = (
                f"No valid draws found after {total_draws} attempts. "
                "Consider relaxing sign restrictions."
            )
            raise ValueError(msg)

        impacts_array = np.array(accepted_impacts)
        median_impact = np.median(impacts_array, axis=0)

        a0 = la.inv(median_impact)
        structural_shocks = (a0 @ self.resid.T).T

        irf_draws_array = np.array(accepted_irfs)

        return SVARResults(
            method="sign",
            A0_inv=median_impact,
            A0=a0,
            B=np.eye(k),
            var_results=self.var_results,
            structural_shocks=structural_shocks,
            sigma_u=self.sigma_u,
            n_obs=self.n_obs,
            k_vars=self.k_vars,
            lags=self.lags,
            coefs=self.coefs,
            intercept=self.intercept,
            resid=self.resid,
            accepted_draws=accepted_impacts,
            irf_draws=irf_draws_array,
        )


def _compute_irf(
    coefs: NDArray[np.floating[Any]],
    impact: NDArray[np.floating[Any]],
    periods: int,
) -> NDArray[np.floating[Any]]:
    """Compute structural IRF from VAR coefficients and impact matrix.

    Parameters
    ----------
    coefs : ndarray of shape (p, K, K)
        VAR coefficient matrices A_1, ..., A_p.
    impact : ndarray of shape (K, K)
        Structural impact matrix (A_0^{-1} * B).
    periods : int
        Number of periods.

    Returns
    -------
    ndarray of shape (periods+1, K, K)
        Structural impulse response functions.
    """
    p = coefs.shape[0]
    k = coefs.shape[1]
    n_horizons = periods + 1

    phi = np.zeros((n_horizons, k, k))
    phi[0] = np.eye(k)

    for h in range(1, n_horizons):
        for s in range(1, min(h, p) + 1):
            phi[h] += phi[h - s] @ coefs[s - 1]

    irf = np.zeros((n_horizons, k, k))
    for h in range(n_horizons):
        irf[h] = phi[h] @ impact

    return irf
