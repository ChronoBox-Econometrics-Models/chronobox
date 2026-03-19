"""
Bayesian Vector Autoregression (BVAR).

Priors:
- Minnesota (Litterman 1986)
- Normal-Wishart (conjugate)
- Sims-Zha (1998, via dummy observations)
- Flat (diffuse / OLS-equivalent)

References
----------
- Litterman, R.B. (1986). JBES, 4(1), 25-38.
- Sims, C.A. & Zha, T. (1998). IER, 39(4), 949-968.
- Koop, G. & Korobilis, D. (2010). Bayesian Multivariate Time Series Methods for
  Empirical Macroeconomics. Foundations and Trends in Econometrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la
from scipy.special import multigammaln


@dataclass
class BVARResults:
    """Results from Bayesian VAR estimation.

    Attributes
    ----------
    coefs_mean : ndarray of shape (p, K, K)
        Posterior mean of VAR coefficient matrices.
    coefs_draws : ndarray of shape (n_draws, p, K, K)
        Posterior draws of VAR coefficients.
    sigma_mean : ndarray of shape (K, K)
        Posterior mean of residual covariance.
    sigma_draws : ndarray of shape (n_draws, K, K)
        Posterior draws of Sigma.
    intercept_mean : ndarray of shape (K,)
        Posterior mean of intercept.
    intercept_draws : ndarray of shape (n_draws, K)
        Posterior draws of intercept.
    log_marginal_likelihood : float
        Log marginal likelihood (for model comparison).
    prior : str
        Prior used.
    n_obs : int
        Number of observations.
    k_vars : int
        Number of endogenous variables.
    lags : int
        Number of lags.
    endog : ndarray of shape (T_total, K)
        Original data.
    """

    coefs_mean: NDArray[np.floating[Any]]
    coefs_draws: NDArray[np.floating[Any]]
    sigma_mean: NDArray[np.floating[Any]]
    sigma_draws: NDArray[np.floating[Any]]
    intercept_mean: NDArray[np.floating[Any]]
    intercept_draws: NDArray[np.floating[Any]]
    log_marginal_likelihood: float
    prior: str
    n_obs: int
    k_vars: int
    lags: int
    endog: NDArray[np.floating[Any]]

    def forecast(
        self,
        steps: int = 12,
        n_draws: int | None = None,
    ) -> dict[str, NDArray[np.floating[Any]]]:
        """Bayesian forecast with predictive distribution.

        Parameters
        ----------
        steps : int
            Forecast horizon.
        n_draws : int or None
            Number of draws to use. If None, uses all available draws.

        Returns
        -------
        dict with keys:
            'mean': ndarray (steps, K) - mean forecast
            'median': ndarray (steps, K) - median forecast
            'lower_68': ndarray (steps, K) - 16th percentile
            'upper_68': ndarray (steps, K) - 84th percentile
            'lower_95': ndarray (steps, K) - 2.5th percentile
            'upper_95': ndarray (steps, K) - 97.5th percentile
            'draws': ndarray (n_draws, steps, K) - all forecast draws
        """
        K = self.k_vars
        p = self.lags
        total_draws = self.coefs_draws.shape[0]
        if n_draws is None:
            n_draws = total_draws
        n_draws = min(n_draws, total_draws)

        rng = np.random.default_rng(42)
        indices = rng.choice(total_draws, size=n_draws, replace=False)

        forecasts = np.zeros((n_draws, steps, K))

        for d_idx, draw_idx in enumerate(indices):
            coefs_d = self.coefs_draws[draw_idx]  # (p, K, K)
            sigma_d = self.sigma_draws[draw_idx]  # (K, K)
            intercept_d = self.intercept_draws[draw_idx]  # (K,)

            # Ensure sigma_d is symmetric positive definite
            sigma_d = (sigma_d + sigma_d.T) / 2.0
            eigvals = la.eigvalsh(sigma_d)
            if np.any(eigvals <= 0):
                sigma_d += (abs(eigvals.min()) + 1e-8) * np.eye(K)

            # Buffer: last p observations + forecast slots
            history = list(self.endog[-(p):])  # list of (K,) arrays

            for h in range(steps):
                pred = intercept_d.copy()
                for s in range(p):
                    pred += history[-(s + 1)] @ coefs_d[s].T

                noise = rng.multivariate_normal(np.zeros(K), sigma_d)
                y_new = pred + noise
                forecasts[d_idx, h] = y_new
                history.append(y_new)

        return {
            "mean": np.mean(forecasts, axis=0),
            "median": np.median(forecasts, axis=0),
            "lower_68": np.percentile(forecasts, 16, axis=0),
            "upper_68": np.percentile(forecasts, 84, axis=0),
            "lower_95": np.percentile(forecasts, 2.5, axis=0),
            "upper_95": np.percentile(forecasts, 97.5, axis=0),
            "draws": forecasts,
        }

    def irf(
        self, periods: int = 20, method: str = "cholesky"
    ) -> NDArray[np.floating[Any]]:
        """Compute impulse response functions using posterior mean.

        Parameters
        ----------
        periods : int
            Number of periods.
        method : str
            Identification method ('cholesky').

        Returns
        -------
        ndarray of shape (periods+1, K, K)
        """
        K = self.k_vars
        p = self.lags
        H = periods + 1

        coefs = self.coefs_mean  # (p, K, K)

        if method == "cholesky":
            sigma = (self.sigma_mean + self.sigma_mean.T) / 2.0
            P = la.cholesky(sigma, lower=True)
        else:
            P = np.eye(K)

        Phi = np.zeros((H, K, K))
        Phi[0] = np.eye(K)
        for h in range(1, H):
            for s in range(1, min(h, p) + 1):
                Phi[h] += Phi[h - s] @ coefs[s - 1]

        irf_out = np.zeros((H, K, K))
        for h in range(H):
            irf_out[h] = Phi[h] @ P

        return irf_out

    def irf_draws_compute(
        self, periods: int = 20, method: str = "cholesky"
    ) -> NDArray[np.floating[Any]]:
        """Compute IRF for each posterior draw.

        Returns
        -------
        ndarray of shape (n_draws, periods+1, K, K)
        """
        n_draws = self.coefs_draws.shape[0]
        K = self.k_vars
        p = self.lags
        H = periods + 1

        irf_all = np.zeros((n_draws, H, K, K))

        for d in range(n_draws):
            coefs = self.coefs_draws[d]
            sigma = self.sigma_draws[d]
            sigma = (sigma + sigma.T) / 2.0

            try:
                P = la.cholesky(sigma, lower=True)
            except la.LinAlgError:
                sigma += 1e-8 * np.eye(K)
                P = la.cholesky(sigma, lower=True)

            Phi = np.zeros((H, K, K))
            Phi[0] = np.eye(K)
            for h in range(1, H):
                for s in range(1, min(h, p) + 1):
                    Phi[h] += Phi[h - s] @ coefs[s - 1]

            for h in range(H):
                irf_all[d, h] = Phi[h] @ P

        return irf_all


class BayesianVAR:
    """Bayesian Vector Autoregression.

    Parameters
    ----------
    lags : int
        Number of lags (default=1).
    prior : str
        Prior type: 'minnesota', 'normal_wishart', 'sims_zha', 'flat'.
    **prior_kwargs
        Additional prior hyperparameters:
        Minnesota: lambda_1 (float), lambda_2 (float), lambda_3 (float), delta (float)
        Normal-Wishart: V_0 (ndarray), S_0 (ndarray), v_0 (float)
    """

    def __init__(
        self, lags: int = 1, prior: str = "minnesota", **prior_kwargs: Any
    ) -> None:
        self.lags = lags
        self.prior = prior.lower()
        self.prior_kwargs = prior_kwargs

        valid_priors = ("minnesota", "normal_wishart", "sims_zha", "flat")
        if self.prior not in valid_priors:
            msg = f"Unknown prior: {self.prior}. Use one of {valid_priors}."
            raise ValueError(msg)

    def fit(
        self,
        endog: NDArray[np.floating[Any]],
        n_draws: int = 5000,
        burnin: int = 1000,
        seed: int = 42,
    ) -> BVARResults:
        """Fit Bayesian VAR.

        Parameters
        ----------
        endog : ndarray of shape (T, K)
            Endogenous variables.
        n_draws : int
            Number of posterior draws to retain.
        burnin : int
            Number of burn-in draws to discard.
        seed : int
            Random seed.

        Returns
        -------
        BVARResults
        """
        endog = np.asarray(endog, dtype=float)
        if endog.ndim == 1:
            endog = endog.reshape(-1, 1)

        T_total, K = endog.shape
        p = self.lags
        T = T_total - p

        if T <= 0:
            msg = f"Not enough observations: T={T_total}, lags={p}"
            raise ValueError(msg)

        # Construct Y and Z matrices
        Y = endog[p:]  # (T, K)
        Z = np.zeros((T, K * p + 1))
        for t in range(T):
            for s in range(p):
                Z[t, s * K : (s + 1) * K] = endog[p + t - s - 1]
            Z[t, -1] = 1.0  # intercept

        M = K * p + 1  # number of regressors per equation

        # OLS estimates
        ZtZ = Z.T @ Z
        ZtY = Z.T @ Y
        try:
            B_ols = la.solve(ZtZ, ZtY)
        except la.LinAlgError:
            B_ols = la.lstsq(Z, Y)[0]
        U_ols = Y - Z @ B_ols
        Sigma_ols = (U_ols.T @ U_ols) / T

        rng = np.random.default_rng(seed)

        if self.prior == "flat":
            return self._fit_flat(
                Y, Z, B_ols, Sigma_ols, endog, T, K, p, M, n_draws, burnin, rng
            )
        if self.prior == "minnesota":
            return self._fit_minnesota(
                Y, Z, B_ols, Sigma_ols, endog, T, K, p, M, n_draws, burnin, rng
            )
        if self.prior == "normal_wishart":
            return self._fit_normal_wishart(
                Y, Z, B_ols, Sigma_ols, endog, T, K, p, M, n_draws, burnin, rng
            )
        # sims_zha
        return self._fit_sims_zha(
            Y, Z, B_ols, Sigma_ols, endog, T, K, p, M, n_draws, burnin, rng
        )

    # ------------------------------------------------------------------
    # Flat (diffuse) prior
    # ------------------------------------------------------------------

    def _fit_flat(
        self,
        Y: NDArray[np.floating[Any]],
        Z: NDArray[np.floating[Any]],
        B_ols: NDArray[np.floating[Any]],
        Sigma_ols: NDArray[np.floating[Any]],
        endog: NDArray[np.floating[Any]],
        T: int,
        K: int,
        p: int,
        M: int,
        n_draws: int,
        burnin: int,
        rng: np.random.Generator,
    ) -> BVARResults:
        ZtZ = Z.T @ Z
        ZtZ_inv = la.inv(ZtZ)

        v_post = T - M
        S_post = T * Sigma_ols
        S_post = (S_post + S_post.T) / 2.0

        total_draws = n_draws + burnin
        coefs_all = np.zeros((total_draws, p, K, K))
        sigma_all = np.zeros((total_draws, K, K))
        intercept_all = np.zeros((total_draws, K))

        for d in range(total_draws):
            Sigma_d = _draw_inv_wishart(S_post, v_post, rng)
            B_d = _draw_matrix_normal(B_ols, Sigma_d, ZtZ_inv, rng)

            for s in range(p):
                coefs_all[d, s] = B_d[s * K : (s + 1) * K].T
            intercept_all[d] = B_d[-1]
            sigma_all[d] = Sigma_d

        coefs_draws = coefs_all[burnin:]
        sigma_draws = sigma_all[burnin:]
        intercept_draws = intercept_all[burnin:]

        log_det_sigma = np.log(la.det(Sigma_ols) + 1e-300)
        log_ml = (
            -(T * K / 2.0) * np.log(2 * np.pi)
            - (T / 2.0) * log_det_sigma
            - T * K / 2.0
        )

        return BVARResults(
            coefs_mean=np.mean(coefs_draws, axis=0),
            coefs_draws=coefs_draws,
            sigma_mean=np.mean(sigma_draws, axis=0),
            sigma_draws=sigma_draws,
            intercept_mean=np.mean(intercept_draws, axis=0),
            intercept_draws=intercept_draws,
            log_marginal_likelihood=float(log_ml),
            prior="flat",
            n_obs=T,
            k_vars=K,
            lags=p,
            endog=endog,
        )

    # ------------------------------------------------------------------
    # Minnesota prior (Litterman 1986)
    # ------------------------------------------------------------------

    def _fit_minnesota(
        self,
        Y: NDArray[np.floating[Any]],
        Z: NDArray[np.floating[Any]],
        B_ols: NDArray[np.floating[Any]],
        Sigma_ols: NDArray[np.floating[Any]],
        endog: NDArray[np.floating[Any]],
        T: int,
        K: int,
        p: int,
        M: int,
        n_draws: int,
        burnin: int,
        rng: np.random.Generator,
    ) -> BVARResults:
        lambda_1 = self.prior_kwargs.get("lambda_1", 0.1)
        lambda_2 = self.prior_kwargs.get("lambda_2", 0.5)
        lambda_3 = self.prior_kwargs.get("lambda_3", 1.0)
        delta = self.prior_kwargs.get("delta", 1.0)

        si = _ar_residual_std(endog, p)

        # Prior mean B_0: (M, K)
        B_0 = np.zeros((M, K))
        for i in range(K):
            B_0[i, i] = delta  # first lag, own variable

        total_draws_needed = n_draws + burnin
        coefs_all = np.zeros((total_draws_needed, p, K, K))
        sigma_all = np.zeros((total_draws_needed, K, K))
        intercept_all = np.zeros((total_draws_needed, K))

        ZtZ = Z.T @ Z

        for eq in range(K):
            v_prior = _minnesota_prior_variance(eq, K, p, M, si, lambda_1, lambda_2, lambda_3)
            V_0_inv_eq = np.diag(1.0 / (v_prior + 1e-20))
            b_0_eq = B_0[:, eq]

            V_post_inv = V_0_inv_eq + ZtZ
            V_post = la.inv(V_post_inv)
            b_post = V_post @ (V_0_inv_eq @ b_0_eq + Z.T @ Y[:, eq])

            sigma_eq = si[eq] ** 2

            for d in range(total_draws_needed):
                b_d = rng.multivariate_normal(b_post, sigma_eq * V_post)
                for s in range(p):
                    coefs_all[d, s, eq, :] = b_d[s * K : (s + 1) * K]
                intercept_all[d, eq] = b_d[-1]

        for d in range(total_draws_needed):
            sigma_all[d] = Sigma_ols

        coefs_draws = coefs_all[burnin:]
        sigma_draws = sigma_all[burnin:]
        intercept_draws = intercept_all[burnin:]

        log_ml = self._compute_minnesota_marginal_likelihood(
            Y, Z, B_0, si, K, p, M, T, lambda_1, lambda_2, lambda_3
        )

        return BVARResults(
            coefs_mean=np.mean(coefs_draws, axis=0),
            coefs_draws=coefs_draws,
            sigma_mean=Sigma_ols,
            sigma_draws=sigma_draws,
            intercept_mean=np.mean(intercept_draws, axis=0),
            intercept_draws=intercept_draws,
            log_marginal_likelihood=float(log_ml),
            prior="minnesota",
            n_obs=T,
            k_vars=K,
            lags=p,
            endog=endog,
        )

    def _compute_minnesota_marginal_likelihood(
        self,
        Y: NDArray[np.floating[Any]],
        Z: NDArray[np.floating[Any]],
        B_0: NDArray[np.floating[Any]],
        si: NDArray[np.floating[Any]],
        K: int,
        p: int,
        M: int,
        T: int,
        lambda_1: float,
        lambda_2: float,
        lambda_3: float,
    ) -> float:
        log_ml = 0.0
        ZtZ = Z.T @ Z

        for eq in range(K):
            v_prior = _minnesota_prior_variance(eq, K, p, M, si, lambda_1, lambda_2, lambda_3)
            V_0_eq = np.diag(v_prior + 1e-20)
            V_0_inv_eq = np.diag(1.0 / (v_prior + 1e-20))
            b_0_eq = B_0[:, eq]
            sigma_eq = si[eq] ** 2

            V_post_inv = V_0_inv_eq + ZtZ
            V_post = la.inv(V_post_inv)
            b_post = V_post @ (V_0_inv_eq @ b_0_eq + Z.T @ Y[:, eq])

            y_eq = Y[:, eq]
            SSR_ols = (
                y_eq @ y_eq - b_post @ V_post_inv @ b_post + b_0_eq @ V_0_inv_eq @ b_0_eq
            )

            log_ml_eq = (
                -(T / 2.0) * np.log(2 * np.pi * sigma_eq)
                + 0.5 * np.log(la.det(V_post) + 1e-300)
                - 0.5 * np.log(la.det(V_0_eq) + 1e-300)
                - SSR_ols / (2.0 * sigma_eq)
            )
            log_ml += log_ml_eq

        return float(log_ml)

    # ------------------------------------------------------------------
    # Normal-Wishart conjugate prior
    # ------------------------------------------------------------------

    def _fit_normal_wishart(
        self,
        Y: NDArray[np.floating[Any]],
        Z: NDArray[np.floating[Any]],
        B_ols: NDArray[np.floating[Any]],
        Sigma_ols: NDArray[np.floating[Any]],
        endog: NDArray[np.floating[Any]],
        T: int,
        K: int,
        p: int,
        M: int,
        n_draws: int,
        burnin: int,
        rng: np.random.Generator,
    ) -> BVARResults:
        V_0 = self.prior_kwargs.get("V_0", np.eye(M) * 10.0)
        v_0 = self.prior_kwargs.get("v_0", float(K + 2))
        S_0 = self.prior_kwargs.get("S_0", np.eye(K) * 0.01)
        B_0 = self.prior_kwargs.get("B_0", np.zeros((M, K)))

        if isinstance(V_0, (int, float)):
            V_0 = np.eye(M) * V_0
        if isinstance(S_0, (int, float)):
            S_0 = np.eye(K) * S_0

        V_0 = np.asarray(V_0, dtype=float)
        S_0 = np.asarray(S_0, dtype=float)
        B_0 = np.asarray(B_0, dtype=float)

        V_0_inv = la.inv(V_0)
        ZtZ = Z.T @ Z

        V_post_inv = V_0_inv + ZtZ
        V_post = la.inv(V_post_inv)
        B_post = V_post @ (V_0_inv @ B_0 + Z.T @ Y)

        v_post = v_0 + T
        S_post = (
            S_0
            + (Y - Z @ B_ols).T @ (Y - Z @ B_ols)
            + (B_ols - B_0).T
            @ la.inv(la.inv(V_0_inv) + la.inv(ZtZ))
            @ (B_ols - B_0)
        )
        S_post = (S_post + S_post.T) / 2.0

        total_draws_needed = n_draws + burnin
        coefs_all = np.zeros((total_draws_needed, p, K, K))
        sigma_all = np.zeros((total_draws_needed, K, K))
        intercept_all = np.zeros((total_draws_needed, K))

        for d in range(total_draws_needed):
            Sigma_d = _draw_inv_wishart(S_post, v_post, rng)
            B_d = _draw_matrix_normal(B_post, Sigma_d, V_post, rng)

            for s in range(p):
                coefs_all[d, s] = B_d[s * K : (s + 1) * K].T
            intercept_all[d] = B_d[-1]
            sigma_all[d] = Sigma_d

        coefs_draws = coefs_all[burnin:]
        sigma_draws = sigma_all[burnin:]
        intercept_draws = intercept_all[burnin:]

        log_ml = self._compute_nw_marginal_likelihood(
            V_0_inv, V_post_inv, S_0, S_post, v_0, v_post, T, K
        )

        return BVARResults(
            coefs_mean=np.mean(coefs_draws, axis=0),
            coefs_draws=coefs_draws,
            sigma_mean=np.mean(sigma_draws, axis=0),
            sigma_draws=sigma_draws,
            intercept_mean=np.mean(intercept_draws, axis=0),
            intercept_draws=intercept_draws,
            log_marginal_likelihood=float(log_ml),
            prior="normal_wishart",
            n_obs=T,
            k_vars=K,
            lags=p,
            endog=endog,
        )

    def _compute_nw_marginal_likelihood(
        self,
        V_0_inv: NDArray[np.floating[Any]],
        V_post_inv: NDArray[np.floating[Any]],
        S_0: NDArray[np.floating[Any]],
        S_post: NDArray[np.floating[Any]],
        v_0: float,
        v_post: float,
        T: int,
        K: int,
    ) -> float:
        log_ml = (
            -(T * K / 2.0) * np.log(np.pi)
            + multigammaln(v_post / 2.0, K)
            - multigammaln(v_0 / 2.0, K)
            + (v_0 / 2.0) * np.log(la.det(S_0) + 1e-300)
            - (v_post / 2.0) * np.log(la.det(S_post) + 1e-300)
            + (K / 2.0) * np.log(la.det(V_0_inv) + 1e-300)
            - (K / 2.0) * np.log(la.det(V_post_inv) + 1e-300)
        )
        return float(log_ml)

    # ------------------------------------------------------------------
    # Sims-Zha prior (via dummy observations)
    # ------------------------------------------------------------------

    def _fit_sims_zha(
        self,
        Y: NDArray[np.floating[Any]],
        Z: NDArray[np.floating[Any]],
        B_ols: NDArray[np.floating[Any]],
        Sigma_ols: NDArray[np.floating[Any]],
        endog: NDArray[np.floating[Any]],
        T: int,
        K: int,
        p: int,
        M: int,
        n_draws: int,
        burnin: int,
        rng: np.random.Generator,
    ) -> BVARResults:
        lambda_1 = self.prior_kwargs.get("lambda_1", 0.1)
        lambda_3 = self.prior_kwargs.get("lambda_3", 1.0)
        mu_5 = self.prior_kwargs.get("mu_5", 1.0)
        mu_6 = self.prior_kwargs.get("mu_6", 1.0)
        delta = self.prior_kwargs.get("delta", 1.0)

        y_bar = np.mean(endog[:p], axis=0)

        si = _ar_residual_std(endog, p)

        dummy_y_list: list[NDArray[np.floating[Any]]] = []
        dummy_z_list: list[NDArray[np.floating[Any]]] = []

        # 1. Minnesota-type dummies (unit root prior)
        for s in range(1, p + 1):
            Y_d = np.zeros((K, K))
            Z_d = np.zeros((K, M))
            for i in range(K):
                scale = delta * si[i] / lambda_1 * s**lambda_3
                Y_d[i, i] = scale
                Z_d[i, (s - 1) * K + i] = scale
            dummy_y_list.append(Y_d)
            dummy_z_list.append(Z_d)

        # 2. Sum-of-coefficients prior (Doan, Litterman & Sims)
        if mu_5 > 0:
            Y_soc = np.diag(delta * y_bar) / mu_5
            Z_soc = np.zeros((K, M))
            for s in range(p):
                Z_soc[:, s * K : (s + 1) * K] = np.diag(delta * y_bar) / mu_5
            Z_soc[:, -1] = 0.0
            dummy_y_list.append(Y_soc)
            dummy_z_list.append(Z_soc)

        # 3. Co-persistence prior
        if mu_6 > 0:
            Y_cop = (delta * y_bar).reshape(1, K) / mu_6
            Z_cop = np.zeros((1, M))
            for s in range(p):
                Z_cop[0, s * K : (s + 1) * K] = delta * y_bar / mu_6
            Z_cop[0, -1] = 1.0 / mu_6
            dummy_y_list.append(Y_cop)
            dummy_z_list.append(Z_cop)

        Y_dummy = np.vstack(dummy_y_list)
        Z_dummy = np.vstack(dummy_z_list)

        Y_aug = np.vstack([Y, Y_dummy])
        Z_aug = np.vstack([Z, Z_dummy])
        T_aug = Y_aug.shape[0]

        ZtZ_aug = Z_aug.T @ Z_aug
        try:
            B_post = la.solve(ZtZ_aug, Z_aug.T @ Y_aug)
        except la.LinAlgError:
            B_post = la.lstsq(Z_aug, Y_aug)[0]

        U_aug = Y_aug - Z_aug @ B_post
        S_post = U_aug.T @ U_aug
        S_post = (S_post + S_post.T) / 2.0
        v_post = float(T_aug - M)

        ZtZ_aug_inv = la.inv(ZtZ_aug)
        total_draws_needed = n_draws + burnin
        coefs_all = np.zeros((total_draws_needed, p, K, K))
        sigma_all = np.zeros((total_draws_needed, K, K))
        intercept_all = np.zeros((total_draws_needed, K))

        for d in range(total_draws_needed):
            Sigma_d = _draw_inv_wishart(S_post, v_post, rng)
            B_d = _draw_matrix_normal(B_post, Sigma_d, ZtZ_aug_inv, rng)

            for s in range(p):
                coefs_all[d, s] = B_d[s * K : (s + 1) * K].T
            intercept_all[d] = B_d[-1]
            sigma_all[d] = Sigma_d

        coefs_draws = coefs_all[burnin:]
        sigma_draws = sigma_all[burnin:]
        intercept_draws = intercept_all[burnin:]

        # Marginal likelihood via dummy-obs approach
        U_dummy = Y_dummy - Z_dummy @ B_post
        S_prior = U_dummy.T @ U_dummy
        S_prior = (S_prior + S_prior.T) / 2.0
        T_dummy = Y_dummy.shape[0]
        v_prior = float(max(T_dummy - M, K + 1))

        V_0_inv_approx = Z_dummy.T @ Z_dummy
        V_post_inv_approx = ZtZ_aug

        log_ml = (
            -(T * K / 2.0) * np.log(np.pi)
            + multigammaln(v_post / 2.0, K)
            - multigammaln(v_prior / 2.0, K)
            + (v_prior / 2.0)
            * np.log(la.det(S_prior + np.eye(K) * 1e-10) + 1e-300)
            - (v_post / 2.0)
            * np.log(la.det(S_post + np.eye(K) * 1e-10) + 1e-300)
            + (K / 2.0)
            * np.log(la.det(V_0_inv_approx + np.eye(M) * 1e-10) + 1e-300)
            - (K / 2.0)
            * np.log(la.det(V_post_inv_approx + np.eye(M) * 1e-10) + 1e-300)
        )

        return BVARResults(
            coefs_mean=np.mean(coefs_draws, axis=0),
            coefs_draws=coefs_draws,
            sigma_mean=np.mean(sigma_draws, axis=0),
            sigma_draws=sigma_draws,
            intercept_mean=np.mean(intercept_draws, axis=0),
            intercept_draws=intercept_draws,
            log_marginal_likelihood=float(log_ml),
            prior="sims_zha",
            n_obs=T,
            k_vars=K,
            lags=p,
            endog=endog,
        )


# ======================================================================
# Helper functions
# ======================================================================


def _ar_residual_std(
    endog: NDArray[np.floating[Any]], p: int
) -> NDArray[np.floating[Any]]:
    """Compute std of residuals from univariate AR(p) for each variable."""
    K = endog.shape[1]
    si = np.zeros(K)
    for i in range(K):
        y_i = endog[:, i]
        T_i = len(y_i) - p
        if T_i <= p:
            si[i] = max(np.std(y_i), 1e-10)
            continue
        Z_i = np.zeros((T_i, p + 1))
        for s_lag in range(p):
            idx_start = p - s_lag - 1
            idx_end = T_i + p - s_lag - 1
            Z_i[:, s_lag] = y_i[idx_start:idx_end]
        Z_i[:, -1] = 1.0
        y_dep = y_i[p:]
        try:
            b_i = la.lstsq(Z_i, y_dep)[0]
            resid_i = y_dep - Z_i @ b_i
            si[i] = np.std(resid_i, ddof=p + 1)
        except Exception:
            si[i] = np.std(y_i)
        if si[i] < 1e-10:
            si[i] = 1e-10
    return si


def _minnesota_prior_variance(
    eq: int,
    K: int,
    p: int,
    M: int,
    si: NDArray[np.floating[Any]],
    lambda_1: float,
    lambda_2: float,
    lambda_3: float,
) -> NDArray[np.floating[Any]]:
    """Compute Minnesota prior variance for a single equation."""
    v_prior = np.zeros(M)
    for s in range(p):
        lag = s + 1
        for j in range(K):
            idx = s * K + j
            if j == eq:
                v_prior[idx] = (lambda_1 / lag**lambda_3) ** 2
            else:
                v_prior[idx] = (
                    (lambda_1 * lambda_2 / lag**lambda_3) ** 2
                    * (si[eq] / si[j]) ** 2
                )
    # Intercept: large variance (uninformative)
    v_prior[-1] = (lambda_1 * 10.0) ** 2
    return v_prior


def _draw_inv_wishart(
    S: NDArray[np.floating[Any]],
    v: float,
    rng: np.random.Generator,
) -> NDArray[np.floating[Any]]:
    """Draw from Inverse-Wishart distribution IW(S, v)."""
    K = S.shape[0]
    v_int = max(int(v), K + 1)

    S_sym = (S + S.T) / 2.0
    eigvals = la.eigvalsh(S_sym)
    if np.any(eigvals <= 0):
        S_sym += (abs(eigvals.min()) + 1e-8) * np.eye(K)

    S_inv = la.inv(S_sym)
    S_inv = (S_inv + S_inv.T) / 2.0
    eigvals = la.eigvalsh(S_inv)
    if np.any(eigvals <= 0):
        S_inv += (abs(eigvals.min()) + 1e-8) * np.eye(K)

    L = la.cholesky(S_inv, lower=True)
    X = rng.standard_normal((v_int, K)) @ L.T
    W = X.T @ X

    try:
        result = la.inv(W)
    except la.LinAlgError:
        result = la.inv(W + np.eye(K) * 1e-8)

    return (result + result.T) / 2.0


def _draw_matrix_normal(
    M_mean: NDArray[np.floating[Any]],
    U: NDArray[np.floating[Any]],
    V: NDArray[np.floating[Any]],
    rng: np.random.Generator,
) -> NDArray[np.floating[Any]]:
    """Draw from Matrix Normal distribution MN(M, U, V).

    vec(X) ~ N(vec(M), U kron V)
    """
    n, p_dim = M_mean.shape

    V_sym = (V + V.T) / 2.0
    eigvals = la.eigvalsh(V_sym)
    if np.any(eigvals <= 0):
        V_sym += (abs(eigvals.min()) + 1e-8) * np.eye(n)
    L_V = la.cholesky(V_sym, lower=True)

    U_sym = (U + U.T) / 2.0
    eigvals = la.eigvalsh(U_sym)
    if np.any(eigvals <= 0):
        U_sym += (abs(eigvals.min()) + 1e-8) * np.eye(p_dim)
    L_U = la.cholesky(U_sym, lower=True)

    Z = rng.standard_normal((n, p_dim))
    return M_mean + L_V @ Z @ L_U.T
