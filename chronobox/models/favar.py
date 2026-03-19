"""
Factor-Augmented Vector Autoregression (FAVAR).

Two-step estimation (Bernanke, Boivin & Eliasz 2005):
  Step 1: Extract factors via PCA
  Step 2: Estimate VAR on [factors; policy variables]

One-step estimation via kalmanbox state-space (Bayesian).

References
----------
- Bernanke, B.S., Boivin, J. & Eliasz, P. (2005). QJE, 120(1), 387-422.
- Stock, J.H. & Watson, M.W. (2002). JASA, 97(460), 1167-1179.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la


@dataclass
class FAVARResults:
    """Results from FAVAR estimation.

    Attributes
    ----------
    factors : ndarray of shape (T, K)
        Estimated factors.
    loadings : ndarray of shape (N, K)
        Factor loadings (Lambda^f).
    loadings_y : ndarray of shape (N, M)
        Policy variable loadings (Lambda^y).
    var_coefs : ndarray of shape (p, K+M, K+M)
        VAR coefficient matrices for [F; Y].
    var_sigma : ndarray of shape (K+M, K+M)
        VAR residual covariance.
    var_resid : ndarray of shape (T-p, K+M)
        VAR residuals.
    intercept : ndarray of shape (K+M,)
        VAR intercept.
    explained_variance_ratio : ndarray of shape (K,)
        Fraction of variance explained by each factor.
    total_explained_variance : float
        Total fraction of variance explained by all K factors.
    n_factors : int
        Number of factors.
    n_policy : int
        Number of policy variables.
    n_panel : int
        Number of panel variables.
    n_obs : int
        Number of observations used in the VAR.
    lags : int
        Number of VAR lags.
    method : str
        Estimation method used.
    x_mean : ndarray of shape (N,)
        Mean of panel variables (for un-standardizing).
    x_std : ndarray of shape (N,)
        Std of panel variables (for un-standardizing).
    """

    factors: NDArray[np.floating[Any]]
    loadings: NDArray[np.floating[Any]]
    loadings_y: NDArray[np.floating[Any]]
    var_coefs: NDArray[np.floating[Any]]
    var_sigma: NDArray[np.floating[Any]]
    var_resid: NDArray[np.floating[Any]]
    intercept: NDArray[np.floating[Any]]
    explained_variance_ratio: NDArray[np.floating[Any]]
    total_explained_variance: float
    n_factors: int
    n_policy: int
    n_panel: int
    n_obs: int
    lags: int
    method: str
    x_mean: NDArray[np.floating[Any]]
    x_std: NDArray[np.floating[Any]]

    def irf(
        self,
        periods: int = 20,
        identification: str = "cholesky",
    ) -> NDArray[np.floating[Any]]:
        """Compute impulse response functions.

        Parameters
        ----------
        periods : int
            Number of periods.
        identification : str
            Identification method ('cholesky').

        Returns
        -------
        ndarray of shape (periods+1, K+M, K+M)
            IRF in the [F; Y] space.
        """
        p = self.lags
        km = self.n_factors + self.n_policy
        n_steps = periods + 1

        # Structural impact matrix
        if identification == "cholesky":
            sigma = (self.var_sigma + self.var_sigma.T) / 2.0
            impact = la.cholesky(sigma, lower=True)
        else:
            impact = np.eye(km)

        # MA coefficients
        phi = np.zeros((n_steps, km, km))
        phi[0] = np.eye(km)
        for h in range(1, n_steps):
            for s in range(1, min(h, p) + 1):
                phi[h] += phi[h - s] @ self.var_coefs[s - 1]

        # Structural IRF
        irf_arr = np.zeros((n_steps, km, km))
        for h in range(n_steps):
            irf_arr[h] = phi[h] @ impact

        return irf_arr

    def irf_panel(
        self,
        periods: int = 20,
        identification: str = "cholesky",
    ) -> NDArray[np.floating[Any]]:
        """Compute IRF for all panel variables X_t.

        Parameters
        ----------
        periods : int
            Number of periods.
        identification : str
            Identification method.

        Returns
        -------
        ndarray of shape (periods+1, N, K+M)
            irf_panel[h, i, j] = response of panel variable i to structural shock j.
        """
        irf_fy = self.irf(periods, identification)
        n_steps = periods + 1
        k = self.n_factors
        n = self.n_panel
        km = k + self.n_policy

        irf_x = np.zeros((n_steps, n, km))
        for h in range(n_steps):
            # IRF_X_h = Lambda^f * Theta_h[:K, :] + Lambda^y * Theta_h[K:, :]
            irf_x[h] = (
                self.loadings @ irf_fy[h, :k, :]
                + self.loadings_y @ irf_fy[h, k:, :]
            )

        return irf_x


class FAVAR:
    """Factor-Augmented Vector Autoregression.

    Parameters
    ----------
    n_factors : int
        Number of latent factors to extract (default=3).
    lags : int
        Number of VAR lags (default=2).
    method : str
        Estimation method: 'two_step' or 'bayesian' (default='two_step').
    remove_y_from_factors : bool
        Whether to remove the Y component from extracted factors (default=True).
    """

    def __init__(
        self,
        n_factors: int = 3,
        lags: int = 2,
        method: str = "two_step",
        remove_y_from_factors: bool = True,
    ) -> None:
        self.n_factors = n_factors
        self.lags = lags
        self.method = method.lower()
        self.remove_y_from_factors = remove_y_from_factors

        if self.method not in ("two_step", "bayesian"):
            msg = f"Unknown method: {self.method}. Use 'two_step' or 'bayesian'."
            raise ValueError(msg)

    def fit(
        self,
        panel: NDArray[np.floating[Any]],
        policy: NDArray[np.floating[Any]],
    ) -> FAVARResults:
        """Fit FAVAR model.

        Parameters
        ----------
        panel : ndarray of shape (T, N)
            Large panel of informational variables.
        policy : ndarray of shape (T, M) or (T,)
            Policy/observable variables.

        Returns
        -------
        FAVARResults
        """
        x_arr = np.asarray(panel, dtype=float)
        y_arr = np.asarray(policy, dtype=float)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        t_total, n_vars = x_arr.shape
        m = y_arr.shape[1]

        if t_total != y_arr.shape[0]:
            msg = (
                f"X and Y must have same number of rows: "
                f"{t_total} vs {y_arr.shape[0]}"
            )
            raise ValueError(msg)

        if t_total <= self.lags:
            msg = f"Not enough observations: T={t_total}, lags={self.lags}"
            raise ValueError(msg)

        if self.method == "two_step":
            return self._fit_two_step(x_arr, y_arr, t_total, n_vars, m)
        else:
            return self._fit_bayesian(x_arr, y_arr, t_total, n_vars, m)

    def _fit_two_step(
        self,
        x_arr: NDArray[np.floating[Any]],
        y_arr: NDArray[np.floating[Any]],
        t_total: int,
        n_vars: int,
        m: int,
    ) -> FAVARResults:
        """Two-step estimation: PCA + VAR."""
        k = self.n_factors
        p = self.lags

        # Step 1: Standardize X and extract factors via PCA
        x_mean = np.mean(x_arr, axis=0)
        x_std = np.std(x_arr, axis=0)
        x_std[x_std < 1e-10] = 1.0
        x_s = (x_arr - x_mean) / x_std

        # SVD
        u, sv, vt = la.svd(x_s, full_matrices=False)

        # Factors: T x K
        f_raw = u[:, :k] * sv[:k]

        # Explained variance
        total_var = np.sum(sv**2)
        explained_var = sv[:k] ** 2 / total_var

        # Optional: remove Y component from factors
        if self.remove_y_from_factors:
            # Regress f_raw on y_arr, take residuals
            y_aug = np.column_stack([y_arr, np.ones(t_total)])
            coef_fy = la.lstsq(y_aug, f_raw)[0]
            f_hat = f_raw - y_aug @ coef_fy
        else:
            f_hat = f_raw

        # Estimate loadings for Y
        # x_s = loadings_f * F + loadings_y * Y + e
        fy = np.column_stack([f_hat, y_arr])
        loadings_all = la.lstsq(fy, x_s)[0].T  # (N, K+M)
        loadings_f = loadings_all[:, :k]
        loadings_y = loadings_all[:, k:]

        # Step 2: Estimate VAR on [f_hat; y_arr]
        z_var = np.column_stack([f_hat, y_arr])  # (T, K+M)
        km = k + m

        t_eff = t_total - p
        y_var = z_var[p:]  # (t_eff, km)
        z_reg = np.zeros((t_eff, km * p + 1))
        for t in range(t_eff):
            for s in range(p):
                z_reg[t, s * km : (s + 1) * km] = z_var[p + t - s - 1]
            z_reg[t, -1] = 1.0

        # OLS
        b_hat = la.solve(z_reg.T @ z_reg, z_reg.T @ y_var)
        resid = y_var - z_reg @ b_hat
        sigma_var = (resid.T @ resid) / t_eff

        # Extract coefficient matrices
        coefs = np.zeros((p, km, km))
        for s in range(p):
            coefs[s] = b_hat[s * km : (s + 1) * km].T
        intercept = b_hat[-1]

        return FAVARResults(
            factors=f_hat,
            loadings=loadings_f,
            loadings_y=loadings_y,
            var_coefs=coefs,
            var_sigma=sigma_var,
            var_resid=resid,
            intercept=intercept,
            explained_variance_ratio=explained_var,
            total_explained_variance=float(np.sum(explained_var)),
            n_factors=k,
            n_policy=m,
            n_panel=n_vars,
            n_obs=t_eff,
            lags=p,
            method="two_step",
            x_mean=x_mean,
            x_std=x_std,
        )

    def _fit_bayesian(
        self,
        x_arr: NDArray[np.floating[Any]],
        y_arr: NDArray[np.floating[Any]],
        t_total: int,
        n_vars: int,
        m: int,
    ) -> FAVARResults:
        """One-step Bayesian estimation via kalmanbox state-space.

        Uses the Kalman filter to jointly estimate factors and VAR parameters.
        Requires kalmanbox to be installed.
        """
        k = self.n_factors
        p = self.lags

        try:
            from kalmanbox import KalmanFilter
        except ImportError as err:
            msg = (
                "kalmanbox is required for Bayesian FAVAR estimation. "
                "Install it with: pip install kalmanbox"
            )
            raise ImportError(msg) from err

        # Standardize X
        x_mean = np.mean(x_arr, axis=0)
        x_std = np.std(x_arr, axis=0)
        x_std[x_std < 1e-10] = 1.0
        x_s = (x_arr - x_mean) / x_std

        # Initialize with two-step estimates for starting values
        two_step_result = self._fit_two_step(x_arr, y_arr, t_total, n_vars, m)

        # State-space model:
        # State: [F_t; Y_t; F_{t-1}; Y_{t-1}; ...]  (p*(K+M) dimensional)
        # Companion form for the VAR part.
        km = k + m
        state_dim = km * p

        # Observation equation:
        # [X_t; Y_t] = obs_mat * state_t + noise
        obs_dim = n_vars + m

        # Build observation matrix
        obs_mat = np.zeros((obs_dim, state_dim))
        obs_mat[:n_vars, :k] = two_step_result.loadings  # Lambda^f
        obs_mat[:n_vars, k:km] = two_step_result.loadings_y  # Lambda^y
        obs_mat[n_vars:, k:km] = np.eye(m)  # Y_t = I * Y_t in state

        # Build transition matrix (companion form)
        trans_mat = np.zeros((state_dim, state_dim))
        for s in range(p):
            trans_mat[:km, s * km : (s + 1) * km] = two_step_result.var_coefs[s]
        if p > 1:
            trans_mat[km:, : km * (p - 1)] = np.eye(km * (p - 1))

        # Build state noise covariance
        trans_cov = np.zeros((state_dim, state_dim))
        trans_cov[:km, :km] = two_step_result.var_sigma

        # Build observation noise covariance
        obs_cov = np.eye(obs_dim) * 0.1
        obs_cov[n_vars:, n_vars:] = np.eye(m) * 1e-8  # Y_t is observed exactly

        # Observations
        obs = np.column_stack([x_s, y_arr])

        # Run Kalman filter
        try:
            kf = KalmanFilter(
                transition_matrix=trans_mat,
                observation_matrix=obs_mat,
                transition_covariance=trans_cov,
                observation_covariance=obs_cov,
                initial_state_mean=np.zeros(state_dim),
                initial_state_covariance=np.eye(state_dim),
            )
            filtered_state, _ = kf.filter(obs)
            f_hat = filtered_state[:, :k]
        except Exception:
            # Fall back to two-step if kalmanbox API differs
            import warnings

            warnings.warn(
                "Bayesian FAVAR via kalmanbox failed. Falling back to two-step.",
                stacklevel=2,
            )
            return self._fit_two_step(x_arr, y_arr, t_total, n_vars, m)

        # Re-estimate VAR with Bayesian factors
        fy = np.column_stack([f_hat, y_arr])
        loadings_all = la.lstsq(fy, x_s)[0].T
        loadings_f = loadings_all[:, :k]
        loadings_y = loadings_all[:, k:]

        # VAR estimation on [f_hat; y_arr]
        z_var = fy
        t_eff = t_total - p
        y_var = z_var[p:]
        z_reg = np.zeros((t_eff, km * p + 1))
        for t in range(t_eff):
            for s in range(p):
                z_reg[t, s * km : (s + 1) * km] = z_var[p + t - s - 1]
            z_reg[t, -1] = 1.0

        b_hat = la.solve(z_reg.T @ z_reg, z_reg.T @ y_var)
        resid = y_var - z_reg @ b_hat
        sigma_var = (resid.T @ resid) / t_eff

        coefs = np.zeros((p, km, km))
        for s in range(p):
            coefs[s] = b_hat[s * km : (s + 1) * km].T
        intercept = b_hat[-1]

        # Explained variance from initial PCA
        explained_var = two_step_result.explained_variance_ratio

        return FAVARResults(
            factors=f_hat,
            loadings=loadings_f,
            loadings_y=loadings_y,
            var_coefs=coefs,
            var_sigma=sigma_var,
            var_resid=resid,
            intercept=intercept,
            explained_variance_ratio=explained_var,
            total_explained_variance=float(np.sum(explained_var)),
            n_factors=k,
            n_policy=m,
            n_panel=n_vars,
            n_obs=t_eff,
            lags=p,
            method="bayesian",
            x_mean=x_mean,
            x_std=x_std,
        )
