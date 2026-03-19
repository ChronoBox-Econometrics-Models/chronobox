"""
Time-Varying Parameter VAR (TVP-VAR).

Estimation via Kalman filter using kalmanbox.

The model allows VAR coefficients to evolve over time as a random walk:
  Y_t = X_t * beta_t + eps_t,  eps_t ~ N(0, H)
  beta_t = beta_{t-1} + u_t,   u_t ~ N(0, Q)

References
----------
- Primiceri, G.E. (2005). RES, 72(3), 821-852.
- Cogley, T. & Sargent, T.J. (2005). RES, 72(2), 367-401.
- Koop, G. & Korobilis, D. (2013). JBES, 31(3), 265-279.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la


@dataclass
class TVPVARResults:
    """Results from TVP-VAR estimation.

    Attributes
    ----------
    beta_filtered : ndarray of shape (T, d)
        Filtered state (time-varying coefficients) at each time t.
        d = K * (K*p + 1).
    beta_smoothed : ndarray of shape (T, d) or None
        Smoothed state (if available from kalmanbox).
    coefs_time : ndarray of shape (T, p, K, K)
        Time-varying coefficient matrices B_1t, ..., B_pt for each t.
    intercept_time : ndarray of shape (T, K)
        Time-varying intercept for each t.
    sigma : ndarray of shape (K, K)
        Observation noise covariance (constant).
    Q : ndarray of shape (d, d)
        State evolution covariance.
    n_obs : int
        Number of observations used.
    k_vars : int
        Number of endogenous variables.
    lags : int
        Number of lags.
    endog : ndarray of shape (T_total, K)
        Original data.
    log_likelihood : float
        Log-likelihood from the Kalman filter.
    """

    beta_filtered: NDArray[np.floating[Any]]
    beta_smoothed: NDArray[np.floating[Any]] | None
    coefs_time: NDArray[np.floating[Any]]
    intercept_time: NDArray[np.floating[Any]]
    sigma: NDArray[np.floating[Any]]
    Q: NDArray[np.floating[Any]]
    n_obs: int
    k_vars: int
    lags: int
    endog: NDArray[np.floating[Any]]
    log_likelihood: float

    def time_varying_irf(
        self,
        t: int,
        periods: int = 20,
        identification: str = "cholesky",
    ) -> NDArray[np.floating[Any]]:
        """Compute structural IRF at a specific time point t.

        Parameters
        ----------
        t : int
            Time index (0-based, relative to the VAR sample).
        periods : int
            Number of IRF periods.
        identification : str
            Identification method ('cholesky').

        Returns
        -------
        ndarray of shape (periods+1, K, K)
        """
        K = self.k_vars
        p = self.lags
        H = periods + 1

        if t < 0 or t >= self.n_obs:
            msg = f"t={t} out of range [0, {self.n_obs - 1}]"
            raise ValueError(msg)

        # Extract coefficients at time t
        coefs_t = self.coefs_time[t]  # (p, K, K)

        # Impact matrix
        if identification == "cholesky":
            sigma = (self.sigma + self.sigma.T) / 2.0
            P = la.cholesky(sigma, lower=True)
        else:
            P = np.eye(K)

        # MA coefficients
        Phi = np.zeros((H, K, K))
        Phi[0] = np.eye(K)
        for h in range(1, H):
            for s in range(1, min(h, p) + 1):
                Phi[h] += Phi[h - s] @ coefs_t[s - 1]

        # Structural IRF
        irf = np.zeros((H, K, K))
        for h in range(H):
            irf[h] = Phi[h] @ P

        return irf

    def coefficient_path(
        self, lag: int, i: int, j: int
    ) -> NDArray[np.floating[Any]]:
        """Extract the time path of a specific VAR coefficient.

        Parameters
        ----------
        lag : int
            Lag index (0-based, i.e., 0 for first lag).
        i : int
            Equation index.
        j : int
            Variable index.

        Returns
        -------
        ndarray of shape (T,)
        """
        return self.coefs_time[:, lag, i, j]


class TVPVAR:
    """Time-Varying Parameter VAR.

    Estimated via Kalman filter using kalmanbox.

    Parameters
    ----------
    lags : int
        Number of lags (default=1).
    stochastic_volatility : bool
        If True, estimate time-varying volatility (not yet implemented,
        reserved for future extension). Default=False.
    Q_scale : float
        Scale factor for the state evolution covariance Q.
        Q = Q_scale^2 * I. Smaller values mean coefficients change slowly.
        Default=0.01.
    """

    def __init__(
        self,
        lags: int = 1,
        stochastic_volatility: bool = False,
        Q_scale: float = 0.01,
    ) -> None:
        self.lags = lags
        self.stochastic_volatility = stochastic_volatility
        self.Q_scale = Q_scale

    def fit(
        self,
        endog: NDArray[np.floating[Any]],
        Q: NDArray[np.floating[Any]] | None = None,
    ) -> TVPVARResults:
        """Fit TVP-VAR model.

        Parameters
        ----------
        endog : ndarray of shape (T, K)
            Endogenous variables.
        Q : ndarray or None
            State evolution covariance. If None, uses Q_scale^2 * I.

        Returns
        -------
        TVPVARResults
        """
        endog = np.asarray(endog, dtype=float)
        if endog.ndim == 1:
            endog = endog.reshape(-1, 1)

        T_total, K = endog.shape
        p = self.lags
        T = T_total - p

        if T <= 0:
            msg = f"Not enough observations: T_total={T_total}, lags={p}"
            raise ValueError(msg)

        # State dimension: beta has K*(K*p + 1) elements
        # Each equation has K*p + 1 regressors, and there are K equations
        n_regressors = K * p + 1  # per equation (lags + intercept)
        d = K * n_regressors  # total state dimension

        # Construct observation data and design matrices
        Y = np.zeros((T, K))
        X_all = np.zeros((T, K, d))

        for t in range(T):
            Y[t] = endog[t + p]

            # Build regressor vector: [Y_{t+p-1}', ..., Y_{t}', 1]
            regressors = np.zeros(n_regressors)
            for s in range(p):
                regressors[s * K : (s + 1) * K] = endog[t + p - s - 1]
            regressors[-1] = 1.0  # intercept

            # X_t = I_K kron regressors' -> each row of equation i uses regressors
            for eq in range(K):
                X_all[t, eq, eq * n_regressors : (eq + 1) * n_regressors] = (
                    regressors
                )

        # Initial OLS estimate for beta_0
        # Stack all observations for a constant-coefficient VAR
        Z_ols = np.zeros((T, n_regressors))
        for t in range(T):
            for s in range(p):
                Z_ols[t, s * K : (s + 1) * K] = endog[t + p - s - 1]
            Z_ols[t, -1] = 1.0

        try:
            B_ols = la.solve(Z_ols.T @ Z_ols, Z_ols.T @ Y)
        except la.LinAlgError:
            B_ols = la.lstsq(Z_ols, Y)[0]

        U_ols = Y - Z_ols @ B_ols
        Sigma_ols = (U_ols.T @ U_ols) / T
        Sigma_ols = (Sigma_ols + Sigma_ols.T) / 2.0

        # Initial state: vec(B_ols)
        beta_0 = np.zeros(d)
        for eq in range(K):
            beta_0[eq * n_regressors : (eq + 1) * n_regressors] = B_ols[:, eq]

        # State evolution covariance Q
        Q_mat = np.asarray(Q, dtype=float) if Q is not None else np.eye(d) * self.Q_scale**2

        # Initial state covariance
        P_0 = np.eye(d) * 4.0  # diffuse

        # Observation noise covariance (constant)
        H = Sigma_ols.copy()

        # Run Kalman filter manually (to avoid kalmanbox API dependency issues)
        beta_filtered, beta_smoothed, log_lik = self._kalman_filter_smoother(
            Y, X_all, beta_0, P_0, Q_mat, H, T, K, d
        )

        # Extract time-varying coefficients
        coefs_time = np.zeros((T, p, K, K))
        intercept_time = np.zeros((T, K))

        for t in range(T):
            for eq in range(K):
                b_eq = beta_filtered[
                    t, eq * n_regressors : (eq + 1) * n_regressors
                ]
                for s in range(p):
                    coefs_time[t, s, eq, :] = b_eq[s * K : (s + 1) * K]
                intercept_time[t, eq] = b_eq[-1]

        return TVPVARResults(
            beta_filtered=beta_filtered,
            beta_smoothed=beta_smoothed,
            coefs_time=coefs_time,
            intercept_time=intercept_time,
            sigma=Sigma_ols,
            Q=Q_mat,
            n_obs=T,
            k_vars=K,
            lags=p,
            endog=endog,
            log_likelihood=log_lik,
        )

    def _kalman_filter_smoother(
        self,
        Y: NDArray[np.floating[Any]],
        X_all: NDArray[np.floating[Any]],
        beta_0: NDArray[np.floating[Any]],
        P_0: NDArray[np.floating[Any]],
        Q: NDArray[np.floating[Any]],
        H: NDArray[np.floating[Any]],
        T: int,
        K: int,
        d: int,
    ) -> tuple[
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        float,
    ]:
        """Run Kalman filter and smoother.

        State equation: beta_t = beta_{t-1} + u_t, u_t ~ N(0, Q)
        Obs equation:   Y_t = X_t * beta_t + eps_t, eps_t ~ N(0, H)

        Parameters
        ----------
        Y : (T, K)
        X_all : (T, K, d) time-varying observation matrices
        beta_0 : (d,) initial state
        P_0 : (d, d) initial state covariance
        Q : (d, d) state noise covariance
        H : (K, K) observation noise covariance
        T : int
        K : int
        d : int

        Returns
        -------
        beta_filtered : (T, d)
        beta_smoothed : (T, d)
        log_likelihood : float
        """
        # Storage
        beta_pred = np.zeros((T, d))
        beta_filt = np.zeros((T, d))
        P_pred = np.zeros((T, d, d))
        P_filt = np.zeros((T, d, d))

        log_lik = 0.0
        beta_prev = beta_0.copy()
        P_prev = P_0.copy()

        for t in range(T):
            # --- Prediction step ---
            beta_pred[t] = beta_prev  # T = I
            P_pred[t] = P_prev + Q

            # --- Update step ---
            Z_t = X_all[t]  # (K, d)
            y_t = Y[t]  # (K,)

            # Innovation
            v_t = y_t - Z_t @ beta_pred[t]  # (K,)

            # Innovation covariance
            F_t = Z_t @ P_pred[t] @ Z_t.T + H  # (K, K)
            F_t = (F_t + F_t.T) / 2.0

            # Kalman gain
            try:
                F_inv = la.inv(F_t)
            except la.LinAlgError:
                F_inv = la.inv(F_t + np.eye(K) * 1e-8)

            K_gain = P_pred[t] @ Z_t.T @ F_inv  # (d, K)

            # Update
            beta_filt[t] = beta_pred[t] + K_gain @ v_t
            P_filt[t] = (np.eye(d) - K_gain @ Z_t) @ P_pred[t]
            P_filt[t] = (P_filt[t] + P_filt[t].T) / 2.0

            # Log-likelihood contribution
            sign, logdet = np.linalg.slogdet(F_t)
            if sign > 0:
                log_lik += -0.5 * (
                    K * np.log(2 * np.pi) + logdet + v_t @ F_inv @ v_t
                )

            beta_prev = beta_filt[t]
            P_prev = P_filt[t]

        # --- Rauch-Tung-Striebel smoother ---
        beta_smooth = np.zeros((T, d))
        beta_smooth[-1] = beta_filt[-1]

        for t in range(T - 2, -1, -1):
            try:
                P_pred_inv = la.inv(P_pred[t + 1])
            except la.LinAlgError:
                P_pred_inv = la.inv(P_pred[t + 1] + np.eye(d) * 1e-8)

            J_t = P_filt[t] @ P_pred_inv  # T = I, so this simplifies
            beta_smooth[t] = beta_filt[t] + J_t @ (
                beta_smooth[t + 1] - beta_pred[t + 1]
            )

        return beta_filt, beta_smooth, log_lik
