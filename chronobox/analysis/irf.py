"""IRF - Impulse Response Functions for VAR models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from chronobox.models.var import VARResults


class IRF:
    """Impulse Response Functions for VAR models.

    Supports orthogonalized (Cholesky) and generalized (Pesaran-Shin) IRFs,
    with optional bootstrap confidence bands.

    Parameters
    ----------
    var_results : VARResults
        Fitted VAR model results.
    periods : int, default 20
        Number of periods for the IRF.
    method : str, default 'cholesky'
        Identification method:
        - 'cholesky': orthogonalized via Cholesky decomposition
        - 'generalized': Pesaran-Shin (1998) generalized IRF
    sigs : float, default 0.95
        Confidence level for bootstrap bands (e.g., 0.95 for 95%).
    runs : int, default 1000
        Number of bootstrap replications for confidence bands.
        Set to 0 to skip bootstrap.
    seed : int or None, default None
        Random seed for bootstrap reproducibility.

    Attributes
    ----------
    irfs : ndarray of shape (periods+1, K, K)
        Point estimates of IRF. irfs[h, i, j] = response of variable i
        at horizon h to a shock in variable j.
    lower : ndarray of shape (periods+1, K, K) or None
        Lower bound of confidence band (None if runs=0).
    upper : ndarray of shape (periods+1, K, K) or None
        Upper bound of confidence band (None if runs=0).
    cum_irfs : ndarray of shape (periods+1, K, K)
        Cumulative IRFs.
    """

    def __init__(
        self,
        var_results: VARResults,
        periods: int = 20,
        method: str = "cholesky",
        sigs: float = 0.95,
        runs: int = 1000,
        seed: int | None = None,
    ) -> None:
        if method not in ("cholesky", "generalized"):
            msg = f"method must be 'cholesky' or 'generalized', got '{method}'"
            raise ValueError(msg)
        if periods < 1:
            msg = f"periods must be >= 1, got {periods}"
            raise ValueError(msg)

        self._var_results = var_results
        self._periods = periods
        self._method = method
        self._sigs = sigs
        self._runs = runs
        self._seed = seed

        self._k = var_results.neqs
        self._names = var_results.names

        # Compute point estimate
        self._irfs = self._compute_irf(var_results)
        self._cum_irfs = np.cumsum(self._irfs, axis=0)

        # Bootstrap confidence bands
        if runs > 0:
            self._lower, self._upper = self._bootstrap_bands()
        else:
            self._lower = None
            self._upper = None

    @property
    def irfs(self) -> NDArray[np.float64]:
        """IRF point estimates of shape (periods+1, K, K)."""
        return self._irfs

    @property
    def lower(self) -> NDArray[np.float64] | None:
        """Lower confidence band of shape (periods+1, K, K), or None."""
        return self._lower

    @property
    def upper(self) -> NDArray[np.float64] | None:
        """Upper confidence band of shape (periods+1, K, K), or None."""
        return self._upper

    @property
    def cum_irfs(self) -> NDArray[np.float64]:
        """Cumulative IRFs of shape (periods+1, K, K)."""
        return self._cum_irfs

    def _compute_ma_coefs(
        self, var_results: VARResults
    ) -> NDArray[np.float64]:
        """Compute MA coefficient matrices Phi_0, ..., Phi_periods.

        Parameters
        ----------
        var_results : VARResults
            VAR results to compute MA coefficients from.

        Returns
        -------
        ndarray of shape (periods+1, K, K)
        """
        return var_results._ma_coefs(self._periods)

    def _compute_irf(
        self, var_results: VARResults
    ) -> NDArray[np.float64]:
        """Compute IRF point estimates.

        Parameters
        ----------
        var_results : VARResults
            VAR results.

        Returns
        -------
        ndarray of shape (periods+1, K, K)
        """
        k = var_results.neqs
        phi = self._compute_ma_coefs(var_results)  # (periods+1, K, K)

        if self._method == "cholesky":
            # Orthogonalized IRF: Theta_h = Phi_h @ P
            p_chol = np.linalg.cholesky(var_results.sigma_u)  # lower triangular
            irf_arr = np.zeros_like(phi)
            for h in range(self._periods + 1):
                irf_arr[h] = phi[h] @ p_chol
        else:
            # Generalized IRF (Pesaran-Shin 1998)
            # GIRF_j(h) = Phi_h @ Sigma_u @ e_j / sqrt(sigma_jj)
            sigma_u = var_results.sigma_u
            irf_arr = np.zeros_like(phi)
            for h in range(self._periods + 1):
                for j in range(k):
                    sigma_jj = sigma_u[j, j]
                    irf_arr[h, :, j] = (phi[h] @ sigma_u[:, j]) / np.sqrt(
                        sigma_jj
                    )

        return irf_arr

    def _bootstrap_bands(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute bootstrap confidence bands via residual resampling.

        Returns
        -------
        tuple of (lower, upper), each of shape (periods+1, K, K)
        """
        from chronobox.models.var import VAR

        vr = self._var_results
        k = vr.neqs
        p = vr.k_ar
        t_total = vr.nobs_total
        resid = vr.resid  # (T', K)
        t_eff = resid.shape[0]

        rng = np.random.default_rng(self._seed)

        alpha_half = (1.0 - self._sigs) / 2.0
        irf_boot = np.zeros(
            (self._runs, self._periods + 1, k, k), dtype=np.float64
        )

        for b in range(self._runs):
            # Resample residuals with replacement
            boot_idx = rng.integers(0, t_eff, size=t_eff)
            u_star = resid[boot_idx]  # (T', K)

            # Generate synthetic data
            y_star = np.zeros((t_total, k), dtype=np.float64)

            # Condition on first p observations from original data
            y_star[:p] = vr.endog[:p]

            # Reconstruct using estimated coefficients
            for t in range(p, t_total):
                y_t = np.zeros(k, dtype=np.float64)
                for lag_i in range(p):
                    y_t += vr.coefs[lag_i] @ y_star[t - lag_i - 1]
                if vr.trend in ("c", "ct", "ctt"):
                    y_t += vr.intercept
                y_t += u_star[t - p]
                y_star[t] = y_t

            # Re-estimate VAR on synthetic data
            try:
                model_star = VAR(lags=p, trend=vr.trend)
                results_star = model_star.fit(y_star)
                irf_star = self._compute_irf(results_star)
                irf_boot[b] = irf_star
            except (np.linalg.LinAlgError, ValueError):
                # If re-estimation fails, use point estimate
                irf_boot[b] = self._irfs

        # Compute percentile bands
        lower = np.percentile(irf_boot, alpha_half * 100, axis=0)
        upper = np.percentile(irf_boot, (1 - alpha_half) * 100, axis=0)

        return lower, upper

    def plot(
        self,
        impulse: str | int | None = None,
        response: str | int | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Plot IRF.

        Parameters
        ----------
        impulse : str, int, or None
            Impulse variable. If None, plots all impulse-response pairs.
        response : str, int, or None
            Response variable. If None, plots all responses.
        figsize : tuple or None
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        names = self._names

        # Resolve indices
        impulse_indices = self._resolve_indices(impulse)
        response_indices = self._resolve_indices(response)

        n_imp = len(impulse_indices)
        n_resp = len(response_indices)
        horizons = np.arange(self._periods + 1)

        if figsize is None:
            figsize = (4 * n_imp, 3 * n_resp)

        fig, axes = plt.subplots(n_resp, n_imp, figsize=figsize, squeeze=False)

        for col, j in enumerate(impulse_indices):
            for row, i in enumerate(response_indices):
                ax = axes[row, col]
                ax.plot(horizons, self._irfs[:, i, j], "b-", linewidth=1.5)
                ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)

                if self._lower is not None and self._upper is not None:
                    ax.fill_between(
                        horizons,
                        self._lower[:, i, j],
                        self._upper[:, i, j],
                        alpha=0.2,
                        color="blue",
                    )

                if row == 0:
                    ax.set_title(f"Shock: {names[j]}", fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"{names[i]}", fontsize=10)
                if row == n_resp - 1:
                    ax.set_xlabel("Horizon", fontsize=9)

                ax.grid(True, alpha=0.3)

        method_label = (
            "Orthogonalized" if self._method == "cholesky" else "Generalized"
        )
        fig.suptitle(
            f"Impulse Response Functions ({method_label})", fontsize=12
        )
        plt.tight_layout()
        return fig

    def plot_cum(
        self,
        impulse: str | int | None = None,
        response: str | int | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Plot cumulative IRF.

        Parameters
        ----------
        impulse : str, int, or None
            Impulse variable.
        response : str, int, or None
            Response variable.
        figsize : tuple or None
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        names = self._names
        impulse_indices = self._resolve_indices(impulse)
        response_indices = self._resolve_indices(response)
        n_imp = len(impulse_indices)
        n_resp = len(response_indices)
        horizons = np.arange(self._periods + 1)

        if figsize is None:
            figsize = (4 * n_imp, 3 * n_resp)

        fig, axes = plt.subplots(n_resp, n_imp, figsize=figsize, squeeze=False)

        for col, j in enumerate(impulse_indices):
            for row, i in enumerate(response_indices):
                ax = axes[row, col]
                ax.plot(
                    horizons, self._cum_irfs[:, i, j], "r-", linewidth=1.5
                )
                ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)

                if row == 0:
                    ax.set_title(f"Shock: {names[j]}", fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"{names[i]}", fontsize=10)
                if row == n_resp - 1:
                    ax.set_xlabel("Horizon", fontsize=9)

                ax.grid(True, alpha=0.3)

        fig.suptitle("Cumulative Impulse Response Functions", fontsize=12)
        plt.tight_layout()
        return fig

    def to_dataframe(self) -> pd.DataFrame:
        """Convert IRF to a long-form DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: horizon, impulse, response, irf, lower, upper.
        """
        records: list[dict[str, Any]] = []
        names = self._names

        for h in range(self._periods + 1):
            for j in range(self._k):
                for i in range(self._k):
                    rec: dict[str, Any] = {
                        "horizon": h,
                        "impulse": names[j],
                        "response": names[i],
                        "irf": self._irfs[h, i, j],
                    }
                    if self._lower is not None:
                        rec["lower"] = self._lower[h, i, j]
                    if self._upper is not None:
                        rec["upper"] = self._upper[h, i, j]
                    records.append(rec)

        return pd.DataFrame(records)

    def _resolve_indices(self, var: str | int | None) -> list[int]:
        """Resolve variable name/index to list of indices."""
        if var is None:
            return list(range(self._k))
        if isinstance(var, int):
            return [var]
        if isinstance(var, str):
            if var in self._names:
                return [self._names.index(var)]
            msg = f"Variable '{var}' not found. Available: {self._names}"
            raise ValueError(msg)
        return list(range(self._k))
