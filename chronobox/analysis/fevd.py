"""FEVD - Forecast Error Variance Decomposition for VAR models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from chronobox.models.var import VARResults


class FEVD:
    """Forecast Error Variance Decomposition for VAR models.

    Decomposes the forecast error variance of each variable into
    contributions from orthogonalized shocks (via Cholesky decomposition).

    Parameters
    ----------
    var_results : VARResults
        Fitted VAR model results.
    periods : int, default 20
        Number of forecast horizons.
    method : str, default 'cholesky'
        Orthogonalization method.

    Attributes
    ----------
    decomp : ndarray of shape (periods+1, K, K)
        decomp[h, i, k] = fraction of forecast error variance of variable i
        at horizon h attributable to shock k.
    """

    def __init__(
        self,
        var_results: VARResults,
        periods: int = 20,
        method: str = "cholesky",
    ) -> None:
        self._var_results = var_results
        self._periods = periods
        self._method = method
        self._k = var_results.neqs
        self._names = var_results.names

        self._decomp = self._compute_fevd()

    @property
    def decomp(self) -> NDArray[np.float64]:
        """FEVD array of shape (periods+1, K, K)."""
        return self._decomp

    def _compute_fevd(self) -> NDArray[np.float64]:
        """Compute the FEVD.

        Returns
        -------
        ndarray of shape (periods+1, K, K)
        """
        k = self._k
        periods = self._periods

        # Get MA coefficients
        phi = self._var_results._ma_coefs(periods)  # (periods+1, K, K)

        # Cholesky factor
        p_chol = np.linalg.cholesky(self._var_results.sigma_u)

        # Orthogonalized MA coefficients: Theta_h = Phi_h @ P
        theta = np.zeros((periods + 1, k, k), dtype=np.float64)
        for h in range(periods + 1):
            theta[h] = phi[h] @ p_chol

        # Compute FEVD
        decomp = np.zeros((periods + 1, k, k), dtype=np.float64)

        for h in range(periods + 1):
            # For each variable i
            for i in range(k):
                # Total MSE for variable i up to horizon h
                mse_i = 0.0
                contrib = np.zeros(k, dtype=np.float64)

                for s in range(h + 1):
                    for kk in range(k):
                        val = theta[s, i, kk] ** 2
                        contrib[kk] += val
                        mse_i += val

                # Normalize
                if mse_i > 0:
                    decomp[h, i, :] = contrib / mse_i
                else:
                    # At h=0 before any forecast error, use identity
                    decomp[h, i, i] = 1.0

        return decomp

    def summary(self) -> str:
        """Formatted summary of the FEVD.

        Returns
        -------
        str
            Summary table showing decomposition at selected horizons.
        """
        lines: list[str] = []
        lines.append("=" * 78)
        lines.append("  Forecast Error Variance Decomposition")
        lines.append("=" * 78)

        # Show selected horizons
        horizons = [0, 1, 2, 5, 10, 20]
        horizons = [h for h in horizons if h <= self._periods]

        names = self._names
        k = self._k

        for i in range(k):
            lines.append("")
            lines.append(f"  Response variable: {names[i]}")
            lines.append("-" * 78)
            header = f"  {'Horizon':>8s}" + "".join(f"{n:>12s}" for n in names)
            lines.append(header)
            lines.append("  " + "-" * (8 + 12 * k))

            for h in horizons:
                row = f"  {h:8d}"
                for j in range(k):
                    row += f"{self._decomp[h, i, j]:12.4f}"
                lines.append(row)

        lines.append("")
        lines.append("=" * 78)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert FEVD to long-form DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: horizon, response, shock, fevd.
        """
        records: list[dict[str, Any]] = []
        names = self._names

        for h in range(self._periods + 1):
            for i in range(self._k):
                for j in range(self._k):
                    records.append({
                        "horizon": h,
                        "response": names[i],
                        "shock": names[j],
                        "fevd": self._decomp[h, i, j],
                    })

        return pd.DataFrame(records)

    def plot(
        self,
        variable: str | int | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Plot FEVD as stacked area chart.

        Parameters
        ----------
        variable : str, int, or None
            Variable to plot. If None, plots all variables.
        figsize : tuple or None
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        k = self._k
        names = self._names

        if variable is None:
            var_indices = list(range(k))
        elif isinstance(variable, int):
            var_indices = [variable]
        elif isinstance(variable, str):
            if variable not in names:
                msg = f"Variable '{variable}' not found. Available: {names}"
                raise ValueError(msg)
            var_indices = [names.index(variable)]
        else:
            var_indices = list(range(k))

        n_vars = len(var_indices)
        if figsize is None:
            figsize = (10, 3 * n_vars)

        fig, axes = plt.subplots(n_vars, 1, figsize=figsize, squeeze=False)

        horizons = np.arange(self._periods + 1)
        colors = plt.cm.tab10(np.linspace(0, 1, k))  # type: ignore[attr-defined]

        for idx, var_i in enumerate(var_indices):
            ax = axes[idx, 0]
            bottom = np.zeros(self._periods + 1, dtype=np.float64)

            for shock_j in range(k):
                values = self._decomp[:, var_i, shock_j]
                ax.fill_between(
                    horizons,
                    bottom,
                    bottom + values,
                    alpha=0.7,
                    color=colors[shock_j],
                    label=names[shock_j],
                )
                bottom += values

            ax.set_title(f"FEVD: {names[var_i]}", fontsize=11)
            ax.set_ylabel("Fraction", fontsize=10)
            ax.set_xlabel("Horizon", fontsize=10)
            ax.set_ylim(0, 1)
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Forecast Error Variance Decomposition", fontsize=13)
        plt.tight_layout()
        return fig
