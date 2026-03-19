"""
Historical Decomposition of structural shocks.

Decomposes observed time series movements into contributions from
each structural shock identified by an SVAR model.

References
----------
- Kilian, L. & Lutkepohl, H. (2017). Structural Vector Autoregressive
  Analysis. Cambridge University Press. Chapter 4.
- Burbidge, J. & Harrison, A. (1985). An historical decomposition of the
  great depression to determine the role of money. Journal of Monetary
  Economics, 16(1), 45-54.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class HistoricalDecompositionResult:
    """Results from historical decomposition.

    Attributes
    ----------
    decomposition : ndarray of shape (T, K_shocks, K_vars)
        HD[t, k, i] = contribution of structural shock k to variable i at time t.
    base : ndarray of shape (T, K_vars)
        Base (deterministic) forecast component.
    observed : ndarray of shape (T, K_vars)
        Observed values of the endogenous variables (within VAR sample).
    structural_shocks : ndarray of shape (T, K)
        Structural shocks eps_t.
    n_obs : int
        Number of observations.
    k_vars : int
        Number of variables.
    shock_names : list[str] | None
        Optional names for structural shocks.
    variable_names : list[str] | None
        Optional names for variables.
    """

    decomposition: NDArray[np.floating[Any]]
    base: NDArray[np.floating[Any]]
    observed: NDArray[np.floating[Any]]
    structural_shocks: NDArray[np.floating[Any]]
    n_obs: int
    k_vars: int
    shock_names: list[str] | None = None
    variable_names: list[str] | None = None

    def verify_decomposition(self, atol: float = 1e-8) -> bool:
        """Verify that base + sum(HD) = observed.

        Parameters
        ----------
        atol : float
            Absolute tolerance for the check.

        Returns
        -------
        bool
            True if the decomposition sums to observed values.
        """
        reconstructed = self.base + self.decomposition.sum(axis=1)
        return bool(np.allclose(reconstructed, self.observed, atol=atol))

    def contribution_of_shock(
        self, shock_index: int
    ) -> NDArray[np.floating[Any]]:
        """Extract the contribution of a specific shock.

        Parameters
        ----------
        shock_index : int
            Index of the structural shock.

        Returns
        -------
        ndarray of shape (T, K_vars)
        """
        return self.decomposition[:, shock_index, :]

    def to_dataframe(self, variable: int) -> Any:
        """Convert decomposition for a specific variable to a pandas DataFrame.

        Parameters
        ----------
        variable : int
            Variable index.

        Returns
        -------
        pd.DataFrame
            Columns are shock contributions + base + observed.
        """
        import pandas as pd

        n_shocks = self.k_vars
        data: dict[str, NDArray[np.floating[Any]]] = {}

        for k in range(n_shocks):
            name = self.shock_names[k] if self.shock_names else f"Shock_{k}"
            data[name] = self.decomposition[:, k, variable]

        data["Base"] = self.base[:, variable]
        data["Observed"] = self.observed[:, variable]

        return pd.DataFrame(data)

    def plot(
        self,
        variable: int,
        stacked: bool = True,
        figsize: tuple[int, int] = (12, 6),
        title: str | None = None,
    ) -> Any:
        """Plot historical decomposition for a specific variable.

        Parameters
        ----------
        variable : int
            Variable index to plot.
        stacked : bool
            If True, use stacked bar chart.
        figsize : tuple
            Figure size.
        title : str or None
            Plot title.

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as plt

        n_shocks = self.k_vars
        n_t = self.n_obs
        t_axis = np.arange(n_t)

        fig, ax = plt.subplots(figsize=figsize)

        if stacked:
            pos_data = np.zeros((n_shocks, n_t))
            neg_data = np.zeros((n_shocks, n_t))

            for k in range(n_shocks):
                contrib = self.decomposition[:, k, variable]
                pos_data[k] = np.maximum(contrib, 0)
                neg_data[k] = np.minimum(contrib, 0)

            bottom_pos = np.zeros(n_t) + self.base[:, variable]
            bottom_neg = np.zeros(n_t) + self.base[:, variable]

            colors = plt.cm.Set3(np.linspace(0, 1, n_shocks))  # type: ignore[attr-defined]

            for k in range(n_shocks):
                label = (
                    self.shock_names[k] if self.shock_names else f"Shock {k}"
                )
                ax.bar(
                    t_axis,
                    pos_data[k],
                    bottom=bottom_pos,
                    width=1.0,
                    color=colors[k],
                    label=label,
                    alpha=0.8,
                )
                ax.bar(
                    t_axis,
                    neg_data[k],
                    bottom=bottom_neg,
                    width=1.0,
                    color=colors[k],
                    alpha=0.8,
                )
                bottom_pos += pos_data[k]
                bottom_neg += neg_data[k]

            ax.plot(
                t_axis,
                self.observed[:, variable],
                "k-",
                linewidth=1.5,
                label="Observed",
                zorder=5,
            )
        else:
            for k in range(n_shocks):
                label = (
                    self.shock_names[k] if self.shock_names else f"Shock {k}"
                )
                ax.plot(
                    t_axis, self.decomposition[:, k, variable], label=label
                )
            ax.plot(
                t_axis, self.observed[:, variable], "k--", label="Observed"
            )

        var_name = (
            self.variable_names[variable]
            if self.variable_names
            else f"Variable {variable}"
        )
        if title is None:
            title = f"Historical Decomposition: {var_name}"
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(loc="best", fontsize="small")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class HistoricalDecomposition:
    """Compute historical decomposition from SVAR results.

    Parameters
    ----------
    svar_results : SVARResults
        Fitted SVAR results with structural identification.
    shock_names : list[str] or None
        Optional names for structural shocks.
    variable_names : list[str] or None
        Optional names for variables.
    """

    def __init__(
        self,
        svar_results: Any,
        shock_names: list[str] | None = None,
        variable_names: list[str] | None = None,
    ) -> None:
        self.svar_results = svar_results
        self.shock_names = shock_names
        self.variable_names = variable_names

        # Extract needed attributes
        self.coefs = np.asarray(svar_results.coefs)  # (p, K, K)
        self.A0_inv = np.asarray(svar_results.A0_inv)  # (K, K)
        self.resid = np.asarray(svar_results.resid)  # (T, K)
        self.structural_shocks = np.asarray(
            svar_results.structural_shocks
        )  # (T, K)
        self._n_vars = int(svar_results.k_vars)
        self._n_lags = int(svar_results.lags)
        self._n_obs = self.resid.shape[0]

        self.intercept = getattr(svar_results, "intercept", None)
        if self.intercept is not None:
            self.intercept = np.asarray(self.intercept)

        self._result: HistoricalDecompositionResult | None = None

    @property
    def decomposition(self) -> NDArray[np.floating[Any]]:
        """Decomposition array of shape (T, K_shocks, K_vars)."""
        if self._result is None:
            self._compute()
        assert self._result is not None
        return self._result.decomposition

    @property
    def base(self) -> NDArray[np.floating[Any]]:
        """Base forecast of shape (T, K_vars)."""
        if self._result is None:
            self._compute()
        assert self._result is not None
        return self._result.base

    @property
    def result(self) -> HistoricalDecompositionResult:
        """Full decomposition result."""
        if self._result is None:
            self._compute()
        assert self._result is not None
        return self._result

    def _compute(self) -> None:
        """Compute the historical decomposition."""
        n_k = self._n_vars
        n_p = self._n_lags
        n_t = self._n_obs

        # Step 1: Compute reduced-form MA coefficients phi
        phi = np.zeros((n_t, n_k, n_k))
        phi[0] = np.eye(n_k)
        for h in range(1, n_t):
            for s in range(1, min(h, n_p) + 1):
                phi[h] += phi[h - s] @ self.coefs[s - 1]

        # Step 2: Structural MA coefficients theta_h = phi_h @ A0_inv
        theta = np.zeros((n_t, n_k, n_k))
        for h in range(n_t):
            theta[h] = phi[h] @ self.A0_inv

        # Step 3: Compute HD for each shock
        eps = self.structural_shocks  # (T, K)
        hd = np.zeros((n_t, n_k, n_k))  # (T, K_shocks, K_vars)

        for t in range(n_t):
            for k in range(n_k):
                for j in range(t + 1):
                    hd[t, k, :] += theta[j][:, k] * eps[t - j, k]

        # Step 4: Get observed values and compute base
        observed = self._get_observed(n_t, n_k)

        # Base = observed - sum(HD)
        base = observed - hd.sum(axis=1)

        self._result = HistoricalDecompositionResult(
            decomposition=hd,
            base=base,
            observed=observed,
            structural_shocks=eps,
            n_obs=n_t,
            k_vars=n_k,
            shock_names=self.shock_names,
            variable_names=self.variable_names,
        )

    def _get_observed(self, n_t: int, n_k: int) -> NDArray[np.floating[Any]]:
        """Get observed values from VAR results or reconstruct."""
        n_p = self._n_lags
        if hasattr(self.svar_results, "var_results") and hasattr(
            self.svar_results.var_results, "endog"
        ):
            var_endog = np.asarray(self.svar_results.var_results.endog)
            if var_endog.shape[0] >= n_t + n_p:
                return np.array(var_endog[n_p : n_p + n_t], dtype=float)

        # Fallback: reconstruct from MA representation
        return self._reconstruct_observed()

    def _reconstruct_observed(self) -> NDArray[np.floating[Any]]:
        """Reconstruct observed Y from MA representation."""
        n_k = self._n_vars
        n_p = self._n_lags
        n_t = self._n_obs
        eps = self.structural_shocks

        phi = np.zeros((n_t, n_k, n_k))
        phi[0] = np.eye(n_k)
        for h in range(1, n_t):
            for s in range(1, min(h, n_p) + 1):
                phi[h] += phi[h - s] @ self.coefs[s - 1]

        theta = np.zeros((n_t, n_k, n_k))
        for h in range(n_t):
            theta[h] = phi[h] @ self.A0_inv

        shock_contrib = np.zeros((n_t, n_k))
        for t in range(n_t):
            for j in range(t + 1):
                shock_contrib[t] += theta[j] @ eps[t - j]

        return shock_contrib

    def plot(
        self,
        variable: int,
        stacked: bool = True,
        figsize: tuple[int, int] = (12, 6),
        title: str | None = None,
    ) -> Any:
        """Plot historical decomposition. See HistoricalDecompositionResult.plot."""
        return self.result.plot(variable, stacked, figsize, title)

    def to_dataframe(self, variable: int) -> Any:
        """Convert to DataFrame. See HistoricalDecompositionResult.to_dataframe."""
        return self.result.to_dataframe(variable)
