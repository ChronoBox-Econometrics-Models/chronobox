"""
Counterfactual analysis based on historical decomposition.

Answers: "What would have happened if shock k had not occurred?"

References
----------
- Kilian, L. & Lutkepohl, H. (2017). Structural Vector Autoregressive
  Analysis. Cambridge University Press. Chapter 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class CounterfactualResult:
    """Results from counterfactual analysis.

    Attributes
    ----------
    counterfactual : ndarray of shape (T, K_vars)
        Counterfactual time series.
    observed : ndarray of shape (T, K_vars)
        Observed time series.
    removed_contribution : ndarray of shape (T, K_vars)
        The contribution that was removed.
    shock_index : int | list[int]
        Index(es) of the shock(s) removed.
    scale : float
        Scale factor applied (0 = full removal, 0.5 = half removal).
    """

    counterfactual: NDArray[np.floating[Any]]
    observed: NDArray[np.floating[Any]]
    removed_contribution: NDArray[np.floating[Any]]
    shock_index: int | list[int]
    scale: float


class Counterfactual:
    """Counterfactual analysis based on historical decomposition.

    Parameters
    ----------
    hd : HistoricalDecomposition or HistoricalDecompositionResult
        Historical decomposition results.
    """

    def __init__(self, hd: Any) -> None:
        # Accept either HistoricalDecomposition or HistoricalDecompositionResult
        if hasattr(hd, "result"):
            self._hd_result = hd.result
        elif hasattr(hd, "decomposition") and hasattr(hd, "base"):
            self._hd_result = hd
        else:
            msg = "hd must be a HistoricalDecomposition or HistoricalDecompositionResult"
            raise TypeError(msg)

    @property
    def decomposition(self) -> NDArray[np.floating[Any]]:
        """Decomposition array (T, K_shocks, K_vars)."""
        return self._hd_result.decomposition

    @property
    def base(self) -> NDArray[np.floating[Any]]:
        """Base forecast (T, K_vars)."""
        return self._hd_result.base

    @property
    def observed(self) -> NDArray[np.floating[Any]]:
        """Observed values (T, K_vars)."""
        return self._hd_result.observed

    def without_shock(
        self, shock_index: int | list[int]
    ) -> NDArray[np.floating[Any]]:
        """Compute counterfactual without specified shock(s).

        Y_cf = Y - HD_k  (for single shock)
        Y_cf = Y - sum(HD_k for k in shock_indices)  (for multiple shocks)

        Parameters
        ----------
        shock_index : int or list[int]
            Index(es) of the structural shock(s) to remove.

        Returns
        -------
        ndarray of shape (T, K_vars)
        """
        if isinstance(shock_index, int):
            shock_index = [shock_index]

        removed = np.zeros_like(self.observed)
        for k in shock_index:
            removed += self.decomposition[:, k, :]

        return self.observed - removed

    def with_modified_shock(
        self, shock_index: int, scale: float = 0.5
    ) -> NDArray[np.floating[Any]]:
        """Compute counterfactual with modified (scaled) shock.

        Y_cf = Y - HD_k * (1 - scale)

        Parameters
        ----------
        shock_index : int
            Index of the structural shock to modify.
        scale : float
            Scale factor. 0 = remove entirely, 1 = keep as-is, 0.5 = halve.

        Returns
        -------
        ndarray of shape (T, K_vars)
        """
        contribution = self.decomposition[:, shock_index, :]
        return self.observed - contribution * (1.0 - scale)

    def without_all_shocks(self) -> NDArray[np.floating[Any]]:
        """Remove all shocks, leaving only the base forecast.

        Returns
        -------
        ndarray of shape (T, K_vars)
            Should equal self.base.
        """
        n_shocks = self.decomposition.shape[1]
        return self.without_shock(list(range(n_shocks)))

    def compute(
        self,
        shock_index: int | list[int],
        scale: float = 0.0,
    ) -> CounterfactualResult:
        """Compute counterfactual with full result object.

        Parameters
        ----------
        shock_index : int or list[int]
            Shock(s) to modify.
        scale : float
            Scale factor (0 = remove, 1 = keep).

        Returns
        -------
        CounterfactualResult
        """
        if isinstance(shock_index, int):
            cf = self.with_modified_shock(shock_index, scale)
            removed = self.decomposition[:, shock_index, :] * (1.0 - scale)
        else:
            removed = np.zeros_like(self.observed)
            for k in shock_index:
                removed += self.decomposition[:, k, :] * (1.0 - scale)
            cf = self.observed - removed

        return CounterfactualResult(
            counterfactual=cf,
            observed=self.observed,
            removed_contribution=removed,
            shock_index=shock_index,
            scale=scale,
        )

    def plot(
        self,
        variable: int,
        shock_index: int | list[int],
        scale: float = 0.0,
        figsize: tuple[int, int] = (12, 6),
        title: str | None = None,
        variable_name: str | None = None,
        shock_name: str | None = None,
    ) -> Any:
        """Plot counterfactual vs observed.

        Parameters
        ----------
        variable : int
            Variable index to plot.
        shock_index : int or list[int]
            Shock(s) to remove.
        scale : float
            Scale factor (0 = remove, 1 = keep).
        figsize : tuple
            Figure size.
        title : str or None
            Plot title.
        variable_name : str or None
            Name of the variable for the plot label.
        shock_name : str or None
            Name of the shock for the plot label.

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as plt

        result = self.compute(shock_index, scale)

        fig, ax = plt.subplots(figsize=figsize)

        n_t = result.observed.shape[0]
        t_axis = np.arange(n_t)

        ax.plot(
            t_axis,
            result.observed[:, variable],
            "b-",
            linewidth=1.5,
            label="Observed",
        )
        ax.plot(
            t_axis,
            result.counterfactual[:, variable],
            "r--",
            linewidth=1.5,
            label="Counterfactual",
        )

        # Shade the difference
        ax.fill_between(
            t_axis,
            result.observed[:, variable],
            result.counterfactual[:, variable],
            alpha=0.2,
            color="gray",
            label="Shock contribution",
        )

        if variable_name is None:
            variable_name = f"Variable {variable}"
        if shock_name is None:
            shock_name = f"Shock {shock_index}"

        if title is None:
            if scale == 0:
                title = (
                    f"Counterfactual: {variable_name} without {shock_name}"
                )
            else:
                title = f"Counterfactual: {variable_name} with {shock_name} scaled by {scale:.1f}"

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
