"""Base transformer for all report transformers.

Provides the abstract interface and shared utility methods for
transforming model results into template contexts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class BaseTransformer(ABC):
    """Abstract base class for report transformers.

    All transformers must implement the `transform` method which takes
    a model results object and returns a dictionary suitable for
    template rendering.
    """

    @abstractmethod
    def transform(self, results: Any, **kwargs: Any) -> dict[str, Any]:
        """Transform model results into a template context.

        Parameters
        ----------
        results : Any
            Model results object.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        dict[str, Any]
            Template context dictionary with keys:
            - title: str
            - model_info: dict
            - sections: list[dict]
            - tables: dict[str, list[dict]]
            - plots: dict[str, str] (base64 encoded)
        """
        ...

    @staticmethod
    def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
        """Safely get attribute from object.

        Parameters
        ----------
        obj : Any
            Object.
        attr : str
            Attribute name. Supports dotted notation (e.g. 'a.b.c').
        default : Any
            Default value if not found.

        Returns
        -------
        Any
            Attribute value or default.
        """
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except (AttributeError, TypeError):
            return default

    @staticmethod
    def _format_params_table(
        names: list[str],
        values: NDArray[np.float64] | list[float],
        std_errors: NDArray[np.float64] | list[float] | None = None,
        t_stats: NDArray[np.float64] | list[float] | None = None,
        p_values: NDArray[np.float64] | list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Format parameter estimates into a table.

        Parameters
        ----------
        names : list[str]
            Parameter names.
        values : array-like
            Parameter estimates.
        std_errors : array-like or None
            Standard errors.
        t_stats : array-like or None
            t-statistics.
        p_values : array-like or None
            p-values.

        Returns
        -------
        list[dict]
            Table rows as dictionaries.
        """
        rows = []
        for i, name in enumerate(names):
            row: dict[str, Any] = {
                "name": name,
                "value": float(values[i]) if i < len(values) else None,
            }
            if std_errors is not None and i < len(std_errors):
                row["std_error"] = float(std_errors[i])
            if t_stats is not None and i < len(t_stats):
                row["t_stat"] = float(t_stats[i])
            if p_values is not None and i < len(p_values):
                pval = float(p_values[i])
                row["p_value"] = pval
                if pval < 0.01:
                    row["significance"] = "***"
                elif pval < 0.05:
                    row["significance"] = "**"
                elif pval < 0.1:
                    row["significance"] = "*"
                else:
                    row["significance"] = ""
            rows.append(row)
        return rows

    @staticmethod
    def _format_ic_table(
        aic: float | None = None,
        bic: float | None = None,
        hqic: float | None = None,
        aicc: float | None = None,
        loglike: float | None = None,
    ) -> dict[str, float | None]:
        """Format information criteria into a dictionary.

        Parameters
        ----------
        aic, bic, hqic, aicc : float or None
            Information criteria values.
        loglike : float or None
            Log-likelihood.

        Returns
        -------
        dict
            Named information criteria.
        """
        ic: dict[str, float | None] = {}
        if loglike is not None:
            ic["Log-Likelihood"] = loglike
        if aic is not None:
            ic["AIC"] = aic
        if bic is not None:
            ic["BIC"] = bic
        if hqic is not None:
            ic["HQIC"] = hqic
        if aicc is not None:
            ic["AICc"] = aicc
        return ic

    @staticmethod
    def _build_section(
        title: str,
        content: str = "",
        table: list[dict[str, Any]] | None = None,
        plot_base64: str | None = None,
        collapsible: bool = False,
    ) -> dict[str, Any]:
        """Build a report section.

        Parameters
        ----------
        title : str
            Section title.
        content : str
            HTML or text content.
        table : list[dict] or None
            Table data.
        plot_base64 : str or None
            Base64-encoded plot image.
        collapsible : bool
            Whether section is collapsible.

        Returns
        -------
        dict
            Section dictionary.
        """
        section: dict[str, Any] = {
            "title": title,
            "content": content,
            "collapsible": collapsible,
            "table": table,
            "plot": plot_base64,
        }
        return section
