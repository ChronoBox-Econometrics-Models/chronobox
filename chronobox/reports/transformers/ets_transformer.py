"""ETS report transformer.

Extracts parameters, decomposition components, forecast, and initial states.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chronobox.reports.transformers.base_transformer import BaseTransformer


class ETSTransformer(BaseTransformer):
    """Transform ETS results into report context."""

    def transform(self, results: Any, **kwargs: Any) -> dict[str, Any]:
        """Transform ETS results.

        Parameters
        ----------
        results : ETSResults
            ETS model results.

        Returns
        -------
        dict
            Template context.
        """
        context: dict[str, Any] = {
            "title": "ETS Model Report",
            "model_info": {},
            "sections": [],
            "tables": {},
            "plots": {},
        }

        # Model info
        for attr in ("error", "trend", "seasonal", "damped", "seasonal_periods"):
            val = self._safe_getattr(results, attr)
            if val is not None:
                context["model_info"][attr] = val

        # Parameters
        params = self._safe_getattr(results, "params")
        if params is not None:
            names = self._safe_getattr(results, "param_names") or [
                f"param_{i}" for i in range(len(params))
            ]
            params_table = self._format_params_table(
                names=list(names),
                values=np.asarray(params),
            )
            context["tables"]["parameters"] = params_table
            context["sections"].append(
                self._build_section("Parameter Estimates", table=params_table)
            )

        # Information criteria
        ic = self._format_ic_table(
            aic=self._safe_getattr(results, "aic"),
            bic=self._safe_getattr(results, "bic"),
            aicc=self._safe_getattr(results, "aicc"),
            loglike=self._safe_getattr(results, "llf"),
        )
        if ic:
            context["information_criteria"] = ic
            context["sections"].append(
                self._build_section("Information Criteria", content=str(ic))
            )

        # Initial state
        initial_state = self._safe_getattr(results, "initial_state")
        if initial_state is not None:
            context["initial_state"] = list(np.asarray(initial_state).ravel())
            context["sections"].append(
                self._build_section(
                    "Initial State",
                    content=f"Initial state values: {context['initial_state']}",
                    collapsible=True,
                )
            )

        return context
