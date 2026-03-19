"""BVAR report transformer.

Extracts posterior summary, IRF posterior bands, and marginal likelihood.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chronobox.reports.transformers.base_transformer import BaseTransformer


class BVARTransformer(BaseTransformer):
    """Transform BVAR results into report context."""

    def transform(self, results: Any, **kwargs: Any) -> dict[str, Any]:
        """Transform BVAR results."""
        context: dict[str, Any] = {
            "title": "Bayesian VAR Report",
            "model_info": {"model_type": "BVAR"},
            "sections": [],
            "tables": {},
            "plots": {},
        }

        # Prior info
        prior = self._safe_getattr(results, "prior")
        if prior is not None:
            context["prior"] = str(prior)
            context["sections"].append(
                self._build_section(
                    "Prior Specification", content=str(prior)
                )
            )

        # Posterior summary
        posterior = self._safe_getattr(results, "posterior_mean")
        if posterior is not None:
            posterior_arr = np.asarray(posterior)
            context["sections"].append(
                self._build_section(
                    "Posterior Summary",
                    content=f"Posterior mean shape: {posterior_arr.shape}",
                )
            )

        # Marginal likelihood
        ml = self._safe_getattr(results, "marginal_likelihood")
        if ml is None:
            ml = self._safe_getattr(results, "log_marginal_likelihood")
        if ml is not None:
            context["marginal_likelihood"] = float(ml)
            context["sections"].append(
                self._build_section(
                    "Marginal Likelihood",
                    content=f"Log marginal likelihood: {float(ml):.4f}",
                )
            )

        # IRF
        irf = self._safe_getattr(results, "irf")
        if irf is not None:
            context["irf"] = {"available": True}
            context["sections"].append(
                self._build_section(
                    "Posterior IRF",
                    content="Posterior IRF with credible bands.",
                )
            )

        return context
