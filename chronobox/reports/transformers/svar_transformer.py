"""SVAR report transformer.

Extracts identification scheme, IRF, FEVD, and historical decomposition.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chronobox.reports.transformers.base_transformer import BaseTransformer


class SVARTransformer(BaseTransformer):
    """Transform SVAR results into report context."""

    def transform(self, results: Any, **kwargs: Any) -> dict[str, Any]:
        """Transform SVAR results."""
        context: dict[str, Any] = {
            "title": "SVAR Model Report",
            "model_info": {"model_type": "SVAR"},
            "sections": [],
            "tables": {},
            "plots": {},
        }

        # Identification
        identification = self._safe_getattr(results, "identification")
        if identification is None:
            identification = self._safe_getattr(
                results, "identification_scheme"
            )
        if identification is not None:
            context["identification"] = identification
            context["sections"].append(
                self._build_section(
                    "Identification Scheme", content=str(identification)
                )
            )

        # A and B matrices
        for matrix_name in ("A", "B", "A_hat", "B_hat"):
            mat = self._safe_getattr(results, matrix_name)
            if mat is not None:
                mat_arr = np.asarray(mat)
                context[f"matrix_{matrix_name}"] = mat_arr.tolist()
                context["sections"].append(
                    self._build_section(
                        f"Matrix {matrix_name}",
                        content=f"<pre>{mat_arr}</pre>",
                        collapsible=True,
                    )
                )

        # IRF
        irf = self._safe_getattr(results, "irf")
        if irf is not None:
            context["irf"] = {"available": True}
            context["sections"].append(
                self._build_section(
                    "Structural IRF", content="Structural IRF plots."
                )
            )

        # FEVD
        fevd = self._safe_getattr(results, "fevd")
        if fevd is not None:
            context["fevd"] = {"available": True}
            context["sections"].append(
                self._build_section("FEVD", content="Structural FEVD plots.")
            )

        # Historical Decomposition
        hd = self._safe_getattr(results, "historical_decomposition")
        if hd is None:
            hd = self._safe_getattr(results, "hd")
        if hd is not None:
            context["hd"] = {"available": True}
            context["sections"].append(
                self._build_section(
                    "Historical Decomposition", content="HD plots."
                )
            )

        return context
