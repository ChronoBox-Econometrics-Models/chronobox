"""ARDL report transformer.

Extracts short-run coefficients, long-run multipliers, bounds test, and ECM.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chronobox.reports.transformers.base_transformer import BaseTransformer


class ARDLTransformer(BaseTransformer):
    """Transform ARDL results into report context."""

    def transform(self, results: Any, **kwargs: Any) -> dict[str, Any]:
        """Transform ARDL results."""
        context: dict[str, Any] = {
            "title": "ARDL Model Report",
            "model_info": {"model_type": "ARDL"},
            "sections": [],
            "tables": {},
            "plots": {},
        }

        # Model order
        order = self._safe_getattr(results, "order")
        if order is not None:
            context["model_info"]["order"] = order

        # Short-run coefficients
        sr_params = self._safe_getattr(results, "short_run_params")
        if sr_params is None:
            sr_params = self._safe_getattr(results, "params")
        if sr_params is not None:
            sr_arr = np.asarray(sr_params)
            names = self._safe_getattr(results, "param_names") or [
                f"param_{i}" for i in range(len(sr_arr))
            ]
            bse = self._safe_getattr(results, "bse")
            pvalues = self._safe_getattr(results, "pvalues")
            sr_table = self._format_params_table(
                names=list(names),
                values=sr_arr,
                std_errors=np.asarray(bse) if bse is not None else None,
                p_values=np.asarray(pvalues) if pvalues is not None else None,
            )
            context["tables"]["short_run"] = sr_table
            context["sections"].append(
                self._build_section("Short-Run Coefficients", table=sr_table)
            )

        # Long-run multipliers
        lr = self._safe_getattr(results, "long_run_multipliers")
        if lr is None:
            lr = self._safe_getattr(results, "long_run")
        if lr is not None:
            if isinstance(lr, dict):
                lr_table = [
                    {"variable": k, "multiplier": float(v)}
                    for k, v in lr.items()
                ]
            else:
                lr_arr = np.asarray(lr)
                lr_table = [
                    {"variable": f"X{i + 1}", "multiplier": float(v)}
                    for i, v in enumerate(lr_arr)
                ]
            context["tables"]["long_run"] = lr_table
            context["sections"].append(
                self._build_section("Long-Run Multipliers", table=lr_table)
            )

        # Bounds test
        bounds = self._safe_getattr(results, "bounds_test")
        if bounds is not None:
            context["bounds_test"] = bounds
            if isinstance(bounds, dict):
                content = f"F-statistic: {bounds.get('f_statistic', 'N/A')}, "
                content += f"Decision: {bounds.get('decision', 'N/A')}"
            else:
                content = str(bounds)
            context["sections"].append(
                self._build_section("Bounds Test", content=content)
            )

        # ECM coefficient
        ecm = self._safe_getattr(results, "ecm_coefficient")
        if ecm is None:
            ecm = self._safe_getattr(results, "adjustment_speed")
        if ecm is not None:
            context["ecm_coefficient"] = float(ecm)
            context["sections"].append(
                self._build_section(
                    "Error Correction",
                    content=f"ECM coefficient: {float(ecm):.4f}",
                )
            )

        return context
