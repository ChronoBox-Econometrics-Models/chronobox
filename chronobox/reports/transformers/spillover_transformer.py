"""Spillover report transformer.

Extracts spillover table, network info, rolling total, and directional indices.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chronobox.reports.transformers.base_transformer import BaseTransformer


class SpilloverTransformer(BaseTransformer):
    """Transform spillover results into report context."""

    def transform(self, results: Any, **kwargs: Any) -> dict[str, Any]:
        """Transform spillover results.

        Parameters
        ----------
        results : SpilloverResults
            Spillover analysis results.

        Returns
        -------
        dict
            Template context.
        """
        context: dict[str, Any] = {
            "title": "Spillover Analysis Report",
            "model_info": {"model_type": "Spillover"},
            "sections": [],
            "tables": {},
            "plots": {},
        }

        # Spillover table
        table = self._safe_getattr(results, "table")
        if table is None:
            table = self._safe_getattr(results, "spillover_table")
        if table is not None:
            table_arr = np.asarray(table, dtype=np.float64)
            var_names = self._safe_getattr(results, "var_names") or [
                f"Var {i + 1}" for i in range(table_arr.shape[0])
            ]

            # Format as list of dicts
            table_rows: list[dict[str, Any]] = []
            for i in range(table_arr.shape[0]):
                row: dict[str, Any] = {
                    "variable": var_names[i]
                    if i < len(var_names)
                    else f"Var {i + 1}"
                }
                for j in range(table_arr.shape[1]):
                    col_name = (
                        var_names[j]
                        if j < len(var_names)
                        else f"Var {j + 1}"
                    )
                    row[col_name] = float(table_arr[i, j])
                table_rows.append(row)

            context["tables"]["spillover"] = table_rows
            context["sections"].append(
                self._build_section("Spillover Table", table=table_rows)
            )

        # Total spillover index
        tsi = self._safe_getattr(results, "total_spillover_index")
        if tsi is None:
            tsi = self._safe_getattr(results, "total_index")
        if tsi is not None:
            context["total_spillover_index"] = float(tsi)
            context["sections"].append(
                self._build_section(
                    "Total Spillover Index",
                    content=f"Total spillover index: {float(tsi):.2f}%",
                )
            )

        # Directional indices
        dir_from = self._safe_getattr(results, "directional_from")
        dir_to = self._safe_getattr(results, "directional_to")
        net = self._safe_getattr(results, "net_spillover")

        if dir_from is not None or dir_to is not None:
            dir_table: list[dict[str, Any]] = []
            var_names_list = self._safe_getattr(results, "var_names") or []
            n = max(
                len(np.asarray(dir_from)) if dir_from is not None else 0,
                len(np.asarray(dir_to)) if dir_to is not None else 0,
            )
            for i in range(n):
                dir_row: dict[str, Any] = {
                    "variable": var_names_list[i]
                    if i < len(var_names_list)
                    else f"Var {i + 1}",
                }
                if dir_from is not None:
                    dir_row["from_others"] = float(np.asarray(dir_from)[i])
                if dir_to is not None:
                    dir_row["to_others"] = float(np.asarray(dir_to)[i])
                if net is not None:
                    dir_row["net"] = float(np.asarray(net)[i])
                dir_table.append(dir_row)

            context["tables"]["directional"] = dir_table
            context["sections"].append(
                self._build_section("Directional Indices", table=dir_table)
            )

        # Rolling
        rolling = self._safe_getattr(results, "rolling_total")
        if rolling is not None:
            context["rolling"] = {"available": True}
            context["sections"].append(
                self._build_section(
                    "Rolling Spillover",
                    content="Rolling total spillover plot.",
                )
            )

        # Network
        context["sections"].append(
            self._build_section(
                "Network Graph",
                content="Spillover network visualization.",
            )
        )

        return context
