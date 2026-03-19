"""Tests report transformer.

Transforms a list or collection of test results into a comparison table
with test names, statistics, p-values, and conclusions.
"""

from __future__ import annotations

from typing import Any

from chronobox.reports.transformers.base_transformer import BaseTransformer


class TestsTransformer(BaseTransformer):
    """Transform test results into a comparison report context."""

    def transform(self, results: Any, **kwargs: Any) -> dict[str, Any]:
        """Transform test results.

        Parameters
        ----------
        results : list of test results, or single test result
            Test results with statistic, p_value, conclusion attributes.

        Returns
        -------
        dict
            Template context.
        """
        context: dict[str, Any] = {
            "title": "Statistical Tests Report",
            "model_info": {"model_type": "Tests"},
            "sections": [],
            "tables": {},
            "plots": {},
        }

        # Normalize to list
        if not isinstance(results, list):
            results = [results]

        test_table: list[dict[str, Any]] = []
        for result in results:
            row = self._extract_test_row(result)
            if row:
                test_table.append(row)

        if test_table:
            context["tables"]["tests"] = test_table
            context["sections"].append(
                self._build_section("Test Results", table=test_table)
            )

            # Summary conclusions
            conclusions = self._generate_conclusions(test_table)
            if conclusions:
                context["conclusions"] = conclusions
                context["sections"].append(
                    self._build_section("Conclusions", content=conclusions)
                )

        return context

    def _extract_test_row(self, result: Any) -> dict[str, Any] | None:
        """Extract a single test result row.

        Parameters
        ----------
        result : Any
            Test result.

        Returns
        -------
        dict or None
        """
        row: dict[str, Any] = {}

        # Test name
        name = self._safe_getattr(result, "test_name")
        if name is None:
            name = self._safe_getattr(result, "name")
        if name is None:
            name = type(result).__name__
        row["test"] = str(name)

        # Statistic
        stat = self._safe_getattr(result, "statistic")
        if stat is None:
            stat = self._safe_getattr(result, "test_statistic")
        if stat is not None:
            row["statistic"] = float(stat)

        # p-value
        pval = self._safe_getattr(result, "p_value")
        if pval is None:
            pval = self._safe_getattr(result, "pvalue")
        if pval is not None:
            row["p_value"] = float(pval)

        # Conclusion
        conclusion = self._safe_getattr(result, "conclusion")
        if conclusion is None:
            conclusion = self._safe_getattr(result, "decision")
        if conclusion is not None:
            row["conclusion"] = str(conclusion)
        elif "p_value" in row:
            row["conclusion"] = (
                "Reject H0" if row["p_value"] < 0.05 else "Fail to reject H0"
            )

        # Critical values
        cv = self._safe_getattr(result, "critical_values")
        if cv is not None:
            row["critical_values"] = cv

        return row if row else None

    @staticmethod
    def _generate_conclusions(table: list[dict[str, Any]]) -> str:
        """Generate summary conclusions from test table.

        Parameters
        ----------
        table : list[dict]
            Test results table.

        Returns
        -------
        str
            Summary text.
        """
        lines = []
        for row in table:
            test = row.get("test", "Unknown")
            conclusion = row.get("conclusion", "N/A")
            pval = row.get("p_value")
            pval_str = f" (p={pval:.4f})" if pval is not None else ""
            lines.append(f"- **{test}**: {conclusion}{pval_str}")
        return "\n".join(lines)
