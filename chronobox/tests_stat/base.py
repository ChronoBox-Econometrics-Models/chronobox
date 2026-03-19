"""
Base classes for statistical tests.

Provides the standardized TestResult dataclass used by all statistical tests
in chronobox.tests_stat.

References
----------
- Provides uniform interface for unit root, cointegration, structural break,
  and specification tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestResult:
    """Standardized result container for all statistical tests.

    Parameters
    ----------
    test_name : str
        Name of the test (e.g., "ADF", "KPSS", "Ljung-Box").
    statistic : float
        Value of the test statistic.
    pvalue : float or None
        p-value of the test. None if not available (e.g., table lookup only).
    critical_values : dict[str, float]
        Critical values at standard significance levels.
        Example: {'1%': -3.43, '5%': -2.86, '10%': -2.57}.
    null_hypothesis : str
        Description of H0.
    alternative_hypothesis : str
        Description of H1.
    reject_at_5pct : bool
        True if H0 is rejected at 5% significance level.
    lags_used : int or None
        Number of lags used in the test (if applicable).
    additional_info : dict
        Additional test-specific information (e.g., break dates, vectors).
    """

    test_name: str
    statistic: float
    pvalue: float | None
    critical_values: dict[str, float]
    null_hypothesis: str
    alternative_hypothesis: str
    reject_at_5pct: bool
    lags_used: int | None = None
    additional_info: dict[str, Any] = field(default_factory=lambda: {})

    def summary(self) -> str:
        """Return a formatted summary string of the test result.

        Returns
        -------
        str
            Multi-line summary including test name, statistic, p-value,
            critical values, hypotheses, and decision.
        """
        lines: list[str] = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  {self.test_name} Test")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Test statistic : {self.statistic:.6f}")

        if self.pvalue is not None:
            lines.append(f"  p-value        : {self.pvalue:.6f}")
        else:
            lines.append("  p-value        : N/A")

        if self.lags_used is not None:
            lines.append(f"  Lags used      : {self.lags_used}")

        lines.append("")
        lines.append(f"  H0: {self.null_hypothesis}")
        lines.append(f"  H1: {self.alternative_hypothesis}")
        lines.append("")

        if self.critical_values:
            lines.append("  Critical Values:")
            for level, value in sorted(self.critical_values.items()):
                lines.append(f"    {level:>5s} : {value:.4f}")

        lines.append("")
        decision = "Reject H0" if self.reject_at_5pct else "Fail to reject H0"
        lines.append(f"  Decision (5%)  : {decision}")

        if self.additional_info:
            lines.append("")
            lines.append("  Additional Info:")
            for key, value in self.additional_info.items():
                if isinstance(value, float):
                    lines.append(f"    {key}: {value:.6f}")
                elif hasattr(value, "__len__") and len(value) > 10:
                    lines.append(f"    {key}: array of length {len(value)}")
                else:
                    lines.append(f"    {key}: {value}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return concise representation of the test result."""
        decision = "Reject H0" if self.reject_at_5pct else "Fail to reject H0"
        pval_str = f"{self.pvalue:.4f}" if self.pvalue is not None else "N/A"
        return (
            f"TestResult({self.test_name}: "
            f"stat={self.statistic:.4f}, "
            f"pval={pval_str}, "
            f"{decision})"
        )
