"""Tests for report transformers.

Tests:
    - ARIMATransformer extracts all expected fields
    - VARTransformer includes coefficients and IRF info
    - SVARTransformer includes HD info
    - TestsTransformer formats table correctly
    - All transformers return valid context structure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chronobox.reports.transformers.ardl_transformer import ARDLTransformer
from chronobox.reports.transformers.arima_transformer import ARIMATransformer
from chronobox.reports.transformers.bvar_transformer import BVARTransformer
from chronobox.reports.transformers.ets_transformer import ETSTransformer
from chronobox.reports.transformers.spillover_transformer import SpilloverTransformer
from chronobox.reports.transformers.svar_transformer import SVARTransformer
from chronobox.reports.transformers.tests_transformer import TestsTransformer
from chronobox.reports.transformers.var_transformer import VARTransformer


@dataclass
class MockARIMAResults:
    """Mock ARIMA results."""

    params: np.ndarray = field(
        default_factory=lambda: np.array([0.5, -0.3, 0.1])
    )
    param_names: list[str] = field(
        default_factory=lambda: ["ar.L1", "ma.L1", "sigma2"]
    )
    bse: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.04, 0.01])
    )
    tvalues: np.ndarray = field(
        default_factory=lambda: np.array([10.0, -7.5, 10.0])
    )
    pvalues: np.ndarray = field(
        default_factory=lambda: np.array([0.001, 0.001, 0.001])
    )
    order: tuple[int, int, int] = (1, 0, 1)
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
    nobs: int = 144
    aic: float = 100.5
    bic: float = 110.2
    hqic: float = 104.3
    aicc: float = 101.0
    llf: float = -47.25
    ljung_box: dict[str, Any] = field(
        default_factory=lambda: {"statistic": 8.5, "p_value": 0.35}
    )
    jarque_bera: tuple[float, float] = (1.2, 0.55)
    resid: np.ndarray = field(
        default_factory=lambda: np.random.default_rng(42).normal(0, 1, 144)
    )
    arroots: np.ndarray = field(
        default_factory=lambda: np.array([1.8 + 0.5j])
    )
    maroots: np.ndarray = field(
        default_factory=lambda: np.array([2.1 - 0.3j])
    )


@dataclass
class MockVARResults:
    """Mock VAR results."""

    params: np.ndarray = field(
        default_factory=lambda: np.random.default_rng(42).normal(0, 1, (6, 3))
    )
    names: list[str] = field(
        default_factory=lambda: ["GDP", "CPI", "Rate"]
    )
    var_names: list[str] = field(
        default_factory=lambda: ["GDP", "CPI", "Rate"]
    )
    bse: np.ndarray = field(
        default_factory=lambda: np.abs(
            np.random.default_rng(42).normal(0.1, 0.05, (6, 3))
        )
    )
    tvalues: np.ndarray = field(
        default_factory=lambda: np.random.default_rng(42).normal(0, 2, (6, 3))
    )
    pvalues: np.ndarray = field(
        default_factory=lambda: np.random.default_rng(42).uniform(0, 1, (6, 3))
    )
    k_ar: int = 2
    neqs: int = 3
    nobs: int = 100
    eigenvalues: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.7 + 0.3j, 0.7 - 0.3j, 0.5, 0.2, -0.1, 0.3]
        )
    )
    irf: object = True
    fevd: object = True


@dataclass
class MockSVARResults:
    """Mock SVAR results."""

    identification: str = "cholesky"
    A_hat: np.ndarray = field(default_factory=lambda: np.eye(3))
    irf: object = True
    fevd: object = True
    hd: np.ndarray = field(
        default_factory=lambda: np.random.default_rng(42).normal(
            0, 1, (100, 3, 3)
        )
    )

    @property
    def historical_decomposition(self) -> np.ndarray:
        """Return HD data."""
        return self.hd


@dataclass
class MockTestResult:
    """Mock test result."""

    test_name: str = "ADF"
    statistic: float = -3.45
    p_value: float = 0.01
    conclusion: str = "Reject unit root"
    critical_values: dict[str, float] = field(
        default_factory=lambda: {"1%": -3.43, "5%": -2.86, "10%": -2.57}
    )


@dataclass
class MockSpilloverResults:
    """Mock spillover results."""

    table: np.ndarray = field(
        default_factory=lambda: np.array(
            [[50, 25, 25], [20, 60, 20], [15, 15, 70]], dtype=float
        )
    )
    var_names: list[str] = field(
        default_factory=lambda: ["GDP", "CPI", "Rate"]
    )
    total_spillover_index: float = 40.0
    directional_from: np.ndarray = field(
        default_factory=lambda: np.array([50.0, 40.0, 30.0])
    )
    directional_to: np.ndarray = field(
        default_factory=lambda: np.array([35.0, 40.0, 45.0])
    )
    net_spillover: np.ndarray = field(
        default_factory=lambda: np.array([-15.0, 0.0, 15.0])
    )


class TestARIMATransformer:
    """Tests for ARIMATransformer."""

    def test_extracts_all_fields(self) -> None:
        """ARIMATransformer extracts params, IC, diagnostics, roots."""
        t = ARIMATransformer()
        ctx = t.transform(MockARIMAResults())

        assert ctx["title"] == "ARIMA Model Report"
        assert "parameters" in ctx["tables"]
        assert len(ctx["tables"]["parameters"]) == 3
        assert "information_criteria" in ctx
        assert "AIC" in ctx["information_criteria"]
        assert "BIC" in ctx["information_criteria"]
        assert "diagnostics" in ctx
        assert "roots" in ctx
        assert len(ctx["sections"]) >= 3

    def test_params_have_significance(self) -> None:
        """Parameters include significance stars."""
        t = ARIMATransformer()
        ctx = t.transform(MockARIMAResults())
        for param in ctx["tables"]["parameters"]:
            assert "significance" in param


class TestVARTransformer:
    """Tests for VARTransformer."""

    def test_includes_coefficients_and_irf(self) -> None:
        """VARTransformer extracts coefficients by equation and IRF."""
        t = VARTransformer()
        ctx = t.transform(MockVARResults())

        assert ctx["title"] == "VAR Model Report"
        assert "coefficients" in ctx["tables"]
        assert "GDP" in ctx["tables"]["coefficients"]
        assert "irf" in ctx
        assert ctx["irf"]["available"] is True

    def test_includes_stability(self) -> None:
        """VARTransformer includes stability analysis."""
        t = VARTransformer()
        ctx = t.transform(MockVARResults())
        assert "stability" in ctx
        assert "is_stable" in ctx["stability"]


class TestSVARTransformer:
    """Tests for SVARTransformer."""

    def test_includes_hd(self) -> None:
        """SVARTransformer includes historical decomposition."""
        t = SVARTransformer()
        results = MockSVARResults()
        ctx = t.transform(results)

        assert "SVAR" in ctx["title"]
        assert any("Identification" in s["title"] for s in ctx["sections"])


class TestTestsTransformer:
    """Tests for TestsTransformer."""

    def test_formats_table(self) -> None:
        """TestsTransformer creates formatted test table."""
        t = TestsTransformer()
        results = [
            MockTestResult(test_name="ADF", statistic=-3.45, p_value=0.01),
            MockTestResult(test_name="KPSS", statistic=0.35, p_value=0.10),
            MockTestResult(test_name="PP", statistic=-4.12, p_value=0.001),
        ]
        ctx = t.transform(results)

        assert "tests" in ctx["tables"]
        assert len(ctx["tables"]["tests"]) == 3
        assert ctx["tables"]["tests"][0]["test"] == "ADF"
        assert ctx["tables"]["tests"][0]["statistic"] == -3.45

    def test_single_result(self) -> None:
        """TestsTransformer handles single result (not list)."""
        t = TestsTransformer()
        ctx = t.transform(MockTestResult())
        assert "tests" in ctx["tables"]
        assert len(ctx["tables"]["tests"]) == 1


class TestSpilloverTransformer:
    """Tests for SpilloverTransformer."""

    def test_extracts_table(self) -> None:
        """SpilloverTransformer extracts spillover table."""
        t = SpilloverTransformer()
        ctx = t.transform(MockSpilloverResults())
        assert "spillover" in ctx["tables"]
        assert ctx["total_spillover_index"] == 40.0

    def test_directional_indices(self) -> None:
        """SpilloverTransformer includes directional indices."""
        t = SpilloverTransformer()
        ctx = t.transform(MockSpilloverResults())
        assert "directional" in ctx["tables"]
        assert len(ctx["tables"]["directional"]) == 3


class TestAllTransformers:
    """Tests that all 8 transformers produce valid context structure."""

    def test_all_return_required_keys(self) -> None:
        """All transformers return context with required keys."""
        transformers: list[tuple[Any, Any]] = [
            (ARIMATransformer(), MockARIMAResults()),
            (ETSTransformer(), MockARIMAResults()),  # reuse mock
            (VARTransformer(), MockVARResults()),
            (SVARTransformer(), MockSVARResults()),
            (BVARTransformer(), MockVARResults()),  # reuse mock
            (ARDLTransformer(), MockARIMAResults()),  # reuse mock
            (TestsTransformer(), [MockTestResult()]),
            (SpilloverTransformer(), MockSpilloverResults()),
        ]

        for transformer, results in transformers:
            ctx = transformer.transform(results)
            assert "title" in ctx, f"{type(transformer).__name__} missing 'title'"
            assert "sections" in ctx, f"{type(transformer).__name__} missing 'sections'"
            assert isinstance(ctx["sections"], list)
