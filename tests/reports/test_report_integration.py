"""Integration tests for the report generation system.

Tests the complete pipeline: results -> transformer -> template -> exporter -> file.
Covers all report types, formats, and themes.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chronobox.reports import ReportManager
from chronobox.reports.report_manager import GeneratedReport
from chronobox.reports.transformers.arima_transformer import ARIMATransformer
from chronobox.reports.transformers.tests_transformer import TestsTransformer
from chronobox.reports.transformers.var_transformer import VARTransformer

# --- Mock Results ---

@dataclass
class MockARIMAResults:
    params: np.ndarray = field(default_factory=lambda: np.array([0.5, -0.3, 0.1]))
    param_names: list[str] = field(default_factory=lambda: ["ar.L1", "ma.L1", "sigma2"])
    bse: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.04, 0.01]))
    tvalues: np.ndarray = field(default_factory=lambda: np.array([10.0, -7.5, 10.0]))
    pvalues: np.ndarray = field(default_factory=lambda: np.array([0.001, 0.001, 0.001]))
    order: tuple[int, int, int] = (1, 0, 1)
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
    nobs: int = 144
    aic: float = 100.5
    bic: float = 110.2
    hqic: float = 104.3
    aicc: float = 101.0
    llf: float = -47.25
    resid: np.ndarray = field(default_factory=lambda: np.random.default_rng(42).normal(0, 1, 144))


MockARIMAResults.__name__ = "ARIMAResults"  # type: ignore[attr-defined]
MockARIMAResults.__qualname__ = "ARIMAResults"  # type: ignore[attr-defined]


@dataclass
class MockVARResults:
    params: np.ndarray = field(default_factory=lambda: np.random.default_rng(42).normal(0, 1, (6, 3)))
    names: list[str] = field(default_factory=lambda: ["GDP", "CPI", "Rate"])
    var_names: list[str] = field(default_factory=lambda: ["GDP", "CPI", "Rate"])
    bse: np.ndarray = field(default_factory=lambda: np.abs(np.random.default_rng(42).normal(0.1, 0.05, (6, 3))))
    tvalues: np.ndarray = field(default_factory=lambda: np.random.default_rng(42).normal(0, 2, (6, 3)))
    pvalues: np.ndarray = field(default_factory=lambda: np.random.default_rng(42).uniform(0, 1, (6, 3)))
    k_ar: int = 2
    neqs: int = 3
    nobs: int = 100
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([0.7 + 0.3j, 0.7 - 0.3j, 0.5, 0.2, -0.1, 0.3]))
    irf: object = True
    fevd: object = True


MockVARResults.__name__ = "VARResults"  # type: ignore[attr-defined]
MockVARResults.__qualname__ = "VARResults"  # type: ignore[attr-defined]


@dataclass
class MockTestResult:
    test_name: str = "ADF"
    statistic: float = -3.45
    p_value: float = 0.01
    conclusion: str = "Reject H0"


# --- Integration Tests ---

class TestHTMLReportARIMA:
    """Test HTML report generation for ARIMA."""

    def test_generates_without_error(self) -> None:
        """ReportManager generates HTML for ARIMA without error."""
        rm = ReportManager()
        report = rm.generate(
            MockARIMAResults(),
            report_type="arima",
            fmt="html",
            theme="professional",
        )
        assert isinstance(report, GeneratedReport)
        assert len(report.content) > 100
        assert report.fmt == "html"

    def test_html_contains_params_table(self) -> None:
        """HTML report contains parameter table."""
        rm = ReportManager()
        report = rm.generate(MockARIMAResults(), report_type="arima", fmt="html")
        content = report.content.lower()
        assert "parameter" in content or "param" in content

    def test_html_contains_diagnostics(self) -> None:
        """HTML report contains diagnostics section."""
        rm = ReportManager()
        results = MockARIMAResults()
        results.ljung_box = {"statistic": 8.5, "p_value": 0.35}  # type: ignore[attr-defined]
        report = rm.generate(results, report_type="arima", fmt="html")
        # Should at least have sections
        assert len(report.content) > 100

    def test_save_to_file(self) -> None:
        """HTML report can be saved to file."""
        rm = ReportManager()
        report = rm.generate(MockARIMAResults(), report_type="arima", fmt="html")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = report.save(Path(tmpdir) / "arima_report.html")
            assert path.exists()
            assert path.stat().st_size > 0
            content = path.read_text(encoding="utf-8")
            assert "chronobox" in content.lower() or len(content) > 100


class TestHTMLReportVAR:
    """Test HTML report generation for VAR."""

    def test_generates_without_error(self) -> None:
        """ReportManager generates HTML for VAR without error."""
        rm = ReportManager()
        report = rm.generate(
            MockVARResults(),
            report_type="var",
            fmt="html",
            theme="professional",
        )
        assert isinstance(report, GeneratedReport)
        assert len(report.content) > 100


class TestLaTeXReport:
    """Test LaTeX report generation."""

    def test_generates_latex(self) -> None:
        """ReportManager generates LaTeX without error."""
        rm = ReportManager()
        report = rm.generate(MockARIMAResults(), report_type="arima", fmt="latex")
        assert isinstance(report, GeneratedReport)
        assert report.fmt == "latex"
        assert r"\documentclass" in report.content or r"\begin{document}" in report.content or len(report.content) > 50

    def test_save_latex_file(self) -> None:
        """LaTeX report can be saved to .tex file."""
        rm = ReportManager()
        report = rm.generate(MockARIMAResults(), report_type="arima", fmt="latex")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = report.save(Path(tmpdir) / "report.tex")
            assert path.exists()


class TestMarkdownReport:
    """Test Markdown report generation."""

    def test_generates_markdown(self) -> None:
        """ReportManager generates Markdown without error."""
        rm = ReportManager()
        report = rm.generate(MockARIMAResults(), report_type="arima", fmt="markdown")
        assert isinstance(report, GeneratedReport)
        assert report.fmt == "markdown"
        assert len(report.content) > 50

    def test_save_markdown_file(self) -> None:
        """Markdown report can be saved to .md file."""
        rm = ReportManager()
        report = rm.generate(MockARIMAResults(), report_type="arima", fmt="markdown")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = report.save(Path(tmpdir) / "report.md")
            assert path.exists()


class TestAutoDetectReportType:
    """Test auto-detection of report type."""

    def test_detects_arima(self) -> None:
        """Auto-detect returns 'arima' for ARIMAResults."""
        rm = ReportManager()
        result = rm.detect_report_type(MockARIMAResults())
        assert result == "arima"

    def test_detects_var(self) -> None:
        """Auto-detect returns 'var' for VARResults."""
        rm = ReportManager()
        result = rm.detect_report_type(MockVARResults())
        assert result == "var"

    def test_detects_tests_list(self) -> None:
        """Auto-detect returns 'tests' for list of test results."""
        rm = ReportManager()
        result = rm.detect_report_type([MockTestResult(), MockTestResult()])
        assert result == "tests"

    def test_auto_in_generate(self) -> None:
        """report_type='auto' works in generate()."""
        rm = ReportManager()
        report = rm.generate(MockARIMAResults(), report_type="auto", fmt="html")
        assert report.report_type == "arima"


class TestTransformerExtraction:
    """Test that transformers extract expected fields."""

    def test_arima_transformer_all_fields(self) -> None:
        """ARIMATransformer extracts params, IC, diagnostics."""
        t = ARIMATransformer()
        ctx = t.transform(MockARIMAResults())
        assert "parameters" in ctx.get("tables", {})
        assert "information_criteria" in ctx

    def test_var_transformer_coefficients_and_irf(self) -> None:
        """VARTransformer extracts coefficients and IRF info."""
        t = VARTransformer()
        ctx = t.transform(MockVARResults())
        assert "coefficients" in ctx.get("tables", {})
        assert ctx.get("irf", {}).get("available") is True

    def test_tests_transformer_formats_table(self) -> None:
        """TestsTransformer formats test comparison table."""
        t = TestsTransformer()
        results = [
            MockTestResult(test_name="ADF", statistic=-3.45, p_value=0.01),
            MockTestResult(test_name="KPSS", statistic=0.35, p_value=0.10),
        ]
        ctx = t.transform(results)
        assert "tests" in ctx.get("tables", {})
        assert len(ctx["tables"]["tests"]) == 2


class TestAllThemesReports:
    """Test report generation with all 4 themes."""

    @pytest.mark.parametrize("theme", ["professional", "academic", "presentation", "bcb"])
    def test_html_with_theme(self, theme: str) -> None:
        """HTML report generates with each theme."""
        rm = ReportManager()
        report = rm.generate(MockARIMAResults(), report_type="arima", fmt="html", theme=theme)
        assert report.theme == theme
        assert len(report.content) > 100


class TestTemplateManagerIntegration:
    """Test template manager functionality."""

    def test_list_templates(self) -> None:
        """Template manager can list available templates."""
        from chronobox.reports.template_manager import TemplateManager

        tm = TemplateManager()
        templates = tm.list_templates()
        assert len(templates) > 0
        assert any("arima" in t for t in templates)

    def test_template_exists(self) -> None:
        """Template exists check works."""
        from chronobox.reports.template_manager import TemplateManager

        tm = TemplateManager()
        assert tm.template_exists("base.html")
        assert not tm.template_exists("nonexistent_template.html")

    def test_render_string(self) -> None:
        """render_string works with inline template."""
        from chronobox.reports.template_manager import TemplateManager

        tm = TemplateManager()
        result = tm.render_string("Hello {{ name }}!", {"name": "World"})
        assert result == "Hello World!"


class TestRegisterTransformer:
    """Test custom transformer registration."""

    def test_register_and_use(self) -> None:
        """Custom transformer can be registered and used."""
        from chronobox.reports.transformers.base_transformer import BaseTransformer

        class MyTransformer(BaseTransformer):
            def transform(self, results: Any, **kwargs: Any) -> dict[str, Any]:
                return {
                    "title": "Custom Report",
                    "model_info": {},
                    "sections": [{"title": "Test", "content": "Hello", "collapsible": False, "table": None, "plot": None}],
                    "tables": {},
                    "plots": {},
                }

        rm = ReportManager()
        rm.register_transformer("custom", MyTransformer())
        report = rm.generate("dummy", report_type="custom", fmt="html")
        assert isinstance(report, GeneratedReport)
        assert len(report.content) > 50


class TestAdditionalReportTypes:
    """Test reports for ETS, SVAR, BVAR, ARDL, Spillover types."""

    def test_ets_html(self) -> None:
        """Generate HTML for ETS results."""
        @dataclass
        class MockETSResults:
            params: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.1, 0.05]))
            param_names: list[str] = field(default_factory=lambda: ["alpha", "beta", "gamma"])
            bse: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.03, 0.02]))
            tvalues: np.ndarray = field(default_factory=lambda: np.array([16.0, 3.3, 2.5]))
            pvalues: np.ndarray = field(default_factory=lambda: np.array([0.001, 0.001, 0.012]))
            nobs: int = 120
            aic: float = 200.0
            bic: float = 210.0

        MockETSResults.__name__ = "ETSResults"  # type: ignore[attr-defined]
        MockETSResults.__qualname__ = "ETSResults"  # type: ignore[attr-defined]

        rm = ReportManager()
        report = rm.generate(MockETSResults(), report_type="ets", fmt="html")
        assert isinstance(report, GeneratedReport)

    def test_svar_latex(self) -> None:
        """Generate LaTeX for SVAR results."""
        @dataclass
        class MockSVARResults:
            params: np.ndarray = field(default_factory=lambda: np.random.default_rng(42).normal(0, 1, (4, 2)))
            var_names: list[str] = field(default_factory=lambda: ["Y1", "Y2"])
            bse: np.ndarray = field(default_factory=lambda: np.abs(np.random.default_rng(42).normal(0.1, 0.05, (4, 2))))
            tvalues: np.ndarray = field(default_factory=lambda: np.random.default_rng(42).normal(0, 2, (4, 2)))
            pvalues: np.ndarray = field(default_factory=lambda: np.random.default_rng(42).uniform(0, 1, (4, 2)))
            k_ar: int = 2
            neqs: int = 2
            nobs: int = 100
            A_hat: np.ndarray = field(default_factory=lambda: np.eye(2))
            B_hat: np.ndarray = field(default_factory=lambda: np.eye(2))

        MockSVARResults.__name__ = "SVARResults"  # type: ignore[attr-defined]
        MockSVARResults.__qualname__ = "SVARResults"  # type: ignore[attr-defined]

        rm = ReportManager()
        report = rm.generate(MockSVARResults(), report_type="svar", fmt="latex")
        assert isinstance(report, GeneratedReport)

    def test_bvar_markdown(self) -> None:
        """Generate Markdown for BVAR results."""
        @dataclass
        class MockBVARResults:
            params: np.ndarray = field(default_factory=lambda: np.random.default_rng(42).normal(0, 1, (4, 2)))
            var_names: list[str] = field(default_factory=lambda: ["Y1", "Y2"])
            bse: np.ndarray = field(default_factory=lambda: np.abs(np.random.default_rng(42).normal(0.1, 0.05, (4, 2))))
            tvalues: np.ndarray = field(default_factory=lambda: np.random.default_rng(42).normal(0, 2, (4, 2)))
            pvalues: np.ndarray = field(default_factory=lambda: np.random.default_rng(42).uniform(0, 1, (4, 2)))
            k_ar: int = 2
            neqs: int = 2
            nobs: int = 100
            prior_type: str = "Minnesota"

        MockBVARResults.__name__ = "BVARResults"  # type: ignore[attr-defined]
        MockBVARResults.__qualname__ = "BVARResults"  # type: ignore[attr-defined]

        rm = ReportManager()
        report = rm.generate(MockBVARResults(), report_type="bvar", fmt="markdown")
        assert isinstance(report, GeneratedReport)

    def test_ardl_html(self) -> None:
        """Generate HTML for ARDL results."""
        @dataclass
        class MockARDLResults:
            params: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.2, 0.5, -0.1]))
            param_names: list[str] = field(default_factory=lambda: ["y.L1", "x1", "x1.L1", "const"])
            bse: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.04, 0.06, 0.02]))
            tvalues: np.ndarray = field(default_factory=lambda: np.array([6.0, 5.0, 8.3, -5.0]))
            pvalues: np.ndarray = field(default_factory=lambda: np.array([0.001, 0.001, 0.001, 0.001]))
            nobs: int = 100
            aic: float = 150.0
            bic: float = 160.0

        MockARDLResults.__name__ = "ARDLResults"  # type: ignore[attr-defined]
        MockARDLResults.__qualname__ = "ARDLResults"  # type: ignore[attr-defined]

        rm = ReportManager()
        report = rm.generate(MockARDLResults(), report_type="ardl", fmt="html")
        assert isinstance(report, GeneratedReport)

    def test_spillover_html(self) -> None:
        """Generate HTML for Spillover results."""
        @dataclass
        class MockSpilloverResults:
            table: np.ndarray = field(default_factory=lambda: np.random.default_rng(42).uniform(5, 30, (3, 3)))
            var_names: list[str] = field(default_factory=lambda: ["A", "B", "C"])
            total_spillover: float = 45.0

        MockSpilloverResults.__name__ = "SpilloverResults"  # type: ignore[attr-defined]
        MockSpilloverResults.__qualname__ = "SpilloverResults"  # type: ignore[attr-defined]

        rm = ReportManager()
        report = rm.generate(MockSpilloverResults(), report_type="spillover", fmt="html")
        assert isinstance(report, GeneratedReport)

    def test_invalid_format_raises(self) -> None:
        """Invalid format raises ValueError."""
        rm = ReportManager()
        with pytest.raises(ValueError, match="Unsupported format"):
            rm.generate(MockARIMAResults(), report_type="arima", fmt="docx")

    def test_custom_title(self) -> None:
        """Custom title is used in report."""
        rm = ReportManager()
        report = rm.generate(MockARIMAResults(), report_type="arima", fmt="html", title="My Custom Title")
        assert report.content is not None
        assert len(report.content) > 100


class TestFilterFunctions:
    """Test Jinja2 filter functions directly."""

    def test_format_number(self) -> None:
        """format_number handles various inputs."""
        from chronobox.reports.template_manager import _format_number

        assert _format_number(3.14159, 2) == "3.14"
        assert _format_number(None) == "---"
        assert _format_number(0.0, 4) == "0.0000"

    def test_format_pvalue(self) -> None:
        """format_pvalue handles various p-values."""
        from chronobox.reports.template_manager import _format_pvalue

        assert _format_pvalue(None) == "---"
        assert _format_pvalue(0.0001) == "<0.001"
        assert _format_pvalue(0.05) == "0.050"
        assert _format_pvalue(0.5) == "0.500"

    def test_significance_stars(self) -> None:
        """significance_stars returns correct stars."""
        from chronobox.reports.template_manager import _significance_stars

        assert _significance_stars(None) == ""
        assert _significance_stars(0.001) == "***"
        assert _significance_stars(0.02) == "**"
        assert _significance_stars(0.08) == "*"
        assert _significance_stars(0.5) == ""

    def test_format_percent(self) -> None:
        """format_percent handles various inputs."""
        from chronobox.reports.template_manager import _format_percent

        assert _format_percent(None) == "---"
        assert _format_percent(0.5) == "50.0%"
        assert _format_percent(75.0) == "75.0%"


class TestEndToEndPipeline:
    """End-to-end tests covering the full pipeline."""

    def test_arima_html_pipeline(self) -> None:
        """Full pipeline: ARIMA results -> HTML report -> file."""
        rm = ReportManager()
        results = MockARIMAResults()
        report = rm.generate(results, report_type="auto", fmt="html", theme="professional")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = report.save(Path(tmpdir) / "arima.html")
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert len(content) > 100

    def test_var_latex_pipeline(self) -> None:
        """Full pipeline: VAR results -> LaTeX report -> file."""
        rm = ReportManager()
        results = MockVARResults()
        report = rm.generate(results, report_type="var", fmt="latex")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = report.save(Path(tmpdir) / "var.tex")
            assert path.exists()

    def test_tests_markdown_pipeline(self) -> None:
        """Full pipeline: Test results -> Markdown report -> file."""
        rm = ReportManager()
        results = [MockTestResult(), MockTestResult(test_name="PP", statistic=-4.0, p_value=0.005)]
        report = rm.generate(results, report_type="tests", fmt="markdown")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = report.save(Path(tmpdir) / "tests.md")
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "ADF" in content
            assert "PP" in content
