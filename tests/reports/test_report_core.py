"""Tests for ReportManager, TemplateManager, and CSSManager.

Tests:
    - ReportManager initializes without error
    - auto_detect_report_type works for known classes
    - auto_detect_report_type raises for unknown classes
    - CSSManager generates non-empty CSS for all 4 themes
    - TemplateManager initializes with Jinja2
    - TemplateManager.render_string works
    - GeneratedReport.save writes to file
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from chronobox.reports.css_manager import CSSManager
from chronobox.reports.report_manager import GeneratedReport, ReportManager


@dataclass
class FakeARIMAResults:
    """Mock ARIMA results for testing."""

    pass


# Rename class to match expected detection
FakeARIMAResults.__name__ = "ARIMAResults"
FakeARIMAResults.__qualname__ = "ARIMAResults"


@dataclass
class FakeVARResults:
    """Mock VAR results for testing."""

    pass


FakeVARResults.__name__ = "VARResults"
FakeVARResults.__qualname__ = "VARResults"


@dataclass
class UnknownResults:
    """Unknown results class for testing."""

    pass


class TestReportManager:
    """Tests for ReportManager."""

    def test_initializes(self) -> None:
        """ReportManager initializes without error."""
        rm = ReportManager()
        assert rm is not None

    def test_auto_detect_arima(self) -> None:
        """auto_detect_report_type returns 'arima' for ARIMAResults."""
        rm = ReportManager()
        result = rm.detect_report_type(FakeARIMAResults())
        assert result == "arima"

    def test_auto_detect_var(self) -> None:
        """auto_detect_report_type returns 'var' for VARResults."""
        rm = ReportManager()
        result = rm.detect_report_type(FakeVARResults())
        assert result == "var"

    def test_auto_detect_unknown_raises(self) -> None:
        """auto_detect_report_type raises for unknown class."""
        rm = ReportManager()
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            rm.detect_report_type(UnknownResults())

    def test_auto_detect_test_list(self) -> None:
        """auto_detect_report_type returns 'tests' for list of test results."""

        @dataclass
        class FakeTestResult:
            statistic: float = 1.0

        rm = ReportManager()
        result = rm.detect_report_type([FakeTestResult(), FakeTestResult()])
        assert result == "tests"


class TestGeneratedReport:
    """Tests for GeneratedReport."""

    def test_save_creates_file(self) -> None:
        """GeneratedReport.save writes content to file."""
        report = GeneratedReport(
            content="<html><body>Test</body></html>",
            fmt="html",
            report_type="arima",
            theme="professional",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = report.save(Path(tmpdir) / "test.html")
            assert path.exists()
            assert path.read_text() == "<html><body>Test</body></html>"

    def test_repr(self) -> None:
        """GeneratedReport has informative repr."""
        report = GeneratedReport(
            content="test content",
            fmt="html",
            report_type="arima",
            theme="professional",
        )
        r = repr(report)
        assert "arima" in r
        assert "html" in r

    def test_str_returns_content(self) -> None:
        """str(report) returns the content."""
        report = GeneratedReport(
            content="hello", fmt="html", report_type="arima", theme="professional"
        )
        assert str(report) == "hello"


class TestCSSManager:
    """Tests for CSSManager."""

    def test_all_4_themes_generate_css(self) -> None:
        """CSSManager generates non-empty CSS for all 4 themes."""
        cm = CSSManager()
        for theme in ["professional", "academic", "presentation", "bcb"]:
            css = cm.get_css(theme)
            assert isinstance(css, str)
            assert len(css) > 100
            assert "--text-color" in css
            assert "--accent-color" in css

    def test_css_has_3_layers(self) -> None:
        """CSS contains base, component, and theme layers."""
        cm = CSSManager()
        css = cm.get_css("professional")
        assert "BASE CSS" in css
        assert "COMPONENT CSS" in css
        assert "THEME" in css

    def test_tables_styles(self) -> None:
        """CSS includes table styling."""
        cm = CSSManager()
        css = cm.get_css("professional")
        assert ".cb-table" in css

    def test_collapsible_styles(self) -> None:
        """CSS includes collapsible section styles."""
        cm = CSSManager()
        css = cm.get_css("professional")
        assert ".cb-collapsible" in css


class TestTemplateManager:
    """Tests for TemplateManager."""

    def test_initializes(self) -> None:
        """TemplateManager initializes with Jinja2."""
        from chronobox.reports.template_manager import TemplateManager

        tm = TemplateManager()
        assert tm is not None

    def test_render_string(self) -> None:
        """TemplateManager.render_string works."""
        from chronobox.reports.template_manager import TemplateManager

        tm = TemplateManager()
        result = tm.render_string("Hello {{ name }}!", {"name": "World"})
        assert result == "Hello World!"

    def test_format_number_filter(self) -> None:
        """format_number filter works in templates."""
        from chronobox.reports.template_manager import TemplateManager

        tm = TemplateManager()
        result = tm.render_string("{{ val|format_number(2) }}", {"val": 3.14159})
        assert result == "3.14"

    def test_significance_stars_filter(self) -> None:
        """significance_stars filter works."""
        from chronobox.reports.template_manager import TemplateManager

        tm = TemplateManager()
        result = tm.render_string("{{ pval|significance_stars }}", {"pval": 0.001})
        assert result == "***"
