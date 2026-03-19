"""Tests for report exporters.

Tests:
    - HTMLExporter generates self-contained HTML
    - LaTeXExporter generates valid LaTeX
    - MarkdownExporter generates valid Markdown
    - All exporters handle empty context gracefully
"""

from __future__ import annotations

import pytest

from chronobox.reports.exporters.html_exporter import HTMLExporter
from chronobox.reports.exporters.latex_exporter import LaTeXExporter
from chronobox.reports.exporters.markdown_exporter import MarkdownExporter


@pytest.fixture
def sample_context() -> dict:
    """Generate sample report context."""
    return {
        "title": "Test Report",
        "css": "body { color: #333; }",
        "model_info": {
            "model_type": "ARIMA",
            "order": "(1, 1, 1)",
            "nobs": 144,
        },
        "sections": [
            {
                "title": "Parameters",
                "content": "Parameter estimates below.",
                "table": [
                    {
                        "name": "ar.L1",
                        "value": 0.5,
                        "std_error": 0.05,
                        "p_value": 0.001,
                        "significance": "***",
                    },
                    {
                        "name": "ma.L1",
                        "value": -0.3,
                        "std_error": 0.04,
                        "p_value": 0.001,
                        "significance": "***",
                    },
                ],
                "collapsible": False,
            },
            {
                "title": "Diagnostics",
                "content": "All diagnostics pass.",
                "table": [
                    {
                        "test": "Ljung-Box",
                        "statistic": 8.5,
                        "p_value": 0.35,
                        "conclusion": "No autocorrelation",
                    },
                ],
                "collapsible": True,
            },
        ],
    }


class TestHTMLExporter:
    """Tests for HTMLExporter."""

    def test_generates_html(self, sample_context: dict) -> None:
        """HTMLExporter produces valid HTML."""
        exporter = HTMLExporter()
        html = exporter.export(sample_context)
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "Test Report" in html
        assert "chronobox" in html

    def test_contains_sidebar(self, sample_context: dict) -> None:
        """HTML contains sidebar navigation."""
        exporter = HTMLExporter()
        html = exporter.export(sample_context)
        assert "report-sidebar" in html
        assert "Parameters" in html

    def test_contains_table(self, sample_context: dict) -> None:
        """HTML contains formatted table."""
        exporter = HTMLExporter()
        html = exporter.export(sample_context)
        assert "cb-table" in html
        assert "ar.L1" in html

    def test_contains_collapsible(self, sample_context: dict) -> None:
        """HTML contains collapsible sections."""
        exporter = HTMLExporter()
        html = exporter.export(sample_context)
        assert "cb-collapsible" in html

    def test_contains_javascript(self, sample_context: dict) -> None:
        """HTML contains interactive JavaScript."""
        exporter = HTMLExporter()
        html = exporter.export(sample_context)
        assert "<script>" in html
        assert "scrollIntoView" in html

    def test_empty_context(self) -> None:
        """HTMLExporter handles empty context."""
        exporter = HTMLExporter()
        html = exporter.export({"title": "Empty"})
        assert "<!DOCTYPE html>" in html


class TestLaTeXExporter:
    """Tests for LaTeXExporter."""

    def test_generates_latex(self, sample_context: dict) -> None:
        """LaTeXExporter produces valid LaTeX."""
        exporter = LaTeXExporter()
        latex = exporter.export(sample_context)
        assert r"\documentclass" in latex
        assert r"\begin{document}" in latex
        assert r"\end{document}" in latex

    def test_uses_booktabs(self, sample_context: dict) -> None:
        """LaTeX uses booktabs for tables."""
        exporter = LaTeXExporter()
        latex = exporter.export(sample_context)
        assert r"\usepackage{booktabs}" in latex
        assert r"\toprule" in latex
        assert r"\midrule" in latex
        assert r"\bottomrule" in latex

    def test_contains_sections(self, sample_context: dict) -> None:
        """LaTeX has sections."""
        exporter = LaTeXExporter()
        latex = exporter.export(sample_context)
        assert r"\section{" in latex

    def test_contains_metadata_footer(self, sample_context: dict) -> None:
        """LaTeX has metadata footer."""
        exporter = LaTeXExporter()
        latex = exporter.export(sample_context)
        assert "chronobox" in latex

    def test_empty_context(self) -> None:
        """LaTeXExporter handles empty context."""
        exporter = LaTeXExporter()
        latex = exporter.export({"title": "Empty"})
        assert r"\begin{document}" in latex


class TestMarkdownExporter:
    """Tests for MarkdownExporter."""

    def test_generates_markdown(self, sample_context: dict) -> None:
        """MarkdownExporter produces valid Markdown."""
        exporter = MarkdownExporter()
        md = exporter.export(sample_context)
        assert "# Test Report" in md
        assert "## Parameters" in md

    def test_contains_table(self, sample_context: dict) -> None:
        """Markdown contains formatted table."""
        exporter = MarkdownExporter()
        md = exporter.export(sample_context)
        assert "|" in md
        assert "ar.L1" in md

    def test_contains_footer(self, sample_context: dict) -> None:
        """Markdown contains footer."""
        exporter = MarkdownExporter()
        md = exporter.export(sample_context)
        assert "chronobox" in md

    def test_empty_context(self) -> None:
        """MarkdownExporter handles empty context."""
        exporter = MarkdownExporter()
        md = exporter.export({"title": "Empty"})
        assert "# Empty" in md

    def test_github_compatible(self, sample_context: dict) -> None:
        """Markdown tables are GitHub-compatible (pipe syntax)."""
        exporter = MarkdownExporter()
        md = exporter.export(sample_context)
        lines = md.split("\n")
        table_lines = [line for line in lines if line.startswith("|")]
        assert len(table_lines) >= 3  # header + separator + at least 1 data row
