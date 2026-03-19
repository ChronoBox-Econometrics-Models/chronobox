"""Report manager - main orchestrator for report generation.

The ReportManager detects the model type from results, applies the appropriate
transformer to build the template context, renders the template, and exports
to the requested format.

Usage:
    from chronobox.reports import ReportManager

    rm = ReportManager()
    report = rm.generate(results, report_type='auto', fmt='html', theme='professional')
    report.save('output.html')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chronobox.reports.css_manager import CSSManager
from chronobox.reports.template_manager import TemplateManager

# Mapping from results class names to report types
_REPORT_TYPE_MAP: dict[str, str] = {
    "ARIMAResults": "arima",
    "SARIMAResults": "arima",
    "TimeSeriesResults": "arima",
    "ETSResults": "ets",
    "VARResults": "var",
    "VECMResults": "var",
    "SVARResults": "svar",
    "BVARResults": "bvar",
    "ARDLResults": "ardl",
    "SpilloverResults": "spillover",
    "TestResult": "tests",
    "TestResults": "tests",
}


@dataclass
class GeneratedReport:
    """Container for a generated report.

    Attributes
    ----------
    content : str
        Rendered report content.
    fmt : str
        Output format ('html', 'latex', 'markdown').
    report_type : str
        Report type (e.g., 'arima', 'var').
    theme : str
        Theme used for rendering.
    metadata : dict
        Additional metadata.
    """

    content: str
    fmt: str
    report_type: str
    theme: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        """Save the report to a file.

        Parameters
        ----------
        path : str or Path
            Output file path.

        Returns
        -------
        Path
            Path to the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.content, encoding="utf-8")
        return path

    def __str__(self) -> str:
        """Return report content as string."""
        return self.content

    def __repr__(self) -> str:
        """Return summary representation."""
        n_chars = len(self.content)
        return (
            f"GeneratedReport(type={self.report_type!r}, fmt={self.fmt!r}, "
            f"theme={self.theme!r}, chars={n_chars})"
        )


class ReportManager:
    """Main orchestrator for report generation.

    Coordinates transformers, template rendering, and export to produce
    complete reports from model results.

    Parameters
    ----------
    template_dir : str or Path or None
        Custom template directory. If None, uses built-in templates.
    """

    def __init__(self, template_dir: str | Path | None = None) -> None:
        self.template_manager = TemplateManager(template_dir=template_dir)
        self.css_manager = CSSManager()
        self._transformers: dict[str, Any] = {}

    def generate(
        self,
        results: Any,
        report_type: str = "auto",
        fmt: str = "html",
        theme: str = "professional",
        title: str | None = None,
        include_plots: bool = True,
        **kwargs: Any,
    ) -> GeneratedReport:
        """Generate a report from model results.

        Parameters
        ----------
        results : Any
            Model results object (ARIMAResults, VARResults, etc.).
        report_type : str
            Report type: 'arima', 'var', 'svar', 'bvar', 'ets', 'ardl',
            'tests', 'spillover', or 'auto' to detect from results class.
        fmt : str
            Output format: 'html', 'latex', 'markdown'.
        theme : str
            Visual theme: 'professional', 'academic', 'presentation', 'bcb'.
        title : str or None
            Custom report title.
        include_plots : bool
            Whether to include inline plots.
        **kwargs
            Additional keyword arguments passed to the transformer.

        Returns
        -------
        GeneratedReport
            Generated report with content and metadata.

        Raises
        ------
        ValueError
            If report_type cannot be determined or format is unsupported.
        """
        # Auto-detect report type
        if report_type == "auto":
            report_type = self.detect_report_type(results)

        # Validate format
        valid_formats = {"html", "latex", "markdown"}
        if fmt not in valid_formats:
            msg = f"Unsupported format '{fmt}'. Supported: {sorted(valid_formats)}"
            raise ValueError(msg)

        # Get transformer
        transformer = self._get_transformer(report_type)

        # Transform results to template context
        context = transformer.transform(results, **kwargs)

        # Add metadata to context
        context["report_type"] = report_type
        context["theme"] = theme
        context["include_plots"] = include_plots
        if title:
            context["title"] = title
        elif "title" not in context:
            context["title"] = f"{report_type.upper()} Report"

        # Get CSS
        context["css"] = self.css_manager.get_css(theme)

        # Generate report based on format
        if fmt == "html":
            content = self._generate_html(report_type, context)
        elif fmt == "latex":
            content = self._generate_latex(report_type, context)
        elif fmt == "markdown":
            content = self._generate_markdown(report_type, context)
        else:
            content = self._generate_html(report_type, context)

        return GeneratedReport(
            content=content,
            fmt=fmt,
            report_type=report_type,
            theme=theme,
            metadata={
                "results_class": type(results).__name__,
                "include_plots": include_plots,
            },
        )

    def detect_report_type(self, results: Any) -> str:
        """Auto-detect report type from results class.

        Parameters
        ----------
        results : Any
            Model results object.

        Returns
        -------
        str
            Detected report type.

        Raises
        ------
        ValueError
            If report type cannot be determined.
        """
        # Check class name
        class_name = type(results).__name__
        if class_name in _REPORT_TYPE_MAP:
            return _REPORT_TYPE_MAP[class_name]

        # Check MRO (method resolution order) for parent classes
        for cls in type(results).__mro__:
            if cls.__name__ in _REPORT_TYPE_MAP:
                return _REPORT_TYPE_MAP[cls.__name__]

        # Check for list of test results
        if isinstance(results, list) and all(
            hasattr(r, "statistic") for r in results
        ):
            return "tests"

        msg = (
            f"Cannot auto-detect report type for {class_name}. "
            f"Specify report_type explicitly."
        )
        raise ValueError(msg)

    def _get_transformer(self, report_type: str) -> Any:
        """Get the transformer for a given report type.

        Parameters
        ----------
        report_type : str
            Report type name.

        Returns
        -------
        transformer instance

        Raises
        ------
        ValueError
            If no transformer is registered for the type.
        """
        if report_type in self._transformers:
            return self._transformers[report_type]

        # Lazy import transformers
        transformer = self._load_transformer(report_type)
        self._transformers[report_type] = transformer
        return transformer

    def _load_transformer(self, report_type: str) -> Any:
        """Lazy-load a transformer by report type.

        Parameters
        ----------
        report_type : str
            Report type name.

        Returns
        -------
        transformer instance
        """
        try:
            if report_type == "arima":
                from chronobox.reports.transformers.arima_transformer import (
                    ARIMATransformer,
                )

                return ARIMATransformer()
            elif report_type == "ets":
                from chronobox.reports.transformers.ets_transformer import (
                    ETSTransformer,
                )

                return ETSTransformer()
            elif report_type == "var":
                from chronobox.reports.transformers.var_transformer import (
                    VARTransformer,
                )

                return VARTransformer()
            elif report_type == "svar":
                from chronobox.reports.transformers.svar_transformer import (
                    SVARTransformer,
                )

                return SVARTransformer()
            elif report_type == "bvar":
                from chronobox.reports.transformers.bvar_transformer import (
                    BVARTransformer,
                )

                return BVARTransformer()
            elif report_type == "ardl":
                from chronobox.reports.transformers.ardl_transformer import (
                    ARDLTransformer,
                )

                return ARDLTransformer()
            elif report_type == "tests":
                from chronobox.reports.transformers.tests_transformer import (
                    TestsTransformer,
                )

                return TestsTransformer()
            elif report_type == "spillover":
                from chronobox.reports.transformers.spillover_transformer import (
                    SpilloverTransformer,
                )

                return SpilloverTransformer()
            else:
                msg = f"No transformer for report type '{report_type}'."
                raise ValueError(msg)
        except ImportError as e:
            msg = f"Transformer for '{report_type}' not available: {e}"
            raise ValueError(msg) from e

    def _generate_html(self, report_type: str, context: dict[str, Any]) -> str:
        """Generate HTML report.

        Parameters
        ----------
        report_type : str
            Report type.
        context : dict
            Template context.

        Returns
        -------
        str
            Rendered HTML.
        """
        template_name = f"{report_type}_report.html"
        try:
            return self.template_manager.render(template_name, context)
        except Exception:
            # Fallback to generic template
            return self.template_manager.render("generic_report.html", context)

    def _generate_latex(self, report_type: str, context: dict[str, Any]) -> str:
        """Generate LaTeX report.

        Parameters
        ----------
        report_type : str
            Report type.
        context : dict
            Template context.

        Returns
        -------
        str
            Rendered LaTeX.
        """
        try:
            from chronobox.reports.exporters.latex_exporter import LaTeXExporter

            exporter = LaTeXExporter()
            return exporter.export(context)
        except ImportError:
            return self._fallback_latex(context)

    def _generate_markdown(self, report_type: str, context: dict[str, Any]) -> str:
        """Generate Markdown report.

        Parameters
        ----------
        report_type : str
            Report type.
        context : dict
            Template context.

        Returns
        -------
        str
            Rendered Markdown.
        """
        try:
            from chronobox.reports.exporters.markdown_exporter import MarkdownExporter

            exporter = MarkdownExporter()
            return exporter.export(context)
        except ImportError:
            return self._fallback_markdown(context)

    def _fallback_latex(self, context: dict[str, Any]) -> str:
        """Fallback LaTeX generation without template.

        Parameters
        ----------
        context : dict
            Template context.

        Returns
        -------
        str
            Basic LaTeX document.
        """
        title = context.get("title", "Report")
        sections = context.get("sections", [])

        lines = [
            r"\documentclass[11pt,a4paper]{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage{booktabs}",
            r"\usepackage{graphicx}",
            r"\usepackage{float}",
            r"\usepackage[margin=2.5cm]{geometry}",
            r"\usepackage{hyperref}",
            "",
            rf"\title{{{title}}}",
            r"\author{Generated by chronobox}",
            r"\date{\today}",
            "",
            r"\begin{document}",
            r"\maketitle",
            "",
        ]

        for section in sections:
            sec_title = section.get("title", "Section")
            sec_content = section.get("content", "")
            lines.append(rf"\section{{{sec_title}}}")
            lines.append(sec_content)
            lines.append("")

        lines.append(r"\end{document}")
        return "\n".join(lines)

    def _fallback_markdown(self, context: dict[str, Any]) -> str:
        """Fallback Markdown generation without template.

        Parameters
        ----------
        context : dict
            Template context.

        Returns
        -------
        str
            Basic Markdown document.
        """
        title = context.get("title", "Report")
        sections = context.get("sections", [])

        lines = [f"# {title}", "", "*Generated by chronobox*", ""]

        for section in sections:
            sec_title = section.get("title", "Section")
            sec_content = section.get("content", "")
            lines.append(f"## {sec_title}")
            lines.append("")
            lines.append(sec_content)
            lines.append("")

        return "\n".join(lines)

    def register_transformer(self, report_type: str, transformer: Any) -> None:
        """Register a custom transformer.

        Parameters
        ----------
        report_type : str
            Report type name.
        transformer : Any
            Transformer instance with a transform(results) method.
        """
        self._transformers[report_type] = transformer
