"""Report generation system for chronobox.

Provides automated report generation from model results with support for
HTML, LaTeX, and Markdown output formats.

Architecture:
    results -> Transformer -> template_context -> TemplateManager -> rendered -> Exporter -> file

Usage:
    from chronobox.reports import ReportManager

    rm = ReportManager()
    report = rm.generate(results, report_type='auto', fmt='html', theme='professional')
    report.save('output.html')
"""

from __future__ import annotations

from chronobox.reports.report_manager import ReportManager

__all__ = ["ReportManager"]
