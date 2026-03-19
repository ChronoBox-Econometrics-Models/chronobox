"""Template manager for Jinja2 template loading and rendering.

Manages template inheritance (base.html -> specific_report.html) and
provides methods to load, render, and list available templates.

Usage:
    from chronobox.reports.template_manager import TemplateManager

    tm = TemplateManager()
    html = tm.render('arima_report.html', context)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import jinja2

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


_BUILTIN_TEMPLATE_DIR = Path(__file__).parent / "templates"


class TemplateManager:
    """Manage Jinja2 templates for report rendering.

    Supports template inheritance via Jinja2's {% extends %} and {% block %}
    directives. Templates are loaded from the built-in template directory
    or a custom directory.

    Parameters
    ----------
    template_dir : str or Path or None
        Custom template directory. If None, uses built-in templates.
    """

    def __init__(self, template_dir: str | Path | None = None) -> None:
        if not HAS_JINJA2:
            msg = (
                "Jinja2 is required for report generation. "
                "Install with: pip install jinja2"
            )
            raise ImportError(msg)

        self._template_dirs: list[Path] = []

        # Add custom directory first (higher priority)
        if template_dir is not None:
            custom_dir = Path(template_dir)
            if custom_dir.is_dir():
                self._template_dirs.append(custom_dir)

        # Add built-in directory
        self._template_dirs.append(_BUILTIN_TEMPLATE_DIR)

        # Create Jinja2 environment
        loaders = [jinja2.FileSystemLoader(str(d)) for d in self._template_dirs]
        self._env = jinja2.Environment(
            loader=jinja2.ChoiceLoader(loaders),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.StrictUndefined,
        )

        # Register custom filters
        self._register_filters()

    def render(self, template_name: str, context: dict[str, Any]) -> str:
        """Render a template with the given context.

        Parameters
        ----------
        template_name : str
            Template filename.
        context : dict
            Template context variables.

        Returns
        -------
        str
            Rendered content.

        Raises
        ------
        jinja2.TemplateNotFound
            If template does not exist.
        """
        try:
            template = self._env.get_template(template_name)
        except jinja2.TemplateNotFound:
            # Try fallback to generic
            try:
                template = self._env.get_template("generic_report.html")
            except jinja2.TemplateNotFound:
                # Generate inline if no template found
                return self._inline_render(context)

        return template.render(**context)

    def render_string(self, template_string: str, context: dict[str, Any]) -> str:
        """Render a template from a string.

        Parameters
        ----------
        template_string : str
            Jinja2 template string.
        context : dict
            Template context variables.

        Returns
        -------
        str
            Rendered content.
        """
        template = self._env.from_string(template_string)
        return template.render(**context)

    def list_templates(self) -> list[str]:
        """List available template names.

        Returns
        -------
        list[str]
            Sorted list of template filenames.
        """
        templates: set[str] = set()
        for d in self._template_dirs:
            if d.is_dir():
                for f in d.rglob("*.html"):
                    templates.add(f.name)
                for f in d.rglob("*.tex"):
                    templates.add(f.name)
                for f in d.rglob("*.md"):
                    templates.add(f.name)
        return sorted(templates)

    def template_exists(self, template_name: str) -> bool:
        """Check if a template exists.

        Parameters
        ----------
        template_name : str
            Template filename.

        Returns
        -------
        bool
        """
        try:
            self._env.get_template(template_name)
            return True
        except jinja2.TemplateNotFound:
            return False

    def _register_filters(self) -> None:
        """Register custom Jinja2 filters."""
        self._env.filters["format_number"] = _format_number
        self._env.filters["format_pvalue"] = _format_pvalue
        self._env.filters["significance_stars"] = _significance_stars
        self._env.filters["format_percent"] = _format_percent

    def _inline_render(self, context: dict[str, Any]) -> str:
        """Generate inline HTML when no template is available.

        Parameters
        ----------
        context : dict
            Template context.

        Returns
        -------
        str
            Basic HTML report.
        """
        title = context.get("title", "Report")
        css = context.get("css", "")
        sections = context.get("sections", [])

        section_html = ""
        for section in sections:
            sec_title = section.get("title", "")
            sec_content = section.get("content", "")
            section_html += f"<section><h2>{sec_title}</h2>{sec_content}</section>\n"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>{css}</style>
</head>
<body>
    <header><h1>{title}</h1></header>
    <main>{section_html}</main>
    <footer><p>Generated by chronobox</p></footer>
</body>
</html>"""


def _format_number(value: float | None, decimals: int = 4) -> str:
    """Format a number with specified decimal places.

    Parameters
    ----------
    value : float or None
        Number to format.
    decimals : int
        Number of decimal places.

    Returns
    -------
    str
        Formatted number.
    """
    if value is None:
        return "---"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_pvalue(value: float | None) -> str:
    """Format a p-value with appropriate precision.

    Parameters
    ----------
    value : float or None
        P-value.

    Returns
    -------
    str
        Formatted p-value.
    """
    if value is None:
        return "---"
    try:
        v = float(value)
        if v < 0.001:
            return "<0.001"
        elif v < 0.01 or v < 0.1:
            return f"{v:.3f}"
        else:
            return f"{v:.3f}"
    except (TypeError, ValueError):
        return str(value)


def _significance_stars(pvalue: float | None) -> str:
    """Return significance stars based on p-value.

    Parameters
    ----------
    pvalue : float or None
        P-value.

    Returns
    -------
    str
        Stars: '***' (p<0.01), '**' (p<0.05), '*' (p<0.1), '' otherwise.
    """
    if pvalue is None:
        return ""
    try:
        v = float(pvalue)
        if v < 0.01:
            return "***"
        elif v < 0.05:
            return "**"
        elif v < 0.1:
            return "*"
        return ""
    except (TypeError, ValueError):
        return ""


def _format_percent(value: float | None, decimals: int = 1) -> str:
    """Format a value as percentage.

    Parameters
    ----------
    value : float or None
        Value (0-1 scale or already percentage).
    decimals : int
        Decimal places.

    Returns
    -------
    str
        Formatted percentage.
    """
    if value is None:
        return "---"
    try:
        v = float(value)
        if abs(v) <= 1.0:
            v *= 100
        return f"{v:.{decimals}f}%"
    except (TypeError, ValueError):
        return str(value)
