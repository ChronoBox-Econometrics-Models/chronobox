"""CSS manager for report styling.

Implements a 3-layer CSS system:
    1. Base: Reset, typography, layout grid
    2. Component: Tables, cards, sections, collapsible elements
    3. Theme: Colors and fonts specific to each visual theme

Usage:
    from chronobox.reports.css_manager import CSSManager

    css = CSSManager()
    styles = css.get_css('professional')
"""

from __future__ import annotations


class CSSManager:
    """Manage CSS styles for report generation.

    Three layers of CSS are combined:
    - Base: Core reset, typography, and layout
    - Component: Reusable UI components (tables, cards, etc.)
    - Theme: Visual theme overrides (colors, fonts)
    """

    def get_css(self, theme: str = "professional") -> str:
        """Get combined CSS for a given theme.

        Parameters
        ----------
        theme : str
            Theme name: 'professional', 'academic', 'presentation', 'bcb'.

        Returns
        -------
        str
            Combined CSS string.
        """
        return "\n".join([
            self._base_css(),
            self._component_css(),
            self._theme_css(theme),
        ])

    def _base_css(self) -> str:
        """Base CSS: reset, typography, layout.

        Returns
        -------
        str
            Base CSS rules.
        """
        return """
/* === BASE CSS === */
*, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html {
    font-size: 16px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

body {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
    color: var(--text-color, #333);
    background: var(--bg-color, #fff);
    font-family: var(--font-family, 'Segoe UI', Helvetica, Arial, sans-serif);
}

h1 {
    font-size: 2rem; margin-bottom: 1rem;
    border-bottom: 2px solid var(--accent-color, #2980b9);
    padding-bottom: 0.5rem;
}
h2 {
    font-size: 1.5rem; margin-top: 2rem;
    margin-bottom: 0.75rem;
    color: var(--heading-color, #2c3e50);
}
h3 {
    font-size: 1.2rem; margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}

p { margin-bottom: 1rem; }
a { color: var(--link-color, #2980b9); text-decoration: none; }
a:hover { text-decoration: underline; }

code {
    font-family: 'Fira Code', 'Consolas', monospace;
    background: var(--code-bg, #f8f9fa);
    padding: 0.15em 0.4em; border-radius: 3px;
    font-size: 0.9em;
}
pre {
    background: var(--code-bg, #f8f9fa);
    padding: 1rem; border-radius: 6px;
    overflow-x: auto; margin: 1rem 0;
}

/* Layout grid */
.report-container { display: flex; min-height: 100vh; }
.report-sidebar {
    width: 250px; position: sticky; top: 0;
    height: 100vh; overflow-y: auto; padding: 1rem;
    border-right: 1px solid var(--border-color, #e0e0e0);
    flex-shrink: 0;
}
.report-main { flex: 1; padding: 0 2rem; min-width: 0; }

.report-sidebar ul { list-style: none; }
.report-sidebar li { margin-bottom: 0.5rem; }
.report-sidebar a { color: var(--sidebar-text, #555); font-size: 0.9rem; }
.report-sidebar a:hover { color: var(--accent-color, #2980b9); }

@media (max-width: 768px) {
    .report-container { flex-direction: column; }
    .report-sidebar {
        width: 100%; height: auto; position: static;
        border-right: none;
        border-bottom: 1px solid var(--border-color, #e0e0e0);
    }
}

.report-header { text-align: center; margin-bottom: 2rem; }
.report-footer {
    margin-top: 3rem; padding-top: 1rem;
    border-top: 1px solid var(--border-color, #e0e0e0);
    font-size: 0.85rem; color: var(--muted-text, #888);
    text-align: center;
}
"""

    def _component_css(self) -> str:
        """Component CSS: tables, cards, sections, collapsible.

        Returns
        -------
        str
            Component CSS rules.
        """
        return """
/* === COMPONENT CSS === */

/* Tables */
.cb-table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.9rem;
}
.cb-table thead {
    background: var(--table-header-bg, #f0f0f0);
    border-bottom: 2px solid var(--table-border, #ccc);
}
.cb-table th {
    padding: 0.6rem 0.8rem;
    text-align: left;
    font-weight: 600;
    white-space: nowrap;
}
.cb-table td {
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid var(--table-border-light, #eee);
}
.cb-table tbody tr:hover {
    background: var(--table-hover, #f9f9f9);
}
.cb-table .num { text-align: right; font-family: 'Fira Code', monospace; }
.cb-table .sig { color: var(--sig-color, #c0392b); font-weight: 700; }

/* Cards */
.cb-card {
    border: 1px solid var(--border-color, #e0e0e0);
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
    background: var(--card-bg, #fff);
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.cb-card-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    color: var(--heading-color, #2c3e50);
}

/* Sections */
.cb-section {
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--section-border, #f0f0f0);
}

/* Collapsible */
.cb-collapsible {
    margin: 0.5rem 0;
}
.cb-collapsible-toggle {
    display: flex;
    align-items: center;
    cursor: pointer;
    padding: 0.5rem;
    background: var(--collapsible-bg, #f8f9fa);
    border: 1px solid var(--border-color, #e0e0e0);
    border-radius: 4px;
    font-weight: 600;
    user-select: none;
}
.cb-collapsible-toggle::before {
    content: '\\25B6';
    margin-right: 0.5rem;
    transition: transform 0.2s;
    font-size: 0.8em;
}
.cb-collapsible.open .cb-collapsible-toggle::before {
    transform: rotate(90deg);
}
.cb-collapsible-content {
    display: none;
    padding: 1rem;
    border: 1px solid var(--border-color, #e0e0e0);
    border-top: none;
    border-radius: 0 0 4px 4px;
}
.cb-collapsible.open .cb-collapsible-content {
    display: block;
}

/* Plots */
.cb-plot {
    text-align: center;
    margin: 1.5rem 0;
}
.cb-plot img {
    max-width: 100%;
    height: auto;
    border: 1px solid var(--border-color, #e0e0e0);
    border-radius: 4px;
}
.cb-plot-caption {
    font-size: 0.85rem;
    color: var(--muted-text, #888);
    margin-top: 0.5rem;
}

/* Info boxes */
.cb-info {
    padding: 1rem; margin: 1rem 0;
    border-left: 4px solid var(--info-color, #3498db);
    background: var(--info-bg, #ebf5fb);
    border-radius: 0 4px 4px 0;
}
.cb-warning {
    padding: 1rem; margin: 1rem 0;
    border-left: 4px solid var(--warning-color, #f39c12);
    background: var(--warning-bg, #fef9e7);
    border-radius: 0 4px 4px 0;
}
.cb-success {
    padding: 1rem; margin: 1rem 0;
    border-left: 4px solid var(--success-color, #27ae60);
    background: var(--success-bg, #eafaf1);
    border-radius: 0 4px 4px 0;
}

/* Metrics grid */
.cb-metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
.cb-metric {
    text-align: center;
    padding: 1rem;
    background: var(--metric-bg, #f8f9fa);
    border-radius: 6px;
}
.cb-metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent-color, #2980b9);
}
.cb-metric-label {
    font-size: 0.8rem;
    color: var(--muted-text, #888);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
"""

    def _theme_css(self, theme: str) -> str:
        """Theme-specific CSS variables.

        Parameters
        ----------
        theme : str
            Theme name.

        Returns
        -------
        str
            CSS custom properties for the theme.
        """
        themes = {
            "professional": """
/* === THEME: Professional === */
:root {
    --text-color: #2c3e50;
    --bg-color: #ffffff;
    --heading-color: #2c3e50;
    --accent-color: #2980b9;
    --link-color: #2980b9;
    --border-color: #ecf0f1;
    --table-header-bg: #f7f9fc;
    --table-border: #d5dbdb;
    --table-border-light: #f2f3f4;
    --table-hover: #fafbfc;
    --card-bg: #ffffff;
    --code-bg: #f7f9fc;
    --sidebar-text: #566573;
    --muted-text: #95a5a6;
    --sig-color: #c0392b;
    --font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
    --section-border: #ecf0f1;
    --collapsible-bg: #f7f9fc;
    --info-color: #2980b9;
    --info-bg: #ebf5fb;
    --warning-color: #f39c12;
    --warning-bg: #fef9e7;
    --success-color: #27ae60;
    --success-bg: #eafaf1;
    --metric-bg: #f7f9fc;
}
""",
            "academic": """
/* === THEME: Academic === */
:root {
    --text-color: #000000;
    --bg-color: #ffffff;
    --heading-color: #000000;
    --accent-color: #333333;
    --link-color: #333333;
    --border-color: #cccccc;
    --table-header-bg: #f0f0f0;
    --table-border: #999999;
    --table-border-light: #e0e0e0;
    --table-hover: #f5f5f5;
    --card-bg: #ffffff;
    --code-bg: #f5f5f5;
    --sidebar-text: #444444;
    --muted-text: #777777;
    --sig-color: #000000;
    --font-family: 'Times New Roman', 'Computer Modern', Georgia, serif;
    --section-border: #cccccc;
    --collapsible-bg: #f5f5f5;
    --info-color: #555555;
    --info-bg: #f5f5f5;
    --warning-color: #888888;
    --warning-bg: #f5f5f5;
    --success-color: #555555;
    --success-bg: #f5f5f5;
    --metric-bg: #f5f5f5;
}
""",
            "presentation": """
/* === THEME: Presentation === */
:root {
    --text-color: #2c3e50;
    --bg-color: #ffffff;
    --heading-color: #2c3e50;
    --accent-color: #e74c3c;
    --link-color: #3498db;
    --border-color: #eeeeee;
    --table-header-bg: #f0f0f0;
    --table-border: #dddddd;
    --table-border-light: #f5f5f5;
    --table-hover: #fafafa;
    --card-bg: #ffffff;
    --code-bg: #f8f8f8;
    --sidebar-text: #555555;
    --muted-text: #999999;
    --sig-color: #e74c3c;
    --font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    --section-border: #eeeeee;
    --collapsible-bg: #f8f8f8;
    --info-color: #3498db;
    --info-bg: #eaf2f8;
    --warning-color: #f1c40f;
    --warning-bg: #fef9e7;
    --success-color: #2ecc71;
    --success-bg: #eafaf1;
    --metric-bg: #f8f8f8;
}
body { font-size: 18px; }
h1 { font-size: 2.5rem; }
h2 { font-size: 1.8rem; }
""",
            "bcb": """
/* === THEME: BCB (Banco Central do Brasil) === */
:root {
    --text-color: #212121;
    --bg-color: #ffffff;
    --heading-color: #004d40;
    --accent-color: #00695c;
    --link-color: #00897b;
    --border-color: #e0e0e0;
    --table-header-bg: #e0f2f1;
    --table-border: #b2dfdb;
    --table-border-light: #e8f5e9;
    --table-hover: #f1f8e9;
    --card-bg: #ffffff;
    --code-bg: #f1f8e9;
    --sidebar-text: #37474f;
    --muted-text: #78909c;
    --sig-color: #b71c1c;
    --font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
    --section-border: #e0e0e0;
    --collapsible-bg: #e0f2f1;
    --info-color: #1565c0;
    --info-bg: #e3f2fd;
    --warning-color: #ff6f00;
    --warning-bg: #fff3e0;
    --success-color: #004d40;
    --success-bg: #e0f2f1;
    --metric-bg: #e0f2f1;
}
""",
        }

        return themes.get(theme, themes["professional"])
