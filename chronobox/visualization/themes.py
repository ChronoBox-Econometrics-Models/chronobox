"""Theme system for chronobox visualization.

Four built-in themes:
    - professional: Blue/gray, clean lines, Helvetica - reports and publications
    - academic: Black/white, minimal, Computer Modern/Times - papers and theses
    - presentation: Vibrant colors, high contrast, large fonts - slides
    - bcb: Green/blue institutional, Banco Central do Brasil style

Usage:
    from chronobox.visualization.themes import set_theme, get_theme

    set_theme('professional')
    theme = get_theme()  # returns current ThemeConfig
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib as mpl


@dataclass
class ThemeConfig:
    """Configuration for a visualization theme.

    Attributes
    ----------
    name : str
        Theme name.
    colors : list[str]
        Ordered color palette (hex codes).
    background : str
        Background color.
    grid_color : str
        Grid line color.
    text_color : str
        Primary text color.
    font_family : str
        Font family name.
    font_size : int
        Base font size in points.
    title_size : int
        Title font size in points.
    label_size : int
        Axis label font size in points.
    tick_size : int
        Tick label font size in points.
    line_width : float
        Default line width.
    grid_alpha : float
        Grid transparency (0-1).
    grid_style : str
        Grid line style.
    spine_visible : bool
        Whether to show axis spines.
    legend_frame : bool
        Whether to frame legends.
    figure_dpi : int
        Default figure DPI.
    confidence_colors : list[str]
        Colors for confidence intervals (dark to light).
    positive_color : str
        Color for positive values / increases.
    negative_color : str
        Color for negative values / decreases.
    highlight_color : str
        Color for highlighting important elements.
    """

    name: str = "professional"
    colors: list[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ])
    background: str = "#ffffff"
    grid_color: str = "#e0e0e0"
    text_color: str = "#333333"
    font_family: str = "sans-serif"
    font_size: int = 11
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    line_width: float = 1.5
    grid_alpha: float = 0.7
    grid_style: str = "-"
    spine_visible: bool = True
    legend_frame: bool = False
    figure_dpi: int = 150
    confidence_colors: list[str] = field(default_factory=lambda: [
        "#1a5276", "#2980b9", "#5dade2", "#aed6f1",
    ])
    positive_color: str = "#27ae60"
    negative_color: str = "#c0392b"
    highlight_color: str = "#e74c3c"


# --- Built-in theme definitions ---

_PROFESSIONAL = ThemeConfig(
    name="professional",
    colors=[
        "#2c3e50", "#2980b9", "#7f8c8d", "#34495e",
        "#1abc9c", "#e74c3c", "#f39c12", "#9b59b6",
    ],
    background="#ffffff",
    grid_color="#ecf0f1",
    text_color="#2c3e50",
    font_family="sans-serif",
    font_size=11,
    title_size=14,
    label_size=12,
    tick_size=10,
    line_width=1.5,
    grid_alpha=0.7,
    grid_style="-",
    spine_visible=True,
    legend_frame=False,
    figure_dpi=150,
    confidence_colors=["#1a5276", "#2980b9", "#5dade2", "#aed6f1"],
    positive_color="#27ae60",
    negative_color="#c0392b",
    highlight_color="#e74c3c",
)

_ACADEMIC = ThemeConfig(
    name="academic",
    colors=[
        "#000000", "#555555", "#888888", "#aaaaaa",
        "#333333", "#666666", "#999999", "#cccccc",
    ],
    background="#ffffff",
    grid_color="#d0d0d0",
    text_color="#000000",
    font_family="serif",
    font_size=10,
    title_size=12,
    label_size=11,
    tick_size=9,
    line_width=1.0,
    grid_alpha=0.5,
    grid_style="--",
    spine_visible=True,
    legend_frame=True,
    figure_dpi=300,
    confidence_colors=["#333333", "#666666", "#999999", "#cccccc"],
    positive_color="#000000",
    negative_color="#555555",
    highlight_color="#000000",
)

_PRESENTATION = ThemeConfig(
    name="presentation",
    colors=[
        "#e74c3c", "#3498db", "#2ecc71", "#f1c40f",
        "#9b59b6", "#e67e22", "#1abc9c", "#34495e",
    ],
    background="#ffffff",
    grid_color="#eeeeee",
    text_color="#2c3e50",
    font_family="sans-serif",
    font_size=14,
    title_size=20,
    label_size=16,
    tick_size=13,
    line_width=2.5,
    grid_alpha=0.4,
    grid_style="-",
    spine_visible=False,
    legend_frame=False,
    figure_dpi=150,
    confidence_colors=["#c0392b", "#e74c3c", "#f1948a", "#fadbd8"],
    positive_color="#2ecc71",
    negative_color="#e74c3c",
    highlight_color="#f1c40f",
)

_BCB = ThemeConfig(
    name="bcb",
    colors=[
        "#004d40", "#00695c", "#00897b", "#26a69a",
        "#1565c0", "#1976d2", "#42a5f5", "#90caf9",
    ],
    background="#ffffff",
    grid_color="#e0e0e0",
    text_color="#212121",
    font_family="sans-serif",
    font_size=11,
    title_size=14,
    label_size=12,
    tick_size=10,
    line_width=1.5,
    grid_alpha=0.6,
    grid_style="-",
    spine_visible=True,
    legend_frame=False,
    figure_dpi=150,
    confidence_colors=["#004d40", "#00897b", "#4db6ac", "#b2dfdb"],
    positive_color="#004d40",
    negative_color="#b71c1c",
    highlight_color="#ff6f00",
)

_THEMES: dict[str, ThemeConfig] = {
    "professional": _PROFESSIONAL,
    "academic": _ACADEMIC,
    "presentation": _PRESENTATION,
    "bcb": _BCB,
}

# Current active theme (module-level state)
_current_theme: ThemeConfig = _PROFESSIONAL


def set_theme(name: str) -> None:
    """Set the active visualization theme.

    Parameters
    ----------
    name : str
        Theme name. One of: 'professional', 'academic', 'presentation', 'bcb'.

    Raises
    ------
    ValueError
        If theme name is not recognized.
    """
    global _current_theme
    if name not in _THEMES:
        available = ", ".join(sorted(_THEMES.keys()))
        msg = f"Unknown theme '{name}'. Available: {available}"
        raise ValueError(msg)
    _current_theme = _THEMES[name]
    _apply_matplotlib_theme(_current_theme)


def get_theme() -> ThemeConfig:
    """Get the current active theme configuration.

    Returns
    -------
    ThemeConfig
        Current theme.
    """
    return _current_theme


def list_themes() -> list[str]:
    """List available theme names.

    Returns
    -------
    list[str]
        Sorted list of theme names.
    """
    return sorted(_THEMES.keys())


def register_theme(name: str, theme: ThemeConfig) -> None:
    """Register a custom theme.

    Parameters
    ----------
    name : str
        Theme name.
    theme : ThemeConfig
        Theme configuration.
    """
    _THEMES[name] = theme


def _apply_matplotlib_theme(theme: ThemeConfig) -> None:
    """Apply a theme to matplotlib's rcParams.

    Parameters
    ----------
    theme : ThemeConfig
        Theme to apply.
    """
    mpl.rcParams.update({
        "figure.facecolor": theme.background,
        "axes.facecolor": theme.background,
        "axes.edgecolor": theme.text_color,
        "axes.labelcolor": theme.text_color,
        "axes.grid": True,
        "grid.color": theme.grid_color,
        "grid.alpha": theme.grid_alpha,
        "grid.linestyle": theme.grid_style,
        "axes.spines.top": theme.spine_visible,
        "axes.spines.right": theme.spine_visible,
        "text.color": theme.text_color,
        "xtick.color": theme.text_color,
        "ytick.color": theme.text_color,
        "font.family": theme.font_family,
        "font.size": theme.font_size,
        "axes.titlesize": theme.title_size,
        "axes.labelsize": theme.label_size,
        "xtick.labelsize": theme.tick_size,
        "ytick.labelsize": theme.tick_size,
        "lines.linewidth": theme.line_width,
        "legend.frameon": theme.legend_frame,
        "figure.dpi": theme.figure_dpi,
        "axes.prop_cycle": mpl.cycler(color=theme.colors),
    })


def get_color_cycle(theme: ThemeConfig | None = None) -> list[str]:
    """Get the color cycle for the given or current theme.

    Parameters
    ----------
    theme : ThemeConfig or None
        Theme to use. If None, uses current theme.

    Returns
    -------
    list[str]
        Color palette.
    """
    if theme is None:
        theme = _current_theme
    return theme.colors


def get_confidence_colors(theme: ThemeConfig | None = None) -> list[str]:
    """Get confidence interval colors (dark to light).

    Parameters
    ----------
    theme : ThemeConfig or None
        Theme to use. If None, uses current theme.

    Returns
    -------
    list[str]
        Confidence interval colors from darkest (tightest) to lightest (widest).
    """
    if theme is None:
        theme = _current_theme
    return theme.confidence_colors
