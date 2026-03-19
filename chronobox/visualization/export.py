"""Figure export utilities for chronobox visualization.

Supports export to PNG, SVG, PDF (via matplotlib) and HTML (via plotly).

Usage:
    from chronobox.visualization.export import save_figure, figure_to_html

    fig = plot_diagnostics(results)
    save_figure(fig, 'diagnostics.png', fmt='png')
    save_figure(fig, 'diagnostics.svg', fmt='svg')
    save_figure(fig, 'diagnostics.pdf', fmt='pdf')

    html_str = figure_to_html(fig)
    save_figure(fig, 'diagnostics.html', fmt='html')
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure


def save_figure(
    fig: Figure | Any,
    path: str | Path,
    fmt: str | None = None,
    dpi: int | None = None,
    transparent: bool = False,
    bbox_inches: str = "tight",
    **kwargs: Any,
) -> Path:
    """Save a matplotlib figure to file in the specified format.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or plotly Figure
        The figure to save.
    path : str or Path
        Output file path.
    fmt : str or None
        Format: 'png', 'svg', 'pdf', 'html'. If None, inferred from extension.
    dpi : int or None
        Resolution in dots per inch. If None, uses theme default.
    transparent : bool
        Whether to use transparent background.
    bbox_inches : str
        Bounding box setting for matplotlib (default 'tight').
    **kwargs
        Additional keyword arguments passed to savefig or plotly write.

    Returns
    -------
    Path
        Path to the saved file.

    Raises
    ------
    ValueError
        If format is not supported or cannot be inferred.
    """
    path = Path(path)

    if fmt is None:
        fmt = path.suffix.lstrip(".").lower()

    if not fmt:
        msg = "Cannot infer format from path. Specify 'fmt' parameter."
        raise ValueError(msg)

    fmt = fmt.lower()
    supported = {"png", "svg", "pdf", "html"}
    if fmt not in supported:
        msg = f"Unsupported format '{fmt}'. Supported: {sorted(supported)}"
        raise ValueError(msg)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "html":
        html_content = figure_to_html(fig)
        path.write_text(html_content, encoding="utf-8")
    elif isinstance(fig, Figure):
        # Matplotlib figure
        save_kwargs: dict[str, Any] = {
            "format": fmt,
            "bbox_inches": bbox_inches,
            "transparent": transparent,
        }
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        save_kwargs.update(kwargs)
        fig.savefig(str(path), **save_kwargs)
    else:
        # Assume plotly figure
        try:
            if fmt == "html":
                fig.write_html(str(path), **kwargs)
            else:
                fig.write_image(str(path), format=fmt, **kwargs)
        except AttributeError:
            msg = f"Figure type {type(fig)} does not support export to {fmt}"
            raise TypeError(msg) from None

    return path


def figure_to_html(
    fig: Figure | Any,
    include_plotlyjs: bool = True,
    full_html: bool = True,
    div_id: str | None = None,
) -> str:
    """Convert a figure to an HTML string.

    For matplotlib figures, converts to a static image embedded in HTML.
    For plotly figures, generates interactive HTML.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or plotly Figure
        The figure to convert.
    include_plotlyjs : bool
        Whether to include plotly.js (for plotly figures).
    full_html : bool
        Whether to generate a full HTML page or just the div.
    div_id : str or None
        Custom div id for the figure element.

    Returns
    -------
    str
        HTML string with the figure.
    """
    if isinstance(fig, Figure):
        return _matplotlib_to_html(fig, full_html=full_html, div_id=div_id)
    else:
        # Assume plotly figure
        try:
            return fig.to_html(
                include_plotlyjs=include_plotlyjs,
                full_html=full_html,
                div_id=div_id,
            )
        except AttributeError:
            return _matplotlib_to_html(fig, full_html=full_html, div_id=div_id)


def figure_to_base64(fig: Figure, fmt: str = "png", dpi: int = 150) -> str:
    """Convert a matplotlib figure to a base64-encoded string.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to convert.
    fmt : str
        Image format ('png', 'svg').
    dpi : int
        Resolution.

    Returns
    -------
    str
        Base64-encoded image string.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return encoded


def _matplotlib_to_html(
    fig: Figure | Any,
    full_html: bool = True,
    div_id: str | None = None,
) -> str:
    """Convert matplotlib figure to HTML with embedded PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to convert.
    full_html : bool
        Whether to wrap in full HTML page.
    div_id : str or None
        Custom div ID.

    Returns
    -------
    str
        HTML string.
    """
    if not isinstance(fig, Figure):
        msg = f"Expected matplotlib Figure, got {type(fig)}"
        raise TypeError(msg)

    encoded = figure_to_base64(fig, fmt="png", dpi=150)
    div_id_attr = f' id="{div_id}"' if div_id else ""

    img_tag = (
        f'<div{div_id_attr} style="text-align: center;">'
        f'<img src="data:image/png;base64,{encoded}" '
        f'style="max-width: 100%; height: auto;" />'
        f"</div>"
    )

    if not full_html:
        return img_tag

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>chronobox figure</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; background: #fff; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
{img_tag}
</body>
</html>"""
    return html


def figures_to_html_gallery(
    figures: list[tuple[str, Figure]],
    title: str = "chronobox Figures",
) -> str:
    """Convert multiple figures to an HTML gallery page.

    Parameters
    ----------
    figures : list of (title, figure) tuples
        Figures to include.
    title : str
        Page title.

    Returns
    -------
    str
        Full HTML page with all figures.
    """
    sections = []
    for fig_title, fig in figures:
        encoded = figure_to_base64(fig, fmt="png", dpi=150)
        sections.append(
            f'<h2>{fig_title}</h2>\n'
            f'<div style="text-align: center; margin-bottom: 30px;">\n'
            f'  <img src="data:image/png;base64,{encoded}" '
            f'style="max-width: 100%; height: auto;" />\n'
            f'</div>'
        )

    body = "\n".join(sections)
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: sans-serif; margin: 40px; background: #fff; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ecf0f1; }}
    </style>
</head>
<body>
<h1>{title}</h1>
{body}
</body>
</html>"""
    return html
