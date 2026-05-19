"""
Report generator for complete workflow cross-validation.

Generates HTML comparison reports from pipeline results across
Python (chronobox), R, and Stata.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def generate_comparison_report(
    title: str,
    python_results: dict[str, Any],
    r_results: dict[str, Any] | None = None,
    stata_results: dict[str, Any] | None = None,
    comparison_df: pd.DataFrame | None = None,
    output_path: str | Path = "outputs/comparison_report.html",
    extra_sections: list[dict[str, str]] | None = None,
) -> Path:
    """Generate an HTML comparison report.

    Parameters
    ----------
    title : str
        Report title.
    python_results : dict
        Results from chronobox.
    r_results : dict, optional
        Results from R.
    stata_results : dict, optional
        Results from Stata.
    comparison_df : pd.DataFrame, optional
        Output of ``cross_validation_report()``.  If *None*, a simple
        side-by-side table is built from the result dicts.
    output_path : str or Path
        Where to write the HTML file.
    extra_sections : list[dict], optional
        Additional ``{"title": ..., "html": ...}`` sections to append.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build comparison table if not provided
    if comparison_df is None:
        comparison_df = _build_simple_comparison(
            python_results, r_results, stata_results
        )

    html = _render_html(
        title=title,
        comparison_df=comparison_df,
        python_results=python_results,
        r_results=r_results,
        stata_results=stata_results,
        extra_sections=extra_sections or [],
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_simple_comparison(
    py: dict, r: dict | None, stata: dict | None
) -> pd.DataFrame:
    metrics = sorted(set(
        list(py.keys())
        + list((r or {}).keys())
        + list((stata or {}).keys())
    ))
    rows = []
    for m in metrics:
        rows.append({
            "metric": m,
            "python": py.get(m),
            "r": (r or {}).get(m),
            "stata": (stata or {}).get(m),
        })
    return pd.DataFrame(rows)


def _df_to_html_table(df: pd.DataFrame) -> str:
    rows_html = []
    # header
    cols = list(df.columns)
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows_html.append(f"<tr>{header}</tr>")
    # body
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                if np.isnan(val):
                    cells.append('<td class="na">—</td>')
                else:
                    cells.append(f"<td>{val:.6f}</td>")
            elif val is None:
                cells.append('<td class="na">—</td>')
            elif isinstance(val, bool):
                cls = "match" if val else "mismatch"
                cells.append(f'<td class="{cls}">{"✓" if val else "✗"}</td>')
            else:
                cells.append(f"<td>{val}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table>{''.join(rows_html)}</table>"


def _result_summary_html(results: dict[str, Any], label: str) -> str:
    if not results:
        return f"<p>No {label} results provided.</p>"
    lines = [f"<h3>{label} Results</h3>", "<ul>"]
    for k, v in results.items():
        if isinstance(v, dict):
            v_str = json.dumps(v, default=_json_default, indent=2)
            lines.append(f"<li><strong>{k}</strong>: <pre>{v_str}</pre></li>")
        elif isinstance(v, (list, np.ndarray)):
            arr = np.asarray(v)
            lines.append(
                f"<li><strong>{k}</strong>: array of length {len(arr)}</li>"
            )
        else:
            lines.append(f"<li><strong>{k}</strong>: {v}</li>")
    lines.append("</ul>")
    return "\n".join(lines)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _render_html(
    title: str,
    comparison_df: pd.DataFrame,
    python_results: dict,
    r_results: dict | None,
    stata_results: dict | None,
    extra_sections: list[dict],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    comparison_table = _df_to_html_table(comparison_df)

    extra_html = ""
    for sec in extra_sections:
        extra_html += f"<h2>{sec['title']}</h2>\n{sec['html']}\n"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 960px;
    margin: 2rem auto;
    padding: 0 1rem;
    color: #333;
    line-height: 1.6;
  }}
  h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 0.3rem; }}
  h2 {{ color: #2e86c1; margin-top: 2rem; }}
  h3 {{ color: #555; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1rem 0;
    font-size: 0.9rem;
  }}
  th, td {{
    border: 1px solid #ddd;
    padding: 0.5rem 0.8rem;
    text-align: right;
  }}
  th {{
    background: #2e86c1;
    color: white;
    text-align: center;
  }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  tr:hover {{ background: #eaf2f8; }}
  td:first-child {{ text-align: left; font-weight: bold; }}
  .match {{ color: #27ae60; font-weight: bold; }}
  .mismatch {{ color: #e74c3c; font-weight: bold; }}
  .na {{ color: #aaa; text-align: center; }}
  pre {{
    background: #f4f4f4;
    padding: 0.8rem;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 0.85rem;
  }}
  .footer {{
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #ddd;
    font-size: 0.8rem;
    color: #888;
  }}
</style>
</head>
<body>

<h1>{title}</h1>
<p>Generated: {now}</p>

<h2>Cross-Validation Comparison</h2>
{comparison_table}

<h2>Detailed Results</h2>
{_result_summary_html(python_results, "Python (chronobox)")}
{_result_summary_html(r_results or {{}}, "R")}
{_result_summary_html(stata_results or {{}}, "Stata")}

{extra_html}

<div class="footer">
  <p>Report generated by <strong>chronobox</strong> complete workflow utilities.</p>
</div>

</body>
</html>
"""
