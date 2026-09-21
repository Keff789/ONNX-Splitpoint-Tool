"""Noninteractive figures for file exports; leave the GUI backend untouched.

Retain the existing energy report's Matplotlib defaults, sizes and grid style.
No pyplot managers or global backend/style changes are used by these helpers.
"""
from __future__ import annotations


def export_subplots(*, figsize):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)
    return figure, figure.subplots()


def table_preview(headers, values, *, caption):
    """Render the same formatted cells as the booktabs export for visual review."""
    import textwrap

    from matplotlib.font_manager import FontProperties
    figure, axis = export_subplots(figsize=(8.2, max(2.0, .32 * (len(values) + 3))))
    renderer = figure.canvas.get_renderer()
    def text_width(text, *, bold=False):
        return renderer.get_text_width_height_descent(
            str(text), FontProperties(size=10, weight="bold" if bold else "normal"),
            ismath=str(text).startswith("$"))[0]
    widths = [max(text_width(header, bold=True),
                  *(text_width(row[i]) for row in values)) + 28
              for i, header in enumerate(headers)]
    figure.set_size_inches(max(8.2, sum(widths) / figure.dpi + .6), figure.get_figheight())
    axis.set_axis_off()
    axis.set_title(textwrap.fill(caption, 110), fontsize=10, pad=18)
    table = axis.table(cellText=values, colLabels=headers, loc="center",
                       cellLoc="right", colWidths=[w / sum(widths) for w in widths])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        cell.visible_edges = "TB" if row == 0 else "B" if row == len(values) else ""
        cell.set_linewidth(.8 if row in (0, len(values)) else 0)
        if row == 0:
            cell.set_text_props(weight="bold")
        if col == 0:
            cell.set_text_props(ha="left")
    figure.tight_layout()
    return figure
