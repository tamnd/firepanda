"""A frame as an HTML table, which is `DataFrame.to_html` and the notebook repr.

A port of pandas' `HTMLFormatter` and `NotebookFormatter`. The cells are the
text `to_string` writes, trimmed, so a float column still shares one number of
digits and a gap is still `NaN`. What this module adds is the markup around
them: the header row, the row labels as `th` cells, a row of dots where rows
were cut and a cell of dots where columns were, and the line under the table
that gives the size.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse, uses_netloc, uses_params, uses_relative

_JUSTIFY = (
    "left",
    "right",
    "center",
    "justify",
    "justify-all",
    "start",
    "end",
    "inherit",
    "match-parent",
    "initial",
    "unset",
)

_SCHEMES = set(uses_relative + uses_netloc + uses_params) - {""}

_STYLE = """<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>"""


def _spaces(labels: list[str], col_space: Any) -> dict[Any, Any]:
    """`col_space` as a width for each label, checked as pandas checks it."""
    if col_space is None:
        return {}
    if isinstance(col_space, (int, str)):
        return {"": col_space, **dict.fromkeys(labels, col_space)}
    if isinstance(col_space, dict):
        for label in col_space:
            if label not in labels and label != "":
                raise ValueError(f"Col_space is defined for an unknown column: {label}")
        return dict(col_space)
    if len(col_space) != len(labels):
        raise ValueError(
            f"Col_space length({len(col_space)}) should match DataFrame number of "
            f"columns({len(labels)})"
        )
    return dict(zip(labels, col_space, strict=True))


class _Writer:
    """Collects the lines of the table, each cell escaped and trimmed as pandas does it."""

    def __init__(self, escape: bool, links: bool, spaces: dict[Any, str], bold: bool) -> None:
        self.lines: list[str] = []
        self.escape = escape
        self.links = links
        self.spaces = spaces
        self.bold = bold

    def write(self, text: str, indent: int = 0) -> None:
        self.lines.append(" " * indent + text)

    def cell(self, value: Any, kind: str, indent: int, header: bool = False) -> None:
        text = value if isinstance(value, str) else str(value)
        start = f"<{kind}>"
        space = self.spaces.get(text) if header else None
        if space is not None:
            start = f'<{kind} style="min-width: {space};">'
        shown = text
        if self.escape:
            shown = shown.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        shown = shown.strip().replace("  ", "&nbsp;&nbsp;")
        end = ""
        if self.links and urlparse(shown).scheme in _SCHEMES:
            start += f'<a href="{text.strip()}" target="_blank">'
            end = "</a>"
        self.write(f"{start}{shown}{end}</{kind}>", indent)

    def row(
        self,
        cells: list[Any],
        indent: int,
        header: bool = False,
        align: str | None = None,
        labels: int = 0,
    ) -> None:
        self.write("<tr>" if align is None else f'<tr style="text-align: {align};">', indent)
        for position, value in enumerate(cells):
            if header or (self.bold and position < labels):
                self.cell(value, "th", indent + 2, header)
            else:
                self.cell(value, "td", indent + 2)
        self.write("</tr>", indent)


def to_html(frame: Any, kw: dict[str, Any], notebook: bool) -> str:
    """The frame as pandas' HTML table, under the options in `kw`."""
    from ._config import get_option
    from ._pandas import _text_cut, _text_float_format, _text_labels, _text_limits, _text_values

    justify = kw["justify"]
    if justify is not None and justify not in _JUSTIFY:
        raise ValueError("Invalid value for justify parameter")
    if justify is None:
        justify = get_option("display.colheader_justify")
    if kw["columns"] is not None:
        frame = frame[list(kw["columns"])]
    labels = list(frame.columns)
    formatters = kw["formatters"]
    if formatters is None:
        formatters = {}
    elif not isinstance(formatters, dict) and len(formatters) != len(labels):
        raise ValueError(
            f"Formatters length({len(formatters)}) should match DataFrame number of "
            f"columns({len(labels)})"
        )
    spaces = {
        label: f"{value}px" if isinstance(value, int) else value
        for label, value in _spaces(labels, kw["col_space"]).items()
    }
    classes = ["dataframe"]
    if not get_option("display.html.use_mathjax"):
        classes += ["tex2jax_ignore", "mathjax_ignore"]
    extra = kw["classes"]
    if extra is not None:
        if isinstance(extra, str):
            extra = extra.split()
        if not isinstance(extra, (list, tuple)):
            raise TypeError(f"classes must be a string, list, or tuple, not {type(extra)}")
        classes.extend(extra)
    border = kw["border"]
    if border is None or border is True:
        border = get_option("display.html.border")
    elif not border:
        border = None

    rows = len(frame)
    header = kw["header"]
    index = kw["index"]
    fitted_rows, fitted_cols = _text_limits(rows, kw, len(labels), bool(header))
    kept_rows, dots_row = _text_cut(rows, fitted_rows, kw["max_rows"])
    kept_cols, dots_col = _text_cut(len(labels), fitted_cols, kw["max_cols"])
    shown = frame
    if dots_row is not None:
        shown = shown.iloc[kept_rows]
    if dots_col is not None:
        shown = shown.iloc[:, kept_cols]
    shown_labels = list(shown.columns)

    names_shown = kw["index_names"]
    column_name = getattr(frame.columns, "name", None)
    row_names = bool(index and names_shown and frame.index.name is not None)
    column_names = bool(column_name is not None and names_shown and header)
    levels = 1 if index or column_names else 0

    writer = _Writer(kw["escape"], kw["render_links"], spaces, kw["bold_rows"])
    table_id = "" if kw["table_id"] is None else f' id="{kw["table_id"]}"'
    border_text = "" if border is None else f' border="{border}"'
    if notebook:
        writer.write("<div>")
        writer.write(_STYLE)
    writer.write(f'<table{border_text} class="{" ".join(classes)}"{table_id}>')
    if header or row_names:
        writer.write("<thead>", 2)
        if header:
            cells: list[Any] = [""] * (levels - 1)
            if index or column_names:
                cells.append((column_name or "") if names_shown else "")
            cells.extend(shown_labels)
            if dots_col is not None:
                cells.insert(levels + dots_col, "...")
            writer.row(cells, 4, header=True, align=justify)
        if row_names:
            cells = [frame.index.name] + [""] * (len(shown_labels) + (dots_col is not None))
            writer.row(cells, 4, header=True)
        writer.write("</thead>", 2)

    writer.write("<tbody>", 2)
    float_format = _text_float_format(kw["float_format"])
    widest = get_option("display.max_colwidth") if notebook else None
    columns = []
    for position, label in enumerate(shown_labels):
        picked = formatters.get(label) if isinstance(formatters, dict) else formatters[position]
        texts = _text_values(
            shown.iloc[:, position], picked, float_format, kw["na_rep"], kw["decimal"], index
        )
        if widest is not None and widest > 3:
            texts = [x[: widest - 3] + "..." if len(x) > widest else x for x in texts]
        columns.append(texts)
    if index:
        mapped = formatters.get("__index__") if isinstance(formatters, dict) else None
        if mapped is not None:
            names = [mapped(label) for label in shown.index.tolist()]
        else:
            names = _text_labels(shown.index, False, None)
    cells = []
    for place in range(len(shown)):
        if dots_row is not None and place == dots_row:
            writer.row(["..."] * len(cells), 4, labels=levels)
        cells = []
        if index:
            cells.append(names[place])
        elif column_names:
            cells.append("")
        cells.extend(texts[place] for texts in columns)
        if dots_col is not None:
            cells.insert(dots_col + levels, "...")
        writer.row(cells, 4, labels=levels)
    writer.write("</tbody>", 2)
    writer.write("</table>")
    dimensions = kw["show_dimensions"]
    if dimensions is True or (dimensions == "truncate" and (dots_row, dots_col) != (None, None)):
        writer.write(f"<p>{rows} rows \N{MULTIPLICATION SIGN} {len(labels)} columns</p>")
    if notebook:
        writer.write("</div>")
    return "\n".join(writer.lines)
