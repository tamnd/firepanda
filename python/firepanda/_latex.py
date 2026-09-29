"""A frame as a LaTeX table, which is `DataFrame.to_latex` and `Series.to_latex`.

pandas builds the table with a `Styler` and two jinja templates, one for a
`tabular` inside an optional `table` float and one for a `longtable`. This
module writes the same lines directly. A float cell has six digits after the
point unless `float_format` or a formatter says otherwise, a gap is `na_rep`,
and `escape` swaps the characters LaTeX treats as special for the commands
that print them.
"""

from __future__ import annotations

import math
from typing import Any

from .errors import InvalidArgumentError

_BACKSLASH = "ab2§=§8yz"


def _escaped(text: str) -> str:
    """The text with LaTeX's special characters escaped, the way pandas does it."""
    return (
        text.replace("\\", _BACKSLASH)
        .replace(_BACKSLASH + " ", _BACKSLASH + "\\space ")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~ ", "~\\space ")
        .replace("~", "\\textasciitilde ")
        .replace("^ ", "^\\space ")
        .replace("^", "\\textasciicircum ")
        .replace(_BACKSLASH, "\\textbackslash ")
    )


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return type(value).__name__ in ("NaTType", "NAType")


def _is_float(value: Any) -> bool:
    return isinstance(value, float) or type(value).__name__.startswith("float")


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float, complex))


def _default(value: Any) -> str:
    if _is_float(value) or isinstance(value, complex):
        return f"{value:.6f}"
    return str(value)


def _cell(value: Any, formatter: Any, options: dict[str, Any]) -> str:
    """One value as the text the Styler would print for it."""
    if _missing(value):
        return options["na_rep"]
    if options["escape"] and isinstance(value, str):
        value = _escaped(value)
    text = formatter(value) if formatter is not None else _default(value)
    text = text if isinstance(text, str) else str(text)
    if options["decimal"] != "." and _is_number(value):
        text = text.replace(".", options["decimal"])
    return text


def _label(value: Any, escape: bool) -> str:
    text = "" if value is None else _default(value)
    return _escaped(text) if escape and isinstance(value, str) else text


def _float_format(float_format: Any) -> Any:
    if float_format is None or callable(float_format):
        return float_format
    return lambda value: float_format % value


def _formatters(labels: list[Any], kinds: list[bool], options: dict[str, Any]) -> list[Any]:
    """The formatter for each column, from `formatters` and `float_format` as pandas pairs them."""
    float_format = _float_format(options["float_format"])
    formatters = options["formatters"]

    def wrapped(formatter: Any) -> Any:
        if float_format is None:
            return formatter
        return lambda value: float_format(value) if _is_float(value) else formatter(value)

    if isinstance(formatters, (list, tuple)):
        chosen = [wrapped(formatter) for formatter in formatters]
    elif isinstance(formatters, dict):
        chosen = [wrapped(formatters[label]) if label in formatters else None for label in labels]
    else:
        chosen = [None] * len(labels)
    return [
        float_format if formatter is None and is_float else formatter
        for formatter, is_float in zip(chosen, kinds, strict=True)
    ]


def _row(cells: list[str]) -> str:
    return " & ".join(cells) + " \\\\"


def to_latex(frame: Any, options: dict[str, Any]) -> str:
    """The lines of the LaTeX table for the frame, joined and ending in a newline."""
    if options["columns"] is not None:
        frame = frame[list(options["columns"])]
    labels = list(frame.columns)
    header = options["header"]
    if not isinstance(header, bool):
        header = list(header)
        if len(header) != len(labels):
            raise InvalidArgumentError(f"Writing {len(labels)} cols but got {len(header)} aliases")
    dtypes = [str(frame[label].dtype) for label in labels]
    kinds = [dtype.startswith("float") for dtype in dtypes]
    numeric = [dtype.lower().startswith(("int", "uint", "float", "bool")) for dtype in dtypes]
    formatters = _formatters(labels, kinds, options)
    index, escape, bold = options["index"], options["escape"], options["bold_rows"]

    column_format = options["column_format"]
    if column_format is None:
        column_format = ("l" if index else "") + "".join("r" if n else "l" for n in numeric)

    head: list[str] = []
    if header is not False and labels:
        shown = header if isinstance(header, list) else labels
        names = [_label(label, escape and not isinstance(header, list)) for label in shown]
        corner = [_label(frame.columns.name, escape)] if index else []
        head.append(_row(corner + names))
    index_name = frame.index.name
    if index and options["index_names"] and index_name is not None:
        head.append(_row([_label(index_name, escape)] + [""] * len(labels)))

    columns = [frame[label].tolist() for label in labels]
    body = []
    for position, row_label in enumerate(frame.index.tolist()):
        cells = [
            _cell(column[position], formatter, options)
            for column, formatter in zip(columns, formatters, strict=True)
        ]
        if index:
            shown = _label(row_label, escape)
            cells.insert(0, f"\\textbf{{{shown}}}" if bold else shown)
        body.append(_row(cells))

    caption, label, position = options["caption"], options["label"], options["position"]
    if isinstance(caption, (tuple, list)):
        full, short = caption
        caption_text = f"\\caption[{short}]{{{full}}}"
    else:
        full = caption
        caption_text = f"\\caption{{{caption}}}" if caption is not None else None
    where = f"[{position}]" if position is not None else ""

    if options["longtable"]:
        lines = [f"\\begin{{longtable}}{where}{{{column_format}}}"]
        titled = [text for text in (caption_text, label and f"\\label{{{label}}}") if text]
        if titled:
            lines.append(" ".join(titled) + " \\\\")
        lines += ["\\toprule", *head, "\\midrule", "\\endfirsthead"]
        if full is not None:
            lines.append(f"\\caption[]{{{full}}} \\\\")
        width = len(labels) + (1 if index else 0)
        lines += ["\\toprule", *head, "\\midrule", "\\endhead", "\\midrule"]
        lines.append(f"\\multicolumn{{{width}}}{{r}}{{Continued on next page}} \\\\")
        lines += ["\\midrule", "\\endfoot", "\\bottomrule", "\\endlastfoot", *body]
        lines.append("\\end{longtable}")
        return "\n".join(lines) + "\n"

    table = caption_text is not None or label is not None or position is not None
    lines = [f"\\begin{{table}}{where}"] if table else []
    if caption_text is not None:
        lines.append(caption_text)
    if label is not None:
        lines.append(f"\\label{{{label}}}")
    lines += [f"\\begin{{tabular}}{{{column_format}}}", "\\toprule", *head, "\\midrule"]
    lines += [*body, "\\bottomrule", "\\end{tabular}"]
    if table:
        lines.append("\\end{table}")
    return "\n".join(lines) + "\n"
