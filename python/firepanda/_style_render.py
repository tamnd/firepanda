"""The renderer under `DataFrame.style`, pandas' `pandas.io.formats.style_render`.

A Styler turns a frame into HTML, LaTeX, Typst or plain text through jinja2
templates. This half of it builds the dictionary the templates read: the header
rows, the body rows, the spans a sparsified MultiIndex label covers, the cells
a trim leaves out, and the CSS each cell was given. It also keeps the display
function for every cell and label, which is what `format`, `format_index` and
`relabel_index` change. This module is pandas' own code, ported function for
function so the markup is the same text, with pandas' internals swapped for the
few lines of plain Python each one amounts to.

jinja2 is imported when this module is, as pandas imports it, so a missing
jinja2 surfaces as pandas' ImportError and `DataFrame.style` checks for it
first. The module is only imported when a Styler is made.
"""

from __future__ import annotations

import itertools
import math
import pathlib
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, TypedDict
from uuid import uuid4

from . import DataFrame, Index, IndexSlice, MultiIndex, Series
from ._config import get_option
from ._optional import imported

jinja2 = imported("jinja2", extra="DataFrame.style requires jinja2.")
from markupsafe import escape as escape_html  # noqa: E402  (markupsafe comes with jinja2)

try:
    import numpy as np
except ImportError:
    np = None

type BaseFormatter = str | Callable
type ExtFormatter = BaseFormatter | dict[Any, BaseFormatter | None]
type CSSPair = tuple[str, str | float]
type CSSList = list[CSSPair]
type CSSProperties = str | CSSList


class CSSDict(TypedDict):
    selector: str
    props: CSSProperties


type CSSStyles = list[CSSDict]
Subset = slice | Sequence | Index


class _NoDefault:
    """The mark a sparsified label leaves, pandas' `lib.no_default`."""

    def __repr__(self) -> str:
        return "<no_default>"


no_default = _NoDefault()


def _missing(x: Any) -> bool:
    """Whether a scalar is one of the spellings of missing."""
    if x is None:
        return True
    if isinstance(x, float):
        return math.isnan(x)
    try:
        from ._na import NA

        if x is NA:
            return True
    except ImportError:
        pass
    if np is not None and isinstance(x, np.floating):
        return bool(np.isnan(x))
    if np is not None and isinstance(x, np.datetime64 | np.timedelta64):
        return bool(np.isnat(x))
    return type(x).__name__ in {"NaTType", "NAType"}


def _is_float(x: Any) -> bool:
    """pandas' `is_float`: a Python or numpy float, never a bool."""
    return isinstance(x, float) or (np is not None and isinstance(x, np.floating))


def _is_integer(x: Any) -> bool:
    """pandas' `is_integer`: a Python or numpy integer, never a bool."""
    if isinstance(x, bool) or (np is not None and isinstance(x, np.bool_)):
        return False
    return isinstance(x, int) or (np is not None and isinstance(x, np.integer))


def _is_complex(x: Any) -> bool:
    """pandas' `is_complex`: a Python or numpy complex number."""
    return isinstance(x, complex) or (np is not None and isinstance(x, np.complexfloating))


def _is_list_like(x: Any) -> bool:
    """pandas' `is_list_like`: iterable, not text, not a class and not a zero dimension array."""
    if isinstance(x, str | bytes | type):
        return False
    if np is not None and isinstance(x, np.ndarray):
        return x.ndim > 0
    return hasattr(x, "__iter__")


def _axis_number(axis: Any) -> int:
    """A frame's axis out of any of its spellings, as `DataFrame._get_axis_number` reads it."""
    if axis in (0, "index", "rows") and not isinstance(axis, bool):
        return 0
    if axis in (1, "columns") and not isinstance(axis, bool):
        return 1
    raise ValueError(f"No axis named {axis} for object type DataFrame")


def _level_number(obj: Any, level: Any) -> int:
    """A level's position by name or number, as `Index._get_level_number` reads it."""
    if isinstance(obj, MultiIndex):
        return obj._level_number(level)
    if isinstance(level, int) and not isinstance(level, bool):
        if level < 0 and level != -1:
            raise IndexError(
                f"Too many levels: Index has only 1 level, {level} is not a valid level number"
            )
        if level > 0:
            raise IndexError(f"Too many levels: Index has only 1 level, not {level + 1}")
        return 0
    if level != obj.name:
        raise KeyError(f"Requested level ({level}) does not match index name ({obj.name})")
    return 0


class StylerRenderer:
    """The rendering half of a Styler: its state, its display functions and its templates."""

    this_dir = pathlib.Path(__file__).parent.resolve()
    template_dir = this_dir / "io" / "formats" / "templates"
    loader = jinja2.FileSystemLoader(template_dir)
    env = jinja2.Environment(loader=loader, trim_blocks=True)
    template_html = env.get_template("html.tpl")
    template_html_table = env.get_template("html_table.tpl")
    template_html_style = env.get_template("html_style.tpl")
    template_latex = env.get_template("latex.tpl")
    template_typst = env.get_template("typst.tpl")
    template_string = env.get_template("string.tpl")

    def __init__(
        self,
        data: DataFrame | Series,
        uuid: str | None = None,
        uuid_len: int = 5,
        table_styles: CSSStyles | None = None,
        table_attributes: str | None = None,
        caption: str | tuple | list | None = None,
        cell_ids: bool = True,
        precision: int | None = None,
    ) -> None:
        if isinstance(data, Series):
            data = data.to_frame()
        if not isinstance(data, DataFrame):
            raise TypeError("``data`` must be a Series or DataFrame")
        self.data: DataFrame = data
        self.index: Index = data.index
        self.columns: Index = data.columns
        if not isinstance(uuid_len, int) or uuid_len < 0:
            raise TypeError("``uuid_len`` must be an integer in range [0, 32].")
        self.uuid = uuid or uuid4().hex[: min(32, uuid_len)]
        self.uuid_len = len(self.uuid)
        self.table_styles = table_styles
        self.table_attributes = table_attributes
        self.caption = caption
        self.cell_ids = cell_ids
        self.css = {
            "row_heading": "row_heading",
            "col_heading": "col_heading",
            "index_name": "index_name",
            "col": "col",
            "row": "row",
            "col_trim": "col_trim",
            "row_trim": "row_trim",
            "level": "level",
            "data": "data",
            "blank": "blank",
            "foot": "foot",
        }
        self.concatenated: list[StylerRenderer] = []
        self.hide_index_names: bool = False
        self.hide_column_names: bool = False
        self.hide_index_: list = [False] * self.index.nlevels
        self.hide_columns_: list = [False] * self.columns.nlevels
        self.hidden_rows: Sequence[int] = []
        self.hidden_columns: Sequence[int] = []
        self.ctx: defaultdict[tuple[int, int], CSSList] = defaultdict(list)
        self.ctx_index: defaultdict[tuple[int, int], CSSList] = defaultdict(list)
        self.ctx_columns: defaultdict[tuple[int, int], CSSList] = defaultdict(list)
        self.cell_context: defaultdict[tuple[int, int], str] = defaultdict(str)
        self._todo: list[tuple[Callable, tuple, dict]] = []
        self.tooltips: Tooltips | None = None
        precision = get_option("styler.format.precision") if precision is None else precision

        def default() -> Callable[[Any], Any]:
            return partial(_default_formatter, precision=precision)

        self._display_funcs: defaultdict[tuple[int, int], Callable[[Any], str]] = defaultdict(
            default
        )
        self._display_funcs_index: defaultdict[tuple[int, int], Callable[[Any], str]] = defaultdict(
            default
        )
        self._display_funcs_index_names: defaultdict[int, Callable[[Any], str]] = defaultdict(
            default
        )
        self._display_funcs_columns: defaultdict[tuple[int, int], Callable[[Any], str]] = (
            defaultdict(default)
        )
        self._display_funcs_column_names: defaultdict[int, Callable[[Any], str]] = defaultdict(
            default
        )

    def _render(
        self,
        sparse_index: bool,
        sparse_columns: bool,
        max_rows: int | None = None,
        max_cols: int | None = None,
        blank: str = "",
    ) -> dict:
        self._compute()
        dxs = []
        ctx_len = len(self.index)
        for i, concatenated in enumerate(self.concatenated):
            concatenated.hide_index_ = self.hide_index_
            concatenated.hidden_columns = self.hidden_columns
            foot = f"{self.css['foot']}{i}"
            concatenated.css = {
                **self.css,
                "data": f"{foot}_data",
                "row_heading": f"{foot}_row_heading",
                "row": f"{foot}_row",
                "foot": f"{foot}_foot",
            }
            dx = concatenated._render(sparse_index, sparse_columns, max_rows, max_cols, blank)
            dxs.append(dx)

            for (r, c), v in concatenated.ctx.items():
                self.ctx[(r + ctx_len, c)] = v
            for (r, c), v in concatenated.ctx_index.items():
                self.ctx_index[(r + ctx_len, c)] = v

            ctx_len += len(concatenated.index)

        return self._translate(sparse_index, sparse_columns, max_rows, max_cols, blank, dxs)

    def _render_html(
        self,
        sparse_index: bool,
        sparse_columns: bool,
        max_rows: int | None = None,
        max_cols: int | None = None,
        **kwargs: Any,
    ) -> str:
        d = self._render(sparse_index, sparse_columns, max_rows, max_cols, "&nbsp;")
        d.update(kwargs)
        return self.template_html.render(
            **d,
            html_table_tpl=self.template_html_table,
            html_style_tpl=self.template_html_style,
        )

    def _render_latex(
        self, sparse_index: bool, sparse_columns: bool, clines: str | None, **kwargs: Any
    ) -> str:
        d = self._render(sparse_index, sparse_columns, None, None)
        self._translate_latex(d, clines=clines)
        self.template_latex.globals["parse_wrap"] = _parse_latex_table_wrapping
        self.template_latex.globals["parse_table"] = _parse_latex_table_styles
        self.template_latex.globals["parse_cell"] = _parse_latex_cell_styles
        self.template_latex.globals["parse_header"] = _parse_latex_header_span
        d.update(kwargs)
        return self.template_latex.render(**d)

    def _render_typst(
        self,
        sparse_index: bool,
        sparse_columns: bool,
        max_rows: int | None = None,
        max_cols: int | None = None,
        **kwargs: Any,
    ) -> str:
        d = self._render(sparse_index, sparse_columns, max_rows, max_cols)
        d.update(kwargs)
        return self.template_typst.render(**d)

    def _render_string(
        self,
        sparse_index: bool,
        sparse_columns: bool,
        max_rows: int | None = None,
        max_cols: int | None = None,
        **kwargs: Any,
    ) -> str:
        d = self._render(sparse_index, sparse_columns, max_rows, max_cols)
        d.update(kwargs)
        return self.template_string.render(**d)

    def _compute(self) -> Any:
        self.ctx.clear()
        self.ctx_index.clear()
        self.ctx_columns.clear()
        r = self
        for func, args, kwargs in self._todo:
            r = func(self)(*args, **kwargs)
        return r

    def _translate(
        self,
        sparse_index: bool,
        sparse_cols: bool,
        max_rows: int | None = None,
        max_cols: int | None = None,
        blank: str = "&nbsp;",
        dxs: list[dict] | None = None,
    ) -> dict:
        if dxs is None:
            dxs = []
        self.css["blank_value"] = blank

        d: dict[str, Any] = {
            "uuid": self.uuid,
            "table_styles": format_table_styles(self.table_styles or []),
            "caption": self.caption,
        }

        max_elements = get_option("styler.render.max_elements")
        max_rows = max_rows if max_rows else get_option("styler.render.max_rows")
        max_cols = max_cols if max_cols else get_option("styler.render.max_columns")
        max_rows, max_cols = _get_trimming_maximums(
            len(self.data.index),
            len(self.data.columns),
            max_elements,
            max_rows,
            max_cols,
        )

        self.cellstyle_map_columns: defaultdict[tuple[CSSPair, ...], list[str]] = defaultdict(list)
        head = self._translate_header(sparse_cols, max_cols)
        d.update({"head": head})

        idx_lengths = _get_level_lengths(self.index, sparse_index, max_rows, self.hidden_rows)
        d.update({"index_lengths": idx_lengths})

        self.cellstyle_map: defaultdict[tuple[CSSPair, ...], list[str]] = defaultdict(list)
        self.cellstyle_map_index: defaultdict[tuple[CSSPair, ...], list[str]] = defaultdict(list)
        body: list = self._translate_body(idx_lengths, max_rows, max_cols)
        d.update({"body": body})

        ctx_maps = {
            "cellstyle": "cellstyle_map",
            "cellstyle_index": "cellstyle_map_index",
            "cellstyle_columns": "cellstyle_map_columns",
        }
        for k, attr in ctx_maps.items():
            styles = [
                {"props": list(props), "selectors": selectors}
                for props, selectors in getattr(self, attr).items()
            ]
            d.update({k: styles})

        for dx in dxs:
            d["body"].extend(dx["body"])
            d["cellstyle"].extend(dx["cellstyle"])
            d["cellstyle_index"].extend(dx["cellstyle_index"])

        table_attr = self.table_attributes
        if not get_option("styler.html.mathjax"):
            table_attr = table_attr or ""
            if 'class="' in table_attr:
                table_attr = table_attr.replace('class="', 'class="tex2jax_ignore mathjax_ignore ')
            else:
                table_attr += ' class="tex2jax_ignore mathjax_ignore"'
        d.update({"table_attributes": table_attr})

        if self.tooltips:
            d = self.tooltips._translate(self, d)

        return d

    def _translate_header(self, sparsify_cols: bool, max_cols: int) -> list:
        col_lengths = _get_level_lengths(self.columns, sparsify_cols, max_cols, self.hidden_columns)

        clabels = self.data.columns.tolist()
        if self.data.columns.nlevels == 1:
            clabels = [[x] for x in clabels]
        clabels = list(zip(*clabels, strict=True))

        head = []
        for r, hide in enumerate(self.hide_columns_):
            if hide or not clabels:
                continue
            head.append(self._generate_col_header_row((r, clabels), max_cols, col_lengths))

        names = list(self.data.index.names)
        if (
            names
            and any(name is not None for name in names)
            and not all(self.hide_index_)
            and not self.hide_index_names
        ):
            head.append(self._generate_index_names_row(clabels, max_cols, col_lengths))

        return head

    def _generate_col_header_row(self, iter: Sequence, max_cols: int, col_lengths: dict) -> list:
        r, clabels = iter

        index_blanks = [_element("th", self.css["blank"], self.css["blank_value"], True)] * (
            self.index.nlevels - sum(self.hide_index_) - 1
        )

        name = self.data.columns.names[r]

        is_display = name is not None and not self.hide_column_names
        value = name if is_display else self.css["blank_value"]
        display_value = self._display_funcs_column_names[r](value) if is_display else None
        column_name = [
            _element(
                "th",
                (
                    f"{self.css['blank']} {self.css['level']}{r}"
                    if name is None
                    else f"{self.css['index_name']} {self.css['level']}{r}"
                ),
                value,
                not all(self.hide_index_),
                display_value=display_value,
            )
        ]

        column_headers: list = []
        visible_col_count: int = 0
        for c, value in enumerate(clabels[r]):
            header_element_visible = _is_visible(c, r, col_lengths)
            if header_element_visible:
                visible_col_count += col_lengths.get((r, c), 0)
            if self._check_trim(
                visible_col_count,
                max_cols,
                column_headers,
                "th",
                f"{self.css['col_heading']} {self.css['level']}{r} {self.css['col_trim']}",
            ):
                break

            header_element = _element(
                "th",
                (f"{self.css['col_heading']} {self.css['level']}{r} {self.css['col']}{c}"),
                value,
                header_element_visible,
                display_value=self._display_funcs_columns[(r, c)](value),
                attributes=(
                    f'colspan="{col_lengths.get((r, c), 0)}"'
                    if col_lengths.get((r, c), 0) > 1
                    else ""
                ),
            )

            if self.cell_ids:
                header_element["id"] = f"{self.css['level']}{r}_{self.css['col']}{c}"
            if header_element_visible and (r, c) in self.ctx_columns and self.ctx_columns[r, c]:
                header_element["id"] = f"{self.css['level']}{r}_{self.css['col']}{c}"
                self.cellstyle_map_columns[tuple(self.ctx_columns[r, c])].append(
                    f"{self.css['level']}{r}_{self.css['col']}{c}"
                )

            column_headers.append(header_element)

        return index_blanks + column_name + column_headers

    def _generate_index_names_row(self, iter: Sequence, max_cols: int, col_lengths: dict) -> list:
        clabels = iter

        index_names = [
            _element(
                "th",
                f"{self.css['index_name']} {self.css['level']}{c}",
                self.css["blank_value"] if name is None else name,
                not self.hide_index_[c],
                display_value=(None if name is None else self._display_funcs_index_names[c](name)),
            )
            for c, name in enumerate(self.data.index.names)
        ]

        column_blanks: list = []
        visible_col_count: int = 0
        if clabels:
            last_level = self.columns.nlevels - 1
            for c, _value in enumerate(clabels[last_level]):
                header_element_visible = _is_visible(c, last_level, col_lengths)
                if header_element_visible:
                    visible_col_count += 1
                if self._check_trim(
                    visible_col_count,
                    max_cols,
                    column_blanks,
                    "th",
                    f"{self.css['blank']} {self.css['col']}{c} {self.css['col_trim']}",
                    self.css["blank_value"],
                ):
                    break

                column_blanks.append(
                    _element(
                        "th",
                        f"{self.css['blank']} {self.css['col']}{c}",
                        self.css["blank_value"],
                        c not in self.hidden_columns,
                    )
                )

        return index_names + column_blanks

    def _translate_body(self, idx_lengths: dict, max_rows: int, max_cols: int) -> list:
        rlabels = self.data.index.tolist()
        if not isinstance(self.data.index, MultiIndex):
            rlabels = [[x] for x in rlabels]

        body: list = []
        visible_row_count: int = 0
        for r, row_tup in [
            z for z in enumerate(self.data.itertuples()) if z[0] not in self.hidden_rows
        ]:
            visible_row_count += 1  # noqa: SIM113  (pandas' counter, kept as pandas has it)
            if self._check_trim(visible_row_count, max_rows, body, "row"):
                break

            body.append(self._generate_body_row((r, row_tup, rlabels), max_cols, idx_lengths))
        return body

    def _check_trim(
        self,
        count: int,
        max: int,
        obj: list,
        element: str,
        css: str | None = None,
        value: str = "...",
    ) -> bool:
        if count > max:
            if element == "row":
                obj.append(self._generate_trimmed_row(max))
            else:
                obj.append(_element(element, css, value, True, attributes=""))
            return True
        return False

    def _generate_trimmed_row(self, max_cols: int) -> list:
        index_headers = [
            _element(
                "th",
                f"{self.css['row_heading']} {self.css['level']}{c} {self.css['row_trim']}",
                "...",
                not self.hide_index_[c],
                attributes="",
            )
            for c in range(self.data.index.nlevels)
        ]

        data: list = []
        visible_col_count: int = 0
        for c, _ in enumerate(self.columns):
            data_element_visible = c not in self.hidden_columns
            if data_element_visible:
                visible_col_count += 1
            if self._check_trim(
                visible_col_count,
                max_cols,
                data,
                "td",
                f"{self.css['data']} {self.css['row_trim']} {self.css['col_trim']}",
            ):
                break

            data.append(
                _element(
                    "td",
                    f"{self.css['data']} {self.css['col']}{c} {self.css['row_trim']}",
                    "...",
                    data_element_visible,
                    attributes="",
                )
            )

        return index_headers + data

    def _generate_body_row(self, iter: tuple, max_cols: int, idx_lengths: dict) -> list:
        r, row_tup, rlabels = iter

        index_headers = []
        for c, value in enumerate(rlabels[r]):
            header_element_visible = _is_visible(r, c, idx_lengths) and not self.hide_index_[c]
            header_element = _element(
                "th",
                f"{self.css['row_heading']} {self.css['level']}{c} {self.css['row']}{r}",
                value,
                header_element_visible,
                display_value=self._display_funcs_index[(r, c)](value),
                attributes=(
                    f'rowspan="{idx_lengths.get((c, r), 0)}"'
                    if idx_lengths.get((c, r), 0) > 1
                    else ""
                ),
            )

            if self.cell_ids:
                header_element["id"] = f"{self.css['level']}{c}_{self.css['row']}{r}"
            if header_element_visible and (r, c) in self.ctx_index and self.ctx_index[r, c]:
                header_element["id"] = f"{self.css['level']}{c}_{self.css['row']}{r}"
                self.cellstyle_map_index[tuple(self.ctx_index[r, c])].append(
                    f"{self.css['level']}{c}_{self.css['row']}{r}"
                )

            index_headers.append(header_element)

        data: list = []
        visible_col_count: int = 0
        for c, value in enumerate(row_tup[1:]):
            data_element_visible = c not in self.hidden_columns and r not in self.hidden_rows
            if data_element_visible:
                visible_col_count += 1
            if self._check_trim(
                visible_col_count,
                max_cols,
                data,
                "td",
                f"{self.css['data']} {self.css['row']}{r} {self.css['col_trim']}",
            ):
                break

            cls = ""
            if (r, c) in self.cell_context:
                cls = " " + self.cell_context[r, c]

            data_element = _element(
                "td",
                f"{self.css['data']} {self.css['row']}{r} {self.css['col']}{c}{cls}",
                value,
                data_element_visible,
                attributes="",
                display_value=self._display_funcs[(r, c)](value),
            )

            if self.cell_ids:
                data_element["id"] = f"{self.css['row']}{r}_{self.css['col']}{c}"
            if data_element_visible and (r, c) in self.ctx and self.ctx[r, c]:
                data_element["id"] = f"{self.css['row']}{r}_{self.css['col']}{c}"
                self.cellstyle_map[tuple(self.ctx[r, c])].append(
                    f"{self.css['row']}{r}_{self.css['col']}{c}"
                )

            data.append(data_element)

        return index_headers + data

    def _translate_latex(self, d: dict, clines: str | None) -> None:
        index_levels = self.index.nlevels
        visible_index_level_n = max(1, index_levels - sum(self.hide_index_))
        d["head"] = [
            [
                {**col, "cellstyle": self.ctx_columns[r, c - visible_index_level_n]}
                for c, col in enumerate(row)
                if col["is_visible"]
            ]
            for r, row in enumerate(d["head"])
        ]

        def _concatenated_visible_rows(obj: Any, n: int, row_indices: list[int]) -> int:
            row_indices.extend([r + n for r in range(len(obj.index)) if r not in obj.hidden_rows])
            n += len(obj.index)
            for concatenated in obj.concatenated:
                n = _concatenated_visible_rows(concatenated, n, row_indices)
            return n

        def concatenated_visible_rows(obj: Any) -> list[int]:
            row_indices: list[int] = []
            _concatenated_visible_rows(obj, 0, row_indices)
            return row_indices

        body = []
        row_body_cells: list = []
        for r, row in zip(concatenated_visible_rows(self), d["body"], strict=True):
            if all(self.hide_index_):
                row_body_headers = []
            else:
                row_body_headers = [
                    {
                        **col,
                        "display_value": (col["display_value"] if col["is_visible"] else ""),
                        "cellstyle": self.ctx_index[r, c],
                    }
                    for c, col in enumerate(row[:index_levels])
                    if (col["type"] == "th" and not self.hide_index_[c])
                ]

            row_body_cells = [
                {**col, "cellstyle": self.ctx[r, c]}
                for c, col in enumerate(row[index_levels:])
                if (col["is_visible"] and col["type"] == "td")
            ]

            body.append(row_body_headers + row_body_cells)
        d["body"] = body

        if clines not in [
            None,
            "all;data",
            "all;index",
            "skip-last;data",
            "skip-last;index",
        ]:
            raise ValueError(
                f"`clines` value of {clines} is invalid. Should either be None or one "
                f"of 'all;data', 'all;index', 'skip-last;data', 'skip-last;index'."
            )
        if clines is not None:
            data_len = len(row_body_cells) if "data" in clines and d["body"] else 0

            d["clines"] = defaultdict(list)
            visible_row_indexes: list[int] = [
                r for r in range(len(self.data.index)) if r not in self.hidden_rows
            ]
            visible_index_levels: list[int] = [
                i for i in range(index_levels) if not self.hide_index_[i]
            ]
            for rn, r in enumerate(visible_row_indexes):
                for lvln, lvl in enumerate(visible_index_levels):
                    if lvl == index_levels - 1 and "skip-last" in clines:
                        continue
                    idx_len = d["index_lengths"].get((lvl, r), None)
                    if idx_len is not None:
                        d["clines"][rn + idx_len].append(
                            f"\\cline{{{lvln + 1}-{len(visible_index_levels) + data_len}}}"
                        )

    def format(
        self,
        formatter: ExtFormatter | None = None,
        subset: Subset | None = None,
        na_rep: str | None = None,
        precision: int | None = None,
        decimal: str = ".",
        thousands: str | None = None,
        escape: str | None = None,
        hyperlinks: str | None = None,
    ) -> StylerRenderer:
        """Sets how the cells in `subset` are written, as `Styler.format` does."""
        if all(
            (
                formatter is None,
                subset is None,
                precision is None,
                decimal == ".",
                thousands is None,
                na_rep is None,
                escape is None,
                hyperlinks is None,
            )
        ):
            self._display_funcs.clear()
            return self

        subset = slice(None) if subset is None else subset
        subset = non_reducing_slice(subset)
        data = self.data.loc[subset]

        if not isinstance(formatter, dict):
            formatter = dict.fromkeys(data.columns, formatter)

        cis = self.columns.get_indexer_for(data.columns)
        ris = self.index.get_indexer_for(data.index)
        for ci in cis:
            format_func = _maybe_wrap_formatter(
                formatter.get(self.columns[ci]),
                na_rep=na_rep,
                precision=precision,
                decimal=decimal,
                thousands=thousands,
                escape=escape,
                hyperlinks=hyperlinks,
            )
            for ri in ris:
                self._display_funcs[(int(ri), int(ci))] = format_func

        return self

    def format_index(
        self,
        formatter: ExtFormatter | None = None,
        axis: Any = 0,
        level: Any = None,
        na_rep: str | None = None,
        precision: int | None = None,
        decimal: str = ".",
        thousands: str | None = None,
        escape: str | None = None,
        hyperlinks: str | None = None,
    ) -> StylerRenderer:
        """Sets how the index or column labels are written, as `Styler.format_index` does."""
        axis = _axis_number(axis)
        if axis == 0:
            display_funcs_, obj = self._display_funcs_index, self.index
        else:
            display_funcs_, obj = self._display_funcs_columns, self.columns
        levels_ = refactor_levels(level, obj)

        if all(
            (
                formatter is None,
                level is None,
                precision is None,
                decimal == ".",
                thousands is None,
                na_rep is None,
                escape is None,
                hyperlinks is None,
            )
        ):
            display_funcs_.clear()
            return self

        if not isinstance(formatter, dict):
            formatter = dict.fromkeys(levels_, formatter)
        else:
            formatter = {
                _level_number(obj, level): formatter_ for level, formatter_ in formatter.items()
            }

        for lvl in levels_:
            format_func = _maybe_wrap_formatter(
                formatter.get(lvl),
                na_rep=na_rep,
                precision=precision,
                decimal=decimal,
                thousands=thousands,
                escape=escape,
                hyperlinks=hyperlinks,
            )

            for idx in [(i, lvl) if axis == 0 else (lvl, i) for i in range(len(obj))]:
                display_funcs_[idx] = format_func

        return self

    def relabel_index(
        self,
        labels: Sequence | Index,
        axis: Any = 0,
        level: Any = None,
    ) -> StylerRenderer:
        """Shows other labels in place of the visible index or column labels."""
        axis = _axis_number(axis)
        if axis == 0:
            display_funcs_, obj = self._display_funcs_index, self.index
            hidden_labels, hidden_lvls = self.hidden_rows, self.hide_index_
        else:
            display_funcs_, obj = self._display_funcs_columns, self.columns
            hidden_labels, hidden_lvls = self.hidden_columns, self.hide_columns_
        visible_len = len(obj) - len(set(hidden_labels))
        if len(labels) != visible_len:
            raise ValueError(
                "``labels`` must be of length equal to the number of "
                f"visible labels along ``axis`` ({visible_len})."
            )

        if level is None:
            level = [i for i in range(obj.nlevels) if not hidden_lvls[i]]
        levels_ = refactor_levels(level, obj)

        def alias_(x: Any, value: Any) -> Any:
            if isinstance(value, str):
                return value.format(x)
            return value

        for ai, i in enumerate([i for i in range(len(obj)) if i not in hidden_labels]):
            if len(levels_) == 1:
                idx = (i, levels_[0]) if axis == 0 else (levels_[0], i)
                display_funcs_[idx] = partial(alias_, value=labels[ai])
            else:
                for aj, lvl in enumerate(levels_):
                    idx = (i, lvl) if axis == 0 else (lvl, i)
                    display_funcs_[idx] = partial(alias_, value=labels[ai][aj])

        return self

    def format_index_names(
        self,
        formatter: ExtFormatter | None = None,
        axis: Any = 0,
        level: Any = None,
        na_rep: str | None = None,
        precision: int | None = None,
        decimal: str = ".",
        thousands: str | None = None,
        escape: str | None = None,
        hyperlinks: str | None = None,
    ) -> StylerRenderer:
        """Sets how the names of the index or column levels are written."""
        axis = _axis_number(axis)
        if axis == 0:
            display_funcs_, obj = self._display_funcs_index_names, self.index
        else:
            display_funcs_, obj = self._display_funcs_column_names, self.columns
        levels_ = refactor_levels(level, obj)

        if all(
            (
                formatter is None,
                level is None,
                precision is None,
                decimal == ".",
                thousands is None,
                na_rep is None,
                escape is None,
                hyperlinks is None,
            )
        ):
            display_funcs_.clear()
            return self

        if not isinstance(formatter, dict):
            formatter = dict.fromkeys(levels_, formatter)
        else:
            formatter = {
                _level_number(obj, level): formatter_ for level, formatter_ in formatter.items()
            }

        for lvl in levels_:
            format_func = _maybe_wrap_formatter(
                formatter.get(lvl),
                na_rep=na_rep,
                precision=precision,
                decimal=decimal,
                thousands=thousands,
                escape=escape,
                hyperlinks=hyperlinks,
            )
            display_funcs_[lvl] = format_func

        return self


def _element(
    html_element: str,
    html_class: str | None,
    value: Any,
    is_visible: bool,
    **kwargs: Any,
) -> dict:
    if "display_value" not in kwargs or kwargs["display_value"] is None:
        kwargs["display_value"] = value
    return {
        "type": html_element,
        "value": value,
        "class": html_class,
        "is_visible": is_visible,
        **kwargs,
    }


def _get_trimming_maximums(
    rn: int,
    cn: int,
    max_elements: int,
    max_rows: int | None = None,
    max_cols: int | None = None,
    scaling_factor: float = 0.8,
) -> tuple[int, int]:
    def scale_down(rn: int, cn: int) -> tuple[int, int]:
        if cn >= rn:
            return rn, int(cn * scaling_factor)
        return int(rn * scaling_factor), cn

    if max_rows:
        rn = max_rows if rn > max_rows else rn
    if max_cols:
        cn = max_cols if cn > max_cols else cn

    while rn * cn > max_elements:
        rn, cn = scale_down(rn, cn)

    return rn, cn


def _label_text(value: Any) -> str:
    """A label as `_format_flat` writes it, which is what sparsifying compares."""
    return "NaN" if _missing(value) else str(value)


def _sparsified_levels(index: MultiIndex) -> list[list[Any]]:
    """Each level's labels with a repeat of the rows above marked, as `_format_multi` gives them.

    A label is marked when it and every level before it read the same as the row
    above. The last level is never marked.
    """
    rows = [tuple(_label_text(part) for part in label) for label in index.tolist()]
    if not rows:
        return []
    levels = len(rows[0])
    result = [list(rows[0])]
    for prev, cur in itertools.pairwise(rows):
        sparse: list[Any] = []
        for i in range(levels):
            if i == levels - 1:
                sparse.append(cur[i])
                break
            if prev[i] == cur[i]:
                sparse.append(no_default)
            else:
                sparse.extend(cur[i:])
                break
        result.append(sparse)
    return [list(level) for level in zip(*result, strict=True)]


def _get_level_lengths(
    index: Index,
    sparsify: bool,
    max_index: int,
    hidden_elements: Sequence[int] | None = None,
) -> dict:
    if hidden_elements is None:
        hidden_elements = []

    lengths: dict = {}
    if not isinstance(index, MultiIndex):
        for i in range(len(index)):
            if i not in hidden_elements:
                lengths[(0, i)] = 1
        return lengths

    levels = _sparsified_levels(index)
    for i, lvl in enumerate(levels):
        visible_row_count = 0
        last_label = 0
        for j, row in enumerate(lvl):
            if visible_row_count > max_index:
                break
            if not sparsify:
                if j not in hidden_elements:
                    lengths[(i, j)] = 1
                    visible_row_count += 1
            elif (row is not no_default) and (j not in hidden_elements):
                last_label = j
                lengths[(i, last_label)] = 1
                visible_row_count += 1
            elif row is not no_default:
                last_label = j
                lengths[(i, last_label)] = 0
            elif j not in hidden_elements:
                visible_row_count += 1
                if visible_row_count > max_index:
                    break
                if lengths[(i, last_label)] == 0:
                    last_label = j
                    lengths[(i, last_label)] = 1
                else:
                    lengths[(i, last_label)] += 1

    return {element: length for element, length in lengths.items() if length >= 1}


def _is_visible(idx_row: int, idx_col: int, lengths: dict) -> bool:
    return (idx_col, idx_row) in lengths


def format_table_styles(styles: CSSStyles) -> CSSStyles:
    return [
        {"selector": selector, "props": css_dict["props"]}
        for css_dict in styles
        for selector in css_dict["selector"].split(",")
    ]


def _default_formatter(x: Any, precision: int, thousands: bool = False) -> Any:
    if _is_float(x) or _is_complex(x):
        return f"{x:,.{precision}f}" if thousands else f"{x:.{precision}f}"
    if _is_integer(x):
        return f"{x:,}" if thousands else str(x)
    return x


def _wrap_decimal_thousands(formatter: Callable, decimal: str, thousands: str | None) -> Callable:
    def wrapper(x: Any) -> Any:
        if _is_float(x) or _is_integer(x) or _is_complex(x):
            if decimal != "." and thousands is not None and thousands != ",":
                return (
                    formatter(x)
                    .replace(",", "§_§-")
                    .replace(".", decimal)
                    .replace("§_§-", thousands)
                )
            if decimal != "." and (thousands is None or thousands == ","):
                return formatter(x).replace(".", decimal)
            if decimal == "." and thousands is not None and thousands != ",":
                return formatter(x).replace(",", thousands)
        return formatter(x)

    return wrapper


def _str_escape(x: Any, escape: str) -> Any:
    if isinstance(x, str):
        if escape == "html":
            return escape_html(x)
        if escape == "latex":
            return _escape_latex(x)
        if escape == "latex-math":
            return _escape_latex_math(x)
        raise ValueError(
            f"`escape` only permitted in {{'html', 'latex', 'latex-math'}}, got {escape}"
        )
    return x


def _render_href(x: Any, format: str) -> Any:
    if isinstance(x, str):
        if format == "html":
            href = '<a href="{0}" target="_blank">{0}</a>'
        elif format == "latex":
            href = r"\href{{{0}}}{{{0}}}"
        else:
            raise ValueError("``hyperlinks`` format can only be 'html' or 'latex'")
        pat = r"((http|ftp)s?:\/\/|www.)[\w/\-?=%.:@]+\.[\w/\-&?=%.,':;~!@#$*()\[\]]+"
        return re.sub(pat, lambda m: href.format(m.group(0)), x)
    return x


def _maybe_wrap_formatter(
    formatter: BaseFormatter | None = None,
    na_rep: str | None = None,
    precision: int | None = None,
    decimal: str = ".",
    thousands: str | None = None,
    escape: str | None = None,
    hyperlinks: str | None = None,
) -> Callable:
    if isinstance(formatter, str):

        def func_0(x: Any) -> Any:
            return formatter.format(x)

    elif callable(formatter):
        func_0 = formatter
    elif formatter is None:
        precision = get_option("styler.format.precision") if precision is None else precision
        func_0 = partial(_default_formatter, precision=precision, thousands=(thousands is not None))
    else:
        raise TypeError(f"'formatter' expected str or callable, got {type(formatter)}")

    if escape is not None:

        def func_1(x: Any) -> Any:
            return func_0(_str_escape(x, escape=escape))

    else:
        func_1 = func_0

    if decimal != "." or (thousands is not None and thousands != ","):
        func_2 = _wrap_decimal_thousands(func_1, decimal=decimal, thousands=thousands)
    else:
        func_2 = func_1

    if hyperlinks is not None:

        def func_3(x: Any) -> Any:
            return func_2(_render_href(x, format=hyperlinks))

    else:
        func_3 = func_2

    if na_rep is None:
        return func_3
    return lambda x: na_rep if _missing(x) else func_3(x)


def non_reducing_slice(slice_: Subset) -> tuple:
    """A subset as a `loc` key that keeps both axes, as pandas' `non_reducing_slice` makes it."""
    kinds: tuple = (Series, Index, MultiIndex, list, str)
    if np is not None:
        kinds += (np.ndarray,)
    if isinstance(slice_, kinds):
        slice_ = IndexSlice[:, slice_]

    def pred(part: Any) -> bool:
        if isinstance(part, tuple):
            return any((isinstance(s, slice) or _is_list_like(s)) for s in part)
        return isinstance(part, slice) or _is_list_like(part)

    if not _is_list_like(slice_):
        slice_ = [slice_] if isinstance(slice_, slice) else [[slice_]]
    else:
        slice_ = [p if pred(p) else [p] for p in slice_]
    return tuple(slice_)


def maybe_convert_css_to_tuples(style: CSSProperties) -> CSSList:
    if isinstance(style, str):
        if style and ":" not in style:
            raise ValueError(
                "Styles supplied as string must follow CSS rule formats, "
                f"for example 'attr: val;'. '{style}' was given."
            )
        s = style.split(";")
        return [
            (x.split(":")[0].strip(), ":".join(x.split(":")[1:]).strip())
            for x in s
            if x.strip() != ""
        ]
    return style


def refactor_levels(level: Any, obj: Index) -> list[int]:
    if level is None:
        levels_: list[int] = list(range(obj.nlevels))
    elif isinstance(level, int):
        levels_ = [level]
    elif isinstance(level, str):
        levels_ = [_level_number(obj, level)]
    elif isinstance(level, list):
        levels_ = [_level_number(obj, lev) if not isinstance(lev, int) else lev for lev in level]
    else:
        raise ValueError("`level` must be of type `int`, `str` or list of such")
    return levels_


class Tooltips:
    """The tooltips a Styler shows over its cells, as CSS pseudo elements or title attributes."""

    def __init__(
        self,
        css_props: CSSProperties = [  # noqa: B006
            ("visibility", "hidden"),
            ("position", "absolute"),
            ("z-index", 1),
            ("background-color", "black"),
            ("color", "white"),
            ("transform", "translate(-20px, -20px)"),
        ],
        css_name: str = "pd-t",
        tooltips: DataFrame | None = None,
        as_title_attribute: bool = False,
    ) -> None:
        self.class_name = css_name
        self.class_properties = css_props
        self.tt_data = DataFrame() if tooltips is None else tooltips
        self.table_styles: CSSStyles = []
        self.as_title_attribute = as_title_attribute

    @property
    def _class_styles(self) -> CSSStyles:
        return [
            {
                "selector": f".{self.class_name}",
                "props": maybe_convert_css_to_tuples(self.class_properties),
            }
        ]

    def _pseudo_css(self, uuid: str, name: str, row: int, col: int, text: str) -> list[CSSDict]:
        selector_id = "#T_" + uuid + "_row" + str(row) + "_col" + str(col)
        return [
            {
                "selector": selector_id + f":hover .{name}",
                "props": [("visibility", "visible")],
            },
            {
                "selector": selector_id + f" .{name}::after",
                "props": [("content", f'"{text}"')],
            },
        ]

    def _translate(self, styler: StylerRenderer, d: dict) -> dict:
        self.tt_data = self.tt_data.reindex_like(styler.data)
        if self.tt_data.empty:
            return d

        values = [list(row[1:]) for row in self.tt_data.itertuples()]
        mask = [[_missing(v) or (isinstance(v, str) and v == "") for v in row] for row in values]
        if not self.as_title_attribute:
            name = self.class_name
            self.table_styles = [
                style
                for sublist in [
                    self._pseudo_css(styler.uuid, name, i, j, str(values[i][j]))
                    for i in range(len(self.tt_data.index))
                    for j in range(len(self.tt_data.columns))
                    if not (mask[i][j] or i in styler.hidden_rows or j in styler.hidden_columns)
                ]
                for style in sublist
            ]

            if self.table_styles:
                for row in d["body"]:
                    for item in row:
                        if item["type"] == "td":
                            item["display_value"] = (
                                str(item["display_value"])
                                + f'<span class="{self.class_name}"></span>'
                            )
                d["table_styles"].extend(self._class_styles)
                d["table_styles"].extend(self.table_styles)
        else:
            index_offset = self.tt_data.index.nlevels
            body = d["body"]
            for i in range(len(self.tt_data.index)):
                for j in range(len(self.tt_data.columns)):
                    if not mask[i][j] or i in styler.hidden_rows or j in styler.hidden_columns:
                        row = body[i]
                        item = row[j + index_offset]
                        value = values[i][j]
                        item["attributes"] += f' title="{value}"'
        return d


def _parse_latex_table_wrapping(table_styles: CSSStyles, caption: str | None) -> bool:
    ignored_wrappers = ["toprule", "midrule", "bottomrule", "column_format"]
    return (
        table_styles is not None
        and any(d["selector"] not in ignored_wrappers for d in table_styles)
    ) or caption is not None


def _parse_latex_table_styles(table_styles: CSSStyles, selector: str) -> str | None:
    for style in table_styles[::-1]:
        if style["selector"] == selector:
            return str(style["props"][0][1]).replace("§", ":")
    return None


def _parse_latex_cell_styles(
    latex_styles: CSSList, display_value: str, convert_css: bool = False
) -> str:
    if convert_css:
        latex_styles = _parse_latex_css_conversion(latex_styles)
    for command, options in latex_styles[::-1]:
        formatter = {
            "--wrap": f"{{\\{command}--to_parse {display_value}}}",
            "--nowrap": f"\\{command}--to_parse {display_value}",
            "--lwrap": f"{{\\{command}--to_parse}} {display_value}",
            "--rwrap": f"\\{command}--to_parse{{{display_value}}}",
            "--dwrap": f"{{\\{command}--to_parse}}{{{display_value}}}",
        }
        display_value = f"\\{command}{options} {display_value}"
        for arg in ["--nowrap", "--wrap", "--lwrap", "--rwrap", "--dwrap"]:
            if arg in str(options):
                display_value = formatter[arg].replace(
                    "--to_parse", _parse_latex_options_strip(value=options, arg=arg)
                )
                break
    return display_value


def _parse_latex_header_span(
    cell: dict[str, Any],
    multirow_align: str,
    multicol_align: str,
    wrap: bool = False,
    convert_css: bool = False,
) -> str:
    display_val = _parse_latex_cell_styles(cell["cellstyle"], cell["display_value"], convert_css)
    if "attributes" in cell:
        attrs = cell["attributes"]
        if 'colspan="' in attrs:
            colspan = attrs[attrs.find('colspan="') + 9 :]
            colspan = int(colspan[: colspan.find('"')])
            if multicol_align == "naive-l":
                out = f"{{{display_val}}}" if wrap else f"{display_val}"
                blanks = " & {}" if wrap else " &"
                return out + blanks * (colspan - 1)
            if multicol_align == "naive-r":
                out = f"{{{display_val}}}" if wrap else f"{display_val}"
                blanks = "{} & " if wrap else "& "
                return blanks * (colspan - 1) + out
            return f"\\multicolumn{{{colspan}}}{{{multicol_align}}}{{{display_val}}}"
        if 'rowspan="' in attrs:
            if multirow_align == "naive":
                return display_val
            rowspan = attrs[attrs.find('rowspan="') + 9 :]
            rowspan = int(rowspan[: rowspan.find('"')])
            return f"\\multirow[{multirow_align}]{{{rowspan}}}{{*}}{{{display_val}}}"
    if wrap:
        return f"{{{display_val}}}"
    return display_val


def _parse_latex_options_strip(value: str | float, arg: str) -> str:
    return str(value).replace(arg, "").replace("/*", "").replace("*/", "").strip()


def _parse_latex_css_conversion(styles: CSSList) -> CSSList:
    def font_weight(value: Any, arg: str) -> tuple[str, str] | None:
        if value in ("bold", "bolder"):
            return "bfseries", f"{arg}"
        return None

    def font_style(value: Any, arg: str) -> tuple[str, str] | None:
        if value == "italic":
            return "itshape", f"{arg}"
        if value == "oblique":
            return "slshape", f"{arg}"
        return None

    def color(value: Any, user_arg: str, command: str, comm_arg: str) -> tuple[str, str]:
        arg = user_arg if user_arg != "" else comm_arg

        if value[0] == "#" and len(value) == 7:
            return command, f"[HTML]{{{value[1:].upper()}}}{arg}"
        if value[0] == "#" and len(value) == 4:
            val = f"{value[1].upper() * 2}{value[2].upper() * 2}{value[3].upper() * 2}"
            return command, f"[HTML]{{{val}}}{arg}"
        if value[:3] == "rgb":
            r = re.findall("(?<=\\()[0-9\\s%]+(?=,)", value)[0].strip()
            r = float(r[:-1]) / 100 if "%" in r else int(r) / 255
            g = re.findall("(?<=,)[0-9\\s%]+(?=,)", value)[0].strip()
            g = float(g[:-1]) / 100 if "%" in g else int(g) / 255
            if value[3] == "a":
                b = re.findall("(?<=,)[0-9\\s%]+(?=,)", value)[1].strip()
            else:
                b = re.findall("(?<=,)[0-9\\s%]+(?=\\))", value)[0].strip()
            b = float(b[:-1]) / 100 if "%" in b else int(b) / 255
            return command, f"[rgb]{{{r:.3f}, {g:.3f}, {b:.3f}}}{arg}"
        return command, f"{{{value}}}{arg}"

    converted: dict[str, Callable] = {
        "font-weight": font_weight,
        "background-color": partial(color, command="cellcolor", comm_arg="--lwrap"),
        "color": partial(color, command="color", comm_arg=""),
        "font-style": font_style,
    }

    latex_styles: CSSList = []
    for attribute, value in styles:
        if isinstance(value, str) and "--latex" in value:
            latex_styles.append((attribute, value.replace("--latex", "")))
        if attribute in converted:
            arg = ""
            for x in ["--wrap", "--nowrap", "--lwrap", "--dwrap", "--rwrap"]:
                if x in str(value):
                    arg, value = x, _parse_latex_options_strip(value, x)
                    break
            latex_style = converted[attribute](value, arg)
            if latex_style is not None:
                latex_styles.extend([latex_style])
    return latex_styles


def _escape_latex(s: str) -> str:
    return (
        s.replace("\\", "ab2§=§8yz")
        .replace("ab2§=§8yz ", "ab2§=§8yz\\space ")
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
        .replace("ab2§=§8yz", "\\textbackslash ")
    )


def _math_mode_with_dollar(s: str) -> str:
    s = s.replace(r"\$", r"rt8§=§7wz")
    pattern = re.compile(r"\$.*?\$")
    pos = 0
    ps = pattern.search(s, pos)
    res = []
    while ps:
        res.append(_escape_latex(s[pos : ps.span()[0]]))
        res.append(ps.group())
        pos = ps.span()[1]
        ps = pattern.search(s, pos)

    res.append(_escape_latex(s[pos : len(s)]))
    return "".join(res).replace(r"rt8§=§7wz", r"\$")


def _math_mode_with_parentheses(s: str) -> str:
    s = s.replace(r"\(", r"LEFT§=§6yzLEFT").replace(r"\)", r"RIGHTab5§=§RIGHT")
    res = []
    for item in re.split(r"LEFT§=§6yz|ab5§=§RIGHT", s):
        if item.startswith("LEFT") and item.endswith("RIGHT"):
            res.append(item.replace("LEFT", r"\(").replace("RIGHT", r"\)"))
        elif "LEFT" in item and "RIGHT" in item:
            res.append(_escape_latex(item).replace("LEFT", r"\(").replace("RIGHT", r"\)"))
        else:
            res.append(
                _escape_latex(item)
                .replace("LEFT", r"\textbackslash (")
                .replace("RIGHT", r"\textbackslash )")
            )
    return "".join(res)


def _escape_latex_math(s: str) -> str:
    s = s.replace(r"\$", r"rt8§=§7wz")
    ps_d = re.compile(r"\$.*?\$").search(s, 0)
    ps_p = re.compile(r"\(.*?\)").search(s, 0)
    mode = []
    if ps_d:
        mode.append(ps_d.span()[0])
    if ps_p:
        mode.append(ps_p.span()[0])
    if len(mode) == 0:
        return _escape_latex(s.replace(r"rt8§=§7wz", r"\$"))
    if s[mode[0]] == r"$":
        return _math_mode_with_dollar(s.replace(r"rt8§=§7wz", r"\$"))
    if s[mode[0] - 1 : mode[0] + 1] == r"\(":
        return _math_mode_with_parentheses(s.replace(r"rt8§=§7wz", r"\$"))
    return _escape_latex(s.replace(r"rt8§=§7wz", r"\$"))
