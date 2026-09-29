"""`Styler`, what `DataFrame.style` answers, pandas' `pandas.io.formats.style`.

A Styler keeps a frame and a queue of styling functions, and renders the frame
with the CSS those functions give each cell, header and label. `apply` and
`map` add to the queue, the highlight, gradient and bar methods are ready made
functions for it, and `to_html`, `to_latex`, `to_typst` and `to_string` run the
queue and fill the templates. This module is pandas' own code, ported method
for method so the output is the same text. Where pandas leans on numpy to hold
a grid of CSS strings this holds a list or a frame instead, so numpy is needed
only for what needs matplotlib, and matplotlib is imported only when a
gradient or a colormap bar is drawn.
"""

from __future__ import annotations

import copy
import math
import operator
import os
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from . import DataFrame, MultiIndex, Series
from ._config import get_option
from ._optional import imported
from ._style_render import (
    CSSProperties,
    CSSStyles,
    ExtFormatter,
    StylerRenderer,
    Subset,
    Tooltips,
    _axis_number,
    _missing,
    format_table_styles,
    jinja2,
    maybe_convert_css_to_tuples,
    non_reducing_slice,
    np,
    refactor_levels,
)
from .api.types import is_numeric_dtype


def _save_to_buffer(string: str, buf: Any = None, encoding: str | None = None) -> str | None:
    """The text, or None once it is written to a path or a handle, as pandas' `save_to_buffer`."""
    if buf is None:
        if encoding is not None:
            raise ValueError("buf is not a file name and encoding is specified.")
        return string
    buf = os.fspath(buf) if isinstance(buf, os.PathLike) else buf
    if isinstance(buf, str):
        with open(buf, "w", encoding=encoding or "utf-8", newline="") as handle:
            handle.write(string)
        return None
    if not hasattr(buf, "write"):
        raise TypeError("buf is not a file name and it has no write method")
    if encoding is not None:
        raise ValueError("buf is not a file name and encoding is specified.")
    buf.write(string)
    return None


def _grid(data: Any) -> list:
    """The values of a Series as a list, or of a frame as a list of rows."""
    if isinstance(data, DataFrame):
        return [list(row[1:]) for row in data.itertuples()]
    return list(data.tolist())


def _shaped(data: Any, cells: list) -> Any:
    """CSS strings in the shape of `data`: a list for a Series, a frame for a frame."""
    if isinstance(data, DataFrame):
        return DataFrame(cells, index=data.index, columns=data.columns)
    return cells


def _flat(values: list) -> list:
    """A grid's values in one list, row by row."""
    if values and isinstance(values[0], list):
        return [v for row in values for v in row]
    return list(values)


def _shape(values: Any) -> tuple:
    """The shape of a list, a list of lists or an array, as numpy writes it."""
    if hasattr(values, "shape"):
        return tuple(values.shape)
    if values and isinstance(values[0], list | tuple):
        return (len(values), len(values[0]))
    return (len(values),)


def _floats(values: list) -> list:
    """The values as floats, with every spelling of missing as NaN."""
    return [math.nan if _missing(v) else float(v) for v in values]


def _numeric_positions(data: DataFrame, with_bool: bool) -> list[int]:
    """The positions of the numeric columns, bool among them when `with_bool` is set."""
    return [
        i
        for i, dtype in enumerate(data.dtypes.tolist())
        if is_numeric_dtype(dtype) and (with_bool or str(dtype) not in {"bool", "boolean"})
    ]


def _pipe(obj: Any, func: Any, *args: Any, **kwargs: Any) -> Any:
    """pandas' `com.pipe`: call `func` with `obj`, or pass it as the keyword a tuple names."""
    if isinstance(func, tuple):
        func, target = func
        if target in kwargs:
            raise ValueError(f"{target} is both the pipe target and a keyword argument")
        kwargs[target] = obj
        return func(*args, **kwargs)
    return func(obj, *args, **kwargs)


class Styler(StylerRenderer):
    """Helps style a DataFrame or Series according to the data with HTML and CSS."""

    def __init__(
        self,
        data: DataFrame | Series,
        precision: int | None = None,
        table_styles: CSSStyles | None = None,
        uuid: str | None = None,
        caption: str | tuple | list | None = None,
        table_attributes: str | None = None,
        cell_ids: bool = True,
        na_rep: str | None = None,
        uuid_len: int = 5,
        decimal: str | None = None,
        thousands: str | None = None,
        escape: str | None = None,
        formatter: ExtFormatter | None = None,
    ) -> None:
        super().__init__(
            data=data,
            uuid=uuid,
            uuid_len=uuid_len,
            table_styles=table_styles,
            table_attributes=table_attributes,
            caption=caption,
            cell_ids=cell_ids,
            precision=precision,
        )

        thousands = thousands or get_option("styler.format.thousands")
        decimal = decimal or get_option("styler.format.decimal")
        na_rep = na_rep or get_option("styler.format.na_rep")
        escape = escape or get_option("styler.format.escape")
        formatter = formatter or get_option("styler.format.formatter")

        self.format(
            formatter=formatter,
            precision=precision,
            na_rep=na_rep,
            escape=escape,
            decimal=decimal,
            thousands=thousands,
        )

    def concat(self, other: Styler) -> Styler:
        """Appends another Styler's rows under this one, as a footer."""
        if not isinstance(other, Styler):
            raise TypeError("`other` must be of type `Styler`")
        if not self.data.columns.equals(other.data.columns):
            raise ValueError("`other.data` must have same columns as `Styler.data`")
        if not self.data.index.nlevels == other.data.index.nlevels:
            raise ValueError(
                "number of index levels must be same in `other` "
                "as in `Styler`. See documentation for suggestions."
            )
        self.concatenated.append(other)
        return self

    def _repr_html_(self) -> str | None:
        if get_option("styler.render.repr") == "html":
            return self.to_html()
        return None

    def _repr_latex_(self) -> str | None:
        if get_option("styler.render.repr") == "latex":
            return self.to_latex()
        return None

    def set_tooltips(
        self,
        ttips: DataFrame,
        props: CSSProperties | None = None,
        css_class: str | None = None,
        as_title_attribute: bool = False,
    ) -> Styler:
        """Shows the strings of `ttips` over the cells they align with."""
        if not self.cell_ids:
            raise NotImplementedError("Tooltips can only render with 'cell_ids' is True.")
        if not ttips.index.is_unique or not ttips.columns.is_unique:
            raise KeyError("Tooltips render only if `ttips` has unique index and columns.")
        if self.tooltips is None:
            self.tooltips = Tooltips()
        self.tooltips.tt_data = ttips
        if not as_title_attribute:
            if props:
                self.tooltips.class_properties = props
            if css_class:
                self.tooltips.class_name = css_class
        else:
            self.tooltips.as_title_attribute = as_title_attribute

        return self

    def to_latex(
        self,
        buf: Any = None,
        *,
        column_format: str | None = None,
        position: str | None = None,
        position_float: str | None = None,
        hrules: bool | None = None,
        clines: str | None = None,
        label: str | None = None,
        caption: str | tuple | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        multirow_align: str | None = None,
        multicol_align: str | None = None,
        siunitx: bool = False,
        environment: str | None = None,
        encoding: str | None = None,
        convert_css: bool = False,
    ) -> str | None:
        """Writes the styled table as LaTeX."""
        obj = self._copy(deepcopy=True)

        table_selectors = (
            [style["selector"] for style in self.table_styles]
            if self.table_styles is not None
            else []
        )

        if column_format is not None:
            obj.set_table_styles(
                [{"selector": "column_format", "props": f":{column_format}"}],
                overwrite=False,
            )
        elif "column_format" in table_selectors:
            pass
        else:
            numeric_cols = _numeric_positions(self.data, with_bool=True)
            column_format = ""
            for level in range(self.index.nlevels):
                column_format += "" if self.hide_index_[level] else "l"
            for ci, _ in enumerate(self.data.columns):
                if ci not in self.hidden_columns:
                    column_format += ("r" if not siunitx else "S") if ci in numeric_cols else "l"
            obj.set_table_styles(
                [{"selector": "column_format", "props": f":{column_format}"}],
                overwrite=False,
            )

        if position:
            obj.set_table_styles(
                [{"selector": "position", "props": f":{position}"}],
                overwrite=False,
            )

        if position_float:
            if environment == "longtable":
                raise ValueError("`position_float` cannot be used in 'longtable' `environment`")
            if position_float not in ["raggedright", "raggedleft", "centering"]:
                raise ValueError(
                    f"`position_float` should be one of "
                    f"'raggedright', 'raggedleft', 'centering', "
                    f"got: '{position_float}'"
                )
            obj.set_table_styles(
                [{"selector": "position_float", "props": f":{position_float}"}],
                overwrite=False,
            )

        hrules = get_option("styler.latex.hrules") if hrules is None else hrules
        if hrules:
            obj.set_table_styles(
                [
                    {"selector": "toprule", "props": ":toprule"},
                    {"selector": "midrule", "props": ":midrule"},
                    {"selector": "bottomrule", "props": ":bottomrule"},
                ],
                overwrite=False,
            )

        if label:
            obj.set_table_styles(
                [{"selector": "label", "props": f":{{{label.replace(':', '§')}}}"}],
                overwrite=False,
            )

        if caption:
            obj.set_caption(caption)

        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")
        environment = environment or get_option("styler.latex.environment")
        multicol_align = multicol_align or get_option("styler.latex.multicol_align")
        multirow_align = multirow_align or get_option("styler.latex.multirow_align")
        latex = obj._render_latex(
            sparse_index=sparse_index,
            sparse_columns=sparse_columns,
            multirow_align=multirow_align,
            multicol_align=multicol_align,
            environment=environment,
            convert_css=convert_css,
            siunitx=siunitx,
            clines=clines,
        )

        encoding = (
            (encoding or get_option("styler.render.encoding")) if isinstance(buf, str) else encoding
        )
        return _save_to_buffer(latex, buf=buf, encoding=encoding)

    def to_typst(
        self,
        buf: Any = None,
        *,
        encoding: str | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
    ) -> str | None:
        """Writes the styled table as Typst."""
        obj = self._copy(deepcopy=True)

        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")

        text = obj._render_typst(
            sparse_columns=sparse_columns,
            sparse_index=sparse_index,
            max_rows=max_rows,
            max_cols=max_columns,
        )
        return _save_to_buffer(text, buf=buf, encoding=(encoding if buf is not None else None))

    def to_html(
        self,
        buf: Any = None,
        *,
        table_uuid: str | None = None,
        table_attributes: str | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        bold_headers: bool = False,
        caption: str | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
        encoding: str | None = None,
        doctype_html: bool = False,
        exclude_styles: bool = False,
        **kwargs: Any,
    ) -> str | None:
        """Writes the styled table as HTML."""
        obj = self._copy(deepcopy=True)

        if table_uuid:
            obj.set_uuid(table_uuid)

        if table_attributes:
            obj.set_table_attributes(table_attributes)

        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")

        if bold_headers:
            obj.set_table_styles(
                [{"selector": "th", "props": "font-weight: bold;"}], overwrite=False
            )

        if caption is not None:
            obj.set_caption(caption)

        html = obj._render_html(
            sparse_index=sparse_index,
            sparse_columns=sparse_columns,
            max_rows=max_rows,
            max_cols=max_columns,
            exclude_styles=exclude_styles,
            encoding=encoding or get_option("styler.render.encoding"),
            doctype_html=doctype_html,
            **kwargs,
        )

        return _save_to_buffer(html, buf=buf, encoding=(encoding if buf is not None else None))

    def to_string(
        self,
        buf: Any = None,
        *,
        encoding: str | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
        delimiter: str = " ",
    ) -> str | None:
        """Writes the styled table as plain text."""
        obj = self._copy(deepcopy=True)

        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")

        text = obj._render_string(
            sparse_columns=sparse_columns,
            sparse_index=sparse_index,
            max_rows=max_rows,
            max_cols=max_columns,
            delimiter=delimiter,
        )
        return _save_to_buffer(text, buf=buf, encoding=(encoding if buf is not None else None))

    def set_td_classes(self, classes: DataFrame) -> Styler:
        """Adds the CSS classes of `classes` to the data cells they align with."""
        if not classes.index.is_unique or not classes.columns.is_unique:
            raise KeyError("Classes render only if `classes` has unique index and columns.")
        classes = classes.reindex_like(self.data)

        for r, row_tup in enumerate(classes.itertuples()):
            for c, value in enumerate(row_tup[1:]):
                if not (_missing(value) or value == ""):
                    self.cell_context[(r, c)] = str(value)

        return self

    def _update_ctx(self, attrs: DataFrame) -> None:
        if not self.index.is_unique or not self.columns.is_unique:
            raise KeyError(
                "`Styler.apply` and `.map` are not compatible with non-unique index or columns."
            )

        for cn in attrs.columns:
            j = self.columns.get_loc(cn)
            ser = attrs[cn]
            for rn, c in ser.items():
                if not c or _missing(c):
                    continue
                css_list = maybe_convert_css_to_tuples(c)
                i = self.index.get_loc(rn)
                self.ctx[(i, j)].extend(css_list)

    def _update_ctx_header(self, attrs: DataFrame, axis: int) -> None:
        for j in attrs.columns:
            ser = attrs[j]
            for i, c in ser.items():
                if not c or _missing(c):
                    continue
                css_list = maybe_convert_css_to_tuples(c)
                if axis == 0:
                    self.ctx_index[(i, j)].extend(css_list)
                else:
                    self.ctx_columns[(j, i)].extend(css_list)

    def _copy(self, deepcopy: bool = False) -> Styler:
        styler = type(self)(self.data)
        shallow = [
            "hide_index_",
            "hide_columns_",
            "hide_column_names",
            "hide_index_names",
            "table_attributes",
            "cell_ids",
            "caption",
            "uuid",
            "uuid_len",
            "template_latex",
            "template_html_style",
            "template_html_table",
            "template_html",
        ]
        deep = [
            "css",
            "concatenated",
            "_display_funcs",
            "_display_funcs_index",
            "_display_funcs_columns",
            "_display_funcs_index_names",
            "_display_funcs_column_names",
            "hidden_rows",
            "hidden_columns",
            "ctx",
            "ctx_index",
            "ctx_columns",
            "cell_context",
            "_todo",
            "table_styles",
            "tooltips",
        ]

        for attr in shallow:
            setattr(styler, attr, getattr(self, attr))

        for attr in deep:
            val = getattr(self, attr)
            setattr(styler, attr, copy.deepcopy(val) if deepcopy else val)

        return styler

    def __copy__(self) -> Styler:
        return self._copy(deepcopy=False)

    def __deepcopy__(self, memo: Any) -> Styler:
        return self._copy(deepcopy=True)

    def clear(self) -> None:
        """Resets the Styler, dropping every style, format and hidden label."""
        clean_copy = Styler(self.data, uuid=self.uuid)
        clean_attrs = [a for a in clean_copy.__dict__ if not callable(a)]
        self_attrs = [a for a in self.__dict__ if not callable(a)]
        for attr in clean_attrs:
            setattr(self, attr, getattr(clean_copy, attr))
        for attr in set(self_attrs).difference(clean_attrs):
            delattr(self, attr)

    def _apply(
        self,
        func: Callable,
        axis: Any = 0,
        subset: Subset | None = None,
        **kwargs: Any,
    ) -> Styler:
        subset = slice(None) if subset is None else subset
        subset = non_reducing_slice(subset)
        data = self.data.loc[subset]
        if data.empty:
            result = DataFrame()
        elif axis is None:
            result = func(data, **kwargs)
            if not isinstance(result, DataFrame):
                if np is None or not isinstance(result, np.ndarray):
                    raise TypeError(
                        f"Function {func!r} must return a DataFrame or ndarray "
                        f"when passed to `Styler.apply` with axis=None"
                    )
                if data.shape != result.shape:
                    raise ValueError(
                        f"Function {func!r} returned ndarray with wrong shape.\n"
                        f"Result has shape: {result.shape}\n"
                        f"Expected shape: {data.shape}"
                    )
                result = DataFrame(result.tolist(), index=data.index, columns=data.columns)
        else:
            axis = _axis_number(axis)
            if axis == 0:
                result = data.apply(func, axis=0, **kwargs)
            else:
                result = data.T.apply(func, axis=0, **kwargs).T

        if isinstance(result, Series):
            raise ValueError(
                f"Function {func!r} resulted in the apply method collapsing to a "
                f"Series.\nUsually, this is the result of the function returning a "
                f"single value, instead of list-like."
            )
        msg = (
            f"Function {func!r} created invalid {{0}} labels.\nUsually, this is "
            f"the result of the function returning a "
            f"{'Series' if axis is not None else 'DataFrame'} which contains invalid "
            f"labels, or returning an incorrectly shaped, list-like object which "
            f"cannot be mapped to labels, possibly due to applying the function along "
            f"the wrong axis.\n"
            f"Result {{0}} has shape: {{1}}\n"
            f"Expected {{0}} shape:   {{2}}"
        )
        if not all(result.index.isin(data.index)):
            raise ValueError(msg.format("index", result.index.shape, data.index.shape))
        if not all(result.columns.isin(data.columns)):
            raise ValueError(msg.format("columns", result.columns.shape, data.columns.shape))
        self._update_ctx(result)
        return self

    def apply(
        self,
        func: Callable,
        axis: Any = 0,
        subset: Subset | None = None,
        **kwargs: Any,
    ) -> Styler:
        """Styles columns, rows or the whole table with a function that answers CSS."""
        self._todo.append((lambda instance: instance._apply, (func, axis, subset), kwargs))
        return self

    def _apply_index(
        self,
        func: Callable,
        axis: Any = 0,
        level: Any = None,
        method: str = "apply",
        **kwargs: Any,
    ) -> Styler:
        axis = _axis_number(axis)
        obj = self.index if axis == 0 else self.columns

        levels_ = refactor_levels(level, obj)
        data = DataFrame(obj.to_list()).loc[:, levels_]

        if method == "apply":
            result = data.apply(func, axis=0, **kwargs)
        elif method == "map":
            result = data.map(func, **kwargs)

        self._update_ctx_header(result, axis)
        return self

    def apply_index(
        self,
        func: Callable,
        axis: Any = 0,
        level: Any = None,
        **kwargs: Any,
    ) -> Styler:
        """Styles the index or column labels a level at a time."""
        self._todo.append(
            (lambda instance: instance._apply_index, (func, axis, level, "apply"), kwargs)
        )
        return self

    def map_index(
        self,
        func: Callable,
        axis: Any = 0,
        level: Any = None,
        **kwargs: Any,
    ) -> Styler:
        """Styles the index or column labels one label at a time."""
        self._todo.append(
            (lambda instance: instance._apply_index, (func, axis, level, "map"), kwargs)
        )
        return self

    def _map(self, func: Callable, subset: Subset | None = None, **kwargs: Any) -> Styler:
        func = partial(func, **kwargs)
        if subset is None:
            subset = slice(None)
        subset = non_reducing_slice(subset)
        result = self.data.loc[subset].map(func)
        self._update_ctx(result)
        return self

    def map(self, func: Callable, subset: Subset | None = None, **kwargs: Any) -> Styler:
        """Styles each cell with a function that answers CSS for one value."""
        self._todo.append((lambda instance: instance._map, (func, subset), kwargs))
        return self

    def set_table_attributes(self, attributes: str) -> Styler:
        """Sets the attributes the `<table>` tag carries."""
        self.table_attributes = attributes
        return self

    def export(self) -> dict[str, Any]:
        """The styles that `use` can put on another Styler."""
        return {
            "apply": copy.copy(self._todo),
            "table_attributes": self.table_attributes,
            "table_styles": copy.copy(self.table_styles),
            "hide_index": all(self.hide_index_),
            "hide_columns": all(self.hide_columns_),
            "hide_index_names": self.hide_index_names,
            "hide_column_names": self.hide_column_names,
            "css": copy.copy(self.css),
        }

    def use(self, styles: dict[str, Any]) -> Styler:
        """Puts styles that `export` answered on this Styler."""
        self._todo.extend(styles.get("apply", []))
        table_attributes: str = self.table_attributes or ""
        obj_table_atts: str = (
            "" if styles.get("table_attributes") is None else str(styles.get("table_attributes"))
        )
        self.set_table_attributes((table_attributes + " " + obj_table_atts).strip())
        if styles.get("table_styles"):
            self.set_table_styles(styles.get("table_styles"), overwrite=False)

        for obj in ["index", "columns"]:
            hide_obj = styles.get("hide_" + obj)
            if hide_obj is not None:
                if isinstance(hide_obj, bool):
                    n = getattr(self, obj).nlevels
                    setattr(self, "hide_" + obj + "_", [hide_obj] * n)
                else:
                    setattr(self, "hide_" + obj + "_", hide_obj)

        self.hide_index_names = styles.get("hide_index_names", False)
        self.hide_column_names = styles.get("hide_column_names", False)
        if styles.get("css"):
            self.css = styles.get("css")
        return self

    def set_uuid(self, uuid: str) -> Styler:
        """Sets the id the table's CSS selectors are made from."""
        self.uuid = uuid
        return self

    def set_caption(self, caption: str | tuple | list) -> Styler:
        """Sets the table's caption."""
        msg = "`caption` must be either a string or 2-tuple of strings."
        if isinstance(caption, list | tuple):
            if (
                len(caption) != 2
                or not isinstance(caption[0], str)
                or not isinstance(caption[1], str)
            ):
                raise ValueError(msg)
        elif not isinstance(caption, str):
            raise ValueError(msg)
        self.caption = caption
        return self

    def set_sticky(
        self,
        axis: Any = 0,
        pixel_size: int | None = None,
        levels: Any = None,
    ) -> Styler:
        """Keeps the index or the column headers in view when the table scrolls."""
        axis = _axis_number(axis)
        obj = self.data.index if axis == 0 else self.data.columns
        pixel_size = pixel_size if pixel_size else (75 if axis == 0 else 25)

        props = "position:sticky; background-color:inherit;"
        if not isinstance(obj, MultiIndex):
            if axis == 1:
                styles: CSSStyles = [
                    {
                        "selector": "thead tr:nth-child(1) th",
                        "props": props + "top:0px; z-index:2;",
                    }
                ]
                if self.index.names[0] is not None:
                    styles[0]["props"] = props + f"top:0px; z-index:2; height:{pixel_size}px;"
                    styles.append(
                        {
                            "selector": "thead tr:nth-child(2) th",
                            "props": props
                            + f"top:{pixel_size}px; z-index:2; height:{pixel_size}px; ",
                        }
                    )
            else:
                styles = [
                    {
                        "selector": "thead tr th:nth-child(1)",
                        "props": props + "left:0px; z-index:3 !important;",
                    },
                    {
                        "selector": "tbody tr th:nth-child(1)",
                        "props": props + "left:0px; z-index:1;",
                    },
                ]

        else:
            range_idx = list(range(obj.nlevels))
            levels_: list[int] = refactor_levels(levels, obj) if levels else range_idx
            levels_ = sorted(levels_)

            if axis == 1:
                styles = []
                for i, level in enumerate(levels_):
                    styles.append(
                        {
                            "selector": f"thead tr:nth-child({level + 1}) th",
                            "props": props
                            + (f"top:{i * pixel_size}px; height:{pixel_size}px; z-index:2;"),
                        }
                    )
                if not all(name is None for name in self.index.names):
                    styles.append(
                        {
                            "selector": f"thead tr:nth-child({obj.nlevels + 1}) th",
                            "props": props
                            + (
                                f"top:{(len(levels_)) * pixel_size}px; "
                                f"height:{pixel_size}px; z-index:2;"
                            ),
                        }
                    )

            else:
                styles = []
                for i, level in enumerate(levels_):
                    props_ = props + (
                        f"left:{i * pixel_size}px; "
                        f"min-width:{pixel_size}px; "
                        f"max-width:{pixel_size}px; "
                    )
                    styles.extend(
                        [
                            {
                                "selector": f"thead tr th:nth-child({level + 1})",
                                "props": props_ + "z-index:3 !important;",
                            },
                            {
                                "selector": f"tbody tr th.level{level}",
                                "props": props_ + "z-index:1;",
                            },
                        ]
                    )

        return self.set_table_styles(styles, overwrite=False)

    def set_table_styles(
        self,
        table_styles: dict[Any, CSSStyles] | CSSStyles | None = None,
        axis: int = 0,
        overwrite: bool = True,
        css_class_names: dict[str, str] | None = None,
    ) -> Styler:
        """Sets CSS rules for the table, or for the rows or columns a dict names."""
        if css_class_names is not None:
            self.css = {**self.css, **css_class_names}

        if table_styles is None:
            return self
        if isinstance(table_styles, dict):
            axis = _axis_number(axis)
            obj = self.data.index if axis == 1 else self.data.columns
            idf = f".{self.css['row']}" if axis == 1 else f".{self.css['col']}"

            table_styles = [
                {
                    "selector": str(s["selector"]) + idf + str(idx),
                    "props": maybe_convert_css_to_tuples(s["props"]),
                }
                for key, styles in table_styles.items()
                for idx in obj.get_indexer_for([key])
                for s in format_table_styles(styles)
            ]
        else:
            table_styles = [
                {
                    "selector": s["selector"],
                    "props": maybe_convert_css_to_tuples(s["props"]),
                }
                for s in table_styles
            ]

        if not overwrite and self.table_styles is not None:
            self.table_styles.extend(table_styles)
        else:
            self.table_styles = table_styles
        return self

    def hide(
        self,
        subset: Subset | None = None,
        axis: Any = 0,
        level: Any = None,
        names: bool = False,
    ) -> Styler:
        """Hides the whole index or columns, some levels of them, or some rows or columns."""
        axis = _axis_number(axis)
        if axis == 0:
            obj, objs, alt = "index", "index", "rows"
        else:
            obj, objs, alt = "column", "columns", "columns"

        if level is not None and subset is not None:
            raise ValueError("`subset` and `level` cannot be passed simultaneously")

        if subset is None:
            if level is None and names:
                setattr(self, f"hide_{obj}_names", True)
                return self

            levels_ = refactor_levels(level, getattr(self, objs))
            setattr(
                self,
                f"hide_{objs}_",
                [lev in levels_ for lev in range(getattr(self, objs).nlevels)],
            )
        else:
            subset_ = (subset, slice(None)) if axis == 0 else (slice(None), subset)
            subset = non_reducing_slice(subset_)
            hide = self.data.loc[subset]
            h_els = [int(at) for at in getattr(self, objs).get_indexer_for(getattr(hide, objs))]
            setattr(self, f"hidden_{alt}", h_els)

        if names:
            setattr(self, f"hide_{obj}_names", True)
        return self

    def _get_numeric_subset_default(self) -> list[bool]:
        numeric = _numeric_positions(self.data, with_bool=False)
        return [at in numeric for at in range(len(self.data.columns))]

    def background_gradient(
        self,
        cmap: Any = "PuBu",
        low: float = 0,
        high: float = 0,
        axis: Any = 0,
        subset: Subset | None = None,
        text_color_threshold: float = 0.408,
        vmin: float | None = None,
        vmax: float | None = None,
        gmap: Sequence | None = None,
    ) -> Styler:
        """Colors the background of each cell by its value on a matplotlib colormap."""
        if subset is None and gmap is None:
            subset = self._get_numeric_subset_default()

        self.apply(
            _background_gradient,
            cmap=cmap,
            subset=subset,
            axis=axis,
            low=low,
            high=high,
            text_color_threshold=text_color_threshold,
            vmin=vmin,
            vmax=vmax,
            gmap=gmap,
        )
        return self

    def text_gradient(
        self,
        cmap: Any = "PuBu",
        low: float = 0,
        high: float = 0,
        axis: Any = 0,
        subset: Subset | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        gmap: Sequence | None = None,
    ) -> Styler:
        """Colors the text of each cell by its value on a matplotlib colormap."""
        if subset is None and gmap is None:
            subset = self._get_numeric_subset_default()

        return self.apply(
            _background_gradient,
            cmap=cmap,
            subset=subset,
            axis=axis,
            low=low,
            high=high,
            vmin=vmin,
            vmax=vmax,
            gmap=gmap,
            text_only=True,
        )

    def set_properties(self, subset: Subset | None = None, **kwargs: Any) -> Styler:
        """Gives every cell in `subset` the same CSS properties."""
        values = "".join([f"{p}: {v};" for p, v in kwargs.items()])
        return self.map(lambda x: values, subset=subset)

    def bar(
        self,
        subset: Subset | None = None,
        axis: Any = 0,
        *,
        color: str | list | tuple | None = None,
        cmap: Any | None = None,
        width: float = 100,
        height: float = 100,
        align: str | float | Callable = "mid",
        vmin: float | None = None,
        vmax: float | None = None,
        props: str = "width: 10em;",
    ) -> Styler:
        """Draws a bar in each cell's background as long as its value."""
        if color is None and cmap is None:
            color = "#d65f5f"
        elif color is not None and cmap is not None:
            raise ValueError("`color` and `cmap` cannot both be given")
        elif color is not None and (
            (isinstance(color, list | tuple) and len(color) > 2)
            or not isinstance(color, str | list | tuple)
        ):
            raise ValueError(
                "`color` must be string or list or tuple of 2 strings,"
                "(eg: color=['#d65f5f', '#5fba7d'])"
            )

        if not 0 <= width <= 100:
            raise ValueError(f"`width` must be a value in [0, 100], got {width}")
        if not 0 <= height <= 100:
            raise ValueError(f"`height` must be a value in [0, 100], got {height}")

        if subset is None:
            subset = self._get_numeric_subset_default()

        self.apply(
            _bar,
            subset=subset,
            axis=axis,
            align=align,
            colors=color,
            cmap=cmap,
            width=width / 100,
            height=height / 100,
            vmin=vmin,
            vmax=vmax,
            base_css=props,
        )

        return self

    def highlight_null(
        self,
        color: str = "red",
        subset: Subset | None = None,
        props: str | None = None,
    ) -> Styler:
        """Highlights the missing values."""

        def f(data: DataFrame, props: str) -> DataFrame:
            cells = [[props if _missing(v) else "" for v in row] for row in _grid(data)]
            return _shaped(data, cells)

        if props is None:
            props = f"background-color: {color};"
        return self.apply(f, axis=None, subset=subset, props=props)

    def highlight_max(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Any = 0,
        props: str | None = None,
    ) -> Styler:
        """Highlights the largest value of each column, row or the whole table."""
        if props is None:
            props = f"background-color: {color};"
        return self.apply(
            partial(_highlight_value, op="max"),
            axis=axis,
            subset=subset,
            props=props,
        )

    def highlight_min(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Any = 0,
        props: str | None = None,
    ) -> Styler:
        """Highlights the smallest value of each column, row or the whole table."""
        if props is None:
            props = f"background-color: {color};"
        return self.apply(
            partial(_highlight_value, op="min"),
            axis=axis,
            subset=subset,
            props=props,
        )

    def highlight_between(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Any = 0,
        left: Any = None,
        right: Any = None,
        inclusive: str = "both",
        props: str | None = None,
    ) -> Styler:
        """Highlights the values between `left` and `right`."""
        if props is None:
            props = f"background-color: {color};"
        return self.apply(
            _highlight_between,
            axis=axis,
            subset=subset,
            props=props,
            left=left,
            right=right,
            inclusive=inclusive,
        )

    def highlight_quantile(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Any = 0,
        q_left: float = 0.0,
        q_right: float = 1.0,
        interpolation: str = "linear",
        inclusive: str = "both",
        props: str | None = None,
    ) -> Styler:
        """Highlights the values between two quantiles."""
        subset_ = slice(None) if subset is None else subset
        subset_ = non_reducing_slice(subset_)
        data = self.data.loc[subset_]

        quantiles = [q_left, q_right]
        if axis is None:
            q = Series(_flat(_grid(data))).quantile(q=quantiles, interpolation=interpolation)
            axis_apply: int | None = None
        else:
            axis = _axis_number(axis)
            q = data.quantile(
                axis=axis, numeric_only=False, q=quantiles, interpolation=interpolation
            )
            axis_apply = 1 - axis

        if props is None:
            props = f"background-color: {color};"
        return self.apply(
            _highlight_between,
            axis=axis_apply,
            subset=subset,
            props=props,
            left=q.iloc[0],
            right=q.iloc[1],
            inclusive=inclusive,
        )

    @classmethod
    def from_custom_template(
        cls,
        searchpath: Sequence[str],
        html_table: str | None = None,
        html_style: str | None = None,
    ) -> type[Styler]:
        """A Styler subclass that renders with templates found on `searchpath`."""
        loader = jinja2.ChoiceLoader([jinja2.FileSystemLoader(searchpath), cls.loader])

        class MyStyler(cls):
            env = jinja2.Environment(loader=loader)
            if html_table:
                template_html_table = env.get_template(html_table)
            if html_style:
                template_html_style = env.get_template(html_style)

        return MyStyler

    def pipe(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Calls `func` with this Styler and answers what it answers."""
        return _pipe(self, func, *args, **kwargs)


def _validate_apply_axis_arg(arg: Any, arg_name: str, dtype: Any, data: Any) -> list:
    """An argument aligned to `data` as a list or list of rows, checked for its shape."""
    if isinstance(arg, Series) and isinstance(data, DataFrame):
        raise ValueError(
            f"'{arg_name}' is a Series but underlying data for operations "
            f"is a DataFrame since 'axis=None'"
        )
    if isinstance(arg, DataFrame) and isinstance(data, Series):
        raise ValueError(
            f"'{arg_name}' is a DataFrame but underlying data for "
            f"operations is a Series with 'axis in [0,1]'"
        )
    if isinstance(arg, Series | DataFrame):
        values = _grid(arg.reindex_like(data))
    else:
        values = (
            arg.tolist()
            if hasattr(arg, "tolist")
            else [list(row) if isinstance(row, list | tuple) else row for row in arg]
        )
        if _shape(values) != tuple(data.shape):
            raise ValueError(
                f"supplied '{arg_name}' is not correct shape for data over "
                f"selected 'axis': got {_shape(values)}, "
                f"expected {tuple(data.shape)}"
            )
    if dtype is float:
        if values and isinstance(values[0], list):
            return [_floats(row) for row in values]
        return _floats(values)
    return values


def _background_gradient(
    data: Any,
    cmap: Any = "PuBu",
    low: float = 0,
    high: float = 0,
    text_color_threshold: float = 0.408,
    vmin: float | None = None,
    vmax: float | None = None,
    gmap: Any = None,
    text_only: bool = False,
) -> Any:
    _matplotlib = imported("matplotlib", extra="Styler.background_gradient requires matplotlib.")
    import matplotlib.colors  # noqa: F401  (loads the submodule the norm and hex come from)
    import numpy

    if gmap is None:
        values = _grid(data)
        if values and isinstance(values[0], list):
            gmap = numpy.array([_floats(row) for row in values], dtype=float)
        else:
            gmap = numpy.array(_floats(values), dtype=float)
    else:
        gmap = numpy.array(_validate_apply_axis_arg(gmap, "gmap", float, data), dtype=float)

    smin = numpy.nanmin(gmap) if vmin is None else vmin
    smax = numpy.nanmax(gmap) if vmax is None else vmax
    rng = smax - smin
    norm = _matplotlib.colors.Normalize(smin - (rng * low), smax + (rng * high))

    if cmap is None:
        rgbas = _matplotlib.colormaps[_matplotlib.rcParams["image.cmap"]](norm(gmap))
    else:
        rgbas = _matplotlib.colormaps.get_cmap(cmap)(norm(gmap))

    def relative_luminance(rgba: Any) -> float:
        r, g, b = (x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in rgba[:3])
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def css(rgba: Any, text_only: bool) -> str:
        if not text_only:
            dark = relative_luminance(rgba) < text_color_threshold
            text_color = "#f1f1f1" if dark else "#000000"
            return f"background-color: {_matplotlib.colors.rgb2hex(rgba)};color: {text_color};"
        return f"color: {_matplotlib.colors.rgb2hex(rgba)};"

    if data.ndim == 1:
        return [css(rgba, text_only) for rgba in rgbas]
    return DataFrame(
        [[css(rgba, text_only) for rgba in row] for row in rgbas],
        index=data.index,
        columns=data.columns,
    )


def _compared(values: list, bound: Any, op: Callable, iterable: bool) -> list:
    """`op` between each value and the bound or its aligned value, False where one is missing."""

    def one(value: Any, other: Any) -> bool:
        if _missing(value) or _missing(other):
            return False
        return bool(op(value, other))

    if values and isinstance(values[0], list):
        return [
            [one(v, bound[i][j] if iterable else bound) for j, v in enumerate(row)]
            for i, row in enumerate(values)
        ]
    return [one(v, bound[i] if iterable else bound) for i, v in enumerate(values)]


def _highlight_between(
    data: Any,
    props: str,
    left: Any = None,
    right: Any = None,
    inclusive: bool | str = True,
) -> Any:
    left_many = _is_iterable(left) and not isinstance(left, str)
    right_many = _is_iterable(right) and not isinstance(right, str)
    if left_many:
        left = _validate_apply_axis_arg(left, "left", None, data)
    if right_many:
        right = _validate_apply_axis_arg(right, "right", None, data)

    if inclusive == "both":
        ops = (operator.ge, operator.le)
    elif inclusive == "neither":
        ops = (operator.gt, operator.lt)
    elif inclusive == "left":
        ops = (operator.ge, operator.lt)
    elif inclusive == "right":
        ops = (operator.gt, operator.le)
    else:
        raise ValueError(
            f"'inclusive' values can be 'both', 'left', 'right', or 'neither' got {inclusive}"
        )

    values = _grid(data)
    everything = _compared(values, True, lambda v, _: True, False)
    g_left = everything if left is None else _compared(values, left, ops[0], left_many)
    l_right = everything if right is None else _compared(values, right, ops[1], right_many)
    if values and isinstance(values[0], list):
        cells = [
            [props if a and b else "" for a, b in zip(ra, rb, strict=True)]
            for ra, rb in zip(g_left, l_right, strict=True)
        ]
    else:
        cells = [props if a and b else "" for a, b in zip(g_left, l_right, strict=True)]
    return _shaped(data, cells)


def _is_iterable(value: Any) -> bool:
    """numpy's `iterable`: whether `iter` accepts it."""
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _highlight_value(data: Any, op: str, props: str) -> Any:
    value = getattr(data, op)(skipna=True)
    if isinstance(data, DataFrame):
        value = getattr(value, op)(skipna=True)

    def one(v: Any) -> str:
        if _missing(v) or _missing(value):
            return ""
        return props if v == value else ""

    values = _grid(data)
    if values and isinstance(values[0], list):
        return _shaped(data, [[one(v) for v in row] for row in values])
    return [one(v) for v in values]


def _bar(
    data: Any,
    align: str | float | Callable,
    colors: str | list | tuple,
    cmap: Any,
    width: float,
    height: float,
    vmin: float | None,
    vmax: float | None,
    base_css: str,
) -> Any:
    def css_bar(start: float, end: float, color: str) -> str:
        cell_css = base_css
        if end > start:
            cell_css += "background: linear-gradient(90deg,"
            if start > 0:
                cell_css += f" transparent {start * 100:.1f}%, {color} {start * 100:.1f}%,"
            cell_css += f" {color} {end * 100:.1f}%, transparent {end * 100:.1f}%)"
        return cell_css

    def css_calc(x: float, left: float, right: float, align: str, color: Any) -> str:
        if math.isnan(x):
            return base_css

        if isinstance(color, list | tuple):
            color = color[0] if x < 0 else color[1]

        x = left if x < left else x
        x = right if x > right else x

        start: float = 0
        end: float = 1

        if align == "left":
            end = (x - left) / (right - left)

        elif align == "right":
            start = (x - left) / (right - left)

        else:
            z_frac: float = 0.5
            if align == "zero":
                limit: float = max(abs(left), abs(right))
                left, right = -limit, limit
            elif align == "mid":
                mid: float = (left + right) / 2
                z_frac = -mid / (right - left) + 0.5 if mid < 0 else -left / (right - left)

            if x < 0:
                start, end = (x - left) / (right - left), z_frac
            else:
                start, end = z_frac, (x - left) / (right - left)

        ret = css_bar(start * width, end * width, color)
        if height < 1 and "background: linear-gradient(" in ret:
            return ret + f" no-repeat center; background-size: 100% {height * 100:.1f}%;"
        return ret

    grid = _grid(data)
    two = bool(grid) and isinstance(grid[0], list)
    values = [_floats(row) for row in grid] if two else _floats(grid)
    present = [v for v in _flat(values) if not math.isnan(v)]
    left = (min(present) if present else math.nan) if vmin is None else vmin
    right = (max(present) if present else math.nan) if vmax is None else vmax
    z: float = 0

    if align == "mid":
        if left >= 0:
            align, left = "left", 0 if vmin is None else vmin
        elif right <= 0:
            align, right = "right", 0 if vmax is None else vmax
    elif align == "mean":
        z, align = (sum(present) / len(present) if present else math.nan), "zero"
    elif callable(align):
        z, align = align(np.array(values) if np is not None else values), "zero"
    elif isinstance(align, float | int):
        z, align = float(align), "zero"
    elif align not in ("left", "right", "zero"):
        raise ValueError(
            "`align` should be in {'left', 'right', 'mid', 'mean', 'zero'} or be a "
            "value defining the center line or a callable that returns a float"
        )

    rgbas = None
    if cmap is not None:
        _matplotlib = imported("matplotlib", extra="Styler.bar requires matplotlib.")
        import matplotlib.colors  # noqa: F401  (loads the submodule the norm and hex come from)
        import numpy

        cmap = _matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
        norm = _matplotlib.colors.Normalize(left, right)
        rgbas = cmap(norm(numpy.array(values, dtype=float)))
        if not two:
            rgbas = [_matplotlib.colors.rgb2hex(rgba) for rgba in rgbas]
        else:
            rgbas = [[_matplotlib.colors.rgb2hex(rgba) for rgba in row] for row in rgbas]

    if not two:
        return [
            css_calc(x - z, left - z, right - z, align, colors if rgbas is None else rgbas[i])
            for i, x in enumerate(values)
        ]
    return _shaped(
        data,
        [
            [
                css_calc(
                    x - z,
                    left - z,
                    right - z,
                    align,
                    colors if rgbas is None else rgbas[i][j],
                )
                for j, x in enumerate(row)
            ]
            for i, row in enumerate(values)
        ],
    )
