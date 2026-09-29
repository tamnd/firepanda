"""The matplotlib plotting backend, pandas' `pandas.plotting._matplotlib` for firepanda.

This is pandas' backend carried over one plot kind at a time, reading the data
through firepanda's public methods, so a plot draws the same artists pandas
draws: the same lines, bars, patches, tick labels, titles and legends, in the
same colours from matplotlib's colour cycle. `_plotting` is the front end that
checks the arguments and calls `plot` here.

One thing is drawn differently. pandas draws a series over a regular date index
against period numbers with its own tick formatter, so that it can mix
frequencies on one axes. Here dates are drawn as matplotlib draws them, and the
tick labels are matplotlib's date labels. Document 115 of the compat notes
describes it.

Importing this module imports matplotlib, so the front end loads it only when a
plot is drawn and turns the ImportError into pandas' sentence.
"""

from __future__ import annotations

import contextlib
import random
import warnings
from collections.abc import Iterable
from math import ceil
from typing import Any, NamedTuple

import matplotlib as mpl
import numpy as np

from . import _plotting

_NO_DEFAULT = object()


def _series(data: Any) -> bool:
    return _plotting._is_series(data)


def _frame(data: Any) -> bool:
    return _plotting._is_frame(data)


def _is_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_list_like(value: Any) -> bool:
    return hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict))


def _is_hashable(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _is_multi(index: Any) -> bool:
    return type(index).__name__ == "MultiIndex"


def pprint_thing(thing: Any) -> str:
    """A label as pandas prints one, with a tuple's parts in parentheses."""
    if isinstance(thing, tuple):
        return "(" + ", ".join(pprint_thing(part) for part in thing) + ")"
    if isinstance(thing, list):
        return "[" + ", ".join(pprint_thing(part) for part in thing) + "]"
    return str(thing)


def _dtype_name(values: Any) -> str:
    return str(values.dtype)


def _real_numeric(dtype: Any) -> bool:
    name = str(dtype).lower()
    return name.startswith(("int", "uint", "float"))


def _numeric(dtype: Any) -> bool:
    name = str(dtype).lower()
    return _real_numeric(dtype) or name.startswith(("bool", "complex"))


def _is_dates(index: Any) -> bool:
    name = str(index.dtype)
    return name.startswith(("datetime64", "period")) or type(index).__name__ in (
        "DatetimeIndex",
        "PeriodIndex",
    )


def _values(column: Any) -> np.ndarray:
    """A column as numpy holds it, a masked column as floats with NaN for its gaps."""
    name = str(column.dtype)
    if name[:1] in "IUF" and name[1:2].isalpha() and name != "category":
        return np.asarray(column.to_numpy(dtype="float", na_value=np.nan))
    if name == "boolean":
        return np.asarray(column.to_numpy(dtype="object", na_value=np.nan))
    if name.startswith("period"):
        return np.asarray(column.dt.to_timestamp().to_numpy())
    return np.asarray(column.to_numpy())


def _index_values(index: Any) -> np.ndarray:
    if type(index).__name__ == "PeriodIndex":
        index = index.to_timestamp()
    return np.asarray(index.to_numpy())


def isna(values: Any) -> np.ndarray:
    """Which of the values are missing, as pandas' isna reads an array."""
    array = np.asarray(values)
    if array.dtype.kind in "fc":
        return np.isnan(array)
    if array.dtype.kind in "mM":
        return np.isnat(array)
    if array.dtype.kind == "O":
        flat = [
            item is None
            or (isinstance(item, float) and item != item)
            or type(item).__name__ in ("NAType", "NaTType")
            for item in array.reshape(-1)
        ]
        return np.array(flat, dtype=bool).reshape(array.shape)
    return np.zeros(array.shape, dtype=bool)


def remove_na_arraylike(values: Any) -> np.ndarray:
    array = np.asarray(values)
    return array[~isna(array)]


def unpack_single_str_list(keys: Any) -> Any:
    if isinstance(keys, list) and len(keys) == 1:
        keys = keys[0]
    return keys


def _maybe_make_list(value: Any) -> Any:
    if value is not None and not isinstance(value, (tuple, list)):
        return [value]
    return value


def _numeric_columns(frame: Any) -> list[Any]:
    return list(frame.select_dtypes(include=["number", "bool"]).columns)


# Colours.


def get_standard_colors(
    num_colors: int,
    colormap: Any = None,
    color_type: str = "default",
    *,
    color: Any = None,
) -> Any:
    """The colours to draw `num_colors` series in, cycled from matplotlib's as pandas does."""
    if isinstance(color, dict):
        return color
    if color is None and colormap is not None:
        colors = _get_colors_from_colormap(colormap, num_colors)
    elif color is not None:
        if colormap is not None:
            warnings.warn(
                "'color' and 'colormap' cannot be used simultaneously. Using 'color'",
                stacklevel=3,
            )
        colors = _get_colors_from_color(color)
    elif color_type == "default":
        prop_cycle = mpl.rcParams["axes.prop_cycle"]
        colors = [c["color"] for c in list(prop_cycle)[: min(num_colors, len(prop_cycle))]]
    elif color_type == "random":
        colors = np.random.default_rng(num_colors).random((num_colors, 3)).tolist()
    else:
        raise ValueError("color_type must be either 'default' or 'random'")
    count = max(num_colors, len(colors))
    return [colors[i % len(colors)] for i in range(count)] if colors else []


def _get_cmap_instance(colormap: Any) -> Any:
    if isinstance(colormap, str):
        return mpl.colormaps[colormap]
    return colormap


def _get_colors_from_colormap(colormap: Any, num_colors: int) -> list[Any]:
    cmap = _get_cmap_instance(colormap)
    return [cmap(num) for num in np.linspace(0, 1, num=num_colors)]


def _is_single_string_color(color: Any) -> bool:
    try:
        mpl.colors.ColorConverter().to_rgba(color)
    except ValueError:
        return False
    return True


def _is_floats_color(color: Any) -> bool:
    return bool(
        _is_list_like(color)
        and len(color) in (3, 4)
        and all(isinstance(x, (int, float)) for x in color)
    )


def _is_single_color(color: Any) -> bool:
    if isinstance(color, str) and _is_single_string_color(color):
        return True
    return _is_floats_color(color)


def _get_colors_from_color(color: Any) -> list[Any]:
    if len(color) == 0:
        raise ValueError(f"Invalid color argument: {color}")
    if _is_single_color(color):
        return [color]
    out = []
    for x in color:
        if not _is_single_color(x):
            raise ValueError(f"Invalid color {x}")
        out.append(x)
    return out


def _color_in_style(style: str) -> bool:
    return not set(mpl.colors.BASE_COLORS).isdisjoint(style)


# Axes layout, from pandas' tools module.


def do_adjust_figure(fig: Any) -> bool:
    if not hasattr(fig, "get_constrained_layout"):
        return False
    return not fig.get_constrained_layout()


def maybe_adjust_figure(fig: Any, *args: Any, **kwargs: Any) -> None:
    if do_adjust_figure(fig):
        fig.subplots_adjust(*args, **kwargs)


def format_date_labels(ax: Any, rot: Any) -> None:
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation(rot)
    fig = ax.get_figure()
    if fig is not None:
        maybe_adjust_figure(fig, bottom=0.2)


def table(ax: Any, data: Any, rowLabels: Any = None, colLabels: Any = None, **kwargs: Any) -> Any:
    """Draw the values of a frame or a column as a matplotlib table on `ax`."""
    import matplotlib.table

    if _series(data):
        data = data.to_frame()
    elif not _frame(data):
        raise ValueError("Input data must be DataFrame or Series")
    if rowLabels is None:
        rowLabels = [pprint_thing(label) for label in data.index]
    if colLabels is None:
        colLabels = [pprint_thing(label) for label in data.columns]
    cellText = np.array([_values(data[col]) for col in data.columns], dtype=object).T
    return matplotlib.table.table(
        ax, cellText=cellText, rowLabels=rowLabels, colLabels=colLabels, **kwargs
    )


def _get_layout(nplots: int, layout: Any = None, layout_type: str = "box") -> tuple[int, int]:
    if layout is not None:
        if not isinstance(layout, (tuple, list)) or len(layout) != 2:
            raise ValueError("Layout must be a tuple of (rows, columns)")
        nrows, ncols = layout
        if nrows == -1 and ncols > 0:
            layout = (ceil(nplots / ncols), ncols)
        elif ncols == -1 and nrows > 0:
            layout = (nrows, ceil(nplots / nrows))
        elif ncols <= 0 and nrows <= 0:
            raise ValueError("At least one dimension of layout must be positive")
        nrows, ncols = layout
        if nrows * ncols < nplots:
            raise ValueError(
                f"Layout of {nrows}x{ncols} must be larger than required size {nplots}"
            )
        return layout
    if layout_type == "single":
        return (1, 1)
    if layout_type == "horizontal":
        return (1, nplots)
    if layout_type == "vertical":
        return (nplots, 1)
    layouts = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2)}
    if nplots in layouts:
        return layouts[nplots]
    k = 1
    while k**2 < nplots:
        k += 1
    if (k - 1) * k >= nplots:
        return k, (k - 1)
    return k, k


def flatten_axes(axes: Any) -> Iterable[Any]:
    if not _is_list_like(axes):
        yield axes
    elif isinstance(axes, np.ndarray):
        yield from np.asarray(axes).reshape(-1)
    else:
        yield from axes


def create_subplots(
    naxes: int,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    subplot_kw: Any = None,
    ax: Any = None,
    layout: Any = None,
    layout_type: str = "box",
    **fig_kw: Any,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    if subplot_kw is None:
        subplot_kw = {}
    if ax is None:
        fig = plt.figure(**fig_kw)
    else:
        if _is_list_like(ax):
            if squeeze:
                ax = np.fromiter(flatten_axes(ax), dtype=object)
            if layout is not None:
                warnings.warn(
                    "When passing multiple axes, layout keyword is ignored.",
                    UserWarning,
                    stacklevel=3,
                )
            if sharex or sharey:
                warnings.warn(
                    "When passing multiple axes, sharex and sharey "
                    "are ignored. These settings must be specified when creating axes.",
                    UserWarning,
                    stacklevel=3,
                )
            ax = np.asarray(ax, dtype=object)
            if ax.size == naxes:
                fig = ax.flat[0].get_figure()
                return fig, ax
            raise ValueError(
                f"The number of passed axes must be {naxes}, the same as the output plot"
            )
        fig = ax.get_figure()
        if naxes == 1:
            if squeeze:
                return fig, ax
            return fig, np.fromiter(flatten_axes(ax), dtype=object)
        warnings.warn(
            "To output multiple subplots, the figure containing the passed axes is being cleared.",
            UserWarning,
            stacklevel=3,
        )
        fig.clear()
    nrows, ncols = _get_layout(naxes, layout=layout, layout_type=layout_type)
    nplots = nrows * ncols
    axarr = np.empty(nplots, dtype=object)
    ax0 = fig.add_subplot(nrows, ncols, 1, **subplot_kw)
    if sharex:
        subplot_kw["sharex"] = ax0
    if sharey:
        subplot_kw["sharey"] = ax0
    axarr[0] = ax0
    for i in range(1, nplots):
        kwds = subplot_kw.copy()
        if i >= naxes:
            kwds["sharex"] = None
            kwds["sharey"] = None
        axarr[i] = fig.add_subplot(nrows, ncols, i + 1, **kwds)
    if naxes != nplots:
        for extra in axarr[naxes:]:
            extra.set_visible(False)
    handle_shared_axes(axarr, nplots, naxes, nrows, ncols, sharex, sharey)
    if squeeze:
        axes = axarr[0] if nplots == 1 else axarr.reshape(nrows, ncols).squeeze()
    else:
        axes = axarr.reshape(nrows, ncols)
    return fig, axes


def _remove_labels_from_axis(axis: Any) -> None:
    for t in axis.get_majorticklabels():
        t.set_visible(False)
    if isinstance(axis.get_minor_locator(), mpl.ticker.NullLocator):
        axis.set_minor_locator(mpl.ticker.AutoLocator())
    if isinstance(axis.get_minor_formatter(), mpl.ticker.NullFormatter):
        axis.set_minor_formatter(mpl.ticker.FormatStrFormatter(""))
    for t in axis.get_minorticklabels():
        t.set_visible(False)
    axis.get_label().set_visible(False)


def _has_externally_shared_axis(ax1: Any, compare_axis: str) -> bool:
    axes = ax1.get_shared_x_axes() if compare_axis == "x" else ax1.get_shared_y_axes()
    ax1_points = ax1.get_position().get_points()
    for ax2 in axes.get_siblings(ax1):
        if not np.array_equal(ax1_points, ax2.get_position().get_points()):
            return True
    return False


def handle_shared_axes(
    axarr: Any, nplots: int, naxes: int, nrows: int, ncols: int, sharex: bool, sharey: bool
) -> None:
    if nplots <= 1:
        return

    def row_num(x: Any) -> int:
        return x.get_subplotspec().rowspan.start

    def col_num(x: Any) -> int:
        return x.get_subplotspec().colspan.start

    if nrows > 1:
        try:
            layout = np.zeros((nrows + 1, ncols + 1), dtype=np.bool_)
            for ax in axarr:
                layout[row_num(ax), col_num(ax)] = ax.get_visible()
            for ax in axarr:
                if not layout[row_num(ax) + 1, col_num(ax)]:
                    continue
                if sharex or _has_externally_shared_axis(ax, "x"):
                    _remove_labels_from_axis(ax.xaxis)
        except IndexError:
            for ax in axarr:
                if ax.get_subplotspec().is_last_row():
                    continue
                if sharex or _has_externally_shared_axis(ax, "x"):
                    _remove_labels_from_axis(ax.xaxis)
    if ncols > 1:
        for ax in axarr:
            if ax.get_subplotspec().is_first_col():
                continue
            if sharey or _has_externally_shared_axis(ax, "y"):
                _remove_labels_from_axis(ax.yaxis)


def set_ticks_props(
    axes: Any, xlabelsize: Any = None, xrot: Any = None, ylabelsize: Any = None, yrot: Any = None
) -> Any:
    for ax in flatten_axes(axes):
        if xlabelsize is not None:
            mpl.artist.setp(ax.get_xticklabels(), fontsize=xlabelsize)
        if xrot is not None:
            mpl.artist.setp(ax.get_xticklabels(), rotation=xrot)
        if ylabelsize is not None:
            mpl.artist.setp(ax.get_yticklabels(), fontsize=ylabelsize)
        if yrot is not None:
            mpl.artist.setp(ax.get_yticklabels(), rotation=yrot)
    return axes


# The plot classes, from pandas' core, hist and boxplot modules.


class MPLPlot:
    """What every plot kind shares: the axes, the colours, the labels and the legend."""

    _kind = "line"
    _layout_type = "vertical"
    _default_rot = 0
    orientation: str | None = None
    _need_to_set_index = False

    def __init__(
        self,
        data: Any,
        kind: Any = None,
        by: Any = None,
        subplots: Any = False,
        sharex: Any = None,
        sharey: bool = False,
        use_index: bool = True,
        figsize: Any = None,
        grid: Any = None,
        legend: Any = True,
        rot: Any = None,
        ax: Any = None,
        fig: Any = None,
        title: Any = None,
        xlim: Any = None,
        ylim: Any = None,
        xticks: Any = None,
        yticks: Any = None,
        xlabel: Any = None,
        ylabel: Any = None,
        fontsize: Any = None,
        secondary_y: Any = False,
        colormap: Any = None,
        table: Any = False,
        layout: Any = None,
        include_bool: bool = False,
        column: Any = None,
        *,
        logx: Any = False,
        logy: Any = False,
        loglog: Any = False,
        mark_right: bool = True,
        stacked: bool = False,
        label: Any = None,
        style: Any = None,
        **kwds: Any,
    ) -> None:
        if isinstance(by, (list, tuple)) and len(by) == 0:
            raise ValueError("No group keys passed!")
        self.by = _maybe_make_list(by)
        if _frame(data):
            if column:
                self.columns = _maybe_make_list(column)
            elif self.by is None:
                self.columns = [c for c in data.columns if _numeric(data[c].dtype)]
            else:
                self.columns = [
                    c for c in data.columns if c not in self.by and _numeric(data[c].dtype)
                ]
        if self.by is not None and self._kind in ("hist", "box"):
            self._groups = [
                (key, group) for key, group in data.groupby(unpack_single_str_list(self.by))
            ]
        self.kind = kind
        self.subplots = self._validate_subplots_kwarg(subplots, data, kind=self._kind)
        self.sharex = self._validate_sharex(sharex, ax, by)
        self.sharey = sharey
        self.figsize = figsize
        self.layout = layout
        self.xticks = xticks
        self.yticks = yticks
        self.xlim = xlim
        self.ylim = ylim
        self.title = title
        self.use_index = use_index
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fontsize = fontsize
        if rot is not None:
            self.rot = rot
            self._rot_set = True
        else:
            self._rot_set = False
            self.rot = self._default_rot
        if grid is None:
            grid = False if secondary_y else mpl.rcParams["axes.grid"]
        self.grid = grid
        self.legend = legend
        self.legend_handles: list[Any] = []
        self.legend_labels: list[Any] = []
        self.logx = self._validate_log_kwd("logx", logx)
        self.logy = self._validate_log_kwd("logy", logy)
        self.loglog = self._validate_log_kwd("loglog", loglog)
        self.label = label
        self.style = style
        self.mark_right = mark_right
        self.stacked = stacked
        self.ax = ax
        xerr = kwds.pop("xerr", None)
        yerr = kwds.pop("yerr", None)
        nseries = self._get_nseries(data)
        xerr, data = self._parse_errorbars("xerr", xerr, data, nseries)
        yerr, data = self._parse_errorbars("yerr", yerr, data, nseries)
        self.errors = {"xerr": xerr, "yerr": yerr}
        self.data = data
        if not isinstance(secondary_y, (bool, tuple, list, np.ndarray)) and not hasattr(
            secondary_y, "tolist"
        ):
            secondary_y = [secondary_y]
        self.secondary_y = secondary_y
        if "cmap" in kwds and colormap:
            raise TypeError("Only specify one of `cmap` and `colormap`.")
        self.colormap = kwds.pop("cmap") if "cmap" in kwds else colormap
        self.table = table
        self.include_bool = include_bool
        self.kwds = kwds
        color = kwds.pop("color", _NO_DEFAULT)
        self.color = self._validate_color_args(color, self.colormap)
        self.data = self._ensure_frame(self.data)
        self.x_compat = _plotting.plot_params["x_compat"]
        if "x_compat" in self.kwds:
            self.x_compat = bool(self.kwds.pop("x_compat"))
        self._axes_fig: tuple[Any, Any] | None = None

    @staticmethod
    def _validate_sharex(sharex: Any, ax: Any, by: Any) -> bool:
        if sharex is None:
            return ax is None and by is None
        if not isinstance(sharex, (bool, np.bool_)):
            raise TypeError("sharex must be a bool or None")
        return bool(sharex)

    @classmethod
    def _validate_log_kwd(cls, kwd: str, value: Any) -> Any:
        if value is None or isinstance(value, bool) or (isinstance(value, str) and value == "sym"):
            return value
        raise ValueError(f"keyword '{kwd}' should be bool, None, or 'sym', not '{value}'")

    @staticmethod
    def _validate_subplots_kwarg(subplots: Any, data: Any, kind: str) -> Any:
        if isinstance(subplots, bool):
            return subplots
        if not isinstance(subplots, Iterable):
            raise ValueError("subplots should be a bool or an iterable")
        supported_kinds = ("line", "bar", "barh", "hist", "kde", "density", "area", "pie")
        if kind not in supported_kinds:
            raise ValueError(
                "When subplots is an iterable, kind must be "
                f"one of {', '.join(supported_kinds)}. Got {kind}."
            )
        if _series(data):
            raise NotImplementedError("An iterable subplots for a Series is not supported.")
        columns = list(data.columns)
        if _is_multi(data.columns):
            raise NotImplementedError(
                "An iterable subplots for a DataFrame with a MultiIndex column is not supported."
            )
        if len(set(columns)) != len(columns):
            raise NotImplementedError(
                "An iterable subplots for a DataFrame with non-unique column "
                "labels is not supported."
            )
        out = []
        seen: set[Any] = set()
        for group in subplots:
            if not _is_list_like(group):
                raise ValueError(
                    "When subplots is an iterable, each entry "
                    "should be a list/tuple of column names."
                )
            bad = [label for label in group if label not in columns]
            if bad:
                raise ValueError(f"Column label(s) {bad} not found in the DataFrame.")
            unique = set(group)
            duplicates = seen.intersection(unique)
            if duplicates:
                raise ValueError(
                    "Each column should be in only one subplot. "
                    f"Columns {duplicates} were found in multiple subplots."
                )
            seen = seen.union(unique)
            out.append(tuple(columns.index(label) for label in group))
        unseen = [c for c in columns if c not in seen]
        with contextlib.suppress(TypeError):
            unseen = sorted(unseen)
        for label in unseen:
            out.append((columns.index(label),))
        return out

    def _validate_color_args(self, color: Any, colormap: Any) -> Any:
        if color is _NO_DEFAULT:
            if "colors" in self.kwds and colormap is not None:
                warnings.warn(
                    "'color' and 'colormap' cannot be used simultaneously. Using 'color'",
                    stacklevel=5,
                )
            return None
        if self.nseries == 1 and color is not None and not _is_list_like(color):
            color = [color]
        if isinstance(color, tuple) and self.nseries == 1 and len(color) in (3, 4):
            color = [color]
        if colormap is not None:
            warnings.warn(
                "'color' and 'colormap' cannot be used simultaneously. Using 'color'",
                stacklevel=5,
            )
        if self.style is not None:
            if isinstance(self.style, dict):
                styles = [self.style[col] for col in self.columns if col in self.style]
            elif _is_list_like(self.style):
                styles = self.style
            else:
                styles = [self.style]
            for s in styles:
                if _color_in_style(s):
                    raise ValueError(
                        "Cannot pass 'style' string with a color symbol and "
                        "'color' keyword argument. Please use one or the "
                        "other or pass 'style' without a color symbol"
                    )
        return color

    @staticmethod
    def _iter_data(data: Any) -> Iterable[tuple[Any, np.ndarray]]:
        if isinstance(data, dict):
            yield from data.items()
            return
        for i, col in enumerate(data.columns):
            yield col, _values(data.iloc[:, i])

    def _get_nseries(self, data: Any) -> int:
        if data.ndim == 1:
            return 1
        if self.by is not None and self._kind == "hist":
            return len(self._groups)
        if self.by is not None and self._kind == "box":
            return len(self.columns)
        return data.shape[1]

    @property
    def nseries(self) -> int:
        return self._get_nseries(self.data)

    def generate(self) -> None:
        self._compute_plot_data()
        fig = self.fig
        self._make_plot(fig)
        self._add_table()
        self._make_legend()
        self._adorn_subplots(fig)
        for ax in self.axes:
            self._post_plot_logic_common(ax)
            self._post_plot_logic(ax, self.data)

    @staticmethod
    def _has_plotted_object(ax: Any) -> bool:
        return len(ax.lines) != 0 or len(ax.artists) != 0 or len(ax.containers) != 0

    def _maybe_right_yaxis(self, ax: Any, axes_num: int) -> Any:
        if not self.on_right(axes_num):
            return self._get_ax_layer(ax)
        if hasattr(ax, "right_ax"):
            return ax.right_ax
        if hasattr(ax, "left_ax"):
            return ax
        orig_ax, new_ax = ax, ax.twinx()
        new_ax._get_lines = orig_ax._get_lines
        new_ax._get_patches_for_fill = orig_ax._get_patches_for_fill
        orig_ax.right_ax, new_ax.left_ax = new_ax, orig_ax
        if not self._has_plotted_object(orig_ax):
            orig_ax.get_yaxis().set_visible(False)
        if self.logy is True or self.loglog is True:
            new_ax.set_yscale("log")
        elif self.logy == "sym" or self.loglog == "sym":
            new_ax.set_yscale("symlog")
        return new_ax

    @property
    def fig(self) -> Any:
        return self._axes_and_fig()[1]

    @property
    def axes(self) -> Any:
        return self._axes_and_fig()[0]

    def _axes_and_fig(self) -> tuple[Any, Any]:
        if self._axes_fig is not None:
            return self._axes_fig
        import matplotlib.pyplot as plt

        if self.subplots:
            naxes = self.nseries if isinstance(self.subplots, bool) else len(self.subplots)
            fig, axes = create_subplots(
                naxes=naxes,
                sharex=self.sharex,
                sharey=self.sharey,
                figsize=self.figsize,
                ax=self.ax,
                layout=self.layout,
                layout_type=self._layout_type,
            )
        elif self.ax is None:
            fig = plt.figure(figsize=self.figsize)
            axes = fig.add_subplot(111)
        else:
            fig = self.ax.get_figure()
            if self.figsize is not None:
                fig.set_size_inches(self.figsize)
            axes = self.ax
        axes = np.fromiter(flatten_axes(axes), dtype=object)
        if self.logx is True or self.loglog is True:
            [a.set_xscale("log") for a in axes]
        elif self.logx == "sym" or self.loglog == "sym":
            [a.set_xscale("symlog") for a in axes]
        if self.logy is True or self.loglog is True:
            [a.set_yscale("log") for a in axes]
        elif self.logy == "sym" or self.loglog == "sym":
            [a.set_yscale("symlog") for a in axes]
        self._axes_fig = (axes, fig)
        return self._axes_fig

    @property
    def result(self) -> Any:
        if self.subplots:
            if self.layout is not None and not _is_list_like(self.ax):
                return self.axes.reshape(*self.layout)
            return self.axes
        sec_true = isinstance(self.secondary_y, bool) and self.secondary_y
        all_sec = _is_list_like(self.secondary_y) and len(self.secondary_y) == self.nseries
        if sec_true or all_sec:
            return self._get_ax_layer(self.axes[0], primary=False)
        return self.axes[0]

    def _ensure_frame(self, data: Any) -> Any:
        if _series(data):
            label = self.label
            if label is None and data.name is None:
                label = ""
            data = data.to_frame() if label is None else data.to_frame(name=label)
        elif self._kind in ("hist", "box"):
            cols = self.columns if self.by is None else self.columns + self.by
            data = data[cols]
        return data

    def _compute_plot_data(self) -> None:
        data = self.data
        if self.by is not None:
            self.subplots = True
            data = data[self.columns]
        data = data.infer_objects()
        include_type: list[Any] = ["number", "datetime", "datetimetz", "timedelta"]
        if self.include_bool is True:
            include_type.append("bool")
        exclude_type = None
        if self._kind == "box":
            include_type = ["number"]
            exclude_type = ["timedelta"]
        if self._kind == "scatter":
            # Scatter also colours by text and category columns, which Arrow keeps as
            # strings and dictionaries rather than as objects, so every column is kept.
            numeric_data = data
        else:
            numeric_data = data.select_dtypes(include=include_type, exclude=exclude_type)
        if numeric_data.shape[-1] == 0:
            raise TypeError("no numeric data to plot")
        self.data = numeric_data

    def _make_plot(self, fig: Any) -> None:
        raise NotImplementedError

    def _add_table(self) -> None:
        if self.table is False:
            return
        data = self.data.transpose() if self.table is True else self.table
        table(self._get_ax(0), data)

    def _post_plot_logic_common(self, ax: Any) -> None:
        if self.orientation == "vertical" or self.orientation is None:
            self._apply_axis_properties(ax.xaxis, rot=self.rot, fontsize=self.fontsize)
            self._apply_axis_properties(ax.yaxis, fontsize=self.fontsize)
        elif self.orientation == "horizontal":
            self._apply_axis_properties(ax.yaxis, rot=self.rot, fontsize=self.fontsize)
            self._apply_axis_properties(ax.xaxis, fontsize=self.fontsize)
        if hasattr(ax, "right_ax"):
            self._apply_axis_properties(ax.right_ax.yaxis, fontsize=self.fontsize)

    def _post_plot_logic(self, ax: Any, data: Any) -> None:
        pass

    def _adorn_subplots(self, fig: Any) -> None:
        if len(self.axes) > 0:
            all_axes = self._get_subplots(fig)
            nrows, ncols = self._get_axes_layout(fig)
            handle_shared_axes(
                axarr=all_axes,
                nplots=len(all_axes),
                naxes=nrows * ncols,
                nrows=nrows,
                ncols=ncols,
                sharex=self.sharex,
                sharey=self.sharey,
            )
        for ax in self.axes:
            ax = getattr(ax, "right_ax", ax)
            if self.yticks is not None:
                ax.set_yticks(self.yticks)
            if self.xticks is not None:
                ax.set_xticks(self.xticks)
            if self.ylim is not None:
                ax.set_ylim(self.ylim)
            if self.xlim is not None:
                ax.set_xlim(self.xlim)
            if self.ylabel is not None:
                ax.set_ylabel(pprint_thing(self.ylabel))
            ax.grid(self.grid)
        if not self.title:
            return
        if self.subplots:
            if _is_list_like(self.title):
                if not isinstance(self.subplots, bool):
                    if len(self.subplots) != len(self.title):
                        raise ValueError(
                            f"The number of titles ({len(self.title)}) must equal "
                            f"the number of subplots ({len(self.subplots)})."
                        )
                elif len(self.title) != self.nseries:
                    raise ValueError(
                        "The length of `title` must equal the number "
                        "of columns if using `title` of type `list` "
                        "and `subplots=True`.\n"
                        f"length of title = {len(self.title)}\n"
                        f"number of columns = {self.nseries}"
                    )
                for ax, title in zip(self.axes, self.title, strict=False):
                    ax.set_title(title)
            else:
                fig.suptitle(self.title)
        else:
            if _is_list_like(self.title):
                raise ValueError(
                    "Using `title` of type `list` is not supported unless `subplots=True` is passed"
                )
            self.axes[0].set_title(self.title)

    @staticmethod
    def _apply_axis_properties(axis: Any, rot: Any = None, fontsize: Any = None) -> None:
        if rot is not None or fontsize is not None:
            for label in axis.get_majorticklabels() + axis.get_minorticklabels():
                if rot is not None:
                    label.set_rotation(rot)
                if fontsize is not None:
                    label.set_fontsize(fontsize)

    @property
    def legend_title(self) -> str | None:
        columns = self.data.columns
        if not _is_multi(columns):
            name = columns.name
            return None if name is None else pprint_thing(name)
        return ",".join(pprint_thing(name) for name in columns.names)

    def _mark_right_label(self, label: str, index: int) -> str:
        if not self.subplots and self.mark_right and self.on_right(index):
            label += " (right)"
        return label

    def _append_legend_handles_labels(self, handle: Any, label: Any) -> None:
        self.legend_handles.append(handle)
        self.legend_labels.append(label)

    def _make_legend(self) -> None:
        ax, leg = self._get_ax_legend(self.axes[0])
        handles: list[Any] = []
        labels: list[Any] = []
        title = ""
        if not self.subplots:
            if leg is not None:
                title = leg.get_title().get_text()
                handles = list(leg.legend_handles)
                labels = [x.get_text() for x in leg.get_texts()]
            if self.legend:
                if self.legend == "reverse":
                    handles += reversed(self.legend_handles)
                    labels += reversed(self.legend_labels)
                else:
                    handles += self.legend_handles
                    labels += self.legend_labels
                if self.legend_title is not None:
                    title = self.legend_title
            if len(handles) > 0:
                ax.legend(handles, labels, loc="best", title=title)
        elif self.subplots and self.legend:
            for ax in self.axes:
                if ax.get_visible():
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", "No artists with labels found to put in legend.", UserWarning
                        )
                        ax.legend(loc="best")

    @staticmethod
    def _get_ax_legend(ax: Any) -> tuple[Any, Any]:
        leg = ax.get_legend()
        other_ax = getattr(ax, "left_ax", None) or getattr(ax, "right_ax", None)
        other_leg = None if other_ax is None else other_ax.get_legend()
        if leg is None and other_leg is not None:
            leg = other_leg
            ax = other_ax
        return ax, leg

    def _get_xticks(self) -> Any:
        index = self.data.index
        if self.use_index:
            if _real_numeric(index.dtype) or _is_dates(index):
                return _index_values(index)
            self._need_to_set_index = True
        return list(range(len(index)))

    @classmethod
    def _plot(
        cls, ax: Any, x: Any, y: Any, style: Any = None, is_errorbar: bool = False, **kwds: Any
    ) -> Any:
        mask = isna(y)
        if mask.any():
            y = np.ma.masked_where(mask, np.ma.array(y))
        if is_errorbar:
            if "xerr" in kwds:
                kwds["xerr"] = np.array(kwds.get("xerr"))
            if "yerr" in kwds:
                kwds["yerr"] = np.array(kwds.get("yerr"))
            return ax.errorbar(x, y, **kwds)
        args = (x, y, style) if style is not None else (x, y)
        return ax.plot(*args, **kwds)

    def _get_custom_index_name(self) -> Any:
        return self.xlabel

    def _get_index_name(self) -> str | None:
        index = self.data.index
        if _is_multi(index):
            names = list(index.names)
            name = (
                ",".join(pprint_thing(x) for x in names)
                if any(x is not None for x in names)
                else None
            )
        else:
            name = index.name
            if name is not None:
                name = pprint_thing(name)
        index_name = self._get_custom_index_name()
        if index_name is not None:
            name = pprint_thing(index_name)
        return name

    @classmethod
    def _get_ax_layer(cls, ax: Any, primary: bool = True) -> Any:
        return getattr(ax, "left_ax", ax) if primary else getattr(ax, "right_ax", ax)

    def _col_idx_to_axis_idx(self, col_idx: int) -> int:
        if isinstance(self.subplots, list):
            return next(i for i, group in enumerate(self.subplots) if col_idx in group)
        return col_idx

    def _get_ax(self, i: int) -> Any:
        if self.subplots:
            i = self._col_idx_to_axis_idx(i)
            ax = self._maybe_right_yaxis(self.axes[i], i)
            self.axes[i] = ax
        else:
            ax = self._maybe_right_yaxis(self.axes[0], i)
        ax.get_yaxis().set_visible(True)
        return ax

    def on_right(self, i: int) -> bool:
        if isinstance(self.secondary_y, bool):
            return self.secondary_y
        return self.data.columns[i] in list(self.secondary_y)

    def _apply_style_colors(
        self, colors: Any, kwds: dict[str, Any], col_num: int, label: Any
    ) -> tuple[Any, dict[str, Any]]:
        style = None
        if self.style is not None:
            if isinstance(self.style, list):
                if col_num < len(self.style):
                    style = self.style[col_num]
            elif isinstance(self.style, dict):
                style = self.style.get(label, style)
            else:
                style = self.style
        has_color = "color" in kwds or self.colormap is not None
        nocolor_style = style is None or not _color_in_style(style)
        if (has_color or self.subplots) and nocolor_style:
            if isinstance(colors, dict):
                kwds["color"] = colors[label]
            else:
                kwds["color"] = colors[col_num % len(colors)]
        return style, kwds

    def _get_colors(self, num_colors: int | None = None, color_kwds: str = "color") -> Any:
        if num_colors is None:
            num_colors = self.nseries
        color = self.color if color_kwds == "color" else self.kwds.get(color_kwds)
        return get_standard_colors(num_colors=num_colors, colormap=self.colormap, color=color)

    @staticmethod
    def _parse_errorbars(label: str, err: Any, data: Any, nseries: int) -> tuple[Any, Any]:
        if err is None:
            return None, data
        if _frame(err):
            err = err.reindex(data.index)
        elif isinstance(err, dict):
            pass
        elif _series(err):
            err = np.tile(np.atleast_2d(_values(err.reindex(data.index))), (nseries, 1))
        elif isinstance(err, str):
            evalues = _values(data[err])
            data = data[[c for c in data.columns if c != err]]
            err = np.tile(np.atleast_2d(evalues), (nseries, 1))
        elif _is_list_like(err):
            err = np.atleast_2d(list(err) if not hasattr(err, "__len__") else err)
            err_shape = err.shape
            if _series(data) and err_shape[0] == 2:
                err = np.expand_dims(err, 0)
                err_shape = err.shape
                if err_shape[2] != len(data):
                    raise ValueError(
                        "Asymmetrical error bars should be provided "
                        f"with the shape (2, {len(data)})"
                    )
            elif _frame(data) and err.ndim == 3:
                if err_shape[0] != nseries or err_shape[1] != 2 or err_shape[2] != len(data):
                    raise ValueError(
                        "Asymmetrical error bars should be provided "
                        f"with the shape ({nseries}, 2, {len(data)})"
                    )
            if len(err) == 1:
                err = np.tile(err, (nseries, 1))
        elif isinstance(err, (int, float, np.number)) and not isinstance(err, bool):
            err = np.tile([err], (nseries, len(data)))
        else:
            raise ValueError(f"No valid {label} detected")
        return err, data

    def _get_errorbars(
        self, label: Any = None, index: Any = None, xerr: bool = True, yerr: bool = True
    ) -> dict[str, Any]:
        errors = {}
        for kw, flag in (("xerr", xerr), ("yerr", yerr)):
            if not flag:
                continue
            err = self.errors[kw]
            if _frame(err) or isinstance(err, dict):
                keys = list(err.keys()) if isinstance(err, dict) else list(err.columns)
                err = err[label] if label is not None and label in keys else None
                if _series(err):
                    err = _values(err)
            elif index is not None and err is not None:
                err = err[index]
            if err is not None:
                errors[kw] = err
        return errors

    def _get_subplots(self, fig: Any) -> list[Any]:
        from matplotlib.axes import Axes

        return [
            ax for ax in fig.get_axes() if isinstance(ax, Axes) and ax.get_subplotspec() is not None
        ]

    def _get_axes_layout(self, fig: Any) -> tuple[int, int]:
        x_set = set()
        y_set = set()
        for ax in self._get_subplots(fig):
            points = ax.get_position().get_points()
            x_set.add(points[0][0])
            y_set.add(points[0][1])
        return (len(y_set), len(x_set))


def _holds_integer(columns: Any) -> bool:
    return str(columns.dtype).lower().startswith(("int", "uint"))


class PlanePlot(MPLPlot):
    """Scatter and hexbin, which plot one column against another."""

    _layout_type = "single"

    def __init__(self, data: Any, x: Any, y: Any, **kwargs: Any) -> None:
        MPLPlot.__init__(self, data, **kwargs)
        if x is None or y is None:
            raise ValueError(self._kind + " requires an x and y column")
        if _is_integer(x) and not _holds_integer(self.data.columns):
            x = self.data.columns[x]
        if _is_integer(y) and not _holds_integer(self.data.columns):
            y = self.data.columns[y]
        self.x = x
        self.y = y

    def _get_nseries(self, data: Any) -> int:
        return 1

    def _post_plot_logic(self, ax: Any, data: Any) -> None:
        ax.set_xlabel(self.xlabel if self.xlabel is not None else pprint_thing(self.x))
        ax.set_ylabel(self.ylabel if self.ylabel is not None else pprint_thing(self.y))

    def _plot_colorbar(self, ax: Any, *, fig: Any, **kwds: Any) -> Any:
        return fig.colorbar(ax.collections[-1], ax=ax, **kwds)


class ScatterPlot(PlanePlot):
    _kind = "scatter"

    def __init__(
        self,
        data: Any,
        x: Any,
        y: Any,
        s: Any = None,
        c: Any = None,
        *,
        colorbar: Any = _NO_DEFAULT,
        norm: Any = None,
        **kwargs: Any,
    ) -> None:
        if s is None:
            s = 20
        elif _is_hashable(s) and s in list(data.columns):
            s = _values(data[s])
        self.s = s
        self.colorbar = colorbar
        self.norm = norm
        super().__init__(data, x, y, **kwargs)
        if _is_integer(c) and not _holds_integer(self.data.columns):
            c = self.data.columns[c]
        self.c = c

    def _make_plot(self, fig: Any) -> None:
        x, y, c, data = self.x, self.y, self.c, self.data
        ax = self.axes[0]
        c_is_column = _is_hashable(c) and c in list(self.data.columns)
        color_by_categorical = c_is_column and str(self.data[c].dtype) == "category"
        c_values = self._get_c_values(self.color, color_by_categorical, c_is_column)
        norm, cmap = self._get_norm_and_cmap(c_values, color_by_categorical)
        cb = self._get_colorbar(c_values, c_is_column)
        label = self.label if self.legend else None
        if not self._are_valid_colors(c_values):
            color_mapping = self._get_color_mapping(c_values)
            c_values = [color_mapping[s] for s in c_values]
            ax.legend(
                handles=[
                    mpl.patches.Circle((0, 0), facecolor=color, label=s)
                    for s, color in color_mapping.items()
                ]
            )
        scatter = ax.scatter(
            _values(data[x]),
            _values(data[y]),
            c=c_values,
            label=label,
            cmap=cmap,
            norm=norm,
            s=self.s,
            **self.kwds,
        )
        if cb:
            cbar = self._plot_colorbar(ax, fig=fig, label=c if c_is_column else "")
            if color_by_categorical:
                categories = list(self.data[c].cat.categories)
                n_cats = len(categories)
                cbar.set_ticks(np.linspace(0.5, n_cats - 0.5, n_cats))
                cbar.ax.set_yticklabels(categories)
        if label is not None:
            self._append_legend_handles_labels(scatter, label)
        errors_x = self._get_errorbars(label=x, index=0, yerr=False)
        errors_y = self._get_errorbars(label=y, index=0, xerr=False)
        if len(errors_x) > 0 or len(errors_y) > 0:
            err_kwds = dict(errors_x, **errors_y)
            err_kwds["ecolor"] = scatter.get_facecolor()[0]
            ax.errorbar(_values(data[x]), _values(data[y]), linestyle="none", **err_kwds)

    def _get_c_values(self, color: Any, color_by_categorical: bool, c_is_column: bool) -> Any:
        c = self.c
        if c is not None and color is not None:
            raise TypeError("Specify exactly one of `c` and `color`")
        if c is None and color is None:
            return mpl.rcParams["patch.facecolor"]
        if color is not None:
            return color
        if color_by_categorical:
            return _values(self.data[c].cat.codes)
        if c_is_column:
            return _values(self.data[c])
        return c

    def _are_valid_colors(self, c_values: Any) -> bool:
        unique = np.unique(c_values)
        try:
            if len(c_values) and all(isinstance(c, str) for c in unique):
                mpl.colors.to_rgba_array(unique)
            return True
        except (TypeError, ValueError):
            return False

    def _get_color_mapping(self, c_values: Any) -> dict[Any, Any]:
        unique = np.unique(c_values)
        cmap = mpl.colormaps.get_cmap(self.colormap)
        colors = cmap(np.linspace(0, 1, len(unique)))
        return dict(zip(unique, colors, strict=True))

    def _get_norm_and_cmap(self, c_values: Any, color_by_categorical: bool) -> tuple[Any, Any]:
        if self.colormap is not None:
            cmap = mpl.colormaps.get_cmap(self.colormap)
        elif isinstance(c_values, np.ndarray) and c_values.dtype.kind in "iu":
            cmap = mpl.colormaps["Greys"]
        else:
            cmap = None
        if color_by_categorical and cmap is not None:
            n_cats = len(self.data[self.c].cat.categories)
            cmap = mpl.colors.ListedColormap([cmap(i) for i in range(cmap.N)])
            bounds = np.linspace(0, n_cats, n_cats + 1)
            return mpl.colors.BoundaryNorm(bounds, cmap.N), cmap
        return self.norm, cmap

    def _get_colorbar(self, c_values: Any, c_is_column: bool) -> Any:
        plot_colorbar = self.colormap or c_is_column
        if self.colorbar is _NO_DEFAULT:
            numeric = isinstance(c_values, np.ndarray) and c_values.dtype.kind in "iufcb"
            return numeric and plot_colorbar
        return self.colorbar


class HexBinPlot(PlanePlot):
    _kind = "hexbin"

    def __init__(
        self, data: Any, x: Any, y: Any, C: Any = None, *, colorbar: bool = True, **kwargs: Any
    ) -> None:
        super().__init__(data, x, y, **kwargs)
        if _is_integer(C) and not _holds_integer(self.data.columns):
            C = self.data.columns[C]
        self.C = C
        self.colorbar = colorbar
        for axis, name in (("x", self.x), ("y", self.y)):
            column = self.data[name]
            if not _numeric(column.dtype) or len(column) == 0:
                raise ValueError(f"{self._kind} requires {axis} column to be numeric")

    def _make_plot(self, fig: Any) -> None:
        x, y, data, C = self.x, self.y, self.data, self.C
        ax = self.axes[0]
        cmap = mpl.colormaps.get_cmap(self.colormap or "BuGn")
        c_values = None if C is None else _values(data[C])
        ax.hexbin(_values(data[x]), _values(data[y]), C=c_values, cmap=cmap, **self.kwds)
        if self.colorbar:
            self._plot_colorbar(ax, fig=fig)

    def _make_legend(self) -> None:
        pass


class LinePlot(MPLPlot):
    _kind = "line"
    _default_rot = 0
    orientation = "vertical"

    def __init__(self, data: Any, **kwargs: Any) -> None:
        MPLPlot.__init__(self, data, **kwargs)
        if self.stacked:
            self.data = self.data.fillna(value=0)

    def _make_plot(self, fig: Any) -> None:
        x = self._get_xticks()
        stacking_id = self._get_stacking_id()
        is_errorbar = any(e is not None for e in self.errors.values())
        colors = self._get_colors()
        for i, (label, y) in enumerate(self._iter_data(data=self.data)):
            ax = self._get_ax(i)
            kwds = self.kwds.copy()
            if self.color is not None:
                kwds["color"] = self.color
            style, kwds = self._apply_style_colors(colors, kwds, i, label)
            errors = self._get_errorbars(label=label, index=i)
            kwds = dict(kwds, **errors)
            label = pprint_thing(label)
            label = self._mark_right_label(label, index=i)
            kwds["label"] = label
            newlines = self._plot(
                ax,
                x,
                y,
                style=style,
                column_num=i,
                stacking_id=stacking_id,
                is_errorbar=is_errorbar,
                **kwds,
            )
            self._append_legend_handles_labels(newlines[0], label)

    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Any,
        x: Any,
        y: Any,
        style: Any = None,
        column_num: Any = None,
        stacking_id: Any = None,
        **kwds: Any,
    ) -> Any:
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(y))
        y_values = cls._get_stacked_values(ax, stacking_id, y, kwds["label"])
        lines = MPLPlot._plot(ax, x, y_values, style=style, **kwds)
        cls._update_stacker(ax, stacking_id, y)
        return lines

    def _get_stacking_id(self) -> int | None:
        return id(self.data) if self.stacked else None

    @classmethod
    def _initialize_stacker(cls, ax: Any, stacking_id: Any, n: int) -> None:
        if stacking_id is None:
            return
        if not hasattr(ax, "_stacker_pos_prior"):
            ax._stacker_pos_prior = {}
        if not hasattr(ax, "_stacker_neg_prior"):
            ax._stacker_neg_prior = {}
        ax._stacker_pos_prior[stacking_id] = np.zeros(n)
        ax._stacker_neg_prior[stacking_id] = np.zeros(n)

    @classmethod
    def _get_stacked_values(cls, ax: Any, stacking_id: Any, values: Any, label: Any) -> Any:
        if stacking_id is None:
            return values
        if not hasattr(ax, "_stacker_pos_prior"):
            cls._initialize_stacker(ax, stacking_id, len(values))
        if (values >= 0).all():
            return ax._stacker_pos_prior[stacking_id] + values
        if (values <= 0).all():
            return ax._stacker_neg_prior[stacking_id] + values
        raise ValueError(
            "When stacked is True, each column must be either "
            "all positive or all negative. "
            f"Column '{label}' contains both positive and negative values"
        )

    @classmethod
    def _update_stacker(cls, ax: Any, stacking_id: Any, values: Any) -> None:
        if stacking_id is None:
            return
        if (values >= 0).all():
            ax._stacker_pos_prior[stacking_id] += values
        elif (values <= 0).all():
            ax._stacker_neg_prior[stacking_id] += values

    def _post_plot_logic(self, ax: Any, data: Any) -> None:
        index = list(data.index)

        def get_label(i: Any) -> str:
            if isinstance(i, (float, np.floating)) and float(i).is_integer():
                i = int(i)
            try:
                return pprint_thing(index[i])
            except (IndexError, TypeError):
                return ""

        if self._need_to_set_index:
            xticks = ax.get_xticks()
            xticklabels = [get_label(x) for x in xticks]
            ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks))
            ax.set_xticklabels(xticklabels)
        condition = (
            _is_dates(data.index)
            and self.use_index
            and (not self.subplots or (self.subplots and self.sharex))
        )
        index_name = self._get_index_name()
        if condition:
            if not self._rot_set:
                self.rot = 30
            format_date_labels(ax, rot=self.rot)
        if index_name is not None and self.use_index:
            ax.set_xlabel(index_name)


class AreaPlot(LinePlot):
    _kind = "area"

    def __init__(self, data: Any, **kwargs: Any) -> None:
        kwargs.setdefault("stacked", True)
        data = data.fillna(value=0)
        LinePlot.__init__(self, data, **kwargs)
        if not self.stacked:
            self.kwds.setdefault("alpha", 0.5)
        if self.logy or self.loglog:
            raise ValueError("Log-y scales are not supported in area plot")

    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Any,
        x: Any,
        y: Any,
        style: Any = None,
        column_num: Any = None,
        stacking_id: Any = None,
        is_errorbar: bool = False,
        **kwds: Any,
    ) -> Any:
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(y))
        y_values = cls._get_stacked_values(ax, stacking_id, y, kwds["label"])
        line_kwds = kwds.copy()
        line_kwds.pop("label")
        lines = MPLPlot._plot(ax, x, y_values, style=style, **line_kwds)
        xdata, y_values = lines[0].get_data(orig=False)
        if stacking_id is None:
            start = np.zeros(len(y))
        elif (y >= 0).all():
            start = ax._stacker_pos_prior[stacking_id]
        elif (y <= 0).all():
            start = ax._stacker_neg_prior[stacking_id]
        else:
            start = np.zeros(len(y))
        if "color" not in kwds:
            kwds["color"] = lines[0].get_color()
        rect = ax.fill_between(xdata, start, y_values, **kwds)
        cls._update_stacker(ax, stacking_id, y)
        return [rect]

    def _post_plot_logic(self, ax: Any, data: Any) -> None:
        LinePlot._post_plot_logic(self, ax, data)
        is_shared_y = len(list(ax.get_shared_y_axes())) > 0
        if self.ylim is None and not is_shared_y:
            values = np.column_stack([v for _, v in self._iter_data(data)]).astype(float)
            if (values >= 0).all():
                ax.set_ylim(0, None)
            elif (values <= 0).all():
                ax.set_ylim(None, 0)


class BarPlot(MPLPlot):
    _kind = "bar"
    _default_rot = 90
    orientation = "vertical"

    def __init__(
        self,
        data: Any,
        *,
        align: str = "center",
        bottom: Any = 0,
        left: Any = 0,
        width: float = 0.5,
        position: float = 0.5,
        log: bool = False,
        **kwargs: Any,
    ) -> None:
        self._is_series = _series(data)
        self.bar_width = width
        self._align = align
        self._position = position
        self.bottom = np.array(bottom) if _is_list_like(bottom) else bottom
        self.left = np.array(left) if _is_list_like(left) else left
        self.log = log
        MPLPlot.__init__(self, data, **kwargs)
        self.tick_pos = np.arange(len(data))

    @property
    def ax_pos(self) -> Any:
        return self.tick_pos - self.tickoffset

    @property
    def tickoffset(self) -> float:
        if self.stacked or self.subplots:
            return self.bar_width * self._position
        if self._align == "edge":
            w = self.bar_width / self.nseries
            return self.bar_width * (self._position - 0.5) + w * 0.5
        return self.bar_width * self._position

    @property
    def lim_offset(self) -> float:
        if self.stacked or self.subplots:
            return self.bar_width / 2 if self._align == "edge" else 0
        if self._align == "edge":
            return self.bar_width / self.nseries * 0.5
        return 0

    @classmethod
    def _plot(  # type: ignore[override]
        cls, ax: Any, x: Any, y: Any, w: Any, start: Any = 0, log: bool = False, **kwds: Any
    ) -> Any:
        return ax.bar(x, y, w, bottom=start, log=log, **kwds)

    @property
    def _start_base(self) -> Any:
        return self.bottom

    def _make_plot(self, fig: Any) -> None:
        colors = self._get_colors()
        ncolors = len(colors)
        pos_prior = neg_prior = np.zeros(len(self.data))
        K = self.nseries
        data = self.data.fillna(0)
        stacked_ind: dict[int, int] = {}
        stacked_offsets: list[tuple[Any, Any]] = []
        if not isinstance(self.subplots, bool) and bool(self.subplots) and self.stacked:
            for i, sub_plot in enumerate(self.subplots):
                if len(sub_plot) <= 1:
                    continue
                for plot in sub_plot:
                    stacked_ind[int(plot)] = i
                stacked_offsets.append((pos_prior, neg_prior))
        for i, (label, y) in enumerate(self._iter_data(data=data)):
            ax = self._get_ax(i)
            kwds = self.kwds.copy()
            if self._is_series:
                kwds["color"] = colors
            elif isinstance(colors, dict):
                kwds["color"] = colors[label]
            else:
                kwds["color"] = colors[i % ncolors]
            errors = self._get_errorbars(label=label, index=i)
            kwds = dict(kwds, **errors)
            label = pprint_thing(label)
            label = self._mark_right_label(label, index=i)
            if ("yerr" in kwds or "xerr" in kwds) and kwds.get("ecolor") is None:
                kwds["ecolor"] = mpl.rcParams["xtick.color"]
            start: Any = 0
            if self.log and (y >= 1).all():
                start = 1
            start = start + self._start_base
            kwds["align"] = self._align
            if i in stacked_ind:
                offset_index = stacked_ind[i]
                pos_prior, neg_prior = stacked_offsets[offset_index]
                mask = y >= 0
                start = np.where(mask, pos_prior, neg_prior) + self._start_base
                w = self.bar_width / 2
                rect = self._plot(
                    ax,
                    self.ax_pos + w,
                    y,
                    self.bar_width,
                    start=start,
                    label=label,
                    log=self.log,
                    **kwds,
                )
                pos_new = pos_prior + np.where(mask, y, 0)
                neg_new = neg_prior + np.where(mask, 0, y)
                stacked_offsets[offset_index] = (pos_new, neg_new)
            elif self.subplots:
                w = self.bar_width / 2
                rect = self._plot(
                    ax,
                    self.ax_pos + w,
                    y,
                    self.bar_width,
                    start=start,
                    label=label,
                    log=self.log,
                    **kwds,
                )
                ax.set_title(label)
            elif self.stacked:
                mask = y >= 0
                start = np.where(mask, pos_prior, neg_prior) + self._start_base
                w = self.bar_width / 2
                rect = self._plot(
                    ax,
                    self.ax_pos + w,
                    y,
                    self.bar_width,
                    start=start,
                    label=label,
                    log=self.log,
                    **kwds,
                )
                pos_prior = pos_prior + np.where(mask, y, 0)
                neg_prior = neg_prior + np.where(mask, 0, y)
            else:
                w = self.bar_width / K
                rect = self._plot(
                    ax,
                    self.ax_pos + (i + 0.5) * w,
                    y,
                    w,
                    start=start,
                    label=label,
                    log=self.log,
                    **kwds,
                )
            self._append_legend_handles_labels(rect, label)

    def _post_plot_logic(self, ax: Any, data: Any) -> None:
        if self.use_index:
            str_index = [pprint_thing(key) for key in data.index]
        else:
            str_index = [pprint_thing(key) for key in range(data.shape[0])]
        s_edge = self.ax_pos[0] - 0.25 + self.lim_offset
        e_edge = self.ax_pos[-1] + 0.25 + self.bar_width + self.lim_offset
        self._decorate_ticks(ax, self._get_index_name(), str_index, s_edge, e_edge)

    def _decorate_ticks(
        self, ax: Any, name: Any, ticklabels: list[str], start_edge: Any, end_edge: Any
    ) -> None:
        ax.set_xlim((start_edge, end_edge))
        if self.xticks is not None:
            ax.set_xticks(np.array(self.xticks))
        else:
            ax.set_xticks(self.tick_pos)
            ax.set_xticklabels(ticklabels)
        if name is not None and self.use_index:
            ax.set_xlabel(name)


class BarhPlot(BarPlot):
    _kind = "barh"
    _default_rot = 0
    orientation = "horizontal"

    @property
    def _start_base(self) -> Any:
        return self.left

    @classmethod
    def _plot(  # type: ignore[override]
        cls, ax: Any, x: Any, y: Any, w: Any, start: Any = 0, log: bool = False, **kwds: Any
    ) -> Any:
        return ax.barh(x, y, w, left=start, log=log, **kwds)

    def _get_custom_index_name(self) -> Any:
        return self.ylabel

    def _decorate_ticks(
        self, ax: Any, name: Any, ticklabels: list[str], start_edge: Any, end_edge: Any
    ) -> None:
        ax.set_ylim((start_edge, end_edge))
        ax.set_yticks(self.tick_pos)
        ax.set_yticklabels(ticklabels)
        if name is not None and self.use_index:
            ax.set_ylabel(name)
        ax.set_xlabel(self.xlabel)


class PiePlot(MPLPlot):
    _kind = "pie"
    _layout_type = "horizontal"

    def __init__(self, data: Any, kind: Any = None, **kwargs: Any) -> None:
        data = data.fillna(value=0)
        values = [_values(data)] if _series(data) else [v for _, v in self._iter_data(data)]
        if any((np.asarray(v, dtype=float) < 0).any() for v in values):
            raise ValueError(f"{self._kind} plot doesn't allow negative values")
        MPLPlot.__init__(self, data, kind=kind, **kwargs)

    @classmethod
    def _validate_log_kwd(cls, kwd: str, value: Any) -> Any:
        super()._validate_log_kwd(kwd=kwd, value=value)
        if value is not False:
            warnings.warn(f"PiePlot ignores the '{kwd}' keyword", UserWarning, stacklevel=5)
        return False

    def _validate_color_args(self, color: Any, colormap: Any) -> None:
        return None

    def _make_plot(self, fig: Any) -> None:
        colors = self._get_colors(num_colors=len(self.data), color_kwds="colors")
        self.kwds.setdefault("colors", colors)
        for i, (_, y) in enumerate(self._iter_data(data=self.data)):
            ax = self._get_ax(i)
            kwds = self.kwds.copy()
            idx = [pprint_thing(v) for v in self.data.index]
            labels = kwds.pop("labels", idx)
            if labels is not None:
                blabels = [
                    "" if value == 0 else left for left, value in zip(labels, y, strict=True)
                ]
            else:
                blabels = None
            results = ax.pie(y, labels=blabels, **kwds)
            if kwds.get("autopct", None) is not None:
                patches, texts, autotexts = results
            else:
                patches, texts = results
                autotexts = []
            if self.fontsize is not None:
                for t in list(texts) + list(autotexts):
                    t.set_fontsize(self.fontsize)
            leglabels = labels if labels is not None else idx
            for patch, leglabel in zip(patches, leglabels, strict=True):
                self._append_legend_handles_labels(patch, leglabel)


def _by_hist_data(plot: MPLPlot) -> dict[Any, np.ndarray]:
    """Each group's columns, one row per value, as pandas' plot by a column iterates them."""
    out = {}
    for key, group in plot._groups:
        columns = [remove_na_arraylike(_values(group[col])) for col in plot.columns]
        out[key] = np.array(columns).T
    return out


def _by_box_data(plot: MPLPlot) -> dict[Any, list[np.ndarray]]:
    """Each column's values in every group, as pandas' box plot by a column iterates them."""
    return {col: [_values(group[col]) for _, group in plot._groups] for col in plot.columns}


class HistPlot(LinePlot):
    _kind = "hist"

    def __init__(
        self,
        data: Any,
        bins: Any = 10,
        bottom: Any = 0,
        *,
        range: Any = None,
        weights: Any = None,
        **kwargs: Any,
    ) -> None:
        self.bottom = np.array(bottom) if _is_list_like(bottom) else bottom
        self._bin_range = range
        self.weights = weights
        self.xlabel = kwargs.get("xlabel")
        self.ylabel = kwargs.get("ylabel")
        MPLPlot.__init__(self, data, **kwargs)
        self.bins = self._adjust_bins(bins)

    def _adjust_bins(self, bins: Any) -> Any:
        if _is_integer(bins):
            if self.by is not None:
                return [
                    self._calculate_bins(group[self.columns], bins) for _, group in self._groups
                ]
            return self._calculate_bins(self.data, bins)
        return bins

    def _calculate_bins(self, data: Any, bins: Any) -> Any:
        if _series(data):
            data = data.to_frame()
        data = data.infer_objects()
        numeric = [col for col in data.columns if _numeric(data[col].dtype)]
        if numeric:
            values = np.concatenate(
                [np.asarray(_values(data[col]), dtype=float) for col in numeric]
            )
        else:
            values = np.array([], dtype=float)
        values = values[~isna(values)]
        return np.histogram_bin_edges(values, bins=bins, range=self._bin_range)

    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Any,
        y: Any,
        style: Any = None,
        bottom: Any = 0,
        column_num: int = 0,
        stacking_id: Any = None,
        *,
        bins: Any,
        **kwds: Any,
    ) -> Any:
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(bins) - 1)
        base = np.zeros(len(bins) - 1)
        bottom = bottom + cls._get_stacked_values(ax, stacking_id, base, kwds["label"])
        n, bins, patches = ax.hist(y, bins=bins, bottom=bottom, **kwds)
        cls._update_stacker(ax, stacking_id, n)
        return patches

    def _make_plot(self, fig: Any) -> None:
        colors = self._get_colors()
        stacking_id = self._get_stacking_id()
        data: Any = _by_hist_data(self) if self.by is not None else self.data
        for i, (label, y) in enumerate(self._iter_data(data=data)):
            ax = self._get_ax(i)
            kwds = self.kwds.copy()
            if self.color is not None:
                kwds["color"] = self.color
            label = pprint_thing(label)
            label = self._mark_right_label(label, index=i)
            kwds["label"] = label
            style, kwds = self._apply_style_colors(colors, kwds, i, label)
            if style is not None:
                kwds["style"] = style
            self._make_plot_keywords(kwds, y)
            if self.by is not None:
                kwds["bins"] = kwds["bins"][i]
                kwds["label"] = self.columns
                kwds.pop("color")
            if self.weights is not None:
                kwds["weights"] = self._get_column_weights(self.weights, i, y)
            if self.by is None:
                y = remove_na_arraylike(y)
            artists = self._plot(ax, y, column_num=i, stacking_id=stacking_id, **kwds)
            if self.by is not None:
                ax.set_title(pprint_thing(label))
            self._append_legend_handles_labels(artists[0], label)

    def _make_plot_keywords(self, kwds: dict[str, Any], y: Any) -> None:
        kwds["bottom"] = self.bottom
        kwds["bins"] = self.bins

    @staticmethod
    def _get_column_weights(weights: Any, i: int, y: Any) -> Any:
        if weights is not None:
            if np.ndim(weights) != 1 and np.shape(weights)[-1] != 1:
                try:
                    weights = weights[:, i]
                except IndexError as err:
                    raise ValueError(
                        "weights must have the same shape as data, or be a single column"
                    ) from err
            weights = weights[~isna(y)]
        return weights

    def _post_plot_logic(self, ax: Any, data: Any) -> None:
        if self.orientation == "horizontal":
            ax.set_xlabel("Frequency" if self.xlabel is None else self.xlabel)
            ax.set_ylabel(self.ylabel)
        else:
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel("Frequency" if self.ylabel is None else self.ylabel)

    @property  # type: ignore[override]
    def orientation(self) -> str:
        if self.kwds.get("orientation", None) == "horizontal":
            return "horizontal"
        return "vertical"


class KdePlot(HistPlot):
    _kind = "kde"
    orientation = "vertical"  # type: ignore[assignment]

    def __init__(
        self,
        data: Any,
        bw_method: Any = None,
        ind: Any = None,
        *,
        weights: Any = None,
        **kwargs: Any,
    ) -> None:
        MPLPlot.__init__(self, data, **kwargs)
        self.bw_method = bw_method
        self.ind = ind
        self.weights = weights

    @staticmethod
    def _get_ind(y: Any, ind: Any) -> Any:
        if ind is None or _is_integer(ind):
            sample_range = np.nanmax(y) - np.nanmin(y)
            ind = np.linspace(
                np.nanmin(y) - 0.5 * sample_range,
                np.nanmax(y) + 0.5 * sample_range,
                1000 if ind is None else ind,
            )
        return ind

    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Any,
        y: Any,
        style: Any = None,
        bw_method: Any = None,
        weights: Any = None,
        ind: Any = None,
        column_num: Any = None,
        stacking_id: Any = None,
        **kwds: Any,
    ) -> Any:
        from scipy.stats import gaussian_kde

        y = remove_na_arraylike(y)
        gkde = gaussian_kde(y, bw_method=bw_method, weights=weights)
        return MPLPlot._plot(ax, ind, gkde.evaluate(ind), style=style, **kwds)

    def _make_plot_keywords(self, kwds: dict[str, Any], y: Any) -> None:
        kwds["bw_method"] = self.bw_method
        kwds["ind"] = self._get_ind(y, ind=self.ind)

    def _post_plot_logic(self, ax: Any, data: Any) -> None:
        ax.set_ylabel("Density")


def _set_ticklabels(ax: Any, labels: list[str], is_vertical: bool, **kwargs: Any) -> None:
    ticks = ax.get_xticks() if is_vertical else ax.get_yticks()
    if len(ticks) != len(labels):
        i, _ = divmod(len(ticks), len(labels))
        labels *= i
    if is_vertical:
        ax.set_xticklabels(labels, **kwargs)
    else:
        ax.set_yticklabels(labels, **kwargs)


class BoxPlot(LinePlot):
    _kind = "box"
    _layout_type = "horizontal"
    _valid_return_types = (None, "axes", "dict", "both")

    class BP(NamedTuple):
        ax: Any
        lines: dict[str, list[Any]]

    def __init__(self, data: Any, return_type: str = "axes", **kwargs: Any) -> None:
        if return_type not in self._valid_return_types:
            raise ValueError("return_type must be {None, 'axes', 'dict', 'both'}")
        self.return_type = return_type
        MPLPlot.__init__(self, data, **kwargs)
        if self.subplots:
            if self.orientation == "vertical":
                self.sharex = False
            else:
                self.sharey = False

    @classmethod
    def _plot(  # type: ignore[override]
        cls, ax: Any, y: Any, column_num: Any = None, return_type: str = "axes", **kwds: Any
    ) -> Any:
        if isinstance(y, list) or np.ndim(y) == 2:
            ys = [remove_na_arraylike(v) for v in y]
            ys = [v if v.size > 0 else np.array([np.nan]) for v in ys]
        else:
            ys = remove_na_arraylike(y)
        bp = ax.boxplot(ys, **kwds)
        if return_type == "dict":
            return bp, bp
        if return_type == "both":
            return cls.BP(ax=ax, lines=bp), bp
        return ax, bp

    def _validate_color_args(self, color: Any, colormap: Any) -> Any:
        if color is _NO_DEFAULT:
            return None
        if colormap is not None:
            warnings.warn(
                "'color' and 'colormap' cannot be used simultaneously. Using 'color'",
                stacklevel=5,
            )
        if isinstance(color, dict):
            valid_keys = ["boxes", "whiskers", "medians", "caps"]
            for key in color:
                if key not in valid_keys:
                    raise ValueError(
                        f"color dict contains invalid key '{key}'. "
                        f"The key must be either {valid_keys}"
                    )
        return color

    def _get_colors(self, num_colors: Any = None, color_kwds: Any = "color") -> None:
        pass

    def maybe_color_bp(self, bp: Any) -> None:
        attrs = get_standard_colors(num_colors=3, colormap=self.colormap, color=None)
        defaults = (attrs[0], attrs[0], attrs[2], attrs[0])
        if isinstance(self.color, dict):
            keys = ("boxes", "whiskers", "medians", "caps")
            color_tup = tuple(self.color.get(k, d) for k, d in zip(keys, defaults, strict=True))
        else:
            color_tup = tuple(self.color or d for d in defaults)
        maybe_color_bp(bp, color_tup=color_tup, **self.kwds)

    def _make_plot(self, fig: Any) -> None:
        if self.subplots:
            obj_axes = []
            obj_labels = []
            data: Any = _by_box_data(self) if self.by is not None else self.data
            for i, (label, y) in enumerate(self._iter_data(data=data)):
                ax = self._get_ax(i)
                kwds = self.kwds.copy()
                if self.by is not None:
                    ax.set_title(pprint_thing(label))
                    ticklabels = [pprint_thing(key) for key, _ in self._groups]
                else:
                    ticklabels = [pprint_thing(label)]
                ret, bp = self._plot(ax, y, column_num=i, return_type=self.return_type, **kwds)
                self.maybe_color_bp(bp)
                obj_axes.append(ret)
                obj_labels.append(label)
                _set_ticklabels(
                    ax=ax, labels=ticklabels, is_vertical=self.orientation == "vertical"
                )
            from ._frame import Series

            self._return_obj = Series(obj_axes, index=obj_labels, dtype="object")
        else:
            y = [v for _, v in self._iter_data(self.data)]
            ax = self._get_ax(0)
            kwds = self.kwds.copy()
            ret, bp = self._plot(ax, y, column_num=0, return_type=self.return_type, **kwds)
            self.maybe_color_bp(bp)
            self._return_obj = ret
            labels = [pprint_thing(left) for left in self.data.columns]
            if not self.use_index:
                labels = [pprint_thing(key) for key in range(len(labels))]
            _set_ticklabels(ax=ax, labels=labels, is_vertical=self.orientation == "vertical")

    def _make_legend(self) -> None:
        pass

    def _post_plot_logic(self, ax: Any, data: Any) -> None:
        if self.xlabel:
            ax.set_xlabel(pprint_thing(self.xlabel))
        if self.ylabel:
            ax.set_ylabel(pprint_thing(self.ylabel))

    @property  # type: ignore[override]
    def orientation(self) -> str:
        return "vertical" if self.kwds.get("vert", True) else "horizontal"

    @property
    def result(self) -> Any:
        if self.return_type is None:
            return super().result
        return self._return_obj


def maybe_color_bp(bp: Any, color_tup: Any, **kwds: Any) -> None:
    if not kwds.get("boxprops"):
        mpl.artist.setp(bp["boxes"], color=color_tup[0], alpha=1)
    if not kwds.get("whiskerprops"):
        mpl.artist.setp(bp["whiskers"], color=color_tup[1], alpha=1)
    if not kwds.get("medianprops"):
        mpl.artist.setp(bp["medians"], color=color_tup[2], alpha=1)
    if not kwds.get("capprops"):
        mpl.artist.setp(bp["caps"], color=color_tup[3], alpha=1)


PLOT_CLASSES: dict[str, type[MPLPlot]] = {
    "line": LinePlot,
    "bar": BarPlot,
    "barh": BarhPlot,
    "kde": KdePlot,
    "hist": HistPlot,
    "box": BoxPlot,
    "area": AreaPlot,
    "pie": PiePlot,
    "scatter": ScatterPlot,
    "hexbin": HexBinPlot,
}


def plot(data: Any, kind: str, **kwargs: Any) -> Any:
    """Draw `data` as a plot of `kind`, the backend's entry point, as pandas' backend has it."""
    import matplotlib.pyplot as plt

    if kwargs.pop("reuse_plot", False):
        ax = kwargs.get("ax")
        if ax is None and len(plt.get_fignums()) > 0:
            with plt.rc_context():
                ax = plt.gca()
            kwargs["ax"] = getattr(ax, "left_ax", ax)
    plot_obj = PLOT_CLASSES[kind](data, **kwargs)
    plot_obj.generate()
    plt.draw_if_interactive()
    return plot_obj.result


# Histograms and box plots drawn outside the plot classes.


def _grouped_plot(
    plotf: Any,
    data: Any,
    column: Any = None,
    by: Any = None,
    numeric_only: bool = True,
    figsize: Any = None,
    sharex: bool = True,
    sharey: bool = True,
    layout: Any = None,
    rot: float = 0,
    ax: Any = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    if figsize == "default":
        raise ValueError(
            "figsize='default' is no longer supported. Specify figure size by tuple instead"
        )
    grouped = data.groupby(by)
    if column is not None:
        grouped = grouped[column]
    groups = list(grouped)
    fig, axes = create_subplots(
        naxes=len(groups), figsize=figsize, sharex=sharex, sharey=sharey, ax=ax, layout=layout
    )
    for ax, (key, group) in zip(flatten_axes(axes), groups, strict=False):
        if numeric_only and _frame(group):
            group = group[_numeric_columns(group)]
        plotf(group, ax, **kwargs)
        ax.set_title(pprint_thing(key))
    return fig, axes


def _grouped_hist(
    data: Any,
    column: Any = None,
    by: Any = None,
    ax: Any = None,
    bins: Any = 50,
    figsize: Any = None,
    layout: Any = None,
    sharex: bool = False,
    sharey: bool = False,
    rot: float = 90,
    grid: bool = True,
    xlabelsize: Any = None,
    xrot: Any = None,
    ylabelsize: Any = None,
    yrot: Any = None,
    legend: bool = False,
    **kwargs: Any,
) -> Any:
    if legend:
        if data.ndim == 1:
            kwargs["label"] = data.name
        elif column is None:
            kwargs["label"] = list(data.columns)
        else:
            kwargs["label"] = column

    def plot_group(group: Any, ax: Any) -> None:
        group = group.dropna()
        if _series(group):
            values = _values(group)
        else:
            values = np.column_stack([_values(group[c]) for c in group.columns])
        ax.hist(values, bins=bins, **kwargs)
        if legend:
            ax.legend()

    if xrot is None:
        xrot = rot
    fig, axes = _grouped_plot(
        plot_group,
        data,
        column=column,
        by=by,
        sharex=sharex,
        sharey=sharey,
        ax=ax,
        figsize=figsize,
        layout=layout,
        rot=rot,
    )
    set_ticks_props(axes, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot)
    maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, hspace=0.5, wspace=0.3)
    return axes


def hist_series(
    self: Any,
    by: Any = None,
    ax: Any = None,
    grid: bool = True,
    xlabelsize: Any = None,
    xrot: Any = None,
    ylabelsize: Any = None,
    yrot: Any = None,
    figsize: Any = None,
    bins: Any = 10,
    legend: bool = False,
    **kwds: Any,
) -> Any:
    import matplotlib.pyplot as plt

    if legend and "label" in kwds:
        raise ValueError("Cannot use both legend and label")
    if by is None:
        if kwds.get("layout") is not None:
            raise ValueError("The 'layout' keyword is not supported when 'by' is None")
        fig = kwds.pop("figure", plt.gcf() if plt.get_fignums() else plt.figure(figsize=figsize))
        if figsize is not None and tuple(figsize) != tuple(fig.get_size_inches()):
            fig.set_size_inches(*figsize, forward=True)
        if ax is None:
            ax = fig.gca()
        elif ax.get_figure() != fig:
            raise AssertionError("passed axis not bound to passed figure")
        values = _values(self.dropna())
        if legend:
            kwds["label"] = self.name
        ax.hist(values, bins=bins, **kwds)
        if legend:
            ax.legend()
        ax.grid(grid)
        axes = np.array([ax])
        set_ticks_props(axes, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot)
    else:
        if "figure" in kwds:
            raise ValueError(
                "Cannot pass 'figure' when using the "
                "'by' argument, since a new 'Figure' instance will be created"
            )
        axes = _grouped_hist(
            self,
            by=by,
            ax=ax,
            grid=grid,
            figsize=figsize,
            bins=bins,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            legend=legend,
            **kwds,
        )
    if hasattr(axes, "ndim") and axes.ndim == 1 and len(axes) == 1:
        return axes[0]
    return axes


def hist_frame(
    data: Any,
    column: Any = None,
    by: Any = None,
    grid: bool = True,
    xlabelsize: Any = None,
    xrot: Any = None,
    ylabelsize: Any = None,
    yrot: Any = None,
    ax: Any = None,
    sharex: bool = False,
    sharey: bool = False,
    figsize: Any = None,
    layout: Any = None,
    bins: Any = 10,
    legend: bool = False,
    **kwds: Any,
) -> Any:
    if legend and "label" in kwds:
        raise ValueError("Cannot use both legend and label")
    if by is not None:
        return _grouped_hist(
            data,
            column=column,
            by=by,
            ax=ax,
            grid=grid,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            layout=layout,
            bins=bins,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            legend=legend,
            **kwds,
        )
    if column is not None:
        if not isinstance(column, (list, np.ndarray)) and not hasattr(column, "tolist"):
            column = [column]
        data = data[list(column)]
    data = data.select_dtypes(include=["number", "datetime64", "datetimetz"], exclude="timedelta")
    naxes = len(data.columns)
    if naxes == 0:
        raise ValueError("hist method requires numerical or datetime columns, nothing to plot.")
    fig, axes = create_subplots(
        naxes=naxes,
        ax=ax,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        layout=layout,
    )
    can_set_label = "label" not in kwds
    for ax, col in zip(flatten_axes(axes), data.columns, strict=False):
        if legend and can_set_label:
            kwds["label"] = col
        ax.hist(_values(data[col].dropna()), bins=bins, **kwds)
        ax.set_title(col)
        ax.grid(grid)
        if legend:
            ax.legend()
    set_ticks_props(axes, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot)
    maybe_adjust_figure(fig, wspace=0.3, hspace=0.3)
    return axes


def _grouped_plot_by_column(
    plotf: Any,
    data: Any,
    columns: Any = None,
    by: Any = None,
    numeric_only: bool = True,
    grid: bool = False,
    figsize: Any = None,
    ax: Any = None,
    layout: Any = None,
    return_type: Any = None,
    **kwargs: Any,
) -> Any:
    from ._frame import Series

    grouped = data.groupby(by, observed=False)
    if columns is None:
        if not isinstance(by, (list, tuple)):
            by = [by]
        columns = [c for c in _numeric_columns(data) if c not in by]
        with contextlib.suppress(TypeError):
            columns = sorted(columns)
    naxes = len(columns)
    fig, axes = create_subplots(
        naxes=naxes,
        sharex=kwargs.pop("sharex", True),
        sharey=kwargs.pop("sharey", True),
        figsize=figsize,
        ax=ax,
        layout=layout,
    )
    xlabel, ylabel = kwargs.pop("xlabel", None), kwargs.pop("ylabel", None)
    if kwargs.get("vert", True):
        xlabel = xlabel or by
    else:
        ylabel = ylabel or by
    ax_values = []
    for ax, col in zip(flatten_axes(axes), columns, strict=False):
        keys, values = zip(*grouped[col], strict=True)
        re_plotf = plotf(keys, values, ax, xlabel=xlabel, ylabel=ylabel, **kwargs)
        ax.set_title(col)
        ax_values.append(re_plotf)
        ax.grid(grid)
    result = Series(ax_values, index=columns, dtype="object")
    if return_type is None:
        result = axes
    byline = by[0] if len(by) == 1 else by
    fig.suptitle(f"Boxplot grouped by {byline}")
    maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, wspace=0.2)
    return result


def boxplot(
    data: Any,
    column: Any = None,
    by: Any = None,
    ax: Any = None,
    fontsize: Any = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Any = None,
    layout: Any = None,
    return_type: Any = None,
    **kwds: Any,
) -> Any:
    import matplotlib.pyplot as plt

    if return_type not in BoxPlot._valid_return_types:
        raise ValueError("return_type must be {'axes', 'dict', 'both'}")
    if _series(data):
        data = data.to_frame("x")
        column = "x"

    def _get_colors() -> Any:
        result_list = get_standard_colors(num_colors=3)
        result = np.take(result_list, [0, 0, 2])
        result = np.append(result, "k")
        colors = kwds.pop("color", None)
        if colors:
            if isinstance(colors, dict):
                valid_keys = ["boxes", "whiskers", "medians", "caps"]
                key_to_index = dict(zip(valid_keys, range(4), strict=True))
                for key, value in colors.items():
                    if key in valid_keys:
                        result[key_to_index[key]] = value
                    else:
                        raise ValueError(
                            f"color dict contains invalid key '{key}'. "
                            f"The key must be either {valid_keys}"
                        )
            else:
                result.fill(colors)
        return result

    def plot_group(keys: Any, values: Any, ax: Any, **kwds: Any) -> Any:
        xlabel, ylabel = kwds.pop("xlabel", None), kwds.pop("ylabel", None)
        if xlabel:
            ax.set_xlabel(pprint_thing(xlabel))
        if ylabel:
            ax.set_ylabel(pprint_thing(ylabel))
        keys = [pprint_thing(x) for x in keys]
        values = [remove_na_arraylike(_values(v) if hasattr(v, "to_numpy") else v) for v in values]
        bp = ax.boxplot(values, **kwds)
        if fontsize is not None:
            ax.tick_params(axis="both", labelsize=fontsize)
        _set_ticklabels(ax=ax, labels=keys, is_vertical=kwds.get("vert", True), rotation=rot)
        maybe_color_bp(bp, color_tup=colors, **kwds)
        if return_type == "dict":
            return bp
        if return_type == "both":
            return BoxPlot.BP(ax=ax, lines=bp)
        return ax

    colors = _get_colors()
    if column is None:
        columns = None
    elif isinstance(column, (list, tuple)):
        columns = column
    else:
        columns = [column]
    if by is not None:
        return _grouped_plot_by_column(
            plot_group,
            data,
            columns=columns,
            by=by,
            grid=grid,
            figsize=figsize,
            ax=ax,
            layout=layout,
            return_type=return_type,
            **kwds,
        )
    if return_type is None:
        return_type = "axes"
    if layout is not None:
        raise ValueError("The 'layout' keyword is not supported when 'by' is None")
    if ax is None:
        rc = {"figure.figsize": figsize} if figsize is not None else {}
        with mpl.rc_context(rc):
            ax = plt.gca()
    data = data[_numeric_columns(data)]
    if len(data.columns) == 0:
        raise ValueError("boxplot method requires numerical columns, nothing to plot.")
    if columns is None:
        columns = list(data.columns)
    else:
        data = data[list(columns)]
    result = plot_group(columns, [_values(data[c]) for c in data.columns], ax, **kwds)
    ax.grid(grid)
    return result


def boxplot_frame(
    self: Any,
    column: Any = None,
    by: Any = None,
    ax: Any = None,
    fontsize: Any = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Any = None,
    layout: Any = None,
    return_type: Any = None,
    **kwds: Any,
) -> Any:
    import matplotlib.pyplot as plt

    ax = boxplot(
        self,
        column=column,
        by=by,
        ax=ax,
        fontsize=fontsize,
        grid=grid,
        rot=rot,
        figsize=figsize,
        layout=layout,
        return_type=return_type,
        **kwds,
    )
    plt.draw_if_interactive()
    return ax


def boxplot_frame_groupby(
    grouped: Any,
    subplots: bool = True,
    column: Any = None,
    fontsize: Any = None,
    rot: int = 0,
    grid: bool = True,
    ax: Any = None,
    figsize: Any = None,
    layout: Any = None,
    sharex: bool = False,
    sharey: bool = True,
    **kwds: Any,
) -> Any:
    from ._frame import Series

    groups = list(grouped)
    if subplots is True:
        fig, axes = create_subplots(
            naxes=len(groups),
            squeeze=False,
            ax=ax,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            layout=layout,
        )
        keys = []
        results = []
        for (key, group), ax in zip(groups, flatten_axes(axes), strict=False):
            d = group.boxplot(ax=ax, column=column, fontsize=fontsize, rot=rot, grid=grid, **kwds)
            ax.set_title(pprint_thing(key))
            keys.append(key)
            results.append(d)
        maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, wspace=0.2)
        return Series(results, index=keys, dtype="object")
    # One box per group and column on one axes, labelled as pandas labels the
    # columns of the groups set side by side.
    import matplotlib.pyplot as plt

    if column is not None and not isinstance(column, (list, tuple)):
        column = [column]
    labels = []
    values = []
    for key, group in groups:
        cols = column if column is not None else _numeric_columns(group)
        for col in cols:
            labels.append(pprint_thing((key, col)))
            values.append(_values(group[col]))
    if ax is None:
        rc = {"figure.figsize": figsize} if figsize is not None else {}
        with mpl.rc_context(rc):
            ax = plt.gca()
    result_list = get_standard_colors(num_colors=3)
    colors = np.append(np.take(result_list, [0, 0, 2]), "k")
    bp = ax.boxplot([remove_na_arraylike(v) for v in values], **kwds)
    if fontsize is not None:
        ax.tick_params(axis="both", labelsize=fontsize)
    _set_ticklabels(ax=ax, labels=labels, is_vertical=kwds.get("vert", True), rotation=rot)
    maybe_color_bp(bp, color_tup=colors, **kwds)
    ax.grid(grid)
    return ax


# The plots of pandas.plotting, from pandas' misc module.


def scatter_matrix(
    frame: Any,
    alpha: float = 0.5,
    figsize: Any = None,
    ax: Any = None,
    grid: bool = False,
    diagonal: str = "hist",
    marker: str = ".",
    density_kwds: Any = None,
    hist_kwds: Any = None,
    range_padding: float = 0.05,
    **kwds: Any,
) -> Any:
    df = frame[_numeric_columns(frame)]
    columns = list(df.columns)
    n = len(columns)
    fig, axes = create_subplots(naxes=n * n, figsize=figsize, ax=ax, squeeze=False)
    maybe_adjust_figure(fig, wspace=0, hspace=0)
    arrays = {a: np.asarray(_values(df[a]), dtype=float) for a in columns}
    mask = {a: ~isna(arrays[a]) for a in columns}
    if marker not in mpl.lines.lineMarkers:
        marker = "o"
    hist_kwds = hist_kwds or {}
    density_kwds = density_kwds or {}
    kwds.setdefault("edgecolors", "none")
    boundaries_list = []
    for a in columns:
        values = arrays[a][mask[a]]
        rmin_, rmax_ = np.min(values), np.max(values)
        rdelta_ext = (rmax_ - rmin_) * range_padding / 2
        boundaries_list.append((rmin_ - rdelta_ext, rmax_ + rdelta_ext))
    for i, a in enumerate(columns):
        for j, b in enumerate(columns):
            ax = axes[i, j]
            if i == j:
                values = arrays[a][mask[a]]
                if diagonal == "hist":
                    ax.hist(values, **hist_kwds)
                elif diagonal in ("kde", "density"):
                    from scipy.stats import gaussian_kde

                    gkde = gaussian_kde(values)
                    ind = np.linspace(values.min(), values.max(), 1000)
                    ax.plot(ind, gkde.evaluate(ind), **density_kwds)
                ax.set_xlim(boundaries_list[i])
            else:
                common = mask[a] & mask[b]
                ax.scatter(arrays[b][common], arrays[a][common], marker=marker, alpha=alpha, **kwds)
                ax.set_xlim(boundaries_list[j])
                ax.set_ylim(boundaries_list[i])
            ax.set_xlabel(b)
            ax.set_ylabel(a)
            if j != 0:
                ax.yaxis.set_visible(False)
            if i != n - 1:
                ax.xaxis.set_visible(False)
    if n > 1:
        lim1 = boundaries_list[0]
        locs = axes[0][1].yaxis.get_majorticklocs()
        locs = locs[(lim1[0] <= locs) & (locs <= lim1[1])]
        adj = (locs - lim1[0]) / (lim1[1] - lim1[0])
        lim0 = axes[0][0].get_ylim()
        adj = adj * (lim0[1] - lim0[0]) + lim0[0]
        axes[0][0].yaxis.set_ticks(adj)
        if np.all(locs == locs.astype(int)):
            locs = locs.astype(int)
        axes[0][0].yaxis.set_ticklabels(locs)
    set_ticks_props(axes, xlabelsize=8, xrot=90, ylabelsize=8, yrot=0)
    return axes


def _class_rows(
    frame: Any, class_column: Any, cols: Any = None
) -> tuple[list[Any], list[Any], Any]:
    """The distinct classes in order, each row's class, and the other columns."""
    class_col = list(frame[class_column])
    classes = list(dict.fromkeys(class_col))
    if cols is None:
        cols = [c for c in frame.columns if c != class_column]
    df = frame[cols]
    return classes, class_col, df


def _rows(df: Any) -> np.ndarray:
    return np.column_stack([np.asarray(_values(df[c]), dtype=float) for c in df.columns])


def radviz(
    frame: Any,
    class_column: Any,
    ax: Any = None,
    color: Any = None,
    colormap: Any = None,
    **kwds: Any,
) -> Any:
    import matplotlib.pyplot as plt

    classes, class_col, df = _class_rows(frame, class_column)
    rows = _rows(df)
    lows = rows.min(axis=0)
    highs = rows.max(axis=0)
    rows = (rows - lows) / (highs - lows)
    if ax is None:
        ax = plt.gca()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    to_plot: dict[Any, list[list[Any]]] = {kls: [[], []] for kls in classes}
    colors = get_standard_colors(
        num_colors=len(classes), colormap=colormap, color_type="random", color=color
    )
    m = len(frame.columns) - 1
    s = np.array([(np.cos(t), np.sin(t)) for t in [2 * np.pi * (i / m) for i in range(m)]])
    for i in range(len(rows)):
        row = rows[i]
        row_ = np.repeat(np.expand_dims(row, axis=1), 2, axis=1)
        y = (s * row_).sum(axis=0) / row.sum()
        kls = class_col[i]
        to_plot[kls][0].append(y[0])
        to_plot[kls][1].append(y[1])
    for i, kls in enumerate(classes):
        ax.scatter(
            to_plot[kls][0], to_plot[kls][1], color=colors[i], label=pprint_thing(kls), **kwds
        )
    ax.legend()
    ax.add_patch(mpl.patches.Circle((0.0, 0.0), radius=1.0, facecolor="none"))
    for xy, name in zip(s, df.columns, strict=True):
        ax.add_patch(mpl.patches.Circle(xy, radius=0.025, facecolor="gray"))
        if xy[0] < 0.0 and xy[1] < 0.0:
            ax.text(xy[0] - 0.025, xy[1] - 0.025, name, ha="right", va="top", size="small")
        elif xy[0] < 0.0 <= xy[1]:
            ax.text(xy[0] - 0.025, xy[1] + 0.025, name, ha="right", va="bottom", size="small")
        elif xy[1] < 0.0 <= xy[0]:
            ax.text(xy[0] + 0.025, xy[1] - 0.025, name, ha="left", va="top", size="small")
        elif xy[0] >= 0.0 and xy[1] >= 0.0:
            ax.text(xy[0] + 0.025, xy[1] + 0.025, name, ha="left", va="bottom", size="small")
    ax.axis("equal")
    return ax


def andrews_curves(
    frame: Any,
    class_column: Any,
    ax: Any = None,
    samples: int = 200,
    color: Any = None,
    colormap: Any = None,
    **kwds: Any,
) -> Any:
    import matplotlib.pyplot as plt

    def function(amplitudes: Any) -> Any:
        def f(t: Any) -> Any:
            result = amplitudes[0] / np.sqrt(2.0)
            coeffs = np.delete(np.copy(amplitudes), 0)
            coeffs = np.resize(coeffs, (int((coeffs.size + 1) / 2), 2))
            harmonics = np.arange(0, coeffs.shape[0]) + 1
            trig_args = np.outer(harmonics, t)
            result += np.sum(
                coeffs[:, 0, np.newaxis] * np.sin(trig_args)
                + coeffs[:, 1, np.newaxis] * np.cos(trig_args),
                axis=0,
            )
            return result

        return f

    classes, class_col, df = _class_rows(frame, class_column)
    rows = _rows(df)
    t = np.linspace(-np.pi, np.pi, samples)
    used_legends: set[str] = set()
    color_values = get_standard_colors(
        num_colors=len(classes), colormap=colormap, color_type="random", color=color
    )
    colors = dict(zip(classes, color_values, strict=False))
    if ax is None:
        ax = plt.gca()
        ax.set_xlim(-np.pi, np.pi)
    for i in range(len(rows)):
        y = function(rows[i])(t)
        kls = class_col[i]
        label = pprint_thing(kls)
        if label not in used_legends:
            used_legends.add(label)
            ax.plot(t, y, color=colors[kls], label=label, **kwds)
        else:
            ax.plot(t, y, color=colors[kls], **kwds)
    ax.legend(loc="upper right")
    ax.grid()
    return ax


def bootstrap_plot(
    series: Any, fig: Any = None, size: int = 50, samples: int = 500, **kwds: Any
) -> Any:
    import matplotlib.pyplot as plt

    data = list(_values(series))
    samplings = [random.sample(data, size) for _ in range(samples)]
    means = np.array([np.mean(sampling) for sampling in samplings])
    medians = np.array([np.median(sampling) for sampling in samplings])
    midranges = np.array([(min(sampling) + max(sampling)) * 0.5 for sampling in samplings])
    if fig is None:
        fig = plt.figure()
    x = list(range(samples))
    axes = []
    for place, xlabel, values, drawn in (
        (1, "Sample", means, "plot"),
        (2, "Sample", medians, "plot"),
        (3, "Sample", midranges, "plot"),
        (4, "Mean", means, "hist"),
        (5, "Median", medians, "hist"),
        (6, "Midrange", midranges, "hist"),
    ):
        ax = fig.add_subplot(2, 3, place)
        ax.set_xlabel(xlabel)
        axes.append(ax)
        if drawn == "plot":
            ax.plot(x, values, **kwds)
        else:
            ax.hist(values, **kwds)
    for axis in axes:
        plt.setp(axis.get_xticklabels(), fontsize=8)
        plt.setp(axis.get_yticklabels(), fontsize=8)
    if do_adjust_figure(fig):
        plt.tight_layout()
    return fig


def parallel_coordinates(
    frame: Any,
    class_column: Any,
    cols: Any = None,
    ax: Any = None,
    color: Any = None,
    use_columns: bool = False,
    xticks: Any = None,
    colormap: Any = None,
    axvlines: bool = True,
    axvlines_kwds: Any = None,
    sort_labels: bool = False,
    **kwds: Any,
) -> Any:
    import matplotlib.pyplot as plt

    if axvlines_kwds is None:
        axvlines_kwds = {"linewidth": 1, "color": "black"}
    classes, class_col, df = _class_rows(frame, class_column, cols)
    rows = _rows(df)
    used_legends: set[str] = set()
    ncols = len(df.columns)
    x: Any
    if use_columns is True:
        if not np.all(np.isreal(list(df.columns))):
            raise ValueError("Columns must be numeric to be used as xticks")
        x = list(df.columns)
    elif xticks is not None:
        if not np.all(np.isreal(xticks)):
            raise ValueError("xticks specified must be numeric")
        if len(xticks) != ncols:
            raise ValueError("Length of xticks must match number of columns")
        x = xticks
    else:
        x = list(range(ncols))
    if ax is None:
        ax = plt.gca()
    color_values = get_standard_colors(
        num_colors=len(classes), colormap=colormap, color_type="random", color=color
    )
    if sort_labels:
        classes = sorted(classes)
        color_values = sorted(color_values)
    colors = dict(zip(classes, color_values, strict=True))
    for i in range(len(rows)):
        kls = class_col[i]
        label = pprint_thing(kls)
        if label not in used_legends:
            used_legends.add(label)
            ax.plot(x, rows[i], color=colors[kls], label=label, **kwds)
        else:
            ax.plot(x, rows[i], color=colors[kls], **kwds)
    if axvlines:
        for i in x:
            ax.axvline(i, **axvlines_kwds)
    ax.set_xticks(x)
    ax.set_xticklabels(list(df.columns))
    ax.set_xlim(x[0], x[-1])
    ax.legend(loc="upper right")
    ax.grid()
    return ax


def lag_plot(series: Any, lag: int = 1, ax: Any = None, **kwds: Any) -> Any:
    import matplotlib.pyplot as plt

    kwds.setdefault("c", plt.rcParams["patch.facecolor"])
    data = _values(series)
    y1 = data[:-lag]
    y2 = data[lag:]
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel("y(t)")
    ax.set_ylabel(f"y(t + {lag})")
    ax.scatter(y1, y2, **kwds)
    return ax


def autocorrelation_plot(series: Any, ax: Any = None, **kwds: Any) -> Any:
    import matplotlib.pyplot as plt

    n = len(series)
    data = np.asarray(_values(series), dtype=float)
    if ax is None:
        ax = plt.gca()
        ax.set_xlim(1, n)
        ax.set_ylim(-1.0, 1.0)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / n

    def r(h: int) -> float:
        return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0

    x = np.arange(n) + 1
    y = [r(loc) for loc in x]
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    ax.axhline(y=z99 / np.sqrt(n), linestyle="--", color="grey")
    ax.axhline(y=z95 / np.sqrt(n), color="grey")
    ax.axhline(y=0.0, color="black")
    ax.axhline(y=-z95 / np.sqrt(n), color="grey")
    ax.axhline(y=-z99 / np.sqrt(n), linestyle="--", color="grey")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.plot(x, y, **kwds)
    if "label" in kwds:
        ax.legend()
    ax.grid()
    return ax


def register() -> None:
    """matplotlib draws dates and times itself, so there is nothing more to register."""


def deregister() -> None:
    """Undo `register`, which registered nothing."""
