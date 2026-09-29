"""The plotting front end, `df.plot`, `s.hist`, `df.boxplot` and `pandas.plotting`.

pandas splits plotting in two. The front end checks the arguments and picks a
backend, and the backend draws. The backend is a module with a `plot` at the top
of it, named by the option `plotting.backend` or the `backend` argument, found
through the `pandas_plotting_backends` entry points or imported by name, which
is how third party backends such as plotly and hvplot plug in. This module is
that front end, word for word where pandas' checks raise, and `_mpl` is the
matplotlib backend, which pandas calls `pandas.plotting._matplotlib`.

matplotlib stays optional. Nothing here imports it until a plot is drawn, and
drawing without it raises pandas' ImportError.
"""

from __future__ import annotations

import contextlib
import importlib
from types import ModuleType
from typing import Any, ClassVar

from ._config import get_option

_MATPLOTLIB = "firepanda._mpl"
_backends: dict[str, ModuleType] = {}


def _load_backend(backend: str) -> ModuleType:
    """The backend module named `backend`, found the three ways pandas looks.

    Raises:
        ImportError: For `matplotlib` when matplotlib is not installed.
        ValueError: When no entry point or module of that name has a `plot`.
    """
    from importlib.metadata import entry_points

    if backend == "matplotlib":
        try:
            return importlib.import_module(_MATPLOTLIB)
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting when the "
                'default backend "matplotlib" is selected.'
            ) from None
    module = None
    for entry_point in entry_points().select(group="pandas_plotting_backends"):
        if entry_point.name == backend:
            module = entry_point.load()
            break
    if module is None:
        try:
            module = importlib.import_module(backend)
        except ImportError:
            module = None
    if module is not None and hasattr(module, "plot"):
        return module
    raise ValueError(
        f"Could not find plotting backend '{backend}'. Ensure that you've "
        f"installed the package providing the '{backend}' entrypoint, or that "
        "the package has a top-level `.plot` method."
    )


def _get_plot_backend(backend: str | None = None) -> ModuleType:
    """The backend asked for, or the one the option names, loaded once and kept."""
    name: str = backend or get_option("plotting.backend")
    if name not in _backends:
        _backends[name] = _load_backend(name)
    return _backends[name]


def _is_series(data: Any) -> bool:
    from ._pandas import SeriesMixin

    return isinstance(data, SeriesMixin)


def _is_frame(data: Any) -> bool:
    from ._pandas import DataFrameMixin

    return isinstance(data, DataFrameMixin)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _holds_integer(columns: Any) -> bool:
    """Whether the column labels are whole numbers, so `x=0` names a column rather than a place."""
    return str(columns.inferred_type) in ("integer", "mixed-integer")


def _is_list_like(value: Any) -> bool:
    return hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict))


def hist_series(
    self: Any,
    by: Any = None,
    ax: Any = None,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot: float | None = None,
    ylabelsize: int | None = None,
    yrot: float | None = None,
    figsize: tuple[int, int] | None = None,
    bins: Any = 10,
    backend: str | None = None,
    legend: bool = False,
    **kwargs: Any,
) -> Any:
    """Draw a histogram of the column with matplotlib, one per group when `by` is given."""
    plot_backend = _get_plot_backend(backend)
    return plot_backend.hist_series(
        self,
        by=by,
        ax=ax,
        grid=grid,
        xlabelsize=xlabelsize,
        xrot=xrot,
        ylabelsize=ylabelsize,
        yrot=yrot,
        figsize=figsize,
        bins=bins,
        legend=legend,
        **kwargs,
    )


def hist_frame(
    data: Any,
    column: Any = None,
    by: Any = None,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot: float | None = None,
    ylabelsize: int | None = None,
    yrot: float | None = None,
    ax: Any = None,
    sharex: bool = False,
    sharey: bool = False,
    figsize: tuple[int, int] | None = None,
    layout: tuple[int, int] | None = None,
    bins: Any = 10,
    backend: str | None = None,
    legend: bool = False,
    **kwargs: Any,
) -> Any:
    """Draw one histogram per numeric column, each on its own axes."""
    plot_backend = _get_plot_backend(backend)
    return plot_backend.hist_frame(
        data,
        column=column,
        by=by,
        grid=grid,
        xlabelsize=xlabelsize,
        xrot=xrot,
        ylabelsize=ylabelsize,
        yrot=yrot,
        ax=ax,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        layout=layout,
        legend=legend,
        bins=bins,
        **kwargs,
    )


def boxplot(
    data: Any,
    column: str | list[str] | None = None,
    by: str | list[str] | None = None,
    ax: Any = None,
    fontsize: float | str | None = None,
    rot: int = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    return_type: str | None = None,
    **kwargs: Any,
) -> Any:
    """Draw a box plot of the columns, always with matplotlib, as pandas does."""
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.boxplot(
        data,
        column=column,
        by=by,
        ax=ax,
        fontsize=fontsize,
        rot=rot,
        grid=grid,
        figsize=figsize,
        layout=layout,
        return_type=return_type,
        **kwargs,
    )


def boxplot_frame(
    self: Any,
    column: Any = None,
    by: Any = None,
    ax: Any = None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: Any = None,
    return_type: Any = None,
    backend: Any = None,
    **kwargs: Any,
) -> Any:
    """`DataFrame.boxplot`, which draws through the backend asked for."""
    plot_backend = _get_plot_backend(backend)
    return plot_backend.boxplot_frame(
        self,
        column=column,
        by=by,
        ax=ax,
        fontsize=fontsize,
        rot=rot,
        grid=grid,
        figsize=figsize,
        layout=layout,
        return_type=return_type,
        **kwargs,
    )


def boxplot_frame_groupby(
    grouped: Any,
    subplots: bool = True,
    column: Any = None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    ax: Any = None,
    figsize: tuple[float, float] | None = None,
    layout: Any = None,
    sharex: bool = False,
    sharey: bool = True,
    backend: Any = None,
    **kwargs: Any,
) -> Any:
    """`DataFrameGroupBy.boxplot`, one box plot per group or one plot of them all."""
    plot_backend = _get_plot_backend(backend)
    return plot_backend.boxplot_frame_groupby(
        grouped,
        subplots=subplots,
        column=column,
        fontsize=fontsize,
        rot=rot,
        grid=grid,
        ax=ax,
        figsize=figsize,
        layout=layout,
        sharex=sharex,
        sharey=sharey,
        **kwargs,
    )


_SERIES_ARGS: list[tuple[str, Any]] = [
    ("kind", "line"),
    ("ax", None),
    ("figsize", None),
    ("use_index", True),
    ("title", None),
    ("grid", None),
    ("legend", False),
    ("style", None),
    ("logx", False),
    ("logy", False),
    ("loglog", False),
    ("xticks", None),
    ("yticks", None),
    ("xlim", None),
    ("ylim", None),
    ("rot", None),
    ("fontsize", None),
    ("colormap", None),
    ("table", False),
    ("yerr", None),
    ("xerr", None),
    ("label", None),
    ("secondary_y", False),
    ("xlabel", None),
    ("ylabel", None),
]

_FRAME_ARGS: list[tuple[str, Any]] = [
    ("x", None),
    ("y", None),
    ("kind", "line"),
    ("ax", None),
    ("subplots", False),
    ("sharex", None),
    ("sharey", False),
    ("layout", None),
    ("figsize", None),
    ("use_index", True),
    ("title", None),
    ("grid", None),
    ("legend", True),
    ("style", None),
    ("logx", False),
    ("logy", False),
    ("loglog", False),
    ("xticks", None),
    ("yticks", None),
    ("xlim", None),
    ("ylim", None),
    ("rot", None),
    ("fontsize", None),
    ("colormap", None),
    ("table", False),
    ("yerr", None),
    ("xerr", None),
    ("secondary_y", False),
    ("xlabel", None),
    ("ylabel", None),
]


class PlotAccessor:
    """`s.plot` and `df.plot`, callable as `df.plot(kind=...)` or through a method per kind."""

    __slots__ = ("_parent",)

    _common_kinds = ("line", "bar", "barh", "kde", "density", "area", "hist", "box")
    _series_kinds = ("pie",)
    _dataframe_kinds = ("scatter", "hexbin")
    _kind_aliases: ClassVar[dict[str, str]] = {"density": "kde"}
    _all_kinds = _common_kinds + _series_kinds + _dataframe_kinds

    def __init__(self, data: Any) -> None:
        self._parent = data

    @staticmethod
    def _get_call_args(
        backend_name: str, data: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Any, Any, str, dict[str, Any]]:
        """The positional and keyword arguments read the way pandas reads them.

        Raises:
            TypeError: For positional arguments to a series, which pandas refuses.
        """
        if _is_series(data):
            arg_def = _SERIES_ARGS
        elif _is_frame(data):
            arg_def = _FRAME_ARGS
        else:
            raise TypeError(
                f"Called plot accessor for type {type(data).__name__}, expected Series or DataFrame"
            )
        if args and _is_series(data):
            positional_args = str(args)[1:-1]
            keyword_args = ", ".join(
                f"{name}={value!r}" for (name, _), value in zip(arg_def, args, strict=False)
            )
            raise TypeError(
                "`Series.plot()` should not be called with positional "
                "arguments, only keyword arguments. The order of "
                "positional arguments will change in the future. "
                f"Use `Series.plot({keyword_args})` instead of "
                f"`Series.plot({positional_args})`."
            )
        pos_args = {name: value for (name, _), value in zip(arg_def, args, strict=False)}
        if backend_name == _MATPLOTLIB:
            kwargs = dict(arg_def, **pos_args, **kwargs)
        else:
            kwargs = dict(pos_args, **kwargs)
        x = kwargs.pop("x", None)
        y = kwargs.pop("y", None)
        kind = kwargs.pop("kind", "line")
        return x, y, kind, kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Draw the plot `kind` asks for, a line plot by default.

        Raises:
            ValueError: For a kind pandas does not know, a frame kind asked of a
                series, a pie of a frame with neither `y` nor `subplots=True`,
                and an `x` or `y` that names no column.
        """
        plot_backend = _get_plot_backend(kwargs.pop("backend", None))
        x, y, kind, kwargs = self._get_call_args(plot_backend.__name__, self._parent, args, kwargs)
        kind = self._kind_aliases.get(kind, kind)
        if plot_backend.__name__ != _MATPLOTLIB:
            return plot_backend.plot(self._parent, x=x, y=y, kind=kind, **kwargs)
        if kind not in self._all_kinds:
            raise ValueError(f"{kind} is not a valid plot kind Valid plot kinds: {self._all_kinds}")
        data = self._parent
        if _is_series(data):
            kwargs["reuse_plot"] = True
        if kind in self._dataframe_kinds:
            if _is_frame(data):
                return plot_backend.plot(data, x=x, y=y, kind=kind, **kwargs)
            raise ValueError(f"plot kind {kind} can only be used for data frames")
        if kind in self._series_kinds:
            if _is_frame(data):
                if y is None and kwargs.get("subplots") is False:
                    raise ValueError(f"{kind} requires either y column or 'subplots=True'")
                if y is not None:
                    if _is_integer(y) and not _holds_integer(data.columns):
                        y = data.columns[y]
                    data = data[y].rename_axis(y)
        elif _is_frame(data):
            data_cols = data.columns
            if x is not None:
                if _is_integer(x) and not _holds_integer(data.columns):
                    x = data_cols[x]
                elif not _is_series(data[x]):
                    raise ValueError("x must be a label or position")
                data = data.set_index(x)
            if y is not None:
                int_ylist = _is_list_like(y) and all(_is_integer(c) for c in y)
                if (_is_integer(y) or int_ylist) and not _holds_integer(data.columns):
                    y = data_cols[y] if _is_integer(y) else [data_cols[c] for c in y]
                label_kw = kwargs.get("label", False)
                for kw in ("xerr", "yerr"):
                    if kw in kwargs and (isinstance(kwargs[kw], str) or _is_integer(kwargs[kw])):
                        with contextlib.suppress(IndexError, KeyError, TypeError):
                            kwargs[kw] = data[kwargs[kw]]
                data = data[y]
                if _is_series(data):
                    data = data.rename(label_kw or y)
                else:
                    match = _is_list_like(label_kw) and len(label_kw) == len(y)
                    if label_kw and not match:
                        raise ValueError("label should be list-like and same length as y")
                    data.columns = label_kw or data.columns
        return plot_backend.plot(data, kind=kind, **kwargs)

    def line(self, x: Any = None, y: Any = None, color: Any = None, **kwargs: Any) -> Any:
        """A line plot, each column a line against the index or `x`."""
        if color is not None:
            kwargs["color"] = color
        return self(kind="line", x=x, y=y, **kwargs)

    def bar(self, x: Any = None, y: Any = None, color: Any = None, **kwargs: Any) -> Any:
        """A vertical bar plot, one group of bars per row."""
        if color is not None:
            kwargs["color"] = color
        return self(kind="bar", x=x, y=y, **kwargs)

    def barh(self, x: Any = None, y: Any = None, color: Any = None, **kwargs: Any) -> Any:
        """A horizontal bar plot, one group of bars per row."""
        if color is not None:
            kwargs["color"] = color
        return self(kind="barh", x=x, y=y, **kwargs)

    def box(self, by: Any = None, **kwargs: Any) -> Any:
        """A box plot of each column's quartiles."""
        return self(kind="box", by=by, **kwargs)

    def hist(self, by: Any = None, bins: int = 10, **kwargs: Any) -> Any:
        """One histogram of every column on one axes."""
        return self(kind="hist", by=by, bins=bins, **kwargs)

    def kde(
        self, bw_method: Any = None, ind: Any = None, weights: Any = None, **kwargs: Any
    ) -> Any:
        """A kernel density estimate of each column, which needs scipy as in pandas."""
        return self(kind="kde", bw_method=bw_method, ind=ind, weights=weights, **kwargs)

    density = kde

    def area(self, x: Any = None, y: Any = None, stacked: bool = True, **kwargs: Any) -> Any:
        """An area plot, stacked unless `stacked=False`."""
        return self(kind="area", x=x, y=y, stacked=stacked, **kwargs)

    def pie(self, y: Any = None, **kwargs: Any) -> Any:
        """A pie of a column, or of each column with `subplots=True`."""
        if y is not None:
            kwargs["y"] = y
        if (
            _is_frame(self._parent)
            and kwargs.get("y") is None
            and not kwargs.get("subplots", False)
        ):
            raise ValueError("pie requires either y column or 'subplots=True'")
        return self(kind="pie", **kwargs)

    def scatter(self, x: Any, y: Any, s: Any = None, c: Any = None, **kwargs: Any) -> Any:
        """A scatter plot of column `y` against column `x`."""
        return self(kind="scatter", x=x, y=y, s=s, c=c, **kwargs)

    def hexbin(
        self,
        x: Any,
        y: Any,
        C: Any = None,
        reduce_C_function: Any = None,
        gridsize: Any = None,
        **kwargs: Any,
    ) -> Any:
        """A hexagonal binning plot of column `y` against column `x`."""
        if reduce_C_function is not None:
            kwargs["reduce_C_function"] = reduce_C_function
        if gridsize is not None:
            kwargs["gridsize"] = gridsize
        return self(kind="hexbin", x=x, y=y, C=C, **kwargs)


class GroupByPlot:
    """`grouped.plot`, which plots each group in turn and answers a series of the results."""

    __slots__ = ("_groupby",)

    def __init__(self, groupby: Any) -> None:
        self._groupby = groupby

    def _each(self, draw: Any) -> Any:
        from ._frame import Series

        keys, results = [], []
        for key, group in self._groupby:
            keys.append(key)
            results.append(draw(group))
        return Series(results, index=keys, dtype="object")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._each(lambda group: group.plot(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def attr(*args: Any, **kwargs: Any) -> Any:
            return self._each(lambda group: getattr(group.plot, name)(*args, **kwargs))

        return attr


class _Options(dict):  # type: ignore[type-arg]
    """`pandas.plotting.plot_params`, a dict of plotting settings with one alias.

    `x_compat` and `xaxis.compat` are one key, and the defaults cannot be removed.
    """

    _ALIASES: ClassVar[dict[str, str]] = {"x_compat": "xaxis.compat"}
    _DEFAULT_KEYS: ClassVar[list[str]] = ["xaxis.compat"]

    def __init__(self) -> None:
        super().__setitem__("xaxis.compat", False)

    def __getitem__(self, key: Any) -> Any:
        key = self._get_canonical_key(key)
        if key not in self:
            raise ValueError(f"{key} is not a valid pandas plotting option")
        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(self._get_canonical_key(key), value)

    def __delitem__(self, key: Any) -> None:
        key = self._get_canonical_key(key)
        if key in self._DEFAULT_KEYS:
            raise ValueError(f"Cannot remove default parameter {key}")
        super().__delitem__(key)

    def __contains__(self, key: Any) -> bool:
        return super().__contains__(self._get_canonical_key(key))

    def reset(self) -> None:
        """Put every setting back to its default."""
        self.__init__()  # type: ignore[misc]

    def _get_canonical_key(self, key: Any) -> Any:
        return self._ALIASES.get(key, key)

    def use(self, key: Any, value: Any) -> Any:
        """A context in which `key` is `value`, put back afterwards."""
        import contextlib

        @contextlib.contextmanager
        def using() -> Any:
            old_value = self[key]
            try:
                self[key] = value
                yield self
            finally:
                self[key] = old_value

        return using()


plot_params = _Options()


def table(ax: Any, data: Any, **kwargs: Any) -> Any:
    """Draw the frame or column as a matplotlib table on `ax`."""
    return _get_plot_backend("matplotlib").table(
        ax=ax, data=data, rowLabels=None, colLabels=None, **kwargs
    )


def register() -> None:
    """Register the date converters with matplotlib, which matplotlib's own already cover."""
    _get_plot_backend("matplotlib").register()


def deregister() -> None:
    """Undo `register`."""
    _get_plot_backend("matplotlib").deregister()


def scatter_matrix(
    frame: Any,
    alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
    ax: Any = None,
    grid: bool = False,
    diagonal: str = "hist",
    marker: str = ".",
    density_kwds: Any = None,
    hist_kwds: Any = None,
    range_padding: float = 0.05,
    **kwargs: Any,
) -> Any:
    """A grid of scatter plots of every numeric column against every other."""
    return _get_plot_backend("matplotlib").scatter_matrix(
        frame=frame,
        alpha=alpha,
        figsize=figsize,
        ax=ax,
        grid=grid,
        diagonal=diagonal,
        marker=marker,
        density_kwds=density_kwds,
        hist_kwds=hist_kwds,
        range_padding=range_padding,
        **kwargs,
    )


def radviz(
    frame: Any,
    class_column: str,
    ax: Any = None,
    color: Any = None,
    colormap: Any = None,
    **kwds: Any,
) -> Any:
    """Each row as a point pulled toward the columns around a circle."""
    return _get_plot_backend("matplotlib").radviz(
        frame=frame, class_column=class_column, ax=ax, color=color, colormap=colormap, **kwds
    )


def andrews_curves(
    frame: Any,
    class_column: str,
    ax: Any = None,
    samples: int = 200,
    color: Any = None,
    colormap: Any = None,
    **kwargs: Any,
) -> Any:
    """Each row as a Fourier series, coloured by its class."""
    return _get_plot_backend("matplotlib").andrews_curves(
        frame=frame,
        class_column=class_column,
        ax=ax,
        samples=samples,
        color=color,
        colormap=colormap,
        **kwargs,
    )


def bootstrap_plot(
    series: Any, fig: Any = None, size: int = 50, samples: int = 500, **kwds: Any
) -> Any:
    """The mean, median and midrange of random samples, with their histograms."""
    return _get_plot_backend("matplotlib").bootstrap_plot(
        series=series, fig=fig, size=size, samples=samples, **kwds
    )


def parallel_coordinates(
    frame: Any,
    class_column: str,
    cols: Any = None,
    ax: Any = None,
    color: Any = None,
    use_columns: bool = False,
    xticks: Any = None,
    colormap: Any = None,
    axvlines: bool = True,
    axvlines_kwds: Any = None,
    sort_labels: bool = False,
    **kwargs: Any,
) -> Any:
    """Each row as a line across the columns, coloured by its class."""
    return _get_plot_backend("matplotlib").parallel_coordinates(
        frame=frame,
        class_column=class_column,
        cols=cols,
        ax=ax,
        color=color,
        use_columns=use_columns,
        xticks=xticks,
        colormap=colormap,
        axvlines=axvlines,
        axvlines_kwds=axvlines_kwds,
        sort_labels=sort_labels,
        **kwargs,
    )


def lag_plot(series: Any, lag: int = 1, ax: Any = None, **kwds: Any) -> Any:
    """Each value against the one `lag` rows later."""
    return _get_plot_backend("matplotlib").lag_plot(series=series, lag=lag, ax=ax, **kwds)


def autocorrelation_plot(series: Any, ax: Any = None, **kwargs: Any) -> Any:
    """The autocorrelation at every lag, with the 95 and 99 percent bands."""
    return _get_plot_backend("matplotlib").autocorrelation_plot(series=series, ax=ax, **kwargs)
