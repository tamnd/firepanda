"""Plotting, `df.plot`, `s.hist`, `df.boxplot` and `pandas.plotting`, compared with pandas.

The checks on the arguments run everywhere, since they are made before a
backend is loaded. The drawing tests need matplotlib, and draw the same plot
with both libraries and compare what ends up on the axes: the lines, the
patches, the tick labels, the titles and the legend.
"""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)
needs_matplotlib = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="matplotlib is not installed"
)


def numbers(lib: ModuleType) -> Any:
    return lib.DataFrame({"a": [1, 2, 3, 4], "b": [4.0, 3.0, 2.0, 1.0]})


def test_plot_is_the_accessor_class_on_the_type() -> None:
    assert fp.Series.plot is fp.plotting.PlotAccessor
    assert fp.DataFrame.plot is fp.plotting.PlotAccessor
    assert isinstance(numbers(fp).plot, fp.plotting.PlotAccessor)


@needs_pandas
def test_the_namespace_is_pandas() -> None:
    import pandas

    assert sorted(fp.plotting.__all__) == sorted(pandas.plotting.__all__)


@needs_pandas
@pytest.mark.parametrize(
    "name",
    [
        "andrews_curves",
        "autocorrelation_plot",
        "bootstrap_plot",
        "boxplot",
        "boxplot_frame",
        "boxplot_frame_groupby",
        "hist_frame",
        "hist_series",
        "lag_plot",
        "parallel_coordinates",
        "radviz",
        "scatter_matrix",
        "table",
    ],
)
def test_the_signatures_are_pandas(name: str) -> None:
    import pandas

    ours = inspect.signature(getattr(fp.plotting, name))
    theirs = inspect.signature(getattr(pandas.plotting, name))
    assert list(ours.parameters) == list(theirs.parameters)


@needs_pandas
@pytest.mark.parametrize(
    "call",
    [
        lambda lib: numbers(lib).plot(kind="nope"),
        lambda lib: numbers(lib)["a"].plot("line", None, 1),
        lambda lib: numbers(lib)["a"].plot.scatter(x=1, y=2),
        lambda lib: numbers(lib).plot.pie(),
        lambda lib: numbers(lib).plot(backend="no_such_backend_here"),
    ],
)
def test_the_arguments_are_checked_before_drawing(call: Callable[[ModuleType], Any]) -> None:
    import pandas

    with pytest.raises(Exception) as ours:
        call(fp)
    with pytest.raises(Exception) as theirs:
        call(pandas)
    assert type(ours.value) is type(theirs.value)
    assert str(ours.value) == str(theirs.value)


def test_no_matplotlib_is_pandas_sentence(monkeypatch: Any) -> None:
    import sys

    from firepanda import _plotting

    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.delitem(sys.modules, "firepanda._mpl", raising=False)
    monkeypatch.setattr(_plotting, "_backends", {})
    with pytest.raises(ImportError, match='default backend "matplotlib" is selected'):
        numbers(fp).plot()


def test_plot_params_answers_both_spellings() -> None:
    params = fp.plotting.plot_params
    assert params["x_compat"] is False
    with params.use("x_compat", True):
        assert params["xaxis.compat"] is True
    assert params["x_compat"] is False


def summary(result: Any) -> Any:
    """What a plot drew, in plain values that compare equal across the two libraries."""
    import numpy as np

    if isinstance(result, dict):
        return sorted(result)
    out = []
    for ax in np.asarray(result, dtype=object).reshape(-1):
        legend = ax.get_legend()
        out.append(
            {
                "title": ax.get_title(),
                "labels": (ax.get_xlabel(), ax.get_ylabel()),
                "xticks": [t.get_text() for t in ax.get_xticklabels()],
                "limits": np.round([ax.get_xlim(), ax.get_ylim()], 6).tolist(),
                "lines": [
                    (np.asarray(line.get_ydata(), dtype=float).round(6).tolist(), line.get_label())
                    for line in ax.get_lines()
                ],
                "colors": [str(line.get_color()) for line in ax.get_lines()],
                "patches": [
                    np.round(patch.get_bbox().bounds, 6).tolist()
                    for patch in ax.patches
                    if hasattr(patch, "get_bbox")
                ],
                "collections": len(ax.collections),
                "legend": None if legend is None else [t.get_text() for t in legend.get_texts()],
                "visible": ax.get_visible(),
            }
        )
    return out


DRAWN: list[Callable[[ModuleType], Any]] = [
    lambda lib: numbers(lib).plot(),
    lambda lib: numbers(lib)["a"].plot(title="t", color="r", style="--"),
    lambda lib: numbers(lib).plot(secondary_y=["b"]),
    lambda lib: numbers(lib).plot(subplots=True, layout=(1, 2), title=["A", "B"]),
    lambda lib: numbers(lib).plot(stacked=True),
    lambda lib: numbers(lib).plot(x="a", y="b"),
    lambda lib: numbers(lib).plot(yerr=0.5),
    lambda lib: numbers(lib).plot(logy=True, grid=True),
    lambda lib: numbers(lib).plot(legend="reverse"),
    lambda lib: numbers(lib).plot(colormap="viridis"),
    lambda lib: numbers(lib).plot(xticks=[0, 2], xlim=(0, 3), ylim=(0, 5), rot=45),
    lambda lib: numbers(lib).plot.bar(),
    lambda lib: numbers(lib).plot.bar(align="edge", width=0.8),
    lambda lib: numbers(lib).plot.barh(stacked=True, xlabel="X"),
    lambda lib: numbers(lib)["a"].plot.bar(rot=0),
    lambda lib: numbers(lib).plot.area(),
    lambda lib: numbers(lib).plot.area(stacked=False),
    lambda lib: numbers(lib).plot.pie(y="a"),
    lambda lib: numbers(lib).plot.pie(subplots=True),
    lambda lib: numbers(lib).plot.scatter(x="a", y="b", c="b"),
    lambda lib: numbers(lib).plot.hexbin(x="a", y="b", gridsize=3),
    lambda lib: numbers(lib).plot.hist(bins=3, alpha=0.5),
    lambda lib: numbers(lib).plot.hist(orientation="horizontal", stacked=True),
    lambda lib: numbers(lib).plot.box(),
    lambda lib: numbers(lib).plot.box(return_type="dict"),
    lambda lib: numbers(lib).assign(k=["x", "y", "x", "y"]).plot.hist(by="k", bins=2),
    lambda lib: numbers(lib).assign(k=["x", "y", "x", "y"]).plot.box(by="k"),
    lambda lib: numbers(lib)["a"].hist(bins=3),
    lambda lib: numbers(lib).hist(bins=2),
    lambda lib: numbers(lib).boxplot(),
    lambda lib: numbers(lib).assign(k=["x", "y", "x", "y"]).boxplot(column=["a"], by="k"),
    lambda lib: numbers(lib).assign(k=["x", "y", "x", "y"]).groupby("k").boxplot(),
    lambda lib: lib.plotting.lag_plot(numbers(lib)["b"]),
    lambda lib: lib.plotting.autocorrelation_plot(numbers(lib)["b"]),
    lambda lib: lib.plotting.scatter_matrix(numbers(lib)),
    lambda lib: lib.plotting.parallel_coordinates(numbers(lib).assign(k=["x", "y", "x", "y"]), "k"),
]


@needs_pandas
@needs_matplotlib
@pytest.mark.parametrize("call", DRAWN)
def test_a_plot_draws_what_pandas_draws(call: Callable[[ModuleType], Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas

    try:
        plt.close("all")
        theirs = summary(call(pandas))
        plt.close("all")
        ours = summary(call(fp))
    finally:
        plt.close("all")
    assert ours == theirs


@needs_pandas
@needs_matplotlib
@pytest.mark.parametrize(
    "call",
    [
        lambda lib: numbers(lib).assign(a=[1, -2, 3, 4]).plot(stacked=True),
        lambda lib: numbers(lib).assign(a=[1, -2, 3, 4]).plot.pie(y="a"),
        lambda lib: numbers(lib).plot(sharex="x"),
        lambda lib: numbers(lib).plot(logx="y"),
        lambda lib: numbers(lib).plot(cmap="a", colormap="b"),
        lambda lib: numbers(lib).plot(style="r--", color="b"),
        lambda lib: numbers(lib).plot(title=["a"]),
        lambda lib: numbers(lib).plot(subplots=True, layout=(1, 1)),
        lambda lib: numbers(lib).plot.area(logy=True),
        lambda lib: numbers(lib).plot.box(return_type="x"),
        lambda lib: numbers(lib).boxplot(layout=(1, 1)),
        lambda lib: numbers(lib)["a"].hist(legend=True, label="x"),
        lambda lib: lib.DataFrame({"s": ["a", "b"]}).plot(),
        lambda lib: lib.DataFrame({"s": ["a", "b"]}).hist(),
        lambda lib: numbers(lib).plot.hist(by=[]),
    ],
)
def test_a_plot_refuses_what_pandas_refuses(call: Callable[[ModuleType], Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas

    try:
        with pytest.raises(Exception) as theirs:
            call(pandas)
        with pytest.raises(Exception) as ours:
            call(fp)
    finally:
        plt.close("all")
    assert type(ours.value) is type(theirs.value)
    assert str(ours.value) == str(theirs.value)


@needs_matplotlib
def test_a_grouped_plot_answers_a_series_of_axes() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = numbers(fp).assign(k=["x", "y", "x", "y"])
    try:
        drawn = frame.groupby("k").plot()
        assert list(drawn.index) == ["x", "y"]
        assert all(hasattr(ax, "get_lines") for ax in drawn)
    finally:
        plt.close("all")


@needs_pandas
@pytest.mark.parametrize("path", ["DataFrameGroupBy", "SeriesGroupBy"])
def test_a_grouped_hist_has_pandas_signature(path: str) -> None:
    from pandas.core import groupby

    frame = numbers(fp).assign(k=["x", "y", "x", "y"])
    grouped = frame.groupby("k") if path == "DataFrameGroupBy" else frame.groupby("k")["a"]
    theirs = inspect.signature(getattr(groupby, path).hist)
    ours = inspect.signature(grouped.hist)
    assert list(ours.parameters) == list(theirs.parameters)[1:]


@needs_matplotlib
def test_a_grouped_hist_draws_each_group() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = numbers(fp).assign(k=["x", "y", "x", "y"])
    try:
        assert list(frame.groupby("k").hist(bins=2).index) == ["x", "y"]
        assert list(frame.groupby("k")["a"].hist(bins=2).index) == ["x", "y"]
    finally:
        plt.close("all")
