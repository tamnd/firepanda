"""`interpolate` on a column and a frame, checked against pandas.

pandas fills each gap on the straight line between the values either side of
it, with numpy's `interp`, and then puts back the gaps its `limit`,
`limit_direction` and `limit_area` say to leave. The line runs along row
positions for `linear` and along the labels for `index` and `values`. The
methods pandas hands to scipy are refused here, since there is no scipy to hand
them to.
"""

from __future__ import annotations

import importlib.util
import inspect
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

NAN = float("nan")
RUNS = [NAN, NAN, 1.0, NAN, NAN, NAN, 5.0, NAN, NAN]


def plain(values: list[Any]) -> list[Any]:
    """The values with every spelling of missing as one word."""
    return ["missing" if value is None or value != value else value for value in values]


def agrees(got: Any, want: Any) -> None:
    """The same values, labels and types, column by column for a frame."""
    assert list(got.index) == list(want.index)
    if hasattr(want, "columns"):
        assert list(got.columns) == list(want.columns)
        for name in want.columns:
            agrees(got[name], want[name])
        return
    assert plain(got.tolist()) == plain(want.tolist())
    assert str(got.dtype) == str(want.dtype)
    assert got.name == want.name


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.Series(RUNS).interpolate(),
    lambda m: m.Series(RUNS).interpolate(limit=1),
    lambda m: m.Series(RUNS).interpolate(limit=2, limit_direction="both"),
    lambda m: m.Series(RUNS).interpolate(limit_direction="backward"),
    lambda m: m.Series(RUNS).interpolate(limit_direction="backward", limit=1),
    lambda m: m.Series(RUNS).interpolate(limit_direction="BOTH"),
    lambda m: m.Series(RUNS).interpolate(limit_area="inside"),
    lambda m: m.Series(RUNS).interpolate(limit_area="outside", limit_direction="both"),
    lambda m: m.Series(RUNS).interpolate(limit_area="outside", limit_direction="both", limit=1),
    lambda m: m.Series(RUNS, index=[9, 8, 7, 6, 5, 4, 3, 2, 1]).rename(4).interpolate(),
    lambda m: m.Series(RUNS, index=[1, 1, 2, 2, 3, 3, 4, 4, 5]).interpolate(),
    lambda m: m.Series([1.0, None, 3], index=[0, 1, 10]).interpolate(method="index"),
    lambda m: m.Series([1.0, None, 3], index=[0, 1, 10]).interpolate(method="values"),
    lambda m: m.Series([0.1, None, None, 0.7]).interpolate(),
    lambda m: m.Series([1.0, None, float("inf"), None, 3]).interpolate(),
    lambda m: m.Series([1.0, None, 3], dtype="float32").interpolate(),
    lambda m: m.Series([None, None], dtype="float64").interpolate(),
    lambda m: m.Series([], dtype="float64").interpolate(method="nope"),
    lambda m: m.Series([1, 2, 3]).interpolate(limit_direction="ignored when nothing is missing"),
    lambda m: m.Series([1.0, None, 3]).interpolate(order=2),
    lambda m: m.DataFrame(
        {"a": [1.0, None, 3], "b": [1, 2, 3], "f": [True, False, True]}
    ).interpolate(),
    lambda m: m.DataFrame({"a": [1.0, None, 3]}, index=[4, 4, 5]).interpolate(),
    lambda m: m.DataFrame({"a": [None, 2.0, None, 8], "b": [1.0, None, None, None]}).interpolate(
        limit_direction="both"
    ),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Each limit and direction, repeated and falling labels, the index as the line,
    an infinity, float32, nothing to fill, and columns pandas leaves alone."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_whole_numbers_with_a_gap_are_floats(firepanda: ModuleType) -> None:
    """pandas holds them as float64 already, so the answer is float64 here too."""
    answer = firepanda.Series([1, None, 3, None]).interpolate()
    assert answer.tolist() == [1.0, 2.0, 3.0, 3.0]
    assert str(answer.dtype) == "float64"


def test_in_place_hands_back_the_object(firepanda: ModuleType) -> None:
    """`interpolate` is in the half of pandas that answers the object itself."""
    import pandas as pd

    for m in (firepanda, pd):
        column = m.Series([1.0, None, 3])
        frame = m.DataFrame({"a": [1.0, None, 3]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert column.interpolate(inplace=True) is column
            assert frame.interpolate(inplace=True) is frame
        assert column.tolist() == [1.0, 2.0, 3.0]
        assert frame["a"].tolist() == [1.0, 2.0, 3.0]


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.Series([1.0, None]).interpolate(method="pad"),
    lambda m: m.Series([1.0, None]).interpolate(method="bfill", limit_direction="x"),
    lambda m: m.Series([1.0, None]).interpolate(method="nope", limit=0),
    lambda m: m.Series([1.0, None]).interpolate(limit_direction="x", limit=0),
    lambda m: m.Series([1.0, None]).interpolate(limit_area="x"),
    lambda m: m.Series([1.0, None]).interpolate(limit=0),
    lambda m: m.Series([1.0, None]).interpolate(limit=1.5),
    lambda m: m.Series([1.0, None]).interpolate(limit=True),
    lambda m: m.Series([1.0, None]).interpolate(method=None),
    lambda m: m.Series([1.0, None]).interpolate(axis=1),
    lambda m: m.Series([1.0, None, 3], index=["a", "b", "c"]).interpolate(method="index"),
    lambda m: m.Series([1.0, None, 3]).interpolate(method="time"),
    lambda m: m.Series([1.0, None, 3]).interpolate(method="spline"),
    lambda m: m.Series(["a", None]).interpolate(limit=0),
    lambda m: m.DataFrame({"a": [1.0, None, 3], "s": ["x", "y", None]}).interpolate(),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The same class or a subclass of it, and the same message, in pandas' order."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert isinstance(mine.value, type(theirs.value))
    assert str(mine.value) == str(theirs.value)


@pytest.mark.parametrize("method", ["nearest", "cubic", "akima"])
def test_a_scipy_method_is_refused(firepanda: ModuleType, method: str) -> None:
    """pandas hands these to scipy, which firepanda does not have."""
    with pytest.raises(NotImplementedError, match="scipy"):
        firepanda.Series([1.0, None, 3]).interpolate(method=method)


@pytest.mark.parametrize("owner", ["DataFrame", "Series"])
def test_the_signature_is_pandas_signature(firepanda: ModuleType, owner: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(getattr(firepanda, owner).interpolate).parameters
    yours = inspect.signature(getattr(pd, owner).interpolate).parameters
    assert [(p.name, p.kind, p.default) for p in ours.values()] == [
        (p.name, p.kind, p.default) for p in yours.values()
    ]
