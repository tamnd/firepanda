"""The window members that read every window in Python, checked against pandas.

`first`, `last`, `nunique`, `apply`, `aggregate`, `pipe`, `cov` and `corr` on
rolling and expanding windows, and `aggregate`, `cov` and `corr` on a decay.
The answers are compared as labels, names and values, with gaps as None and
numbers to twelve places, because a covariance worked out from a window's own
mean and one worked out from running sums land on different last digits.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def plain(value: Any) -> Any:
    """A value as compared: a gap as None and a number rounded."""
    if value is None or value != value:
        return None
    return round(float(value), 12) if isinstance(value, float) else value


def facts(answer: Any) -> tuple[Any, ...]:
    """What is compared: the kind, the labels, the names, the types and the values."""
    if hasattr(answer, "columns"):
        return (
            "frame",
            list(answer.columns),
            answer.index.tolist(),
            [str(answer[c].dtype) for c in answer.columns],
            [[plain(v) for v in answer[c].tolist()] for c in answer.columns],
        )
    return (
        "column",
        answer.name,
        answer.index.tolist(),
        str(answer.dtype),
        [plain(v) for v in answer.tolist()],
    )


def column(m: ModuleType) -> Any:
    """Six values with a gap and a repeat."""
    return m.Series([1.0, 2.0, None, 4.0, 4.0, 6.0], name="x")


def other(m: ModuleType) -> Any:
    """A second column on the same labels."""
    return m.Series([2.0, 1.0, 3.0, 5.0, 4.0, 6.0])


def gappy(m: ModuleType) -> Any:
    """A second column with its gap in another row."""
    return m.Series([2.0, 1.0, 3.0, 5.0, None, 6.0])


def frame(m: ModuleType) -> Any:
    """A whole number column and a float one."""
    return m.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5.0, 3.0, 4.0, 1.0, 2.0]})


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: column(m).rolling(2).first(),
    lambda m: column(m).rolling(2).last(),
    lambda m: column(m).rolling(3, min_periods=1).nunique(),
    lambda m: column(m).rolling(2, min_periods=0).first(),
    lambda m: column(m).rolling(2, min_periods=0).nunique(),
    lambda m: column(m).rolling(3, center=True).first(),
    lambda m: column(m).rolling(3, closed="both").last(),
    lambda m: column(m).rolling(3, closed="neither", min_periods=1).last(),
    lambda m: column(m).rolling(3, closed="left", min_periods=1).first(),
    lambda m: column(m).expanding().first(),
    lambda m: column(m).expanding().nunique(),
    lambda m: m.Series([1, 2, 3]).rolling(2).first(),
    lambda m: frame(m).rolling(2).first(),
    lambda m: frame(m).rolling(3, min_periods=1).nunique(),
    lambda m: column(m).rolling(2).apply(lambda w: w.sum() * 2),
    lambda m: column(m).rolling(2).apply(lambda w: w[0], raw=True),
    lambda m: column(m).rolling(2, min_periods=1).apply(lambda w: w.index[-1]),
    lambda m: column(m).rolling(2, min_periods=1).apply(len),
    lambda m: column(m).rolling(2).apply(lambda w: True),
    lambda m: (
        column(m)
        .rolling(2, min_periods=1)
        .apply(lambda w, k, n=1: w.sum() * k + n, args=(3,), kwargs={"n": 5})
    ),
    lambda m: column(m).rolling(2, step=2).apply(sum),
    lambda m: column(m).rolling(2, closed="both").apply(len),
    lambda m: column(m).rolling(0).apply(len),
    lambda m: column(m).expanding(2).apply(len),
    lambda m: m.Series([None, None, 1.0]).rolling(2, min_periods=0).apply(len),
    lambda m: frame(m).rolling(2).apply(lambda w: w.iloc[-1] - w.iloc[0]),
    lambda m: column(m).rolling(2).agg("sum"),
    lambda m: column(m).rolling(2).aggregate("max"),
    lambda m: column(m).rolling(2).agg("quantile", 0.5),
    lambda m: column(m).rolling(2).agg(["sum", "mean"]),
    lambda m: column(m).rolling(2).agg(["sum", lambda w: 1]),
    lambda m: column(m).rolling(2).agg({"p": "sum"}),
    lambda m: column(m).rolling(2).agg(sum),
    lambda m: column(m).expanding().agg(["min", "max"]),
    lambda m: frame(m).rolling(2).agg({"a": "sum", "b": "max"}),
    lambda m: frame(m).rolling(2).agg(lambda w: w.max()),
    lambda m: column(m).rolling(2).pipe(lambda r: r.sum()),
    lambda m: column(m).rolling(2).pipe((lambda a, r: r.sum(), "r"), 1),
    lambda m: column(m).rolling(3).corr(other(m)),
    lambda m: column(m).rolling(3).cov(other(m)),
    lambda m: column(m).rolling(3).cov(other(m), ddof=0),
    lambda m: column(m).rolling(3, min_periods=2).corr(),
    lambda m: column(m).rolling(2).corr(m.Series([1.0, 3.0, 2.0], index=[3, 4, 5])),
    lambda m: column(m).expanding().cov(other(m)),
    lambda m: column(m).expanding(2).corr(other(m)),
    lambda m: m.Series([1.0, 1, 1, 1]).rolling(3).corr(m.Series([1.0, 2, 3, 4])),
    lambda m: frame(m).rolling(3).corr(frame(m)["a"]),
    lambda m: (
        frame(m).rolling(3).cov(m.DataFrame({"b": [1.0, 2.0, 4.0, 8.0, 16.0], "c": [1.0] * 5}))
    ),
    lambda m: frame(m).rolling(3).cov(frame(m), pairwise=False),
    lambda m: frame(m).rolling(3).corr(pairwise=False),
    lambda m: column(m).rolling(3).corr(frame(m)),
    lambda m: column(m).ewm(span=3).cov(gappy(m)),
    lambda m: column(m).ewm(span=3).cov(gappy(m), bias=True),
    lambda m: column(m).ewm(span=3).corr(gappy(m)),
    lambda m: column(m).ewm(com=0.5, adjust=False).cov(gappy(m)),
    lambda m: column(m).ewm(com=0.5, adjust=False).corr(gappy(m)),
    lambda m: column(m).ewm(alpha=0.3, ignore_na=True).cov(gappy(m), bias=True),
    lambda m: column(m).ewm(alpha=0.3, ignore_na=True).corr(gappy(m)),
    lambda m: column(m).ewm(halflife=2, min_periods=3).cov(gappy(m)),
    lambda m: column(m).ewm(span=3, adjust=False, ignore_na=True).corr(gappy(m)),
    lambda m: column(m).ewm(span=3).corr(),
    lambda m: column(m).ewm(span=3).cov(),
    lambda m: column(m).ewm(span=3).cov(m.Series([1.0, 3.0, 2.0], index=[3, 4, 5])),
    lambda m: m.Series([1.0, 1, 1, 1]).ewm(span=3).corr(m.Series([1.0, 2, 3, 4])),
    lambda m: frame(m).ewm(span=3).cov(frame(m)["a"]),
    lambda m: frame(m).ewm(span=3).corr(pairwise=False),
    lambda m: column(m).ewm(span=3).agg("mean"),
    lambda m: column(m).ewm(span=3).agg(["mean", "std"]),
    lambda m: frame(m).ewm(span=3).agg({"a": "mean"}),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Every member over every placement, on a column and on a frame."""
    import pandas as pd

    assert facts(build(firepanda)) == facts(build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: column(m).rolling(2).apply(5),
    lambda m: column(m).rolling(2).apply(lambda w: "x"),
    lambda m: column(m).rolling(2).apply(lambda w: None),
    lambda m: column(m).rolling(2).apply(sum, raw=1),
    lambda m: column(m).rolling(2).apply(sum, engine="numba"),
    lambda m: column(m).rolling(2).apply(sum, engine="other"),
    lambda m: column(m).rolling(2).agg("nope"),
    lambda m: column(m).expanding().agg("nope"),
    lambda m: frame(m).rolling(2).agg({"z": "sum"}),
    lambda m: column(m).rolling(3).corr([1, 2, 3]),
    lambda m: m.Series(["a", "b"]).rolling(1).first(),
    lambda m: m.Series(["a", "b"]).rolling(1).apply(len),
    lambda m: m.DataFrame({"a": [1, 2], "t": ["x", "y"]}).rolling(1).nunique(),
    lambda m: column(m).rolling(2).pipe((lambda r: r, "r"), r=1),
    lambda m: column(m).ewm(span=3).agg(lambda w: 1),
    lambda m: column(m).ewm(span=3).agg("nope"),
    lambda m: column(m).ewm(span=3).agg("apply"),
    lambda m: column(m).ewm(span=3).corr([1]),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_mistakes_raise_as_pandas_raises(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """The same error type and words as pandas."""
    import pandas as pd

    with pytest.raises(Exception) as expected:
        build(pd)
    kind = type(expected.value)
    if kind.__name__ == "DataError":
        from firepanda.errors import DataError

        kind = DataError
    with pytest.raises(kind, match="^" + re.escape(str(expected.value)) + "$"):
        build(firepanda)


REFUSED: list[Callable[[Any], Any]] = [
    lambda m: frame(m).rolling(2).agg(["sum", "min"]),
    lambda m: frame(m).rolling(2).agg({"a": ["sum", "max"]}),
    lambda m: frame(m).rolling(2).corr(),
    lambda m: frame(m).rolling(2).cov(frame(m), pairwise=True),
    lambda m: frame(m).rolling(2).first(numeric_only=True),
    lambda m: frame(m).ewm(span=3).corr(),
]


@pytest.mark.parametrize("build", REFUSED)
def test_answers_with_two_levels_of_labels_are_refused(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """pandas answers these with two levels of labels, which is not here yet."""
    with pytest.raises(NotImplementedError):
        build(firepanda)


def test_a_window_without_an_answer_is_a_nan_and_not_a_gap(firepanda: ModuleType) -> None:
    """pandas answers NaN there, and so do the kernel reductions, so a null would stand out."""
    s = column(firepanda)
    answers = [
        s.rolling(2).first(),
        s.rolling(2).nunique(),
        s.rolling(2).apply(len),
        s.rolling(3).cov(other(firepanda)),
        firepanda.Series([1.0, 1.0, 1.0]).rolling(2).corr(firepanda.Series([1.0, 2.0, 3.0])),
    ]
    for answer in answers:
        first = answer.tolist()[0]
        assert first is not None and first != first
