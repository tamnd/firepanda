"""`merge_asof`, checked against pandas.

Each left row takes the last right row at or before its key, or with a
direction the first at or after it or the nearest, among right rows with the
same `by` values and within a tolerance. The answers are compared as columns,
types and labels.
"""

from __future__ import annotations

import datetime
import importlib.util
import re
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def facts(frame: Any) -> tuple[Any, ...]:
    """What is compared: labels, index, column types and values with gaps as None."""

    def plain(value: Any) -> Any:
        if value is None or value != value:
            return None
        return str(value) if hasattr(value, "value") else value

    return (
        list(frame.columns),
        [plain(v) for v in frame.index.tolist()],
        frame.index.name,
        ["string" if str(frame[c].dtype) == "str" else str(frame[c].dtype) for c in frame.columns],
        [[plain(v) for v in frame[c].tolist()] for c in frame.columns],
    )


def quotes(m: ModuleType) -> tuple[Any, Any]:
    """Trades and quotes on whole number times, two tickers."""
    trades = m.DataFrame(
        {
            "time": [1, 3, 5, 5, 8, 12],
            "ticker": ["A", "B", "A", "B", "A", "C"],
            "qty": [10, 20, 30, 40, 50, 60],
        }
    )
    quotes = m.DataFrame(
        {
            "time": [0, 2, 3, 5, 7, 9],
            "ticker": ["A", "B", "A", "B", "A", "B"],
            "bid": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            "size": [1, 2, 3, 4, 5, 6],
        }
    )
    return trades, quotes


def timed(m: ModuleType) -> tuple[Any, Any]:
    """The same shape on instants."""
    left = m.DataFrame(
        {
            "t": m.to_datetime(
                m.Series(["2020-01-01 00:00:00", "2020-01-01 00:00:05", "2020-01-01 00:00:30"])
            ),
            "a": [1, 2, 3],
        }
    )
    right = m.DataFrame(
        {
            "t": m.to_datetime(m.Series(["2020-01-01 00:00:01", "2020-01-01 00:00:04"])),
            "b": [1.5, 2.5],
        }
    )
    return left, right


def floats(m: ModuleType) -> tuple[Any, Any]:
    """Float keys with names that clash on both sides."""
    left = m.DataFrame({"k": [0.5, 1.0, 2.5, 4.0], "v": [1, 2, 3, 4]})
    right = m.DataFrame({"k": [1.0, 2.0, 3.0], "v": [10.0, 20.0, 30.0], "w": [1, 2, 3]})
    return left, right


def asof(m: ModuleType, sides: tuple[Any, Any], **kw: Any) -> Any:
    """merge_asof on a pair of frames."""
    return m.merge_asof(sides[0], sides[1], **kw)


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: asof(m, quotes(m), on="time"),
    lambda m: asof(m, quotes(m), on="time", by="ticker"),
    lambda m: asof(m, quotes(m), on="time", by=["ticker"], direction="forward"),
    lambda m: asof(m, quotes(m), on="time", by="ticker", direction="nearest"),
    lambda m: asof(m, quotes(m), on="time", direction="nearest"),
    lambda m: asof(m, quotes(m), on="time", allow_exact_matches=False),
    lambda m: asof(m, quotes(m), on="time", direction="forward", allow_exact_matches=False),
    lambda m: asof(m, quotes(m), on="time", direction="nearest", allow_exact_matches=False),
    lambda m: asof(m, quotes(m), on="time", tolerance=1),
    lambda m: asof(m, quotes(m), on="time", tolerance=0, direction="nearest"),
    lambda m: asof(m, quotes(m), on="time", by="ticker", tolerance=2, direction="forward"),
    lambda m: asof(
        m, quotes(m), left_on="time", right_on="time", left_by="ticker", right_by="ticker"
    ),
    lambda m: m.merge_asof(quotes(m)[0], quotes(m)[1].iloc[:0], on="time"),
    lambda m: m.merge_asof(
        quotes(m)[0].rename(columns={"time": "at"}), quotes(m)[1], left_on="at", right_on="time"
    ),
    lambda m: m.merge_asof(
        quotes(m)[0].rename(columns={"ticker": "sym"}),
        quotes(m)[1],
        on="time",
        left_by="sym",
        right_by="ticker",
    ),
    lambda m: asof(m, floats(m), on="k"),
    lambda m: asof(m, floats(m), on="k", suffixes=("_left", "_right")),
    lambda m: asof(m, floats(m), on="k", tolerance=0.25, direction="nearest"),
    lambda m: asof(m, floats(m), on="k", tolerance=1),
    lambda m: asof(m, timed(m), on="t"),
    lambda m: asof(m, timed(m), on="t", tolerance=m.Timedelta("2s")),
    lambda m: asof(m, timed(m), on="t", tolerance=datetime.timedelta(seconds=10)),
    lambda m: asof(m, timed(m), on="t", direction="forward"),
    lambda m: m.merge_asof(
        floats(m)[0].set_index("k"),
        floats(m)[1].set_index("k"),
        left_index=True,
        right_index=True,
    ),
    lambda m: asof(m, quotes(m), on="time", by="ticker", direction="backward").tail(3),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Every direction, tolerance, `by`, key spelling and key type."""
    import pandas as pd

    assert facts(build(firepanda)) == facts(build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: asof(m, quotes(m), on="time", direction="sideways"),
    lambda m: asof(m, quotes(m), on="time", allow_exact_matches=1),
    lambda m: asof(m, quotes(m), on="time", left_on="time"),
    lambda m: asof(m, quotes(m), on=["time", "ticker"]),
    lambda m: asof(m, quotes(m), left_on="time"),
    lambda m: asof(m, quotes(m), right_on="time"),
    lambda m: asof(m, quotes(m), on="ticker"),
    lambda m: asof(m, quotes(m), on="time", by="ticker", left_by="ticker"),
    lambda m: asof(m, quotes(m), on="time", right_by="ticker"),
    lambda m: asof(m, quotes(m), on="time", left_by="ticker"),
    lambda m: asof(m, quotes(m), on="time", left_by=["ticker", "qty"], right_by=["ticker"]),
    lambda m: asof(m, quotes(m), on="time", tolerance=-1),
    lambda m: asof(m, quotes(m), on="time", tolerance=1.5),
    lambda m: asof(m, floats(m), on="k", tolerance="x"),
    lambda m: asof(m, timed(m), on="t", tolerance=1),
    lambda m: asof(m, timed(m), on="t", tolerance=m.Timedelta("-1s")),
    lambda m: m.merge_asof(floats(m)[0], quotes(m)[1].rename(columns={"time": "k"}), on="k"),
    lambda m: m.merge_asof(quotes(m)[0].iloc[::-1], quotes(m)[1], on="time"),
    lambda m: m.merge_asof(quotes(m)[0], quotes(m)[1].iloc[::-1], on="time"),
    lambda m: m.merge_asof(floats(m)[0].assign(k=[None, 1.0, 2.0, 3.0]), floats(m)[1], on="k"),
    lambda m: m.merge_asof(quotes(m)[0], [1, 2], on="time"),
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
    if kind.__name__ == "MergeError":
        from firepanda.errors import MergeError

        kind = MergeError
    with pytest.raises(kind, match="^" + re.escape(str(expected.value)) + "$"):
        build(firepanda)


def test_a_key_on_the_labels_of_one_side_is_refused(firepanda: ModuleType) -> None:
    """pandas fills the other side's key column from the labels; that is not supported yet."""
    left, right = floats(firepanda)
    with pytest.raises(NotImplementedError, match="labels of one side"):
        firepanda.merge_asof(left.set_index("k"), right, left_index=True, right_on="k")
