"""`merge_ordered`, checked against pandas.

An ordered merge is a merge sorted by the key, with an option to carry each
side's last row down over the rows where it had no match, and an option to
merge group by group. The answers are compared as columns, types and labels.
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


def facts(frame: Any) -> tuple[Any, ...]:
    """What is compared: labels, index, column types and values with gaps as None."""

    def plain(value: Any) -> Any:
        if value is None or value != value:
            return None
        return str(value) if hasattr(value, "value") else value

    return (
        list(frame.columns),
        [plain(v) for v in frame.index.tolist()],
        ["string" if str(frame[c].dtype) == "str" else str(frame[c].dtype) for c in frame.columns],
        [[plain(v) for v in frame[c].tolist()] for c in frame.columns],
    )


def grouped(m: ModuleType) -> tuple[Any, Any]:
    """The pandas documentation's frames: a left in two groups and a right with no group."""
    left = m.DataFrame(
        {
            "key": ["a", "c", "e", "a", "c", "e"],
            "lvalue": [1, 2, 3, 1, 2, 3],
            "group": ["a", "a", "a", "b", "b", "b"],
        }
    )
    right = m.DataFrame({"key": ["b", "c", "d"], "rvalue": [1, 2, 3]})
    return left, right


def numbers(m: ModuleType) -> tuple[Any, Any]:
    """Unsorted integer keys, with a gap in the right values."""
    left = m.DataFrame({"k": [3, 1, 2], "v": [1, 2, 3]})
    right = m.DataFrame({"k": [2, 5, 0], "w": [1.5, 2.5, None]})
    return left, right


def named(m: ModuleType) -> tuple[Any, Any]:
    """Keys with different names and a clashing text column."""
    left = m.DataFrame({"a": [1, 3, 5], "s": ["p", "q", "r"]})
    right = m.DataFrame({"b": [2, 3, 6], "s": ["x", "y", "z"]})
    return left, right


def timed(m: ModuleType) -> tuple[Any, Any]:
    """Instant keys and an instant value."""
    left = m.DataFrame({"t": m.to_datetime(m.Series(["2020-01-01", "2020-01-03"])), "a": [1, 2]})
    right = m.DataFrame(
        {
            "t": m.to_datetime(m.Series(["2020-01-02", "2020-01-04"])),
            "seen": m.to_datetime(m.Series(["2021-05-05", "2021-06-06"])),
        }
    )
    return left, right


def sides(m: ModuleType) -> tuple[Any, Any]:
    """A right frame in groups whose values the left lacks for one group."""
    left = m.DataFrame({"k": [1, 2, 3], "g": ["x", "x", "y"], "v": [10, 20, 30]})
    right = m.DataFrame({"k": [2, 3, 1, 4], "g": ["x", "x", "z", "y"], "w": [1, 2, 3, 4]})
    return left, right


def ordered(m: ModuleType, pair: tuple[Any, Any], **kw: Any) -> Any:
    """merge_ordered on a pair of frames."""
    return m.merge_ordered(pair[0], pair[1], **kw)


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: ordered(m, grouped(m), fill_method="ffill", left_by="group"),
    lambda m: ordered(m, grouped(m), left_by="group"),
    lambda m: ordered(m, grouped(m), left_by=["group"], how="inner"),
    lambda m: ordered(m, grouped(m), on="key"),
    lambda m: ordered(m, grouped(m), on="key", fill_method="ffill"),
    lambda m: ordered(m, grouped(m), on="key", how="left"),
    lambda m: ordered(m, numbers(m), on="k"),
    lambda m: ordered(m, numbers(m), on="k", fill_method="ffill"),
    lambda m: ordered(m, numbers(m), on="k", how="right", fill_method="ffill"),
    lambda m: ordered(m, numbers(m), on="k", how="inner"),
    lambda m: ordered(m, numbers(m), on="k", how="left"),
    lambda m: ordered(m, numbers(m)),
    lambda m: ordered(m, named(m), left_on="a", right_on="b"),
    lambda m: ordered(m, named(m), left_on="a", right_on="b", fill_method="ffill"),
    lambda m: ordered(m, named(m), left_on="a", right_on="b", suffixes=("_l", "_r")),
    lambda m: ordered(m, timed(m), on="t"),
    lambda m: ordered(m, timed(m), on="t", fill_method="ffill"),
    lambda m: ordered(m, sides(m), on="k", left_by="g"),
    lambda m: ordered(m, sides(m), on="k", left_by="g", fill_method="ffill"),
    lambda m: ordered(m, sides(m), on="k", right_by="g"),
    lambda m: ordered(m, (sides(m)[1], sides(m)[0]), on="k", right_by="g", fill_method="ffill"),
    lambda m: ordered(m, (grouped(m)[1], grouped(m)[0]), on="key", right_by="group"),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Every join, fill, group and key spelling."""
    import pandas as pd

    assert facts(build(firepanda)) == facts(build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: ordered(m, grouped(m), on="key", fill_method="bfill"),
    lambda m: ordered(m, grouped(m), left_by="group", right_by="key"),
    lambda m: ordered(m, grouped(m), left_by="nope"),
    lambda m: ordered(m, grouped(m), right_by="group"),
    lambda m: ordered(m, grouped(m), on="nope"),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_mistakes_raise_as_pandas_raises(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """The same error type and words as pandas."""
    import pandas as pd

    with pytest.raises(Exception) as expected:
        build(pd)
    with pytest.raises(type(expected.value), match="^" + re.escape(str(expected.value)) + "$"):
        build(firepanda)
