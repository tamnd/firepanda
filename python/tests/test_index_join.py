"""`Index.join` and the index members around it, checked against pandas.

The join is compared over every pairing of a few indexes, sorted and not,
unique and not, named and not, for each `how` and `sort`, because pandas takes
a different path through each of those and the paths answer rows in different
orders. The rest are `asof`, `asof_locs`, `get_indexer_for`,
`get_indexer_non_unique`, `sortlevel`, `groupby`, `view`, `str` and `shift`.
"""

from __future__ import annotations

import importlib.util
import itertools
import re
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def facts(answer: Any) -> Any:
    """What is compared: an index as its type, labels and name, an array as a list."""
    if isinstance(answer, tuple):
        return tuple(facts(each) for each in answer)
    if isinstance(answer, dict):
        return {plain(key): facts(each) for key, each in answer.items()}
    if isinstance(answer, list):
        return [plain(each) for each in answer]
    if hasattr(answer, "name") and hasattr(answer, "tolist"):
        return (type(answer).__name__, [plain(v) for v in answer.tolist()], answer.name)
    if hasattr(answer, "tolist") and hasattr(answer, "__len__"):
        return [plain(v) for v in answer.tolist()]
    return plain(answer)


def plain(value: Any) -> Any:
    """A value as compared: a gap as None, an instant as text, a numpy number as Python."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    if type(value).__name__ in ("Timestamp", "NaTType"):
        return None if str(value) == "NaT" else str(value)
    return value.item() if hasattr(value, "item") and not hasattr(value, "__len__") else value


LEFT = {
    "unique": ([3, 1, 2], "k"),
    "repeated": ([3, 1, 2, 1], "k"),
    "sorted": ([1, 2, 3], None),
    "sorted-repeated": ([1, 1, 2, 4], "j"),
}
RIGHT = {
    "unique": ([2, 4, 3], "k"),
    "unnamed": ([2, 4, 3], None),
    "renamed": ([2, 4, 3], "j"),
    "repeated": ([2, 1, 1, 5], None),
    "empty": ([], None),
}
JOINS = list(itertools.product(LEFT, RIGHT, ["left", "right", "inner", "outer"], [False, True]))


@pytest.mark.parametrize(("left", "right", "how", "sort"), JOINS)
def test_a_join_answers_pandas_rows_in_pandas_order(
    firepanda: ModuleType, left: str, right: str, how: str, sort: bool
) -> None:
    """The joined labels, their name and both sides' positions, None where pandas says None."""
    import pandas as pd

    def join(m: ModuleType) -> Any:
        mine, theirs = m.Index(LEFT[left][0], name=LEFT[left][1]), RIGHT[right]
        other = m.Index(theirs[0] or [0], name=theirs[1])[: len(theirs[0])]
        return mine.join(other, how=how, sort=sort, return_indexers=True)

    assert facts(join(firepanda)) == facts(join(pd))


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.Index([3, 1, 2]).join(m.Index([2, 4]), how="outer"),
    lambda m: m.Index([3, 1, 2], name="k").join([1, 2]),
    lambda m: m.Index([1, 2]).join(m.Index([1.5, 2.0]), how="outer", return_indexers=True),
    lambda m: m.Index([1.0, None]).join(m.Index([None, 2.0]), how="outer", return_indexers=True),
    lambda m: m.Index(["b", "a", "c"], name="t").join(m.Index(["a", "z"]), how="outer"),
    lambda m: m.Index([1, 2]).join(m.Index([1, 2]), how="inner", return_indexers=True),
    lambda m: m.Index([1]).join(m.Index([1]), level=0),
    lambda m: m.DatetimeIndex(["2024-01-03", "2024-01-01"], name="w").join(
        m.DatetimeIndex(["2024-01-01", "2024-01-05"]), how="outer", return_indexers=True
    ),
    lambda m: m.DatetimeIndex(["2024-01-03", "2024-01-01"], name="w").join(
        m.DatetimeIndex(["2024-01-01"]), how="inner"
    ),
    lambda m: m.Index([3, 1, 2]).get_indexer_for([1, 5, 3]),
    lambda m: m.Index([3, 1, 2, 1]).get_indexer_for([1, 5, 3]),
    lambda m: m.Index([3, 1, 2, 1]).get_indexer_non_unique([1, 5, 3]),
    lambda m: m.Index([3, 1, 2]).get_indexer_non_unique(m.Index([2, 9])),
    lambda m: m.Index([1.0, None]).get_indexer_non_unique([float("nan")]),
    lambda m: m.DatetimeIndex(["2024-01-02", "2024-01-01"]).get_indexer_for(
        m.DatetimeIndex(["2024-01-01", "2024-01-09"])
    ),
    lambda m: m.DatetimeIndex(["2024-01-02", "2024-01-01"]).get_indexer(["2024-01-01"]),
    lambda m: m.Index([3, 1, 2], name="k").sortlevel(),
    lambda m: m.Index([3, 1, 2]).sortlevel(ascending=False),
    lambda m: m.Index([3, 1, 2]).sortlevel(ascending=[False]),
    lambda m: m.Index(["b", "a", "c"]).sortlevel(),
    lambda m: m.Index([1.0, None, 3.0]).sortlevel(),
    lambda m: m.Index([1.0, None, 3.0]).sortlevel(na_position="last"),
    lambda m: m.Index([1, 3, 5]).asof(4),
    lambda m: m.Index([1, 3, 5]).asof(3),
    lambda m: m.Index([1, 3, 5]).asof(0),
    lambda m: m.Index([1, 3, 5]).asof(9),
    lambda m: m.Index([1, 3, 3, 5]).asof(3),
    lambda m: m.Index([5, 3, 1]).asof(4),
    lambda m: m.Index([5, 3, 1]).asof(0),
    lambda m: m.Index([5, 3, 1]).asof(6),
    lambda m: m.Index(["a", "c"]).asof("b"),
    lambda m: m.Index([]).asof(1) if m.__name__ == "pandas" else m.Index([0])[:0].asof(1),
    lambda m: m.DatetimeIndex(["2024-01-01", "2024-01-03"]).asof("2024-01-02"),
    lambda m: m.Index([1, 3, 5, 7]).asof_locs(m.Index([0, 2, 3, 8]), _mask(m, [1, 0, 1, 1])),
    lambda m: m.Index([1, 3, 5, 7]).asof_locs(m.Index([4, 1]), _mask(m, [0, 1, 1, 1])),
    lambda m: m.Index(["b", "a", "c"], name="t").groupby(["x", "y", "x"]),
    lambda m: m.Index(["b", "a", "c"]).groupby(m.Index(["x", "y", "x"])),
    lambda m: m.Index(["b", "a", "c"]).groupby(["x", "y"]),
    lambda m: m.Index(["b", "a", "c"]).groupby(["x", None, "x"]),
    lambda m: m.Index([1, 2, 3]).groupby([2, 1, 2]),
    lambda m: m.DatetimeIndex(["2024-01-01", "2024-01-02"]).groupby(["a", "a"]),
    lambda m: m.Index(["b", "a"], name="t").view(),
    lambda m: type(m.DatetimeIndex(["2024-01-01"]).view()).__name__,
    lambda m: m.Index(["b", "a"]).view().is_(m.Index(["b", "a"])),
    lambda m: m.Index(["b", "a"], name="t").str.upper(),
    lambda m: m.Index(["b", "ab"], name="t").str.len(),
    lambda m: m.Index(["b", "ab"]).str.contains("a"),
    lambda m: m.Index(["b", "ab"]).str.startswith("a"),
    lambda m: m.Index(["b", "ab"]).str.replace("b", "c"),
    lambda m: m.Index(["b", "ab"]).str.cat(sep="-"),
    lambda m: list(m.Index([3, 1]).array),
]


def _mask(m: ModuleType, flags: list[int]) -> Any:
    """A mask as pandas wants it, a numpy array, or a list for firepanda."""
    if m.__name__ != "pandas":
        return [bool(flag) for flag in flags]
    import numpy as np

    return np.array(flags, dtype=bool)


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Every member, on labels of each kind, against pandas."""
    import pandas as pd

    assert facts(build(firepanda)) == facts(build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.Index([1]).join(m.Index([1]), how="nope"),
    lambda m: m.Index([3, 1]).sortlevel(ascending=[False, True]),
    lambda m: m.Index([3, 1]).sortlevel(ascending="x"),
    lambda m: m.Index([3, 1]).sortlevel(ascending=[1]),
    lambda m: m.Index([3, 1]).shift(),
    lambda m: m.Index([3, 1]).shift(1, freq="D"),
    lambda m: m.Index([3, 1, 2]).asof(2),
    lambda m: m.Index([1, 3, 3, 5]).asof(4),
    lambda m: m.Index([1, 3]).asof([2]),
    lambda m: m.Index([1, 3]).str,
    lambda m: m.Index([1.0, None, 3.0]).asof(2),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_mistakes_raise_as_pandas_raises(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """The same error type as pandas, and the same words where they are not about types.

    The type is compared by name, since pandas' own errors are its own classes,
    and firepanda's `InvalidArgumentError` is the `ValueError` it subclasses.
    """
    import pandas as pd

    with pytest.raises(Exception) as expected:
        build(pd)
    words = str(expected.value)
    ending = "$"
    if words.startswith("Can only use .str"):
        words, ending = "Can only use .str accessor with string values, not ", ""
    with pytest.raises(Exception, match="^" + re.escape(words) + ending) as found:
        build(firepanda)
    named = type(found.value).__name__.replace("InvalidArgumentError", "ValueError")
    assert named == type(expected.value).__name__


def test_str_is_the_accessor_off_the_class(firepanda: ModuleType) -> None:
    """Read off the class it is something called with the labels, as in pandas."""
    import inspect

    assert list(inspect.signature(firepanda.Index.str).parameters) == ["data"]


def test_a_join_of_text_and_numbers_is_refused(firepanda: ModuleType) -> None:
    """pandas answers labels of mixed types, which firepanda has no index for."""
    with pytest.raises(NotImplementedError, match="mixed values"):
        firepanda.Index(["a"]).join(firepanda.Index([1]), how="outer")


def test_view_as_another_type_is_refused(firepanda: ModuleType) -> None:
    """Reading the labels' bytes as another type needs a buffer of fixed width values."""
    with pytest.raises(NotImplementedError):
        firepanda.Index([1, 2]).view("int32")
