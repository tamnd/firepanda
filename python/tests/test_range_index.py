"""`RangeIndex`, checked against pandas.

The repr, the three numbers, the labels and the name are compared for every
way of giving the range, and the mistakes by type and words.
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


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.RangeIndex(5),
    lambda m: m.RangeIndex(1, 10, 3, name="r"),
    lambda m: m.RangeIndex(range(2, 6)),
    lambda m: m.RangeIndex.from_range(range(2, 6), name="x"),
    lambda m: m.RangeIndex(5, dtype="int32"),
    lambda m: (
        m.RangeIndex(1, 10, 3).start,
        m.RangeIndex(1, 10, 3).stop,
        m.RangeIndex(1, 10, 3).step,
        str(m.RangeIndex(1, 10, 3).dtype),
    ),
    lambda m: m.RangeIndex(1, 10, 3).tolist(),
    lambda m: m.RangeIndex(10, 1, -3).tolist(),
    lambda m: m.RangeIndex(m.RangeIndex(3, name="a")),
    lambda m: m.RangeIndex(m.RangeIndex(3, name="a"), name="b"),
    lambda m: m.RangeIndex(1.0, 4.0),
    lambda m: m.RangeIndex(stop=3),
    lambda m: m.RangeIndex(step=3),
    lambda m: m.RangeIndex(2).name,
    lambda m: len(m.RangeIndex(1, 10, 3)),
    lambda m: m.RangeIndex(1, 10, 3).tolist()[1:],
    lambda m: m.RangeIndex(-5).tolist(),
    lambda m: m.RangeIndex(3, 0, -1).tolist(),
    lambda m: str(m.RangeIndex(3, name="n")),
    lambda m: m.RangeIndex(3).rename("z").tolist(),
    lambda m: m.Series([1, 2, 3], index=m.RangeIndex(10, 13)).index.tolist(),
    lambda m: m.DataFrame({"a": [1, 2]}).set_axis(m.RangeIndex(5, 7)).index.tolist(),
    lambda m: m.RangeIndex(1, 10, 3).equals(m.Index([1, 4, 7])),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Every way of giving the range, compared by repr."""
    import pandas as pd

    assert repr(build(firepanda)) == repr(build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.RangeIndex(),
    lambda m: m.RangeIndex(1, 5, 0),
    lambda m: m.RangeIndex(1.5),
    lambda m: m.RangeIndex("a"),
    lambda m: m.RangeIndex(5, dtype="float64"),
    lambda m: m.RangeIndex.from_range([1, 2]),
    lambda m: m.RangeIndex(None),
    lambda m: m.RangeIndex(True),
    lambda m: m.RangeIndex(5, dtype="uint8"),
    lambda m: m.RangeIndex(5, dtype="bool"),
    lambda m: m.RangeIndex([1, 2]),
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


def test_the_labels_are_int64(firepanda: ModuleType) -> None:
    """Whatever signed type was asked for, as in pandas."""
    assert str(firepanda.RangeIndex(3, dtype="int32").dtype) == "int64"
