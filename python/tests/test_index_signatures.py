"""The signatures `inspect` reads off the index classes, which have to be pandas' own.

`inspect` reads `__new__` before `__init__`, and the index's `__new__` takes
anything so that it can answer a list of tuples with a `MultiIndex`. Each class
that keeps it shows the parameters of its own `__init__` instead, so the names,
kinds and order match pandas'.
"""

from __future__ import annotations

import inspect
from typing import Any

import pandas as pd
import pytest

CLASSES = [
    "Index",
    "CategoricalIndex",
    "RangeIndex",
    "PeriodIndex",
    "DatetimeIndex",
    "TimedeltaIndex",
    "IntervalIndex",
    "MultiIndex",
]


def shape(cls: Any) -> list[tuple[str, Any]]:
    return [(p.name, p.kind) for p in inspect.signature(cls).parameters.values()]


@pytest.mark.parametrize("name", CLASSES)
def test_index_signature_is_pandas(firepanda: Any, name: str) -> None:
    assert shape(getattr(firepanda, name)) == shape(getattr(pd, name))


def test_index_still_builds_what_pandas_builds(firepanda: Any) -> None:
    assert repr(firepanda.Index([1, 2], name="x")) == repr(pd.Index([1, 2], name="x"))
    assert repr(firepanda.Index([(1, 2)])) == repr(pd.Index([(1, 2)]))
