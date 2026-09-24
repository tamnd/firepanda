"""`filter` by `like` and `regex` over labels with gaps, checked against pandas.

pandas renders every label with `str` before it looks, so a missing label is the
text `nan`, or `NaT` for instants, and a rule can match that text.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def texts(m: ModuleType) -> Any:
    """A column whose text labels have a gap."""
    return m.Series([1, 2, 3], index=m.Index(["a", None, "b"]))


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: texts(m).filter(like="a"),
    lambda m: texts(m).filter(like="n"),
    lambda m: texts(m).filter(regex="^n"),
    lambda m: texts(m).filter(items=["b", "a"]),
    lambda m: m.Series([1, 2], index=[1.5, None]).filter(like="nan"),
    lambda m: m.Series([1, 2], index=m.to_datetime(["2024-01-01", None])).filter(like="NaT"),
    lambda m: m.DataFrame({"x": [1, 2, 3]}, index=m.Index(["a", None, "b"])).filter(
        like="a", axis=0
    ),
    lambda m: m.DataFrame({"x": [1, 2, 3]}, index=m.Index(["a", None, "b"])).filter(
        regex="n", axis=0
    ),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """A gap in text, numbers and instants, over a column and over a frame's rows."""
    import pandas as pd

    got, want = build(firepanda), build(pd)
    assert len(got) == len(want)
    if hasattr(want, "columns"):
        assert got["x"].tolist() == want["x"].tolist()
    else:
        assert got.tolist() == want.tolist()
