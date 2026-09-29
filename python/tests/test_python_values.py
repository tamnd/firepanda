"""Answers made of Python values, compared with pandas.

`dt.to_pydatetime`, `str.encode`, `str.decode` and `DataFrame.to_records` all
hand back values Python or numpy holds rather than a typed column. Each case
runs in both libraries and the answers are compared by their repr, or for a
mistake by its class name.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")
pytest.importorskip("numpy")


def moments(lib: ModuleType) -> Any:
    return lib.Series(lib.to_datetime(["2020-01-01 10:00", None]), name="m")


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    try:
        got = call(lib)
    except Exception as error:
        return "error", type(error).__name__
    return repr(got)


CASES: list[Callable[[ModuleType], Any]] = [
    lambda lib: moments(lib).dt.to_pydatetime(),
    lambda lib: moments(lib).dt.to_pydatetime().tolist(),
    lambda lib: (
        lib.Series(lib.to_datetime(["2020-01-01 10:00"]).tz_localize("UTC"))
        .dt.to_pydatetime()
        .tolist()
    ),
    lambda lib: lib.Series(lib.to_timedelta(["1s"])).dt.to_pydatetime(),
    lambda lib: lib.Series(["a", "é", None], name="n").str.encode("utf-8"),
    lambda lib: lib.Series(["a", "é", None]).str.encode("utf-8").tolist(),
    lambda lib: lib.Series(["a", "é", None]).str.encode("utf-8").str.decode("utf-8"),
    lambda lib: lib.Series(["a", "é"]).str.encode("ascii", errors="ignore").tolist(),
    lambda lib: lib.Series(["é"]).str.encode("ascii"),
    lambda lib: lib.Series([b"a", 1]).str.decode("utf-8"),
    lambda lib: lib.DataFrame({"a": [1, 2], "b": ["x", "y"]}, index=["p", "q"]).to_records(),
    lambda lib: lib.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_records(index=False),
    lambda lib: lib.DataFrame({"a": [1, 2], "b": [1.5, 2.5]}).to_records(
        column_dtypes={"a": "int32"}
    ),
    lambda lib: lib.DataFrame({"a": [1, 2]}).rename_axis("k").to_records(index_dtypes="<i4"),
    lambda lib: lib.DataFrame(
        {"a": [1, 2]}, index=lib.MultiIndex.from_tuples([("x", 1), ("y", 2)])
    ).to_records(),
]


@pytest.mark.parametrize("call", CASES)
def test_the_answer_is_pandas_answer(call: Callable[[ModuleType], Any]) -> None:
    assert outcome(call, fp) == outcome(call, pd)
