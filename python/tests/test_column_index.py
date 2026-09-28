"""A frame's column labels handed out as an index, compared with pandas.

Each case runs in both libraries and the answers are compared by their text,
which for an index is its class, its repr and its labels, and for a mistake is
its class name and its message.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def frame(lib: ModuleType) -> Any:
    return lib.DataFrame({"x": [1, 2], "y": [3.0, 4.0], "z": ["a", "b"]})


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    try:
        got = call(lib)
    except Exception as error:
        return "error", type(error).__name__, str(error)
    if isinstance(got, (fp.Index, pd.Index)):
        return type(got).__name__, repr(got), got.tolist(), got.name
    return str(got)


CASES: list[Callable[[ModuleType], Any]] = [
    lambda lib: frame(lib).columns,
    lambda lib: lib.DataFrame().columns,
    lambda lib: lib.DataFrame({"a": []}).columns,
    lambda lib: frame(lib).columns.tolist(),
    lambda lib: frame(lib).columns.name,
    lambda lib: frame(lib).columns[1],
    lambda lib: frame(lib).columns[1:],
    lambda lib: frame(lib).columns.get_loc("y"),
    lambda lib: "y" in frame(lib).columns,
    lambda lib: list(frame(lib).columns),
    lambda lib: len(frame(lib).columns),
    lambda lib: frame(lib).columns.equals(frame(lib).columns),
    lambda lib: frame(lib).columns.intersection(["z", "y", "w"]),
    lambda lib: frame(lib).columns.difference(["y"]),
    lambda lib: frame(lib).columns.union(["w"]),
    lambda lib: [bool(one) for one in frame(lib).columns.isin(["x", "z"])],
    lambda lib: frame(lib)[frame(lib).columns[1:]].columns,
    lambda lib: frame(lib)[lib.Series(["z", "x"])].columns,
    lambda lib: frame(lib)[["z", "x"]].columns,
    lambda lib: frame(lib).rename(columns={"x": "w"}).columns,
    lambda lib: frame(lib).drop(columns="y").columns,
]


@pytest.mark.parametrize("call", CASES)
def test_the_answer_is_pandas_answer(call: Callable[[ModuleType], Any]) -> None:
    assert outcome(call, fp) == outcome(call, pd)


def test_the_columns_can_select_the_frame_they_came_from() -> None:
    ours = frame(fp)
    assert ours[ours.columns[:2]].columns.tolist() == ["x", "y"]
