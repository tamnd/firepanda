"""The operations that make column names that are not text, compared with pandas.

Each case runs in both libraries and the answers are compared by their repr,
or for a mistake by its class name. A pivot is compared by its names and its
cells rather than its repr, because pandas also names the column axis after
the pivoted column and firepanda has no place to hold that name yet.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def long(lib: ModuleType) -> Any:
    return lib.DataFrame(
        {
            "k": ["a", "a", "b"],
            "h": [1, 2, 1],
            "v": [10, 20, 30],
            "d": lib.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
        }
    )


def wide(frame: Any) -> Any:
    return frame.columns.tolist(), frame.to_dict()


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    try:
        got = call(lib)
    except Exception as error:
        return "error", type(error).__name__
    return repr(got)


CSV = "1,2,x\n3,4,y\n"

CASES: list[Callable[[ModuleType], Any]] = [
    lambda lib: wide(long(lib).pivot(index="k", columns="h", values="v")),
    lambda lib: wide(long(lib).pivot_table(index="k", columns="h", values="v", aggfunc="sum")),
    lambda lib: wide(long(lib).pivot_table(index="k", columns="d", values="v", aggfunc="sum")),
    lambda lib: wide(lib.crosstab(long(lib)["k"], long(lib)["h"])),
    lambda lib: wide(long(lib).set_index(["k", "h"])["v"].unstack()),
    lambda lib: long(lib).pivot(index="k", columns="h", values="v")[1],
    lambda lib: wide(long(lib).pivot(index="h", columns="v", values="k")),
    lambda lib: lib.get_dummies(long(lib)["h"]),
    lambda lib: lib.get_dummies(long(lib)["h"], dtype=int),
    lambda lib: lib.get_dummies(long(lib)[["h"]], columns=["h"]),
    lambda lib: lib.Series(["a-b", "c-d"]).str.partition("-"),
    lambda lib: lib.Series(["a-b", "c-d"]).str.rpartition("-"),
    lambda lib: lib.Series(["a1", "b2"]).str.extract(r"(\w)(\d)"),
    lambda lib: lib.Series(["a b", "c d e"]).str.split(expand=True),
    lambda lib: lib.Series(["a b", "c d e"]).str.rsplit(expand=True, n=1),
    lambda lib: lib.DataFrame.from_dict({"r": [1, 2]}, orient="index"),
    lambda lib: lib.DataFrame.from_records([(1, 2), (3, 4)]),
    lambda lib: lib.concat([lib.Series([1]), lib.Series([2])], axis=1),
    lambda lib: list(lib.concat([lib.Series([1], name="a"), lib.Series([2])], axis=1)),
    lambda lib: lib.read_csv(io.StringIO(CSV), header=None),
    lambda lib: lib.read_csv(io.StringIO(CSV), header=None).columns.tolist(),
    lambda lib: lib.read_csv(io.StringIO(CSV), header=None, index_col=0),
    lambda lib: lib.read_csv(io.StringIO(CSV), header=None, index_col=0).index.name,
    lambda lib: lib.read_csv(io.StringIO(CSV), header=None, usecols=[0, 2]),
    lambda lib: lib.read_csv(io.StringIO(CSV), header=None, dtype={1: "float64"}),
    lambda lib: lib.read_json(io.StringIO("[[1,2],[3,4]]")).columns.tolist(),
    lambda lib: lib.read_json(io.StringIO('{"0":{"a":1},"1":{"a":2}}')).columns.tolist(),
    lambda lib: lib.read_json(io.StringIO('{"0":{"a":1}}'), convert_axes=False).columns,
    lambda lib: lib.read_json(io.StringIO('[{"0":1,"b":2}]')).columns,
    lambda lib: lib.read_json(io.StringIO('{"1.5":[1]}')).columns,
    lambda lib: lib.DataFrame({"a": [1, 2]}).rename_axis(5).index.name,
    lambda lib: lib.DataFrame({"a": [1, 2]}).rename_axis(5),
    lambda lib: lib.Series([1, 2]).rename_axis(2.5).index.name,
    lambda lib: lib.get_dummies(lib.Series([1.5, 2.0])),
    lambda lib: lib.get_dummies(lib.Series(["a", "b"]), dummy_na=True),
    lambda lib: wide(lib.get_dummies(lib.Series([True, False]))),
    lambda lib: wide(lib.crosstab(lib.Series(["p", "q"]), lib.Series([1, 2]))),
    lambda lib: lib.DataFrame({"a": [1, 2]}).dot([[1, 2]]),
    lambda lib: list(lib.concat([lib.DataFrame({"a": [1]}), lib.Series([1])])),
    lambda lib: lib.Index([1, 2]).rename(3).name,
    lambda lib: lib.Index([1, 2], name="x").rename(None).name,
]


@pytest.mark.parametrize("call", CASES)
def test_the_answer_is_pandas_answer(call: Callable[[ModuleType], Any]) -> None:
    assert outcome(call, fp) == outcome(call, pd)


def test_a_tuple_level_name_is_still_its_text() -> None:
    assert fp.Index([1, 2]).rename(("a", 1)).name == "('a', 1)"
