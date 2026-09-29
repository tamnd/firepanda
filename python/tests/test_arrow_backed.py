"""Columns backed by Arrow, `ArrowDtype`, and the list and struct accessors, against pandas.

firepanda holds such a column as an object column whose cells carry the Arrow
type, which is document 98. Each case runs in both libraries and the answers
are compared by their repr, or for a mistake by its class name.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")

LISTS = [[1, 2, 3], [], None, [4]]
STRUCTS = [{"a": 1, "b": "x"}, None, {"a": 2, "b": None}]
DEEP = [{"inner": {"deep": 1.5}}, {"inner": None}]


def arrow(lib: ModuleType, values: list[Any]) -> Any:
    column = lib.Series(values)
    return column.astype(lib.ArrowDtype(pa.array(values).type))


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "list-dtype": lambda lib: str(arrow(lib, LISTS).dtype),
    "list-repr": lambda lib: arrow(lib, LISTS),
    "list-len": lambda lib: arrow(lib, LISTS).list.len(),
    "list-getitem-out-of-bounds": lambda lib: arrow(lib, LISTS).list[0],
    "list-getitem": lambda lib: arrow(lib, [[1, 2], [3]]).list[0],
    "list-slice": lambda lib: arrow(lib, LISTS).list[0:2],
    "list-flatten": lambda lib: arrow(lib, LISTS).list.flatten(),
    "list-isna": lambda lib: arrow(lib, LISTS).isna(),
    "list-cell": lambda lib: arrow(lib, LISTS).iloc[0],
    "list-tolist": lambda lib: arrow(lib, LISTS).tolist(),
    "list-in-frame": lambda lib: lib.DataFrame({"v": arrow(lib, LISTS), "r": range(4)}),
    "list-on-numbers": lambda lib: lib.Series([1, 2]).list,
    "struct-repr": lambda lib: arrow(lib, STRUCTS),
    "struct-dtypes": lambda lib: arrow(lib, STRUCTS).struct.dtypes.astype(str),
    "struct-field-name": lambda lib: arrow(lib, STRUCTS).struct.field("a"),
    "struct-field-index": lambda lib: arrow(lib, STRUCTS).struct.field(1),
    "struct-field-missing": lambda lib: arrow(lib, STRUCTS).struct.field("z"),
    "struct-field-path": lambda lib: arrow(lib, DEEP).struct.field(["inner", "deep"]),
    "struct-field-nested": lambda lib: arrow(lib, DEEP).struct.field("inner"),
    "struct-field-isna": lambda lib: arrow(lib, STRUCTS).struct.field("a").isna(),
    "struct-explode": lambda lib: arrow(lib, STRUCTS).struct.explode(),
    "flat-constructor": lambda lib: lib.Series([1, None], dtype=lib.ArrowDtype(pa.int64())),
    "dtype-equals-name": lambda lib: lib.ArrowDtype(pa.int64()) == "int64[pyarrow]",
}


def outcome(build: Callable[[], Any]) -> str:
    try:
        return repr(build())
    except Exception as error:
        return type(error).__name__


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_arrow_backed_columns_answer_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


def test_the_type_survives_moving_rows() -> None:
    column = arrow(fp, LISTS)
    assert str(column.iloc[[3, 0]].dtype) == "list<item: int64>[pyarrow]"
    assert str(column.sort_index(ascending=False).dtype) == "list<item: int64>[pyarrow]"


def test_a_nan_among_floats_is_a_value_when_exported() -> None:
    column = arrow(fp, [[1.5, float("nan")], None]).list.flatten()
    assert pa.array(column).null_count == 0
