"""Columns of any Python value, pandas' object dtype, compared with pandas.

Each case runs in both libraries and the answers are compared by their repr,
or for a mistake by its class name. A dtype is compared by its text, since
firepanda spells a dtype as a word where pandas hands out a numpy dtype.
"""

from __future__ import annotations

import datetime
import decimal
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp
from firepanda import _objects

pd = pytest.importorskip("pandas")


def mixed(lib: ModuleType) -> Any:
    return lib.Series([3, "a", None, 1.5, "a"])


def lists(lib: ModuleType) -> Any:
    return lib.Series([[1, 2], [3], None])


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    try:
        got = call(lib)
    except Exception as error:
        return "error", type(error).__name__
    return repr(got)


CASES: list[Callable[[ModuleType], Any]] = [
    lambda lib: mixed(lib),
    lambda lib: str(mixed(lib).dtype),
    lambda lib: mixed(lib).tolist(),
    lambda lib: list(mixed(lib)),
    lambda lib: mixed(lib)[1],
    lambda lib: mixed(lib).loc[3],
    lambda lib: mixed(lib).iloc[[4, 0]],
    lambda lib: lists(lib),
    lambda lib: lists(lib).tolist(),
    lambda lib: lib.Series([{"a": 1}, (1, "x"), {2, 3}]).tolist(),
    lambda lib: lib.Series([datetime.time(1, 2)]),
    lambda lib: lib.Series([decimal.Decimal("1.5"), "a"]).tolist(),
    lambda lib: lib.Series([2**70, 1]).tolist(),
    lambda lib: lib.Series(["a", "b"], dtype=object),
    lambda lib: lib.Series([1, 2], dtype="O").tolist(),
    lambda lib: lib.Series([1, "a"], index=["x", "y"], name="n"),
    lambda lib: lib.Series([1, 2]).astype(object),
    lambda lib: lib.Series([1, 2]).astype(object).astype("int64"),
    lambda lib: lib.Series([1, "a"]).astype(str),
    lambda lib: lib.DataFrame({"a": [1, 2], "b": [1, "x"]}),
    lambda lib: lib.DataFrame({"a": [1, 2], "b": [1, "x"]}).dtypes.astype(str).tolist(),
    lambda lib: lib.DataFrame({"a": lists(lib)}),
    lambda lib: lib.concat([lib.Series([1, "a"]), lib.Series([2, "b"])]).tolist(),
    lambda lib: mixed(lib).to_dict(),
    lambda lib: mixed(lib).to_numpy().tolist(),
    lambda lib: lists(lib).to_numpy().shape,
    lambda lib: mixed(lib).isna().tolist(),
    lambda lib: mixed(lib).dropna().tolist(),
    lambda lib: mixed(lib).fillna(0).tolist(),
    lambda lib: list(mixed(lib).unique()),
    lambda lib: mixed(lib).nunique(),
    lambda lib: mixed(lib).value_counts().to_dict(),
    lambda lib: mixed(lib).duplicated().tolist(),
    lambda lib: mixed(lib).equals(mixed(lib)),
    lambda lib: mixed(lib).shift(1).tolist()[1:],
    lambda lib: (mixed(lib) == "a").tolist(),
    lambda lib: (mixed(lib) != 3).tolist(),
    lambda lib: (lib.Series([1, 2.5], dtype=object) < 2).tolist(),
    lambda lib: (lib.Series(["a", 1]) * 2).tolist(),
    lambda lib: (2 * lib.Series(["a", 1])).tolist(),
    lambda lib: (lib.Series(["a", "b"], dtype=object) + "x").tolist(),
    lambda lib: str((lib.Series(["a", 1]) * 2).dtype),
    lambda lib: lib.Series(["a", 1]) - 1,
    lambda lib: lib.Series(["b", 1, "a"]).sort_values(),
    lambda lib: lib.Series(["b", "c", "a", "c"], dtype=object).sort_values().tolist(),
    lambda lib: lib.Series(["b", "c", "a", "c"], dtype=object).sort_values().index.tolist(),
    lambda lib: (
        lib.Series(["b", None, "a"], dtype=object)
        .sort_values(ascending=False, na_position="first")
        .index.tolist()
    ),
    lambda lib: lib.Series(["b", "c", "a"], dtype=object).min(),
    lambda lib: lib.Series(["b", "c", "a"], dtype=object).max(),
    lambda lib: lib.Series(["b", "c", "a"], dtype=object).sum(),
    lambda lib: lib.Series([[1, 2], [3]]).sum(),
    lambda lib: mixed(lib).isin(["a", 3]).tolist(),
    lambda lib: mixed(lib).map(lambda v: type(v).__name__).tolist(),
    lambda lib: lists(lib).apply(lambda v: v).tolist(),
    lambda lib: str(lib.Series([1, 2], dtype=object).infer_objects().dtype),
    lambda lib: lib.Series(["a", 3], dtype=object).str.upper().tolist()[:1],
    lambda lib: (
        lib.DataFrame({"k": lib.Series([1, "a", 1]), "v": [1, 2, 3]})
        .groupby("k")["v"]
        .sum()
        .to_dict()
    ),
    lambda lib: lib.DataFrame({"a": mixed(lib)}).to_csv(),
    lambda lib: lib.DataFrame({"a": mixed(lib)}).iloc[1, 0],
]


@pytest.mark.parametrize("call", CASES)
def test_the_answer_is_pandas_answer(call: Callable[[ModuleType], Any]) -> None:
    assert outcome(call, fp) == outcome(call, pd)


def test_the_extension_holds_written_text() -> None:
    inner = fp.Series([1, "a"])._inner
    assert inner.dtype() == "string"
    assert all(cell.startswith(_objects.MARK) for cell in inner.to_list())


def test_every_kind_reads_back_as_it_was() -> None:
    values = ["a\x00b", 7, -(2**63), 2.5, float("-inf"), True, [1, [None, "x"]], (2, ()), {"k": 1}]
    assert _objects.values(_objects.cells(values)) == values


def test_ordinary_text_is_not_an_object_column() -> None:
    assert fp.Series(["a", None, "b"]).dtype != "object"


def test_the_string_accessor_refuses_lists_for_now() -> None:
    with pytest.raises(NotImplementedError, match="lists"):
        fp.Series([[1], [2]]).str.len()


def test_text_that_starts_with_the_mark_is_still_text() -> None:
    rows = ["\x1c", "\x1cs\x01x", "\x1cz\x01"]
    assert fp.Series(rows).dtype != "object"
    assert fp.Series(rows).tolist() == rows
