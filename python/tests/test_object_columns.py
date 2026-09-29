"""Columns of any Python value, pandas' object dtype, compared with pandas.

Each case runs in both libraries and the answers are compared by their repr,
or for a mistake by its class name. A dtype is compared by its text, since
firepanda spells a dtype as a word where pandas hands out a numpy dtype.
"""

from __future__ import annotations

import datetime
import decimal
import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp
from firepanda import _objects

pd = pytest.importorskip("pandas")


def mixed(lib: ModuleType) -> Any:
    return lib.Series([3, "a", None, 1.5, "a"])


def typed(lib: ModuleType) -> Any:
    return lib.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1.5, None]})


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
    lambda lib: typed(lib).iloc[0],
    lambda lib: typed(lib).loc[1],
    lambda lib: [row.tolist() for _, row in typed(lib).iterrows()],
    lambda lib: typed(lib).T,
    lambda lib: typed(lib).T.dtypes.astype(str).tolist(),
    lambda lib: typed(lib).melt(),
    lambda lib: lib.DataFrame({"a": [1.5], "b": [True]}).iloc[0],
    lambda lib: lib.concat([lib.Series([1, 2]), lib.Series(["a"])]),
    lambda lib: lib.concat([lib.DataFrame({"a": [True]}), lib.DataFrame({"b": [1]})]),
    lambda lib: lib.Series(["a", "b", "a", None]).describe(),
    lambda lib: lib.Series(["b", "a", "a", "b"], dtype="category").describe(),
    lambda lib: lib.Series([True, False, True]).describe(),
    lambda lib: lib.Series(["a", None]).iloc[1:].describe(),
    lambda lib: lib.DataFrame({"a": ["x", "y", "x"]}).describe(),
    lambda lib: lib.DataFrame({"a": ["x", "y", "x"], "b": [1, 2, 3]}).describe(include="all"),
    lambda lib: lib.Series(["a b", "c", None]).str.split(),
    lambda lib: lib.Series(["a,b,c"]).str.split(",", n=1),
    lambda lib: lib.Series(["a b c", "c"]).str.rsplit(n=1),
    lambda lib: lib.Series(["a1b2", "c", None]).str.findall(r"\d"),
    lambda lib: lib.Series(["a1b2"]).str.findall(r"(\w)(\d)"),
    lambda lib: lib.Series(["A1a2"]).str.findall(r"a\d", flags=2),
    lambda lib: lib.Series(lib.to_datetime(["2020-01-01 10:30:05", None])).dt.time,
    lambda lib: lib.Series(lib.to_datetime(["2020-01-01 10:30:05"])).dt.time[0],
    lambda lib: lib.Series([{"a"}, frozenset(["b"]), ("x",), ["y", ("z", "w")], {"k": ["v"]}]),
    lambda lib: lib.Series([1.5, "a", float("nan")]),
    lambda lib: lib.Series([1.5, "a", None]),
    lambda lib: lists(lib).str.len(),
    lambda lib: lists(lib).str.get(0),
    lambda lib: lists(lib).str.get(-1),
    lambda lib: lists(lib).str[:1],
    lambda lib: lists(lib).str[0],
    lambda lib: lib.Series([[1, 2], [3]]).str.len(),
    lambda lib: lib.Series([["a", "b"], ["c"], ("d", "e")]).str.join(","),
    lambda lib: lib.Series([["a", 1], ["c"], 5]).str.join(","),
    lambda lib: lib.Series(["abc", [1, 2, 3], 5, ("x", "y")]).str.len(),
    lambda lib: lib.Series(["abc", [1, 2, 3], 5, ("x", "y")]).str[::-1],
    lambda lib: lib.Series([{"k": 1}, {"j": 2}]).str.get("k"),
    lambda lib: lib.Series([[1, 2], [3]]).str.get(9),
    lambda lib: lib.Series(["ab", "cd"], dtype=object).str.len(),
    lambda lib: lib.Series(["ab", "cd"], dtype=object).str.get(0),
    lambda lib: lib.Series([[1, 2]], index=["r"], name="n").str.len(),
    lambda lib: lib.Series(["abc", "de"]).str[0],
    lambda lib: lib.Series(["abc", None]).str[1:],
    lambda lib: lib.Series(["abc", "de"]).str[-1],
    lambda lib: lib.Series([[1, 2], [], 3]).explode(),
    lambda lib: lib.Series([[1, 2], [3]], index=["a", "b"], name="n").explode(),
    lambda lib: lib.Series([[1, 2], [3]]).explode(ignore_index=True),
    lambda lib: lib.Series([("x", "y"), "z"]).explode().tolist(),
    lambda lib: lib.DataFrame({"k": [1, 2], "v": lists(lib)[:2]}).explode("v"),
    lambda lib: lib.DataFrame({"k": [1, 2], "v": lists(lib)[:2]}).explode("v", ignore_index=True),
    lambda lib: lib.DataFrame(
        {"v": lib.Series([[1, 2], [3]]), "w": lib.Series([["a", "b"], ["c"]])}
    ).explode(["v", "w"]),
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


def test_a_gap_reads_as_the_list_wrote_it() -> None:
    assert fp.Series([1, "a", None]).tolist()[2] is None
    assert math.isnan(fp.Series([1, "a", float("nan")]).tolist()[2])
    assert math.isnan(fp.Series([1, 2]).astype(object).reindex([0, 5]).tolist()[1])


def test_text_methods_read_a_list_as_a_gap() -> None:
    assert fp.Series([[1], ["a"]]).str.upper().isna().tolist() == [True, True]


def test_text_that_starts_with_the_mark_is_still_text() -> None:
    rows = ["\x1c", "\x1cs\x01x", "\x1cz\x01"]
    assert fp.Series(rows).dtype != "object"
    assert fp.Series(rows).tolist() == rows


def test_a_number_with_a_zone_counts_from_the_epoch() -> None:
    assert repr(fp.Timestamp(1609524245000000000, tz="Asia/Tokyo")) == repr(
        pd.Timestamp(1609524245000000000, tz="Asia/Tokyo")
    )
    assert repr(fp.Timestamp(1609524245, unit="s", tz="Asia/Tokyo")) == repr(
        pd.Timestamp(1609524245, unit="s", tz="Asia/Tokyo")
    )


def test_an_object_column_exports_its_values_to_arrow() -> None:
    pa = pytest.importorskip("pyarrow")
    assert pa.array(fp.Series([[1, 2], None])).to_pylist() == [[1, 2], None]
    frame = fp.DataFrame({"a": [1], "b": fp.Series([[1]])})
    assert pa.table(frame).to_pydict() == {"a": [1], "b": [[1]]}
    with pytest.raises(pa.ArrowInvalid):
        pa.array(fp.Series([1, "two"]))


def test_a_nested_arrow_table_reads_as_object_columns() -> None:
    pa = pytest.importorskip("pyarrow")
    table = pa.table(
        {
            "row": [0, 1, 2],
            "items": pa.array([[1, 2], [], None], type=pa.large_list(pa.int64())),
            "pair": pa.array([{"a": 1}, None, {"a": 3}], type=pa.struct([("a", pa.int64())])),
        }
    )
    frame = fp.DataFrame.from_arrow(table)
    assert frame.dtypes.astype(str).tolist() == ["int64", "object", "object"]
    assert frame["items"].tolist() == [[1, 2], [], None]
    assert frame["pair"].tolist() == [{"a": 1}, None, {"a": 3}]
    assert frame.explode("items")["items"].tolist()[:2] == [1, 2]
    assert frame.explode("items")["items"].isna().tolist() == [False, False, True, True]


@pytest.mark.parametrize(
    ("columns", "message"),
    [([], "nonempty"), (["v", "v"], "unique"), (["v", "w"], "matching element counts")],
)
def test_explode_refuses_what_pandas_refuses(columns: list[str], message: str) -> None:
    frame = fp.DataFrame({"v": fp.Series([[1, 2], [3]]), "w": fp.Series([[1], [2]])})
    with pytest.raises(ValueError, match=message):
        frame.explode(columns)
    with pytest.raises(ValueError, match=message):
        pd.DataFrame({"v": [[1, 2], [3]], "w": [[1], [2]]}).explode(columns)


def test_a_nested_arrow_column_reads_the_numbers_pandas_reads() -> None:
    pa = pytest.importorskip("pyarrow")
    table = pa.table(
        {
            "items": pa.array([[1, 2], [None]], type=pa.large_list(pa.int64())),
            "pair": pa.array([{"a": 1}, {"a": None}], type=pa.struct([("a", pa.int64())])),
        }
    )
    frame = fp.DataFrame.from_arrow(table)
    expected = table.to_pandas()
    assert frame["items"].tolist()[0] == expected["items"][0].tolist()
    assert math.isnan(frame["items"].tolist()[1][0])
    assert frame["pair"].tolist() == expected["pair"].tolist()


class Thing:
    """A value of a class of the caller's, which a column hands back as itself."""


def test_an_object_of_the_callers_comes_back_as_itself() -> None:
    thing = Thing()
    series = fp.Series([thing, thing], index=["a", "b"])
    assert series["a"] is thing
    assert series.iloc[1] is thing
    assert str(series.dtype) == "object"


def test_an_object_that_cannot_be_pickled_is_held_by_the_column() -> None:
    held = fp.Series([lambda value: value + 1, None])
    assert held[0](1) == 2


def test_a_value_pandas_copies_is_still_written_as_a_value() -> None:
    assert _objects._written(decimal.Decimal("1.5")).startswith("p")
    assert _objects._written({"a": 1}).startswith("p")
    assert _objects._written(Thing()).startswith("r")


def test_a_column_of_arrays_builds_and_prints_as_pandas() -> None:
    np = pytest.importorskip("numpy")

    def build(lib: ModuleType) -> Any:
        return lib.Series([np.array([1.5, 2]), np.array([[3]], dtype=object)], dtype="object")

    ours, theirs = build(fp), build(pd)
    assert repr(ours) == repr(theirs)
    assert [a.tolist() for a in ours.tolist()] == [a.tolist() for a in theirs.tolist()]
