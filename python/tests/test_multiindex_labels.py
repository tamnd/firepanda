"""Frames and columns whose rows are labelled by a MultiIndex, compared with pandas.

Each case runs in both libraries and is compared by its text: a frame or a
column by what it prints and by its labels, level names and values, and a
mistake by its class name and message.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest
from firepanda._levels import read, written

import firepanda as fp

pd = pytest.importorskip("pandas")


def frame(lib: ModuleType) -> Any:
    return lib.DataFrame(
        {"a": ["x", "y", "x", "x"], "b": [1, 2, 3, 3], "v": [0.5, 1.5, 2.5, 3.0], "w": [1, 2, 3, 4]}
    )


def keyed(lib: ModuleType) -> Any:
    return frame(lib).set_index(["a", "b"])


def pairs(lib: ModuleType) -> Any:
    return lib.MultiIndex.from_tuples([("b", 2), ("a", 1), ("a", 3), (None, 4)], names=["k", "n"])


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    try:
        got = call(lib)
    except Exception as error:
        return type(error).__name__, str(error)
    if isinstance(got, (fp.DataFrame, pd.DataFrame)):
        return str(got), str(got.index.tolist()), list(got.index.names), str(got.to_dict("list"))
    if isinstance(got, (fp.Series, pd.Series)):
        return str(got), str(got.index.tolist()), list(got.index.names), str(got.tolist())
    return str(got)


CASES: list[Callable[[ModuleType], Any]] = [
    lambda lib: lib.Series([10, 20, 30, 40], index=pairs(lib), name="v"),
    lambda lib: lib.Series([10, 20, 30, 40], index=pairs(lib)).sort_index(),
    lambda lib: lib.DataFrame({"x": [1.5, 2, 3, 4]}, index=pairs(lib)).sort_index(),
    lambda lib: lib.Series([1, 2], index=pairs(lib)[:2]) + lib.Series([5], index=pairs(lib)[1:2]),
    keyed,
    lambda lib: keyed(lib).sort_index(),
    lambda lib: keyed(lib)["v"],
    lambda lib: keyed(lib)[keyed(lib)["v"] > 1],
    lambda lib: frame(lib).set_index(["a", "b"], drop=False),
    lambda lib: frame(lib).set_index("a").set_index("b", append=True),
    lambda lib: frame(lib).set_index(["a", "zz"]),
    lambda lib: keyed(lib).reset_index(),
    lambda lib: keyed(lib).reset_index(level="b"),
    lambda lib: keyed(lib).reset_index(level=[0, 1]),
    lambda lib: keyed(lib).reset_index(drop=True),
    lambda lib: keyed(lib)["v"].reset_index(),
    lambda lib: lib.DataFrame({"v": [1]}, index=lib.MultiIndex.from_tuples([(1, "a")])),
    lambda lib: lib.DataFrame(
        {"v": [1]}, index=lib.MultiIndex.from_tuples([(1, "a")])
    ).reset_index(),
    lambda lib: lib.Series([1], index=lib.MultiIndex.from_tuples([("a", 1)], names=[None, "n"])),
    lambda lib: lib.Series(range(80), index=lib.MultiIndex.from_product([["a", "b"], range(40)])),
    lambda lib: keyed(lib).rename_axis(["p", "q"]),
    lambda lib: keyed(lib).iloc[:3].reindex(lib.MultiIndex.from_tuples([("x", 3), ("z", 9)])),
    lambda lib: keyed(lib).to_csv(),
    lambda lib: keyed(lib).equals(keyed(lib).copy()),
    lambda lib: list(keyed(lib)["v"].items()),
    lambda lib: ("x", 3) in keyed(lib)["v"],
    lambda lib: ("x", 2) in keyed(lib)["v"],
    lambda lib: list(keyed(lib).itertuples()),
    lambda lib: frame(lib).groupby(["a", "b"]).sum(),
    lambda lib: frame(lib).groupby(["a", "b"]).mean(),
    lambda lib: frame(lib).groupby(["a", "b"]).size(),
    lambda lib: frame(lib).groupby(["a", "b"]).count(),
    lambda lib: frame(lib).groupby(["a", "b"]).first(),
    lambda lib: frame(lib).groupby(["a", "b"]).std(),
    lambda lib: frame(lib).groupby(["a", "b"])["v"].sum(),
    lambda lib: frame(lib).groupby(["a", "b"]).v.max(),
    lambda lib: frame(lib).groupby(["a", "b"]).agg({"v": "sum", "w": "max"}),
    lambda lib: frame(lib).groupby(["a", "b"])["v"].agg(["sum", "max"]),
    lambda lib: frame(lib).groupby(["b", "a"], sort=False).w.sum(),
    lambda lib: frame(lib).groupby(["a", "b"]).v.apply(lambda s: s.max()),
    lambda lib: frame(lib).groupby(["a", "b"]).v.quantile(0.5),
    lambda lib: frame(lib).groupby(["a", "b"], as_index=False).sum(),
    lambda lib: frame(lib).groupby("a").zz,
]


@pytest.mark.parametrize("call", CASES)
def test_the_answer_is_pandas_answer(call: Callable[[ModuleType], Any]) -> None:
    assert outcome(call, fp) == outcome(call, pd)


VALUES: list[tuple[Any, ...]] = [
    ("", "a", "a\x00", "a\x01", "a\x02", "a\x03", "ab", "b"),
    (-(2**63), -5, -1, 0, 1, 7, 2**63 - 1),
    (float("-inf"), -2.5, -0.0, 1e-300, 1.5, float("inf")),
    (False, True),
]


@pytest.mark.parametrize("values", VALUES)
def test_written_labels_sort_as_their_values_do(values: tuple[Any, ...]) -> None:
    labels = [written((value, 0)) for value in values]
    assert labels == sorted(labels)
    assert [read(label)[0] for label in labels] == list(values)


def test_a_gap_sorts_after_every_value_and_reads_back_as_nan() -> None:
    assert written(("zzz",)) < written((None,))
    assert written((10**18,)) < written((float("nan"),))
    back = read(written((None, 1)))
    assert back[0] != back[0] and back[1] == 1


def test_instants_and_spans_keep_their_unit_and_zone() -> None:
    moment = fp.Timestamp("2024-01-01 01:02", tz="UTC").as_unit("s")
    span = fp.Timedelta("90min").as_unit("ms")
    got = read(written((moment, span)))
    assert got == (moment, span)
    assert (got[0].unit, str(got[0].tz), got[1].unit) == ("s", "UTC", "ms")
    assert written((fp.Timestamp("2024-01-01"),)) < written((fp.Timestamp("2024-01-02"),))


def test_a_value_that_cannot_be_written_is_refused_by_its_type() -> None:
    with pytest.raises(NotImplementedError, match="of type bytes"):
        written((b"x",))
