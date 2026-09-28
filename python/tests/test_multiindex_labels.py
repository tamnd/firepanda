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


def gaps(lib: ModuleType) -> Any:
    return lib.DataFrame(
        {
            "k": ["b", "a", "b", "a", "b", None],
            "v": [1.0, 2.0, 3.0, None, 5.0, 6.0],
            "w": [1, 2, 3, 4, 5, 6],
            "c": ["p", "q", "p", "p", "q", "q"],
        },
        index=[10, 11, 12, 13, 14, 15],
    )


def pairs_series(lib: ModuleType, names: list[Any] | None = None) -> Any:
    rows = [("b", "y"), ("a", "x"), ("a", "y")]
    return lib.Series([1, 2, 3], index=lib.MultiIndex.from_tuples(rows, names=names), name="n")


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    try:
        got = call(lib)
    except Exception as error:
        # firepanda's own mistakes are subclasses of the builtin pandas raises.
        kind = next(k for k in type(error).__mro__ if k.__module__ == "builtins")
        return kind.__name__, str(error)
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
    lambda lib: keyed(lib).loc[("y", 2)],
    lambda lib: keyed(lib).loc[("x", 3)],
    lambda lib: keyed(lib).iloc[:3].loc[("y", 2)],
    lambda lib: keyed(lib).loc["x"],
    lambda lib: keyed(lib).loc[["y"]],
    lambda lib: keyed(lib).loc["x", "v"],
    lambda lib: keyed(lib).loc[("y", 2), "v"],
    lambda lib: keyed(lib).loc[("z", 1)],
    lambda lib: keyed(lib)["v"].loc[("y", 2)],
    lambda lib: keyed(lib)["v"].loc["x"],
    lambda lib: keyed(lib)["v"]["x"],
    lambda lib: keyed(lib).xs("x"),
    lambda lib: keyed(lib).xs(3, level="b"),
    lambda lib: keyed(lib).xs(3, level="b", drop_level=False),
    lambda lib: keyed(lib).xs(("y", 2)),
    lambda lib: keyed(lib).xs("q"),
    lambda lib: keyed(lib).xs("v", axis=1),
    lambda lib: keyed(lib)["v"].xs("x"),
    lambda lib: keyed(lib)["v"].xs(3, level=1),
    lambda lib: keyed(lib).droplevel(0),
    lambda lib: keyed(lib).droplevel("b"),
    lambda lib: keyed(lib).droplevel([0, 1]),
    lambda lib: keyed(lib)["v"].droplevel("a"),
    lambda lib: frame(lib).droplevel(0),
    lambda lib: keyed(lib).swaplevel(),
    lambda lib: keyed(lib)["v"].swaplevel(0, 1),
    lambda lib: frame(lib).swaplevel(),
    lambda lib: keyed(lib).reorder_levels(["b", "a"]),
    lambda lib: keyed(lib).reorder_levels(["b"]),
    lambda lib: keyed(lib)["v"].reorder_levels([1, 0]),
    lambda lib: frame(lib).reorder_levels([0]),
    lambda lib: keyed(lib).sort_index(level="b"),
    lambda lib: keyed(lib).sort_index(level=1, ascending=False),
    lambda lib: keyed(lib)["v"].sort_index(level=1),
    lambda lib: frame(lib).value_counts(["a", "b"]),
    lambda lib: frame(lib)[["a", "b"]].value_counts(),
    lambda lib: frame(lib)[["a", "b"]].value_counts(normalize=True),
    lambda lib: frame(lib)[["a", "b"]].value_counts(ascending=True),
    lambda lib: frame(lib)[["a", "b"]].value_counts(sort=False),
    lambda lib: frame(lib).value_counts("a"),
    lambda lib: lib.DataFrame({"a": ["y", "x", "z", "x"]}).value_counts(),
    lambda lib: lib.DataFrame({"a": ["y", "x", "z"]}).value_counts(ascending=True),
    lambda lib: frame(lib).value_counts(["b"]),
    lambda lib: frame(lib).groupby("a")["b"].value_counts(),
    lambda lib: frame(lib).groupby("a")["b"].value_counts(normalize=True),
    lambda lib: frame(lib).groupby("a")["b"].value_counts(ascending=True, sort=False),
    lambda lib: frame(lib).groupby("a", as_index=False)["b"].value_counts(),
    lambda lib: frame(lib).groupby(["a", "b"])["w"].value_counts(),
    lambda lib: frame(lib).groupby("a").apply(lambda d: d * 2),
    lambda lib: frame(lib).groupby("a").apply(lambda d: d.head(1)),
    lambda lib: frame(lib).groupby("a", sort=False).apply(lambda d: d.tail(1)),
    lambda lib: frame(lib).groupby("a", as_index=False).apply(lambda d: d * 2),
    lambda lib: frame(lib).groupby(["a", "b"]).apply(lambda d: d.reset_index()),
    lambda lib: frame(lib).groupby("a")["v"].apply(lambda s: s.head(1)),
    lambda lib: keyed(lib).groupby("w").apply(lambda d: d),
    lambda lib: frame(lib).groupby("a")[["v", "w"]].corr().round(12),
    lambda lib: frame(lib).groupby("a")[["b", "v"]].corr(method="spearman").round(12),
    lambda lib: frame(lib).groupby("a")[["b", "w"]].cov().round(12),
    lambda lib: frame(lib).groupby("a")[["b", "w"]].cov(ddof=0).round(12),
    lambda lib: frame(lib).groupby("a")["v"].corr(frame(lib)["w"]).round(12),
    lambda lib: frame(lib).groupby("a")["v"].cov(frame(lib)["b"]).round(12),
    lambda lib: gaps(lib).groupby("k")["v"].rolling(2).sum(),
    lambda lib: gaps(lib).groupby("k")[["v", "w"]].rolling(2, min_periods=1).mean(),
    lambda lib: gaps(lib).groupby("k")["w"].expanding().sum(),
    lambda lib: gaps(lib).groupby("k")["w"].ewm(span=3).mean().round(12),
    lambda lib: gaps(lib).groupby("k")["w"].rolling(-1),
    lambda lib: gaps(lib).groupby("k")["w"].rolling(2).nope,
    lambda lib: gaps(lib).groupby("k").take([0]),
    lambda lib: gaps(lib).groupby("k")["w"].take([-1]),
    lambda lib: gaps(lib).groupby("k").value_counts(),
    lambda lib: gaps(lib).groupby("k").value_counts(["c"], normalize=True),
    lambda lib: gaps(lib).groupby("k")[["c"]].value_counts(ascending=True),
    lambda lib: gaps(lib).groupby("k", as_index=False).value_counts(["c"], sort=False),
    lambda lib: gaps(lib).groupby("k", sort=False)["c"].value_counts(sort=False),
    lambda lib: gaps(lib).groupby("k")[["v", "w"]].corrwith(gaps(lib)["w"]).round(12),
    lambda lib: lib.concat([gaps(lib).head(2), gaps(lib).tail(1)], keys=["x", "y"]),
    lambda lib: lib.concat([gaps(lib)["w"].head(2), gaps(lib)["w"].tail(1)], keys=["x", "y"]),
    lambda lib: lib.concat({"x": gaps(lib).head(1), "n": None, "y": gaps(lib).tail(1)}),
    lambda lib: lib.concat(
        [gaps(lib).head(1), gaps(lib).tail(1)], keys=[("x", 1), ("y", 2)], names=["a", "b"]
    ),
    lambda lib: lib.concat([gaps(lib)["w"]], keys=["x"], names=["part"]),
    lambda lib: lib.concat([keyed(lib).head(2)], keys=["x"]),
    lambda lib: lib.concat([gaps(lib)], keys=["x", "y"]),
    lambda lib: lib.concat([gaps(lib)], keys=["x"], ignore_index=True),
    lambda lib: pairs_series(lib).unstack(),
    lambda lib: pairs_series(lib, ["p", "q"]).unstack(0).to_dict(),
    lambda lib: pairs_series(lib, ["p", "q"]).unstack(fill_value=0).to_dict(),
    lambda lib: pairs_series(lib).unstack(sort=False),
    lambda lib: lib.Series([1, 2]).unstack(),
    lambda lib: lib.Series([1, 2], index=lib.MultiIndex.from_tuples([("a", "x")] * 2)).unstack(),
    lambda lib: lib.DataFrame({"a": [1, 2], "b": [3.5, 4]}, index=["r", "s"]).stack(),
    lambda lib: lib.DataFrame({"a": [1, 2], "b": [3, 4]}).stack(),
    lambda lib: lib.DataFrame({"a": [1, 2], "b": [3, 4]}).rename_axis("k").stack(),
    lambda lib: lib.DataFrame({"a": [1, 2]}).stack(dropna=True),
    lambda lib: frame(lib).pivot_table(index=["a", "b"], values="v", aggfunc="sum"),
    lambda lib: frame(lib).pivot_table(index=["a", "b"], values="v", aggfunc=lambda s: s.max()),
    lambda lib: (
        frame(lib)
        .assign(c=["p", "q", "q", "p"])
        .pivot_table(index=["a", "b"], columns="c", values="v")
        .to_dict()
    ),
    lambda lib: (
        frame(lib)
        .assign(c=["p", "q", "q", "r"])
        .pivot(index=["a", "b"], columns="c", values="v")
        .to_dict()
    ),
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
