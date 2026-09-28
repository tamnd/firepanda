"""`attrs`, `flags` and `set_flags`, compared with pandas.

Which answers keep `attrs` is a list in pandas rather than a rule, so the
sweep below calls every method either library answers with no arguments, on
both, and asks that the same ones keep them. The cases after it are the ones
that need arguments.
"""

from __future__ import annotations

import importlib.util
import pickle
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def held(m: ModuleType, attrs: dict[str, Any] | None = None) -> Any:
    frame = m.DataFrame({"a": [1, 2, 1], "b": [1.0, 2, 3], "c": [3, 1, 2]})
    frame.attrs.update({"x": [1]} if attrs is None else attrs)
    return frame


def dated(m: ModuleType) -> Any:
    column = m.Series(m.to_datetime(["2024-01-01", "2024-02-03"]))
    column.attrs["x"] = [1]
    return column


def resampled(m: ModuleType) -> Any:
    frame = dated(m).to_frame("t").assign(v=1).set_index("t")
    frame.attrs["x"] = [1]
    return frame.resample("D")


SOURCES: dict[str, Callable[[ModuleType], Any]] = {
    "frame": held,
    "column": lambda m: held(m)["b"],
    "group by": lambda m: held(m).groupby("a"),
    "column group by": lambda m: held(m).groupby("a")["b"],
    "str": lambda m: held(m)["a"].astype(str).str,
    "dt": lambda m: dated(m).dt,
    "cat": lambda m: held(m)["a"].astype(str).astype("category").cat,
    "resample": resampled,
    "rolling": lambda m: held(m).rolling(2),
    "expanding": lambda m: held(m)["b"].expanding(),
    "ewm": lambda m: held(m)["b"].ewm(1),
}

OUTSIDE = {"plot", "hist", "boxplot", "info", "style", "sparse", "to_clipboard"}


def kept(m: ModuleType, source: str) -> dict[str, bool]:
    """Each method of the source answered with no arguments, and whether its answer kept them."""
    answers: dict[str, bool] = {}
    for name in dir(SOURCES[source](m)):
        if name.startswith("_") or name.startswith("to_") or name in OUTSIDE:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                member = getattr(SOURCES[source](m), name)
                answer = member() if callable(member) else member
        except Exception:
            continue
        if isinstance(answer, m.DataFrame | m.Series):
            answers[name] = bool(answer.attrs)
    return answers


@needs_pandas
@pytest.mark.parametrize("source", list(SOURCES))
def test_every_method_keeps_them_where_pandas_does(firepanda: ModuleType, source: str) -> None:
    """Every method both answer with no arguments keeps them or drops them alike."""
    import pandas as pd

    mine, theirs = kept(firepanda, source), kept(pd, source)
    both = sorted(set(mine) & set(theirs))
    assert both
    assert {name: mine[name] for name in both} == {name: theirs[name] for name in both}


def other(m: ModuleType) -> Any:
    return held(m, {"y": 2})


def plain(m: ModuleType) -> Any:
    return m.DataFrame({"a": [1, 2, 1], "b": [1.0, 2, 3], "c": [3, 1, 2]})


ANSWERS: dict[str, Callable[[ModuleType], Any]] = {
    "the other side's": lambda m: held(m) + other(m),
    "the other side's the other way": lambda m: other(m) + held(m),
    "a plain other side": lambda m: plain(m) + held(m),
    "a number": lambda m: held(m) * 2,
    "a method operator": lambda m: held(m).eq(other(m)),
    "a column and a plain column": lambda m: plain(m)["a"] + held(m)["a"],
    "negated": lambda m: -held(m),
    "merge alike": lambda m: held(m).merge(held(m)),
    "merge unlike": lambda m: held(m).merge(plain(m)),
    "the merge function": lambda m: m.merge(held(m), held(m)),
    "join unlike": lambda m: held(m).join(plain(m), rsuffix="r"),
    "join alike": lambda m: held(m).join(held(m), rsuffix="r"),
    "concat alike": lambda m: m.concat([held(m), held(m)]),
    "concat unlike": lambda m: m.concat([plain(m), held(m)]),
    "concat columns": lambda m: m.concat([held(m)["a"], held(m)["a"]]),
    "agg a name": lambda m: held(m).agg("sum"),
    "agg a list": lambda m: held(m).agg(["sum", "min"]),
    "agg a dictionary": lambda m: held(m).agg({"a": "sum"}),
    "agg a function": lambda m: held(m)["b"].agg(lambda v: v + 1),
    "group agg a name": lambda m: held(m).groupby("a").agg("sum"),
    "group agg a dropped name": lambda m: held(m).groupby("a").agg("std"),
    "group agg a dictionary": lambda m: held(m).groupby("a").agg({"b": "sum"}),
    "group agg a list": lambda m: held(m).groupby("a")["b"].agg(["sum", "min"]),
    "group agg a function": lambda m: held(m).groupby("a").agg(lambda v: v.sum()),
    "group transform": lambda m: held(m).groupby("a").transform("sum"),
    "group get_group": lambda m: held(m).groupby("a").get_group(1),
    "group apply": lambda m: held(m).groupby("a").apply(lambda g: g.sum()),
    "group filter": lambda m: held(m).groupby("a").filter(lambda g: True),
    "a column of a group": lambda m: held(m).groupby("a")["b"].mean(),
    "a group of a column": lambda m: held(m)["b"].groupby(held(m)["a"]).sum(),
    "loc": lambda m: held(m).loc[[0], ["a"]],
    "iloc": lambda m: held(m).iloc[:1],
    "a mask": lambda m: held(m)[held(m)["a"] > 1],
    "columns": lambda m: held(m)[["a"]],
    "assign": lambda m: held(m).assign(d=1),
    "where": lambda m: held(m).where(held(m) > 1, other(m)),
    "query": lambda m: held(m).query("a > 1"),
    "sort_values": lambda m: held(m).sort_values("a"),
    "set_index": lambda m: held(m).set_index("a"),
    "rename": lambda m: held(m).rename(columns={"a": "z"}),
    "fillna": lambda m: held(m).fillna(0),
    "astype": lambda m: held(m).astype(float),
    "apply": lambda m: held(m).apply(lambda c: c),
    "map": lambda m: held(m)["a"].map(str),
    "melt as a method": lambda m: held(m).melt(),
    "melt as a function": lambda m: m.melt(held(m)),
    "pivot_table": lambda m: held(m).pivot_table(index="a", values="b", aggfunc="sum"),
    "get_dummies": lambda m: m.get_dummies(held(m)),
    "to_numeric": lambda m: m.to_numeric(held(m)["a"]),
    "cut": lambda m: m.cut(held(m)["b"], 2, labels=False),
    "a window of a selection": lambda m: held(m).rolling(2)["a"].sum(),
    "set_flags": lambda m: held(m).set_flags(allows_duplicate_labels=False),
    "a row of iterrows": lambda m: next(held(m).iterrows())[1],
    "a column of items": lambda m: next(iter(held(m).items()))[1],
    "a group of iterating": lambda m: next(iter(held(m).groupby("a")))[1],
    "pipe": lambda m: held(m).pipe(lambda f: plain(m)),
}


@needs_pandas
@pytest.mark.parametrize("name", list(ANSWERS))
def test_an_answer_keeps_what_pandas_keeps(firepanda: ModuleType, name: str) -> None:
    """The `attrs` of the answer, and whether it refuses duplicate labels."""
    import pandas as pd

    def shown(m: ModuleType) -> Any:
        answer = ANSWERS[name](m)
        return answer.attrs, answer.flags.allows_duplicate_labels

    assert shown(firepanda) == shown(pd)


def refused(m: ModuleType) -> Any:
    return m.DataFrame({"a": [1, 2]}).set_flags(allows_duplicate_labels=False)


def refusing(obj: Any) -> Any:
    """Refuses duplicate labels on an object the caller keeps.

    pandas' flags hold their object weakly, so on an object nobody holds they
    say the object has been deleted instead.
    """
    obj.flags.allows_duplicate_labels = False
    return obj


KEPT: list[Any] = []


def kept_alive(obj: Any) -> Any:
    KEPT.append(obj)
    return obj


MISTAKES: dict[str, Callable[[ModuleType], Any]] = {
    "repeated labels": lambda m: refusing(
        kept_alive(m.DataFrame({"a": [1, 2, 3, 4]}, index=[5, 5, 6, 6]))
    ),
    "repeated text labels": lambda m: kept_alive(
        m.Series([1, 2, 3, 4, 5], index=["x", "yy", "x", "zz", "yy"])
    ).set_flags(allows_duplicate_labels=False),
    "a concat that repeats": lambda m: m.concat([kept_alive(refused(m)), kept_alive(refused(m))]),
    "a reindex that repeats": lambda m: kept_alive(refused(m)).reindex([0, 0]),
    "an unknown flag": lambda m: refused(m).flags["x"],
    "setting an unknown flag": lambda m: refused(m).flags.__setitem__("x", 1),
}


@needs_pandas
@pytest.mark.parametrize("name", list(MISTAKES))
def test_a_mistake_raises_what_pandas_raises(firepanda: ModuleType, name: str) -> None:
    """The same kind of error and the same words, the table of positions included.

    pandas' `DuplicateLabelError` is its own class, so the kinds are compared by name.
    """
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        MISTAKES[name](pd)
    with pytest.raises(Exception) as mine:
        MISTAKES[name](firepanda)
    assert type(mine.value).__name__ == type(theirs.value).__name__
    assert isinstance(mine.value, ValueError) == isinstance(theirs.value, ValueError)
    assert str(mine.value) == str(theirs.value)


@needs_pandas
def test_the_flags_read_as_pandas_flags_do(firepanda: ModuleType) -> None:
    """The repr, equality, the item form, and a flags object made by hand."""
    import pandas as pd

    def shown(m: ModuleType) -> Any:
        frame = m.DataFrame({"a": [1, 2]})
        flags = frame.flags
        flags["allows_duplicate_labels"] = False
        by_hand = m.Flags(frame, allows_duplicate_labels=True)
        return (
            repr(flags),
            repr(frame.flags),
            flags == frame.flags,
            flags == by_hand,
            flags == 1,
            by_hand["allows_duplicate_labels"],
            repr(frame.set_flags(allows_duplicate_labels=True).flags),
        )

    assert shown(firepanda) == shown(pd)


def test_attrs_are_copied_deeply(firepanda: ModuleType) -> None:
    """An answer's `attrs` are its own, so changing them leaves the source alone."""
    frame = held(firepanda)
    answer = frame.head()
    answer.attrs["x"].append(2)
    answer.attrs["z"] = 3
    assert frame.attrs == {"x": [1]}


def test_attrs_are_set_as_a_dictionary(firepanda: ModuleType) -> None:
    """Setting `attrs` takes anything `dict` takes, and a frame starts with none."""
    frame = firepanda.DataFrame({"a": [1]})
    assert frame.attrs == {}
    frame.attrs = [("k", 1)]
    assert frame.attrs == {"k": 1}
    assert frame.head().attrs == {"k": 1}


def test_the_copy_keyword_is_deprecated(firepanda: ModuleType) -> None:
    """`copy` warns and does nothing, as it does across pandas 3."""
    with pytest.warns(DeprecationWarning, match="copy keyword is deprecated"):
        firepanda.DataFrame({"a": [1]}).set_flags(copy=True)


def test_a_frame_with_no_attrs_is_not_touched(firepanda: ModuleType) -> None:
    """The answer of a frame holding nothing holds nothing either, and pickles as before."""
    frame = firepanda.DataFrame({"a": [1, 2]})
    assert (frame + 1).attrs == {}
    assert frame.flags.allows_duplicate_labels
    with pytest.raises(TypeError):
        pickle.dumps(frame)
