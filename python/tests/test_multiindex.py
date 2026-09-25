"""`MultiIndex`, checked against pandas.

Every way of building one, every member that answers another index, a lookup
or an array, and the mistakes pandas refuses. An index is compared as its rows,
names, levels and codes, an array as a list, and a flat index as its labels and
name, with gaps as None and the text type spelled one way.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def plain(value: Any) -> Any:
    """A value as compared: a gap as None, a row value by value, a numpy number as Python."""
    if isinstance(value, tuple):
        return tuple(plain(v) for v in value)
    if value is None or (isinstance(value, float) and value != value):
        return None
    return value.item() if hasattr(value, "item") and not hasattr(value, "__len__") else value


def facts(answer: Any) -> Any:
    """What is compared, for whatever kind of answer a member gives."""
    if type(answer).__name__ == "MultiIndex":
        return (
            "multi",
            [plain(row) for row in answer.tolist()],
            list(answer.names),
            [[plain(v) for v in level.tolist()] for level in answer.levels],
            [[int(c) for c in code] for code in answer.codes],
        )
    if type(answer).__name__ == "Index":
        return ("index", [plain(v) for v in answer.tolist()], answer.name)
    if hasattr(answer, "columns"):
        return (
            "frame",
            list(answer.columns),
            [[plain(v) for v in answer[c].tolist()] for c in answer.columns],
        )
    if hasattr(answer, "tolist") and hasattr(answer, "__len__"):
        return ("array", [plain(v) for v in answer.tolist()])
    if isinstance(answer, slice):
        return ("slice", plain(answer.start), plain(answer.stop), answer.step)
    if isinstance(answer, tuple):
        return tuple(facts(each) for each in answer)
    if isinstance(answer, list):
        return [facts(each) for each in answer]
    if isinstance(answer, dict):
        return {key: facts(each) for key, each in answer.items()}
    return plain(answer)


def keyed(m: ModuleType) -> Any:
    """Four rows out of order, named."""
    return m.MultiIndex.from_arrays([["b", "a", "b", "a"], [2, 1, 1, 2]], names=["k", "n"])


def ordered(m: ModuleType) -> Any:
    """Four sorted rows, named."""
    return m.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 3)], names=["k", "n"])


def gappy(m: ModuleType) -> Any:
    """Three rows with a gap on each level."""
    return m.MultiIndex.from_arrays([["x", None, "y"], [1.5, 2.5, None]])


def repeated(m: ModuleType) -> Any:
    """Four rows, one of them three times."""
    return m.MultiIndex.from_arrays([["a", "a", "b", "a"], [1, 1, 2, 1]])


BUILDS: list[Callable[[Any], Any]] = [
    keyed,
    ordered,
    gappy,
    lambda m: m.MultiIndex.from_product([[2, 1], ["p", "q"]], names=["n", "c"]),
    lambda m: m.MultiIndex.from_frame(m.DataFrame({"x": [1, 2], "y": ["a", "b"]})),
    lambda m: m.MultiIndex.from_arrays([m.Series(["a", "b"], name="s"), [1, 2]]),
    lambda m: m.MultiIndex.from_tuples([(1, 2), (1,)]),
    lambda m: m.MultiIndex.from_tuples([], names=["a", "b"]),
    lambda m: m.MultiIndex(
        levels=[["a", "b"], [1, 2]], codes=[[0, 1, 1], [1, 0, 1]], names=["k", None]
    ),
    lambda m: m.MultiIndex(levels=[["a", "b"]], codes=[[0, -1]], name=["only"]),
    lambda m: (keyed(m).nlevels, keyed(m).levshape, keyed(m).shape, keyed(m).size),
    lambda m: (keyed(m).ndim, keyed(m).empty, len(keyed(m)), keyed(m).name),
    lambda m: (keyed(m).is_unique, repeated(m).has_duplicates, keyed(m).inferred_type),
    lambda m: (ordered(m).is_monotonic_increasing, keyed(m).is_monotonic_increasing),
    lambda m: ordered(m)[::-1].is_monotonic_decreasing,
    lambda m: list(keyed(m)),
    lambda m: keyed(m).tolist() == keyed(m).to_list(),
    lambda m: keyed(m).to_numpy(),
    lambda m: keyed(m).values,
    lambda m: keyed(m).codes,
    lambda m: keyed(m).levels,
    lambda m: keyed(m)[1],
    lambda m: keyed(m)[-1],
    lambda m: keyed(m)[1:3],
    lambda m: keyed(m)[[3, 0]],
    lambda m: keyed(m)[[True, False, True, False]],
    lambda m: (("a", 1) in keyed(m), "a" in keyed(m), ("a", 5) in keyed(m), "z" in keyed(m)),
    lambda m: keyed(m).get_level_values(0),
    lambda m: keyed(m).get_level_values("n"),
    lambda m: keyed(m).get_level_values(-1),
    lambda m: gappy(m).get_level_values(1),
    lambda m: keyed(m).droplevel(0),
    lambda m: keyed(m).droplevel("n"),
    lambda m: m.MultiIndex.from_arrays([[1, 2], [3, 4], [5, 6]], names=list("abc")).droplevel(
        ["a", "b"]
    ),
    lambda m: m.MultiIndex.from_arrays([[1, 2], [3, 4], [5, 6]], names=list("abc")).droplevel("b"),
    lambda m: keyed(m).swaplevel(),
    lambda m: keyed(m).swaplevel(0, "n"),
    lambda m: keyed(m).reorder_levels(["n", "k"]),
    lambda m: keyed(m).copy(names=["p", "q"]),
    lambda m: keyed(m).copy(),
    lambda m: keyed(m).rename(["p", "q"]),
    lambda m: keyed(m).set_names("z", level=1),
    lambda m: keyed(m).set_names(["y", "z"], level=["n", "k"]),
    lambda m: keyed(m).set_names({"k": "K"}),
    lambda m: keyed(m).set_levels(["A", "B"], level=0),
    lambda m: keyed(m).set_levels([["A", "B"], [10, 20]]),
    lambda m: keyed(m).set_codes([0, 0, 1, 1], level=0),
    lambda m: keyed(m).set_codes([[0, 0, 1, 1], [0, 1, 0, 1]]),
    lambda m: keyed(m)[:2].remove_unused_levels(),
    lambda m: keyed(m).sortlevel(),
    lambda m: keyed(m).sortlevel(1, ascending=False),
    lambda m: keyed(m).sortlevel([0, 1], ascending=[True, False]),
    lambda m: keyed(m).sortlevel("n", sort_remaining=False),
    lambda m: keyed(m).sort_values(),
    lambda m: keyed(m).sort_values(ascending=False, return_indexer=True),
    lambda m: gappy(m).sort_values(),
    lambda m: keyed(m).argsort(),
    lambda m: keyed(m).take([2, 0]),
    lambda m: keyed(m).take([-1]),
    lambda m: keyed(m).delete(1),
    lambda m: keyed(m).delete([0, -1]),
    lambda m: keyed(m).insert(1, ("c", 9)),
    lambda m: keyed(m).insert(0, ("a", 2)),
    lambda m: keyed(m).append(ordered(m)),
    lambda m: keyed(m).append([ordered(m), keyed(m)]),
    lambda m: keyed(m).repeat(2),
    lambda m: keyed(m).repeat([1, 0, 2, 1]),
    lambda m: keyed(m).drop([("a", 1)]),
    lambda m: keyed(m).drop("a", level=0),
    lambda m: ordered(m).drop("a"),
    lambda m: keyed(m).drop([("z", 1)], errors="ignore"),
    lambda m: repeated(m).duplicated(),
    lambda m: repeated(m).duplicated(keep="last"),
    lambda m: repeated(m).duplicated(keep=False),
    lambda m: repeated(m).drop_duplicates(),
    lambda m: repeated(m).drop_duplicates(keep="last"),
    lambda m: repeated(m).unique(),
    lambda m: repeated(m).unique(level=0),
    lambda m: (repeated(m).nunique(), keyed(m).nunique()),
    lambda m: gappy(m).dropna(),
    lambda m: gappy(m).dropna(how="all"),
    lambda m: ordered(m).get_loc(("a", 2)),
    lambda m: ordered(m).get_loc("a"),
    lambda m: ordered(m).get_loc("b"),
    lambda m: keyed(m).get_loc("a"),
    lambda m: keyed(m).get_loc(("a", 1)),
    lambda m: repeated(m).get_loc(("a", 1)),
    lambda m: ordered(m).get_locs(["a", 2]),
    lambda m: ordered(m).get_locs([["a", "b"], 1]),
    lambda m: ordered(m).get_locs([slice(None), 1]),
    lambda m: ordered(m).get_loc_level("a"),
    lambda m: ordered(m).get_loc_level(1, level=1),
    lambda m: ordered(m).get_indexer([("b", 1), ("z", 0), ("a", 1)]),
    lambda m: ordered(m).get_indexer(keyed(m)),
    lambda m: ordered(m).get_indexer_for([("b", 3)]),
    lambda m: repeated(m).get_indexer_non_unique([("a", 1), ("q", 1)]),
    lambda m: keyed(m).isin([("a", 1), ("b", 1)]),
    lambda m: keyed(m).isin(["a"], level=0),
    lambda m: keyed(m).isin([1], level="n"),
    lambda m: keyed(m).union(ordered(m)),
    lambda m: keyed(m).union(ordered(m), sort=False),
    lambda m: keyed(m).union([("z", 9)]),
    lambda m: keyed(m).intersection(ordered(m)),
    lambda m: keyed(m).difference(ordered(m)),
    lambda m: keyed(m).symmetric_difference(ordered(m)),
    lambda m: keyed(m).join(ordered(m), how="inner"),
    lambda m: (keyed(m).equals(keyed(m)), keyed(m).equals(ordered(m)), keyed(m).equals([1])),
    lambda m: (keyed(m).identical(keyed(m)), keyed(m).identical(keyed(m).rename(["p", "q"]))),
    lambda m: (keyed(m).equal_levels(keyed(m)), keyed(m).equal_levels(ordered(m))),
    lambda m: (keyed(m).min(), keyed(m).max(), keyed(m).argmin(), keyed(m).argmax()),
    lambda m: keyed(m).factorize(),
    lambda m: keyed(m).map(lambda row: row[1]),
    lambda m: keyed(m).putmask([True, False, False, False], ordered(m)),
    lambda m: keyed(m).astype(object),
    lambda m: ordered(m).slice_locs("a", "a"),
    lambda m: ordered(m).slice_locs(("a", 2), ("b", 1)),
    lambda m: ordered(m).slice_indexer("a", "b"),
    lambda m: (
        ordered(m).get_slice_bound("a", "right"),
        ordered(m).get_slice_bound(("b", 1), "left"),
    ),
    lambda m: keyed(m).groupby([1, 2, 1, 2]),
    lambda m: keyed(m)[:1].item(),
    lambda m: (keyed(m).ravel(), keyed(m).view(), keyed(m).T, keyed(m).transpose()),
    lambda m: keyed(m).to_frame(index=False),
    lambda m: keyed(m).to_frame(index=False, name=["p", "q"]),
    lambda m: gappy(m).to_frame(index=False, name=["s", "f"]),
    lambda m: keyed(m).reindex([("a", 1), ("q", 1)]),
    lambda m: keyed(m).reindex(keyed(m)),
    lambda m: repr(keyed(m)),
    lambda m: repr(gappy(m)),
    lambda m: repr(m.MultiIndex.from_tuples([], names=["a", "b"])),
    lambda m: [str(code.dtype) for code in keyed(m).codes],
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Every way in, every member that answers an index, a lookup or an array."""
    import pandas as pd

    assert facts(build(firepanda)) == facts(build(pd))


def test_the_level_types_are_pandas_types(firepanda: ModuleType) -> None:
    """Each level's type, with text spelled the way firepanda spells it."""
    import pandas as pd

    def types(m: ModuleType) -> list[str]:
        return [str(t).replace("string", "str") for t in keyed(m).dtypes.tolist()]

    assert types(firepanda) == types(pd)
    assert list(keyed(firepanda).dtypes.index) == ["k", "n"]


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.MultiIndex(levels=[["a", "b"]]),
    lambda m: m.MultiIndex(levels=[["a", "b"]], codes=[[0, 2]]),
    lambda m: m.MultiIndex(levels=[["a", "b"], [1]], codes=[[0, 1], [0]]),
    lambda m: m.MultiIndex(levels=[["a", "a"]], codes=[[0, 1]]),
    lambda m: m.MultiIndex(levels=[["a", "b"]], codes=[[0, 1], [0, 1]]),
    lambda m: m.MultiIndex(levels=[], codes=[]),
    lambda m: m.MultiIndex(levels=[["a"]], codes=[[-2]]),
    lambda m: m.MultiIndex.from_arrays([[1, 2], [1]]),
    lambda m: m.MultiIndex.from_arrays(5),
    lambda m: m.MultiIndex.from_arrays([["a", "b"], [1, 2]], names=["x"]),
    lambda m: m.MultiIndex.from_arrays([["a", "b"], [1, 2]], names="x"),
    lambda m: m.MultiIndex.from_tuples([]),
    lambda m: m.MultiIndex.from_tuples([1, 2]),
    lambda m: m.MultiIndex.from_product(5),
    lambda m: keyed(m).array,
    lambda m: keyed(m).get_level_values("z"),
    lambda m: keyed(m).get_level_values(5),
    lambda m: keyed(m).get_level_values(-3),
    lambda m: keyed(m)[5],
    lambda m: keyed(m).take([5]),
    lambda m: keyed(m).swaplevel(0, "q"),
    lambda m: keyed(m).reorder_levels([0]),
    lambda m: keyed(m).droplevel([0, 1]),
    lambda m: keyed(m).droplevel("zz"),
    lambda m: keyed(m).set_names(["p"]),
    lambda m: keyed(m).set_names("a"),
    lambda m: keyed(m).set_names(["a", "b"], level=[0]),
    lambda m: keyed(m).set_levels(["A", "A"], level=0),
    lambda m: keyed(m).set_levels(["A"], level=0),
    lambda m: keyed(m).set_codes([[0, 5], [0, 0]]),
    lambda m: keyed(m).sortlevel("zz"),
    lambda m: keyed(m).sortlevel(0, ascending=[True, False]),
    lambda m: keyed(m).insert(0, ("a",)),
    lambda m: keyed(m).duplicated(keep="x"),
    lambda m: keyed(m).drop([("z", 1)]),
    lambda m: ordered(m).get_loc(("a", 9)),
    lambda m: ordered(m).get_loc("z"),
    lambda m: keyed(m).get_loc(("a", 1, 3)),
    lambda m: keyed(m).union([1, 2]),
    lambda m: keyed(m).item(),
    lambda m: keyed(m).isna(),
    lambda m: keyed(m).notna(),
    lambda m: keyed(m).hasnans,
    lambda m: keyed(m).fillna(1),
    lambda m: keyed(m).where([True, False, True, True]),
    lambda m: keyed(m).shift(),
    lambda m: keyed(m).round(),
    lambda m: keyed(m).diff(),
    lambda m: keyed(m).all(),
    lambda m: keyed(m).any(),
    lambda m: keyed(m).astype("int64"),
    lambda m: keyed(m).infer_objects(),
    lambda m: keyed(m).str,
    lambda m: gappy(m).dropna(how="some"),
    lambda m: keyed(m).to_frame(index=False, name=["p"]),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_mistakes_raise_as_pandas_raises(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """The same error type and words as pandas."""
    import pandas as pd

    with pytest.raises(Exception) as expected:
        build(pd)
    with pytest.raises(type(expected.value), match="^" + re.escape(str(expected.value)) + "$"):
        build(firepanda)


REFUSED: list[Callable[[Any], Any]] = [
    lambda m: keyed(m).to_frame(),
    lambda m: keyed(m).to_series(),
    lambda m: keyed(m).to_flat_index(),
    lambda m: keyed(m).value_counts(),
    lambda m: keyed(m).sort_values(key=lambda level: level),
    lambda m: keyed(m).append(m.Index([1])),
    lambda m: keyed(m).get_indexer([("a", 1)], method="pad"),
]


@pytest.mark.parametrize("build", REFUSED)
def test_what_needs_tuples_in_a_column_is_refused(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """A frame or column labelled by a MultiIndex, or holding tuples, is not there yet."""
    with pytest.raises(NotImplementedError):
        build(firepanda)


def test_the_str_member_is_on_the_class_and_not_on_an_index(firepanda: ModuleType) -> None:
    """pandas declares `str` on every index class and refuses it on this one."""
    assert callable(firepanda.MultiIndex.str)
    assert not hasattr(keyed(firepanda), "str")


def test_an_index_is_not_hashable(firepanda: ModuleType) -> None:
    """As in pandas, since it compares row by row."""
    with pytest.raises(TypeError):
        hash(keyed(firepanda))
