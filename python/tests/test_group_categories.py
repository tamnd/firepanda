"""Grouping by a category column, with and without the categories that have no rows.

The groups come in the order of the categories, and the key comes back as a
category column carrying every category. Under `observed=False` each category
with no rows is a group too, and a reduction answers what pandas answers for an
empty group: nothing for a count, a sum or `nunique`, one for a product, false
for `any`, true for `all`, and a gap for the rest.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def frame(m: ModuleType, ordered: bool = False) -> Any:
    key = (
        m.Series(["b", "a", "b", None, "a"])
        .astype("category")
        .cat.set_categories(["c", "b", "a", "d"], ordered=ordered)
    )
    return m.DataFrame(
        {"k": key, "v": [1, 2, 3, 4, 5], "f": [1.5, 2.5, None, 3.0, 4.0], "t": list("xyzwu")}
    )


def every(m: ModuleType, **kwargs: Any) -> Any:
    return frame(m).groupby("k", observed=False, **kwargs)


ANSWERS: dict[str, Callable[[ModuleType], Any]] = {
    "size": lambda m: frame(m).groupby("k").size(),
    "sum": lambda m: frame(m).groupby("k")[["v", "f"]].sum(),
    "one column": lambda m: frame(m).groupby("k")["v"].mean(),
    "ordered": lambda m: frame(m, True).groupby("k")["v"].max(),
    "unsorted": lambda m: frame(m).groupby("k", sort=False)["v"].sum(),
    "as columns": lambda m: frame(m).groupby("k", as_index=False)["v"].sum(),
    "gap kept": lambda m: frame(m).groupby("k", dropna=False)["v"].sum(),
    "agg": lambda m: frame(m).groupby("k").agg({"v": "sum", "f": "min"}),
    "transform": lambda m: frame(m).groupby("k")["v"].transform("sum"),
    "cumsum": lambda m: frame(m).groupby("k")["v"].cumsum(),
    "head": lambda m: frame(m).groupby("k").head(1),
    "filter": lambda m: frame(m).groupby("k").filter(lambda g: len(g) > 1),
    "get_group": lambda m: frame(m).groupby("k").get_group("b"),
    "a column by a category": lambda m: frame(m)["v"].groupby(frame(m)["k"]).sum(),
    "every size": lambda m: every(m).size(),
    "every count": lambda m: every(m).count(),
    "every sum": lambda m: every(m)["v"].sum(),
    "every sum of floats": lambda m: every(m)["f"].sum(),
    "every prod": lambda m: every(m)[["v", "f"]].prod(),
    "every nunique": lambda m: every(m)["v"].nunique(),
    "every mean": lambda m: every(m)[["v", "f"]].mean(),
    "every min": lambda m: every(m)["v"].min(),
    "every first": lambda m: every(m)["f"].first(),
    "every any": lambda m: every(m)["v"].any(),
    "every all": lambda m: every(m)["v"].all(),
    "every std": lambda m: every(m)["v"].std(),
    "every agg name": lambda m: every(m)["v"].agg("sum"),
    "every agg list": lambda m: every(m)["v"].agg(["sum", "mean", "size"]),
    "every agg dict": lambda m: every(m).agg({"v": "sum", "f": "mean"}),
    "every unsorted": lambda m: every(m, sort=False)["v"].sum(),
    "every as columns": lambda m: every(m, as_index=False)["v"].sum(),
    "every as columns size": lambda m: every(m, as_index=False).size(),
    "every gap kept": lambda m: every(m, dropna=False)["v"].sum(),
    "every ordered": lambda m: frame(m, True).groupby("k", observed=False)["v"].max(),
    "every transform": lambda m: every(m)["v"].transform("sum"),
    "every head": lambda m: every(m).head(1),
    "every ngroup": lambda m: every(m).ngroup(),
    "every a column by a category": lambda m: (
        frame(m)["v"].groupby(frame(m)["k"], observed=False).sum()
    ),
    "observed on other keys": lambda m: frame(m).groupby("v", observed=False)["f"].sum(),
}

MISTAKES: dict[str, Callable[[ModuleType], Any]] = {
    "idxmax with a category with no rows": lambda m: every(m)["v"].idxmax(),
    "a group that is not a category": lambda m: frame(m).groupby("k").get_group("z"),
}


def kind(column: Any) -> str:
    """A column's type as pandas prints it, where firepanda's text is `string`."""
    return "str" if str(column.dtype) == "string" else str(column.dtype)


def labels(answer: Any) -> str:
    """The labels as pandas prints a category's, and as a list otherwise.

    A frame with a category column has plain labels in firepanda rather than a
    `RangeIndex`, which is not what these tests are about.
    """
    index = answer.index
    return repr(index) if str(index.dtype) == "category" else repr(index.tolist())


def shown(answer: Any) -> Any:
    """An answer's shape, names, labels and values, spelled the same in both."""
    if hasattr(answer, "columns"):
        values = {name: (repr(answer[name].tolist()), kind(answer[name])) for name in answer}
        return list(answer.columns), labels(answer), values
    return answer.name, labels(answer), repr(answer.tolist()), kind(answer)


@needs_pandas
@pytest.mark.parametrize("name", list(ANSWERS))
def test_an_answer_matches_pandas(firepanda: ModuleType, name: str) -> None:
    """The columns, their types, the labels as pandas prints them, and every value."""
    import pandas as pd

    assert shown(ANSWERS[name](firepanda)) == shown(ANSWERS[name](pd))


@needs_pandas
@pytest.mark.parametrize("name", list(MISTAKES))
def test_a_mistake_raises_what_pandas_raises(firepanda: ModuleType, name: str) -> None:
    """The same kind of error and the same words."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        MISTAKES[name](pd)
    with pytest.raises(theirs.type) as mine:
        MISTAKES[name](firepanda)
    assert str(mine.value) == str(theirs.value)


@needs_pandas
def test_iterating_hands_back_the_categories(firepanda: ModuleType) -> None:
    """Each key is the category, and each group keeps the key as a category column."""
    import pandas as pd

    def groups(m: ModuleType) -> Any:
        return [(k, g["v"].tolist(), str(g["k"].dtype)) for k, g in frame(m).groupby("k")]

    assert groups(firepanda) == groups(pd)


@needs_pandas
def test_the_labels_of_a_category_answer_are_a_categorical_index(firepanda: ModuleType) -> None:
    """The categories, the order flag and the codes, as `CategoricalIndex` has them."""
    import pandas as pd

    def held(m: ModuleType) -> Any:
        index = frame(m, True).groupby("k", dropna=False)["v"].sum().index
        return index.categories.tolist(), index.ordered, index.codes.tolist()

    assert held(firepanda) == held(pd)
    assert not hasattr(firepanda.Index([1, 2]), "categories")


def test_every_category_over_several_keys_is_refused(firepanda: ModuleType) -> None:
    """Every mix of the keys' categories is labelled by a MultiIndex."""
    with pytest.raises(NotImplementedError):
        frame(firepanda).groupby(["k", "v"], observed=False)


def test_apply_with_every_category_is_refused(firepanda: ModuleType) -> None:
    """pandas calls the function on an empty group for each category with no rows."""
    with pytest.raises(NotImplementedError):
        every(firepanda)["v"].apply(lambda g: g.sum())
