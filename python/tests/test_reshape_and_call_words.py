"""Reshaping mistakes and wrong calls, refused with the sentences pandas prints.

A wrong call is refused by Python before the method runs, and the sentence names
the function, so each method carries the name pandas' own would have: the class
for most, `NDFrame` for those pandas shares between a frame and a column, and
the bare name where numpy's argument checks speak.
"""

from __future__ import annotations

from types import ModuleType

import pytest


def _frame(firepanda: ModuleType) -> object:
    return firepanda.DataFrame({"k": ["a", "a", "b"], "c": ["x", "y", "x"], "v": [1, 2, 3]})


def test_pivot_table_names_the_frame_group_by(firepanda: ModuleType) -> None:
    """With or without a columns key, the function is looked up on the values."""
    frame = _frame(firepanda)
    words = r"^'bogus' is not a valid function for 'DataFrameGroupBy' object$"
    with pytest.raises(AttributeError, match=words):
        frame.pivot_table(index="k", values="v", aggfunc="bogus")
    with pytest.raises(AttributeError, match=words):
        frame.pivot_table(index="k", columns="c", values="v", aggfunc="bogus")


@pytest.mark.parametrize(
    ("keys", "name", "words"),
    [
        ({"index": "k", "columns": "c"}, "a", r'^Conflicting name "a" in margins$'),
        ({"index": "c", "columns": "k"}, "b", r'^Conflicting name "b" in margins$'),
        ({"index": "k"}, "a", r'^Conflicting name "a" in margins$'),
        ({"index": "k", "columns": "c"}, 1, r"^margins_name argument must be a string$"),
    ],
)
def test_margins_need_a_name_no_label_has(
    firepanda: ModuleType, keys: dict[str, str], name: object, words: str
) -> None:
    """The totals row and column cannot share a label with the table."""
    with pytest.raises(ValueError, match=words):
        _frame(firepanda).pivot_table(values="v", margins=True, margins_name=name, **keys)


def test_margins_with_a_free_name_still_total(firepanda: ModuleType) -> None:
    """The default name is not a label, so the totals are added."""
    table = _frame(firepanda).pivot_table(index="k", values="v", margins=True)
    assert table.index.tolist() == ["a", "b", "All"]


def test_get_dummies_selects_its_columns_first(firepanda: ModuleType) -> None:
    """A missing column is the KeyError selecting it gives."""
    frame = _frame(firepanda)
    with pytest.raises(KeyError, match=r"None of \[Index\(\['zz'\], dtype='str'\)\]"):
        firepanda.get_dummies(frame, columns=["zz"])
    with pytest.raises(KeyError, match=r"\['zz'\] not in index"):
        firepanda.get_dummies(frame, columns=["zz", "k"])


@pytest.mark.parametrize(
    ("call", "words"),
    [
        (lambda f: f.pivot(index="k"), r"^DataFrame\.pivot\(\) missing 1 required keyword-only"),
        (lambda f: f.head(zz=1), r"^NDFrame\.head\(\) got an unexpected keyword argument 'zz'$"),
        (lambda f: f.fillna(0, method="ffill"), r"^NDFrame\.fillna\(\) got an unexpected"),
        (lambda f: f.merge(), r"^DataFrame\.merge\(\) missing 1 required positional"),
        (lambda f: f.div(), r"^DataFrame\.truediv\(\) missing 1 required positional"),
        (lambda f: f.keys(1), r"^NDFrame\.keys\(\) takes 1 positional argument"),
        (lambda f: f["v"].tolist(zz=1), r"^IndexOpsMixin\.tolist\(\) got an unexpected"),
        (lambda f: f["v"].shift(zz=1), r"^NDFrame\.shift\(\) got an unexpected"),
        (lambda f: f["v"].sort_values(zz=1), r"^Series\.sort_values\(\) got an unexpected"),
        (lambda f: f["v"].keys(1), r"^Series\.keys\(\) takes 1 positional argument"),
    ],
)
def test_wrong_calls_name_what_pandas_names(
    firepanda: ModuleType, call: object, words: str
) -> None:
    """The qualified name in Python's own sentence is pandas' one."""
    with pytest.raises(TypeError, match=words):
        call(_frame(firepanda))  # type: ignore[operator]
