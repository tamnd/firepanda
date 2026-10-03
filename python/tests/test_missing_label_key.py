"""A label that is not there is a KeyError carrying the label, as pandas raises it.

A caller who catches the error reads the label back out of `args`, so the
error has to hold the label alone rather than a sentence about it.
"""

from __future__ import annotations

from types import ModuleType

import pytest


@pytest.mark.parametrize(
    ("labels", "missing"),
    [(["p", "q", "r"], "zz"), ([1, 2], 5)],
)
def test_get_loc_of_a_missing_label(
    firepanda: ModuleType, labels: list[object], missing: object
) -> None:
    """The error's one argument is the label that was asked for."""
    with pytest.raises(KeyError) as caught:
        firepanda.Index(labels).get_loc(missing)
    assert caught.value.args == (missing,)


def test_a_range_get_loc_of_a_missing_label(firepanda: ModuleType) -> None:
    """A range of labels says the same."""
    with pytest.raises(KeyError) as caught:
        firepanda.RangeIndex(3).get_loc(7)
    assert caught.value.args == (7,)


def test_at_a_missing_row(firepanda: ModuleType) -> None:
    """`at` finds the row through `get_loc`, so it says the same."""
    frame = firepanda.DataFrame({"a": [1, 2, 3]}, index=["p", "q", "r"])
    with pytest.raises(KeyError) as caught:
        frame.at["zz", "a"]
    assert caught.value.args == ("zz",)
    assert frame.at["q", "a"] == 2
