"""`Timestamp(fold=)` and `MultiIndex.sortorder`, the way pandas takes them.

pandas takes a fold only beside a wall clock with no zone, a naive `datetime`
or the fields given by name, where it picks the side of a repeated hour, and
refuses it beside anything else. A sortorder is kept as given, carried by a
copy, a mask and a forward slice, and checked against the rows when the index
is built from codes or a product. Each test runs the same code on both
libraries and compares what they print.
"""

from __future__ import annotations

import datetime
import re
from typing import Any

import pandas as pd
import pytest

NAIVE = datetime.datetime(2024, 11, 3, 1, 30)

FOLDS = {
    "naive first": lambda lib: lib.Timestamp(NAIVE, tz="US/Eastern", fold=0),
    "naive second": lambda lib: lib.Timestamp(NAIVE, tz="US/Eastern", fold=1),
    "named fields": lambda lib: lib.Timestamp(
        year=2024, month=11, day=3, hour=1, minute=30, tz="US/Eastern", fold=1
    ),
    "plain": lambda lib: lib.Timestamp(datetime.datetime(2024, 1, 1), fold=1).fold,
}

REFUSED = {
    "text": lambda lib: lib.Timestamp("2024-11-03 01:30", tz="US/Eastern", fold=1),
    "number": lambda lib: lib.Timestamp(0, fold=1),
    "zoned": lambda lib: lib.Timestamp(datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC), fold=1),
    "positional fields": lambda lib: lib.Timestamp(2024, 11, 3, 1, 30, fold=0),
    "date": lambda lib: lib.Timestamp(datetime.date(2024, 1, 1), fold=1),
    "two": lambda lib: lib.Timestamp(NAIVE, fold=2),
}


def mi(lib: Any, **kwargs: Any) -> Any:
    return lib.MultiIndex.from_arrays([["a", "a", "b"], [2, 1, 3]], **kwargs)


SORTORDERS = {
    "default": lambda lib: mi(lib).sortorder,
    "arrays": lambda lib: mi(lib, sortorder=5).sortorder,
    "tuples": lambda lib: lib.MultiIndex.from_tuples([("a", 1)], sortorder=2).sortorder,
    "product": lambda lib: lib.MultiIndex.from_product([["a"], [1, 2]], sortorder=2).sortorder,
    "codes": lambda lib: (
        lib.MultiIndex([["a", "b"], [1, 2]], [[0, 0, 1], [1, 0, 1]], sortorder=1).sortorder
    ),
    "unverified": lambda lib: (
        lib.MultiIndex(
            [["a", "b"], [1, 2]], [[0, 0, 1], [1, 0, 1]], sortorder=2, verify_integrity=False
        ).sortorder
    ),
    "slice": lambda lib: mi(lib, sortorder=1)[:2].sortorder,
    "reversed": lambda lib: mi(lib, sortorder=1)[::-1].sortorder,
    "mask": lambda lib: mi(lib, sortorder=1)[[True, False, True]].sortorder,
    "take": lambda lib: mi(lib, sortorder=1).take([0, 1]).sortorder,
    "copy": lambda lib: mi(lib, sortorder=1).copy().sortorder,
}


@pytest.mark.parametrize("make", FOLDS.values(), ids=FOLDS.keys())
def test_fold_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


@pytest.mark.parametrize("make", REFUSED.values(), ids=REFUSED.keys())
def test_fold_is_refused_as_pandas_refuses_it(firepanda: Any, make: Any) -> None:
    with pytest.raises(ValueError) as expected:
        make(pd)
    with pytest.raises(ValueError, match=re.escape(str(expected.value)[:40])):
        make(firepanda)


@pytest.mark.parametrize("make", SORTORDERS.values(), ids=SORTORDERS.keys())
def test_sortorder_is_pandas(firepanda: Any, make: Any) -> None:
    assert make(firepanda) == make(pd)


def test_a_sortorder_past_the_sorted_levels_is_refused(firepanda: Any) -> None:
    with pytest.raises(ValueError, match="sortorder 2 with lexsort_depth 1"):
        firepanda.MultiIndex([["a", "b"], [1, 2]], [[0, 0, 1], [1, 0, 1]], sortorder=2)
    with pytest.raises(ValueError, match="sortorder 2 with lexsort_depth 0"):
        firepanda.MultiIndex.from_product([["b", "a"], [1, 2]], sortorder=2)
