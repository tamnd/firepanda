"""Categories of instants and spans, which pandas keeps in their own type.

pandas makes a category column of a datetime or timedelta column with the
categories still instants or spans. A gap reads as NaT, the labels print as
their own index prints them, and the fields under `.dt` come off the values.
Each test runs the same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def dates(lib: Any) -> Any:
    return lib.Series(lib.to_datetime(["2024-01-02", "2024-01-01", None, "2024-01-02"]))


def spans(lib: Any) -> Any:
    return lib.Series(lib.to_timedelta(["1D", "2h", "1D"]))


def held(lib: Any) -> Any:
    return dates(lib).astype("category")


def plain(lib: Any) -> Any:
    return lib.to_datetime(["2024-01-02", "2024-01-01", "2024-01-02"])


BUILDS = {
    "dates": held,
    "categories": lambda lib: held(lib).cat.categories,
    "codes": lambda lib: held(lib).cat.codes.tolist(),
    "back": lambda lib: held(lib).astype("datetime64[us]"),
    "values": lambda lib: held(lib).tolist(),
    "value_counts": lambda lib: held(lib).value_counts(),
    "spans": lambda lib: spans(lib).astype("category"),
    "span categories": lambda lib: spans(lib).astype("category").cat.categories,
    "span counts": lambda lib: spans(lib).astype("category").value_counts(),
    "zoned": lambda lib: dates(lib).dropna().dt.tz_localize("UTC").astype("category"),
    "frame": lambda lib: lib.DataFrame({"d": dates(lib)}).astype("category"),
    "item": lambda lib: held(lib)[0],
    "group by": lambda lib: (
        lib.DataFrame({"d": held(lib), "v": [1, 2, 3, 4]}).groupby("d", observed=True).sum()
    ),
    "equal": lambda lib: (held(lib) == lib.Timestamp("2024-01-02")).tolist(),
    "max": lambda lib: held(lib).cat.as_ordered().max(),
    "sort": lambda lib: held(lib).sort_values(),
    "unique": lambda lib: held(lib).unique(),
    "categorical": lambda lib: lib.Categorical(plain(lib)),
    "dtype": lambda lib: lib.Series(plain(lib), dtype="category"),
    "index": lambda lib: lib.CategoricalIndex(plain(lib)),
    "index astype": lambda lib: lib.Index(plain(lib)).astype("category"),
    "unobserved": lambda lib: (
        lib.DataFrame({"d": lib.Series(plain(lib)).astype("category"), "v": [1, 2, 3]})
        .groupby("d", observed=False)
        .v.sum()
    ),
    "set_index": lambda lib: lib.DataFrame(
        {"d": lib.Series(plain(lib)).astype("category"), "v": [1, 2, 3]}
    ).set_index("d"),
    "day": lambda lib: lib.Series(plain(lib)).astype("category").dt.day,
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_timed_categories_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_an_ordered_minimum_is_the_value(firepanda: Any) -> None:
    for values in ([3, 1, 2], ["b", "a"]):
        mine = firepanda.Series(firepanda.Categorical(values, ordered=True))
        theirs = pd.Series(pd.Categorical(values, ordered=True))
        assert (mine.min(), mine.max()) == (theirs.min(), theirs.max())
