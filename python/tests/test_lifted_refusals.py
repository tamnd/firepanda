"""Resample keys, blanked filters, counted `nth`, number windows, copies and freqs.

pandas sets `convention` aside for timestamps, puts the bins in front of what
`apply` answers with `group_keys=True`, blanks the rows `filter` drops with
`dropna=False`, counts only whole rows in `nth` with `dropna=`, reads only the
number columns of a window with `numeric_only=True` and accepts `copy=` in its
constructors. Along the way a concat keeps a frequency only when its pieces
join up and a label at midnight shows its time under a step shorter than a
day. Each test here runs the same code on both libraries and compares what
they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

NAN = float("nan")


def stamped(lib: Any) -> Any:
    return lib.Series([1, 2, 3, 4], index=lib.date_range("2024-01-01", periods=4, freq="30min"))


def gapped(lib: Any) -> Any:
    return lib.DataFrame(
        {
            "k": ["a", "b", "a", "b", "a", None, "b", "a", "a", "b"],
            "x": [NAN, 1.0, 3.0, NAN, NAN, 5.0, NAN, NAN, 2.0, 4.0],
            "y": [1.5, NAN, NAN, 2.5, 3.5, NAN, 6.0, NAN, NAN, 1.0],
            "n": [4, 4, 1, 2, 2, 7, 9, 4, 3, 8],
        }
    )


def mixed(lib: Any) -> Any:
    return lib.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "s": ["w", "x", "y", "z"],
            "f": [True, False, True, True],
            "b": [5.0, 3.0, 4.0, 1.0],
        }
    )


BUILDS: dict[str, Any] = {
    f"group_keys {keyed} {name}": (
        lambda lib, keyed=keyed, func=func: stamped(lib).resample("h", group_keys=keyed).apply(func)
    )
    for keyed in (False, True)
    for name, func in (
        ("sum", lambda x: x.sum()),
        ("same rows", lambda x: x * 2),
        ("one row", lambda x: x.head(1)),
    )
}
BUILDS |= {
    f"convention {c}": lambda lib, c=c: stamped(lib).resample("h", convention=c).sum()
    for c in ("start", "end", "s", "e")
}
BUILDS |= {
    "concat apart": lambda lib: (
        lib.concat([stamped(lib).iloc[[0, 1]].head(1), stamped(lib).iloc[[2, 3]].head(1)]).index
    ),
    "concat joined": lambda lib: lib.concat([stamped(lib).iloc[:2], stamped(lib).iloc[2:]]).index,
    "midnight under a short step": lambda lib: stamped(lib).iloc[:1].index,
    "midnight column": lambda lib: stamped(lib).iloc[:1],
    "midnight daily": lambda lib: lib.date_range("2024-01-01", periods=2, freq="D"),
    "midnight 24h": lambda lib: lib.date_range("2024-01-01", periods=2, freq="24h"),
    "filter blanked": lambda lib: (
        gapped(lib).groupby("k").filter(lambda g: len(g) > 4, dropna=False)
    ),
    "filter blanked column": lambda lib: (
        gapped(lib).groupby("k")["n"].filter(lambda g: g.sum() > 20, dropna=False)
    ),
    "filter blanked picked": lambda lib: (
        gapped(lib).groupby("k")[["x", "n"]].filter(lambda g: len(g) > 4, dropna=False)
    ),
    "filter blanked by a series": lambda lib: (
        gapped(lib).groupby(gapped(lib)["n"] % 2).filter(lambda g: len(g) > 5, dropna=False)
    ),
    "filter picked": lambda lib: gapped(lib).groupby("k")[["x"]].filter(lambda g: len(g) > 4),
    "head picked": lambda lib: gapped(lib).groupby("k")[["x"]].head(1),
    "nth any": lambda lib: gapped(lib).groupby("k").nth(0, dropna="any"),
    "nth all": lambda lib: gapped(lib).groupby("k").nth(1, dropna="all"),
    "nth column": lambda lib: gapped(lib).groupby("k")["x"].nth(0, dropna="any"),
    "nth from the back": lambda lib: gapped(lib).groupby("k")["x"].nth(-1, dropna="any"),
    "rolling cov numbers": lambda lib: mixed(lib).rolling(2).cov(numeric_only=True),
    "expanding numbers": lambda lib: mixed(lib).expanding().mean(numeric_only=True),
    "ewm numbers": lambda lib: mixed(lib).ewm(com=1).mean(numeric_only=True),
    "ewm std numbers": lambda lib: mixed(lib).ewm(com=1).std(numeric_only=True),
    "ewm cov numbers": lambda lib: mixed(lib).ewm(com=1).cov(numeric_only=True),
    "quantile numbers": lambda lib: mixed(lib).rolling(2).quantile(0.5, numeric_only=True),
    "grouped rolling numbers": lambda lib: (
        mixed(lib).assign(k=["p", "p", "q", "q"]).groupby("k").rolling(2).sum(numeric_only=True)
    ),
    "copy": lambda lib: lib.DataFrame({"a": [1]}, copy=True),
    "no copy": lambda lib: lib.DataFrame({"a": [1]}, copy=False),
    "series copy": lambda lib: lib.Series([1, 2], copy=True),
}
BUILDS |= {
    f"rolling {how} numbers": lambda lib, how=how: getattr(mixed(lib).rolling(2), how)(
        numeric_only=True
    )
    for how in ("sum", "mean", "first", "last", "nunique", "std", "max", "median", "count")
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_what_was_refused_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


REFUSED = {
    "convention": lambda lib: stamped(lib).resample("h", convention="x"),
    "nth list": lambda lib: gapped(lib).groupby("k").nth([0, 1], dropna="any"),
    "nth how": lambda lib: gapped(lib).groupby("k").nth(0, dropna="some"),
}


@pytest.mark.parametrize("make", REFUSED.values(), ids=REFUSED.keys())
def test_what_pandas_refuses_is_refused_in_its_words(firepanda: Any, make: Any) -> None:
    with pytest.raises(ValueError) as theirs:
        make(pd)
    with pytest.raises(ValueError) as mine:
        make(firepanda)
    assert str(mine.value) == str(theirs.value)
