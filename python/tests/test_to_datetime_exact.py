"""`to_datetime(exact=False)`, which lets the format match part of the text.

pandas searches each row for the first place the format matches and passes
over whatever is either side of it, and a row the format matches nowhere is an
error, or missing under `errors="coerce"`. The unit is microseconds, except
where pandas' ISO 8601 reader takes a row past the end of an ISO format and
holds the column at nanoseconds. Each test runs the same code on both libraries
and compares the answer or the error.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

CALLS = {
    "surrounded": lambda lib: lib.to_datetime(["on 2020-01-15 ok"], format="%Y-%m-%d", exact=False),
    "day first": lambda lib: lib.to_datetime(
        ["x 15/01/2020 y", "02/03/2021!"], format="%d/%m/%Y", exact=False
    ),
    "gap": lambda lib: lib.to_datetime(
        ["15/01/2020 and more", None], format="%d/%m/%Y", exact=False
    ),
    "coerced": lambda lib: lib.to_datetime(
        ["x 15/01/2020", "bad"], format="%d/%m/%Y", exact=False, errors="coerce"
    ),
    "all missing": lambda lib: lib.to_datetime(
        ["x", None], format="%Y-%m-%d", exact=False, errors="coerce"
    ),
    "time after a day": lambda lib: lib.to_datetime(
        ["2020-01-15 10:00"], format="%Y-%m-%d", exact=False
    ),
    "day after a month": lambda lib: lib.to_datetime(
        ["2020-01-15 10:00"], format="%Y-%m", exact=False
    ),
    "slashes": lambda lib: lib.to_datetime(["2020/01/15 99"], format="%Y/%m/%d", exact=False),
    "word after a month": lambda lib: lib.to_datetime(["2020-01 x"], format="%Y-%m", exact=False),
    "dash after a day": lambda lib: lib.to_datetime(
        ["2020-01-15-3"], format="%Y-%m-%d", exact=False
    ),
    "minutes after an hour": lambda lib: lib.to_datetime(
        ["2020-01-15 10:30"], format="%Y-%m-%d %H", exact=False
    ),
    "column": lambda lib: lib.to_datetime(
        lib.Series(["a 2020-01-15", "b 2021-02-16"]), format="%Y-%m-%d", exact=False
    ),
    "exact": lambda lib: lib.to_datetime(["2020-01-15"], format="%Y-%m-%d", exact=True),
    "no format": lambda lib: lib.to_datetime(["2020-01-15"], exact=False),
}

MISTAKES = {
    "short": lambda lib: lib.to_datetime(["2020-01"], format="%Y-%m-%d", exact=False),
    "nowhere": lambda lib: lib.to_datetime(["abc"], format="%d/%m/%Y", exact=False),
    "longer format": lambda lib: lib.to_datetime(
        ["2020-01-15 10:00"], format="%Y-%m-%d %H:%M:%S", exact=False
    ),
}


@pytest.mark.parametrize("make", CALLS.values(), ids=CALLS.keys())
def test_exact_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


@pytest.mark.parametrize("make", MISTAKES.values(), ids=MISTAKES.keys())
def test_what_pandas_refuses_is_refused(firepanda: Any, make: Any) -> None:
    with pytest.raises(ValueError) as theirs:
        make(pd)
    with pytest.raises(ValueError) as mine:
        make(firepanda)
    assert str(mine.value) == str(theirs.value)
