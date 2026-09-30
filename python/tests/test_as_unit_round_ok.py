"""`as_unit(round_ok=False)`, which refuses a cast that would drop a fraction.

pandas names the first instant a coarser unit cannot hold exactly by its whole
count in the unit it is held in, and a finer unit, or instants with nothing to
drop, cast as usual. Spans do not take the argument at all. Each test runs the
same code on both libraries and compares the answer or the error.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def stamps(lib: Any, texts: list[Any]) -> Any:
    return lib.Series(lib.to_datetime(texts, format="ISO8601"))


def spans(lib: Any) -> Any:
    return lib.Series(lib.to_timedelta(["-1.5s", "2s", None]))


EXACT = ["2024-01-01 00:00:01", None]
ROUNDED = ["2024-01-01 00:00:01", "2024-01-01 00:00:02.25", "1969-12-31 23:59:59.5"]

CALLS = {
    "exact": lambda lib: stamps(lib, EXACT).dt.as_unit("s", round_ok=False),
    "finer": lambda lib: stamps(lib, ROUNDED).dt.as_unit("ns", round_ok=False),
    "rounded": lambda lib: stamps(lib, ROUNDED).dt.as_unit("s", round_ok=False),
    "before 1970": lambda lib: stamps(lib, ["1969-12-31 23:59:59.999"]).dt.as_unit(
        "s", round_ok=False
    ),
    "nanoseconds": lambda lib: stamps(lib, ["2024-01-01 00:00:01.123456789"]).dt.as_unit(
        "us", round_ok=False
    ),
    "zoned": lambda lib: (
        stamps(lib, ["2024-01-01 00:00:01.5"]).dt.tz_localize("UTC").dt.as_unit("s", round_ok=False)
    ),
    "index exact": lambda lib: lib.DatetimeIndex(stamps(lib, EXACT)).as_unit("s", round_ok=False),
    "index rounded": lambda lib: lib.DatetimeIndex(stamps(lib, ROUNDED)).as_unit(
        "s", round_ok=False
    ),
    "spans": lambda lib: spans(lib).dt.as_unit("s", round_ok=False),
    "span index": lambda lib: lib.TimedeltaIndex(spans(lib)).as_unit("s", round_ok=False),
}


def outcome(make: Any, lib: Any) -> Any:
    try:
        return repr(make(lib))
    except (ValueError, TypeError) as error:
        return (isinstance(error, ValueError), str(error))


@pytest.mark.parametrize("make", CALLS.values(), ids=CALLS.keys())
def test_round_ok_is_pandas(firepanda: Any, make: Any) -> None:
    assert outcome(make, firepanda) == outcome(make, pd)
