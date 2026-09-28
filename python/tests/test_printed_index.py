"""What an index prints at a prompt, checked against pandas.

The core writes an index as a summary, and a zoned instant as its count, so the
repr is a port of pandas' `format_object_summary`: the labels in brackets,
wrapped at `display.width` and cut past `display.max_seq_items`, then the type,
the name, the length when cut and, for instants and spans, the frequency.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

ZONED = ["2024-01-02T10:00:00+01:00", None, "2024-07-02T10:00:00.5+01:00"]

SHAPES: dict[str, Callable[[Any], Any]] = {
    "ints": lambda m: m.Index([1, 2, 3]),
    "one": lambda m: m.Index([1.0]),
    "two": lambda m: m.Index([4, 5]),
    "floats with a gap": lambda m: m.Index([1.5, None, 3.25]),
    "odd floats": lambda m: m.Index([0.1 + 0.2, 1e20, -1.5, 1e-7]),
    "named text": lambda m: m.Index(["a", None, "c"], name="k"),
    "escaped text": lambda m: m.Index(["tab\there", "new\nline", "it's"]),
    "flags": lambda m: m.Index([True, False, True]),
    "thirty": lambda m: m.Index(list(range(30)), name="x"),
    "long ints": lambda m: m.Index(list(range(150))),
    "long floats": lambda m: m.Index([i * 1.25 for i in range(120)]),
    "long labels": lambda m: m.Index([f"label_number_{i}" for i in range(40)]),
    "many labels": lambda m: m.Index([f"x{i}" for i in range(101)]),
    "dates": lambda m: m.DatetimeIndex(["2024-01-02", None, "2024-01-04"]),
    "dates and times": lambda m: m.DatetimeIndex(["2024-01-02 10:00:00.123", "2024-01-03"]),
    "named dates": lambda m: m.DatetimeIndex(["2024-01-02"] * 12, name="when"),
    "zoned": lambda m: m.DatetimeIndex(m.to_datetime(ZONED, format="mixed")),
    "utc": lambda m: m.DatetimeIndex(m.to_datetime(["2024-01-02", "2024-01-03"], utc=True)),
    "days": lambda m: m.TimedeltaIndex(["-1 days", "2 days", None]),
    "spans": lambda m: m.TimedeltaIndex(["1 days 00:00:00.5", "-2 hours"]),
    "frame index": lambda m: (
        m.DataFrame({"t": m.to_datetime(["2024-01-02", "2024-01-03"]), "v": [1, 2]})
        .set_index("t")
        .index
    ),
}


@pytest.mark.parametrize("shape", SHAPES)
def test_repr_and_str_print_what_pandas_prints(firepanda: ModuleType, shape: str) -> None:
    """The whole text, under pandas' default options."""
    import pandas as pd

    ours, theirs = SHAPES[shape](firepanda), SHAPES[shape](pd)
    assert repr(ours) == repr(theirs)
    assert str(ours) == str(theirs)


@pytest.mark.parametrize(
    "options",
    [
        {"display.max_seq_items": 6, "display.width": 40},
        {"display.max_seq_items": 1},
        {"display.max_seq_items": None},
        {"display.width": 30},
    ],
)
@pytest.mark.parametrize("shape", ["thirty", "long ints", "long labels", "dates"])
def test_the_display_options_are_read_as_pandas_reads_them(
    firepanda: ModuleType, shape: str, options: dict[str, Any]
) -> None:
    """Each option changes the text the same way in both."""
    import pandas as pd

    pairs = [item for pair in options.items() for item in pair]
    with firepanda.option_context(*pairs):
        ours = repr(SHAPES[shape](firepanda))
    with pd.option_context(*pairs):
        theirs = repr(SHAPES[shape](pd))
    assert ours == theirs
