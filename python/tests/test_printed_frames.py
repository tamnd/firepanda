"""What a frame or a column prints at a prompt, checked against pandas.

The core writes a frame as a summary of its columns and an instant with a zone
as its count, so both reprs go through the text formatter, under the display
options pandas reads: `max_rows` and `min_rows` for the rows, `max_columns`
and `width` for the columns, and `show_dimensions` for the shape at the end.
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
    "text": lambda m: m.DataFrame({"t": ["a", None], "x": [1, 2]}),
    "floats": lambda m: m.DataFrame({"f": [1.5, None], "g": [1e10, 2.0]}),
    "named index": lambda m: m.DataFrame({"a": [1, 2]}, index=m.Index(["x", "y"], name="k")),
    "dates": lambda m: m.DataFrame({"t": m.to_datetime(["2024-01-02", None])}),
    "zoned": lambda m: m.DataFrame({"t": m.to_datetime(ZONED, format="mixed")}),
    "zoned column": lambda m: m.Series(m.to_datetime(ZONED, format="mixed"), name="t"),
    "zoned midnight": lambda m: m.Series(m.to_datetime(["2024-01-02"] * 2)).dt.tz_localize("UTC"),
    "named zone": lambda m: m.Series(m.to_datetime(["2024-01-02 10:00"])).dt.tz_localize(
        "Europe/Paris"
    ),
    "long frame": lambda m: m.DataFrame({"x": range(70), "y": [i / 7 for i in range(70)]}),
    "long column": lambda m: m.Series(range(100), name="n"),
    "sixty": lambda m: m.Series(range(60)),
    "sixty one": lambda m: m.Series(range(61)),
    "many columns": lambda m: m.DataFrame({f"c{i}": [i, i * 1000] for i in range(30)}),
    "wide columns": lambda m: m.DataFrame({f"column_{i}": [i * 1.5, 2.25] for i in range(12)}),
    "long and wide": lambda m: m.DataFrame({f"c{i}": range(100) for i in range(25)}),
    "empty": lambda m: m.DataFrame({"a": []}),
    "empty column": lambda m: m.Series([], dtype="float64"),
    "mixed": lambda m: m.DataFrame(
        {"a": [1, 2], "b": ["x", "y"], "c": [0.5, 1.25], "d": m.to_datetime(["2024-01-01"] * 2)}
    ),
    "long text": lambda m: m.DataFrame({"t": ["x" * 80, "y"]}),
    "spans": lambda m: m.Series(m.to_timedelta(["1 days", "2 hours"])),
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
        {"display.max_rows": 10, "display.min_rows": 4},
        {"display.max_rows": 10, "display.min_rows": 0},
        {"display.max_columns": 4},
        {"display.show_dimensions": True},
        {"display.show_dimensions": False},
        {"display.width": 40},
        {"display.expand_frame_repr": False},
        {"display.max_colwidth": 5},
    ],
)
@pytest.mark.parametrize("shape", ["long frame", "long column", "many columns", "long text"])
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
