"""`json_normalize`, checked against pandas.

The function is a port of pandas' own, working on dicts and lists before a frame
exists, so every input that ends in columns of numbers or text gives the same
frame. A record whose cell is a dict or a list needs a column of objects, which
firepanda does not have, so a `max_level` short of the depth is left out.
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

STATES = [
    {
        "state": "Florida",
        "shortname": "FL",
        "info": {"governor": "Rick Scott"},
        "counties": [
            {"name": "Dade", "population": 12345},
            {"name": "Broward", "population": 40000},
        ],
    },
    {
        "state": "Ohio",
        "shortname": "OH",
        "info": {"governor": "John Kasich"},
        "counties": [{"name": "Summit", "population": 1234}],
    },
]
PEOPLE = [
    {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
    {"name": {"given": "Mark", "family": "Regner"}},
    {"id": 2, "name": "Faye Raker"},
]
DEEP = [{"a": 1, "b": {"c": 2.5, "d": {"e": "x", "f": {"g": 3}}}}, {"a": 2}]

CALLS: dict[str, Callable[[Any], Any]] = {
    "nested": lambda jn: jn(PEOPLE),
    "one level": lambda jn: jn(PEOPLE, max_level=1),
    "deep enough": lambda jn: jn(DEEP, max_level=3),
    "every level": lambda jn: jn(DEEP),
    "separator": lambda jn: jn(PEOPLE, sep="_"),
    "record prefix": lambda jn: jn(PEOPLE, record_prefix="r."),
    "records and meta": lambda jn: jn(
        STATES, "counties", ["state", "shortname", ["info", "governor"]]
    ),
    "prefixes": lambda jn: jn(STATES, "counties", ["state"], record_prefix="c.", meta_prefix="m."),
    "meta by path": lambda jn: jn(STATES, ["counties"], [["info", "governor"]]),
    "scalar records": lambda jn: jn({"A": [1, 2]}, "A", record_prefix="Prefix."),
    "one dict": lambda jn: jn({"a": 1, "b": {"c": 2.5, "d": {"e": "x"}}}),
    "nothing": lambda jn: jn([]),
    "a gap": lambda jn: jn([{"a": 1}, None, {"a": 3}]),
    "a generator": lambda jn: jn(row for row in [{"a": 1}, {"a": 2, "b": "x"}]),
}

MISTAKES: dict[str, Callable[[Any], Any]] = {
    "missing meta": lambda jn: jn(STATES, "counties", ["missing"]),
    "missing path": lambda jn: jn(STATES, "nothing"),
    "meta clashes": lambda jn: jn(STATES, "counties", ["name"]),
    "not dicts": lambda jn: jn([1, 2]),
    "meta not text": lambda jn: jn(STATES, "counties", [1]),
    "path to a value": lambda jn: jn(STATES, "state"),
}


@pytest.mark.parametrize("call", CALLS)
def test_the_frame_is_the_one_pandas_builds(firepanda: ModuleType, call: str) -> None:
    """The same columns, in the same order, printing the same values."""
    import pandas as pd

    ours, theirs = CALLS[call](firepanda.json_normalize), CALLS[call](pd.json_normalize)
    assert list(ours.columns) == list(theirs.columns)
    assert repr(ours) == repr(theirs)


@pytest.mark.parametrize("call", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, call: str) -> None:
    """The same class of error, with the same words."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        MISTAKES[call](pd.json_normalize)
    with pytest.raises(type(theirs.value)) as ours:
        MISTAKES[call](firepanda.json_normalize)
    assert str(ours.value) == str(theirs.value)
