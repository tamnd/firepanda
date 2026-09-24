"""`to_numeric`, checked against pandas.

pandas reads whole numbers in text as int64 and anything with a decimal point
or an exponent as float64, lets space surround a number but not an infinity,
and reads empty text as NaN. A column answers a column, an index an index, a
list a numpy array and one value one number, and `downcast` walks from the
smallest type of a kind up to the first that holds every value.
"""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

TEXTS = [
    "1",
    " 2 ",
    "1.5",
    "1e3",
    "inf",
    "-Infinity",
    "",
    "+3",
    "-4",
    "1.",
    ".5",
    "1.5 ",
    "\t7\n",
    "+inf",
    "INFINITY",
    "-0",
    "00012",
    "1E-2",
]
NOT_NUMBERS = ["nan", "1_000", "0x10", "abc", " ", "- 1", "1e", " inf", "True"]


def shown(answer: Any) -> Any:
    """An answer as plain Python, with its type, name and labels."""
    if isinstance(answer, np.ndarray):
        return ("array", plain(answer.tolist()), str(answer.dtype))
    if hasattr(answer, "iloc"):
        return (
            "column",
            plain(answer.tolist()),
            str(answer.dtype),
            answer.name,
            list(answer.index),
        )
    if hasattr(answer, "dtype") and hasattr(answer, "tolist") and hasattr(answer, "name"):
        return ("index", plain(answer.tolist()), str(answer.dtype), answer.name)
    return ("value", type(answer).__name__, "missing" if answer != answer else answer)


def plain(values: list[Any]) -> list[Any]:
    """The values with every spelling of missing as one word."""
    return ["missing" if value is None or value != value else value for value in values]


def downcasts() -> list[Callable[[Any], Any]]:
    """Every kind of downcast over values that fit some types and not others."""
    builds: list[Callable[[Any], Any]] = []
    for values in ([0.1, 0.2], [1e300], [1.5, None], [3.0, None], [1, 300], [-1, 2], [-1, 70000]):
        dtype = "float64" if any(not isinstance(one, int) for one in values) else "int64"
        for kind in ("integer", "signed", "unsigned", "float"):
            builds.append(
                lambda m, values=values, dtype=dtype, kind=kind: m.to_numeric(
                    m.Series(values, dtype=dtype), downcast=kind
                )
            )
    return builds


BUILDS: list[Callable[[Any], Any]] = [
    *[lambda m, text=text: m.to_numeric(m.Series([text])) for text in TEXTS],
    *[
        lambda m, text=text: m.to_numeric(m.Series([text, "x"]), errors="coerce")
        for text in TEXTS[:6]
    ],
    lambda m: m.to_numeric(m.Series(["1", "2", None], name="n", index=["a", "b", "c"])),
    lambda m: m.to_numeric(m.Series(["1", "x"]), errors="coerce"),
    lambda m: m.to_numeric("5"),
    lambda m: m.to_numeric("5.5"),
    lambda m: m.to_numeric(5),
    lambda m: m.to_numeric(None),
    lambda m: m.to_numeric(True),
    lambda m: m.to_numeric("x", errors="coerce"),
    lambda m: m.to_numeric(["1", "2"]),
    lambda m: m.to_numeric([1, "2", 3.5]),
    lambda m: m.to_numeric(np.array(["1", "2"])),
    lambda m: m.to_numeric(["9223372036854775808"]),
    lambda m: m.to_numeric(m.Index(["1", "2"], name="k")),
    lambda m: m.to_numeric(m.Series([True, False])),
    lambda m: m.to_numeric(m.Series([1.5, 2.5], name="f")),
    lambda m: m.to_numeric(m.to_datetime(m.Series(["2024-01-01"]))),
    lambda m: m.to_numeric(m.Series(["1", "2"]), downcast="integer"),
    lambda m: m.to_numeric(m.Series(["1", "x"]), errors="coerce", downcast="integer"),
    lambda m: m.to_numeric(m.Series([True]), downcast="integer"),
    lambda m: m.to_numeric(m.Series([], dtype="float64"), downcast="unsigned"),
    lambda m: m.to_numeric("7", downcast="unsigned"),
    *downcasts(),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Text in every spelling, each kind of argument, and every downcast."""
    import pandas as pd

    assert shown(build(firepanda)) == shown(build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    *[lambda m, text=text: m.to_numeric(m.Series(["1", text])) for text in NOT_NUMBERS],
    lambda m: m.to_numeric("x"),
    lambda m: m.to_numeric(m.Series(["1"]), errors="ignore"),
    lambda m: m.to_numeric(m.Series(["1"]), downcast="x"),
    lambda m: m.to_numeric(m.Series(["1"]), dtype_backend="x"),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The same class or a subclass of it, and the same message."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert isinstance(mine.value, type(theirs.value))
    assert str(mine.value) == str(theirs.value)


@pytest.mark.parametrize(
    "build",
    [
        lambda m: m.to_numeric(m.Series(["1"]), dtype_backend="pyarrow"),
        lambda m: m.to_numeric(m.Series(["9223372036854775808"])),
        lambda m: m.to_numeric(m.Series(["18446744073709551616"])),
    ],
)
def test_what_firepanda_cannot_hold_is_refused(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """pandas' nullable types, and whole numbers past int64 in a column."""
    with pytest.raises(NotImplementedError, match="to_numeric"):
        build(firepanda)


def test_the_signature_is_pandas_signature(firepanda: ModuleType) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(firepanda.to_numeric).parameters
    yours = inspect.signature(pd.to_numeric).parameters
    assert [(p.name, p.kind) for p in ours.values()] == [(p.name, p.kind) for p in yours.values()]
    assert [p.default for p in ours.values()][:3] == [p.default for p in yours.values()][:3]
