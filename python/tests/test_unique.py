"""`unique` and `factorize` on a column, an index and an array, checked against pandas.

pandas answers `Series.unique` with an array and `factorize` with an array of
codes and an index of uniques, so every test here compares the values of each
part and, for an index, its name, and leaves the storage classes out, since
pandas has a different one for text, numbers and categories and firepanda has
one for all of them.
"""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

NAN = float("nan")
TEXT = ["b", "a", None, "b", "c", None, "a"]
NUMBERS = [2.0, NAN, 1.0, NAN, 2.0, 7.5]
WHOLE = [5, 3, 5, 9, 3, 3]


def plain(values: list[Any]) -> list[Any]:
    """The values with every spelling of missing as one word."""
    return ["missing" if value is None or value != value else value for value in values]


def agrees(got: Any, want: Any) -> None:
    """The same values in the same order, and the same name for an index."""
    if isinstance(want, tuple):
        assert isinstance(got, tuple)
        assert len(got) == len(want)
        for mine, theirs in zip(got, want, strict=True):
            agrees(mine, theirs)
        return
    assert plain(list(got)) == plain(want.tolist())
    assert len(got) == len(want)
    if hasattr(want, "names"):
        assert hasattr(got, "names")
        assert got.name == want.name


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.Series(TEXT, name="k").unique(),
    lambda m: m.Series(NUMBERS).unique(),
    lambda m: m.Series(WHOLE).unique(),
    lambda m: m.Series([True, False, True]).unique(),
    lambda m: m.Series([], dtype="float64").unique(),
    lambda m: m.Series(TEXT).astype("category").unique(),
    lambda m: m.unique(m.Series(TEXT)),
    lambda m: m.unique(m.Index(WHOLE, name="i")),
    lambda m: m.Series(TEXT, name="k").factorize(),
    lambda m: m.Series(TEXT).factorize(sort=True),
    lambda m: m.Series(TEXT).factorize(use_na_sentinel=False),
    lambda m: m.Series(TEXT).factorize(sort=True, use_na_sentinel=False),
    lambda m: m.Series(NUMBERS).factorize(),
    lambda m: m.Series(NUMBERS).factorize(sort=True),
    lambda m: m.Series(NUMBERS).factorize(use_na_sentinel=False),
    lambda m: m.Series(WHOLE).factorize(sort=True),
    lambda m: m.Series([None, None], dtype="string").factorize(),
    lambda m: m.factorize(m.Series(TEXT, name="k")),
    lambda m: m.factorize(m.Series(WHOLE), sort=True, size_hint=10),
    lambda m: m.factorize(m.Index(["q", "p", "q"], name="i")),
    lambda m: m.Index(["q", "p", "q"], name="i").factorize(sort=True),
    lambda m: m.factorize(m.Series(TEXT).unique()),
    lambda m: m.factorize(m.Series(TEXT).array, sort=True),
    lambda m: m.Series(TEXT).array,
    lambda m: m.Series(TEXT).unique()[1:],
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Text, floats with NaN, whole numbers, flags, categories, an index and an array."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_the_codes_are_whole_numbers(firepanda: ModuleType) -> None:
    """int64 whether or not a value was missing, as numpy's intp is on a 64 bit machine."""
    for values in (TEXT, NUMBERS, WHOLE):
        codes, _ = firepanda.Series(values).factorize()
        assert codes.dtype == "int64"


def test_a_categorical_unique_keeps_every_category(firepanda: ModuleType) -> None:
    """Crossing to Arrow as a dictionary that still holds the unused category."""
    import pyarrow as pa

    column = firepanda.Series(["b", "a", "z", "b"]).astype("category")
    kept = pa.array(column[column != "z"].unique())
    assert pa.types.is_dictionary(kept.type)
    assert kept.dictionary.to_pylist() == ["a", "b", "z"]
    assert kept.to_pylist() == ["b", "a"]


def test_an_array_reads_like_one(firepanda: ModuleType) -> None:
    """Length, shape, a position, a slice, membership and the Arrow export."""
    import pyarrow as pa

    values = firepanda.Series(TEXT, name="k").unique()
    assert len(values) == values.size == 4
    assert values.shape == (4,)
    assert values.ndim == 1
    assert values[0] == "b"
    assert values[1:].tolist() == ["a", None, "c"]
    assert "c" in values
    assert pa.array(values).to_pylist() == ["b", "a", None, "c"]
    assert values.isna().tolist() == [False, False, True, False]


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.unique(["b", "a"]),
    lambda m: m.factorize(("b", "a")),
    lambda m: m.unique("text"),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The same class and the same message."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert isinstance(mine.value, type(theirs.value))
    assert str(mine.value) == str(theirs.value)


def test_a_categorical_factorize_is_refused(firepanda: ModuleType) -> None:
    """Its uniques are a categorical index, which is not held yet."""
    with pytest.raises(NotImplementedError):
        firepanda.Series(["b", "a"]).astype("category").factorize()


@pytest.mark.parametrize(
    ("owner", "name"),
    [("Series", "unique"), ("Series", "factorize"), ("Index", "factorize")],
)
def test_the_method_signature_is_pandas_signature(
    firepanda: ModuleType, owner: str, name: str
) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(getattr(getattr(firepanda, owner), name)).parameters
    yours = inspect.signature(getattr(getattr(pd, owner), name)).parameters
    assert [(p.name, p.kind, p.default) for p in ours.values()] == [
        (p.name, p.kind, p.default) for p in yours.values()
    ]


@pytest.mark.parametrize("name", ["unique", "factorize"])
def test_the_function_signature_is_pandas_signature(firepanda: ModuleType, name: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(getattr(firepanda, name)).parameters
    yours = inspect.signature(getattr(pd, name)).parameters
    assert [(p.name, p.kind, p.default) for p in ours.values()] == [
        (p.name, p.kind, p.default) for p in yours.values()
    ]
