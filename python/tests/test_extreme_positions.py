"""`idxmax`, `idxmin`, `argmax`, `argmin` and `between` on a series, against pandas.

The four position methods find the extreme with the reduction and the first row
equal to it with a filter, so a tie answers the first row, as numpy's does. The
three errors are pandas' sentences, and an empty series is the empty sequence
error whatever `skipna` says, because pandas checks that first.
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

METHODS = ["idxmax", "idxmin", "argmax", "argmin"]


def labelled(m: Any, values: list[Any], labels: list[str]) -> Any:
    """A series with string labels, built the one way both libraries take."""
    frame = m.DataFrame({"k": labels, "v": values}).set_index("k")
    return frame["v"].rename(None).rename_axis(None)


ANSWERED: list[Callable[[Any], Any]] = [
    lambda m: m.Series([1.0, float("nan"), 3.0, 3.0]),
    lambda m: m.Series(["b", "a", "c"]),
    lambda m: m.Series([True, False]),
    lambda m: labelled(m, [1, 5, 5], ["x", "y", "z"]),
    lambda m: m.Series([2, 1, 1, 7]).iloc[1:],
]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("build", ANSWERED)
def test_the_first_extreme(firepanda: ModuleType, method: str, build: Any) -> None:
    """The label or the position of the first row holding the extreme."""
    import pandas as pd

    assert getattr(build(firepanda), method)() == getattr(build(pd), method)()


REFUSED: list[tuple[Callable[[Any], Any], bool]] = [
    (lambda m: m.Series([1.0, float("nan"), 3.0]), False),
    (lambda m: m.Series([float("nan")] * 2), True),
    (lambda m: m.Series([float("nan")] * 2), False),
    (lambda m: m.Series([], dtype="float64"), True),
    (lambda m: m.Series([], dtype="float64"), False),
]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(("build", "skipna"), REFUSED)
def test_the_three_errors(firepanda: ModuleType, method: str, build: Any, skipna: bool) -> None:
    """A gap under `skipna=False`, nothing but gaps, and nothing at all."""
    import pandas as pd

    with pytest.raises(ValueError) as theirs:
        getattr(build(pd), method)(skipna=skipna)
    with pytest.raises(ValueError) as mine:
        getattr(build(firepanda), method)(skipna=skipna)
    assert str(mine.value) == str(theirs.value)


@pytest.mark.parametrize("inclusive", ["both", "neither", "left", "right"])
def test_between(firepanda: ModuleType, inclusive: str) -> None:
    """Each way of closing the ends, with a missing value False and the name kept."""
    import pandas as pd

    rows = [1.0, None, 3.0, 5.0, 2.0]
    got = firepanda.Series(rows, name="n").between(1, 3, inclusive=inclusive)
    want = pd.Series(rows, name="n").between(1, 3, inclusive=inclusive)
    assert got.tolist() == want.tolist()
    assert got.name == want.name
    assert got.dtype == str(want.dtype)


def test_between_strings_and_bounds_the_wrong_way(firepanda: ModuleType) -> None:
    """Text compares in order, and a low bound above the high one is all False."""
    import pandas as pd

    for m in (firepanda, pd):
        assert m.Series(["a", "b", "c"]).between("a", "b").tolist() == [True, True, False]
        assert m.Series([1, 2, 3]).between(3, 1).tolist() == [False, False, False]


def test_between_refuses_a_word_it_does_not_know(firepanda: ModuleType) -> None:
    """In pandas' words."""
    import pandas as pd

    with pytest.raises(ValueError) as theirs:
        pd.Series([1]).between(0, 2, inclusive="up")
    with pytest.raises(ValueError) as mine:
        firepanda.Series([1]).between(0, 2, inclusive="up")
    assert str(mine.value) == str(theirs.value)
