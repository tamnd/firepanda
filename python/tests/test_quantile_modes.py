"""The four picking interpolations and a list of quantiles, checked against pandas.

`lower`, `higher`, `midpoint` and `nearest` land on a value in the column or
halfway between two of them, so they have to match pandas exactly, including
`nearest` rounding a half to the even position the way `numpy.around` does. A
list of quantiles answers a Series labelled by the quantiles on a Series and a
frame with a row per quantile on a frame. `linear` is compared with a tolerance,
because the kernel weights the two values with one formula and numpy with two
and they can part in the last bit.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

PICKED = ["lower", "higher", "midpoint", "nearest"]
COLUMNS = [
    [3.0, 1.0, None, 7.5, -2.0, 4.0],
    [1.0, 2.0],
    [5.0],
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [None, None],
    [],
]
QS = [0.0, 0.1, 0.125, 0.25, 1 / 3, 0.5, 2 / 3, 0.75, 0.875, 1.0]


def same(got: list[Any], want: list[Any]) -> bool:
    """Equal row by row, with NaN equal to NaN."""
    return len(got) == len(want) and all(
        a == b or (a != a and b != b) for a, b in zip(got, want, strict=True)
    )


@pytest.mark.parametrize("how", PICKED)
@pytest.mark.parametrize("rows", COLUMNS)
@pytest.mark.parametrize("q", QS)
def test_a_picked_quantile_is_exactly_pandas(
    firepanda: ModuleType, how: str, rows: list[Any], q: float
) -> None:
    """One quantile, every rule that picks, every length from nothing to six."""
    import pandas as pd

    got = firepanda.Series(rows, dtype="float64").quantile(q, interpolation=how)
    want = pd.Series(rows, dtype="float64").quantile(q, interpolation=how)
    assert same([float(got)], [float(want)])


@pytest.mark.parametrize("how", PICKED)
@pytest.mark.parametrize("rows", COLUMNS)
def test_a_list_of_picked_quantiles_is_a_series(
    firepanda: ModuleType, how: str, rows: list[Any]
) -> None:
    """Labelled by the quantiles, and named for the column."""
    import pandas as pd

    got = firepanda.Series(rows, dtype="float64", name="x").quantile(QS, interpolation=how)
    want = pd.Series(rows, dtype="float64", name="x").quantile(QS, interpolation=how)
    assert same(got.tolist(), want.tolist())
    assert got.index.tolist() == want.index.tolist()
    assert got.name == want.name


def test_a_list_of_linear_quantiles_is_close(firepanda: ModuleType) -> None:
    """The kernel answers each one, so this is the shape more than the values."""
    import pandas as pd

    rows = COLUMNS[0]
    got = firepanda.Series(rows).quantile(QS)
    want = pd.Series(rows).quantile(QS)
    assert got.tolist() == pytest.approx(want.tolist())
    assert got.index.tolist() == want.index.tolist()


def test_nearest_rounds_a_half_to_the_even_position(firepanda: ModuleType) -> None:
    """A position of one and a half goes to two and one of two and a half to two."""
    import pandas as pd

    for rows in ([10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]):
        got = firepanda.Series(rows).quantile(0.5, interpolation="nearest")
        assert got == pd.Series(rows).quantile(0.5, interpolation="nearest")


def test_integers_pick_an_integer_and_average_to_a_float(firepanda: ModuleType) -> None:
    """What pandas answers for an int64 column under each rule."""
    import pandas as pd

    rows = [4, 1, 3, 2]
    for how in PICKED:
        got = firepanda.Series(rows).quantile(0.5, interpolation=how)
        want = pd.Series(rows).quantile(0.5, interpolation=how)
        assert float(got) == float(want)


@pytest.mark.parametrize("how", ["linear", *PICKED])
def test_a_frame_answers_a_row_per_quantile(firepanda: ModuleType, how: str) -> None:
    """With the quantiles as the index and a column per numeric column."""
    import pandas as pd

    data = {"a": [1.0, 4.0, None, 2.0], "b": [7, 5, 6, 8]}
    got = firepanda.DataFrame(data).quantile([0.25, 0.5], interpolation=how)
    want = pd.DataFrame(data).quantile([0.25, 0.5], interpolation=how)
    assert list(got.columns) == list(want.columns)
    assert got.index.tolist() == want.index.tolist()
    for name in want.columns:
        assert got[name].tolist() == pytest.approx(want[name].tolist())


@pytest.mark.parametrize("how", PICKED)
def test_a_frame_answers_one_quantile_per_column(firepanda: ModuleType, how: str) -> None:
    """One quantile under a picking rule is a Series labelled by the columns."""
    import pandas as pd

    data = {"a": [1.0, 4.0, None, 2.0], "b": [7, 5, 6, 8], "s": ["x", "y", "z", "w"]}
    got = firepanda.DataFrame(data).quantile(0.5, interpolation=how, numeric_only=True)
    want = pd.DataFrame(data).quantile(0.5, interpolation=how, numeric_only=True)
    assert got.index.tolist() == want.index.tolist()
    assert same(got.tolist(), want.tolist())


def test_a_quantile_column_keeps_its_name_beside_the_index(firepanda: ModuleType) -> None:
    """A column called quantile does not collide with the key the rows are built on."""
    import pandas as pd

    data = {"quantile": [1.0, 2.0, 3.0]}
    got = firepanda.DataFrame(data).quantile([0.5])
    want = pd.DataFrame(data).quantile([0.5])
    assert list(got.columns) == list(want.columns)
    assert got["quantile"].tolist() == want["quantile"].tolist()


@pytest.mark.parametrize("q", [[0.5, 1.5], 1.5, -0.1])
def test_a_quantile_outside_the_interval_says_so(firepanda: ModuleType, q: Any) -> None:
    """In pandas' words, which are the same for one quantile and a list of them."""
    import pandas as pd

    with pytest.raises(ValueError) as theirs:
        pd.Series([1.0]).quantile(q, interpolation="lower")
    with pytest.raises(ValueError) as mine:
        firepanda.Series([1.0]).quantile(q, interpolation="lower")
    assert str(mine.value) == str(theirs.value)


def test_the_eight_newer_rules_are_refused(firepanda: ModuleType) -> None:
    """They are pandas rules firepanda has not written, not typos."""
    with pytest.raises(NotImplementedError, match="interpolation='weibull'"):
        firepanda.Series([1.0, 2.0]).quantile([0.5], interpolation="weibull")


def test_a_frame_along_its_rows_is_refused(firepanda: ModuleType) -> None:
    """Only down the columns so far."""
    with pytest.raises(NotImplementedError):
        firepanda.DataFrame({"a": [1.0], "b": [2.0]}).quantile([0.5], axis=1)


def test_a_grouped_quantile_outside_the_interval_says_what_groupby_says(
    firepanda: ModuleType,
) -> None:
    """groupby has a sentence of its own for the same mistake."""
    import pandas as pd

    data = {"k": [0, 0], "a": [1.0, 2.0]}
    with pytest.raises(ValueError) as theirs:
        pd.DataFrame(data).groupby("k").quantile(1.5)
    with pytest.raises(ValueError) as mine:
        firepanda.DataFrame(data).groupby("k").quantile(1.5)
    assert str(mine.value) == str(theirs.value)
