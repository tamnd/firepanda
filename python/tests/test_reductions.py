"""The twelve reductions, checked against a running pandas rather than a table.

`s.sum()` is the most written reduction in pandas and until now the Python layer
had none of them at all, while the core had seventeen. So these tests are less
about whether a total is right, which `tests/test_agg.mojo` already measures in
Mojo over more dtypes than are reachable from here, than about whether the
answer that comes out of the boundary is the answer pandas gives: the same
number, the same shape, and the same thing when there is no answer.

Comparing against a live pandas rather than against constants is deliberate. A
constant records what somebody believed pandas does on the day they wrote it
down, and the whole claim of this project is about what pandas actually does.

The refusals are tested as carefully as the answers. Every one of the arguments
that is declared and not implemented raises rather than being ignored, because a
declared parameter that is quietly dropped is the failure that takes longest to
find, and the test that a refusal still refuses is what stops one from being
dropped by accident later.
"""

from __future__ import annotations

import importlib.util
import math
from types import ModuleType

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

VALUES = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
"""One column both libraries get, with no missing value in it.

Floats rather than integers, because an integer column with a missing value is
where the two libraries disagree about the dtype and that argument belongs in
the astype and missing value documents rather than in here.
"""

PLAIN = ["sum", "mean", "min", "max", "median", "skew", "std", "var", "sem", "nunique"]
"""The ten that take no argument worth varying."""


@needs_pandas
@pytest.mark.parametrize("name", PLAIN)
def test_a_series_reduction_gives_the_pandas_answer(firepanda: ModuleType, name: str) -> None:
    """The whole point, once per reduction."""
    import pandas as pd

    mine = getattr(firepanda.Series(VALUES), name)()
    theirs = getattr(pd.Series(VALUES), name)()
    assert mine == pytest.approx(theirs)


@needs_pandas
@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.9, 1.0])
def test_the_quantile_lands_where_pandas_lands(firepanda: ModuleType, q: float) -> None:
    """Including the two ends, where there is nothing to interpolate between."""
    import pandas as pd

    assert firepanda.Series(VALUES).quantile(q) == pytest.approx(pd.Series(VALUES).quantile(q))


@needs_pandas
@pytest.mark.parametrize("ddof", [0, 1, 2])
def test_the_delta_degrees_of_freedom_reaches_the_divisor(firepanda: ModuleType, ddof: int) -> None:
    """The three that take a number take it, rather than declaring it.

    A `ddof` that is declared and dropped gives the right answer at the default
    and the wrong one everywhere else, which is why zero and two are here.
    """
    import pandas as pd

    for name in ("std", "var", "sem"):
        mine = getattr(firepanda.Series(VALUES), name)(ddof=ddof)
        theirs = getattr(pd.Series(VALUES), name)(ddof=ddof)
        assert mine == pytest.approx(theirs), name


@needs_pandas
def test_a_missing_value_is_skipped_the_way_pandas_skips_it(firepanda: ModuleType) -> None:
    """A hole in the column changes the divisor and not just the total."""
    import pandas as pd

    holed = [1.0, None, 3.0, None, 5.0]
    for name in ("sum", "mean", "count", "std", "median", "nunique"):
        mine = getattr(firepanda.Series(holed), name)()
        theirs = getattr(pd.Series(holed), name)()
        assert mine == pytest.approx(theirs), name


@needs_pandas
def test_a_reduction_with_no_answer_is_a_nan_and_not_a_none(firepanda: ModuleType) -> None:
    """The mean of nothing.

    The boundary hands back `None` for a reduction with no answer, because that
    is what an absent Arrow value is, and pandas gives a float NaN. The Python
    layer is where that turn happens, and a caller who wrote `math.isnan` would
    get a TypeError if it did not.
    """
    import pandas as pd

    empty = firepanda.Series([])
    assert math.isnan(empty.mean())
    assert math.isnan(pd.Series([], dtype="float64").mean())


@needs_pandas
def test_a_sum_of_nothing_is_zero_in_both(firepanda: ModuleType) -> None:
    """The one reduction that has an answer for a column with nothing in it."""
    import pandas as pd

    assert firepanda.Series([]).sum() == pd.Series([], dtype="float64").sum()


@needs_pandas
@pytest.mark.parametrize("name", ["sum", "mean", "min", "max", "count", "nunique"])
def test_a_frame_reduction_is_a_series_labelled_by_the_columns(
    firepanda: ModuleType, name: str
) -> None:
    """A frame reduces to one value per column, which is a series and not a row.

    This is the shape that is not the core's. `DataFrame.agg_all` gives a one row
    frame, which keeps every column's own type, and pandas gives a series, which
    has to pick one type for all of them.
    """
    import pandas as pd

    data = {"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]}
    mine = getattr(firepanda.DataFrame(data), name)()
    theirs = getattr(pd.DataFrame(data), name)()
    assert isinstance(mine, firepanda.Series)
    assert list(mine.index) == list(theirs.index)
    assert mine.tolist() == pytest.approx(theirs.tolist())


@needs_pandas
def test_a_mix_of_numbers_reduces_to_float64_as_pandas_does(firepanda: ModuleType) -> None:
    """An integer column and a float column have one type between them."""
    import pandas as pd

    data = {"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]}
    mine = firepanda.DataFrame(data).sum()
    theirs = pd.DataFrame(data).sum()
    assert mine.dtype == str(theirs.dtype)
    assert mine.tolist() == pytest.approx(theirs.tolist())


def test_a_mix_with_nothing_in_common_says_so(firepanda: ModuleType) -> None:
    """Rather than picking text, which is what a widening to strings would do.

    pandas answers an object series here, holding a number and a string side by
    side. There is no object column in Arrow, so the honest answer is a refusal
    with the reason in it.
    """
    with pytest.raises(TypeError, match="nothing in common"):
        firepanda.DataFrame({"a": [1, 2], "b": ["x", "y"]}).min()


def test_an_empty_frame_reduces_to_an_empty_series(firepanda: ModuleType) -> None:
    """No columns is no answers, rather than an error."""
    out = firepanda.DataFrame({}).sum()
    assert isinstance(out, firepanda.Series)
    assert out.tolist() == []


@pytest.mark.parametrize(
    ("call", "arguments", "expected"),
    [
        ("sum", {"skipna": False}, "skipna"),
        ("sum", {"min_count": 1}, "min_count"),
        ("mean", {"numeric_only": True}, "numeric_only"),
        ("std", {"numeric_only": True}, "numeric_only"),
        ("quantile", {"interpolation": "lower"}, "interpolation"),
        ("quantile", {"q": [0.1, 0.9]}, "single quantile"),
        ("nunique", {"dropna": False}, "dropna"),
    ],
)
def test_a_declared_argument_that_is_not_implemented_refuses(
    firepanda: ModuleType, call: str, arguments: dict[str, object], expected: str
) -> None:
    """Every one of them, by name, with the reason in the message."""
    with pytest.raises(NotImplementedError, match=expected):
        getattr(firepanda.Series(VALUES), call)(**arguments)


def test_reducing_across_a_row_is_refused_and_not_transposed(firepanda: ModuleType) -> None:
    """`axis=1` on a frame is a different kernel rather than the same one."""
    with pytest.raises(NotImplementedError, match="axis=1"):
        firepanda.DataFrame({"a": [1.0], "b": [2.0]}).sum(axis=1)


def test_a_series_has_only_the_one_axis(firepanda: ModuleType) -> None:
    """And says what pandas says, which names the axis and the type."""
    with pytest.raises(ValueError, match="No axis named 1"):
        firepanda.Series(VALUES).mean(axis=1)


def test_a_quantile_outside_the_interval_is_a_value_error(firepanda: ModuleType) -> None:
    """With the pandas message, which is the one a caller will have seen before."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        firepanda.Series(VALUES).quantile(1.5)
