"""The five window reductions, checked against a running pandas.

The kernel has its own tests in `tests/test_window.mojo` and they check the
arithmetic. These check the surface: that the nine arguments `s.rolling(...)`
takes are spelled the way pandas spells them, that the ones this library has no
implementation for are refused by name rather than ignored, and that the answers
still match once the arguments have crossed the boundary.

Every answer is compared against pandas rather than against a written down
constant, for the reason `test_astype.py` gives, with one exception. The
infinities are compared against what is true, because pandas carries one running
total and cannot recover from an infinity passing through it, and the two tests
at the bottom assert the difference rather than working around it.
"""

from __future__ import annotations

import importlib.util
import math
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

ROWS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
"""Ten rows where every window sums to something a reader can check by hand.

Deliberately not random. The point of these tests is which rows a window covers,
and a column of counting numbers makes a wrong window visible as a wrong number
rather than as two long lists that have to be diffed.
"""

HOLED = [1.0, None, 3.0, None, 5.0, 6.0]
"""Six rows with the gaps arranged so that no three wide window holds three
values, which is what makes `min_periods` visible."""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda float column."""
    return firepanda.Series(values, name="v")


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas."""
    import pandas as pd

    return pd.Series(values, name="v", dtype="float64")


def same(mine: Any, them: Any) -> bool:
    """Compares two answers, reading a missing row as equal to a missing row."""
    ours = mine.tolist()
    other = them.tolist()
    if len(ours) != len(other):
        return False
    for one, two in zip(ours, other, strict=True):
        if one is None or (isinstance(one, float) and math.isnan(one)):
            if two is None or (isinstance(two, float) and math.isnan(two)):
                continue
            return False
        if one != two:
            return False
    return list(mine.index) == list(them.index)


@needs_pandas
@pytest.mark.parametrize("kind", ["sum", "mean", "count", "min", "max"])
def test_every_reduction_matches_over_a_plain_window(firepanda: ModuleType, kind: str) -> None:
    """The five names, over a five wide window with nothing unusual about it."""
    assert same(
        getattr(made(firepanda).rolling(5), kind)(),
        getattr(theirs().rolling(5), kind)(),
    )


@needs_pandas
@pytest.mark.parametrize("kind", ["sum", "mean", "count", "min", "max"])
def test_every_reduction_matches_over_an_expanding_window(firepanda: ModuleType, kind: str) -> None:
    """The same five with no near end, which is the other half of the surface."""
    assert same(
        getattr(made(firepanda).expanding(), kind)(),
        getattr(theirs().expanding(), kind)(),
    )


@needs_pandas
@pytest.mark.parametrize(
    "arguments",
    [
        {"window": 5},
        {"window": 5, "min_periods": 1},
        {"window": 4, "min_periods": 3},
        {"window": 5, "center": True},
        {"window": 4, "center": True},
        {"window": 5, "closed": "left"},
        {"window": 5, "closed": "both"},
        {"window": 5, "closed": "neither"},
        {"window": 1},
        {"window": 0},
        {"window": 100},
        {"window": 10, "step": 3},
        {"window": 4, "step": 2, "center": True},
    ],
)
def test_where_the_window_sits_matches(firepanda: ModuleType, arguments: dict[str, Any]) -> None:
    """Thirteen spellings of the same question, which is which rows to read.

    A step is in here twice because it changes the height of the answer as well
    as its values, and `same` compares the row labels, so a stepped window that
    kept the wrong ones would fail here rather than looking right.
    """
    assert same(
        made(firepanda).rolling(**arguments).sum(),
        theirs().rolling(**arguments).sum(),
    )


@needs_pandas
@pytest.mark.parametrize("periods", [0, 1, 3, 5, 10])
def test_how_long_an_expanding_window_waits_matches(firepanda: ModuleType, periods: int) -> None:
    """The one parameter an expanding window has, at every value worth trying."""
    assert same(
        made(firepanda).expanding(min_periods=periods).sum(),
        theirs().expanding(min_periods=periods).sum(),
    )


@needs_pandas
@pytest.mark.parametrize("kind", ["sum", "mean", "count", "min", "max"])
def test_a_missing_row_is_stepped_over(firepanda: ModuleType, kind: str) -> None:
    """A column with gaps in it, at the one `min_periods` that answers anything.

    `count` is the one that reads differently, and the difference is pandas' and
    not this library's: it tests `min_periods` against how many rows the window
    covers rather than how many hold a value, because it computes the count as a
    rolling sum over the presence indicator.
    """
    assert same(
        getattr(made(firepanda, HOLED).rolling(3, min_periods=1), kind)(),
        getattr(theirs(HOLED).rolling(3, min_periods=1), kind)(),
    )


@needs_pandas
def test_a_window_over_gaps_that_never_fills_is_missing_everywhere(
    firepanda: ModuleType,
) -> None:
    """The default `min_periods` on the same column, where no window ever has
    three values in it and pandas answers a column of nulls rather than raising."""
    assert same(made(firepanda, HOLED).rolling(3).sum(), theirs(HOLED).rolling(3).sum())


@needs_pandas
def test_an_integer_column_comes_back_as_float64(firepanda: ModuleType) -> None:
    """Every window reduction answers float64 in pandas, including `count` and
    including the extremes, because there is nowhere else to put the holes at the
    top of the column."""
    import pandas as pd

    mine = firepanda.Series([1, 2, 3, 4], name="v")
    them = pd.Series([1, 2, 3, 4], name="v")
    for kind in ("sum", "mean", "count", "min", "max"):
        ours = getattr(mine.rolling(2), kind)()
        assert str(ours.dtype) == "float64"
        assert same(ours, getattr(them.rolling(2), kind)())


@needs_pandas
def test_the_two_classes_are_named_what_pandas_names_them(firepanda: ModuleType) -> None:
    """A program that checks what it was handed back checks the name, and the
    conformance board is one of those programs."""
    assert type(made(firepanda).rolling(2)).__name__ == "Rolling"
    assert type(made(firepanda).expanding()).__name__ == "Expanding"
    assert type(theirs().rolling(2)).__name__ == "Rolling"
    assert type(theirs().expanding()).__name__ == "Expanding"


@needs_pandas
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"window": -1}, "window must be an integer 0 or greater"),
        ({"window": 2.5}, "window must be an integer 0 or greater"),
        ({"window": "2s"}, "window must be an integer 0 or greater"),
        ({"window": 2, "min_periods": -1}, "min_periods must be >= 0"),
        ({"window": 2, "min_periods": 5}, "min_periods 5 must be <= window 2"),
        ({"window": 2, "closed": "outer"}, "closed must be 'right', 'left', 'both' or 'neither'"),
        ({"window": 2, "center": 1}, "center must be a boolean"),
    ],
)
def test_the_arguments_that_do_not_describe_a_window_are_refused_the_same_way(
    firepanda: ModuleType, arguments: dict[str, Any], message: str
) -> None:
    """Seven bad spellings, each raising a `ValueError` with pandas' own sentence.

    Raised out of `rolling` and not out of the reduction after it, which is also
    where pandas raises it. A program that catches the wrong line is a program
    whose error handling does not run.
    """
    with pytest.raises(ValueError, match=message.replace("(", r"\(")):
        made(firepanda).rolling(**arguments)
    with pytest.raises(ValueError, match=message.replace("(", r"\(")):
        theirs().rolling(**arguments)


@needs_pandas
def test_a_step_of_zero_is_refused_here_and_divides_by_zero_there(
    firepanda: ModuleType,
) -> None:
    """The one argument check that is not pandas'.

    pandas accepts a step of nought when the window is built and divides by it
    when the reduction runs, so what a caller sees is a `ZeroDivisionError` out
    of `.sum()` with nothing in it naming the argument that caused it. A step of
    nought asks for the same row forever, so it is refused where the other four
    are.
    """
    with pytest.raises(ValueError, match="step must be >= 1"):
        made(firepanda).rolling(2, step=0)
    with pytest.raises(ZeroDivisionError):
        theirs().rolling(2, step=0).sum()


@needs_pandas
def test_the_arguments_with_no_implementation_behind_them_are_refused_by_name(
    firepanda: ModuleType,
) -> None:
    """Four declared parameters that are not honoured, each naming itself.

    Declared rather than left out, for the reason document 18 gives: the
    signature parity check compares the whole parameter list against a running
    pandas, and a caller who passes one gets a sentence about it rather than a
    TypeError about an unexpected keyword.
    """
    series = made(firepanda)
    with pytest.raises(NotImplementedError, match="win_type"):
        series.rolling(2, win_type="boxcar")
    with pytest.raises(NotImplementedError, match="on"):
        series.rolling(2, on="v")
    with pytest.raises(NotImplementedError, match="method"):
        series.rolling(2, method="table")
    with pytest.raises(NotImplementedError, match="engine"):
        series.rolling(2).sum(engine="numba")


@needs_pandas
def test_the_default_engine_spelled_out_is_still_the_default(firepanda: ModuleType) -> None:
    """`cython` names the path this library is on, so it is accepted rather than
    refused, and `numeric_only` is accepted at both values because on a column
    the two agree everywhere there is an answer."""
    assert same(
        made(firepanda).rolling(3).sum(engine="cython"),
        theirs().rolling(3).sum(engine="cython"),
    )
    assert same(
        made(firepanda).rolling(3).sum(numeric_only=True),
        theirs().rolling(3).sum(numeric_only=True),
    )


@needs_pandas
def test_a_text_column_has_nothing_to_reduce(firepanda: ModuleType) -> None:
    """Both refuse, and the class each raises is measured rather than assumed.

    pandas raises `pandas.errors.DataError`, which inherits from `Exception` and
    from nothing else, so `except TypeError` does not catch it and neither does
    `except ValueError`. This raises `DTypeError`, which is a `TypeError`,
    because an argument of the wrong type is what happened and that is where the
    rest of this library puts it. Catching `Exception` catches both, and this
    test records the difference rather than papering over it.
    """
    import pandas as pd

    with pytest.raises(TypeError):
        firepanda.Series(["a", "b", "c"], name="v").rolling(2).sum()
    with pytest.raises(pd.errors.DataError):
        pd.Series(["a", "b", "c"], name="v").rolling(2).sum()
    assert not issubclass(pd.errors.DataError, TypeError)


def test_a_window_holding_one_infinity_sums_to_it(firepanda: ModuleType) -> None:
    """The first of the two asserted differences, and the reason for both.

    pandas carries one running total, so an infinity entering the window makes
    it infinite and subtracting the infinity again gives a NaN rather than
    giving the total back. Every window after that reads NaN until the window
    empties. That is not a rounding difference, it is a wrong answer on real
    data, and this counts the infinities beside the total instead.
    """
    rows = [1.0, 2.0, math.inf, 3.0, 4.0, 5.0, 6.0]
    got = firepanda.Series(rows, name="v").rolling(3).sum().tolist()
    assert math.isinf(got[2]) and got[2] > 0
    assert math.isinf(got[3]) and math.isinf(got[4])
    assert got[5] == 12.0
    assert got[6] == 15.0


def test_a_window_holding_both_infinities_is_not_a_number(firepanda: ModuleType) -> None:
    """The second, which is true rather than merely different, and it also has to
    survive both of them leaving the window again."""
    rows = [math.inf, -math.inf, 1.0, 2.0, 3.0, 4.0]
    got = firepanda.Series(rows, name="v").rolling(3).sum().tolist()
    assert math.isnan(got[2])
    assert got[4] == 6.0
    assert got[5] == 9.0


def test_an_extreme_over_a_window_holding_an_infinity_answers_it(
    firepanda: ModuleType,
) -> None:
    """pandas answers a NaN here because it seeds its running maximum with
    negative infinity and reads a result equal to that seed as an empty window.
    An infinity in a column is an ordinary value and this answers it."""
    rows = [1.0, math.inf, 2.0, 3.0]
    got = firepanda.Series(rows, name="v").rolling(2).max().tolist()
    assert math.isinf(got[1]) and math.isinf(got[2])
    assert got[3] == 3.0


def test_the_low_bits_survive_a_row_leaving_the_window(firepanda: ModuleType) -> None:
    """A large value beside small ones, which is where a running total that threw
    away the bits an addition dropped would answer one rather than two once the
    large value had left."""
    rows = [1e16, 1.0, 1.0, 1.0, 1.0]
    got = firepanda.Series(rows, name="v").rolling(2).sum().tolist()
    assert got[1] == 1e16
    assert got[2] == 2.0
    assert got[3] == 2.0
