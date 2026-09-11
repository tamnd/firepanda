"""The eight window reductions, checked against a running pandas.

The kernel has its own tests in `tests/test_window.mojo` and `tests/
test_spread.mojo` and they check the arithmetic. These check the surface: that
the nine arguments `s.rolling(...)` takes are spelled the way pandas spells them,
that the ones this library has no implementation for are refused by name rather
than ignored, and that the answers still match once the arguments have crossed
the boundary.

Every answer is compared against pandas rather than against a written down
constant, for the reason `test_astype.py` gives, with one exception. The
infinities are compared against what is true, because pandas replaces every
infinity in a window with a missing value before its kernel sees the column, and
the tests that assert the difference say so.

The last section is the same surface over a frame. It repeats the reductions and
the placements rather than trusting that a frame window is the columns windowed
one at a time, because that is the claim being made and a test that assumes it
tests nothing. What is genuinely new down there is the row labels a step leaves
behind, the eleven properties a window object reports about itself, and the two
arguments that mean something different on a frame from what they mean on a
column.
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

WINDOWED = ["sum", "mean", "count", "min", "max", "var", "std", "sem"]
"""The eight reductions, every one of which has to answer over every placement.

Written once because a reduction added to the library and not added here would
leave four tests passing on seven names and looking complete.
"""


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
@pytest.mark.parametrize("kind", WINDOWED)
def test_every_reduction_matches_over_a_plain_window(firepanda: ModuleType, kind: str) -> None:
    """The eight names, over a five wide window with nothing unusual about it."""
    assert same(
        getattr(made(firepanda).rolling(5), kind)(),
        getattr(theirs().rolling(5), kind)(),
    )


@needs_pandas
@pytest.mark.parametrize("kind", WINDOWED)
def test_every_reduction_matches_over_an_expanding_window(firepanda: ModuleType, kind: str) -> None:
    """The same eight with no near end, which is the other half of the surface."""
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
@pytest.mark.parametrize("kind", WINDOWED)
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
    for kind in WINDOWED:
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
    """The first of the asserted differences, and the reason for all of them.

    pandas cannot get an infinity out of a window, and not because it disagrees
    about the arithmetic. `BaseWindow._prep_values` replaces every infinity in
    the column with a NaN before the kernel runs, so an infinity is a missing
    value to a pandas window and the window is reduced over the rows either side
    of it. Its `count` is the one reduction that does not see the replacement, so
    pandas will tell you a window holds two values and then answer the sum of
    one of them. An infinity in a column is an ordinary value here and this
    counts the infinities beside the total.
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
    """pandas answers the other row of the window here, for the reason the sum
    test above gives, which is that the infinity was gone before its kernel
    started. An infinity in a column is an ordinary value and this answers it."""
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


@needs_pandas
@pytest.mark.parametrize("kind", ["var", "std", "sem"])
@pytest.mark.parametrize("ddof", [0, 1, 2, 3, -1])
def test_the_degrees_of_freedom_cross_the_boundary(
    firepanda: ModuleType, kind: str, ddof: int
) -> None:
    """The one argument the three spreads have that the other five do not.

    Five values including a negative one, which pandas accepts and which divides
    by more than the count rather than less, and a three that leaves a four wide
    window one degree of freedom and a three wide window none.
    """
    assert same(
        getattr(made(firepanda).rolling(4), kind)(ddof=ddof),
        getattr(theirs().rolling(4), kind)(ddof=ddof),
    )
    assert same(
        getattr(made(firepanda).expanding(), kind)(ddof=ddof),
        getattr(theirs().expanding(), kind)(ddof=ddof),
    )


@needs_pandas
@pytest.mark.parametrize("kind", ["var", "std", "sem"])
def test_a_window_with_no_degrees_of_freedom_left_is_missing(
    firepanda: ModuleType, kind: str
) -> None:
    """A count equal to the degrees of freedom leaves no divisor, and pandas
    answers a column of nulls rather than dividing by nought, so a one wide
    window is missing everywhere at the default."""
    assert same(
        getattr(made(firepanda).rolling(1), kind)(),
        getattr(theirs().rolling(1), kind)(),
    )
    assert same(
        getattr(made(firepanda).rolling(1), kind)(ddof=0),
        getattr(theirs().rolling(1), kind)(ddof=0),
    )


@needs_pandas
def test_the_degrees_of_freedom_have_to_be_a_whole_number(firepanda: ModuleType) -> None:
    """pandas truncates a float here, so `ddof=1.5` quietly answers the `ddof=1`
    column and a caller who wrote that meant something and did not get it. This
    refuses it with a sentence about the argument, which is the one place in this
    file where the difference is on purpose and is not about arithmetic."""
    with pytest.raises(ValueError, match="ddof must be an integer"):
        made(firepanda).rolling(3).var(ddof=1.5)
    assert same(theirs().rolling(3).var(ddof=1.5), theirs().rolling(3).var(ddof=1))


@needs_pandas
def test_the_standard_error_is_the_one_spread_with_no_engine_to_choose(
    firepanda: ModuleType,
) -> None:
    """pandas writes `sem` as `std(ddof) / count ** 0.5` rather than as a kernel,
    so it never had a numba path to offer and its signature has no `engine` on
    it. Declaring one here that pandas does not have would fail the signature
    parity check, so `sem` is the one reduction of the eight that takes only the
    two arguments."""
    with pytest.raises(TypeError):
        made(firepanda).rolling(3).sem(engine="cython")
    with pytest.raises(TypeError):
        theirs().rolling(3).sem(engine="cython")


@needs_pandas
def test_a_spread_over_a_window_holding_an_infinity_is_not_a_number(
    firepanda: ModuleType,
) -> None:
    """pandas answers `[0, 0, 0, 0.25, 0.25]` here and the middle three are
    wrong, for the reason the sum test above gives.

    The mean of a set holding an infinity is an infinity, every deviation from it
    is an infinity minus an infinity, and there is no number there. pandas had
    already replaced the infinity with a missing value, so its window over rows
    nought and one is a window over one value and it answers the variance of one
    value. Worse, it is not even consistent with itself across the degrees of
    freedom: the same windows are NaN at the default, because one value leaves no
    divisor there, and nought at `ddof=0`, because one value leaves a divisor of
    one.
    """
    rows = [1.0, math.inf, 2.0, 3.0, 4.0]
    got = firepanda.Series(rows, name="v").rolling(2, min_periods=1).var(ddof=0).tolist()
    assert got[0] == 0.0
    assert math.isnan(got[1]) and math.isnan(got[2])
    assert got[3] == 0.25
    assert got[4] == 0.25
    them = theirs(rows).rolling(2, min_periods=1).var(ddof=0).tolist()
    assert them[1] == 0.0 and them[2] == 0.0


@needs_pandas
def test_a_spread_that_overflowed_recovers_once_the_value_leaves(
    firepanda: ModuleType,
) -> None:
    """pandas answers `[nan, 0, inf, inf, inf]` here and the last two are wrong.

    A window holding ten to the two hundred and a small number has a variance
    too large for a double and genuinely is an infinity. Every window after it
    holds small numbers only. pandas cannot get back to them because its
    accumulated deviations went infinite and no later subtraction brings them
    back, which is the same failure as the running total and not the same cause
    as the infinities above. The carried state here notices that it has stopped
    being a number and rebuilds the window from its own rows.
    """
    rows = [1e200, 1e200, 1.0, 2.0, 3.0]
    got = firepanda.Series(rows, name="v").rolling(2).var().tolist()
    assert math.isnan(got[0])
    assert got[1] == 0.0
    assert math.isinf(got[2])
    assert got[3] == 0.5
    assert got[4] == 0.5
    them = theirs(rows).rolling(2).var().tolist()
    assert math.isinf(them[3]) and math.isinf(them[4])


@needs_pandas
def test_a_large_value_leaving_the_window_does_not_take_the_answer_with_it(
    firepanda: ModuleType,
) -> None:
    """A large value beside small ones, where subtracting it back out of the
    accumulated deviations leaves the remainder as the difference of two large
    numbers and every digit of it can be wrong. Carrying the state answers three
    quarters for the window over one, two and three, so the state is rebuilt from
    its own rows when its error bound says the digits it is holding are no longer
    digits of the answer. pandas gets this one right too and the two agree."""
    rows = [1e8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    got = firepanda.Series(rows, name="v").rolling(3).var()
    assert got.tolist()[3:] == [1.0, 1.0, 1.0, 1.0]
    assert same(got, theirs(rows).rolling(3).var())


COLUMNS: dict[str, list[Any]] = {"a": ROWS, "b": [x * 3 for x in ROWS], "c": HOLED + ROWS[:4]}
"""Three columns of ten rows, one of them holed, so that a frame window has both
a column where every window fills and one where they do not."""


def framed(firepanda: ModuleType) -> Any:
    """The three columns as a firepanda frame."""
    return firepanda.DataFrame(COLUMNS)


def their_frame() -> Any:
    """The same three columns in pandas."""
    import pandas as pd

    return pd.DataFrame(COLUMNS, dtype="float64")


def matching(mine: Any, them: Any) -> bool:
    """Compares two frames column by column, reading a missing row as equal to a
    missing row, which is what `same` does for a column."""
    if list(mine.columns) != list(them.columns):
        return False
    return all(same(mine[name], them[name]) for name in list(mine.columns))


@needs_pandas
@pytest.mark.parametrize("kind", WINDOWED)
def test_every_reduction_matches_over_a_frame(firepanda: ModuleType, kind: str) -> None:
    """The same eight names over a frame, which answers a frame of the same
    columns in the same order rather than a column."""
    mine = getattr(framed(firepanda).rolling(4), kind)()
    them = getattr(their_frame().rolling(4), kind)()
    assert type(mine).__name__ == "DataFrame"
    assert matching(mine, them)


@needs_pandas
@pytest.mark.parametrize("kind", WINDOWED)
def test_every_reduction_matches_over_an_expanding_frame(firepanda: ModuleType, kind: str) -> None:
    """The eight again over the window with no near end."""
    assert matching(
        getattr(framed(firepanda).expanding(), kind)(),
        getattr(their_frame().expanding(), kind)(),
    )


@needs_pandas
@pytest.mark.parametrize(
    "arguments",
    [
        {"window": 3, "center": True},
        {"window": 4, "center": True},
        {"window": 3, "closed": "left"},
        {"window": 3, "closed": "both"},
        {"window": 3, "closed": "neither"},
        {"window": 3, "min_periods": 1},
        {"window": 10, "step": 3},
        {"window": 1},
        {"window": 0},
        {"window": 40},
    ],
)
def test_where_the_window_sits_matches_over_a_frame(
    firepanda: ModuleType, arguments: dict[str, Any]
) -> None:
    """Ten placements, each answered by both.

    The same list the column form is checked against, because a frame window is
    the columns windowed one at a time and there is nothing about where a window
    sits that a frame can get wrong on its own. What this catches is the row
    labels, which `step` makes different from the frame's own and which are
    taken off the first answered column rather than off the frame.
    """
    assert matching(
        framed(firepanda).rolling(**arguments).sum(),
        their_frame().rolling(**arguments).sum(),
    )


@needs_pandas
def test_a_frame_window_is_built_off_the_frame_and_named_the_same(
    firepanda: ModuleType,
) -> None:
    """One class for a column and a frame here, where pandas has two, and both of
    pandas' are named what this one is named. A program that checks what it was
    handed back checks the name."""
    assert type(framed(firepanda).rolling(2)).__name__ == "Rolling"
    assert type(framed(firepanda).expanding()).__name__ == "Expanding"
    assert type(their_frame().rolling(2)).__name__ == "Rolling"
    assert type(their_frame().expanding()).__name__ == "Expanding"


@needs_pandas
def test_the_window_reports_back_what_it_was_given(firepanda: ModuleType) -> None:
    """The eleven properties pandas puts on a window object, read off both.

    Six of them are the arguments handed straight back, and the reason they are
    kept rather than resolved is here: pandas answers None for a `closed` that
    was not given and for a `min_periods` that was not given, so filling in the
    default when the object is built would report a decision as an argument.
    """
    named = ("window", "min_periods", "center", "closed", "step", "method", "win_type", "on")
    mine = framed(firepanda).rolling(3, center=True, closed="left", step=2)
    them = their_frame().rolling(3, center=True, closed="left", step=2)
    for name in named:
        assert getattr(mine, name) == getattr(them, name), name
    assert mine.ndim == them.ndim == 2
    assert mine.exclusions == them.exclusions == frozenset()
    assert list(mine.obj.columns) == list(them.obj.columns)

    plain = framed(firepanda).rolling(3)
    other = their_frame().rolling(3)
    assert plain.closed is other.closed is None
    assert plain.min_periods is other.min_periods is None

    grown = framed(firepanda).expanding(2)
    same_grown = their_frame().expanding(2)
    for name in named:
        assert getattr(grown, name) == getattr(same_grown, name), name


@needs_pandas
def test_a_column_window_reports_itself_as_one_dimensional(firepanda: ModuleType) -> None:
    """The one property that is not the same on both owners, and the only reason
    the mixin has to know which of the two it is holding outside the reduction."""
    mine = made(firepanda).rolling(2)
    them = theirs().rolling(2)
    assert mine.ndim == them.ndim == 1
    assert type(mine.obj).__name__ == type(them.obj).__name__ == "Series"


@needs_pandas
def test_a_frame_with_a_text_column_in_it_names_the_column(firepanda: ModuleType) -> None:
    """Both refuse, and this library says which column it was.

    pandas says `Cannot aggregate non-numeric type: str`, which over a frame of
    forty columns sends the reader back to look for the column themselves. The
    class is the same difference `test_a_text_column_has_nothing_to_reduce`
    measures on a column.
    """
    import pandas as pd

    mine = firepanda.DataFrame({"a": [1.0, 2.0, 3.0], "t": ["x", "y", "z"]})
    them = pd.DataFrame({"a": [1.0, 2.0, 3.0], "t": ["x", "y", "z"]})
    with pytest.raises(TypeError, match="'t'"):
        mine.rolling(2).sum()
    with pytest.raises(pd.errors.DataError):
        them.rolling(2).sum()


@needs_pandas
def test_the_column_that_cannot_be_reduced_is_found_before_any_column_is_read(
    firepanda: ModuleType,
) -> None:
    """The text column is last, and it still raises rather than half answering.

    Cheap to get wrong and invisible when it is, because the answer is thrown
    away either way. It matters because a caller who gets an error should not
    have to wonder what was already spent, and it is the one thing about a frame
    window that is not per column.
    """
    mine = firepanda.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "t": ["x", "y"]})
    with pytest.raises(TypeError, match="'t'"):
        mine.rolling(2).sum()


@needs_pandas
def test_dropping_the_columns_a_window_cannot_read_is_refused(
    firepanda: ModuleType,
) -> None:
    """`numeric_only` is one name asking two questions.

    On a column both values agree everywhere there is an answer, so both are
    accepted, which the test above this section checks. On a frame True says to
    drop the columns rather than refuse them, which decides which columns come
    back, so it is refused the way the group by path refuses it.
    """
    import pandas as pd

    mine = firepanda.DataFrame({"a": [1.0, 2.0], "t": ["x", "y"]})
    them = pd.DataFrame({"a": [1.0, 2.0], "t": ["x", "y"]})
    with pytest.raises(NotImplementedError, match="numeric_only"):
        mine.rolling(2).sum(numeric_only=True)
    assert list(them.rolling(2).sum(numeric_only=True).columns) == ["a"]
    assert matching(
        framed(firepanda).rolling(2).sum(numeric_only=False),
        their_frame().rolling(2).sum(numeric_only=False),
    )


@needs_pandas
def test_ordering_the_window_by_a_column_is_refused_on_a_frame_too(
    firepanda: ModuleType,
) -> None:
    """`on` is the one refusal whose reason changes on a frame and whose answer
    does not.

    On a frame pandas both orders the window by the named column and carries it
    through into the answer unreduced. Carrying it is the easy half and the
    ordering is the point, and ordering by a column means a window given as a
    duration, so writing the copying half would be a `rolling("2D", on="t")`
    that silently counted rows.
    """
    with pytest.raises(NotImplementedError, match="on"):
        framed(firepanda).rolling(2, on="a")
    with pytest.raises(NotImplementedError, match="win_type"):
        framed(firepanda).rolling(2, win_type="boxcar")
    with pytest.raises(NotImplementedError, match="method"):
        framed(firepanda).rolling(2, method="table")
    with pytest.raises(NotImplementedError, match="method"):
        framed(firepanda).expanding(method="table")


@needs_pandas
def test_a_frame_of_no_columns_is_handed_back(firepanda: ModuleType) -> None:
    """There was nothing to reduce and nothing was reduced.

    Worth a test because the obvious implementation puts the answered columns
    back together and putting nothing together makes a frame of no rows, which
    is a different frame from one of no columns.
    """
    mine = firepanda.DataFrame({})
    assert mine.rolling(2).sum().shape == (0, 0)
    assert mine.expanding().sum().shape == (0, 0)


@needs_pandas
def test_an_integer_frame_comes_back_as_float64(firepanda: ModuleType) -> None:
    """Every column of the answer is float64, the same as on a column, because
    there is nowhere else to put the holes at the top."""
    import pandas as pd

    mine = firepanda.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    them = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    got = mine.rolling(2).sum()
    assert [str(got[name].dtype) for name in list(got.columns)] == ["float64", "float64"]
    assert matching(got, them.rolling(2).sum())
