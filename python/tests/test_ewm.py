"""Tests for `ewm` and the four reductions on what it hands back.

Every assertion here is against a running pandas rather than against a written
down number, which is the opposite of the arrangement in the kernel tests and is
deliberate. The kernel tests own the question of whether the recurrence is right
and quote pandas' answers so that the question can be answered without pandas
installed. These own the question of whether the pandas surface over it is right,
which is a question about signatures, defaults, property values and the class of
an exception, and none of those can be written down and trusted.

The comparisons carry a small relative tolerance rather than none, which the
window tests mostly do not. The reason is not the recurrence. The pandas wheel
these run against was compiled with the multiply and the add of the fold
contracted into one fused instruction, which rounds once where two operations
round twice, so a handful of rows differ in the last bit however correct both
sides are. `firepanda/kernel/ewm.mojo` says the same thing at more length.
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
"""Ten rows where a wrong decay is visible as a number that stops climbing."""

HOLED = [1.0, None, 3.0, None, None, 7.0, 9.0]
"""Seven rows with a single gap and then a double one.

One gap cannot tell the two readings of `ignore_na` apart by much and two can,
which is the whole reason the column is shaped this way rather than evenly.
"""

DECAYS = (
    {"span": 5.0},
    {"com": 2.0},
    {"halflife": 3.0},
    {"alpha": 0.25},
)
"""The four spellings of the decay, each on its own.

The first two describe the same window on purpose, because a caller relies on
`span=5` and `com=2` agreeing without ever checking it.
"""

REDUCED = ("mean", "sum", "var", "std")
"""The four reductions, written once so that a fifth added to the library and not
added here cannot leave these tests passing on four names and looking
complete."""

NEAR = 1e-15
"""How far apart two answers are allowed to be, relative to the larger of the two.

About four units in the last place, which covers the fused multiply-add the
module docstring describes and nothing larger than it.
"""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda float column."""
    return firepanda.Series(values, name="v")


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas."""
    import pandas as pd

    return pd.Series(values, name="v", dtype="float64")


def same(mine: Any, them: Any) -> bool:
    """Compares two answers, reading a missing row as equal to a missing row.

    Args:
        mine: What this library answered.
        them: What pandas answered.

    Returns:
        True if every row agrees to within `NEAR` and the labels agree exactly.
    """
    ours = mine.tolist()
    other = them.tolist()
    if len(ours) != len(other):
        return False
    for one, two in zip(ours, other, strict=True):
        missing = one is None or (isinstance(one, float) and math.isnan(one))
        theirs_missing = two is None or (isinstance(two, float) and math.isnan(two))
        if missing or theirs_missing:
            if missing and theirs_missing:
                continue
            return False
        if one != two and abs(one - two) > NEAR * max(abs(one), abs(two), 1.0):
            return False
    return list(mine.index) == list(them.index)


@needs_pandas
@pytest.mark.parametrize("decay", DECAYS)
@pytest.mark.parametrize("kind", REDUCED)
def test_every_reduction_matches_under_every_spelling_of_the_decay(
    firepanda: ModuleType, decay: dict[str, float], kind: str
) -> None:
    """Four reductions by four decays, which is the whole of the default path."""
    mine = getattr(made(firepanda).ewm(**decay), kind)()
    them = getattr(theirs().ewm(**decay), kind)()
    assert same(mine, them)


@needs_pandas
def test_a_span_and_a_centre_of_mass_describe_the_same_window(
    firepanda: ModuleType,
) -> None:
    """`span=5` and `com=2` are one number written two ways, on both sides."""
    assert made(firepanda).ewm(span=5).mean().tolist() == (
        made(firepanda).ewm(com=2).mean().tolist()
    )


@needs_pandas
@pytest.mark.parametrize("adjust", [True, False])
@pytest.mark.parametrize("ignore_na", [True, False])
@pytest.mark.parametrize("kind", ["mean", "var", "std"])
def test_the_two_flags_are_read_the_same_way_over_gaps(
    firepanda: ModuleType, adjust: bool, ignore_na: bool, kind: str
) -> None:
    """All four combinations over the column with the two gaps in it.

    These are four genuinely different answers and not four roundings of one, so
    a flag that is quietly dropped fails here rather than somewhere downstream.
    `sum` is left out because pandas has no answer for it with `adjust` off, and
    the test below is about that.
    """
    mine = getattr(
        made(firepanda, HOLED).ewm(alpha=0.3, adjust=adjust, ignore_na=ignore_na), kind
    )()
    them = getattr(theirs(HOLED).ewm(alpha=0.3, adjust=adjust, ignore_na=ignore_na), kind)()
    assert same(mine, them)


@needs_pandas
@pytest.mark.parametrize("periods", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("kind", REDUCED)
def test_how_many_values_a_row_waits_for_matches(
    firepanda: ModuleType, periods: int, kind: str
) -> None:
    """`min_periods` counts values and not rows, so the gaps make it visible."""
    mine = getattr(made(firepanda, HOLED).ewm(alpha=0.3, min_periods=periods), kind)()
    them = getattr(theirs(HOLED).ewm(alpha=0.3, min_periods=periods), kind)()
    assert same(mine, them)


@needs_pandas
def test_a_count_of_nought_is_reported_back_as_one(firepanda: ModuleType) -> None:
    """The one argument pandas resolves in its constructor, so this one does too."""
    assert made(firepanda).ewm(span=5).min_periods == theirs().ewm(span=5).min_periods
    assert (
        made(firepanda).ewm(span=5, min_periods=0).min_periods
        == theirs().ewm(span=5, min_periods=0).min_periods
    )
    assert (
        made(firepanda).ewm(span=5, min_periods=-1).min_periods
        == theirs().ewm(span=5, min_periods=-1).min_periods
    )


@needs_pandas
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("kind", ["var", "std"])
def test_the_correction_the_two_spreads_read_matches(
    firepanda: ModuleType, bias: bool, kind: str
) -> None:
    """`bias` is the one parameter a reduction here reads and the decay does not."""
    mine = getattr(made(firepanda, HOLED).ewm(alpha=0.3), kind)(bias=bias)
    them = getattr(theirs(HOLED).ewm(alpha=0.3), kind)(bias=bias)
    assert same(mine, them)


@needs_pandas
def test_the_deviation_is_the_root_of_the_variance(firepanda: ModuleType) -> None:
    """Checked against itself as well as against pandas, since it is one line."""
    window = made(firepanda, HOLED).ewm(alpha=0.3)
    variances = window.var().tolist()
    deviations = window.std().tolist()
    for one, two in zip(variances, deviations, strict=True):
        if math.isnan(one):
            assert math.isnan(two)
        else:
            assert abs(math.sqrt(one) - two) <= NEAR * max(abs(two), 1.0)


@needs_pandas
def test_a_total_with_the_unadjusted_recurrence_is_refused_the_same_way(
    firepanda: ModuleType,
) -> None:
    """A real pandas signature behind which pandas has no answer.

    pandas raises `NotImplementedError` rather than choosing one of the two things
    such a total could mean, and inventing an answer would be the one place in
    this family where firepanda is not compatible. So the class and the sentence
    are both copied.
    """
    with pytest.raises(NotImplementedError) as theirs_raised:
        theirs().ewm(alpha=0.3, adjust=False).sum()
    with pytest.raises(NotImplementedError) as mine_raised:
        made(firepanda).ewm(alpha=0.3, adjust=False).sum()
    assert str(mine_raised.value) == str(theirs_raised.value)


@needs_pandas
@pytest.mark.parametrize(
    "decay",
    [
        {},
        {"span": 5.0, "com": 2.0},
        {"halflife": 3.0, "alpha": 0.5},
        {"span": 0.5},
        {"com": -1.0},
        {"halflife": 0.0},
        {"alpha": 0.0},
        {"alpha": 1.5},
    ],
)
def test_a_decay_that_is_not_one_number_is_refused_the_same_way(
    firepanda: ModuleType, decay: dict[str, float]
) -> None:
    """None of the four, two of the four, and each of the four out of its range.

    The sentences are compared and not just the class, because every one of these
    is a caller writing something they meant and the sentence is how they find
    out which part of it was wrong.
    """
    with pytest.raises(ValueError) as theirs_raised:
        theirs().ewm(**decay)
    with pytest.raises(ValueError) as mine_raised:
        made(firepanda).ewm(**decay)
    assert str(mine_raised.value) == str(theirs_raised.value)


def test_a_decay_that_is_not_a_number_at_all_is_refused(firepanda: ModuleType) -> None:
    """A half life is the one of the four with a second reading, and it needs a
    calendar."""
    with pytest.raises(ValueError, match="halflife must be a real number"):
        made(firepanda).ewm(halflife="2D")
    with pytest.raises(ValueError, match="span must be a real number"):
        made(firepanda).ewm(span="5")


def test_the_arguments_with_no_implementation_behind_them_are_refused_by_name(
    firepanda: ModuleType,
) -> None:
    """Declared so the signature matches, refused so nothing is quietly ignored."""
    with pytest.raises(NotImplementedError, match="times"):
        made(firepanda).ewm(span=5, times=[1, 2, 3])
    with pytest.raises(NotImplementedError, match="method"):
        made(firepanda).ewm(span=5, method="table")


def test_the_default_engine_spelled_out_is_still_the_default(
    firepanda: ModuleType,
) -> None:
    """`cython` names the one implementation here, and numba names none of it."""
    plain = made(firepanda).ewm(span=5).mean().tolist()
    assert made(firepanda).ewm(span=5).mean(engine="cython").tolist() == plain
    with pytest.raises(NotImplementedError, match="numba"):
        made(firepanda).ewm(span=5).mean(engine="numba")
    with pytest.raises(NotImplementedError, match="engine_kwargs"):
        made(firepanda).ewm(span=5).mean(engine_kwargs={"nopython": True})


def test_the_flags_have_to_be_flags(firepanda: ModuleType) -> None:
    """pandas reads anything here for truth, so `adjust="no"` means yes there.

    A caller who wrote that meant the opposite of what they got, which is the
    reason `center` and `pct` are checked the same way on the rolling side.
    """
    with pytest.raises(ValueError, match="adjust must be a boolean"):
        made(firepanda).ewm(span=5, adjust="no")
    with pytest.raises(ValueError, match="ignore_na must be a boolean"):
        made(firepanda).ewm(span=5, ignore_na="no")
    with pytest.raises(ValueError, match="bias must be a boolean"):
        made(firepanda).ewm(span=5).var(bias="no")


@needs_pandas
@pytest.mark.parametrize("kind", REDUCED)
def test_a_frame_decays_a_column_at_a_time(firepanda: ModuleType, kind: str) -> None:
    """Every column of a frame has the same rows, so there is nothing else to do."""
    import pandas as pd

    data = {"a": ROWS, "b": [v * 2.0 for v in ROWS]}
    mine = getattr(firepanda.DataFrame(data).ewm(span=5), kind)()
    them = getattr(pd.DataFrame(data).ewm(span=5), kind)()
    assert list(mine.columns) == list(them.columns)
    for name in them.columns:
        assert same(mine[name], them[name])


def test_a_frame_is_not_asked_to_drop_the_columns_it_cannot_read(
    firepanda: ModuleType,
) -> None:
    """Which columns come back is a decision, and this library does not make it."""
    frame = firepanda.DataFrame({"a": ROWS, "b": [str(v) for v in ROWS]})
    with pytest.raises(NotImplementedError, match="numeric_only"):
        frame.ewm(span=5).mean(numeric_only=True)


def test_a_text_column_has_nothing_to_decay(firepanda: ModuleType) -> None:
    """The same refusal the rolling window gives, for the same reason."""
    with pytest.raises(TypeError, match="nothing to aggregate"):
        firepanda.Series(["a", "b", "c"], name="v").ewm(span=5).mean()


@needs_pandas
def test_an_integer_column_comes_back_as_float64(firepanda: ModuleType) -> None:
    """A weighted mean of whole numbers is not a whole number."""
    values = [1, 2, 3, 4]
    mine = firepanda.Series(values, name="v").ewm(span=3).mean()
    them = theirs([float(v) for v in values]).ewm(span=3).mean()
    assert str(mine.dtype) == "float64"
    assert same(mine, them)


@needs_pandas
def test_the_class_is_named_what_pandas_names_it(firepanda: ModuleType) -> None:
    """Code in the wild reads the type name, and `ewm` is reached from both owners."""
    column = made(firepanda).ewm(span=5)
    frame = firepanda.DataFrame({"a": ROWS}).ewm(span=5)
    assert type(column).__name__ == "ExponentialMovingWindow"
    assert type(frame).__name__ == "ExponentialMovingWindow"
    assert type(theirs().ewm(span=5)).__name__ == "ExponentialMovingWindow"


@needs_pandas
@pytest.mark.parametrize(
    "name",
    [
        "adjust",
        "alpha",
        "center",
        "closed",
        "com",
        "exclusions",
        "halflife",
        "ignore_na",
        "method",
        "min_periods",
        "ndim",
        "on",
        "span",
        "step",
        "times",
        "win_type",
        "window",
    ],
)
def test_what_a_window_object_reports_about_itself_matches(
    firepanda: ModuleType, name: str
) -> None:
    """Seventeen of the eighteen, and every one of them read out of pandas.

    `obj` is the one left out, because it answers the data itself and a firepanda
    column is not a pandas column. The rest are the arguments back and the seven
    questions a rolling window answers that this one answers with nothing, and
    those seven are the ones worth checking against pandas rather than against a
    written down None: an absent attribute is not the same answer as None and only
    pandas can say which of the two it gives.
    """
    assert getattr(made(firepanda).ewm(span=5), name) == getattr(theirs().ewm(span=5), name)


@needs_pandas
def test_the_decay_is_reported_back_the_way_it_arrived(firepanda: ModuleType) -> None:
    """`ewm(span=5).com` is None in pandas, so a conversion is not reported."""
    mine = made(firepanda).ewm(com=2)
    them = theirs().ewm(com=2)
    for name in ("com", "span", "halflife", "alpha"):
        assert getattr(mine, name) == getattr(them, name)


def test_the_data_is_reported_back(firepanda: ModuleType) -> None:
    """`obj` is what was windowed, which is the one thing pandas cannot be asked
    about here."""
    column = made(firepanda)
    assert made(firepanda).ewm(span=5).obj is not None
    assert column.ewm(span=5).obj is column
    assert column.ewm(span=5).ndim == 1
    assert firepanda.DataFrame({"a": ROWS}).ewm(span=5).ndim == 2


@needs_pandas
def test_a_column_of_nothing_decays_to_nothing(firepanda: ModuleType) -> None:
    """No value ever arrives, so the recurrence never starts."""
    values: list[Any] = [None, None, None]
    assert same(
        made(firepanda, values).ewm(alpha=0.3).mean(),
        theirs(values).ewm(alpha=0.3).mean(),
    )


@needs_pandas
def test_one_value_repeated_comes_back_unchanged(firepanda: ModuleType) -> None:
    """A flat column stays flat, which takes a guard in the fold to be true."""
    values = [2.5] * 6
    assert made(firepanda, values).ewm(alpha=0.3).mean().tolist() == values


@needs_pandas
def test_an_empty_column_decays_to_an_empty_column(firepanda: ModuleType) -> None:
    """Nothing to read and nothing to say about it."""
    values: list[Any] = []
    assert made(firepanda, values).ewm(span=5).mean().tolist() == []
    assert theirs(values).ewm(span=5).mean().tolist() == []
