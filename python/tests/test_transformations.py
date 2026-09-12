"""The twelve transformations, checked against a running pandas.

Same argument as `test_reductions.py`. A reduction folds a column down to one
number and a transformation hands back a column, and both are measured against a
live pandas rather than against constants, because a constant records what
somebody believed pandas did on the day they wrote it down.

What is new here and is not in the reduction tests is the index. A reduction
throws the labels away, so nothing about them can be wrong. A transformation
carries them, and it carries them differently depending on which one it is:
`isna` keeps every label because it answers a row per row, `dropna` keeps only
the labels of the rows that survived, and `shift` keeps all of them and moves
the values underneath. Those are three different behaviours that all look like
"the answer has an index" from a distance, so each is checked.

The refusals are tested as carefully as the answers, for the reason that file
gives: a declared parameter that is quietly dropped is the failure that takes
longest to find.
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

HOLED = [1.0, None, 3.0, None, 5.0]
"""One column with gaps in it, which is where every one of these differs.

A column with no missing values would let `ffill`, `bfill`, `dropna`, `isna` and
the four scans all pass while doing nothing, so the corpus for a transformation
has to have holes in it in a way the corpus for a sum does not.
"""

WHOLE = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
"""One column with no gaps, for the ones whose answer a gap would hide."""

PER_COLUMN = [
    "isna",
    "notna",
    "ffill",
    "bfill",
    "shift",
    "diff",
    "pct_change",
    "cumsum",
    "cumprod",
    "cummax",
    "cummin",
]
"""The eleven that mean the same thing to a column and to a frame."""


def like(mine: list[Any], theirs: list[Any]) -> bool:
    """Whether two lists of values agree, counting every missing spelling as one.

    firepanda hands a missing value back as `None` where it came out of a
    validity bit and as a NaN where it came out of a float, and pandas hands
    back a NaN either way. Both mean the row is not there, and this is where
    that reading is written down so the tests below can compare values rather
    than spellings.
    """

    def absent(one: Any) -> bool:
        return one is None or (isinstance(one, float) and math.isnan(one))

    if len(mine) != len(theirs):
        return False
    return all(
        absent(a) == absent(b) and (absent(a) or a == pytest.approx(b))
        for a, b in zip(mine, theirs, strict=True)
    )


@needs_pandas
@pytest.mark.parametrize("name", [*PER_COLUMN, "dropna"])
def test_a_series_transformation_gives_the_pandas_answer(firepanda: ModuleType, name: str) -> None:
    """The whole point, once per transformation."""
    import pandas as pd

    mine = getattr(firepanda.Series(HOLED), name)().tolist()
    theirs = getattr(pd.Series(HOLED), name)().tolist()
    assert like(mine, theirs), f"{name}: {mine} against {theirs}"


@needs_pandas
@pytest.mark.parametrize("name", PER_COLUMN)
def test_a_frame_transformation_runs_down_each_column(firepanda: ModuleType, name: str) -> None:
    """A frame is the columns done one at a time, rather than anything new.

    Worth its own test because the alternative reading is available and wrong.
    `df.cumsum()` could plausibly total across the row, and it does not, and a
    frame with two columns of different heights of missing value is where the
    two readings come apart.
    """
    import pandas as pd

    data = {"a": HOLED, "b": [10.0, 20.0, None, 40.0, 50.0]}
    mine = getattr(firepanda.DataFrame(data), name)()
    theirs = getattr(pd.DataFrame(data), name)()
    for column in ("a", "b"):
        assert like(mine[column].tolist(), theirs[column].tolist()), f"{name}.{column}"


@needs_pandas
@pytest.mark.parametrize("periods", [-2, -1, 0, 1, 2, 9])
def test_the_periods_reach_the_kernel(firepanda: ModuleType, periods: int) -> None:
    """Including zero, both directions, and further than the column is tall.

    A `periods` that is declared and dropped gives the right answer at one and
    the wrong one everywhere else, which is why this is parametrized rather than
    being a single call.
    """
    import pandas as pd

    for name in ("shift", "diff", "pct_change"):
        mine = getattr(firepanda.Series(WHOLE), name)(periods).tolist()
        theirs = getattr(pd.Series(WHOLE), name)(periods).tolist()
        assert like(mine, theirs), f"{name}({periods}): {mine} against {theirs}"


@needs_pandas
@pytest.mark.parametrize("limit", [1, 2, 5])
def test_the_fill_limit_reaches_the_kernel(firepanda: ModuleType, limit: int) -> None:
    """A limit stops a fill part way through a run of missing rows."""
    import pandas as pd

    run = [1.0, None, None, None, 5.0]
    for name in ("ffill", "bfill"):
        mine = getattr(firepanda.Series(run), name)(limit=limit).tolist()
        theirs = getattr(pd.Series(run), name)(limit=limit).tolist()
        assert like(mine, theirs), f"{name}(limit={limit}): {mine} against {theirs}"


@needs_pandas
def test_no_limit_and_a_limit_of_zero_are_not_the_same_thing(firepanda: ModuleType) -> None:
    """The core spells no limit as zero and pandas spells it as None.

    That is one line of translation at the boundary and it is the kind of line
    that gets written the wrong way round, so a caller who writes `limit=0` has
    to be refused rather than handed the answer for no limit at all.
    """
    import pandas as pd

    with pytest.raises(ValueError, match="greater than 0"):
        firepanda.Series(HOLED).ffill(limit=0)
    with pytest.raises(ValueError, match="greater than 0"):
        pd.Series(HOLED).ffill(limit=0)


@needs_pandas
def test_a_shift_widens_a_column_of_whole_numbers(firepanda: ModuleType) -> None:
    """Because the gap it makes has nothing an integer can hold.

    This is the missing value policy arriving somewhere new. Moving the rows of
    a complete int64 column makes room with no value in it, and pandas has no
    integer that means absent, so the column becomes float64 with a NaN in the
    gap. A shift of zero makes no gap and stays whole.
    """
    import pandas as pd

    mine, theirs = firepanda.Series([1, 2, 3]), pd.Series([1, 2, 3])
    assert mine.shift().dtype == str(theirs.shift().dtype) == "float64"
    assert like(mine.shift().tolist(), theirs.shift().tolist())
    assert mine.shift(0).dtype == str(theirs.shift(0).dtype) == "int64"


@needs_pandas
def test_a_scan_steps_over_a_missing_row_rather_than_stopping_at_it(
    firepanda: ModuleType,
) -> None:
    """The gap neither restarts the running total nor poisons it.

    A cumulative sum written in an afternoon does one of those two, so this is
    the case that separates the implementation from the obvious loop.
    """
    import pandas as pd

    mine = firepanda.Series(HOLED).cumsum().tolist()
    theirs = pd.Series(HOLED).cumsum().tolist()
    assert like(mine, theirs)
    assert like(mine, [1.0, None, 4.0, None, 9.0])


@needs_pandas
def test_the_labels_come_through_the_way_pandas_brings_them(firepanda: ModuleType) -> None:
    """Three different behaviours that all look like keeping an index."""
    import pandas as pd

    mine, theirs = firepanda.Series(HOLED), pd.Series(HOLED)
    assert mine.isna().index.tolist() == theirs.isna().index.tolist()
    assert mine.dropna().index.tolist() == theirs.dropna().index.tolist() == [0, 2, 4]
    assert mine.shift().index.tolist() == theirs.shift().index.tolist()


@needs_pandas
def test_a_frame_dropna_removes_rows_and_a_series_dropna_removes_values(
    firepanda: ModuleType,
) -> None:
    """The one name on the list that means two different things.

    A column `dropna` looks at that column. A frame `dropna` looks at every
    column at once and takes whole rows out, so no per column loop produces it,
    which is why it has its own door at the boundary.
    """
    import pandas as pd

    data = {"a": [1.0, None, 3.0], "b": [10.0, 20.0, None]}
    mine, theirs = firepanda.DataFrame(data), pd.DataFrame(data)
    assert len(mine.dropna()) == len(theirs.dropna()) == 1
    assert mine.dropna().index.tolist() == theirs.dropna().index.tolist() == [0]
    assert len(mine["a"].dropna()) == len(theirs["a"].dropna()) == 2


@needs_pandas
def test_a_subset_narrows_which_columns_can_disqualify_a_row(firepanda: ModuleType) -> None:
    """A string and a list of strings both work, the way pandas takes both."""
    import pandas as pd

    data = {"a": [1.0, None, 3.0], "b": [10.0, 20.0, None]}
    mine, theirs = firepanda.DataFrame(data), pd.DataFrame(data)
    for subset in ("b", ["b"], ["a", "b"]):
        assert len(mine.dropna(subset=subset)) == len(theirs.dropna(subset=subset)), subset


@needs_pandas
@pytest.mark.parametrize("values", [[3.0, 1.0, 2.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], HOLED])
def test_monotonic_answers_what_pandas_answers(firepanda: ModuleType, values: list[Any]) -> None:
    """Including that a column with a hole in it is monotonic in neither direction.

    Which is not obvious, since the present values in `HOLED` are increasing. A
    value that is not there cannot be said to be in order with respect to
    anything, and both libraries agree.
    """
    import pandas as pd

    mine, theirs = firepanda.Series(values), pd.Series(values)
    assert mine.is_monotonic_increasing == theirs.is_monotonic_increasing
    assert mine.is_monotonic_decreasing == theirs.is_monotonic_decreasing


def test_an_empty_column_transforms_to_an_empty_column(firepanda: ModuleType) -> None:
    """Rather than raising, which is the easy thing for a loop to do at length zero."""
    for name in ("dropna", "isna", "ffill", "cumsum", "shift"):
        assert getattr(firepanda.Series([]), name)().tolist() == [], name


@pytest.mark.parametrize(
    ("owner", "call", "arguments", "expected"),
    [
        ("Series", "ffill", {"limit_area": "inside"}, "limit_area"),
        ("Series", "shift", {"freq": "D"}, "freq"),
        ("Series", "shift", {"fill_value": 0}, "fill_value"),
        ("Series", "shift", {"periods": [1, 2]}, "single number"),
        ("Series", "shift", {"suffix": "_x"}, "suffix"),
        ("Series", "pct_change", {"fill_method": "pad"}, "fill_method"),
        ("Series", "cumsum", {"skipna": False}, "skipna"),
        ("Series", "dropna", {"ignore_index": True}, "ignore_index"),
        ("DataFrame", "cumsum", {"numeric_only": True}, "numeric_only"),
        ("DataFrame", "dropna", {"how": "all"}, "how"),
        ("DataFrame", "dropna", {"thresh": 1}, "thresh"),
        ("DataFrame", "cumsum", {"axis": 1}, "axis=1"),
    ],
)
def test_a_declared_argument_that_is_not_implemented_refuses(
    firepanda: ModuleType, owner: str, call: str, arguments: dict[str, object], expected: str
) -> None:
    """Every one of them, by name, with the reason in the message."""
    holder = firepanda.Series(HOLED) if owner == "Series" else firepanda.DataFrame({"a": HOLED})
    with pytest.raises(NotImplementedError, match=expected):
        getattr(holder, call)(**arguments)


def test_a_series_has_only_the_one_axis(firepanda: ModuleType) -> None:
    """And says what pandas says, which names the axis and the type."""
    with pytest.raises(ValueError, match="No axis named 1"):
        firepanda.Series(HOLED).cumsum(axis=1)


def test_a_frame_will_not_transform_through_the_dropna_door(firepanda: ModuleType) -> None:
    """The boundary refuses it too, rather than trusting the layer above.

    `DataFrame.dropna` never reaches `transform`, because the generated member
    calls the other helper. This asserts the boundary would refuse it anyway,
    since the two are separated by a convention and a convention that nothing
    checks is a comment.

    Calling `_inner` reaches the extension without going through the Python
    layer, so the tag arrives as written rather than as the `ValueError` a
    caller would see. That is what makes this a test of the boundary and not of
    the translation, which the error tests cover on their own.
    """
    frame = firepanda.DataFrame({"a": HOLED})
    with pytest.raises(Exception, match=r"firepanda:value: .*removes rows"):
        frame._inner.transform("dropna", 0)


def test_an_unknown_transformation_says_so(firepanda: ModuleType) -> None:
    """The name crosses the boundary as a string, so something has to check it.

    Tagged `value` and not `dtype`, which is why the check sits outside the
    handler that turns a kernel complaint about a column into a type error. A
    word nobody implements is a value the caller chose, and the tag has to say
    so or a caller reads it as a complaint about their data.

    Reached through `_inner` for the reason the test above gives, so the tag is
    the untranslated one.
    """
    with pytest.raises(Exception, match="firepanda:value: unknown transformation"):
        firepanda.Series(HOLED)._inner.transform("cumquat", 0)
