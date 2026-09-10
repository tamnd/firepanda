"""Converting a column to another type, checked against a running pandas.

`astype` is one method with two halves. One half is the conversion, which is a
kernel that was already written and tested in Mojo. The other half is deciding
what type the caller asked for, and that half is nearly all of the work, because
pandas resolves a dtype through numpy and numpy answers to about sixty different
spellings of a dozen types. `int64` is also `int`, `long`, `q` and `i8`, and a
program that says `astype("l")` means the same thing as one that says
`astype(numpy.int64)`.

So the sweep below is the important test in this file. It walks the whole name
table rather than a sample of it, and it asks a live pandas what each name means
rather than comparing against a constant, because a constant records what
somebody believed numpy did on the day they wrote it down. The table is imported
from the module under test on purpose: it makes the sweep exhaustive by
construction, so a row added to the table without being measured fails here.

The refusals get the same treatment. A type pandas has and firepanda does not
has to say so by name, and the ones that are refused despite firepanda having
the type are the rows worth staring at: a cast to `datetime64[ns]` would hand
back the integers underneath, which is a wrong answer that looks right, and it
is refused rather than approximated.
"""

from __future__ import annotations

import importlib.util
import warnings
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

WHOLE = [1, 2, 3]
"""Three small whole numbers, which every numeric type can hold exactly.

The values are deliberately boring. A sweep over sixty type names is measuring
the names, and a value that overflows a uint8 or loses precision in a float16
would fail rows for a reason that has nothing to do with the name being read
correctly.
"""

TEXT = ["1", "2", "3"]
"""The same three, written down, for the conversions that go through text."""


def names(firepanda: ModuleType) -> dict[str, str]:
    """The spelling table, out of the module the test is about."""
    return dict(firepanda._pandas._DTYPE_NAMES)


@needs_pandas
def test_every_spelling_means_what_pandas_says_it_means(firepanda: ModuleType) -> None:
    """The sweep, over the whole table, against a live pandas.

    Three names are excused. `category` is skipped outright, for the reason in
    the loop, and two are checked here and then excused from the pandas half of
    the
    comparison. `str` and `string` both name text, and pandas 3 prints them as
    `str` and `string` where firepanda prints `string` for both, since firepanda
    has one text type and pandas has two spellings of one. That is a difference
    in what the dtype is called and not in what the column holds, and the values
    are compared below to show it.
    """
    import pandas as pd

    wrong: list[str] = []
    for name, want in names(firepanda).items():
        # The one name in the table that is not a layout, so the boring integer
        # values the rest of the sweep uses are the wrong input for it: firepanda
        # can only encode text. It has its own tests below.
        if name == "category":
            continue
        mine = firepanda.Series(WHOLE).astype(name).dtype
        if mine != want:
            wrong.append(f"{name}: firepanda gave {mine} and the table says {want}")
            continue
        theirs = str(pd.Series(WHOLE).astype(name).dtype)
        if name in {"str", "string"}:
            continue
        if theirs != mine:
            wrong.append(f"{name}: firepanda gave {mine} and pandas gave {theirs}")
    assert not wrong, "\n".join(wrong)


@needs_pandas
def test_text_is_one_type_here_and_two_spellings_there(firepanda: ModuleType) -> None:
    """Both of pandas' text names land on the one text column, with the values."""
    import pandas as pd

    for name in ("str", "string"):
        mine = firepanda.Series(WHOLE).astype(name)
        theirs = pd.Series(WHOLE).astype(name)
        assert mine.dtype == "string"
        assert mine.tolist() == theirs.tolist()


@needs_pandas
@pytest.mark.parametrize(
    ("values", "name"),
    [
        (WHOLE, "float64"),
        (WHOLE, "int8"),
        (WHOLE, "bool"),
        (WHOLE, "str"),
        (TEXT, "int64"),
        (TEXT, "float64"),
        ([1.7, -1.7, 0.0], "int64"),
        ([True, False, True], "int64"),
        ([1, 0, 2], "bool"),
    ],
)
def test_the_values_come_out_the_way_pandas_converts_them(
    firepanda: ModuleType, values: list[Any], name: str
) -> None:
    """The conversions themselves, one pair of types at a time.

    The float to integer row is the one with a decision in it. Truncating toward
    zero and rounding to nearest disagree on every value with a fraction, and
    `-1.7` is in the list because truncation gives `-1` and rounding gives `-2`,
    so a row that only had positive numbers in it would pass either way.
    """
    import pandas as pd

    mine = firepanda.Series(values).astype(name).tolist()
    theirs = pd.Series(values).astype(name).tolist()
    assert mine == theirs


@needs_pandas
def test_a_text_column_becomes_a_category_the_way_pandas_makes_one(
    firepanda: ModuleType,
) -> None:
    """The dtype name and the values, against a live pandas."""
    import pandas as pd

    words = ["rivet", "bolt", "rivet", "anchor"]
    mine = firepanda.Series(words).astype("category")
    theirs = pd.Series(words).astype("category")
    assert mine.dtype == "category"
    assert str(theirs.dtype) == "category"
    assert mine.astype("str").tolist() == list(theirs.astype("str"))


@needs_pandas
def test_the_categories_are_sorted_and_not_in_the_order_they_appeared(
    firepanda: ModuleType,
) -> None:
    """First appearance would be rivet, bolt, anchor, and pandas sorts.

    Asserted through pyarrow rather than through a `.cat` namespace firepanda
    does not have yet, which is the only way to see the categories from Python
    today and is also the way a consumer would see them.
    """
    pa = pytest.importorskip("pyarrow")
    import pandas as pd

    words = ["rivet", "bolt", "rivet", "anchor"]
    mine = pa.table(firepanda.DataFrame({"part": words}).astype({"part": "category"}))
    assert mine.column("part").chunk(0).dictionary.to_pylist() == ["anchor", "bolt", "rivet"]
    assert list(pd.Series(words).astype("category").cat.categories) == [
        "anchor",
        "bolt",
        "rivet",
    ]


@needs_pandas
def test_a_missing_value_is_not_one_of_the_categories(firepanda: ModuleType) -> None:
    """A null stays a null and does not become a category of its own."""
    pa = pytest.importorskip("pyarrow")
    import pandas as pd

    words = ["rivet", None, "bolt", None]
    mine = pa.table(firepanda.DataFrame({"part": words}).astype({"part": "category"}))
    assert mine.column("part").chunk(0).dictionary.to_pylist() == ["bolt", "rivet"]
    assert mine.column("part").to_pylist() == words
    assert list(pd.Series(words).astype("category").cat.categories) == ["bolt", "rivet"]


def test_a_category_cast_back_gives_the_values_and_not_the_codes(
    firepanda: ModuleType,
) -> None:
    """The codes here are 1, 0, 1 and the values are 20, 10, 20."""
    made = firepanda.Series(["20", "10", "20"]).astype("category")
    assert made.astype("int64").tolist() == [20, 10, 20]
    assert made.astype("str").tolist() == ["20", "10", "20"]


def test_a_column_that_is_not_text_cannot_be_encoded_yet(firepanda: ModuleType) -> None:
    """Pandas does this, so it is a gap and says so rather than a type error."""
    with pytest.raises(NotImplementedError, match="not supported"):
        firepanda.Series(WHOLE).astype("category")
    with pytest.raises(NotImplementedError, match="not supported"):
        firepanda.DataFrame({"n": WHOLE}).astype({"n": "category"})


def test_a_python_type_is_a_dtype_too(firepanda: ModuleType) -> None:
    """`astype(float)` is what a lot of code writes, and pandas takes it."""
    assert firepanda.Series(WHOLE).astype(float).dtype == "float64"
    assert firepanda.Series(WHOLE).astype(int).dtype == "int64"
    assert firepanda.Series(WHOLE).astype(bool).dtype == "bool"
    assert firepanda.Series(WHOLE).astype(str).dtype == "string"


def test_a_numpy_dtype_object_is_read_off_its_name(firepanda: ModuleType) -> None:
    """Without importing numpy, which is the point of reading it by duck typing.

    A stand in with a `name` attribute goes down the same path a real numpy
    dtype does. Using a stand in rather than numpy keeps the test running on a
    machine that has no numpy, and firepanda has no numpy dependency to lose.
    """

    class Pretend:
        name = "float32"

    assert firepanda.Series(WHOLE).astype(Pretend()).dtype == "float32"


@needs_pandas
def test_a_name_nothing_answers_to_is_a_type_error(firepanda: ModuleType) -> None:
    """With the pandas message, which is a `TypeError` despite reading like one."""
    import pandas as pd

    with pytest.raises(TypeError) as theirs:
        pd.Series(WHOLE).astype("nonsense")
    with pytest.raises(TypeError) as mine:
        firepanda.Series(WHOLE).astype("nonsense")
    assert str(mine.value) == str(theirs.value)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Int64", "nullable"),
        ("UInt8", "nullable"),
        ("Float32", "nullable"),
        ("boolean", "nullable"),
        ("object", "no type that holds anything"),
        ("O", "no type that holds anything"),
        ("U", "no type that holds anything"),
        ("unicode", "no type that holds anything"),
        ("str_", "no type that holds anything"),
        ("datetime64[ns]", "column of counts"),
        ("timedelta64[ns]", "column of counts"),
        ("date32[day]", "day numbers"),
        ("binary", "column of text"),
        ("complex128", "no complex column"),
        ("period[D]", "no period column"),
        ("interval", "no interval column"),
        ("S21", "fixed width byte string"),
        ("bytes", "fixed width byte string"),
        ("void", "no void column"),
        ("longdouble", "extended precision"),
        ("g", "extended precision"),
        (">i8", "big endian"),
    ],
)
def test_a_type_firepanda_does_not_have_says_so_by_name(
    firepanda: ModuleType, name: str, expected: str
) -> None:
    """Every refusal, with the reason in the message.

    A refusal that says only that something failed is worth very little, since
    the caller already knew that. What makes these worth the lines they cost is
    that each one says which type was asked for and why there is no column for
    it, so a reader can tell a missing feature from a mistake.
    """
    with pytest.raises(NotImplementedError, match=expected):
        firepanda.Series(WHOLE).astype(name)


def test_the_two_marks_that_mean_native_are_dropped(firepanda: ModuleType) -> None:
    """Arrow is little endian, so `<i8`, `=i8` and `|i1` are the plain types."""
    assert firepanda.Series(WHOLE).astype("<i8").dtype == "int64"
    assert firepanda.Series(WHOLE).astype("=i8").dtype == "int64"
    assert firepanda.Series(WHOLE).astype("|i1").dtype == "int8"


@needs_pandas
def test_a_value_that_will_not_convert_raises(firepanda: ModuleType) -> None:
    """A `ValueError` in both, saying the same sentence, plus the row here.

    The class is the part that matters, because `except ValueError` around a
    cast is ordinary code and a `TypeError` walks straight past it. The value
    was the wrong value and not the wrong type, which is the distinction Python
    draws and the one pandas draws here.

    The message is the pandas one word for word up to the row number, so that a
    program matching on it keeps working. The row number is ours: pandas says
    which value would not read and not where it was, and on a column of any size
    that is the first thing a person then has to go and find out.
    """
    import pandas as pd

    for series in (firepanda.Series(["1", "x"]), pd.Series(["1", "x"])):
        with pytest.raises(ValueError, match=r"invalid literal for int\(\)"):
            series.astype("int64")

    with pytest.raises(ValueError, match="at row 1"):
        firepanda.Series(["1", "x"]).astype("int64")


@needs_pandas
def test_a_value_that_will_not_read_as_a_float_says_float(
    firepanda: ModuleType,
) -> None:
    """A different sentence for a different target, and again the pandas one."""
    import pandas as pd

    for series in (firepanda.Series(["1", "x"]), pd.Series(["1", "x"])):
        with pytest.raises(ValueError, match="could not convert string to float"):
            series.astype("float64")


@needs_pandas
@pytest.mark.parametrize(
    "values",
    [
        pytest.param([1.0, float("nan")], id="nan"),
        pytest.param([1.0, float("inf")], id="inf"),
        pytest.param([1.0, float("-inf")], id="negative-inf"),
        pytest.param([1.0, None], id="missing"),
    ],
)
def test_nothing_a_float_holds_and_an_integer_cannot_gets_through(
    firepanda: ModuleType, values: list[float | None]
) -> None:
    """The three shapes of the same refusal, which pandas has a class for.

    Before this, firepanda handed back a null in an integer column for the first
    and the last of these and the largest int64 there is for the two infinities.
    The null is the worse of the two, because nothing failed and the caller ended
    up holding a column pandas could not have made.
    """
    import pandas as pd

    for series in (firepanda.Series(values), pd.Series(values)):
        with pytest.raises(ValueError, match="Cannot convert non-finite values"):
            series.astype("int64")


@needs_pandas
def test_the_refusal_has_the_name_pandas_gives_it(firepanda: ModuleType) -> None:
    """`IntCastingNaNError`, caught by name and by the broad class both.

    pandas has a class with this exact name in `pandas.errors` and a program
    that catches it is asking a specific question. It is a `ValueError` in both
    libraries, so the broad catch keeps working either way.
    """
    import pandas as pd

    with pytest.raises(pd.errors.IntCastingNaNError):
        pd.Series([1.0, float("nan")]).astype("int64")
    with pytest.raises(firepanda.errors.IntCastingNaNError):
        firepanda.Series([1.0, float("nan")]).astype("int64")

    assert issubclass(firepanda.errors.IntCastingNaNError, ValueError)


@needs_pandas
def test_an_integer_column_holding_a_null_is_refused_too(
    firepanda: ModuleType,
) -> None:
    """A door pandas does not have, answered the way pandas would have answered.

    `firepanda.Series([1, None, 3])` is an int64 column with a null in it, where
    the pandas one is a float64 column with a NaN, so a caller here can ask to
    convert an integer column that already holds a missing value. There is
    nowhere for it to go, and the column pandas would have had is the one it
    refuses, so this refuses it too.
    """
    with pytest.raises(firepanda.errors.IntCastingNaNError):
        firepanda.Series([1, None, 3]).astype("int64")


@needs_pandas
def test_a_column_with_room_for_the_value_is_not_refused(
    firepanda: ModuleType,
) -> None:
    """The check is about integers and asks nothing of any other target."""
    import math

    values = [1.0, float("nan"), float("inf")]
    assert math.isnan(firepanda.Series(values).astype("float32").tolist()[1])
    assert firepanda.Series(values).astype("string").tolist()[0] == "1.0"


@needs_pandas
def test_a_frame_refuses_the_same_column_the_same_way(firepanda: ModuleType) -> None:
    """The check sits under both surfaces rather than under the Series alone."""
    import pandas as pd

    values = {"a": [1.0, float("nan")], "b": [1.0, 2.0]}
    for frame in (firepanda.DataFrame(values), pd.DataFrame(values)):
        with pytest.raises(ValueError, match="Cannot convert non-finite values"):
            frame.astype({"a": "int64"})
        # The clean column on its own goes through, so it is the values that
        # decided and not the frame having a bad column somewhere in it.
        assert frame.astype({"b": "int64"})["b"].tolist() == [1, 2]


@needs_pandas
def test_errors_ignore_hands_the_column_back_unchanged(firepanda: ModuleType) -> None:
    """Which is what it means, rather than filling the bad row in with nothing.

    Worth saying out loud because the kernel underneath has a flag that does the
    other thing, and turning a value that will not convert into a missing one is
    a reasonable behaviour that pandas does not have. `errors="ignore"` is read
    in the Python layer and never reaches that flag.
    """
    import pandas as pd

    mine = firepanda.Series(["1", "x"]).astype("int64", errors="ignore")
    theirs = pd.Series(["1", "x"]).astype("int64", errors="ignore")
    assert mine.tolist() == theirs.tolist() == ["1", "x"]
    assert mine.dtype == "string"


@needs_pandas
def test_errors_ignore_covers_the_non_finite_refusal_as_well(
    firepanda: ModuleType,
) -> None:
    """It is a `ValueError`, and `errors="ignore"` swallows every one of those.

    Worth its own test because this refusal is raised before the conversion
    starts rather than by the conversion failing, and a check in the wrong place
    would have escaped the handler that reads the keyword.
    """
    import pandas as pd

    values = [1.0, float("nan")]
    mine = firepanda.Series(values).astype("int64", errors="ignore")
    theirs = pd.Series(values).astype("int64", errors="ignore")
    assert mine.dtype == "float64" == str(theirs.dtype)
    assert mine.tolist()[0] == theirs.tolist()[0] == 1.0


@needs_pandas
def test_errors_ignore_does_not_excuse_a_type_that_does_not_exist(
    firepanda: ModuleType,
) -> None:
    """The keyword is about values, and pandas raises on the name either way."""
    import pandas as pd

    with pytest.raises(TypeError):
        firepanda.Series(WHOLE).astype("nonsense", errors="ignore")
    with pytest.raises(TypeError):
        pd.Series(WHOLE).astype("nonsense", errors="ignore")


@needs_pandas
def test_a_third_word_for_errors_is_refused_the_way_pandas_refuses_it(
    firepanda: ModuleType,
) -> None:
    """Same message, down to the quotes around the value."""
    import pandas as pd

    with pytest.raises(ValueError) as theirs:
        pd.Series(WHOLE).astype("int64", errors="nope")
    with pytest.raises(ValueError) as mine:
        firepanda.Series(WHOLE).astype("int64", errors="nope")
    assert str(mine.value) == str(theirs.value)


def test_the_errors_keyword_is_read_before_the_type_is(firepanda: ModuleType) -> None:
    """Both wrong at once answers about the keyword, which is what pandas does."""
    with pytest.raises(ValueError, match="errors"):
        firepanda.Series(WHOLE).astype("nonsense", errors="nope")


def test_copy_warns_and_is_otherwise_ignored(firepanda: ModuleType) -> None:
    """It is deprecated in pandas 3 and does nothing there or here.

    Refusing it would have been the other choice and it is the wrong one, since
    code that passes `copy=False` gets no complaint from pandas and would get
    one here for a keyword that changes nothing.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        answer = firepanda.Series(WHOLE).astype("float64", copy=False)
    assert answer.dtype == "float64"
    assert [one.category for one in caught] == [DeprecationWarning]
    assert "copy keyword is deprecated" in str(caught[0].message)


def test_not_passing_copy_says_nothing(firepanda: ModuleType) -> None:
    """The warning is about the keyword arriving, not about the method running."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        firepanda.Series(WHOLE).astype("float64")
    assert caught == []


@needs_pandas
def test_a_frame_converts_every_column_when_it_is_given_one_type(
    firepanda: ModuleType,
) -> None:
    """One name for the whole frame, which is the shorter of the two forms."""
    import pandas as pd

    data = {"a": [1, 2], "b": [3, 4]}
    mine = firepanda.DataFrame(data).astype("float32")
    theirs = pd.DataFrame(data).astype("float32")
    assert [mine[one].dtype for one in ("a", "b")] == [str(one) for one in theirs.dtypes]


@needs_pandas
def test_a_dict_converts_the_columns_it_names_and_leaves_the_rest(
    firepanda: ModuleType,
) -> None:
    """The longer form, which is the one worth having."""
    import pandas as pd

    data = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
    mine = firepanda.DataFrame(data).astype({"a": "float32", "c": "str"})
    theirs = pd.DataFrame(data).astype({"a": "float32", "c": "str"})
    assert mine["a"].dtype == str(theirs["a"].dtype) == "float32"
    assert mine["b"].dtype == str(theirs["b"].dtype) == "int64"
    assert mine["c"].dtype == "string"
    assert mine["c"].tolist() == theirs["c"].tolist()


@needs_pandas
def test_a_key_that_is_not_a_column_says_so_before_anything_converts(
    firepanda: ModuleType,
) -> None:
    """With the pandas message, and with the frame left alone.

    The second half is the part a caller notices. Converting the columns that
    were named and then failing on the one that was not would leave the caller
    holding a frame that is half converted, and since the answer is a new frame
    the original is untouched either way, so this asserts the original rather
    than the answer.
    """
    import pandas as pd

    frame = firepanda.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(KeyError) as mine:
        frame.astype({"a": "float32", "missing": "int64"})
    with pytest.raises(KeyError) as theirs:
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).astype({"a": "float32", "missing": "int64"})
    assert str(mine.value) == str(theirs.value)
    assert frame["a"].dtype == "int64"


@needs_pandas
def test_the_dtype_argument_to_a_constructor_is_the_same_cast(
    firepanda: ModuleType,
) -> None:
    """Which is what it is in pandas, and the reason it is written that way here.

    Inference then conversion is one pass more than reading the values straight
    into the asked for type, and it is the reading that decides what a value
    means, so the two agree on every answer. If they ever stop agreeing, this is
    the test that says so.
    """
    import pandas as pd

    mine = firepanda.Series(TEXT, dtype="int64")
    assert mine.dtype == "int64"
    assert mine.tolist() == pd.Series(TEXT, dtype="int64").tolist()
    assert mine.tolist() == firepanda.Series(TEXT).astype("int64").tolist()


@needs_pandas
def test_a_frame_constructor_takes_a_dtype_for_every_column(firepanda: ModuleType) -> None:
    """The frame half of the same thing. pandas takes one name, not a dict."""
    import pandas as pd

    data = {"a": [1, 2], "b": [3, 4]}
    mine = firepanda.DataFrame(data, dtype="float32")
    theirs = pd.DataFrame(data, dtype="float32")
    assert [mine[one].dtype for one in ("a", "b")] == [str(one) for one in theirs.dtypes]


def test_a_constructor_dtype_that_does_not_exist_refuses(firepanda: ModuleType) -> None:
    """The same resolution, so the same message, rather than a second table."""
    with pytest.raises(TypeError, match="not understood"):
        firepanda.Series(WHOLE, dtype="nonsense")
    with pytest.raises(NotImplementedError, match="nullable"):
        firepanda.DataFrame({"a": WHOLE}, dtype="Int64")


def test_a_cast_to_the_type_it_already_is_still_answers_a_new_column(
    firepanda: ModuleType,
) -> None:
    """pandas copies here too, and a caller who mutates the answer relies on it."""
    column = firepanda.Series(WHOLE)
    answer = column.astype("int64")
    assert answer is not column
    assert answer.tolist() == column.tolist()


def test_an_empty_column_converts_to_an_empty_column(firepanda: ModuleType) -> None:
    """Nothing to convert is not the same as nothing to do."""
    answer = firepanda.Series([]).astype("float64")
    assert answer.dtype == "float64"
    assert answer.tolist() == []


@needs_pandas
def test_text_to_bool_reads_the_text_here_and_does_not_there(firepanda: ModuleType) -> None:
    """The one divergence in this slice, written down rather than left to be found.

    pandas asks numpy whether each string is truthy, so `"x"` and `"0"` and
    `"false"` are all `True` and only the empty string is `False`. Nothing is
    parsed and no input is ever rejected, which means `astype("bool")` on a
    column of text is an expensive way of asking whether each value is non
    empty. firepanda parses instead, and a value that is not a boolean is an
    error rather than a `True`.

    This is a deliberate divergence and not an oversight. It is recorded here so
    that a future change of mind is a change to a failing test rather than to
    nothing at all.
    """
    import pandas as pd

    assert pd.Series(["x", "0", ""]).astype("bool").tolist() == [True, True, False]
    with pytest.raises(ValueError, match="not a bool"):
        firepanda.Series(["x", "0", ""]).astype("bool")
