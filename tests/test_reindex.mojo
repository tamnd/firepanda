"""Tests for putting a frame onto a set of labels it may not have.

Two halves that share a name and share very little else. The row half is a
lookup in the index and a gather, so its tests are about what happens to a label
that is not there: which row comes back, what type the column has afterwards and
what a fill value changes about both. The column half is a lookup by name in the
schema, so its tests are about ordering and about the column that has to be made
out of nothing.

The widening is the part worth writing down. An integer column that gains a
missing row comes back as a float column, which is a surprising answer until you
remember that it is the answer pandas gives and for the same reason. Every test
here that checks a dtype is checking that rule and not the gather.
"""

from std.math import isnan
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.array.strings import strings_from_list
from firepanda.array.value import Value
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series


def _labels(values: List[Int64]) raises -> AnyArray:
    """An int64 column of labels, all present."""
    return AnyArray(from_list[DType.int64](values))


def _keyed() raises -> DataFrame:
    """A frame of three rows labelled ten, twenty and thirty.

    The labels are not the row numbers on purpose. A test whose labels happen to
    be zero, one and two cannot tell a lookup from a position and would pass
    against a library that ignored the index entirely.

    Returns:
        The frame, with an int64 column, a float64 column and a string column.
    """
    var count = Array[DType.int64](3)
    var size = Array[DType.float64](3)
    for i in range(3):
        count.set_valid(i, Int64(i + 1))
        size.set_valid(i, Float64(i) + 0.5)
    var series = List[Series]()
    series.append(Series("key", _labels([Int64(10), 20, 30])))
    series.append(Series("count", count^))
    series.append(Series("size", size^))
    series.append(
        Series("word", AnyArray(strings_from_list(["red", "green", "blue"])))
    )
    return DataFrame.from_series(series^).set_index("key")


def _numbers(hole: Bool = False) raises -> DataFrame:
    """A frame of nothing but numbers, labelled ten, twenty and thirty.

    The fill value tests want a frame with no words in it, because a fill that
    suits a number does not suit a string and the pair is refused rather than
    half filled.

    Args:
        hole: Whether to leave the middle row of `count` missing, which is how
            the tests tell a hole the frame arrived with from one a lookup made.

    Returns:
        The frame.
    """
    var count = Array[DType.int64](3)
    for i in range(3):
        count.set_valid(i, Int64(i + 1))
    if hole:
        count.set_null(1)
    var series = List[Series]()
    series.append(Series("key", _labels([Int64(10), 20, 30])))
    series.append(Series("count", count^))
    return DataFrame.from_series(series^).set_index("key")


def _column(df: DataFrame, name: String) raises -> AnyArray:
    """Takes one column out of a frame, by name and as a column."""
    return AnyArray(copy=df.column(name).values)


def _counts_of(df: DataFrame) raises -> List[Int]:
    """Reads the `count` column, with a missing row written as minus one.

    The column arrives as an integer one or as a float one depending on whether
    anything went missing, and both are read here, because half these tests are
    about which of the two came back and the other half do not care.
    """
    var column = _column(df, "count")
    var out = List[Int](capacity=len(column))
    for i in range(len(column)):
        if not column.is_valid(i):
            out.append(-1)
        elif column.dtype() == DType.float64:
            var number = column.as_typed_view[DType.float64]()[i]
            # A widened column carries its missing rows as NaN and drops the
            # bitmap, so the check above sees nothing and this is where the
            # holes in an integer column that gained one turn up.
            out.append(-1 if isnan(number) else Int(number))
        else:
            out.append(Int(column.as_typed_view[DType.int64]()[i]))
    return out^


def _label_list(df: DataFrame) raises -> List[Int]:
    """Reads the row labels of a result as a list of numbers."""
    var found = df.index.materialize().as_typed[DType.int64]()
    var out = List[Int](capacity=len(found))
    for i in range(len(found)):
        out.append(Int(found[i]))
    return out^


def test_a_label_the_frame_has_brings_its_own_row() raises:
    var got = _keyed().reindex(_labels([Int64(30), 10]))
    assert_equal(len(got), 2, "one row per label")
    assert_equal(_counts_of(got), [3, 1], "the rows those labels sit in")
    assert_equal(_label_list(got), [30, 10], "the labels that were asked for")


def test_a_label_the_frame_does_not_have_gives_a_missing_row() raises:
    var got = _keyed().reindex(_labels([Int64(10), 99]))
    assert_equal(_counts_of(got), [1, -1], "the second row came from nowhere")
    var word = _column(got, "word")
    assert_true(word.is_valid(0), "the first word is there")
    assert_false(word.is_valid(1), "and a string goes missing the same way")


def test_an_integer_column_widens_when_a_row_goes_missing() raises:
    """The rule that makes this operation change a type at all.

    pandas has one missing value for a number and it is NaN, so a column that
    cannot hold a NaN has to become one that can.
    """
    var whole = _keyed().reindex(_labels([Int64(10), 20]))
    assert_true(
        _column(whole, "count").dtype() == DType.int64,
        "nothing went missing, so nothing widened",
    )
    var holed = _keyed().reindex(_labels([Int64(10), 99]))
    assert_true(
        _column(holed, "count").dtype() == DType.float64,
        "a hole in an integer column widens it",
    )


def test_a_fill_value_keeps_the_column_as_it_was() raises:
    var got = _numbers().reindex(_labels([Int64(20), 99]), Value(Int64(0)))
    assert_true(
        _column(got, "count").dtype() == DType.int64,
        "nothing is missing, so there is nothing to widen for",
    )
    assert_equal(_counts_of(got), [2, 0], "the row that was not there is zero")


def test_a_fill_value_leaves_a_hole_that_was_already_there_alone() raises:
    """What makes this a gather and not a fill.

    `fill_value` is about a row the frame does not have and not about a value
    the frame does not have, and a fill written as a pass over the answer would
    have filled both. pandas fills only the first, which is why the fill goes in
    as a row of its own that the missing labels point at.
    """
    var got = _numbers(hole=True).reindex(
        _labels([Int64(10), 20, 99]), Value(Int64(7))
    )
    assert_equal(
        _counts_of(got), [1, -1, 7], "only the row from nowhere was filled"
    )


def test_a_label_asked_for_twice_brings_its_row_twice() raises:
    var got = _keyed().reindex(_labels([Int64(20), 20]))
    assert_equal(_counts_of(got), [2, 2], "the same row both times")
    assert_equal(_label_list(got), [20, 20], "and the label twice as well")


def test_asking_for_no_labels_gives_a_frame_of_no_rows() raises:
    """The case that must not reach the lookup.

    A list with no values in it has no type in it either, so asking an int64
    index where an empty column's labels sit is a question about two dtypes that
    nobody asked.
    """
    var empty = AnyArray(from_list[DType.float64](List[Float64]()))
    var got = _keyed().reindex(empty)
    assert_equal(len(got), 0, "no rows")
    assert_equal(got.width(), 3, "and the columns are all still there")


def test_the_labels_keep_the_name_the_index_had() raises:
    var got = _keyed().reindex(_labels([Int64(30)]))
    assert_true(Bool(got.index.name), "the index is still named")
    assert_equal(got.index.name.value(), "key", "and named what it was")


def test_a_repeated_label_in_the_frame_is_refused() raises:
    """A label sitting in two rows has no single row to answer with."""
    var count = Array[DType.int64](3)
    for i in range(3):
        count.set_valid(i, Int64(i + 1))
    var series = List[Series]()
    series.append(Series("key", _labels([Int64(10), 10, 30])))
    series.append(Series("count", count^))
    var df = DataFrame.from_series(series^).set_index("key")

    with assert_raises(contains="unique"):
        _ = df.reindex(_labels([Int64(10)]))


def test_a_word_cannot_fill_a_column_of_numbers() raises:
    """The one mismatch that would otherwise be silent.

    A number asked for its bytes raises. A word asked for its number does not:
    it reads the store's integer field, which for a word is a zero, so the frame
    would come back holding zeros nobody wrote.
    """
    with assert_raises(contains="fill_value"):
        _ = _keyed().reindex(_labels([Int64(99)]), Value(String("x")))


def test_a_word_fills_a_column_of_words() raises:
    var count = Array[DType.int64](2)
    for i in range(2):
        count.set_valid(i, Int64(i))
    var series = List[Series]()
    series.append(Series("key", _labels([Int64(10), 20])))
    series.append(Series("word", AnyArray(strings_from_list(["red", "green"]))))
    var df = DataFrame.from_series(series^).set_index("key")

    var got = df.reindex(_labels([Int64(20), 99]), Value(String("none")))
    var word = _column(got, "word")
    ref text = word.strings()
    assert_equal(
        String(StringSlice(unsafe_from_utf8=text.unsafe_bytes(0))),
        "green",
        "the row that was there",
    )
    assert_equal(
        String(StringSlice(unsafe_from_utf8=text.unsafe_bytes(1))),
        "none",
        "and the row that was not",
    )


def test_a_fill_is_only_refused_when_there_is_a_row_for_it() raises:
    """The fill is checked against the columns on the way past and not first.

    That is pandas' order, so a whole number offered to a frame with a column of
    words in it is a mistake only when some label was missing, and is never
    looked at otherwise.
    """
    var whole = _keyed().reindex(_labels([Int64(30), 10]), Value(Int64(0)))
    assert_equal(_counts_of(whole), [3, 1], "nothing was filled, nothing said")
    with assert_raises(contains="fill_value"):
        _ = _keyed().reindex(_labels([Int64(30), 99]), Value(Int64(0)))


def test_a_frame_labelled_by_position_is_looked_up_by_arithmetic() raises:
    """The range fast path, which is a separate implementation of the lookup."""
    var count = Array[DType.int64](3)
    for i in range(3):
        count.set_valid(i, Int64(i + 1))
    var series = List[Series]()
    series.append(Series("count", count^))
    var df = DataFrame.from_series(series^)

    var got = df.reindex(_labels([Int64(2), 0, 7]))
    assert_equal(_counts_of(got), [3, 1, -1], "two rows and one from nowhere")
    assert_equal(_label_list(got), [2, 0, 7], "labelled as asked")


def test_the_columns_come_back_in_the_order_they_were_asked_for() raises:
    var got = _keyed().reindex_columns(["word", "count"])
    assert_equal(got.names(), ["word", "count"], "that order and no other")
    assert_equal(len(got), 3, "every row is still there")
    assert_equal(_counts_of(got), [1, 2, 3], "with what it held")


def test_leaving_a_column_out_is_how_a_column_is_dropped() raises:
    var got = _keyed().reindex_columns(["count"])
    assert_equal(got.width(), 1, "one column asked for is one column back")


def test_a_column_that_is_not_there_is_made_out_of_nothing() raises:
    var got = _keyed().reindex_columns(["count", "missing"])
    var made = _column(got, "missing")
    assert_equal(len(made), 3, "as tall as the frame")
    assert_true(
        made.dtype() == DType.float64,
        "float64, because a missing column says nothing about its own type",
    )
    assert_equal(made.null_count(), 3, "and every row of it is missing")


def test_a_made_up_column_takes_the_fill_values_type() raises:
    """Why the fill is read for its type and not only for its value.

    Asking for a column that is not there and filling it with a whole number
    gives an integer column in pandas, so the type of the answer depends on how
    the fill was written.
    """
    var got = _keyed().reindex_columns(["missing"], Value(Int64(7)))
    var made = _column(got, "missing")
    assert_true(
        made.dtype() == DType.int64, "an integer fill, an integer column"
    )
    assert_equal(made.null_count(), 0, "nothing missing in it")
    ref numbers = made.as_typed_view[DType.int64]()
    assert_equal(Int(numbers[0]), 7, "the value that was asked for")
    assert_equal(Int(numbers[2]), 7, "in every row")


def test_a_column_asked_for_twice_is_refused() raises:
    """Two columns under one name would leave the second unaddressable."""
    with assert_raises(contains="twice"):
        _ = _keyed().reindex_columns(["count", "count"])


def test_the_row_labels_survive_a_reindex_of_the_columns() raises:
    var got = _keyed().reindex_columns(["size"])
    assert_equal(_label_list(got), [10, 20, 30], "the labels are untouched")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
