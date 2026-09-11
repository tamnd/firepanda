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

`Series.reindex` and `Index.reindex` are at the bottom of the file. The first is
the row half with one column in it and the tests for it are the same tests, run
again because a second copy of a rule is a second place to get it wrong. The
second is a stranger thing: an index carries no data, so reindexing one moves
nothing and the answer is the target plus the lookup, and every test for it is
about the lookup or about whose name the result ends up under.
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
from firepanda.frame.index import Index
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
    # A NaN in the values rather than a cleared bit in a bitmap, which is how
    # the row half spells a missing number as well. pandas has one missing
    # value for a number and this is it, so a column made out of nothing is
    # made out of the same nothing pandas would have read.
    assert_equal(made.null_count(), 0, "and it says so in its values")
    ref values = made.as_typed_view[DType.float64]()
    for i in range(3):
        assert_true(isnan(values[i]), "every row of it is missing")


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


def _counted(hole: Bool = False) raises -> Series:
    """A series of three whole numbers labelled ten, twenty and thirty.

    Args:
        hole: Whether to leave the middle row missing, which is how the fill
            value tests tell a hole the series arrived with from one the lookup
            made.

    Returns:
        The series.
    """
    var values = Array[DType.int64](3)
    for i in range(3):
        values.set_valid(i, Int64(i + 1))
    if hole:
        values.set_null(1)
    var out = Series("count", values^)
    out.index = Index(_labels([Int64(10), 20, 30]), String("key"))
    return out^


def _numbers_of(series: Series) raises -> List[Int]:
    """Reads a series as whole numbers, with a missing row written as minus one.

    The same two readings `_counts_of` does above, for the same reason: the
    column comes back integer or float depending on whether the lookup missed,
    and a widened one carries its holes as NaN rather than in a bitmap.
    """
    var out = List[Int](capacity=len(series))
    for i in range(len(series)):
        if not series.is_valid(i):
            out.append(-1)
        elif series.dtype() == DType.float64:
            var number = series.values.as_typed_view[DType.float64]()[i]
            out.append(-1 if isnan(number) else Int(number))
        else:
            out.append(Int(series.values.as_typed_view[DType.int64]()[i]))
    return out^


def _labels_of(index: Index) raises -> List[Int]:
    """Reads an index as a list of whole numbers."""
    var found = index.materialize().as_typed[DType.int64]()
    var out = List[Int](capacity=len(found))
    for i in range(len(found)):
        out.append(Int(found[i]))
    return out^


def test_a_series_comes_back_on_the_labels_it_was_given() raises:
    var got = _counted().reindex(_labels([Int64(30), 10]))
    assert_equal(_numbers_of(got), [3, 1], "the rows those labels sit in")
    assert_equal(_labels_of(got.index), [30, 10], "in the order asked for")
    assert_equal(got.name, "count", "and under the name it already had")


def test_a_label_the_series_does_not_have_is_a_missing_row() raises:
    var got = _counted().reindex(_labels([Int64(10), 99]))
    assert_equal(_numbers_of(got), [1, -1], "the second row came from nowhere")
    assert_equal(
        got.dtype(), DType.float64, "and the column widened to hold it"
    )


def test_a_series_that_loses_nothing_keeps_its_type() raises:
    var got = _counted().reindex(_labels([Int64(30), 30, 10]))
    assert_equal(got.dtype(), DType.int64, "no hole, so no widening")
    assert_equal(_numbers_of(got), [3, 3, 1], "and a label may repeat")


def test_a_fill_value_stops_a_series_widening() raises:
    var got = _counted().reindex(_labels([Int64(10), 99]), Value(Int64(0)))
    assert_equal(got.dtype(), DType.int64, "there is nothing to widen for")
    assert_equal(_numbers_of(got), [1, 0], "and the value is in the row")


def test_a_fill_value_leaves_a_hole_the_series_already_had() raises:
    # The same rule the frame half is tested for, and the reason the fill is a
    # row rather than a pass over the answer. A null that was in the series
    # before the lookup is still a null after it.
    var got = _counted(hole=True).reindex(
        _labels([Int64(20), 99]), Value(Int64(7))
    )
    assert_equal(_numbers_of(got), [-1, 7], "only the row nobody found")


def test_a_series_on_no_labels_at_all_is_empty() raises:
    var got = _counted().reindex(_labels(List[Int64]()))
    assert_equal(len(got), 0, "no rows")
    assert_equal(got.dtype(), DType.int64, "and nothing widened on the way")


def test_a_word_cannot_fill_a_series_of_numbers() raises:
    with assert_raises(contains="nothing to put in the row"):
        _ = _counted().reindex(_labels([Int64(99)]), Value(String("nothing")))


def test_a_series_whose_labels_repeat_is_refused() raises:
    var series = Series("count", _labels([Int64(1), 2]))
    series.index = Index(_labels([Int64(10), 10]), String("key"))
    with assert_raises(contains="unique"):
        _ = series.reindex(_labels([Int64(10)]))


def test_reindexing_an_index_answers_the_labels_and_the_lookup() raises:
    var index = Index(_labels([Int64(10), 20, 30]), String("key"))
    var got = index.reindex(_labels([Int64(30), 99]))
    assert_equal(_labels_of(got.index), [30, 99], "the labels asked for")
    assert_true(got.positions, "and where to find each of them")
    ref found = got.positions.value()
    assert_equal(Int(found[0]), 2, "thirty is the third row")
    assert_equal(Int(found[1]), -1, "and ninety nine is nowhere")


def test_an_index_reindexed_onto_itself_has_nothing_to_move() raises:
    # pandas answers the second half with nothing rather than with the range a
    # gather would have been, which is how a caller learns it can skip the move.
    var index = Index(_labels([Int64(10), 20, 30]), String("key"))
    var got = index.reindex(_labels([Int64(10), 20, 30]))
    assert_false(got.positions, "there is nothing to gather")
    assert_equal(
        _labels_of(got.index), [10, 20, 30], "and the labels are those"
    )


def test_an_index_reindexed_onto_nothing_answers_an_empty_lookup() raises:
    var index = Index(_labels([Int64(10), 20]), String("key"))
    var got = index.reindex(_labels(List[Int64]()))
    assert_equal(len(got.index), 0, "no labels")
    assert_true(got.positions, "and an answer of no positions, not no answer")
    assert_equal(len(got.positions.value()), 0, "which is empty")


def test_a_list_of_labels_takes_the_name_the_index_had() raises:
    var index = Index(_labels([Int64(10), 20]), String("key"))
    var got = index.reindex(_labels([Int64(20)]))
    assert_true(got.index.name, "the result is named")
    assert_equal(
        got.index.name.value(), "key", "after the index that was asked"
    )


def test_an_index_of_labels_keeps_its_own_name() raises:
    # The rule that reads backwards until you notice it is the same rule twice:
    # the name belongs to whoever was in a position to say what it was, and a
    # target that is an index was.
    var index = Index(_labels([Int64(10), 20]), String("key"))
    var wanted = Index(_labels([Int64(20)]), String("other"))
    var got = index.reindex(wanted)
    assert_equal(got.index.name.value(), "other", "the target's own name")


def test_an_index_that_repeats_a_label_cannot_be_reindexed() raises:
    var index = Index(_labels([Int64(10), 10]), String("key"))
    with assert_raises(contains="unique"):
        _ = index.reindex(_labels([Int64(10)]))


def _other() raises -> DataFrame:
    """A frame labelled twenty, thirty and forty, holding one column of the
    three `_keyed` has and one it does not."""
    var value = Array[DType.int64](3)
    for i in range(3):
        value.set_valid(i, Int64(0))
    var series = List[Series]()
    series.append(Series("other", _labels([Int64(20), 30, 40])))
    series.append(Series("count", value^))
    series.append(Series("extra", Array[DType.int64](3)))
    return DataFrame.from_series(series^).set_index("other")


def test_a_frame_reindexed_onto_an_index_takes_that_index_name() raises:
    # The whole reason the overload exists. A bare set of labels leaves the
    # answer named after the frame that was asked, and an index says what the
    # answer should be called as well as what should be in it.
    var got = _keyed().reindex(Index(_labels([Int64(30), 10]), String("other")))
    assert_equal(got.index.name.value(), "other", "the target's own name")
    assert_equal(_label_list(got), [30, 10], "and the labels it asked for")


def test_a_frame_reindexed_onto_an_empty_index_takes_its_name_too() raises:
    # The short circuit is a second path through the method and it has its own
    # chance to drop the name, so it is asked separately.
    var got = _keyed().reindex(Index(_labels(List[Int64]()), String("other")))
    assert_equal(got.rows, 0, "no rows came back")
    assert_equal(got.index.name.value(), "other", "under the name asked for")


def test_a_series_reindexed_onto_an_index_takes_that_index_name() raises:
    var got = _counted().reindex(Index(_labels([Int64(30)]), String("other")))
    assert_equal(got.index.name.value(), "other", "the target's own name")
    assert_equal(got.name, "count", "and the series keeps its own")


def test_a_frame_shaped_like_another_takes_its_labels_and_its_columns() raises:
    var got = _keyed().reindex_like(_other())
    assert_equal(_label_list(got), [20, 30, 40], "the other frame's labels")
    assert_equal(got.names(), ["count", "extra"], "and its column names")


def test_a_frame_shaped_like_another_is_labelled_the_way_it_is() raises:
    # The reason this is a method rather than two calls at the boundary: a
    # caller who took the labels out and handed them over would lose the name.
    var got = _keyed().reindex_like(_other())
    assert_equal(got.index.name.value(), "other", "the other frame's name")


def test_a_row_the_other_frame_has_and_this_one_does_not_goes_missing() raises:
    var got = _keyed().reindex_like(_other())
    assert_equal(_counts_of(got), [2, 3, -1], "forty was not there to bring")


def test_a_column_the_other_frame_has_and_this_one_does_not_is_made() raises:
    var got = _keyed().reindex_like(_other())
    var made = _column(got, "extra")
    assert_true(
        made.dtype() == DType.float64, "made as a float column of nothing"
    )
    ref values = made.as_typed_view[DType.float64]()
    for i in range(3):
        assert_true(isnan(values[i]), "every row of it is missing")


def test_a_frame_shaped_like_itself_comes_back_as_it_was() raises:
    var got = _keyed().reindex_like(_keyed())
    assert_equal(_label_list(got), [10, 20, 30], "the labels it already had")
    assert_equal(_counts_of(got), [1, 2, 3], "and the rows it already had")
    assert_equal(got.names(), ["count", "size", "word"], "and its columns")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
