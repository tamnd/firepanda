"""Tests for arithmetic and comparison on a frame.

Every expected value in here was read off a running pandas 3.0. A frame aligns
on two axes rather than one, so on top of everything `tests/test_series_arith.mojo`
already covers there are three more questions, and all three of them have an
answer that is easy to get wrong.

The first is what the result's columns are. Two frames whose column names are
the same list in the same order keep that order. Anything else is the sorted
union, which means two frames holding the same names in a different order come
back sorted rather than in either operand's order, and a name that only one of
them has is in the result holding nothing.

The second is that the row plan is worked out once and used for every column.
That is not observable from the outside, which is exactly why it is worth a test
that two frames with several columns and differing labels still put every column
on the same rows.

The third is `df + s`, which reads like adding a column and is not. The series
labels are matched against the column names, so a series is broadcast across the
rows and its values act as one constant per column. `axis=0` is the other
reading and has to be asked for.

The unary operations need none of that. They are here because a frame applies
them column by column and because `-` on a boolean column is the logical not,
which is pandas rather than numpy.

The same two divergences as the series tests show up. An aligned result stays an
integer with nulls in it where pandas widens to float64 for a NaN, and a
comparison of two frames that are not shaped alike answers a missing value where
pandas answers False.
"""

from std.collections import Optional
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.array.strings import strings_from_list
from firepanda.array.value import Value
from firepanda.frame.frame import DataFrame
from firepanda.frame.index import Index
from firepanda.frame.series import Series
from firepanda.kernel.binary import BinaryOp


def unnamed() -> Optional[String]:
    """A level name of `None`, spelled once because it is written a lot."""
    return Optional[String]()


def col(name: String, values: List[Int64]) raises -> Series:
    """One int64 column, with no labels of its own to speak of."""
    return Series(name, from_list[DType.int64](values))


def labelled(var columns: List[Series], at: List[Int64]) raises -> DataFrame:
    """A frame of int64 columns under the given row labels."""
    var out = DataFrame.from_series(columns^)
    out.index = Index(AnyArray(from_list[DType.int64](at)), unnamed())
    return out^


def frame(var a: Series, at: List[Int64]) raises -> DataFrame:
    """A one column frame under the given row labels."""
    var columns = List[Series]()
    columns.append(a^)
    return labelled(columns^, at)


def frame(var a: Series, var b: Series, at: List[Int64]) raises -> DataFrame:
    """A two column frame under the given row labels."""
    var columns = List[Series]()
    columns.append(a^)
    columns.append(b^)
    return labelled(columns^, at)


def frame(
    var a: Series, var b: Series, var c: Series, at: List[Int64]
) raises -> DataFrame:
    """A three column frame under the given row labels."""
    var columns = List[Series]()
    columns.append(a^)
    columns.append(b^)
    columns.append(c^)
    return labelled(columns^, at)


def flag_frame(name: String, values: List[Bool]) raises -> DataFrame:
    """A one column boolean frame under a default range index."""
    var out = Array[DType.bool](len(values))
    for i in range(len(values)):
        out.set_valid(i, values[i])
    var columns = List[Series]()
    columns.append(Series(name, out^))
    return DataFrame.from_series(columns^)


def series(name: String, values: List[Int64], at: List[Int64]) raises -> Series:
    """An int64 series under integer row labels."""
    var out = Series(name, from_list[DType.int64](values))
    out.index = Index(AnyArray(from_list[DType.int64](at)), unnamed())
    return out^


def text_labelled(
    name: String, values: List[Int64], at: List[String]
) raises -> Series:
    """An int64 series under text row labels, which is what naming columns
    needs."""
    var out = Series(name, from_list[DType.int64](values))
    out.index = Index(AnyArray(strings_from_list(at)), unnamed())
    return out^


def ints(df: DataFrame, name: String) raises -> List[Int64]:
    """One column's values, with a null read as zero."""
    var column = df.column(name)
    ref view = column.values.as_typed_view[DType.int64]()
    var out = List[Int64](capacity=len(view))
    for i in range(len(view)):
        out.append(view[i])
    return out^


def floats(df: DataFrame, name: String) raises -> List[Float64]:
    """One float64 column's values."""
    var column = df.column(name)
    ref view = column.values.as_typed_view[DType.float64]()
    var out = List[Float64](capacity=len(view))
    for i in range(len(view)):
        out.append(view[i])
    return out^


def flags(df: DataFrame, name: String) raises -> List[Bool]:
    """One boolean column's values."""
    var column = df.column(name)
    ref view = column.values.as_typed_view[DType.bool]()
    var out = List[Bool](capacity=len(view))
    for i in range(len(view)):
        out.append(view[i])
    return out^


def here(df: DataFrame, name: String, i: Int) raises -> Bool:
    """Whether one cell holds a value at all."""
    return df.column(name).values.is_valid(i)


def label_at(df: DataFrame, i: Int) raises -> Int:
    """The row label in position `i`, as an integer."""
    return Int(df.index.materialize().as_typed[DType.int64]()[i])


def test_two_frames_on_the_same_labels_and_columns_add_cell_by_cell() raises:
    """The short circuit on both axes. Two frames that came out of the same file
    hit this, and it does no union, no lookup and no gather."""
    var a = frame(col("x", [1, 2]), col("y", [3, 4]), [10, 11])
    var b = frame(col("x", [10, 20]), col("y", [30, 40]), [10, 11])
    var got = a + b
    assert_equal(len(got.names()), 2, "two columns")
    assert_equal(ints(got, "x")[0], Int64(11), "x row 0")
    assert_equal(ints(got, "y")[1], Int64(44), "y row 1")
    assert_equal(label_at(got, 0), 10, "the labels came through")


def test_two_frames_align_on_both_the_rows_and_the_columns() raises:
    """The result is on the union of the labels and the union of the names, so
    two frames of two columns and two rows each can come back three by three
    with one cell in it that both sides had."""
    var a = frame(col("x", [1, 2]), col("y", [3, 4]), [1, 2])
    var b = frame(col("y", [10, 20]), col("z", [30, 40]), [2, 3])
    var got = a + b
    assert_equal(len(got), 3, "three rows")
    assert_equal(len(got.names()), 3, "three columns")
    assert_equal(ints(got, "y")[1], Int64(14), "the one cell both sides had")
    assert_equal(here(got, "y", 0), False, "row 1 is on the left only")
    assert_equal(here(got, "y", 2), False, "row 3 is on the right only")
    assert_equal(here(got, "x", 1), False, "x is on the left only")
    assert_equal(here(got, "z", 1), False, "z is on the right only")


def test_a_column_only_one_frame_has_comes_back_holding_nothing() raises:
    """It is in the result because the columns are unioned, and it is empty
    because the other frame has no values to combine with it. pandas gives a
    float64 column of NaN; this gives an int64 column of nulls, which is the
    same answer without the widening."""
    var a = frame(col("x", [1, 2]), [1, 2])
    var b = frame(col("y", [10, 20]), [1, 2])
    var got = a + b
    assert_equal(len(got.names()), 2, "both columns")
    assert_equal(here(got, "x", 0), False, "x is empty")
    assert_equal(here(got, "y", 1), False, "y is empty")


def test_a_fill_value_puts_a_one_sided_column_back() raises:
    """`fill_value` fills a side that is missing, and a column the other frame
    does not have is missing on every row, so the whole column survives. This is
    the case that makes the missing side a column of nulls rather than a special
    path: filling it needs it to be there."""
    var a = frame(col("x", [1, 2]), col("y", [3, 4]), [1, 2])
    var b = frame(col("y", [10, 20]), [1, 2])
    var got = a.binary(b, BinaryOp.ADD, Value(Int64(0)))
    assert_equal(ints(got, "x")[0], Int64(1), "x came back")
    assert_equal(ints(got, "x")[1], Int64(2), "x came back whole")
    assert_equal(ints(got, "y")[0], Int64(13), "y still added")


def test_a_fill_value_leaves_a_cell_both_sides_are_missing_alone() raises:
    """The rule is one sided and not "fill every null", same as on a series."""
    var a = frame(col("x", [1, 2]), [1, 2])
    var b = frame(col("x", [10, 20]), [3, 4])
    var got = a.binary(b, BinaryOp.ADD, Value(Int64(0)))
    assert_equal(len(got), 4, "the union of the labels")
    assert_equal(ints(got, "x")[0], Int64(1), "label 1 is on the left only")
    assert_equal(ints(got, "x")[2], Int64(10), "label 3 is on the right only")
    assert_true(here(got, "x", 0), "and both of those were filled")


def test_the_column_order_is_kept_when_the_two_frames_agree() raises:
    """Two frames built the same way keep the order they were built in, which is
    the case where a reordering would be most surprising."""
    var a = frame(col("z", [1]), col("a", [2]), [1])
    var b = frame(col("z", [10]), col("a", [20]), [1])
    var got = a + b
    assert_equal(got.names()[0], "z", "z first")
    assert_equal(got.names()[1], "a", "a second")


def test_the_column_order_is_sorted_when_the_two_frames_differ() raises:
    """Two frames holding the same names in a different order come back sorted.
    pandas picks neither side, which is measured and not guessed."""
    var a = frame(col("z", [1]), col("a", [2]), [1])
    var b = frame(col("a", [20]), col("z", [10]), [1])
    var got = a + b
    assert_equal(got.names()[0], "a", "a first")
    assert_equal(got.names()[1], "z", "z second")


def test_every_column_lands_on_the_same_rows() raises:
    """The row plan is worked out once and reused, so a frame of several columns
    cannot end up with them on different rows. Nothing outside can see the plan,
    so the test is that the labels and the values agree column by column."""
    var a = frame(
        col("x", [1, 2]),
        col("y", [3, 4]),
        col("w", [5, 6]),
        [1, 3],
    )
    var b = frame(
        col("x", [10, 20]),
        col("y", [30, 40]),
        col("w", [50, 60]),
        [2, 3],
    )
    var got = a + b
    assert_equal(len(got), 3, "labels 1, 2 and 3")
    assert_equal(label_at(got, 2), 3, "the shared label is last")
    assert_equal(ints(got, "x")[2], Int64(22), "x on label 3")
    assert_equal(ints(got, "y")[2], Int64(44), "y on label 3")
    assert_equal(ints(got, "w")[2], Int64(66), "w on label 3")
    assert_equal(here(got, "x", 1), False, "and nothing on label 2")


def test_the_seven_operators_against_a_constant() raises:
    """A constant has no labels, so nothing aligns and every column gets the
    same treatment."""
    var a = frame(col("x", [7, 8]), col("y", [2, 3]), [1, 2])
    assert_equal(ints(a + Value(Int64(3)), "x")[0], Int64(10), "add")
    assert_equal(ints(a - Value(Int64(3)), "y")[0], Int64(-1), "sub")
    assert_equal(ints(a * Value(Int64(3)), "y")[1], Int64(9), "mul")
    assert_equal(floats(a / Value(Int64(2)), "x")[0], Float64(3.5), "div")
    assert_equal(ints(a // Value(Int64(3)), "x")[0], Int64(2), "floordiv")
    assert_equal(ints(a % Value(Int64(3)), "x")[0], Int64(1), "mod")
    assert_equal(ints(a ** Value(Int64(2)), "y")[1], Int64(9), "pow")


def test_a_constant_on_the_left_is_not_the_same_as_one_on_the_right() raises:
    """The reflected forms, which matter for the three operations that are not
    commutative."""
    var a = frame(col("x", [7, 8]), [1, 2])
    assert_equal(ints(a.__rsub__(Value(Int64(10))), "x")[0], Int64(3), "10 - 7")
    assert_equal(
        ints(a.__rfloordiv__(Value(Int64(20))), "x")[0], Int64(2), "20 // 7"
    )
    assert_equal(ints(a.__rmod__(Value(Int64(20))), "x")[0], Int64(6), "20 % 7")


def test_the_six_comparisons_against_a_constant() raises:
    """A comparison with a constant answers a boolean column per input
    column."""
    var a = frame(col("x", [7, 8]), col("y", [2, 3]), [1, 2])
    var three = Value(Int64(3))
    assert_equal(flags(a == three, "y")[1], True, "eq")
    assert_equal(flags(a != three, "y")[1], False, "ne")
    assert_equal(flags(a < three, "y")[0], True, "lt")
    assert_equal(flags(a <= three, "y")[1], True, "le")
    assert_equal(flags(a > three, "x")[0], True, "gt")
    assert_equal(flags(a >= three, "y")[1], True, "ge")


def test_comparing_two_frames_needs_the_same_labels_and_columns() raises:
    """`ValueError: Can only compare identically-labeled (both index and
    columns) DataFrame objects` is what pandas raises here, and it raises it for
    a difference on either axis. The flexible forms align instead."""
    var a = frame(col("x", [1, 2]), [1, 2])
    var other_rows = frame(col("x", [1, 2]), [3, 4])
    var other_columns = frame(col("y", [1, 2]), [1, 2])
    with assert_raises(contains="identically-labeled"):
        _ = a == other_rows
    with assert_raises(contains="identically-labeled"):
        _ = a == other_columns


def test_two_frames_with_the_same_names_in_a_different_order_do_not_compare() raises:
    """The column names have to be the same list and not the same set, because
    the comparison walks the two schemas in step."""
    var a = frame(col("x", [1]), col("y", [2]), [1])
    var b = frame(col("y", [2]), col("x", [1]), [1])
    with assert_raises(contains="identically-labeled"):
        _ = a == b


def test_comparing_two_shaped_alike_frames_is_cell_by_cell() raises:
    """Which is the whole point of refusing the rest: the operator only runs
    where the two frames describe the same cells."""
    var a = frame(col("x", [1, 2]), col("y", [5, 6]), [1, 2])
    var b = frame(col("x", [1, 9]), col("y", [5, 5]), [1, 2])
    var got = a == b
    assert_equal(flags(got, "x")[0], True, "x row 0")
    assert_equal(flags(got, "x")[1], False, "x row 1")
    assert_equal(flags(got, "y")[1], False, "y row 1")


def test_the_erased_comparison_aligns_where_the_operator_refuses() raises:
    """The flexible forms go through here, and they can align because a cell
    that only one side has is a missing answer rather than a False one. That
    last part is a divergence: pandas says False."""
    var a = frame(col("x", [1, 2]), [1, 2])
    var b = frame(col("x", [1, 2]), [2, 3])
    var got = a.binary(b, BinaryOp.EQ)
    assert_equal(len(got), 3, "the union of the labels")
    assert_equal(
        flags(got, "x")[1], False, "2 on the left is not 1 on the right"
    )
    assert_equal(here(got, "x", 0), False, "and label 1 has no answer")


def test_a_series_is_broadcast_along_the_columns_by_name() raises:
    """`df + s` matches the series labels against the column names, which reads
    like adding a column and is adding a row."""
    var a = frame(col("x", [1, 2]), col("y", [3, 4]), [1, 2])
    var s = text_labelled("s", [10, 100], ["y", "x"])
    var got = a + s
    assert_equal(ints(got, "x")[0], Int64(101), "x got the x label")
    assert_equal(ints(got, "y")[1], Int64(14), "y got the y label")
    assert_equal(label_at(got, 0), 1, "the row labels are untouched")


def test_a_label_no_column_answers_to_is_still_a_column() raises:
    """The column names and the series labels are unioned, same as two frames
    are, so a label with no column comes back as a column with nothing in it."""
    var a = frame(col("x", [1, 2]), [1, 2])
    var s = text_labelled("s", [10, 100], ["x", "z"])
    var got = a + s
    assert_equal(len(got.names()), 2, "x and z")
    assert_equal(ints(got, "x")[0], Int64(11), "x was added to")
    assert_equal(here(got, "z", 0), False, "and z holds nothing")


def test_a_column_the_series_says_nothing_about_holds_nothing() raises:
    """The other way round. pandas answers NaN for a column no label matched,
    because there is no value to combine the column with."""
    var a = frame(col("x", [1, 2]), col("y", [3, 4]), [1, 2])
    var s = text_labelled("s", [10], ["x"])
    var got = a + s
    assert_equal(ints(got, "x")[1], Int64(12), "x was added to")
    assert_equal(here(got, "y", 0), False, "and y holds nothing")


def test_a_series_broadcast_along_the_columns_can_be_turned_round() raises:
    """`rsub` along the columns, which is `s - df` written the way a frame can
    say it."""
    var a = frame(col("x", [1, 2]), [1, 2])
    var s = text_labelled("s", [10], ["x"])
    var got = a.binary(s, BinaryOp.SUB, axis=1, flip=True)
    assert_equal(ints(got, "x")[0], Int64(9), "10 - 1")
    assert_equal(ints(got, "x")[1], Int64(8), "10 - 2")


def test_a_series_along_the_index_is_used_as_one_more_column() raises:
    """`axis=0` is the other reading, where the series labels are row labels and
    every column of the frame meets the same series."""
    var a = frame(col("x", [1, 2]), col("y", [3, 4]), [1, 2])
    var s = series("s", [5, 6], [2, 3])
    var got = a.binary(s, BinaryOp.ADD, axis=0)
    assert_equal(len(got), 3, "the union of the labels")
    assert_equal(ints(got, "x")[1], Int64(7), "2 + 5 on label 2")
    assert_equal(ints(got, "y")[1], Int64(9), "4 + 5 on label 2")
    assert_equal(here(got, "x", 0), False, "label 1 is not in the series")
    assert_equal(here(got, "y", 2), False, "label 3 is not in the frame")


def test_a_series_along_the_index_can_be_turned_round() raises:
    """Subtraction is the operation where the two readings differ, so it is the
    one worth checking the flag on."""
    var a = frame(col("x", [1, 2]), [1, 2])
    var s = series("s", [10, 20], [1, 2])
    var got = a.binary(s, BinaryOp.SUB, axis=0, flip=True)
    assert_equal(ints(got, "x")[0], Int64(9), "10 - 1")
    assert_equal(ints(got, "x")[1], Int64(18), "20 - 2")


def test_a_series_with_labels_that_are_not_text_cannot_name_columns() raises:
    """Broadcasting along the columns matches labels against names, and a name
    is text, so a series labelled with integers has nothing to match with. The
    message says to use the other axis."""
    var a = frame(col("x", [1, 2]), [1, 2])
    var s = series("s", [10, 20], [1, 2])
    with assert_raises(contains="row labels have to be text"):
        _ = a + s


def test_an_axis_that_is_neither_the_rows_nor_the_columns_is_refused() raises:
    """A frame has two of them."""
    var a = frame(col("x", [1, 2]), [1, 2])
    var s = series("s", [10, 20], [1, 2])
    with assert_raises(contains="axis must be 0"):
        _ = a.binary(s, BinaryOp.ADD, axis=2)


def test_the_four_unary_operations_on_an_integer_frame() raises:
    """One operand, so there is nothing to line up and the only question is what
    each operation does to each column."""
    var a = frame(col("x", [7, -7]), col("y", [1, -1]), [1, 2])
    assert_equal(ints(-a, "x")[0], Int64(-7), "neg")
    assert_equal(ints(+a, "x")[1], Int64(-7), "pos")
    assert_equal(ints(a.abs(), "x")[1], Int64(7), "abs")
    assert_equal(ints(~a, "y")[0], Int64(-2), "invert")


def test_a_unary_operation_keeps_the_labels_and_the_column_names() raises:
    """Nothing about negating a number renames the row it was on."""
    var a = frame(col("x", [7, 8]), [10, 11])
    var got = -a
    assert_equal(got.names()[0], "x", "the column name")
    assert_equal(label_at(got, 1), 11, "the row label")


def test_negating_a_bool_frame_is_the_logical_not() raises:
    """Which is what pandas does, where numpy refuses the expression outright.
    """
    var a = flag_frame("x", [True, False])
    assert_equal(flags(-a, "x")[0], False, "row 0")
    assert_equal(flags(-a, "x")[1], True, "row 1")


def test_a_duplicated_row_label_is_refused_when_the_labels_differ() raises:
    """Same rule as on a series, and for the same reason: pandas pairs a
    repeated label with every copy of it on the other side, which is a join."""
    var a = frame(col("x", [1, 2]), [1, 1])
    var b = frame(col("x", [10, 20]), [1, 2])
    with assert_raises(contains="duplicated row label"):
        _ = a + b


def test_two_equal_indexes_with_duplicates_still_align() raises:
    """Because equal indexes align in position order, which needs no lookup and
    so has nothing to be ambiguous about."""
    var a = frame(col("x", [1, 2]), [1, 1])
    var b = frame(col("x", [10, 20]), [1, 1])
    var got = a + b
    assert_equal(ints(got, "x")[0], Int64(11), "row 0")
    assert_equal(ints(got, "x")[1], Int64(22), "row 1")


def test_the_erased_entry_point_takes_the_operation_as_a_value() raises:
    """The form a plan calls, where the operation is a runtime value rather than
    seven methods."""
    var a = frame(col("x", [1, 2]), [1, 2])
    var b = frame(col("x", [10, 20]), [1, 2])
    assert_equal(ints(a.binary(b, BinaryOp.ADD), "x")[0], Int64(11), "add")
    assert_equal(
        ints(a.binary(b, BinaryOp.SUB, flip=True), "x")[0], Int64(9), "flipped"
    )
    assert_equal(
        ints(a.binary(Value(Int64(4)), BinaryOp.SUB, value_on_left=True), "x")[
            0
        ],
        Int64(3),
        "constant on the left",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
