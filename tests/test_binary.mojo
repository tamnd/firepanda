"""Tests for arithmetic and comparison over columns whose dtypes are values.

The loops these dispatch to are tested in `test_kernel.mojo` against their
scalar twins, one dtype at a time, so nothing here is checking whether addition
adds. What is being checked is the three things that only exist once the dtype
stops being a parameter: that the promotion picks the type NumPy and pandas pick,
that both operands actually arrive at that type before the loop runs, and that a
pair with no common type is an error rather than a reinterpretation.

The type is asserted separately from the values in most of these, because the
whole point of promoting on the types is that the answer's type does not depend
on what is in the column. A test that only looked at values would pass on int32
plus int32 overflowing into the right int64 answer by accident.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import strings_from_list
from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.kernel.binary import (
    BinaryOp,
    binary_any,
    binary_type,
    binary_value_any,
)


def typed[dt: DType](values: List[Scalar[dt]]) raises -> AnyArray:
    """Builds a fully valid column of one dtype."""
    var col = Array[dt](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    return AnyArray(col^)


def holes[
    dt: DType
](values: List[Scalar[dt]], present: List[Bool]) raises -> AnyArray:
    """Builds a column with nulls where asked."""
    var col = Array[dt](len(values))
    for i in range(len(values)):
        if present[i]:
            col.set_valid(i, values[i])
        else:
            col.set_null(i)
    return AnyArray(col^)


def read[dt: DType](col: AnyArray) raises -> List[Scalar[dt]]:
    """Reads a column out as a plain list."""
    ref view = col.as_typed_view[dt]()
    var out = List[Scalar[dt]](capacity=len(view))
    for i in range(len(view)):
        out.append(view[i])
    return out^


def test_two_columns_of_one_type_keep_that_type() raises:
    var got = binary_any(
        typed[DType.int64]([1, 2, 3]),
        typed[DType.int64]([10, 20, 30]),
        BinaryOp.ADD,
    )
    assert_true(got.type == LogicalType.INT64, "result type")
    var values = read[DType.int64](got)
    assert_equal(values[0], Int64(11), "row 0")
    assert_equal(values[1], Int64(22), "row 1")
    assert_equal(values[2], Int64(33), "row 2")


def test_a_narrower_operand_is_widened_to_the_other() raises:
    """An int32 with an int64 is an int64, and the narrow side has to be
    converted rather than read at the wrong width."""
    var got = binary_any(
        typed[DType.int32]([1, 2]), typed[DType.int64]([5, 6]), BinaryOp.MUL
    )
    assert_true(got.type == LogicalType.INT64, "result type")
    var values = read[DType.int64](got)
    assert_equal(values[0], Int64(5), "row 0")
    assert_equal(values[1], Int64(12), "row 1")


def test_an_integer_wider_than_the_mantissa_forces_float64() raises:
    """An int32 has thirty one significant bits and a float32 has twenty four,
    so NumPy answers float64 here and so do we."""
    var type = binary_type(BinaryOp.ADD, LogicalType.INT32, LogicalType.FLOAT32)
    assert_true(type == LogicalType.FLOAT64, "promoted type")
    var got = binary_any(
        typed[DType.int32]([3]), typed[DType.float32]([0.5]), BinaryOp.ADD
    )
    assert_true(got.type == LogicalType.FLOAT64, "result type")
    assert_equal(read[DType.float64](got)[0], Float64(3.5), "row 0")


def test_uint64_with_a_signed_type_goes_to_float64() raises:
    """No signed integer holds the top of uint64, so the answer is lossy and it
    is lossy in the same way NumPy is."""
    var got = binary_any(
        typed[DType.uint64]([4]), typed[DType.int32]([2]), BinaryOp.SUB
    )
    assert_true(got.type == LogicalType.FLOAT64, "result type")
    assert_equal(read[DType.float64](got)[0], Float64(2.0), "row 0")


def test_division_always_answers_float64() raises:
    var got = binary_any(
        typed[DType.int64]([7, 8]),
        typed[DType.int64]([2, 4]),
        BinaryOp.DIV,
    )
    assert_true(got.type == LogicalType.FLOAT64, "result type")
    var values = read[DType.float64](got)
    assert_equal(values[0], Float64(3.5), "an integer division does not floor")
    assert_equal(values[1], Float64(2.0), "row 1")


def test_a_comparison_answers_bool_whatever_went_in() raises:
    var got = binary_any(
        typed[DType.float32]([1.5, 2.5]),
        typed[DType.int16]([2, 2]),
        BinaryOp.LT,
    )
    assert_true(got.type == LogicalType.BOOL, "result type")
    var values = read[DType.bool](got)
    assert_true(values[0], "1.5 is less than 2")
    assert_true(not values[1], "2.5 is not less than 2")


def test_a_comparison_across_widths_compares_the_values() raises:
    """The whole risk in an erased comparison is reading one side at the other
    side's width, which would make 256 as an int16 equal to 0 as an int8."""
    var got = binary_any(
        typed[DType.int16]([256, 1]), typed[DType.int8]([0, 1]), BinaryOp.EQ
    )
    var values = read[DType.bool](got)
    assert_true(not values[0], "256 does not equal 0")
    assert_true(values[1], "1 equals 1")


def test_a_null_on_either_side_makes_the_answer_null() raises:
    var left = holes[DType.int64]([1, 2, 3], [True, False, True])
    var right = holes[DType.int64]([1, 2, 3], [True, True, False])
    var got = binary_any(left, right, BinaryOp.ADD)
    assert_true(got.is_valid(0), "both present")
    assert_true(not got.is_valid(1), "the left is null")
    assert_true(not got.is_valid(2), "the right is null")
    assert_equal(got.null_count(), 2, "nulls")


def test_a_null_survives_the_conversion_the_promotion_asks_for() raises:
    """The cast that widens an operand has to carry the validity across, and a
    null that turned into a zero would be an answer rather than a null."""
    var left = holes[DType.int32]([1, 2], [True, False])
    var got = binary_any(left, typed[DType.int64]([10, 10]), BinaryOp.ADD)
    assert_true(got.is_valid(0), "row 0 is present")
    assert_true(not got.is_valid(1), "row 1 is still null")


def test_bool_columns_compare_and_do_not_add() raises:
    var left = typed[DType.bool]([True, False])
    var right = typed[DType.bool]([True, True])
    var got = binary_any(left, right, BinaryOp.EQ)
    var values = read[DType.bool](got)
    assert_true(values[0], "true equals true")
    assert_true(not values[1], "false does not equal true")

    with assert_raises(contains="is not defined on"):
        _ = binary_any(
            typed[DType.bool]([True]), typed[DType.bool]([True]), BinaryOp.ADD
        )


def test_a_bool_with_a_number_takes_the_number() raises:
    var got = binary_any(
        typed[DType.bool]([True, False]),
        typed[DType.int64]([5, 5]),
        BinaryOp.ADD,
    )
    assert_true(got.type == LogicalType.INT64, "result type")
    var values = read[DType.int64](got)
    assert_equal(values[0], Int64(6), "true is one")
    assert_equal(values[1], Int64(5), "false is zero")


def test_text_with_a_number_has_no_common_type() raises:
    var words = AnyArray(strings_from_list(["a", "b"]))
    with assert_raises(contains="no common type"):
        _ = binary_any(words, typed[DType.int64]([1, 2]), BinaryOp.EQ)


def test_text_with_text_compares_the_elements_not_the_bytes() raises:
    """A string column's physical dtype is uint8, so the one thing that must not
    happen is falling through to the uint8 arm and comparing first bytes."""
    var left = AnyArray(strings_from_list(["ab", "cd"]))
    var right = AnyArray(strings_from_list(["ab", "ce"]))
    var got = binary_any(left, right, BinaryOp.EQ)
    assert_true(got.type == LogicalType.BOOL, "result type")
    var values = read[DType.bool](got)
    assert_true(values[0], "same")
    assert_true(not values[1], "differ in the last byte")


def test_arithmetic_on_text_is_still_an_error() raises:
    var left = AnyArray(strings_from_list(["ab", "cd"]))
    var right = AnyArray(strings_from_list(["ab", "ce"]))
    with assert_raises(contains="is not defined on"):
        _ = binary_any(left, right, BinaryOp.ADD)


def test_columns_of_different_lengths_are_refused() raises:
    with assert_raises(contains="the right has 2"):
        _ = binary_any(
            typed[DType.int64]([1, 2, 3]),
            typed[DType.int64]([1, 2]),
            BinaryOp.ADD,
        )


def test_the_result_type_is_known_without_a_column() raises:
    """A plan needs the type before any row moves, which is the only reason this
    is a function of the types alone."""
    assert_true(
        binary_type(BinaryOp.DIV, LogicalType.INT8, LogicalType.INT8)
        == LogicalType.FLOAT64,
        "division",
    )
    assert_true(
        binary_type(BinaryOp.GE, LogicalType.FLOAT64, LogicalType.INT8)
        == LogicalType.BOOL,
        "comparison",
    )
    assert_true(
        binary_type(BinaryOp.SUB, LogicalType.INT16, LogicalType.UINT16)
        == LogicalType.INT32,
        "signed with unsigned of the same width goes up a width",
    )
    with assert_raises(contains="is not defined on"):
        _ = binary_type(BinaryOp.MUL, LogicalType.BOOL, LogicalType.BOOL)


def test_every_operation_prints_as_the_symbol_it_is_written_with() raises:
    assert_equal(String(BinaryOp.ADD), "+", "add")
    assert_equal(String(BinaryOp.SUB), "-", "subtract")
    assert_equal(String(BinaryOp.MUL), "*", "multiply")
    assert_equal(String(BinaryOp.DIV), "/", "divide")
    assert_equal(String(BinaryOp.EQ), "==", "equal")
    assert_equal(String(BinaryOp.NE), "!=", "not equal")
    assert_equal(String(BinaryOp.LT), "<", "less")
    assert_equal(String(BinaryOp.LE), "<=", "less or equal")
    assert_equal(String(BinaryOp.GT), ">", "greater")
    assert_equal(String(BinaryOp.GE), ">=", "greater or equal")


def test_only_the_six_comparisons_say_they_are_comparisons() raises:
    assert_true(not BinaryOp.ADD.is_comparison(), "add")
    assert_true(not BinaryOp.DIV.is_comparison(), "divide")
    assert_true(BinaryOp.EQ.is_comparison(), "equal")
    assert_true(BinaryOp.GE.is_comparison(), "greater or equal")


def test_a_constant_on_the_right_is_applied_to_every_row() raises:
    var got = binary_value_any(
        typed[DType.int64]([1, 2, 3]), Value(Int64(10)), BinaryOp.ADD
    )
    assert_true(got.type == LogicalType.INT64, "result type")
    var values = read[DType.int64](got)
    assert_equal(values[0], Int64(11), "row 0")
    assert_equal(values[1], Int64(12), "row 1")
    assert_equal(values[2], Int64(13), "row 2")


def test_a_constant_promotes_on_its_type_not_its_value() raises:
    """An int32 column with a constant that arrived as an int32 stays int32.

    Promoting on the value would let `x + 1` widen or not depending on how the
    literal was spelled, which would make an expression's type depend on
    something the plan cannot see."""
    var narrow = binary_value_any(
        typed[DType.int32]([1, 2]), Value(Int32(1)), BinaryOp.ADD
    )
    assert_true(narrow.type == LogicalType.INT32, "int32 with an int32")
    var wide = binary_value_any(
        typed[DType.int32]([1, 2]), Value(Int64(1)), BinaryOp.ADD
    )
    assert_true(wide.type == LogicalType.INT64, "int32 with an int64")


def test_a_float_constant_promotes_an_integer_column() raises:
    var got = binary_value_any(
        typed[DType.int32]([1, 2]), Value(Float64(0.5)), BinaryOp.MUL
    )
    assert_true(got.type == LogicalType.FLOAT64, "result type")
    var values = read[DType.float64](got)
    assert_equal(values[0], Float64(0.5), "row 0")
    assert_equal(values[1], Float64(1.0), "row 1")


def test_subtraction_knows_which_side_the_constant_is_on() raises:
    """`x - 5` and `5 - x` are different answers, so the flag has to reach the
    loop rather than being lost in the erasure."""
    var right = binary_value_any(
        typed[DType.int64]([1, 2]), Value(Int64(5)), BinaryOp.SUB
    )
    var left = binary_value_any(
        typed[DType.int64]([1, 2]), Value(Int64(5)), BinaryOp.SUB, True
    )
    assert_equal(read[DType.int64](right)[0], Int64(-4), "x - 5")
    assert_equal(read[DType.int64](left)[0], Int64(4), "5 - x")


def test_division_knows_which_side_the_constant_is_on() raises:
    var right = binary_value_any(
        typed[DType.int64]([1, 2]), Value(Int64(4)), BinaryOp.DIV
    )
    var left = binary_value_any(
        typed[DType.int64]([1, 2]), Value(Int64(4)), BinaryOp.DIV, True
    )
    assert_true(right.type == LogicalType.FLOAT64, "result type")
    assert_equal(read[DType.float64](right)[0], Float64(0.25), "1 / 4")
    assert_equal(read[DType.float64](left)[0], Float64(4.0), "4 / 1")


def test_a_constant_on_the_left_of_a_comparison_is_turned_round() raises:
    """`5 < x` is `x > 5`, and the answer has to be the first reading."""
    var got = binary_value_any(
        typed[DType.int64]([1, 5, 9]), Value(Int64(5)), BinaryOp.LT, True
    )
    assert_true(got.type == LogicalType.BOOL, "result type")
    var values = read[DType.bool](got)
    assert_equal(values[0], False, "5 < 1")
    assert_equal(values[1], False, "5 < 5")
    assert_equal(values[2], True, "5 < 9")


def test_the_ordered_comparisons_mirror_and_the_others_do_not() raises:
    assert_true(BinaryOp.LT.mirrored() == BinaryOp.GT, "less than")
    assert_true(BinaryOp.LE.mirrored() == BinaryOp.GE, "less or equal")
    assert_true(BinaryOp.GT.mirrored() == BinaryOp.LT, "greater than")
    assert_true(BinaryOp.GE.mirrored() == BinaryOp.LE, "greater or equal")
    assert_true(BinaryOp.EQ.mirrored() == BinaryOp.EQ, "equal")
    assert_true(BinaryOp.NE.mirrored() == BinaryOp.NE, "not equal")


def test_a_comparison_against_a_constant_answers_bool() raises:
    var got = binary_value_any(
        typed[DType.float64]([1.0, 2.5, 4.0]), Value(Float64(2.5)), BinaryOp.GE
    )
    var values = read[DType.bool](got)
    assert_equal(values[0], False, "row 0")
    assert_equal(values[1], True, "row 1")
    assert_equal(values[2], True, "row 2")


def test_a_null_in_the_column_stays_null_against_a_constant() raises:
    var column = holes[DType.int64]([1, 2, 3], [True, False, True])
    var got = binary_value_any(column, Value(Int64(10)), BinaryOp.ADD)
    assert_true(got.is_valid(0), "row 0")
    assert_true(not got.is_valid(1), "row 1 was null")
    assert_true(got.is_valid(2), "row 2")
    assert_equal(got.null_count(), 1, "nulls")


def test_a_null_constant_makes_every_row_null() raises:
    """The constant is missing, so every answer is missing, and the type of the
    answer is still the type the operation would have produced."""
    var got = binary_value_any(
        typed[DType.int64]([1, 2, 3]),
        Value(null=LogicalType.INT64),
        BinaryOp.ADD,
    )
    assert_true(got.type == LogicalType.INT64, "result type")
    assert_equal(len(got), 3, "rows")
    assert_equal(got.null_count(), 3, "nulls")


def test_a_null_constant_in_a_comparison_answers_a_bool_column_of_nulls() raises:
    var got = binary_value_any(
        typed[DType.int64]([1, 2]), Value(null=LogicalType.INT64), BinaryOp.LT
    )
    assert_true(got.type == LogicalType.BOOL, "result type")
    assert_equal(got.null_count(), 2, "nulls")


def test_a_text_constant_against_a_number_has_no_common_type() raises:
    with assert_raises(contains="no common type"):
        _ = binary_value_any(
            typed[DType.int64]([1, 2]), Value(String("five")), BinaryOp.ADD
        )


def test_a_constant_cannot_be_added_to_a_bool_column() raises:
    with assert_raises(contains="is not defined on"):
        _ = binary_value_any(
            typed[DType.bool]([True, False]), Value(True), BinaryOp.ADD
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
