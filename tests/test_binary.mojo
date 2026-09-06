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

Floor division, the remainder and the power are the exception to the first
paragraph, because their values are the argument. Every number asserted about
them was read off a running pandas 3.0 rather than worked out here, and the one
place the two disagree is written down as a disagreement with pandas' own four
answers next to it.
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
    resolve_constant,
    weak_operand_type,
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


def test_dividing_two_integers_answers_float64() raises:
    var got = binary_any(
        typed[DType.int64]([7, 8]),
        typed[DType.int64]([2, 4]),
        BinaryOp.DIV,
    )
    assert_true(got.type == LogicalType.FLOAT64, "result type")
    var values = read[DType.float64](got)
    assert_equal(values[0], Float64(3.5), "an integer division does not floor")
    assert_equal(values[1], Float64(2.0), "row 1")


def test_division_keeps_a_float_operand_at_its_own_width() raises:
    """`pd.Series([7.0, 8.0], dtype="float32") / ...` is float32 in pandas.

    Division answers a float because two integers have no integer answer, and
    that is the whole of the rule. Where one side is already a float there is
    nothing to widen, and answering float64 here would double the memory of a
    column for a precision nobody asked for.
    """
    var got = binary_any(
        typed[DType.float32]([7.0, 8.0]),
        typed[DType.float32]([2.0, 4.0]),
        BinaryOp.DIV,
    )
    assert_true(got.type == LogicalType.FLOAT32, "result type")
    var values = read[DType.float32](got)
    assert_equal(values[0], Float32(3.5), "row 0")
    assert_equal(values[1], Float32(2.0), "row 1")


def test_a_float32_divided_by_an_integer_stays_float32() raises:
    """pandas answers float32, because `promote` has already decided that an
    int8 is representable in a float32 and there is nothing left to widen."""
    var got = binary_any(
        typed[DType.float32]([7.0, 8.0]),
        typed[DType.int8]([2, 4]),
        BinaryOp.DIV,
    )
    assert_true(got.type == LogicalType.FLOAT32, "result type")
    var values = read[DType.float32](got)
    assert_equal(values[0], Float32(3.5), "row 0")
    assert_equal(values[1], Float32(2.0), "row 1")


def test_an_int64_divided_by_a_float32_is_float64() raises:
    """The other side of the same rule. An int64 is not representable in a
    float32, so `promote` gives float64 and the division keeps it."""
    var got = binary_any(
        typed[DType.int64]([7, 8]),
        typed[DType.float32]([2.0, 4.0]),
        BinaryOp.DIV,
    )
    assert_true(got.type == LogicalType.FLOAT64, "result type")
    var values = read[DType.float64](got)
    assert_equal(values[0], Float64(3.5), "row 0")


def test_a_float32_column_divided_by_a_constant_stays_float32() raises:
    """The constant path has its own loop, so it needs its own test.

    `binary_type` is what decides the answer and both paths ask it, but the
    kernel underneath is a different function, and a division that keeps its
    width in the column form while widening in the constant form would be a
    disagreement nothing about the declared type would catch.
    """
    var right = binary_value_any(
        typed[DType.float32]([7.0, 8.0]), Value(Float32(2.0)), BinaryOp.DIV
    )
    var left = binary_value_any(
        typed[DType.float32]([2.0, 4.0]),
        Value(Float32(8.0)),
        BinaryOp.DIV,
        True,
    )
    assert_true(right.type == LogicalType.FLOAT32, "result type")
    assert_true(left.type == LogicalType.FLOAT32, "flipped result type")
    assert_equal(read[DType.float32](right)[0], Float32(3.5), "7 / 2")
    assert_equal(read[DType.float32](left)[0], Float32(4.0), "8 / 2")
    assert_equal(read[DType.float32](left)[1], Float32(2.0), "8 / 4")


def test_floor_division_keeps_the_operand_type_and_floors() raises:
    """`pd.Series([-7, 7, -7, 7]) // pd.Series([3, 3, -3, -3])` is int64 and
    `[-3, 2, 2, -3]`, which is the Python rounding and not the C one."""
    var got = binary_any(
        typed[DType.int64]([-7, 7, -7, 7]),
        typed[DType.int64]([3, 3, -3, -3]),
        BinaryOp.FLOORDIV,
    )
    assert_true(got.type == LogicalType.INT64, "result type")
    var values = read[DType.int64](got)
    assert_equal(values[0], Int64(-3), "-7 // 3 rounds away from zero")
    assert_equal(values[1], Int64(2), "7 // 3")
    assert_equal(values[2], Int64(2), "-7 // -3")
    assert_equal(values[3], Int64(-3), "7 // -3 rounds away from zero")


def test_the_remainder_takes_the_sign_of_the_divisor() raises:
    """The same pair through `%` in pandas is int64 and `[2, 1, -1, -2]`, which
    is the remainder that goes with the floor above."""
    var got = binary_any(
        typed[DType.int64]([-7, 7, -7, 7]),
        typed[DType.int64]([3, 3, -3, -3]),
        BinaryOp.MOD,
    )
    assert_true(got.type == LogicalType.INT64, "result type")
    var values = read[DType.int64](got)
    assert_equal(values[0], Int64(2), "-7 % 3 is positive")
    assert_equal(values[1], Int64(1), "7 % 3")
    assert_equal(values[2], Int64(-1), "-7 % -3 is negative")
    assert_equal(values[3], Int64(-2), "7 % -3 is negative")


def test_an_integer_divided_by_zero_is_null_and_stays_an_integer() raises:
    """This is the registered divergence, and it is registered because pandas
    has four answers to it and none of them is a function of the types alone.

    A numpy backed series widens the column to float64 and puts an infinity
    there, a masked Int64 series answers zero and says nothing, an Arrow backed
    series raises `ArrowInvalid`, and `%` on an Arrow backed series raises
    `NotImplementedError` instead. firepanda answers a null and keeps int64."""
    var divisors = typed[DType.int64]([3, 0, 3, 0])
    var quotient = binary_any(
        typed[DType.int64]([-7, 7, -7, 7]), divisors, BinaryOp.FLOORDIV
    )
    assert_true(quotient.type == LogicalType.INT64, "the type does not widen")
    assert_true(quotient.is_valid(0), "row 0 divides")
    assert_true(not quotient.is_valid(1), "row 1 has a zero divisor")
    assert_equal(quotient.null_count(), 2, "nulls")
    assert_equal(read[DType.int64](quotient)[0], Int64(-3), "row 0")
    assert_equal(
        read[DType.int64](quotient)[1], Int64(0), "a null holds a zero"
    )

    var remainder = binary_any(
        typed[DType.int64]([-7, 7, -7, 7]), divisors, BinaryOp.MOD
    )
    assert_true(remainder.type == LogicalType.INT64, "the type does not widen")
    assert_equal(remainder.null_count(), 2, "nulls")
    assert_equal(read[DType.int64](remainder)[0], Int64(2), "row 0")


def test_a_zero_divisor_past_the_last_full_register_is_still_found() raises:
    """The loop reads a whole register at the end of the column and the padding
    behind it is zero, so the row that clears the bits has to stop at the length
    and the rows before it still have to be checked. Nine rows is a partial tail
    at every register width the loop is compiled for."""
    var numerators = typed[DType.int64]([1, 2, 3, 4, 5, 6, 7, 8, 9])
    var divisors = typed[DType.int64]([1, 1, 1, 1, 1, 1, 1, 1, 0])
    var got = binary_any(numerators, divisors, BinaryOp.FLOORDIV)
    assert_equal(got.null_count(), 1, "only the last row is null")
    assert_true(not got.is_valid(8), "the last row")
    assert_equal(read[DType.int64](got)[7], Int64(8), "the row before it")


def test_a_zero_divisor_is_found_in_every_morsel_of_a_long_column() raises:
    """A column past the morsel size is divided on several cores at once and
    each of them writes to the same validity bitmap. A morsel boundary is a
    multiple of sixty four rows, so no two of them share a byte, and this is the
    test that would fail if that ever stopped being true."""
    var rows = 300000
    var numerators = Array[DType.int64](rows)
    var divisors = Array[DType.int64](rows)
    for i in range(rows):
        numerators.set_valid(i, Int64(i))
        divisors.set_valid(i, Int64(0) if i % 100000 == 7 else Int64(2))

    var got = binary_any(
        AnyArray(numerators^), AnyArray(divisors^), BinaryOp.FLOORDIV
    )
    assert_equal(got.null_count(), 3, "one null per hundred thousand rows")
    assert_true(not got.is_valid(7), "the first")
    assert_true(not got.is_valid(200007), "the last")
    assert_true(got.is_valid(200008), "the row after it")
    assert_equal(read[DType.int64](got)[9], Int64(4), "9 // 2")


def test_a_float_divided_by_zero_is_an_infinity_rather_than_a_null() raises:
    """A float column has somewhere to put an infinity, so there is nothing to
    diverge about. pandas answers `[-3.0, inf, inf, nan]` and so does this."""
    var got = binary_any(
        typed[DType.float64]([-7.0, 7.0, 1.0, 0.0]),
        typed[DType.float64]([3.0, 0.0, 0.0, 0.0]),
        BinaryOp.FLOORDIV,
    )
    assert_true(got.type == LogicalType.FLOAT64, "result type")
    assert_equal(got.null_count(), 0, "no nulls")
    var values = read[DType.float64](got)
    assert_equal(values[0], Float64(-3.0), "-7.0 // 3.0")
    assert_true(values[1] > Float64(1e308), "7.0 // 0.0 is an infinity")
    assert_true(values[3] != values[3], "0.0 // 0.0 is a NaN")


def test_a_float_remainder_by_zero_is_a_nan_rather_than_a_null() raises:
    var got = binary_any(
        typed[DType.float64]([-7.0, 7.0]),
        typed[DType.float64]([3.0, 0.0]),
        BinaryOp.MOD,
    )
    assert_equal(got.null_count(), 0, "no nulls")
    var values = read[DType.float64](got)
    assert_equal(values[0], Float64(2.0), "-7.0 % 3.0 takes the divisor's sign")
    assert_true(values[1] != values[1], "7.0 % 0.0 is a NaN")


def test_a_power_keeps_the_operand_type() raises:
    """`pd.Series([2, 3, 0, -2]) ** pd.Series([3, 0, 0, 3])` is int64 and
    `[8, 1, 1, -8]`, so zero to the zero is one and the type does not widen."""
    var got = binary_any(
        typed[DType.int64]([2, 3, 0, -2]),
        typed[DType.int64]([3, 0, 0, 3]),
        BinaryOp.POW,
    )
    assert_true(got.type == LogicalType.INT64, "result type")
    var values = read[DType.int64](got)
    assert_equal(values[0], Int64(8), "2 ** 3")
    assert_equal(values[1], Int64(1), "anything to the zero")
    assert_equal(values[2], Int64(1), "zero to the zero")
    assert_equal(values[3], Int64(-8), "a negative base keeps its sign")


def test_a_negative_integer_exponent_is_refused() raises:
    """There is no integer answer, and numpy raises rather than truncating to
    zero, so the loop raises rather than answering the zero Mojo would give."""
    with assert_raises(contains="negative integer powers"):
        _ = binary_any(
            typed[DType.int64]([2, 2]),
            typed[DType.int64]([-1, -1]),
            BinaryOp.POW,
        )


def test_a_negative_float_exponent_is_a_fraction_and_not_an_error() raises:
    var got = binary_any(
        typed[DType.float64]([2.0, 4.0]),
        typed[DType.float64]([-1.0, -0.5]),
        BinaryOp.POW,
    )
    var values = read[DType.float64](got)
    assert_equal(values[0], Float64(0.5), "2.0 ** -1.0")
    assert_equal(values[1], Float64(0.5), "4.0 ** -0.5")


def test_a_float_power_is_the_answer_numpy_gives_to_the_last_bit() raises:
    """The vector power instruction answers `1.4142135623734946` for the square
    root of two, which is out by six hundred of the last bits and would show up
    as a failure against pandas rather than as a rounding nobody notices. These
    three are what numpy prints, and the third is the one that catches a loop
    that quietly computed in single precision."""
    var got = binary_any(
        typed[DType.float64]([2.0, 10.0, 3.0]),
        typed[DType.float64]([0.5, 3.0, 0.3]),
        BinaryOp.POW,
    )
    var values = read[DType.float64](got)
    assert_equal(values[0], Float64(1.4142135623730951), "2.0 ** 0.5")
    assert_equal(values[1], Float64(1000.0), "10.0 ** 3.0")
    assert_equal(values[2], Float64(1.3903891703159093), "3.0 ** 0.3")


def test_a_null_row_does_not_look_like_a_zero_divisor() raises:
    """A null holds a zero underneath, so the divisor check sees one there. The
    row is null either way, and what this is checking is that a null on the
    other side has not turned the whole column into nulls."""
    var got = binary_any(
        typed[DType.int64]([10, 20, 30]),
        holes[DType.int64]([2, 2, 2], [True, False, True]),
        BinaryOp.FLOORDIV,
    )
    assert_equal(got.null_count(), 1, "only the null row")
    var values = read[DType.int64](got)
    assert_equal(values[0], Int64(5), "row 0")
    assert_equal(values[2], Int64(15), "row 2")


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
    assert_true(
        binary_type(BinaryOp.FLOORDIV, LogicalType.INT8, LogicalType.INT8)
        == LogicalType.INT8,
        "floor division keeps the operand type where division does not",
    )
    assert_true(
        binary_type(BinaryOp.MOD, LogicalType.INT32, LogicalType.FLOAT32)
        == LogicalType.FLOAT64,
        "the remainder promotes the same way addition does",
    )
    assert_true(
        binary_type(BinaryOp.POW, LogicalType.INT16, LogicalType.INT16)
        == LogicalType.INT16,
        "a power keeps the operand type",
    )
    with assert_raises(contains="is not defined on"):
        _ = binary_type(BinaryOp.MUL, LogicalType.BOOL, LogicalType.BOOL)
    with assert_raises(contains="is not defined on"):
        _ = binary_type(BinaryOp.FLOORDIV, LogicalType.BOOL, LogicalType.BOOL)


def test_every_operation_prints_as_the_symbol_it_is_written_with() raises:
    assert_equal(String(BinaryOp.ADD), "+", "add")
    assert_equal(String(BinaryOp.SUB), "-", "subtract")
    assert_equal(String(BinaryOp.MUL), "*", "multiply")
    assert_equal(String(BinaryOp.DIV), "/", "divide")
    assert_equal(String(BinaryOp.FLOORDIV), "//", "floor divide")
    assert_equal(String(BinaryOp.MOD), "%", "remainder")
    assert_equal(String(BinaryOp.POW), "**", "power")
    assert_equal(String(BinaryOp.EQ), "==", "equal")
    assert_equal(String(BinaryOp.NE), "!=", "not equal")
    assert_equal(String(BinaryOp.LT), "<", "less")
    assert_equal(String(BinaryOp.LE), "<=", "less or equal")
    assert_equal(String(BinaryOp.GT), ">", "greater")
    assert_equal(String(BinaryOp.GE), ">=", "greater or equal")


def test_only_the_six_comparisons_say_they_are_comparisons() raises:
    """`is_comparison` is one integer test against the first comparison's code,
    so an operation added on the wrong side of that line would answer this
    wrongly rather than fail to compile. Every arithmetic code is named here for
    that reason."""
    assert_true(not BinaryOp.ADD.is_comparison(), "add")
    assert_true(not BinaryOp.SUB.is_comparison(), "subtract")
    assert_true(not BinaryOp.MUL.is_comparison(), "multiply")
    assert_true(not BinaryOp.DIV.is_comparison(), "divide")
    assert_true(not BinaryOp.FLOORDIV.is_comparison(), "floor divide")
    assert_true(not BinaryOp.MOD.is_comparison(), "remainder")
    assert_true(not BinaryOp.POW.is_comparison(), "power")
    assert_true(BinaryOp.EQ.is_comparison(), "equal")
    assert_true(BinaryOp.NE.is_comparison(), "not equal")
    assert_true(BinaryOp.LT.is_comparison(), "less than")
    assert_true(BinaryOp.LE.is_comparison(), "less or equal")
    assert_true(BinaryOp.GT.is_comparison(), "greater than")
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


def test_floor_division_knows_which_side_the_constant_is_on() raises:
    var right = binary_value_any(
        typed[DType.int64]([7, 8]), Value(Int64(2)), BinaryOp.FLOORDIV
    )
    var left = binary_value_any(
        typed[DType.int64]([2, 3]), Value(Int64(7)), BinaryOp.FLOORDIV, True
    )
    assert_true(right.type == LogicalType.INT64, "result type")
    assert_equal(read[DType.int64](right)[0], Int64(3), "7 // 2")
    assert_equal(read[DType.int64](left)[0], Int64(3), "7 // 2 the other way")
    assert_equal(read[DType.int64](left)[1], Int64(2), "7 // 3")


def test_the_remainder_knows_which_side_the_constant_is_on() raises:
    var right = binary_value_any(
        typed[DType.int64]([7, 8]), Value(Int64(3)), BinaryOp.MOD
    )
    var left = binary_value_any(
        typed[DType.int64]([3, 5]), Value(Int64(7)), BinaryOp.MOD, True
    )
    assert_equal(read[DType.int64](right)[0], Int64(1), "7 % 3")
    assert_equal(read[DType.int64](right)[1], Int64(2), "8 % 3")
    assert_equal(read[DType.int64](left)[0], Int64(1), "7 % 3 the other way")
    assert_equal(read[DType.int64](left)[1], Int64(2), "7 % 5")


def test_a_constant_zero_divisor_makes_every_row_null() raises:
    """It is one test rather than one per register, and there is no loop left to
    run once it fires."""
    var got = binary_value_any(
        typed[DType.int64]([1, 2, 3]), Value(Int64(0)), BinaryOp.FLOORDIV
    )
    assert_true(got.type == LogicalType.INT64, "the type does not widen")
    assert_equal(len(got), 3, "rows")
    assert_equal(got.null_count(), 3, "nulls")


def test_a_zero_in_the_column_nulls_a_row_of_a_flipped_division() raises:
    """Flipped, the divisor is the column, so the check that could be hoisted
    above the loop for `x // 5` has to happen inside it for `5 // x`."""
    var got = binary_value_any(
        typed[DType.int64]([3, 0, 4]), Value(Int64(12)), BinaryOp.FLOORDIV, True
    )
    assert_equal(got.null_count(), 1, "one null")
    assert_true(not got.is_valid(1), "the zero row")
    var values = read[DType.int64](got)
    assert_equal(values[0], Int64(4), "12 // 3")
    assert_equal(values[2], Int64(3), "12 // 4")


def test_a_constant_negative_exponent_is_refused_on_either_side() raises:
    with assert_raises(contains="negative integer powers"):
        _ = binary_value_any(
            typed[DType.int64]([2, 3]), Value(Int64(-2)), BinaryOp.POW
        )
    with assert_raises(contains="negative integer powers"):
        _ = binary_value_any(
            typed[DType.int64]([1, -2]), Value(Int64(3)), BinaryOp.POW, True
        )


def test_a_constant_power_applies_to_every_row() raises:
    var right = binary_value_any(
        typed[DType.int64]([2, 3, 0]), Value(Int64(2)), BinaryOp.POW
    )
    var left = binary_value_any(
        typed[DType.int64]([3, 0]), Value(Int64(2)), BinaryOp.POW, True
    )
    assert_true(right.type == LogicalType.INT64, "result type")
    assert_equal(read[DType.int64](right)[0], Int64(4), "2 ** 2")
    assert_equal(read[DType.int64](right)[2], Int64(0), "0 ** 2")
    assert_equal(read[DType.int64](left)[0], Int64(8), "2 ** 3")
    assert_equal(read[DType.int64](left)[1], Int64(1), "2 ** 0")


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


def test_a_weak_integer_takes_the_dtype_of_the_column() raises:
    """`s + 2` on an int8 column stays int8, which is the whole rule in one line.

    A Python integer has no width. NumPy 2 and pandas 3 stopped inventing one
    for it and made it take the width of whatever it meets, so the column
    decides and a small number never widens anything."""
    var got = binary_value_any(
        typed[DType.int8]([1, 2, 3]),
        Value(Int64(2)).weakened(),
        BinaryOp.ADD,
    )
    assert_true(got.type == LogicalType.INT8, "result type")
    var values = read[DType.int8](got)
    assert_equal(values[0], 3, "row 0")
    assert_equal(values[2], 5, "row 2")


def test_a_constant_that_arrived_with_a_dtype_still_promotes() raises:
    """The flag is what changes the answer, so the same number without it does
    not. This is the pair to the test above and the reason `Value.weak` is a
    field rather than a guess made from the type: an int64 that a caller asked
    for is an int64 and it widens the column the way it always did."""
    var got = binary_value_any(
        typed[DType.int8]([1, 2, 3]), Value(Int64(2)), BinaryOp.ADD
    )
    assert_true(got.type == LogicalType.INT64, "result type")


def test_a_weak_integer_too_large_for_the_column_is_refused() raises:
    """Taking the column's width means a number that does not fit has nowhere to
    go, and pandas raises rather than widening. The message is pandas' own,
    word for word, because that is the string somebody will search for."""
    with assert_raises(contains="Python integer 200 out of bounds for int8"):
        _ = binary_value_any(
            typed[DType.int8]([1, 2, 3]),
            Value(Int64(200)).weakened(),
            BinaryOp.ADD,
        )


def test_a_comparison_against_a_weak_integer_never_narrows() raises:
    """`s > 200` on an int8 column answers False rather than raising.

    Comparisons are the exception to the rule above and they are the exception
    in pandas too, which is not an inconsistency: asking whether a byte is
    larger than two hundred has a perfectly good answer, and refusing to answer
    it because the number does not fit would be useless."""
    var got = binary_value_any(
        typed[DType.int8]([1, 100, 127]),
        Value(Int64(200)).weakened(),
        BinaryOp.GT,
    )
    assert_true(got.type == LogicalType.BOOL, "result type")
    var values = read[DType.bool](got)
    assert_equal(values[0], False, "row 0")
    assert_equal(values[2], False, "row 2, the largest int8 there is")


def test_a_weak_float_takes_the_width_of_a_float_column() raises:
    """A Python float is weak the same way a Python integer is, so a float32
    column stays float32 rather than being pulled up to double by a literal."""
    var got = binary_value_any(
        typed[DType.float32]([1.0, 2.0]),
        Value(Float64(1.5)).weakened(),
        BinaryOp.ADD,
    )
    assert_true(got.type == LogicalType.FLOAT32, "result type")
    var values = read[DType.float32](got)
    assert_equal(values[0], Float32(2.5), "row 0")


def test_a_weak_float_still_widens_an_integer_column() raises:
    """Weak means it has no width of its own, not that it has no kind. A float
    meeting an integer column has to become a float somewhere, and float64 is
    where pandas puts it because there is no narrower float to pick from an
    int64."""
    var got = binary_value_any(
        typed[DType.int64]([1, 2]),
        Value(Float64(0.5)).weakened(),
        BinaryOp.ADD,
    )
    assert_true(got.type == LogicalType.FLOAT64, "result type")


def test_a_weak_float_too_large_for_a_float32_column_is_not_refused() raises:
    """The bounds check is an integer thing, because a float has infinities to
    overflow into and NumPy is content to let it. This is one of the two places
    firepanda has to be careful not to be stricter than the thing it copies."""
    var got = binary_value_any(
        typed[DType.float32]([1.0, 2.0]),
        Value(Float64(1.0e300)).weakened(),
        BinaryOp.ADD,
    )
    assert_true(got.type == LogicalType.FLOAT32, "result type")


def test_a_weak_bool_takes_the_dtype_of_a_number_column() raises:
    """`True` is a Python `bool` and a Python `bool` is an `int`, so it lands on
    the column the way `1` does rather than turning the answer into a bool."""
    var got = binary_value_any(
        typed[DType.int16]([1, 2]), Value(True).weakened(), BinaryOp.ADD
    )
    assert_true(got.type == LogicalType.INT16, "result type")
    var values = read[DType.int16](got)
    assert_equal(values[0], 2, "row 0")


def test_a_weak_constant_against_text_is_left_alone() raises:
    """There is no width to take from a text column, so the rule does not apply
    and the failure is the one it always was."""
    with assert_raises(contains="no common type"):
        _ = binary_value_any(
            AnyArray(strings_from_list(["a", "b"])),
            Value(Int64(2)).weakened(),
            BinaryOp.ADD,
        )


def test_which_dtype_a_weak_scalar_lands_on() raises:
    """The rule on its own, without a column of data underneath it, because the
    tests above each pin one row of it and this says what the table is."""
    assert_true(
        weak_operand_type(LogicalType.INT8, LogicalType.INT64)
        == LogicalType.INT8,
        "an integer takes the integer column",
    )
    assert_true(
        weak_operand_type(LogicalType.FLOAT32, LogicalType.INT64)
        == LogicalType.FLOAT32,
        "an integer takes the float column",
    )
    assert_true(
        weak_operand_type(LogicalType.INT8, LogicalType.FLOAT64)
        == LogicalType.FLOAT64,
        "a float on an integer column has no narrower float to pick",
    )
    assert_true(
        weak_operand_type(LogicalType.FLOAT32, LogicalType.FLOAT64)
        == LogicalType.FLOAT32,
        "a float takes the float column",
    )
    assert_true(
        weak_operand_type(LogicalType.INT16, LogicalType.BOOL)
        == LogicalType.INT16,
        "a bool is an integer and takes the column",
    )
    assert_true(
        weak_operand_type(LogicalType.BOOL, LogicalType.INT64)
        == LogicalType.INT64,
        "a number on a bool column is the one case the column does not win",
    )
    assert_true(
        weak_operand_type(LogicalType.STRING, LogicalType.INT64)
        == LogicalType.INT64,
        "a text column has no width to give",
    )


def test_resolving_a_constant_twice_is_the_same_answer() raises:
    """`ComputeNode.schema` declares the dtype and `binary_value_any` produces
    it, and they get there by calling this, so the answer has to be the one it
    was the first time. It is a pure function of the dtype and the number and
    this is the test that says so out loud."""
    var value = Value(Int64(2)).weakened()
    var once = resolve_constant(LogicalType.INT8, value, BinaryOp.ADD)
    var twice = resolve_constant(LogicalType.INT8, once, BinaryOp.ADD)
    assert_true(once.type == LogicalType.INT8, "once")
    assert_true(twice.type == LogicalType.INT8, "twice")
    assert_true(not once.weak, "a resolved constant is no longer weak")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
