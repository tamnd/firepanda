"""Tests for the four operations that read one column and write one.

These are not four spellings of the same loop. `-x` and `~x` are the same
operation on a bool column and two different ones on an integer column, `abs` is
the column unchanged on half the dtypes there are, and `+x` is the column
unchanged on all of them. So most of what is asserted here is which of those a
type gets rather than whether a sign got flipped.

Every value was read off a running pandas 3.0 rather than worked out here, which
matters more than usual because pandas does not pass these through to numpy. On
a bool column numpy refuses `-x` outright and pandas answers the logical not, so
copying numpy would have produced a library that raises on an expression pandas
answers.

The wrapping cases are here for the same reason the divide by zero cases are in
`test_binary.mojo`: they are the places where an implementation might reasonably
have decided to raise, and pandas does not, so a later change that starts
raising should fail a test rather than read like an improvement.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import strings_from_list
from firepanda.dtype.logical import LogicalType
from firepanda.kernel.unary import (
    UnaryOp,
    absolute,
    invert,
    negate,
    unary_any,
    unary_type,
)


def typed[dt: DType](values: List[Scalar[dt]]) raises -> Array[dt]:
    """Builds a fully valid column of one dtype."""
    var col = Array[dt](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    return col^


def erased[dt: DType](values: List[Scalar[dt]]) raises -> AnyArray:
    """Builds the same column with its dtype carried as a field."""
    return AnyArray(typed[dt](values))


def read[dt: DType](col: AnyArray) raises -> List[Scalar[dt]]:
    """Reads a column out as a plain list."""
    ref view = col.as_typed_view[dt]()
    var out = List[Scalar[dt]](capacity=len(view))
    for i in range(len(view)):
        out.append(view[i])
    return out^


def test_negating_a_column_flips_every_sign() raises:
    var got = negate(typed[DType.int64]([1, -2, 0]))
    assert_equal(got[0], Int64(-1), "row 0")
    assert_equal(got[1], Int64(2), "row 1")
    assert_equal(got[2], Int64(0), "row 2")


def test_negating_the_most_negative_integer_wraps() raises:
    """`-Series([-128], dtype='int8')` is `[-128]` in pandas, because the
    positive 128 does not fit and the hardware answers the input again. A
    checked negation would be a divergence rather than a fix."""
    var got = negate(typed[DType.int8]([-128, -3, 7]))
    assert_equal(got[0], Int8(-128), "row 0")
    assert_equal(got[1], Int8(3), "row 1")
    assert_equal(got[2], Int8(-7), "row 2")


def test_negating_an_unsigned_column_wraps_to_the_complement() raises:
    """`-Series([1], dtype='uint8')` is `[255]`. There is no negative in the
    type, so the answer is the two's complement read as unsigned, which is what
    pandas hands back and what the instruction produces."""
    var got = negate(typed[DType.uint8]([1, 0, 200]))
    assert_equal(got[0], UInt8(255), "row 0")
    assert_equal(got[1], UInt8(0), "row 1")
    assert_equal(got[2], UInt8(56), "row 2")


def test_negating_a_zero_float_gives_a_signed_zero() raises:
    """This is the test that rules out writing `-x` as `0 - x`. `-Series([0.0])`
    has the sign bit set in pandas and `0.0 - 0.0` does not, so the two
    expressions differ on a value that compares equal to itself and would never
    have shown up in a test that only compared numbers."""
    var got = negate(typed[DType.float64]([0.0, -0.0]))
    assert_true(Float64(1.0) / got[0] < 0, "negative zero on row 0")
    assert_true(Float64(1.0) / got[1] > 0, "positive zero on row 1")


def test_negating_a_bool_column_is_the_logical_not() raises:
    """A column of True and False comes back False and True.

    numpy refuses this expression and says the boolean negative is not
    supported. pandas catches that and answers the inversion instead."""
    var got = negate(typed[DType.bool]([True, False, True]))
    assert_equal(got[0], False, "row 0")
    assert_equal(got[1], True, "row 1")
    assert_equal(got[2], False, "row 2")


def test_the_absolute_value_flips_only_the_negatives() raises:
    var got = absolute(typed[DType.float64]([-1.5, 2.5, -0.0]))
    assert_equal(got[0], Float64(1.5), "row 0")
    assert_equal(got[1], Float64(2.5), "row 1")
    assert_true(Float64(1.0) / got[2] > 0, "positive zero on row 2")


def test_the_absolute_value_of_the_most_negative_integer_wraps() raises:
    """`abs(Series([-128], dtype='int8'))` is `[-128]`, the same wrap as the
    negation and for the same reason."""
    var got = absolute(typed[DType.int8]([-128, -3, 7]))
    assert_equal(got[0], Int8(-128), "row 0")
    assert_equal(got[1], Int8(3), "row 1")
    assert_equal(got[2], Int8(7), "row 2")


def test_the_absolute_value_of_a_bool_column_is_the_column() raises:
    """`abs(Series([True, False]))` is `[True, False]`, still bool. The values
    are already zero and one, so this is a copy and not a pass."""
    var got = absolute(typed[DType.bool]([True, False]))
    assert_equal(got[0], True, "row 0")
    assert_equal(got[1], False, "row 1")


def test_the_absolute_value_of_an_unsigned_column_is_the_column() raises:
    var got = absolute(typed[DType.uint8]([0, 1, 255]))
    assert_equal(got[0], UInt8(0), "row 0")
    assert_equal(got[1], UInt8(1), "row 1")
    assert_equal(got[2], UInt8(255), "row 2")


def test_inverting_an_integer_column_is_the_bitwise_not() raises:
    """`~Series([1, 2])` is `[-2, -3]` and not `[False, False]`, which is the
    difference between this operator on an integer column and the same operator
    on a bool one."""
    var got = invert(typed[DType.int64]([1, 2, 0, -1]))
    assert_equal(got[0], Int64(-2), "row 0")
    assert_equal(got[1], Int64(-3), "row 1")
    assert_equal(got[2], Int64(-1), "row 2")
    assert_equal(got[3], Int64(0), "row 3")


def test_inverting_an_unsigned_column_is_still_the_bitwise_not() raises:
    """`~Series([0], dtype='uint8')` is `[255]`, so the operator does not
    quietly become a logical not on a type whose values are all positive."""
    var got = invert(typed[DType.uint8]([0, 1]))
    assert_equal(got[0], UInt8(255), "row 0")
    assert_equal(got[1], UInt8(254), "row 1")


def test_inverting_a_bool_column_is_the_logical_not() raises:
    var got = invert(typed[DType.bool]([True, False]))
    assert_equal(got[0], False, "row 0")
    assert_equal(got[1], True, "row 1")


def test_a_null_stays_null_through_every_operation() raises:
    """None of the three loops can change which rows are present, so the null
    count is the input's null count in all of them. The value underneath a null
    is repaired to zero the same way the binary kernels repair it."""
    var col = Array[DType.int32](3)
    col.set_valid(0, -5)
    col.set_null(1)
    col.set_valid(2, 6)

    var negated = negate(Array[DType.int32](copy=col))
    assert_equal(negated.null_count(), 1, "nulls after negating")
    assert_true(not negated.is_valid(1), "row 1 still null")
    assert_equal(negated[1], Int32(0), "value under the null")

    var magnitude = absolute(Array[DType.int32](copy=col))
    assert_equal(magnitude.null_count(), 1, "nulls after the absolute value")
    assert_equal(magnitude[0], Int32(5), "row 0")

    var inverted = invert(Array[DType.int32](copy=col))
    assert_equal(inverted.null_count(), 1, "nulls after inverting")
    assert_equal(inverted[0], Int32(4), "row 0")


def test_every_row_of_a_long_column_is_reached() raises:
    """Long enough to run as more than one morsel, so that the repair of the
    null rows is checked on a range that does not start at zero. The nulls are
    spread on a stride that is not a multiple of the register width or of the
    morsel size."""
    var n = 300000
    var col = Array[DType.int64](n)
    for i in range(n):
        if i % 9973 == 11:
            col.set_null(i)
        else:
            col.set_valid(i, Int64(i) - 150000)

    var got = negate(col^)
    assert_equal(len(got), n, "length")
    for i in range(n):
        if i % 9973 == 11:
            assert_true(not got.is_valid(i), String("row ", i, " null"))
            assert_equal(got[i], Int64(0), String("row ", i, " value"))
        else:
            assert_equal(got[i], Int64(150000 - i), String("row ", i))


def test_unary_plus_is_the_column_unchanged() raises:
    """On every type that has it at all, including bool, where pandas answers
    the column back rather than promoting it to an integer."""
    var numbers = unary_any(erased[DType.int64]([1, -2]), UnaryOp.POS)
    assert_true(numbers.type == LogicalType.INT64, "number type")
    assert_equal(read[DType.int64](numbers)[0], Int64(1), "row 0")
    assert_equal(read[DType.int64](numbers)[1], Int64(-2), "row 1")

    var flags = unary_any(erased[DType.bool]([True, False]), UnaryOp.POS)
    assert_true(flags.type == LogicalType.BOOL, "bool type")
    assert_equal(read[DType.bool](flags)[0], True, "row 0")


def test_the_erased_entry_point_keeps_the_operand_type() raises:
    """All four answer the type they were given, which is what makes
    `unary_type` a check rather than a computation."""
    var negated = unary_any(erased[DType.int32]([1, -2]), UnaryOp.NEG)
    assert_true(negated.type == LogicalType.INT32, "negated type")
    assert_equal(read[DType.int32](negated)[1], Int32(2), "row 1")

    var magnitude = unary_any(erased[DType.float32]([-1.5]), UnaryOp.ABS)
    assert_true(magnitude.type == LogicalType.FLOAT32, "absolute type")
    assert_equal(read[DType.float32](magnitude)[0], Float32(1.5), "row 0")

    var inverted = unary_any(erased[DType.bool]([True]), UnaryOp.INVERT)
    assert_true(inverted.type == LogicalType.BOOL, "inverted type")
    assert_equal(read[DType.bool](inverted)[0], False, "row 0")


def test_the_result_type_is_known_without_a_column() raises:
    """A plan has to be able to ask, which is the whole reason this is a
    function of the type rather than something the loop finds out."""
    assert_true(
        unary_type(UnaryOp.NEG, LogicalType.INT16) == LogicalType.INT16,
        "negated int16",
    )
    assert_true(
        unary_type(UnaryOp.ABS, LogicalType.FLOAT64) == LogicalType.FLOAT64,
        "absolute float64",
    )
    assert_true(
        unary_type(UnaryOp.INVERT, LogicalType.BOOL) == LogicalType.BOOL,
        "inverted bool",
    )
    assert_true(
        unary_type(UnaryOp.POS, LogicalType.UINT8) == LogicalType.UINT8,
        "unary plus on uint8",
    )


def test_inverting_a_float_column_is_refused() raises:
    """`~Series([1.0])` raises in pandas too, saying the invert ufunc is not
    supported for the input types. There is no bitwise not of a float and
    answering one would have to invent a meaning for it."""
    with assert_raises(contains="~"):
        _ = unary_any(erased[DType.float64]([1.0]), UnaryOp.INVERT)


def test_a_text_column_has_no_unary_arithmetic() raises:
    """Refusing is the same answer as refusing to add a string to a number.

    pandas raises a `TypeError` naming the operator here."""
    var text = AnyArray(strings_from_list(["a", "b"]))
    with assert_raises(contains="is not defined on"):
        _ = unary_any(text, UnaryOp.NEG)


def test_every_operation_prints_as_the_symbol_it_is_written_with() raises:
    """The messages above quote the operator, so the rendering is part of what a
    caller reads when something is refused."""
    assert_equal(String(UnaryOp.NEG), "-", "negation")
    assert_equal(String(UnaryOp.POS), "+", "unary plus")
    assert_equal(String(UnaryOp.ABS), "abs", "absolute value")
    assert_equal(String(UnaryOp.INVERT), "~", "inversion")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
