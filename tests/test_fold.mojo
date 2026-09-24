"""Tests for the sum of a column and a constant that builds no column.

Every answer is checked against the two pass route it replaces, which is
`binary_value_any` and then `reduce_any`, bit for bit. That route is the
definition, and the fused one is only allowed to be faster.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.value import Value
from firepanda.bitmap.bitmap import Bitmap
from firepanda.exec import MORSEL_ROWS
from firepanda.kernel.binary import BinaryOp, binary_value_any
from firepanda.kernel.fold import reduce_value_any
from firepanda.kernel.group import AggKind
from firepanda.kernel.reduce import reduce_any


def _same_as_two_passes(
    a: AnyArray, b: Value, op: BinaryOp, left: Bool, kind: AggKind
) raises:
    var want = reduce_any(binary_value_any(a, b, op, left), kind)
    var got = reduce_value_any(a, b, op, left, kind, False)
    assert_true(Bool(got))
    var answer = got.take()
    assert_equal(len(answer), 1)
    assert_true(answer.type == want.type)
    assert_equal(
        answer.unsafe_ptr[DType.uint64]().unsafe_load(),
        want.unsafe_ptr[DType.uint64]().unsafe_load(),
    )


def _every_pairing(a: AnyArray) raises:
    var constants = [
        Value(Int64(89)),
        Value(Int64(-5)),
        Value(Int64(40000)),
        Value(Int64(9_000_000_000)),
    ]
    for k in range(len(constants)):
        for op in [BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL]:
            for left in [False, True]:
                for kind in [AggKind.SUM, AggKind.COUNT]:
                    _same_as_two_passes(a, constants[k], op, left, kind)


def test_a_narrow_column_widens_the_way_the_two_passes_do() raises:
    var n = 3 * MORSEL_ROWS + 17
    var x = Array[DType.int16](n)
    for i in range(n):
        x[i] = Int16((i * 7919) % 65536 - 32768)
    _every_pairing(AnyArray(x^))


def test_an_int64_sum_that_wraps_wraps_the_same_way() raises:
    var n = MORSEL_ROWS + 3
    var x = Array[DType.int64](n)
    for i in range(n):
        x[i] = Int64(i) * 3_000_000_000_000_000
    _every_pairing(AnyArray(x^))


def test_an_unsigned_column_answers_as_the_two_passes_do() raises:
    var n = 1001
    var x = Array[DType.uint8](n)
    for i in range(n):
        x[i] = UInt8(i % 256)
    _every_pairing(AnyArray(x^))


def test_a_null_row_adds_nothing_and_is_not_counted() raises:
    var n = 2 * MORSEL_ROWS + 5
    var x = Array[DType.int32](n)
    for i in range(n):
        x[i] = Int32(i * 104729)
    var a = AnyArray(x^)
    var bits = Bitmap(n)
    for i in range(0, n, 3):
        bits.set(i, False)
    a.data.validity = bits^
    _every_pairing(a)


def test_a_float_column_is_left_to_the_two_passes() raises:
    var x = Array[DType.float64](10)
    var got = reduce_value_any(
        AnyArray(x^), Value(Int64(1)), BinaryOp.ADD, False, AggKind.SUM, False
    )
    assert_false(Bool(got))


def test_a_division_is_left_to_the_two_passes() raises:
    var x = Array[DType.int64](10)
    var got = reduce_value_any(
        AnyArray(x^),
        Value(Int64(2)),
        BinaryOp.SQLDIV,
        False,
        AggKind.SUM,
        False,
    )
    assert_false(Bool(got))


def test_a_sum_taken_as_a_float_is_left_to_the_two_passes() raises:
    var x = Array[DType.int64](10)
    var got = reduce_value_any(
        AnyArray(x^), Value(Int64(2)), BinaryOp.ADD, False, AggKind.SUM, True
    )
    assert_false(Bool(got))


def test_a_minimum_is_left_to_the_two_passes() raises:
    var x = Array[DType.int64](10)
    var got = reduce_value_any(
        AnyArray(x^), Value(Int64(2)), BinaryOp.ADD, False, AggKind.MIN, False
    )
    assert_false(Bool(got))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
