"""Kernels that read a string column held as codes into its distinct values.

Each test runs the same operation on the encoded column and on its decoded twin
and asks for the same answer, because that is the whole promise of an encoding:
the user cannot tell. The column has a category longer than twelve bytes, so
the payload side of a view is exercised, and a null row, so the codes' validity
is.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import strings_from_list
from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.frame import DataFrame
from firepanda.frame.groupby import AggKind, AggSpec
from firepanda.frame.series import Series
from firepanda.kernel.binary import BinaryOp, binary_value_any
from firepanda.io.arrow_export import export_array, export_schema
from firepanda.io.arrow_import import import_array
from firepanda.io.arrow_ipc import read_ipc_stream
from firepanda.io.arrow_ipc_write import write_ipc_stream_bytes
from firepanda.kernel.member import is_in_any
from firepanda.kernel.sort import argsort_any, argsort_multi, is_sorted_any
from firepanda.kernel.select import filter_any, gather_any, take_any


def status() -> AnyArray:
    var codes = Array[DType.int32](8)
    var picked: List[Int32] = [2, 0, 1, 0, 1, 2, 0, 0]
    for i in range(8):
        codes.set_valid(i, picked[i])
    codes.set_null(7)
    return AnyArray.dictionary_encoded(
        codes^, strings_from_list(["ok", "late", "a status too long to inline"])
    )


def same_text(a: AnyArray, b: AnyArray) raises:
    ref x = a.decoded().strings()
    ref y = b.decoded().strings()
    assert_equal(len(x), len(y), "rows")
    for i in range(len(x)):
        assert_equal(x.is_valid(i), y.is_valid(i), "null at " + String(i))
        if x.is_valid(i):
            assert_equal(x[i], y[i], "row " + String(i))


def same_mask(a: Array[DType.bool], b: Array[DType.bool]) raises:
    assert_equal(len(a), len(b), "rows")
    for i in range(len(a)):
        assert_equal(a.is_valid(i), b.is_valid(i), "null at " + String(i))
        if a.is_valid(i):
            assert_equal(a[i], b[i], "row " + String(i))


def test_moving_rows_keeps_the_encoding() raises:
    var col = status()
    var flat = col.decoded()
    var taken = take_any(col, [5, -1, 2, 7, 0])
    assert_false(taken.is_flat())
    same_text(taken, take_any(flat, [5, -1, 2, 7, 0]))
    var picks: List[UInt32] = [0, 3, 4, 7]
    var gathered = gather_any(col, picks)
    assert_false(gathered.is_flat())
    same_text(gathered, gather_any(flat, picks))
    var mask = Array[DType.bool](8)
    for i in range(8):
        mask.set_valid(i, i % 3 != 1)
    var kept = filter_any(col, mask)
    assert_false(kept.is_flat())
    assert_true(kept.type == LogicalType.STRING)
    same_text(kept, filter_any(flat, mask))


def test_a_comparison_with_a_constant_matches_the_flat_one() raises:
    var col = status()
    var flat = col.decoded()
    var ops: List[BinaryOp] = [
        BinaryOp.EQ,
        BinaryOp.NE,
        BinaryOp.LT,
        BinaryOp.GE,
    ]
    for k in range(len(ops)):
        var got = binary_value_any(col, Value(String("late")), ops[k])
        var want = binary_value_any(flat, Value(String("late")), ops[k])
        same_mask(got.as_typed[DType.bool](), want.as_typed[DType.bool]())
    # A constant on the left is turned round, and has to come out the same.
    var left = binary_value_any(
        col, Value(String("m")), BinaryOp.LT, value_on_left=True
    )
    var left_flat = binary_value_any(
        flat, Value(String("m")), BinaryOp.LT, value_on_left=True
    )
    same_mask(left.as_typed[DType.bool](), left_flat.as_typed[DType.bool]())


def test_a_lookup_in_a_set_matches_the_flat_one() raises:
    var col = status()
    var wanted = AnyArray(
        strings_from_list(["late", "a status too long to inline"])
    )
    same_mask(is_in_any(col, wanted), is_in_any(col.decoded(), wanted))
    # The set held encoded as well.
    same_mask(is_in_any(col, col), is_in_any(col.decoded(), col.decoded()))


def frame(var text: AnyArray) raises -> DataFrame:
    var n = Array[DType.int64](8)
    for i in range(8):
        n.set_valid(i, Int64(i + 1))
    var columns = List[AnyArray]()
    columns.append(text^)
    columns.append(AnyArray(n^))
    var fields = List[Field]()
    fields.append(Field("status", LogicalType.STRING))
    fields.append(Field("n", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def test_a_group_by_on_an_encoded_key_matches_the_flat_one() raises:
    var specs: List[AggSpec] = [AggSpec("n", AggKind.SUM, "total")]
    var by: List[String] = ["status"]
    var got = frame(status()).group_by(by.copy(), specs.copy(), False, True)
    var want = frame(status().decoded()).group_by(by^, specs^, False, True)
    assert_equal(len(got), len(want), "groups")
    assert_true(got[0].is_flat(), "a group's key comes out flat")
    same_text(got[0], want[0])
    ref a = got[1].as_typed_view[DType.int64]()
    ref b = want[1].as_typed_view[DType.int64]()
    for i in range(len(a)):
        assert_equal(a[i], b[i], "group " + String(i))


def test_the_text_tests_match_the_flat_ones() raises:
    var held = Series(String("s"), status())
    var flat = Series(String("s"), status().decoded())
    same_mask(held.str_contains("at"), flat.str_contains("at"))
    same_mask(held.str_contains("long"), flat.str_contains("long"))
    same_mask(
        held.str_contains_in_order("a", "o"),
        flat.str_contains_in_order("a", "o"),
    )
    same_mask(held.str_starts_with("a st"), flat.str_starts_with("a st"))
    same_mask(held.str_ends_with("te"), flat.str_ends_with("te"))
    same_text(held.str_slice(0, 3).values, flat.str_slice(0, 3).values)


def same_order(a: Array[DType.uint32], b: Array[DType.uint32]) raises:
    assert_equal(len(a), len(b), "rows")
    for i in range(len(a)):
        assert_equal(a[i], b[i], "place " + String(i))


def test_a_sort_on_an_encoded_key_matches_the_flat_one() raises:
    var col = status()
    var flat = col.decoded()
    for way in range(4):
        var descending = way % 2 == 1
        var nulls_first = way >= 2
        same_order(
            argsort_any(col, descending, nulls_first),
            argsort_any(flat, descending, nulls_first),
        )
    # Ties on the encoded key are broken by the next one, so the encoded pass
    # has to be stable over the order the later key gave.
    var n = Array[DType.int64](8)
    var picked: List[Int64] = [3, 1, 4, 1, 5, 9, 2, 6]
    for i in range(8):
        n.set_valid(i, picked[i])
    var held: List[AnyArray] = [col.copy(), AnyArray(n.copy())]
    var bare: List[AnyArray] = [flat.copy(), AnyArray(n^)]
    var descending: List[Bool] = [False, True]
    var nulls_first: List[Bool] = [True, False]
    same_order(
        argsort_multi(held, descending, nulls_first),
        argsort_multi(bare, descending, nulls_first),
    )
    assert_equal(is_sorted_any(col), is_sorted_any(flat))
    var by: List[String] = ["status"]
    var up: List[Bool] = [False]
    same_order(
        frame(status()).argsort_limit(by.copy(), up.copy(), up.copy(), 3),
        frame(flat.copy()).argsort_limit(by.copy(), up.copy(), up.copy(), 3),
    )
    var order = argsort_any(col)
    var picks = List[UInt32]()
    for i in range(len(order)):
        picks.append(order[i])
    var sorted = gather_any(col, picks)
    assert_false(sorted.is_flat())
    assert_true(is_sorted_any(sorted))


def test_an_export_hands_out_the_strings() raises:
    var schema = export_schema(LogicalType.STRING)
    var array = export_array(status())
    var back = import_array(schema, array)
    assert_true(back.is_flat())
    same_text(back, status())
    var bytes = write_ipc_stream_bytes(frame(status()))
    var read = read_ipc_stream(Span(bytes))
    assert_true(read[0].is_flat())
    same_text(read[0], status())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
