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
from firepanda.io.write import WriteOptions, write_csv_bytes
from firepanda.kernel.cast import cast_any
from firepanda.kernel.concat import concat_two_any
from firepanda.kernel.nulls import coalesce_any, fill_forward_any
from firepanda.kernel.reduce import distinct_count_any, reduce_any
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


def same_numbers[dt: DType](a: AnyArray, b: AnyArray) raises:
    ref x = a.as_typed_view[dt]()
    ref y = b.as_typed_view[dt]()
    assert_equal(len(x), len(y), "rows")
    for i in range(len(x)):
        assert_equal(x.is_valid(i), y.is_valid(i), "null at " + String(i))
        if x.is_valid(i):
            assert_equal(x[i], y[i], "row " + String(i))


def test_the_text_methods_match_the_flat_ones() raises:
    var held = Series(String("s"), status())
    var flat = Series(String("s"), status().decoded())
    same_text(held.chars_upper().values, flat.chars_upper().values)
    same_text(
        held.chars_replace("at", "AT", -1).values,
        flat.chars_replace("at", "AT", -1).values,
    )
    same_text(held.chars_get(1).values, flat.chars_get(1).values)
    same_numbers[DType.int64](
        held.chars_length().values, flat.chars_length().values
    )
    same_numbers[DType.int64](
        held.chars_find("t", None, None, False).values,
        flat.chars_find("t", None, None, False).values,
    )
    same_numbers[DType.bool](
        held.chars_is_alpha().values, flat.chars_is_alpha().values
    )
    var parts = held.chars_partition(" ", False)
    var bare = flat.chars_partition(" ", False)
    for k in range(3):
        same_text(parts[k].values, bare[k].values)
    for i in range(8):
        assert_equal(held.text(i), flat.text(i), "text " + String(i))
    same_text(AnyArray(held.as_strings()), AnyArray(flat.as_strings()))
    assert_equal(held.chars_upper().name.value(), "s")


def test_a_string_answer_per_category_goes_out_flat() raises:
    # Upper casing two categories into one answer has to leave codes into a
    # list without a repeat in it, so the answer comes back flat.
    var codes = Array[DType.int32](4)
    var picked: List[Int32] = [0, 1, 1, 0]
    for i in range(4):
        codes.set_valid(i, picked[i])
    var col = AnyArray.dictionary_encoded(codes^, strings_from_list(["a", "A"]))
    var up = Series(String("s"), col^).chars_upper()
    assert_true(up.values.is_flat())
    for i in range(4):
        assert_equal(up.text(i), "A")


def test_the_kernels_that_decode_match_the_flat_ones() raises:
    var col = status()
    var flat = col.decoded()
    same_text(concat_two_any(col, flat), concat_two_any(flat, flat))
    same_text(fill_forward_any(col), fill_forward_any(flat))
    same_text(coalesce_any(col, flat), coalesce_any(flat, flat))
    same_text(
        cast_any(col, LogicalType.STRING), cast_any(flat, LogicalType.STRING)
    )
    same_text(reduce_any(col, AggKind.MIN), reduce_any(flat, AggKind.MIN))
    same_text(reduce_any(col, AggKind.MAX), reduce_any(flat, AggKind.MAX))
    assert_equal(distinct_count_any(col), 3)
    # A filter keeps every category, so a count of them would say three here.
    var mask = Array[DType.bool](8)
    for i in range(8):
        mask.set_valid(i, i != 0 and i != 5)
    var kept = filter_any(col, mask)
    assert_false(kept.is_flat())
    assert_equal(distinct_count_any(kept), 2)
    same_text(
        reduce_any(kept, AggKind.MAX), reduce_any(kept.decoded(), AggKind.MAX)
    )


def test_a_frame_with_a_column_held_as_codes_joins_and_writes() raises:
    var on: List[String] = ["status"]
    var got = frame(status()).join_on(frame(status()), on, on)
    var want = frame(status().decoded()).join_on(
        frame(status().decoded()), on, on
    )
    assert_equal(len(got), len(want), "rows")
    for c in range(got.width()):
        if want[c].is_string():
            same_text(got[c], want[c])
    var specs: List[AggSpec] = [AggSpec("status", AggKind.MAX, "top")]
    var by: List[String] = ["n"]
    var grouped = frame(status()).group_by(by.copy(), specs.copy(), False, True)
    var bare = frame(status().decoded()).group_by(by^, specs^, False, True)
    same_text(grouped[1], bare[1])
    assert_equal(
        String(from_utf8=Span(write_csv_bytes(frame(status()), WriteOptions()))),
        String(
            from_utf8=Span(
                write_csv_bytes(frame(status().decoded()), WriteOptions())
            )
        ),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
