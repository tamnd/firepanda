"""Columns held as positions into the column their rows came from.

A selection has to read back as exactly the rows a gather would have made, and
a kernel that moves rows has to keep it a selection over the same source rather
than gathering. Each test builds the same rows both ways and compares them.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.encoding import NO_LAYOUT
from firepanda.array.strings import strings_from_list
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.frame.frame import DataFrame
from firepanda.frame.groupby import AggSpec
from firepanda.frame.series import Series
from firepanda.kernel.group import AggKind
from firepanda.kernel.select import (
    SELECT_MIN_ROWS,
    RowPicker,
    filter_any,
    gather_any,
    take_any,
)


def numbers() -> AnyArray:
    var values = Array[DType.int64](6)
    for i in range(6):
        values.set_valid(i, Int64(i * 10))
    values.set_null(4)
    return AnyArray(values^)


def words() -> AnyArray:
    return AnyArray(
        strings_from_list(
            ["a", "bb", "a string too long to inline", "", "cc", "d"]
        )
    )


def select(col: AnyArray, picks: List[Int]) raises -> AnyArray:
    var picker = RowPicker(SELECT_MIN_ROWS)
    return picker.pick(col, picks)


def same(a: AnyArray, b: AnyArray) raises:
    var x = a.decoded()
    var y = b.decoded()
    assert_equal(len(x), len(y), "rows")
    assert_true(x.type == y.type, "type")
    for i in range(len(x)):
        assert_equal(x.is_valid(i), y.is_valid(i), "null at " + String(i))
        if not x.is_valid(i):
            continue
        if x.is_string():
            assert_equal(x.strings()[i], y.strings()[i], "row " + String(i))
        else:
            assert_equal(
                x.as_typed_view[DType.int64]()[i],
                y.as_typed_view[DType.int64]()[i],
                "row " + String(i),
            )


def test_a_selection_reads_back_as_the_gather() raises:
    var picks: List[Int] = [5, 4, -1, 0, 2, 2]
    for col in [numbers(), words()]:
        var held = select(col, picks)
        assert_true(held.is_selected())
        assert_equal(len(held), 6)
        assert_equal(held.null_count(), take_any(col, picks).null_count())
        same(held, take_any(col, picks))
        assert_true(held.decoded().is_flat())


def test_moving_rows_keeps_the_selection() raises:
    var picks: List[Int] = [5, 4, -1, 0, 2, 3]
    for col in [numbers(), words()]:
        var held = select(col, picks)
        var flat = take_any(col, picks)
        var again: List[Int] = [1, 0, 5, -1, 2]
        var taken = take_any(held, again)
        assert_true(taken.is_selected())
        same(taken, take_any(flat, again))
        var picked: List[UInt32] = [0, 2, 3, 5]
        var gathered = gather_any(held, picked)
        assert_true(gathered.is_selected())
        same(gathered, gather_any(flat, picked))
        var mask = Array[DType.bool](6)
        for i in range(6):
            mask.set_valid(i, i % 2 == 0)
        var kept = filter_any(held, mask)
        assert_true(kept.is_selected())
        same(kept, filter_any(flat, mask))
        var cut = held.slice(1, 4)
        assert_true(cut.is_selected())
        same(cut, flat.slice(1, 4))
        var seen = held.window(0, 3)
        assert_true(seen.is_selected())
        same(seen, flat.slice(0, 3))
        if col.is_string():
            assert_equal(held.text_at(2), flat.text_at(2))
            assert_equal(held.text_at(1), flat.text_at(1))


def test_a_selection_is_refused_by_a_kernel_that_reads_values() raises:
    # Its buffer holds positions. A kernel that dispatches on the dtype must
    # not match an arm and read them as values.
    var held = select(numbers(), [0, 1, 2])
    assert_true(held.dtype() == NO_LAYOUT)
    assert_equal(String(held.encoding), "selection")
    with assert_raises(contains="call decoded() first"):
        _ = held.as_typed_view[DType.int64]()[0]
    var text = select(words(), [0, 1, 2])
    assert_false(text.is_string())
    with assert_raises(contains="call decoded() first"):
        _ = len(text.strings())


def test_a_category_or_coded_column_is_gathered_instead() raises:
    var codes = Array[DType.int32](3)
    for i in range(3):
        codes.set_valid(i, Int32(i % 2))
    var coded = AnyArray.dictionary_encoded(
        codes^, strings_from_list(["x", "y"])
    )
    var taken = select(coded, [2, 0])
    assert_false(taken.is_selected())
    assert_true(taken.is_coded())
    with assert_raises(contains="does not hold"):
        _ = AnyArray.selection(coded, Buffer(0), Bitmap(0), 0)


def test_one_side_shares_one_list_of_positions() raises:
    var picks: List[Int] = [3, 1, 2]
    var picker = RowPicker(SELECT_MIN_ROWS)
    var a = picker.pick(numbers(), picks)
    var b = picker.pick(words(), picks)
    assert_true(a.data.values.unsafe_ptr() == b.data.values.unsafe_ptr())
    # And a later join composing both composes once.
    var later = RowPicker(SELECT_MIN_ROWS)
    var again: List[Int] = [2, 0]
    var c = later.pick(a, again)
    var d = later.pick(b, again)
    assert_true(c.data.values.unsafe_ptr() == d.data.values.unsafe_ptr())
    same(c, take_any(take_any(numbers(), picks), again))
    same(d, take_any(take_any(words(), picks), again))


def test_a_short_join_output_is_gathered() raises:
    var picker = RowPicker(SELECT_MIN_ROWS - 1)
    assert_false(picker.pick(numbers(), [1, 2]).is_selected())


def test_a_tall_join_holds_positions_and_a_second_join_thins_them() raises:
    var n = SELECT_MIN_ROWS + 1000
    var keys = Array[DType.int64](n)
    var values = Array[DType.int64](n)
    var right_keys = Array[DType.int64](n)
    var tags = List[String]()
    for i in range(n):
        keys.set_valid(i, Int64(i))
        values.set_valid(i, Int64(i * 3))
        right_keys.set_valid(i, Int64(n - 1 - i))
        tags.append(String("t", n - 1 - i))
    var left_cols = List[Series]()
    left_cols.append(Series("k", AnyArray(keys^)))
    left_cols.append(Series("v", AnyArray(values^)))
    var left = DataFrame.from_series(left_cols^)
    var right_cols = List[Series]()
    right_cols.append(Series("rk", AnyArray(right_keys^)))
    right_cols.append(Series("tag", strings_from_list(tags)))
    var right = DataFrame.from_series(right_cols^)
    var joined = left.join_on(right, ["k"], ["rk"])
    assert_equal(len(joined), n)
    assert_true(joined.columns[1].held().is_selected())
    assert_true(joined.columns[3].held().is_selected())
    # A column handed out as a series is flat.
    assert_true(joined.column("v").values.is_flat())
    # Reading a column gathers it where it lies, and only that one.
    assert_equal(joined[1].as_typed_view[DType.int64]()[5], 15)
    assert_true(joined.columns[1].held().is_flat())
    assert_true(joined.columns[3].held().is_selected())
    # So does a method that reads values.
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM))
    var summed = joined.group_by(["tag"], specs)
    assert_equal(len(summed), n)
    assert_equal(summed.column("tag").text(0), "t0")
    var sorted = joined.sort_by("v", descending=True)
    assert_equal(sorted.column("k").as_typed[DType.int64]()[0], Int64(n - 1))

    var few = Array[DType.int64](3)
    few.set_valid(0, 7)
    few.set_valid(1, Int64(n - 1))
    few.set_valid(2, Int64(n // 2))
    var few_cols = List[Series]()
    few_cols.append(Series("w", AnyArray(few^)))
    var wanted = DataFrame.from_series(few_cols^)
    var thin = joined.join_on(wanted, ["rk"], ["w"])
    assert_equal(len(thin), 3)
    var k = thin.column("k").values.copy()
    var v = thin.column("v").values.copy()
    var tag = thin.column("tag").values.copy()
    for i in range(3):
        var at = k.as_typed_view[DType.int64]()[i]
        assert_equal(v.as_typed_view[DType.int64]()[i], at * 3)
        assert_equal(tag.strings()[i], String("t", at))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
