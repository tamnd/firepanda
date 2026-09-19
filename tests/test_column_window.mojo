"""Tests for sharing part of a column instead of copying it.

A window is what a slice would be if it did not allocate. The pieces a scan
hands out are windows, so what is checked here is that a window reads exactly
what the slice of the same range reads, that writing through one leaves the
column it came from alone, and that taking one really does share the bytes
rather than quietly copying them and giving the right answer anyway.

The last of those is the one worth having a test for. A window that copied would
pass every correctness test in this file and would give back the whole reason
for having it, which is the same trap `test_buffer.mojo` describes for the
sharing copy, so the addresses are compared and not just the contents.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.array.data import ColumnData
from firepanda.array.nested import ITEM, ROOT, NestedNode
from firepanda.array.strings import StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import ALIGNMENT, Buffer
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.exec.morsel import MORSEL_ROWS
from firepanda.exec import Node, Pipeline, Project, Scan
from firepanda.frame.frame import DataFrame


def test_a_buffer_window_shares_the_allocation() raises:
    var parent = Buffer(4 * ALIGNMENT)
    for i in range(4 * ALIGNMENT):
        parent.unsafe_mut_ptr().unsafe_offset(i).unsafe_write(UInt8(i & 0xFF))

    var window = Buffer(window_of=parent, at=ALIGNMENT, size=2 * ALIGNMENT)
    assert_equal(len(window), 2 * ALIGNMENT)
    assert_true(window.is_shared())
    assert_equal(Int(window.unsafe_ptr()), Int(parent.unsafe_ptr()) + ALIGNMENT)
    for i in range(2 * ALIGNMENT):
        assert_equal(
            window.unsafe_ptr().unsafe_offset(i).unsafe_load(),
            UInt8((ALIGNMENT + i) & 0xFF),
        )


def test_a_buffer_window_is_still_aligned() raises:
    var parent = Buffer(8 * ALIGNMENT)
    for at in [0, ALIGNMENT, 3 * ALIGNMENT, 7 * ALIGNMENT]:
        var window = Buffer(window_of=parent, at=at, size=ALIGNMENT)
        assert_true(window.is_aligned())


def test_writing_a_window_leaves_the_parent_alone() raises:
    var parent = Buffer(4 * ALIGNMENT)
    for i in range(4 * ALIGNMENT):
        parent.unsafe_mut_ptr().unsafe_offset(i).unsafe_write(UInt8(7))

    var window = Buffer(window_of=parent, at=ALIGNMENT, size=ALIGNMENT)
    for i in range(ALIGNMENT):
        window.unsafe_mut_ptr().unsafe_offset(i).unsafe_write(UInt8(9))

    # The window took its own copy on the write, so it reads nines and the
    # buffer it was cut from still reads sevens all the way across.
    for i in range(ALIGNMENT):
        assert_equal(
            window.unsafe_ptr().unsafe_offset(i).unsafe_load(), UInt8(9)
        )
    for i in range(4 * ALIGNMENT):
        assert_equal(
            parent.unsafe_ptr().unsafe_offset(i).unsafe_load(), UInt8(7)
        )
    assert_false(window.is_shared())


def test_a_window_that_went_private_has_zero_padding() raises:
    # The last window of a column is the one that can end partway through a
    # 64-byte unit, and while it shares it borrows the parent's own padding for
    # the tail. The moment it is written it is an ordinary buffer again, so it
    # owes the zero padding that every allocated buffer owes.
    var parent = Buffer(3 * ALIGNMENT + 8)
    for i in range(3 * ALIGNMENT + 8):
        parent.unsafe_mut_ptr().unsafe_offset(i).unsafe_write(UInt8(0xFF))

    var window = Buffer(window_of=parent, at=2 * ALIGNMENT, size=ALIGNMENT + 8)
    window.unsafe_mut_ptr().unsafe_offset(0).unsafe_write(UInt8(1))
    assert_false(window.is_shared())
    for i in range(ALIGNMENT + 8, window.capacity()):
        assert_equal(
            window.unsafe_ptr().unsafe_offset(i).unsafe_load(), UInt8(0)
        )
    # And the parent it was cut from still reads what it read.
    for i in range(3 * ALIGNMENT + 8):
        assert_equal(
            parent.unsafe_ptr().unsafe_offset(i).unsafe_load(), UInt8(0xFF)
        )


def test_a_bitmap_window_reads_what_the_slice_reads() raises:
    var bits = ALIGNMENT * 8 * 3
    var parent = Bitmap(bits, all_valid=True)
    for i in range(0, bits, 3):
        parent.set(i, False)

    var at = ALIGNMENT * 8
    var window = Bitmap(window_of=parent, at=at, length=ALIGNMENT * 8)
    var copied = parent.slice(at, at + ALIGNMENT * 8)
    assert_equal(len(window), len(copied))
    for i in range(len(window)):
        assert_equal(window.get(i), copied.get(i))
    assert_equal(window.count_ones(), copied.count_ones())


def test_a_bitmap_window_that_runs_to_the_end_may_be_short() raises:
    var bits = ALIGNMENT * 8 + 13
    var parent = Bitmap(bits, all_valid=True)
    parent.set(ALIGNMENT * 8 + 5, False)

    var window = Bitmap(window_of=parent, at=ALIGNMENT * 8, length=13)
    assert_equal(len(window), 13)
    assert_equal(window.count_ones(), 12)
    assert_false(window.all_valid())


def _numbers(rows: Int) raises -> AnyArray:
    var column = Array[DType.int64](rows)
    for i in range(rows):
        column[i] = Int64(i * 3)
    for i in range(0, rows, 7):
        column.set_null(i)
    return AnyArray(column^)


def test_a_column_window_reads_what_the_slice_reads() raises:
    var rows = MORSEL_ROWS * 2 + 11
    var column = _numbers(rows)
    var starts = [0, MORSEL_ROWS, MORSEL_ROWS * 2]
    for k in range(len(starts)):
        var at = starts[k]
        var take = rows - at
        if take > MORSEL_ROWS:
            take = MORSEL_ROWS
        var window = column.window(at, take)
        var copied = column.slice(at, at + take)
        assert_equal(len(window), len(copied))
        assert_equal(window.null_count(), copied.null_count())
        for i in range(len(window)):
            assert_equal(window.is_valid(i), copied.is_valid(i))
            if window.is_valid(i):
                assert_equal(
                    window.unsafe_ptr[DType.int64]()
                    .unsafe_offset(i)
                    .unsafe_load(),
                    copied.unsafe_ptr[DType.int64]()
                    .unsafe_offset(i)
                    .unsafe_load(),
                )


def test_a_column_window_copies_nothing() raises:
    var column = _numbers(MORSEL_ROWS * 2)
    var window = column.window(MORSEL_ROWS, MORSEL_ROWS)
    assert_equal(
        Int(window.unsafe_ptr[DType.int64]()),
        Int(column.unsafe_ptr[DType.int64]()) + MORSEL_ROWS * 8,
    )


def _name(i: Int) -> String:
    return String("name-of-element-number-", i)


def test_a_window_of_a_column_of_names_reads_the_same_bytes() raises:
    var rows = MORSEL_ROWS + 5
    var checked = [
        0,
        1,
        1000,
        MORSEL_ROWS - 1,
        MORSEL_ROWS,
        MORSEL_ROWS + 1,
        MORSEL_ROWS + 2,
        MORSEL_ROWS + 3,
        MORSEL_ROWS + 4,
    ]
    var builder = StringBuilder(capacity=rows)
    for i in range(rows):
        # Every element is longer than the twelve bytes that fit inside a view,
        # so the window is exercised against the block and offset route. Only
        # the elements that get read back are given their own text, because
        # formatting a hundred and thirty thousand strings is most of what this
        # test would otherwise spend its time on.
        if i in checked:
            builder.append(_name(i).as_bytes())
        else:
            builder.append("an element out in the payload".as_bytes())
    var column = AnyArray(builder^.finish())

    var window = column.window(MORSEL_ROWS, 5)
    assert_equal(len(window), 5)
    for i in range(5):
        assert_equal(
            String(
                StringSlice(unsafe_from_utf8=window.strings().unsafe_bytes(i))
            ),
            _name(MORSEL_ROWS + i),
        )

    var first = column.window(0, MORSEL_ROWS)
    assert_equal(len(first), MORSEL_ROWS)
    for i in [0, 1, 1000, MORSEL_ROWS - 1]:
        assert_equal(
            String(
                StringSlice(unsafe_from_utf8=first.strings().unsafe_bytes(i))
            ),
            _name(i),
        )


def test_a_window_outside_the_column_is_an_error() raises:
    var column = _numbers(MORSEL_ROWS)
    var raised = False
    try:
        _ = column.window(MORSEL_ROWS, 1)
    except:
        raised = True
    assert_true(raised)


def _lists(rows: Int) raises -> AnyArray:
    """Builds a list column of `rows` rows, each holding one element."""
    var offsets = List[Int32](capacity=rows + 1)
    var items = List[Int64](capacity=rows)
    for i in range(rows + 1):
        offsets.append(Int32(i))
    for i in range(rows):
        items.append(Int64(i))

    var edges = AnyArray(from_list[DType.int32](offsets))
    var storage = edges^.into_node(String(ITEM), ROOT).take_data()
    storage.validity = Bitmap(rows)
    storage.length = rows

    var nodes = List[NestedNode]()
    nodes.append(
        NestedNode(
            String("col"), LogicalType.list_of(DType.int64), ROOT, storage^
        )
    )
    var leaf = AnyArray(from_list[DType.int64](items))
    nodes.append(leaf^.into_node(String(ITEM), 0))
    return AnyArray.nested_from(nodes^)


def _nested_frame(rows: Int) raises -> DataFrame:
    """One chunk of `rows` rows, a number beside a list of one element."""
    var plain = Array[DType.int64](rows)
    for i in range(rows):
        plain[i] = Int64(i)

    var columns = List[AnyArray]()
    columns.append(AnyArray(plain^))
    columns.append(_lists(rows))
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("items", LogicalType.list_of(DType.int64)))
    return DataFrame(Schema(fields^), columns^)


def test_a_frame_holding_a_list_column_is_not_cut_at_all() raises:
    # A scan that is asked to cut a tall chunk into morsels cuts every column,
    # and a list column is the one shape it cannot cut. If it cut the others
    # anyway the frame would be chunked differently from column to column and
    # the scan would refuse its own work, so asking is refused quietly here and
    # the frame is left whole.
    var rows = MORSEL_ROWS + 3
    var scan = Scan(_nested_frame(rows))
    scan.cut()
    assert_equal(scan.num_chunks(), 1, "the frame is left whole")

    var pipeline = Pipeline(_nested_frame(rows))
    pipeline.add(Node(Project([0])))
    var out = pipeline^.run()
    assert_equal(out.rows, rows)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
