"""Tests for the column types.

The round trip test is the M0 exit criterion: for every dtype in `ALL`, build a
typed array, write values and nulls into it, erase the dtype, put it back, and get
the same values out. The erasure step is the one that has been rewritten most
often, because moving buffers out of a typed column into an untyped one without
copying them is the whole reason `ColumnData` exists.
"""

from std.builtin.rebind import rebind
from std.sys.info import size_of
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.array.data import ColumnData
from firepanda.array.chunked import ChunkedArray
from firepanda.array.strings import StringBuilder
from firepanda.dtype.lists import ALL, NUMERIC
from firepanda.dtype.logical import LogicalType, logical_for


def sample[dt: DType](i: Int) -> Scalar[dt]:
    """Returns a value of `dt` that varies with the position.

    Args:
        i: The position.

    Parameters:
        dt: The dtype.

    Returns:
        A value small enough to be exact in float16 and to fit in int8.
    """
    comptime if dt == DType.bool:
        # A `comptime if` does not narrow `dt`, so the bool scalar has to be
        # built at its own type and then rebound. Without the rebind the compiler
        # picks the Intable constructor, which rejects a non-integral dtype.
        return rebind[Scalar[dt]](Scalar[DType.bool](i % 3 == 0))
    else:
        return Scalar[dt]((i * 7) % 100)


def test_every_dtype_round_trips() raises:
    comptime for dt in ALL:
        var typed = Array[dt](64)
        assert_equal(len(typed), 64)
        assert_equal(typed.null_count(), 0)

        for i in range(64):
            typed[i] = sample[dt](i)
        for i in range(0, 64, 5):
            typed.set_null(i)

        var expected_nulls = typed.null_count()
        var erased = AnyArray(typed^)
        assert_equal(len(erased), 64)
        assert_equal(erased.dtype(), dt)
        assert_equal(erased.null_count(), expected_nulls)

        var back = erased^.into_typed[dt]()
        assert_equal(len(back), 64)
        assert_equal(back.null_count(), expected_nulls)
        for i in range(64):
            if i % 5 == 0:
                assert_false(back.is_valid(i))
                assert_equal(back[i], Scalar[dt](0))
            else:
                assert_true(back.is_valid(i))
                assert_equal(back[i], sample[dt](i))


def test_from_list_and_to_list() raises:
    var values: List[Int64] = [Int64(3), Int64(-1), Int64(0), Int64(99)]
    var array = from_list[DType.int64](values)
    assert_equal(len(array), 4)
    var out = array.to_list()
    for i in range(4):
        assert_equal(out[i], values[i])


def test_new_array_is_zero_and_valid() raises:
    var array = Array[DType.float64](10)
    for i in range(10):
        assert_equal(array[i], Float64(0))
        assert_true(array.is_valid(i))


def test_set_null_zeroes_the_value() raises:
    # Kernels read under nulls without masking in several places. That is only
    # correct if a null position holds zero.
    var array = Array[DType.int32](4)
    array[2] = Int32(1234)
    array.set_null(2)
    assert_false(array.is_valid(2))
    assert_equal(array[2], Int32(0))


def test_set_valid() raises:
    var array = Array[DType.int32](4)
    array.set_null(1)
    array.set_valid(1, Int32(77))
    assert_true(array.is_valid(1))
    assert_equal(array[1], Int32(77))


def test_simd_load_and_store() raises:
    var array = Array[DType.int32](16)
    array.store[4](0, SIMD[DType.int32, 4](1, 2, 3, 4))
    var loaded = array.load[4](0)
    assert_equal(loaded[0], Int32(1))
    assert_equal(loaded[3], Int32(4))


def test_load_may_run_past_the_end() raises:
    # The buffer is padded to a 64-byte multiple, so a tail load of a register
    # that straddles the logical end reads zeros rather than someone else's page.
    var array = Array[DType.int32](5)
    for i in range(5):
        array[i] = Int32(i + 1)
    var loaded = array.load[8](0)
    assert_equal(loaded[4], Int32(5))
    assert_equal(loaded[5], Int32(0))
    assert_equal(loaded[7], Int32(0))


def test_slice_copies_values_and_validity() raises:
    var array = Array[DType.int64](20)
    for i in range(20):
        array[i] = Int64(i)
    array.set_null(7)
    var part = array.slice(5, 15)
    assert_equal(len(part), 10)
    for i in range(10):
        if i == 2:
            assert_false(part.is_valid(i))
        else:
            assert_true(part.is_valid(i))
            assert_equal(part[i], Int64(i + 5))


def test_copy_is_deep() raises:
    var original = Array[DType.int64](8)
    original[0] = Int64(1)
    var duplicate = Array[DType.int64](copy=original)
    duplicate[0] = Int64(2)
    assert_equal(original[0], Int64(1))
    assert_equal(duplicate[0], Int64(2))


def test_any_array_as_typed_copies() raises:
    var typed = Array[DType.int64](4)
    typed[0] = Int64(5)
    var erased = AnyArray(typed^)
    var view = erased.as_typed[DType.int64]()
    view[0] = Int64(6)
    assert_equal(erased.unsafe_ptr[DType.int64]().unsafe_load(), Int64(5))


def test_as_typed_with_the_wrong_dtype_raises() raises:
    var erased = AnyArray(Array[DType.int64](4))
    with assert_raises(contains="dtype mismatch"):
        _ = erased.as_typed[DType.float64]()


def test_any_array_carries_its_logical_type() raises:
    var erased = AnyArray(Array[DType.float32](2))
    assert_equal(erased.type, LogicalType.FLOAT32)
    assert_equal(erased.dtype(), DType.float32)


def _grades() raises -> AnyArray:
    """Builds a four row categorical over three categories, one of them unused.

    Returns:
        The column.
    """
    var codes = Array[DType.int8](4)
    codes.set_valid(0, Int8(1))
    codes.set_valid(1, Int8(0))
    codes.set_null(2)
    codes.set_valid(3, Int8(1))
    var levels = StringBuilder()
    levels.append(String("low").as_bytes())
    levels.append(String("high").as_bytes())
    levels.append(String("unused").as_bytes())
    return AnyArray.dictionary[DType.int8](codes^, levels^.finish(), True)


def test_a_dictionary_column_holds_its_categories_apart_from_its_values() raises:
    var column = _grades()
    assert_true(column.is_dictionary())
    assert_equal(len(column), 4)
    # Four rows and three categories, and the two numbers being different is
    # the whole point of the type.
    assert_equal(len(column.categories()), 3)
    assert_equal(column.categories()[2], "unused")
    assert_equal(column.codes[DType.int8]()[0], Int8(1))
    assert_false(column.is_valid(2))
    assert_equal(String(column.type), "category")
    assert_true(column.type.ordered)


def test_a_dictionary_column_is_not_a_string_column() raises:
    # Both hold text and only one of them holds a value per row, so a
    # categorical answering yes here would let `strings()` hand three
    # categories back to a caller that asked for four rows.
    var column = _grades()
    assert_false(column.is_string())
    with assert_raises(contains="not a string column"):
        _ = column.strings()


def test_the_codes_of_a_dictionary_column_are_not_its_values() raises:
    # The refusal that makes the type safe. The codes are a valid int8 buffer,
    # so without this a mean over a categorical would return the average of the
    # category positions and nothing about the answer would look wrong.
    var column = _grades()
    with assert_raises(contains="positions into its categories"):
        _ = column.as_typed[DType.int8]()
    with assert_raises(contains="positions into its categories"):
        _ = column.as_typed_view[DType.int8]()
    with assert_raises(contains="dtype mismatch"):
        _ = column.codes[DType.int32]()


def test_a_dictionary_column_counts_both_halves_of_itself() raises:
    # pandas counts the codes and the categories for a categorical, and so does
    # this, because a million rows over four categories is four megabytes of
    # codes next to a few dozen bytes of text and reporting either half alone
    # makes the saving look imaginary or free.
    var column = _grades()
    ref categories = column.categories()
    var expected = (
        column.data.validity.byte_length()
        + len(column.data.values)
        + len(categories.views)
        + len(categories.payload)
    )
    assert_equal(column.nbytes(), expected)
    assert_true(column.nbytes() > 4)


def test_copying_a_dictionary_column_copies_its_categories() raises:
    var column = _grades()
    var duplicate = AnyArray(copy=column)
    assert_equal(len(duplicate.categories()), 3)
    assert_equal(duplicate.categories()[0], "low")
    assert_true(duplicate.type.ordered)


def test_chunked_lengths_add_up() raises:
    var column = ChunkedArray(AnyArray(Array[DType.int64](10)))
    column.append(AnyArray(Array[DType.int64](5)))
    column.append(AnyArray(Array[DType.int64](7)))
    assert_equal(len(column), 22)
    assert_equal(column.num_chunks(), 3)
    assert_equal(column.dtype(), DType.int64)


def test_chunked_rejects_a_mismatched_chunk() raises:
    var column = ChunkedArray(AnyArray(Array[DType.int64](4)))
    with assert_raises(contains="does not match"):
        column.append(AnyArray(Array[DType.float64](4)))


def test_chunked_locate() raises:
    var column = ChunkedArray(AnyArray(Array[DType.int64](10)))
    column.append(AnyArray(Array[DType.int64](5)))

    var first = column.locate(3)
    assert_equal(first[0], 0)
    assert_equal(first[1], 3)

    var second = column.locate(12)
    assert_equal(second[0], 1)
    assert_equal(second[1], 2)

    with assert_raises(contains="out of range"):
        _ = column.locate(15)


def test_chunked_null_count_sums_chunks() raises:
    var first = Array[DType.int64](4)
    first.set_null(0)
    var second = Array[DType.int64](4)
    second.set_null(1)
    second.set_null(2)

    var column = ChunkedArray(AnyArray(first^))
    column.append(AnyArray(second^))
    assert_equal(column.null_count(), 3)


def test_empty_chunked_column() raises:
    var column = ChunkedArray(logical_for(DType.int64))
    assert_equal(len(column), 0)
    assert_equal(column.num_chunks(), 0)
    assert_equal(column.null_count(), 0)


def test_chunked_locate_walks_a_column_of_many_chunks() raises:
    # `locate` binary searches a prefix sum rather than adding lengths up again,
    # so the shape worth testing is one where the answer differs per chunk and
    # the boundaries land where an off by one would be visible. The chunks here
    # have different lengths on purpose, since equal ones would let a division
    # pass the test without the search being right.
    var column = ChunkedArray(AnyArray(Array[DType.int64](3)))
    var sizes = List[Int]()
    sizes.append(1)
    sizes.append(4)
    sizes.append(1)
    sizes.append(9)
    sizes.append(2)
    for i in range(len(sizes)):
        column.append(AnyArray(Array[DType.int64](sizes[i])))
    assert_equal(len(column), 20)
    assert_equal(column.num_chunks(), 6)

    # Every row, checked against a walk that adds the lengths up the slow way.
    var chunk = 0
    var seen = 0
    for row in range(20):
        while row - seen >= len(column.chunks[chunk]):
            seen += len(column.chunks[chunk])
            chunk += 1
        var found = column.locate(row)
        assert_equal(found[0], chunk, String("wrong chunk for row ", row))
        assert_equal(found[1], row - seen, String("wrong offset for row ", row))

    with assert_raises(contains="out of range"):
        _ = column.locate(20)
    with assert_raises(contains="out of range"):
        _ = column.locate(-1)


def test_chunked_drops_an_empty_chunk() raises:
    # An empty chunk would put two equal entries in the prefix sum, and then a
    # row position would name two chunks and the search could return the one
    # holding nothing. It is dropped on the way in instead.
    var column = ChunkedArray(AnyArray(Array[DType.int64](2)))
    column.append(AnyArray(Array[DType.int64](0)))
    column.append(AnyArray(Array[DType.int64](3)))
    assert_equal(column.num_chunks(), 2)
    assert_equal(len(column), 5)

    var found = column.locate(2)
    assert_equal(found[0], 1)
    assert_equal(found[1], 0)


def test_chunked_only_borrows_the_single_chunk() raises:
    # `only` is what lets an operator that has not been taught about chunks read
    # a chunked column, and it is only worth having if it copies nothing. That is
    # one pointer comparison, the same claim `as_typed_view` asserts.
    var typed = Array[DType.int64](4)
    for i in range(4):
        typed[i] = Int64(i * 3)
    var inner = AnyArray(typed^)
    var address = Int(inner.data.values.unsafe_ptr())

    var column = ChunkedArray(inner^)
    ref chunk = column.only()
    assert_equal(len(chunk), 4)
    assert_equal(chunk.as_typed_view[DType.int64]()[3], Int64(9))
    assert_equal(Int(chunk.data.values.unsafe_ptr()), address)

    column.append(AnyArray(Array[DType.int64](2)))
    with assert_raises(contains="not one"):
        _ = len(column.only())


def test_chunked_combine_of_one_chunk_moves_rather_than_copies() raises:
    var typed = Array[DType.int64](6)
    for i in range(6):
        typed[i] = Int64(i)
    typed.set_null(4)
    var inner = AnyArray(typed^)
    var address = Int(inner.data.values.unsafe_ptr())

    var column = ChunkedArray(inner^)
    var flat = column^.combine()
    assert_equal(len(flat), 6)
    assert_equal(flat.null_count(), 1)
    assert_equal(
        Int(flat.data.values.unsafe_ptr()),
        address,
        "combining one chunk should hand back its own buffer",
    )


def test_chunked_combine_stacks_several_chunks() raises:
    var first = Array[DType.int64](3)
    for i in range(3):
        first[i] = Int64(i)
    var second = Array[DType.int64](2)
    second[0] = Int64(10)
    second[1] = Int64(11)
    second.set_null(1)

    var column = ChunkedArray(AnyArray(first^))
    column.append(AnyArray(second^))
    assert_equal(column.null_count(), 1)

    var flat = column^.combine()
    assert_equal(len(flat), 5)
    ref view = flat.as_typed_view[DType.int64]()
    assert_equal(view[0], Int64(0))
    assert_equal(view[2], Int64(2))
    assert_equal(view[3], Int64(10))
    assert_false(view.is_valid(4))
    assert_equal(flat.null_count(), 1)


def test_chunked_combine_of_nothing_is_an_empty_column() raises:
    var column = ChunkedArray(logical_for(DType.int64))
    var flat = column^.combine()
    assert_equal(len(flat), 0)
    assert_equal(flat.dtype(), DType.int64)


def test_a_typed_array_has_the_layout_of_the_storage_it_holds() raises:
    """The assumption `AnyArray.as_typed_view` reinterprets a pointer under.

    A struct of one field has that field's layout, so a pointer to a
    `ColumnData` is a pointer to an `Array` over it. That is true today and
    nothing in the language promises it stays true if `Array` gains a second
    field, so it is asserted here rather than left to be discovered by a group
    by reading the wrong bytes.
    """
    assert_equal(size_of[Array[DType.int64]](), size_of[ColumnData]())
    assert_equal(size_of[Array[DType.uint8]](), size_of[ColumnData]())
    assert_equal(size_of[Array[DType.float64]](), size_of[ColumnData]())


def test_a_typed_view_reads_the_column_without_copying_it() raises:
    var typed = Array[DType.int64](4)
    for i in range(4):
        typed[i] = Int64(i * 7)
    typed.set_null(2)

    var column = AnyArray(typed^)
    ref view = column.as_typed_view[DType.int64]()

    assert_equal(len(view), 4)
    assert_equal(view[0], Int64(0))
    assert_equal(view[3], Int64(21))
    assert_false(view.is_valid(2))
    assert_equal(view.null_count(), 1)
    # The point of the view is that it is the column's own storage and not a
    # copy of it, which is one pointer comparison to establish.
    assert_true(view.unsafe_ptr() == column.unsafe_ptr[DType.int64]())


def test_a_typed_view_refuses_the_wrong_dtype() raises:
    var typed = Array[DType.int64](2)
    var column = AnyArray(typed^)
    with assert_raises(contains="dtype mismatch"):
        ref view = column.as_typed_view[DType.float64]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
