"""Tests for the allocator wrapper.

The two properties every kernel in the engine is allowed to assume are checked
here: the base address is 64-byte aligned, and the allocation is a whole number
of 64-byte blocks so a vectorized loop can run one register past the end of the
data. If either of these stops holding, kernels start reading pages they do not
own, and they will do it rarely enough to look like a miscompile.

The third property is newer and is the reason the accessors come in pairs: a
copy of a buffer shares the bytes and only takes its own when somebody asks for
a pointer they could write through. The tests below check the address and not
just the contents, because a reader that quietly un-shared would still give the
right answer and would give back the memcpy this was written to remove.
"""

from std.testing import TestSuite, assert_equal, assert_true

from firepanda.buffer.buffer import ALIGNMENT, Buffer, round_up
from firepanda.buffer.pool import BufferPool, class_capacity, size_class


def test_round_up() raises:
    assert_equal(round_up(0, 64), 0)
    assert_equal(round_up(1, 64), 64)
    assert_equal(round_up(63, 64), 64)
    assert_equal(round_up(64, 64), 64)
    assert_equal(round_up(65, 64), 128)
    assert_equal(round_up(4096, 64), 4096)


def test_capacity_is_a_whole_number_of_blocks() raises:
    for size in [0, 1, 7, 63, 64, 65, 1000, 65536]:
        var buffer = Buffer(size)
        assert_equal(len(buffer), size)
        assert_equal(buffer.capacity() % ALIGNMENT, 0)
        assert_true(buffer.capacity() >= size)
        assert_true(buffer.capacity() >= ALIGNMENT)


def test_alignment_holds_for_every_size() raises:
    # Allocators tend to align large requests for free and small ones not at all,
    # so the small sizes are the interesting ones here.
    for size in [1, 3, 8, 17, 64, 100, 4096, 1 << 20]:
        var buffer = Buffer(size)
        assert_true(buffer.is_aligned())


def test_new_buffers_are_zeroed() raises:
    var buffer = Buffer(1000)
    var ptr = buffer.unsafe_ptr()
    for i in range(buffer.capacity()):
        assert_equal(ptr.unsafe_offset(i).unsafe_load(), UInt8(0))


def test_padding_is_zeroed_too() raises:
    # A masked tail load reads the padding. If it held garbage, a sum over a
    # column whose length is not a multiple of the register width would be wrong
    # by an amount that depends on what the allocator handed back.
    var buffer = Buffer(65)
    var ptr = buffer.unsafe_ptr()
    for i in range(65, buffer.capacity()):
        assert_equal(ptr.unsafe_offset(i).unsafe_load(), UInt8(0))


def test_a_copy_shares_the_bytes() raises:
    # This is what makes a projection cheap. `select` clones every column it
    # keeps, and before this the clone was a memcpy of the whole column.
    var original = Buffer(128)
    original.unsafe_mut_ptr().unsafe_offset(7).unsafe_write(UInt8(42))
    var duplicate = Buffer(copy=original)

    assert_true(original.is_shared(), "the original knows it is shared")
    assert_true(duplicate.is_shared(), "and so does the copy")
    assert_equal(
        Int(duplicate.unsafe_ptr()),
        Int(original.unsafe_ptr()),
        "same allocation, not a copy of one",
    )


def test_reading_a_shared_buffer_does_not_copy_it() raises:
    # A reader that un-shared would give back the memcpy this change removes,
    # and it would do it quietly, so the address is checked and not just the
    # bytes.
    var original = Buffer(128)
    var duplicate = Buffer(copy=original)
    var address = Int(original.unsafe_ptr())
    # Mojo destroys a value at its last use, so whichever of two sharing buffers
    # is asked about last is by then the only holder and reports itself
    # unshared. That is correct and it is not what this test is about, so a
    # third holder is kept alive past every assertion below to stop the count
    # from reaching one while anyone is looking. Without it the test fails on a
    # library that is behaving.
    var witness = Buffer(copy=original)

    for i in range(len(duplicate)):
        _ = duplicate.unsafe_ptr().unsafe_offset(i).unsafe_load()
    _ = duplicate.bitcast[DType.int32]().unsafe_load()

    assert_true(duplicate.is_shared(), "still shared after reading it")
    assert_equal(Int(duplicate.unsafe_ptr()), address, "still the same bytes")
    assert_equal(Int(witness.unsafe_ptr()), address, "for every holder of them")


def test_writing_a_shared_buffer_takes_a_private_copy() raises:
    var original = Buffer(128)
    original.unsafe_mut_ptr().unsafe_offset(7).unsafe_write(UInt8(42))
    var duplicate = Buffer(copy=original)

    duplicate.unsafe_mut_ptr().unsafe_offset(7).unsafe_write(UInt8(9))

    assert_true(not original.is_shared(), "the writer left")
    assert_true(not duplicate.is_shared(), "and took its own allocation")
    assert_equal(
        original.unsafe_ptr().unsafe_offset(7).unsafe_load(),
        UInt8(42),
        "the other side did not see the write",
    )
    assert_equal(
        duplicate.unsafe_ptr().unsafe_offset(7).unsafe_load(), UInt8(9)
    )


def test_make_private_unshares_without_a_write() raises:
    # The lazy copy cannot serve a buffer that several workers are about to
    # write, because they would all reach the un-share at once, all allocate,
    # and all take the source's refcount down for the one copy that exists.
    # A kernel that starts from a copy of its input says so up front instead.
    var original = Buffer(128)
    original.unsafe_mut_ptr().unsafe_offset(7).unsafe_write(UInt8(42))
    var duplicate = Buffer(copy=original)
    assert_true(duplicate.is_shared(), "shared to begin with")

    duplicate.make_private()

    assert_true(not duplicate.is_shared(), "took its own allocation")
    assert_true(not original.is_shared(), "and left the original alone")
    assert_equal(
        duplicate.unsafe_ptr().unsafe_offset(7).unsafe_load(),
        UInt8(42),
        "with the bytes it was copied from",
    )


def test_make_private_on_a_buffer_that_is_already_alone() raises:
    var buffer = Buffer(128)
    var address = Int(buffer.unsafe_ptr())
    buffer.make_private()
    assert_equal(
        Int(buffer.unsafe_ptr()), address, "nothing to un-share, nothing copied"
    )


def test_unsharing_carries_the_padding_over() raises:
    # The pad past the logical size is what a vectorized kernel reads when it
    # runs one register off the end, so a copy that stopped at the size would
    # leave whatever the allocator last put there in the way of a masked load.
    var original = Buffer(65)
    var duplicate = Buffer(copy=original)
    duplicate.unsafe_mut_ptr().unsafe_write(UInt8(1))

    for i in range(65, duplicate.capacity()):
        assert_equal(
            duplicate.unsafe_ptr().unsafe_offset(i).unsafe_load(),
            UInt8(0),
            "pad byte " + String(i),
        )


def test_a_buffer_that_is_alone_keeps_its_allocation() raises:
    var buffer = Buffer(128)
    var address = Int(buffer.unsafe_ptr())
    buffer.unsafe_mut_ptr().unsafe_write(UInt8(3))
    assert_equal(
        Int(buffer.unsafe_ptr()), address, "nothing to un-share, nothing copied"
    )


def test_bitcast_reads_the_same_bytes() raises:
    var buffer = Buffer(64)
    var typed = buffer.mut_bitcast[DType.int32]()
    typed.unsafe_offset(3).unsafe_write(Int32(-7))
    assert_equal(
        buffer.bitcast[DType.int32]().unsafe_offset(3).unsafe_load(), Int32(-7)
    )


def test_zero_clears_everything() raises:
    var buffer = Buffer(100)
    var ptr = buffer.unsafe_mut_ptr()
    for i in range(buffer.capacity()):
        ptr.unsafe_offset(i).unsafe_write(UInt8(255))
    buffer.zero()
    for i in range(buffer.capacity()):
        assert_equal(
            buffer.unsafe_ptr().unsafe_offset(i).unsafe_load(), UInt8(0)
        )


def test_overwritten_still_zeroes_the_pad() raises:
    # The caller's own bytes are left alone, but the pad between the size asked
    # for and the 64-byte capacity is what a vectorized kernel reads when it
    # runs one register past the end, so it has to be zero.
    for size in [1, 63, 64, 65, 100, 4095]:
        var buffer = Buffer(overwritten=size)
        assert_equal(len(buffer), size, "size is what was asked for")
        assert_true(
            buffer.capacity() >= size, "capacity covers the requested size"
        )
        for i in range(size, buffer.capacity()):
            assert_equal(
                buffer.unsafe_ptr().unsafe_offset(i).unsafe_load(),
                UInt8(0),
                "pad byte " + String(i) + " of " + String(size),
            )


def test_overwritten_of_nothing_is_still_a_buffer() raises:
    var buffer = Buffer(overwritten=0)
    assert_equal(len(buffer), 0, "empty")
    for i in range(buffer.capacity()):
        assert_equal(
            buffer.unsafe_ptr().unsafe_offset(i).unsafe_load(), UInt8(0)
        )


def test_size_classes_are_monotonic_and_sufficient() raises:
    var previous = -1
    for size in [1, 64, 65, 128, 129, 4096, 100000]:
        var index = size_class(size)
        assert_true(index >= previous)
        assert_true(class_capacity(index) >= size)
        previous = index


def test_size_class_capacity_round_trips() raises:
    for index in range(0, 20):
        assert_equal(size_class(class_capacity(index)), index)


def test_pool_recycles_and_counts() raises:
    var pool = BufferPool()
    var first = pool.take(1000)
    assert_equal(pool.misses(), 1)
    assert_equal(pool.hits(), 0)

    var address = Int(first.unsafe_ptr())
    pool.give(first^)
    assert_true(pool.pooled_bytes() > 0)

    var second = pool.take(1000)
    assert_equal(pool.hits(), 1)
    assert_equal(Int(second.unsafe_ptr()), address)
    pool.give(second^)


def test_pool_hands_back_zeroed_memory() raises:
    # A recycled buffer that still held the previous column's bytes would show up
    # as a wrong answer under nulls, because a null position is supposed to read
    # as zero.
    var pool = BufferPool()
    var first = pool.take(256)
    for i in range(256):
        first.unsafe_mut_ptr().unsafe_offset(i).unsafe_write(UInt8(0xAB))
    pool.give(first^)

    var second = pool.take(256)
    for i in range(256):
        assert_equal(
            second.unsafe_ptr().unsafe_offset(i).unsafe_load(), UInt8(0)
        )
    pool.give(second^)


def test_pool_respects_its_limit() raises:
    var pool = BufferPool(limit_per_class=2)
    for _ in range(5):
        pool.give(Buffer(64))
    assert_equal(pool.pooled_bytes(), 128)
    pool.clear()
    assert_equal(pool.pooled_bytes(), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
