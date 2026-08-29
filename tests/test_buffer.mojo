"""Tests for the allocator wrapper.

The two properties every kernel in the engine is allowed to assume are checked
here: the base address is 64-byte aligned, and the allocation is a whole number
of 64-byte blocks so a vectorized loop can run one register past the end of the
data. If either of these stops holding, kernels start reading pages they do not
own, and they will do it rarely enough to look like a miscompile.
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


def test_copy_is_deep() raises:
    var original = Buffer(128)
    original.unsafe_ptr().unsafe_offset(7).unsafe_write(UInt8(42))
    var duplicate = Buffer(copy=original)
    assert_equal(
        duplicate.unsafe_ptr().unsafe_offset(7).unsafe_load(), UInt8(42)
    )

    duplicate.unsafe_ptr().unsafe_offset(7).unsafe_write(UInt8(9))
    assert_equal(
        original.unsafe_ptr().unsafe_offset(7).unsafe_load(), UInt8(42)
    )
    assert_equal(
        duplicate.unsafe_ptr().unsafe_offset(7).unsafe_load(), UInt8(9)
    )


def test_bitcast_reads_the_same_bytes() raises:
    var buffer = Buffer(64)
    var typed = buffer.bitcast[DType.int32]()
    typed.unsafe_offset(3).unsafe_write(Int32(-7))
    assert_equal(
        buffer.bitcast[DType.int32]().unsafe_offset(3).unsafe_load(), Int32(-7)
    )


def test_zero_clears_everything() raises:
    var buffer = Buffer(100)
    var ptr = buffer.unsafe_ptr()
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
        first.unsafe_ptr().unsafe_offset(i).unsafe_write(UInt8(0xAB))
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
