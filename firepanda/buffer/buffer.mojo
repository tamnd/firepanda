"""The one place firepanda asks the allocator for memory.

Every array, bitmap and offsets vector in the engine is a `Buffer` underneath.
Buffers are always 64-byte aligned and always allocated in whole 64-byte
multiples. That is not a micro-optimization, it is what makes the kernels simple:
a kernel can read a full SIMD register past the logical end of a column without
touching a page it does not own, so the tail of a loop needs masking but not a
separate scalar path.

See docs/specs/02-architecture.md.
"""

from std.memory import unsafe_memcpy, unsafe_memset_zero
from std.memory.alloc import Layout, ManagedAllocation, alloc

comptime ALIGNMENT = 64
"""Bytes. One cache line on x86-64 and Apple silicon, and the AVX-512 register width."""


def round_up(n: Int, multiple: Int) -> Int:
    """Rounds a byte count up to a multiple.

    Args:
        n: The count to round.
        multiple: The multiple to round to. Must be positive.

    Returns:
        The smallest multiple of `multiple` that is at least `n`.
    """
    return ((n + multiple - 1) // multiple) * multiple


struct Buffer(Copyable, Movable, Sized):
    """An owned, 64-byte aligned, zero-initialized run of bytes."""

    var _mem: ManagedAllocation[UInt8]
    var _capacity: Int
    var _size: Int

    def __init__(out self, size: Int):
        """Allocates a zeroed buffer.

        The allocation is rounded up to a whole number of 64-byte blocks so that
        vectorized kernels can overrun the logical end by up to one register.

        Args:
            size: The number of bytes the caller intends to use.
        """
        var capacity = round_up(size, ALIGNMENT)
        if capacity == 0:
            capacity = ALIGNMENT
        var allocation = alloc(
            Layout[UInt8](count=capacity, alignment=ALIGNMENT)
        )
        unsafe_memset_zero(allocation.unsafe_ptr(), capacity)
        self._mem = allocation^.into_managed()
        self._capacity = capacity
        self._size = size

    def __init__(out self, *, overwritten: Int):
        """Allocates a buffer whose bytes the caller promises to write.

        The zeroing an ordinary `Buffer` does is a full pass over the
        allocation, and a caller that is about to memcpy over every byte of it
        pays for that pass twice. A concat is the case that matters: it writes
        every view and every payload byte of its output and nothing else, and on
        a ten million row string column the memset alone was a third of it.

        The pad between the requested size and the 64-byte capacity is still
        zeroed, so a vectorized kernel reading one register past the logical end
        sees zeroes rather than whatever the allocator left there, which is the
        invariant the rest of the engine is written against. Only the caller's
        own bytes are left alone, and reading one before writing it is a bug in
        the caller.

        Args:
            overwritten: The number of bytes the caller will write, all of them.
        """
        var capacity = round_up(overwritten, ALIGNMENT)
        if capacity == 0:
            capacity = ALIGNMENT
        var allocation = alloc(
            Layout[UInt8](count=capacity, alignment=ALIGNMENT)
        )
        var pad = capacity - overwritten
        if pad > 0:
            unsafe_memset_zero(
                allocation.unsafe_ptr().unsafe_offset(overwritten), pad
            )
        self._mem = allocation^.into_managed()
        self._capacity = capacity
        self._size = overwritten

    def __init__(out self, *, copy: Self):
        """Copies a buffer's bytes into a fresh allocation.

        Buffers are deep-copied because the alternative, refcounting, would put an
        atomic on the hot path of every column clone. Copies are rare; kernels
        move buffers rather than copying them.

        Args:
            copy: The buffer to copy.
        """
        self = Self(copy._size)
        unsafe_memcpy(
            dest=self._mem.unsafe_ptr(),
            src=copy._mem.unsafe_ptr(),
            count=copy._size,
        )

    def __len__(self) -> Int:
        """Returns the number of bytes the caller asked for.

        Returns:
            The logical size, not the allocated capacity.
        """
        return self._size

    def set_size(mut self, size: Int):
        """Changes the logical size without touching the allocation.

        This exists for the pool, which hands out an allocation rounded up to a
        size class and still owes the caller a buffer whose length is the length
        that was asked for. Growing back up to the capacity is allowed and the
        bytes in between are still zero, because nothing outside the logical size
        is ever written.

        Args:
            size: The new logical size. Must not exceed `capacity()`.
        """
        debug_assert(
            size <= self._capacity,
            "buffer size ",
            size,
            " exceeds capacity ",
            self._capacity,
        )
        self._size = size

    def capacity(self) -> Int:
        """Returns the number of bytes actually allocated.

        Returns:
            The logical size rounded up to a 64-byte multiple.
        """
        return self._capacity

    def unsafe_ptr(ref self) -> Pointer[UInt8, origin_of(self)]:
        """Returns a pointer to the first byte.

        Returns:
            A pointer valid for `capacity()` bytes.
        """
        return self._mem.unsafe_ptr().unsafe_origin_cast[origin_of(self)]()

    def bitcast[dt: DType](ref self) -> Pointer[Scalar[dt], origin_of(self)]:
        """Reinterprets the bytes as elements of a dtype.

        Parameters:
            dt: The dtype to view the bytes as.

        Returns:
            A typed pointer to the first element.
        """
        return (
            self._mem.unsafe_ptr()
            .unsafe_origin_cast[origin_of(self)]()
            .unsafe_bitcast[Scalar[dt]]()
        )

    def is_aligned(self) -> Bool:
        """Reports whether the allocation meets the alignment contract.

        Returns:
            True if the base address is a multiple of 64.
        """
        return Int(self._mem.unsafe_ptr()) % ALIGNMENT == 0

    def zero(mut self):
        """Sets every allocated byte, including the padding, to zero."""
        unsafe_memset_zero(self._mem.unsafe_ptr(), self._capacity)
