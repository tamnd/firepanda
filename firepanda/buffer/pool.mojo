"""A size-class free list for buffers.

Group-by and join build and discard buffers at a rate where malloc shows up in a
profile. The pool keeps retired buffers in per-size-class free lists and hands
them back instead of returning them to the allocator.

The pool is an explicit object rather than a global. A global would be shared
mutable state, which the engine confines to `firepanda/exec/shared.mojo` and
guards in CI, and it would need a lock or a thread-local. When the executor lands
it will own one pool per worker thread and pass it down; until then callers that
want pooling construct one and hold it. Callers that do not care keep using
`Buffer` directly and pay the allocator.

See docs/specs/02-architecture.md.
"""

from .buffer import ALIGNMENT, Buffer, round_up

comptime MIN_CLASS_SHIFT = 6
"""Buffers below 64 bytes all land in the smallest class."""

comptime NUM_CLASSES = 27
"""Size classes from 64 bytes up to 4 GiB. Anything larger bypasses the pool."""

comptime MAX_POOLED = 1 << (MIN_CLASS_SHIFT + NUM_CLASSES - 1)
"""Largest byte count the pool will retain."""


def size_class(nbytes: Int) -> Int:
    """Returns the index of the size class a byte count falls into.

    Classes are powers of two starting at 64. A request is served from the class
    whose capacity is at least the request, so a 100-byte request uses the
    128-byte class and wastes 28 bytes. That waste is the price of not calling
    the allocator.

    Args:
        nbytes: The requested byte count.

    Returns:
        The class index, or -1 if the request is too large to pool.
    """
    if nbytes > MAX_POOLED:
        return -1
    var capacity = 1 << MIN_CLASS_SHIFT
    var index = 0
    while capacity < nbytes:
        capacity <<= 1
        index += 1
    return index


def class_capacity(index: Int) -> Int:
    """Returns the byte capacity of a size class.

    Args:
        index: The class index.

    Returns:
        The number of bytes buffers in that class hold.
    """
    return 1 << (MIN_CLASS_SHIFT + index)


struct BufferPool(Movable):
    """Per-size-class free lists of retired buffers."""

    var _free: List[List[Buffer]]
    var _limit_per_class: Int
    var _hits: Int
    var _misses: Int

    def __init__(out self, limit_per_class: Int = 8):
        """Constructs an empty pool.

        Args:
            limit_per_class: How many retired buffers to keep per size class
                before releasing them back to the allocator. Eight is enough to
                cover a hash table rebuild without holding a working set hostage.
        """
        self._free = List[List[Buffer]]()
        for _ in range(NUM_CLASSES):
            self._free.append(List[Buffer]())
        self._limit_per_class = limit_per_class
        self._hits = 0
        self._misses = 0

    def take(mut self, size: Int) -> Buffer:
        """Returns a zeroed buffer whose length is exactly `size` bytes.

        The allocation behind it is rounded up to a size class, which is what
        makes recycling possible, but the buffer reports the length the caller
        asked for. A caller that got back a longer buffer than it requested would
        build a column with the wrong row count.

        Args:
            size: The number of bytes needed.

        Returns:
            A buffer whose contents are zero, from the free list if one is
            available and from the allocator otherwise.
        """
        var index = size_class(size)
        if index >= 0 and len(self._free[index]) > 0:
            var buffer = self._free[index].pop()
            # Recycled memory carries the previous tenant's bytes. Callers are
            # entitled to assume a buffer is zeroed, and a validity bitmap that
            # is not zeroed reads as a column full of stale nulls.
            buffer.zero()
            buffer.set_size(size)
            self._hits += 1
            return buffer^
        self._misses += 1
        if index < 0:
            return Buffer(size)
        var buffer = Buffer(class_capacity(index))
        buffer.set_size(size)
        return buffer^

    def give(mut self, var buffer: Buffer):
        """Returns a buffer to the pool.

        Buffers that do not fit a size class, or whose class free list is full,
        are dropped here and released by their destructor.

        Args:
            buffer: The buffer to retire.
        """
        var index = size_class(buffer.capacity())
        if index < 0:
            return
        if len(self._free[index]) >= self._limit_per_class:
            return
        self._free[index].append(buffer^)

    def hits(self) -> Int:
        """Returns how many requests were served from a free list.

        Returns:
            The hit count since construction.
        """
        return self._hits

    def misses(self) -> Int:
        """Returns how many requests went to the allocator.

        Returns:
            The miss count since construction.
        """
        return self._misses

    def pooled_bytes(self) -> Int:
        """Returns the total capacity currently held on the free lists.

        Returns:
            Bytes retained by the pool and not in use by any caller.
        """
        var total = 0
        for index in range(NUM_CLASSES):
            total += len(self._free[index]) * class_capacity(index)
        return total

    def clear(mut self):
        """Releases every retired buffer back to the allocator."""
        for index in range(NUM_CLASSES):
            self._free[index].clear()
