"""The one place firepanda asks the allocator for memory.

Every array, bitmap and offsets vector in the engine is a `Buffer` underneath.
Buffers are always 64-byte aligned and always allocated in whole 64-byte
multiples. That is not a micro-optimization, it is what makes the kernels simple:
a kernel can read a full SIMD register past the logical end of a column without
touching a page it does not own, so the tail of a loop needs masking but not a
separate scalar path.

See docs/specs/02-architecture.md.
"""

from std.memory import ArcPointer, unsafe_memcpy, unsafe_memset_zero
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
    """A 64-byte aligned, zero-initialized run of bytes, shared until written.

    The allocation is behind a refcount and a copy of a buffer shares it. The
    copy becomes real the first time somebody asks for a pointer they could
    write through, which is what `unsafe_mut_ptr` and `mut_bitcast` are for and
    what `unsafe_ptr` and `bitcast` deliberately are not. So the split in those
    four names is not a style, it is the whole mechanism: a reader cannot
    accidentally pay for a copy and a writer cannot accidentally skip one,
    because the compiler refuses to write through what a reader is handed.

    This is what makes a projection cheap. `select` clones the columns it keeps,
    which used to memcpy them, and on TPC-H that copy was most of what separated
    us from polars: 15.8 ms on q5, 16.6 on q8 and 30.3 on q9, measured in place
    by inserting a second redundant projection and taking the difference. None
    of it computed anything. See #406.

    The refcount is atomic and the un-sharing is not, and that is the one rule
    a caller has to keep. A shared buffer must be made private before several
    workers write it, because otherwise each of them reaches `_unshare` at the
    same time, each sees a count above one and each allocates: one wins the
    assignment, the rest leak, and the buffer they all copied from has its count
    taken down once per worker for the single copy that exists. Nothing looks
    wrong at the time. It surfaces much later as a free of memory somebody is
    still reading, which is how this was found, by the sanitizer rather than by
    a wrong answer.

    `make_private` is how a caller says so, and it is only needed where the
    thing being written in parallel was made by copying rather than by
    allocating. Most kernels allocate their output and are already private, so
    the rule bites in one place: a kernel that starts from a copy of its input
    and edits it, such as `_drop_nans` clearing the bits of the rows holding a
    NaN.
    """

    var _mem: ArcPointer[ManagedAllocation[UInt8]]
    var _capacity: Int
    var _size: Int
    var _offset: Int
    """Bytes from the allocation's first byte to this buffer's first byte.

    Zero for a buffer that was allocated, and a multiple of 64 for one that is a
    window onto part of another. `_capacity` is measured from here rather than
    from the allocation, so every pointer this hands out and every length it
    reports are about the window and a caller cannot tell the difference.

    A window's padding is the next window's rows rather than zeroes, which is
    the one thing that is not the same. `Buffer(window_of=)` says what a caller
    has to promise so that nothing reads it.
    """

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
        self._mem = ArcPointer(allocation^.into_managed())
        self._capacity = capacity
        self._size = size
        self._offset = 0

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
        self._mem = ArcPointer(allocation^.into_managed())
        self._capacity = capacity
        self._size = overwritten
        self._offset = 0

    def __init__(out self, *, copy: Self):
        """Shares a buffer's bytes rather than copying them.

        This used to memcpy, on the argument that copies are rare because
        kernels move buffers rather than copying them. The kernels do, and the
        frame layer does not: `select` clones every column it keeps, and
        projection is the second thing a query does. On TPC-H that memcpy was
        most of what separated us from polars, so the cost of one atomic per
        column clone is bought back many times over by not moving the column.

        The bytes become private again the first time somebody asks for a
        pointer they could write through. Nothing is copied for a reader.

        Args:
            copy: The buffer to share with.
        """
        self._mem = copy._mem
        self._capacity = copy._capacity
        self._size = copy._size
        self._offset = copy._offset

    def __init__(out self, *, window_of: Self, at: Int, size: Int):
        """Shares part of a buffer, copying nothing at all.

        What `Buffer(copy=)` is to a whole column this is to a range of one.
        Both share the allocation and both go private on the first write, and
        the difference is that this one starts partway in and stops early, so a
        column can be cut into pieces for nothing rather than for a memcpy per
        piece.

        The caller owes one promise and it is about the padding. An allocated
        buffer is rounded up to 64 bytes and the bytes between the logical size
        and that boundary are zero, which is what lets a kernel read a whole
        register past the end of a column and mask the answer. A window that
        stops before its parent does has the next window's rows there instead of
        zeroes, so the promise is that the window never has a tail to mask:
        `at` and `size` are both multiples of 64, or the window runs to the end
        of the parent and inherits the parent's own padding.

        A row count that is a multiple of 64 satisfies that for every fixed
        width dtype we have, so in practice the promise is that a caller cuts on
        whole morsels. `Scan` is the caller that does.

        The window's capacity is its size rather than its parent's, so nothing
        downstream can read past it by asking how much room there is. Writing
        through it takes a private copy of the window and not of the parent, so
        a worker that edits one piece of a column does not pay for the rest.

        Args:
            window_of: The buffer to share part of.
            at: The first byte, counted from that buffer's own first byte. A
                multiple of 64.
            size: The number of bytes. A multiple of 64, unless `at + size` is
                that buffer's whole size.
        """
        debug_assert(
            at % ALIGNMENT == 0,
            "buffer window starts at ",
            at,
            " which is not a multiple of ",
            ALIGNMENT,
        )
        debug_assert(
            size % ALIGNMENT == 0 or at + size == window_of._size,
            "buffer window of ",
            size,
            " bytes at ",
            at,
            " neither lands on ",
            ALIGNMENT,
            " nor reaches the end of the ",
            window_of._size,
            " it is cut from",
        )
        debug_assert(
            at + size <= window_of._size,
            "buffer window of ",
            size,
            " bytes at ",
            at,
            " runs past the ",
            window_of._size,
            " it is cut from",
        )
        self._mem = window_of._mem
        self._size = size
        self._offset = window_of._offset + at
        var room = window_of._capacity - at
        var wanted = round_up(size, ALIGNMENT)
        self._capacity = wanted if wanted < room else room

    def _unshare(mut self):
        """Gives this buffer an allocation nobody else is holding.

        Called on the way to handing out a mutable pointer. A buffer that is
        already alone keeps its allocation and pays a load and a branch, which
        is what every write costs after this change and is not measurable next
        to the write itself.

        The whole capacity is copied and not just the logical size, because the
        pad up to the 64-byte boundary is guaranteed to be zero and a kernel is
        allowed to read a full register past the end. Copying only the size
        would leave that pad holding whatever the allocator last put there.

        A window copies its own bytes rather than its parent's, which is the
        point of having one: a worker that writes one morsel of a column pays
        for that morsel. What it gets back is an ordinary buffer starting at
        zero, so the padding it inherits is zero the way an allocated buffer's
        is, and it stops being a window at the moment it stops sharing.
        """
        if self._mem.count() == 1 and self._offset == 0:
            return
        var allocation = alloc(
            Layout[UInt8](count=self._capacity, alignment=ALIGNMENT)
        )
        unsafe_memcpy(
            dest=allocation.unsafe_ptr(),
            src=self._mem[].unsafe_ptr().unsafe_offset(self._offset),
            count=self._size,
        )
        var pad = self._capacity - self._size
        if pad > 0:
            unsafe_memset_zero(
                allocation.unsafe_ptr().unsafe_offset(self._size), pad
            )
        self._mem = ArcPointer(allocation^.into_managed())
        self._offset = 0

    def make_private(mut self):
        """Takes this buffer's own copy of the bytes now rather than on a write.

        For the one shape the lazy copy cannot serve: a buffer that was made by
        copying another and is about to be written by several workers at once.
        Each of them would reach `_unshare` at the same time, all of them would
        see a count above one, and all of them would allocate. One would win the
        assignment and the rest would leak, and the count on the buffer they
        copied from would go down once per worker for the one copy that was
        made. It is a torn refcount rather than torn data, so it does not show
        up as a wrong answer, it shows up later as a free of something still in
        use.

        Calling this on the thread that made the copy, before the workers start,
        makes the buffer private while there is still only one of them, and
        every worker then finds a count of one and takes the early return. A
        buffer that was allocated rather than copied is already private and this
        costs it a load and a branch.
        """
        self._unshare()

    def is_shared(self) -> Bool:
        """Reports whether another buffer is holding the same allocation.

        For tests, which is the only way to observe from the outside that a
        clone did not copy. Nothing in the engine branches on it.

        Returns:
            True if the refcount is above one.
        """
        return self._mem.count() > 1

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

    def unsafe_ptr(self) -> Pointer[UInt8, origin_of(self)]:
        """Returns a pointer to the first byte, for reading.

        Borrowed rather than `ref`, so what comes back cannot be written
        through. That is the point: a shared buffer stays shared for readers,
        and a caller who means to write says so by name and calls
        `unsafe_mut_ptr`, which is where the copy happens. The compiler is what
        enforces the distinction, so it holds for call sites written later
        without anyone having to remember this paragraph.

        Returns:
            A pointer valid for `capacity()` bytes.
        """
        return (
            self._mem[]
            .unsafe_ptr()
            .unsafe_offset(self._offset)
            .as_imm()
            .unsafe_origin_cast[origin_of(self)]()
        )

    def unsafe_mut_ptr(mut self) -> Pointer[UInt8, origin_of(self)]:
        """Returns a pointer to the first byte, for writing.

        Takes a private copy of the allocation first if anyone else is holding
        it, so the bytes this writes into belong to this buffer alone.

        Returns:
            A pointer valid for `capacity()` bytes.
        """
        self._unshare()
        return (
            self._mem[]
            .unsafe_ptr()
            .unsafe_offset(self._offset)
            .unsafe_origin_cast[origin_of(self)]()
        )

    def bitcast[dt: DType](self) -> Pointer[Scalar[dt], origin_of(self)]:
        """Reinterprets the bytes as elements of a dtype, for reading.

        Parameters:
            dt: The dtype to view the bytes as.

        Returns:
            A typed pointer to the first element.
        """
        return (
            self._mem[]
            .unsafe_ptr()
            .unsafe_offset(self._offset)
            .as_imm()
            .unsafe_origin_cast[origin_of(self)]()
            .unsafe_bitcast[Scalar[dt]]()
        )

    def mut_bitcast[
        dt: DType
    ](mut self) -> Pointer[Scalar[dt], origin_of(self)]:
        """Reinterprets the bytes as elements of a dtype, for writing.

        Takes a private copy of the allocation first if anyone else is holding
        it.

        Parameters:
            dt: The dtype to view the bytes as.

        Returns:
            A typed pointer to the first element.
        """
        self._unshare()
        return (
            self._mem[]
            .unsafe_ptr()
            .unsafe_offset(self._offset)
            .unsafe_origin_cast[origin_of(self)]()
            .unsafe_bitcast[Scalar[dt]]()
        )

    def is_aligned(self) -> Bool:
        """Reports whether the allocation meets the alignment contract.

        Returns:
            True if the base address is a multiple of 64.
        """
        return (
            Int(self._mem[].unsafe_ptr().unsafe_offset(self._offset))
            % ALIGNMENT
            == 0
        )

    def zero(mut self):
        """Sets every allocated byte, including the padding, to zero."""
        self._unshare()
        unsafe_memset_zero(
            self._mem[].unsafe_ptr().unsafe_offset(self._offset), self._capacity
        )
