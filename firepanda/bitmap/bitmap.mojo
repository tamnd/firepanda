"""The validity bitmap.

One bit per element, least significant bit first within each byte, which is the
Arrow layout. A set bit means the value at that position is present; a clear bit
means it is null. Every array in firepanda carries one.

The bit order matters for interop and it also matters for speed: with LSB-first
packing, `count_ones` over a run of bytes is a popcount loop with no shuffling,
and appending a value is a shift and an or rather than a reversal.

Trailing bits in the final byte, past `length`, are always zero. Every method
here maintains that, and `count_ones` depends on it.

See docs/specs/02-architecture.md.
"""

from std.bit import pop_count
from std.memory import unsafe_memcpy

from firepanda.buffer.buffer import Buffer, round_up


def bytes_for(length: Int) -> Int:
    """Returns the number of bytes needed to hold a bitmap of a given length.

    Args:
        length: The number of bits.

    Returns:
        The byte count, rounded up.
    """
    return (length + 7) // 8


struct Bitmap(Copyable, Movable, Sized):
    """A packed run of validity bits, one per element."""

    var _buffer: Buffer
    var _length: Int

    def __init__(out self, length: Int, all_valid: Bool = True):
        """Constructs a bitmap of a given length.

        Args:
            length: The number of bits.
            all_valid: Whether every bit starts set. A freshly built column with
                no nulls is the common case, so this defaults to True.
        """
        self._buffer = Buffer(bytes_for(length))
        self._length = length
        if all_valid:
            self.set_all()

    def __init__(out self, *, copy: Self):
        """Copies a bitmap.

        Args:
            copy: The bitmap to copy.
        """
        self._buffer = Buffer(copy=copy._buffer)
        self._length = copy._length

    def __len__(self) -> Int:
        """Returns the number of bits.

        Returns:
            The bit count.
        """
        return self._length

    def byte_length(self) -> Int:
        """Returns the number of bytes the bits occupy.

        Returns:
            The byte count.
        """
        return bytes_for(self._length)

    def unsafe_ptr(ref self) -> Pointer[UInt8, origin_of(self)]:
        """Returns a pointer to the packed bytes.

        Returns:
            A pointer to the first byte.
        """
        return self._buffer.unsafe_ptr().unsafe_origin_cast[origin_of(self)]()

    def get(self, i: Int) -> Bool:
        """Returns the bit at a position without bounds checking.

        Args:
            i: The bit position. Must be less than `len(self)`.

        Returns:
            True if the value at `i` is present.
        """
        var byte = self._buffer.unsafe_ptr().unsafe_offset(i >> 3).unsafe_load()
        return ((byte >> UInt8(i & 7)) & 1) != 0

    def set(mut self, i: Int, value: Bool):
        """Sets the bit at a position without bounds checking.

        Args:
            i: The bit position. Must be less than `len(self)`.
            value: True to mark the value present, False to mark it null.
        """
        var slot = self._buffer.unsafe_ptr().unsafe_offset(i >> 3)
        var mask = UInt8(1) << UInt8(i & 7)
        var byte = slot.unsafe_load()
        if value:
            slot.unsafe_write(byte | mask)
        else:
            slot.unsafe_write(byte & ~mask)

    def set_all(mut self):
        """Marks every value present."""
        var nbytes = self.byte_length()
        var ptr = self._buffer.unsafe_ptr()
        for i in range(nbytes):
            ptr.unsafe_offset(i).unsafe_write(0xFF)
        self._clear_tail()

    def clear_all(mut self):
        """Marks every value null."""
        self._buffer.zero()

    def _clear_tail(mut self):
        """Zeroes the bits past `length` in the final byte.

        Every mutating path calls this. `count_ones` counts whole bytes and would
        otherwise report set padding bits as present values.
        """
        var used = self._length & 7
        if used == 0:
            return
        var last = self.byte_length() - 1
        var slot = self._buffer.unsafe_ptr().unsafe_offset(last)
        var mask = UInt8((1 << used) - 1)
        slot.unsafe_write(slot.unsafe_load() & mask)

    def count_ones(self) -> Int:
        """Returns the number of present values.

        Returns:
            The number of set bits.
        """
        var total = 0
        var ptr = self._buffer.unsafe_ptr()
        for i in range(self.byte_length()):
            total += Int(pop_count(ptr.unsafe_offset(i).unsafe_load()))
        return total

    def null_count(self) -> Int:
        """Returns the number of null values.

        Returns:
            The number of clear bits below `length`.
        """
        return self._length - self.count_ones()

    def all_valid(self) -> Bool:
        """Reports whether the bitmap contains no nulls.

        Returns:
            True if every bit below `length` is set.
        """
        return self.count_ones() == self._length

    def any_valid(self) -> Bool:
        """Reports whether the bitmap contains at least one present value.

        Returns:
            True if any bit below `length` is set.
        """
        var ptr = self._buffer.unsafe_ptr()
        for i in range(self.byte_length()):
            if ptr.unsafe_offset(i).unsafe_load() != 0:
                return True
        return False

    def set_range(mut self, start: Int, end: Int, value: Bool):
        """Sets a half-open range of bits.

        The whole bytes in the middle are written directly and only the two
        partial bytes at the ends go through the bit path, which is what makes
        this worth having over a loop of `set`.

        Args:
            start: The first bit position, inclusive.
            end: The last bit position, exclusive.
            value: True to mark present, False to mark null.
        """
        if start >= end:
            return
        var first_full = round_up(start, 8) // 8
        var last_full = end // 8
        if first_full >= last_full:
            for i in range(start, end):
                self.set(i, value)
            return
        for i in range(start, first_full * 8):
            self.set(i, value)
        var fill = UInt8(0xFF) if value else UInt8(0)
        var ptr = self._buffer.unsafe_ptr()
        for byte in range(first_full, last_full):
            ptr.unsafe_offset(byte).unsafe_write(fill)
        for i in range(last_full * 8, end):
            self.set(i, value)
        self._clear_tail()

    def and_with(mut self, other: Self):
        """Intersects this bitmap with another in place.

        A binary operation on two columns produces a null wherever either input
        is null, which is this.

        Args:
            other: The bitmap to intersect with. Must be the same length.
        """
        var ptr = self._buffer.unsafe_ptr()
        var rhs = other._buffer.unsafe_ptr()
        for i in range(self.byte_length()):
            var slot = ptr.unsafe_offset(i)
            slot.unsafe_write(
                slot.unsafe_load() & rhs.unsafe_offset(i).unsafe_load()
            )
        self._clear_tail()

    def or_with(mut self, other: Self):
        """Unions this bitmap with another in place.

        Args:
            other: The bitmap to union with. Must be the same length.
        """
        var ptr = self._buffer.unsafe_ptr()
        var rhs = other._buffer.unsafe_ptr()
        for i in range(self.byte_length()):
            var slot = ptr.unsafe_offset(i)
            slot.unsafe_write(
                slot.unsafe_load() | rhs.unsafe_offset(i).unsafe_load()
            )
        self._clear_tail()

    def invert(mut self):
        """Flips every bit below `length`."""
        var ptr = self._buffer.unsafe_ptr()
        for i in range(self.byte_length()):
            var slot = ptr.unsafe_offset(i)
            slot.unsafe_write(~slot.unsafe_load())
        self._clear_tail()

    def slice(self, start: Int, end: Int) -> Self:
        """Returns a new bitmap holding the bits in a half-open range.

        This copies rather than aliasing. An Arrow-style zero-copy slice would
        need a bit offset on every read, which costs a shift in the inner loop of
        every kernel. Slices of validity are rare enough that the copy is the
        better trade; see docs/specs/02-architecture.md.

        Args:
            start: The first bit position, inclusive.
            end: The last bit position, exclusive.

        Returns:
            A bitmap of length `end - start`.
        """
        var out = Self(end - start, all_valid=False)
        if start & 7 == 0:
            # Byte aligned, so the copy is a memcpy plus a tail fixup.
            var nbytes = bytes_for(end - start)
            unsafe_memcpy(
                dest=out._buffer.unsafe_ptr(),
                src=self._buffer.unsafe_ptr().unsafe_offset(start >> 3),
                count=nbytes,
            )
            out._clear_tail()
            return out^
        for i in range(start, end):
            out.set(i - start, self.get(i))
        return out^

    def to_list(self) -> List[Bool]:
        """Returns the bits as a list, for tests and for small interop paths.

        Returns:
            A list of length `len(self)`.
        """
        var out = List[Bool](capacity=self._length)
        for i in range(self._length):
            out.append(self.get(i))
        return out^
