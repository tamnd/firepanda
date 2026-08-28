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


def _low_mask(bits: Int) -> UInt64:
    """Returns a word with its lowest `bits` bits set.

    The full-word case is spelled out rather than shifted into, because shifting
    a 64-bit word by 64 is undefined and on x86 quietly shifts by zero, which
    would produce a mask of one bit where the caller asked for all of them.

    Args:
        bits: How many low bits to set, from 0 to 64.

    Returns:
        The mask.
    """
    if bits >= 64:
        return UInt64.MAX
    return (UInt64(1) << UInt64(bits)) - 1


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

    def word_count(self) -> Int:
        """Returns how many 64-bit words cover the bitmap.

        Returns:
            The word count, rounded up. Reading every one of them is safe: the
            buffer is padded to a 64-byte multiple and the padding is zero.
        """
        return (self._length + 63) // 64

    def unsafe_word(self, w: Int) -> UInt64:
        """Returns the validity bits for values `w * 64` through `w * 64 + 63`.

        Bit 0 of the result is the lowest-numbered value, matching the Arrow
        packing. Words past the end read as zero, which is to say as null, so a
        caller that bounds its loop by length rather than by word count gets the
        right answer either way.

        Args:
            w: The word index. Must be less than `word_count()`.

        Returns:
            The word.
        """
        return (
            self._buffer.bitcast[DType.uint64]().unsafe_offset(w).unsafe_load()
        )

    def unsafe_set_word(mut self, w: Int, value: UInt64):
        """Writes the validity bits for values `w * 64` through `w * 64 + 63`.

        The whole word goes down, including the bits past `length` in the last
        one. That is in bounds because the buffer is padded, and it is correct
        because a caller building a bitmap forward never sets a bit it has not
        reached. A caller that does set one has broken the tail-is-zero rule that
        `count_ones` depends on.

        Args:
            w: The word index. Must be less than `word_count()`.
            value: The word.
        """
        self._buffer.bitcast[DType.uint64]().unsafe_offset(w).unsafe_write(
            value
        )

    def count_ones(self) -> Int:
        """Returns the number of present values.

        A word at a time rather than a byte at a time. The tail bits past
        `length` are cleared by every mutating path and the buffer padding is
        zero, so the last word needs no masking and the loop needs no remainder.

        Returns:
            The number of set bits.
        """
        var total = 0
        for w in range(self.word_count()):
            total += Int(pop_count(self.unsafe_word(w)))
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

    def paste(mut self, at: Int, other: Self, count: Int):
        """Copies the first `count` bits of another bitmap in at a bit offset.

        `slice` reads a run out of a bitmap and this writes one in, and the
        asymmetry between them is that the destination here is shared: the bits
        on either side of the run belong to somebody else and have to come
        through untouched. So every word is a read, a mask and a write rather
        than a store, and a run that does not start on a word boundary is
        written as two halves of the word it straddles.

        This is what a concat of columns with nulls needs. Without it the only
        way to place a part's validity at an arbitrary row is a loop over bits,
        which is sixty four times the work and was the shape of the loop at the
        bottom of this package's concat kernel.

        Args:
            at: The destination bit position the run starts at. The run must fit
                inside this bitmap.
            other: The bitmap to read from.
            count: How many bits to copy, starting from the first bit of
                `other`. Must not exceed either bitmap.
        """
        if count <= 0:
            return
        for w in range((count + 63) // 64):
            var bits = count - w * 64
            if bits > 64:
                bits = 64
            var value = other.unsafe_word(w) & _low_mask(bits)

            var target = at + w * 64
            var word = target >> 6
            var offset = target & 63
            var low = 64 - offset
            if low > bits:
                low = bits

            var mask = _low_mask(low) << UInt64(offset)
            self.unsafe_set_word(
                word,
                (self.unsafe_word(word) & ~mask)
                | ((value << UInt64(offset)) & mask),
            )
            if bits > low:
                # Only reachable when the run is unaligned, so `low` is under 64
                # and the shift below is defined.
                var rest = _low_mask(bits - low)
                self.unsafe_set_word(
                    word + 1,
                    (self.unsafe_word(word + 1) & ~rest)
                    | ((value >> UInt64(low)) & rest),
                )

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
        if end <= start:
            return out^

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

        # Unaligned, so every output byte straddles two input bytes. Shifting a
        # byte at a time is eight times less work than walking bits and is the
        # only reason this branch is not the one that shows up in a profile. A
        # word at a time would be another eight, and needs unaligned 64-bit loads
        # that have to be assembled by hand near the end of the buffer, which is
        # a trade worth making the day a profile asks for it and not before.
        var shift = UInt8(start & 7)
        var carry = UInt8(8) - shift
        var source = self._buffer.unsafe_ptr()
        var target = out._buffer.unsafe_ptr()
        var first = start >> 3
        var available = self.byte_length()
        var nbytes = bytes_for(end - start)

        for j in range(nbytes):
            var low = source.unsafe_offset(first + j).unsafe_load() >> shift
            var high = UInt8(0)
            if first + j + 1 < available:
                high = (
                    source.unsafe_offset(first + j + 1).unsafe_load() << carry
                )
            target.unsafe_offset(j).unsafe_write(low | high)

        out._clear_tail()
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
