"""The variable width string column.

`Array[dt]` cannot hold a string, because every element of it is the same number
of bytes. This is the column that can, and it is the last thing missing before a
CSV reader has somewhere to put a text field.

The layout is the one `strview.mojo` describes, which has been sitting in the
repository unused since M0. Every element is a 16 byte `StringView`, and a string
of at most twelve bytes lives inside its own view. Only the long ones go to the
payload, so a column of country codes or status labels is one flat array with no
indirection at all, and a column of long strings still compares on its first four
bytes before either payload is touched.

That is the difference from the classic Arrow layout, where every element is an
offset into a data buffer and reading any string costs two dependent loads. The
cost of this layout is 16 bytes per element against Arrow's 4 or 8, which is a
loss on a column of long strings and a large win on the short ones that dataframes
are actually full of.

A finished column has exactly one payload block, so every long view carries block
index zero. The block field is not wasted: it is what lets a `ChunkedArray` of
strings later share payload across chunks without rewriting the views. The
builder accumulates into a payload buffer that doubles as it fills, so a string is
copied in once when it arrives and the buffer is handed to the column whole.

Building goes through `StringBuilder` rather than through the column, because the
column is immutable once it exists and because the builder is the shape a reader
wants: append a field, append a null, ask for the result. `finish` consumes it.

Slicing, taking and filtering all copy, which is what `Array` does and for the
same reason recorded on `Bitmap.slice`. A view into another column's payload
would make every column's lifetime depend on every column it was ever cut from.
"""

from std.collections.span import Span
from std.memory import unsafe_memcpy

from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.logical import LogicalType

comptime WORD = 8
"""Bytes compared at once when two long elements have to be walked.

One 64-bit word. A wider register would settle a long field in fewer steps, but
the fields this runs on are names and labels rather than paragraphs, and a
register that is wider than the field is a tail loop wearing a costume.
"""


from .strview import (
    INLINE_CAPACITY,
    PREFIX_LENGTH,
    StringView,
    VIEW_SIZE,
    make_inline,
    make_long,
    views_equal_short,
)


struct StringArray(Copyable, Movable, Sized):
    """A nullable column of variable width byte strings.

    The bytes are not validated as UTF-8. A CSV field is bytes, a Parquet byte
    array is bytes, and a column that refuses to hold what the file contains is a
    column that cannot read the file. Validation belongs to whatever asks for a
    `String` out of it.
    """

    var views: Buffer
    """One 16 byte `StringView` per element."""

    var payload: Buffer
    """The bytes of every element longer than twelve bytes, back to back."""

    var validity: Bitmap
    """One bit per element. Set means present."""

    var length: Int
    """The number of elements."""

    def __init__(
        out self,
        var views: Buffer,
        var payload: Buffer,
        var validity: Bitmap,
        length: Int,
    ):
        """Constructs a column from buffers the caller already built.

        Args:
            views: The views buffer, at least `length * 16` bytes.
            payload: The bytes of the long elements.
            validity: The validity bitmap.
            length: The number of elements.
        """
        self.views = views^
        self.payload = payload^
        self.validity = validity^
        self.length = length

    def __init__(out self, *, copy: Self):
        """Deep-copies a column.

        Args:
            copy: The column to copy.
        """
        self.views = Buffer(copy=copy.views)
        self.payload = Buffer(copy=copy.payload)
        self.validity = Bitmap(copy=copy.validity)
        self.length = copy.length

    def __len__(self) -> Int:
        """Returns the number of elements.

        Returns:
            The length.
        """
        return self.length

    def dtype(self) -> LogicalType:
        """Returns the logical type of the column.

        Returns:
            `LogicalType.STRING`.
        """
        return LogicalType.STRING

    def is_valid(self, i: Int) -> Bool:
        """Reports whether an element is present.

        Args:
            i: The element index.

        Returns:
            True if the element is not null.
        """
        return self.validity.get(i)

    def null_count(self) -> Int:
        """Returns how many elements are null.

        Returns:
            The null count.
        """
        return self.validity.null_count()

    def view(self, i: Int) -> StringView:
        """Returns one element's view.

        A null element's view is the view of the empty string, because the
        builder writes nothing for a null and the buffer arrives zeroed.

        Args:
            i: The element index.

        Returns:
            The view.
        """
        return (
            self.views.unsafe_ptr()
            .unsafe_bitcast[StringView]()
            .unsafe_offset(i)[]
        )

    def byte_length(self, i: Int) -> Int:
        """Returns the length of one element in bytes.

        This reads the view and nothing else, so it costs one load whether the
        element is inline or not.

        Args:
            i: The element index.

        Returns:
            The byte length. Zero for a null.
        """
        return len(self.view(i))

    def unsafe_bytes(ref self, i: Int) -> Span[UInt8, origin_of(self)]:
        """Returns one element's bytes without copying them.

        The span points either into the views buffer or into the payload, so it
        is valid exactly as long as the column is. Nothing here can outlive the
        column, which is what makes this safe to hand to a kernel and unsafe to
        store.

        Args:
            i: The element index.

        Returns:
            The bytes. Empty for a null.
        """
        var element = self.view(i)
        var count = len(element)
        if element.is_inline():
            return Span[UInt8, origin_of(self)](
                unsafe_ptr=self.views.unsafe_ptr()
                .unsafe_offset(i * VIEW_SIZE + PREFIX_LENGTH)
                .unsafe_origin_cast[origin_of(self)](),
                length=count,
            )
        return Span[UInt8, origin_of(self)](
            unsafe_ptr=self.payload.unsafe_ptr()
            .unsafe_offset(element.offset())
            .unsafe_origin_cast[origin_of(self)](),
            length=count,
        )

    def __getitem__(self, i: Int) -> String:
        """Returns one element as a string.

        This copies. `unsafe_bytes` is the one that does not, and is what a
        kernel should use.

        Args:
            i: The element index.

        Returns:
            The element's bytes as a string. Empty for a null.
        """
        var out = String()
        var bytes = self.unsafe_bytes(i)
        for j in range(len(bytes)):
            out += chr(Int(bytes[j]))
        return out^

    def equals(self, i: Int, other: Span[UInt8, _]) -> Bool:
        """Compares one element against a run of bytes.

        Args:
            i: The element index.
            other: The bytes to compare against.

        Returns:
            True if the element is present and byte-identical to `other`.
        """
        if not self.is_valid(i):
            return False
        var element = self.view(i)
        if len(element) != len(other):
            return False
        if element.is_inline():
            # Both are short, so build the other side's view and compare four
            # words. This is why short views are zero padded on construction.
            return views_equal_short(element, make_inline(other))
        return _bytes_equal(self.unsafe_bytes(i), other)

    def element_equals(self, i: Int, j: Int) -> Bool:
        """Compares two elements of the same column.

        The prefix check is the reason this layout exists. Two long strings that
        differ in their first four bytes are settled without either payload being
        read, and in a join key or a sort that is the common case.

        Args:
            i: The left element index.
            j: The right element index.

        Returns:
            True if both are present and byte-identical.
        """
        if not self.is_valid(i) or not self.is_valid(j):
            return False
        var left = self.view(i)
        var right = self.view(j)
        if len(left) != len(right) or left.prefix() != right.prefix():
            return False
        if left.is_inline():
            return views_equal_short(left, right)
        return _bytes_equal(self.unsafe_bytes(i), self.unsafe_bytes(j))

    def slice(self, start: Int, end: Int) raises -> Self:
        """Returns a copy of a contiguous run of elements.

        Args:
            start: The first element to keep.
            end: One past the last element to keep.

        Returns:
            A new column holding the run.

        Raises:
            If the bounds are reversed or outside the column.
        """
        if start < 0 or end > self.length or start > end:
            raise Error(
                String(
                    "slice [",
                    start,
                    ", ",
                    end,
                    ") is outside a column of ",
                    self.length,
                )
            )
        var builder = StringBuilder(capacity=end - start)
        for i in range(start, end):
            if self.is_valid(i):
                builder.append(self.unsafe_bytes(i))
            else:
                builder.append_null()
        return builder^.finish()

    def take(self, indices: List[Int]) raises -> Self:
        """Returns the elements at the given positions, in that order.

        Args:
            indices: The positions to gather. Repeats are allowed.

        Returns:
            A new column of the same length as `indices`.

        Raises:
            If any index is outside the column.
        """
        var builder = StringBuilder(capacity=len(indices))
        for k in range(len(indices)):
            var i = indices[k]
            if i < 0 or i >= self.length:
                raise Error(
                    String(
                        "take index ",
                        i,
                        " is outside a column of ",
                        self.length,
                    )
                )
            if self.is_valid(i):
                builder.append(self.unsafe_bytes(i))
            else:
                builder.append_null()
        return builder^.finish()

    def filter(self, mask: Bitmap) raises -> Self:
        """Returns the elements whose mask bit is set.

        Args:
            mask: One bit per element. Set means keep.

        Returns:
            A new column holding the kept elements in order.

        Raises:
            If the mask is shorter than the column.
        """
        if mask.byte_length() * 8 < self.length:
            raise Error("filter mask is shorter than the column")
        var builder = StringBuilder(capacity=self.length)
        for i in range(self.length):
            if not mask.get(i):
                continue
            if self.is_valid(i):
                builder.append(self.unsafe_bytes(i))
            else:
                builder.append_null()
        return builder^.finish()

    def to_list(self) -> List[String]:
        """Copies the column into a list of strings.

        A null becomes the empty string, which is lossy and is why this exists
        for tests and printing rather than for kernels.

        Returns:
            The elements in order.
        """
        var out = List[String](capacity=self.length)
        for i in range(self.length):
            out.append(self[i])
        return out^


struct StringBuilder(Movable, Sized):
    """Accumulates elements and hands back a finished column.

    The payload grows by doubling and is handed to the column by `finish`, so a
    long string is copied into it once when it arrives and again only when the
    buffer outgrows itself. Short strings never enter the payload at all, which
    means a column of labels does no payload work whatsoever.
    """

    var _views: List[StringView]
    var _payload: Buffer
    var _payload_size: Int
    var _nulls: List[Bool]

    def __init__(out self, capacity: Int = 0):
        """Constructs an empty builder.

        Args:
            capacity: How many elements are expected. A hint only.
        """
        self._views = List[StringView](capacity=capacity)
        self._payload = Buffer(0)
        self._payload_size = 0
        self._nulls = List[Bool](capacity=capacity)

    def __len__(self) -> Int:
        """Returns how many elements have been appended.

        Returns:
            The count.
        """
        return len(self._views)

    def append(mut self, bytes: Span[UInt8, _]):
        """Appends one present element.

        Args:
            bytes: The element's bytes. Copied.
        """
        if len(bytes) <= INLINE_CAPACITY:
            self._views.append(make_inline(bytes))
        else:
            var offset = self._payload_size
            self._reserve(offset + len(bytes))
            unsafe_memcpy(
                dest=self._payload.unsafe_ptr().unsafe_offset(offset),
                src=bytes.unsafe_ptr(),
                count=len(bytes),
            )
            self._payload_size = offset + len(bytes)
            self._views.append(make_long(bytes, 0, offset))
        self._nulls.append(False)

    def _reserve(mut self, needed: Int):
        """Makes room for at least `needed` payload bytes.

        The doubling is the point. A `Buffer` sized to exactly what was asked for
        turns a column of long strings into a quadratic copy, which is what the
        first version of this did and what made a `take` of a quarter of a million
        rows take three seconds instead of ten milliseconds.

        Args:
            needed: The payload size that has to fit.
        """
        if needed <= self._payload.capacity():
            return
        var grown = self._payload.capacity() * 2
        if grown < needed:
            grown = needed
        if grown < 64:
            grown = 64
        var bigger = Buffer(grown)
        if self._payload_size > 0:
            unsafe_memcpy(
                dest=bigger.unsafe_ptr(),
                src=self._payload.unsafe_ptr(),
                count=self._payload_size,
            )
        self._payload = bigger^

    def append_null(mut self):
        """Appends one null element.

        The view written is the view of the empty string, so reading a null's
        bytes gives an empty span rather than reading uninitialized memory.
        """
        self._views.append(StringView())
        self._nulls.append(True)

    def finish(deinit self) -> StringArray:
        """Builds the column and consumes the builder.

        Returns:
            The finished column.
        """
        var count = len(self._views)
        var views = Buffer(count * VIEW_SIZE)
        var target = views.unsafe_ptr().unsafe_bitcast[StringView]()
        for i in range(count):
            target.unsafe_offset(i)[] = self._views[i]

        # The payload moves out of the builder rather than being copied into a
        # buffer of the exact size. Slack at the end of it costs nothing, since
        # every read into the payload goes through an offset in a view.
        var payload = self._payload^
        payload.set_size(self._payload_size)

        var validity = Bitmap(count)
        for i in range(count):
            if self._nulls[i]:
                validity.set(i, False)

        return StringArray(views^, payload^, validity^, count)


def strings_from_list(values: List[String]) -> StringArray:
    """Builds a column with no nulls from a list of strings.

    Args:
        values: The elements, in order.

    Returns:
        The column.
    """
    var builder = StringBuilder(capacity=len(values))
    for i in range(len(values)):
        builder.append(values[i].as_bytes())
    return builder^.finish()


def _bytes_equal(a: Span[UInt8, _], b: Span[UInt8, _]) -> Bool:
    """Compares two runs of bytes of known equal length.

    A word at a time while there is a word left, then the tail. This only ever
    runs on elements longer than twelve bytes whose first four bytes already
    matched, so there is always at least one word to do and the loop is worth
    having.

    Args:
        a: The left bytes.
        b: The right bytes.

    Returns:
        True if every byte matches.
    """
    var count = len(a)
    var left = a.unsafe_ptr()
    var right = b.unsafe_ptr()
    var i = 0
    while i + WORD <= count:
        var chunk = left.unsafe_offset(i).unsafe_load[width=WORD]()
        var other = right.unsafe_offset(i).unsafe_load[width=WORD]()
        if chunk.ne(other).reduce_or():
            return False
        i += WORD
    while i < count:
        if a[i] != b[i]:
            return False
        i += 1
    return True
