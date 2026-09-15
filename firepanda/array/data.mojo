"""The storage a column is made of.

`Array[dt]` and `AnyArray` are the same three things: a values buffer, a validity
bitmap, and a length. The only difference between them is whether the dtype is a
parameter or a field. Naming the shared part is worth doing for its own sake, and
it also has a mechanical benefit: erasing a typed column into an untyped one is a
single move of one field rather than a piecewise teardown, which Mojo does not
allow across a type that owns memory.
"""

from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer


struct ColumnData(Copyable, Movable):
    """The buffers and length behind any column."""

    var values: Buffer
    """The packed values."""

    var validity: Bitmap
    """One bit per value. Set means present."""

    var length: Int
    """The number of values."""

    def __init__(
        out self, var values: Buffer, var validity: Bitmap, length: Int
    ):
        """Constructs storage from buffers the caller already built.

        Args:
            values: The values buffer.
            validity: The validity bitmap.
            length: The number of values.
        """
        self.values = values^
        self.validity = validity^
        self.length = length

    def __init__(out self, *, byte_size: Int, length: Int):
        """Allocates zeroed storage with every value marked present.

        Args:
            byte_size: The size of the values buffer in bytes.
            length: The number of values.
        """
        self.values = Buffer(byte_size)
        self.validity = Bitmap(length)
        self.length = length

    def __init__(out self, *, overwritten_bytes: Int, length: Int):
        """Allocates unzeroed storage with every value marked present.

        See `Buffer.__init__(overwritten=)` for what the caller is promising.

        Args:
            overwritten_bytes: The size of the values buffer in bytes, every one
                of which the caller will write before reading it.
            length: The number of values.
        """
        self.values = Buffer(overwritten=overwritten_bytes)
        self.validity = Bitmap(length)
        self.length = length

    def __init__(out self, *, copy: Self):
        """Deep-copies storage.

        Args:
            copy: The storage to copy.
        """
        self.values = Buffer(copy=copy.values)
        self.validity = Bitmap(copy=copy.validity)
        self.length = copy.length

    def __init__(
        out self, *, window_of: Self, at: Int, length: Int, width: Int
    ):
        """Shares a run of rows of a column, copying nothing.

        The values and the bits both become windows onto what they were part
        of, so the piece costs an atomic per buffer rather than a memcpy per
        row. Writing through either one copies that piece and leaves the rest
        of the column where it is.

        Both windows carry the alignment promise their own constructors
        document, and both are satisfied by cutting on a morsel: a hundred and
        twenty eight thousand values is a multiple of 64 bytes at every fixed
        width we have, and a hundred and twenty eight thousand bits is a
        multiple of 64 bytes too.

        Args:
            window_of: The storage to share part of.
            at: The first row.
            length: The number of rows.
            width: The bytes per value, from `dtype_size`.
        """
        self.values = Buffer(
            window_of=window_of.values, at=at * width, size=length * width
        )
        self.validity = Bitmap(
            window_of=window_of.validity, at=at, length=length
        )
        self.length = length
