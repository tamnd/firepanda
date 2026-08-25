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

    def __init__(out self, *, copy: Self):
        """Deep-copies storage.

        Args:
            copy: The storage to copy.
        """
        self.values = Buffer(copy=copy.values)
        self.validity = Bitmap(copy=copy.validity)
        self.length = copy.length
