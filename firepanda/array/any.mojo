"""The type-erased column.

A frame holds columns of different types in one list, so at the frame boundary
the dtype has to stop being a parameter and start being a value. `AnyArray` is
that boundary. It holds the same storage `Array[dt]` holds plus the dtype as a
field, and it hands out a typed column through `as_typed`.

`as_typed[dt]()` is a checked reinterpretation, not a conversion. It raises if the
requested dtype does not match the one in the field. That check is the only thing
standing between a wrong dispatch and reading an int64 column as float64, so it is
never elided, not even in release; the cost is one comparison against a value that
is in cache and predicted.
"""

from std.memory import unsafe_memcpy

from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.lists import dtype_size
from firepanda.dtype.logical import LogicalType, logical_for

from .array import Array
from .data import ColumnData


struct AnyArray(Copyable, Movable, Sized):
    """A column whose dtype is a runtime value."""

    var data: ColumnData
    """The values buffer, validity bitmap and length."""

    var type: LogicalType
    """The column type, carried as data rather than as a parameter."""

    def __init__(out self, var data: ColumnData, type: LogicalType):
        """Constructs a type-erased column over storage the caller built.

        Args:
            data: The storage.
            type: The column type.
        """
        self.data = data^
        self.type = type

    def __init__[dt: DType](out self, var typed: Array[dt]):
        """Erases the dtype of a typed array, taking ownership of its buffers.

        Args:
            typed: The array to erase. Consumed, with no copy of the buffers.

        Parameters:
            dt: The dtype being erased.
        """
        self.data = typed^.into_data()
        self.type = logical_for(dt)

    def __init__(out self, *, copy: Self):
        """Deep-copies a column.

        Args:
            copy: The column to copy.
        """
        self.data = ColumnData(copy=copy.data)
        self.type = copy.type

    def __len__(self) -> Int:
        """Returns the number of values.

        Returns:
            The length.
        """
        return self.data.length

    def dtype(self) -> DType:
        """Returns the physical dtype.

        Returns:
            The dtype the values buffer is laid out as.
        """
        return self.type.physical

    def is_valid(self, i: Int) -> Bool:
        """Reports whether the value at a position is present.

        Args:
            i: The position. Must be less than the length.

        Returns:
            True if the value is not null.
        """
        return self.data.validity.get(i)

    def null_count(self) -> Int:
        """Returns the number of null values.

        Returns:
            The count of clear validity bits.
        """
        return self.data.validity.null_count()

    def check_dtype[dt: DType](self) raises:
        """Raises unless the column has a given dtype.

        Parameters:
            dt: The expected dtype.

        Raises:
            If the column's dtype differs.
        """
        if self.type.physical != dt:
            raise Error(
                "dtype mismatch: column is "
                + String(self.type.physical)
                + ", requested "
                + String(dt)
            )

    def as_typed[dt: DType](self) raises -> Array[dt]:
        """Returns a typed copy of the column.

        The copy is what makes this safe to hand out from a borrowed column. On
        the hot path, `dispatch` reads through `unsafe_ptr` instead, having
        already proved the dtype.

        Parameters:
            dt: The dtype to read the column as.

        Returns:
            A typed array holding the same values.

        Raises:
            If `dt` is not the column's dtype.
        """
        self.check_dtype[dt]()
        return Array[dt](ColumnData(copy=self.data))

    def into_typed[dt: DType](deinit self) raises -> Array[dt]:
        """Converts to a typed column without copying, consuming this one.

        Parameters:
            dt: The dtype to read the column as.

        Returns:
            A typed array over the same buffers.

        Raises:
            If `dt` is not the column's dtype.
        """
        self.check_dtype[dt]()
        return Array[dt](self.data^)

    def slice(self, start: Int, end: Int) -> Self:
        """Returns a copy of a half-open range of the column.

        This is the one column operation that needs no dispatch at all. A slice
        moves bytes and does not look at them, so the element width coming from
        `dtype_size` as a runtime value is all it needs, and the whole thing
        compiles to a single copy of the loop instead of one per dtype.

        Args:
            start: The first position, inclusive.
            end: The last position, exclusive.

        Returns:
            A new column of length `end - start` and the same dtype.
        """
        var width = dtype_size(self.type.physical)
        var n = end - start
        var values = Buffer(n * width)
        if n > 0:
            unsafe_memcpy(
                dest=values.unsafe_ptr(),
                src=self.data.values.unsafe_ptr().unsafe_offset(start * width),
                count=n * width,
            )
        return Self(
            ColumnData(values^, self.data.validity.slice(start, end), n),
            self.type,
        )

    def unsafe_ptr[dt: DType](ref self) -> Pointer[Scalar[dt], origin_of(self)]:
        """Returns a typed pointer to the values without checking the dtype.

        Callers must have checked the dtype already, normally by going through
        `dispatch`. This exists so that dispatch does not pay for a second check
        and a buffer copy on the hot path.

        Parameters:
            dt: The dtype to view the values as.

        Returns:
            A pointer to the first value.
        """
        return self.data.values.bitcast[dt]().unsafe_origin_cast[
            origin_of(self)
        ]()
