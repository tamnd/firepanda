"""The typed column.

`Array[dt]` is the unit every kernel operates on: a values buffer, a validity
bitmap, and a length. It owns its memory. There is no offset field, because a
sliced array copies; see the note on `Bitmap.slice` for why.

The dtype is a parameter rather than a field, so `Array[DType.int64]` and
`Array[DType.float64]` are different types and a kernel written against one is
compiled for one. That is the whole performance argument for this library and it
is also the whole code-size problem; `AnyArray` in `any.mojo` is the seam where
the parameter becomes a value again.
"""

from std.sys.info import size_of

from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.logical import LogicalType, logical_for

from .data import ColumnData


struct Array[dt: DType](Copyable, Movable, Sized):
    """A typed, nullable column of fixed-width values."""

    var data: ColumnData
    """The values buffer, validity bitmap and length."""

    def __init__(out self, length: Int):
        """Constructs an array of zeros with every value marked present.

        Args:
            length: The number of values.
        """
        self.data = ColumnData(
            byte_size=length * size_of[Self.dt](), length=length
        )

    def __init__(out self, var data: ColumnData):
        """Constructs an array over storage the caller already built.

        Args:
            data: The storage. Its values buffer must hold at least `length`
                elements of `dt`.
        """
        self.data = data^

    def __init__(out self, *, copy: Self):
        """Deep-copies an array.

        Args:
            copy: The array to copy.
        """
        self.data = ColumnData(copy=copy.data)

    def into_data(deinit self) -> ColumnData:
        """Gives up the storage without copying it.

        This is how a typed column becomes an untyped one. It consumes the array.

        Returns:
            The storage.
        """
        return self.data^

    def __len__(self) -> Int:
        """Returns the number of values.

        Returns:
            The length.
        """
        return self.data.length

    def dtype(self) -> LogicalType:
        """Returns the logical type of the column.

        Returns:
            The default logical type for the physical dtype.
        """
        return logical_for(Self.dt)

    def unsafe_ptr(ref self) -> Pointer[Scalar[Self.dt], origin_of(self)]:
        """Returns a typed pointer to the first value.

        Returns:
            A pointer valid for at least `len(self)` elements, and readable for a
            further SIMD register beyond that thanks to buffer padding.
        """
        return self.data.values.bitcast[Self.dt]().unsafe_origin_cast[
            origin_of(self)
        ]()

    def __getitem__(self, i: Int) -> Scalar[Self.dt]:
        """Returns the value at a position, present or not.

        A null position holds whatever the buffer holds there, which is zero for
        a freshly built array. Callers that care must check `is_valid`.

        Args:
            i: The position. Must be less than `len(self)`.

        Returns:
            The value.
        """
        return (
            self.data.values.bitcast[Self.dt]().unsafe_offset(i).unsafe_load()
        )

    def __setitem__(mut self, i: Int, value: Scalar[Self.dt]):
        """Writes the value at a position and leaves validity alone.

        Args:
            i: The position. Must be less than `len(self)`.
            value: The value to write.
        """
        self.data.values.bitcast[Self.dt]().unsafe_offset(i).unsafe_write(value)

    def is_valid(self, i: Int) -> Bool:
        """Reports whether the value at a position is present.

        Args:
            i: The position. Must be less than `len(self)`.

        Returns:
            True if the value is not null.
        """
        return self.data.validity.get(i)

    def set_null(mut self, i: Int):
        """Marks a position null and zeroes its value.

        The value is zeroed so that a null never leaks a stale number into a
        kernel that reads without masking. Kernels are allowed to rely on this.

        Args:
            i: The position. Must be less than `len(self)`.
        """
        self.data.validity.set(i, False)
        self[i] = Scalar[Self.dt](0)

    def set_valid(mut self, i: Int, value: Scalar[Self.dt]):
        """Writes a value and marks the position present.

        Args:
            i: The position. Must be less than `len(self)`.
            value: The value to write.
        """
        self[i] = value
        self.data.validity.set(i, True)

    def null_count(self) -> Int:
        """Returns the number of null values.

        Returns:
            The count of clear validity bits.
        """
        return self.data.validity.null_count()

    def load[width: Int](self, i: Int) -> SIMD[Self.dt, width]:
        """Loads a SIMD register of values starting at a position.

        Args:
            i: The first position. May run up to one register past the length
                because the values buffer is padded to a 64-byte multiple.

        Parameters:
            width: The register width in elements.

        Returns:
            The loaded register.
        """
        return (
            self.data.values.bitcast[Self.dt]()
            .unsafe_offset(i)
            .unsafe_load[width=width]()
        )

    def store[width: Int](mut self, i: Int, value: SIMD[Self.dt, width]):
        """Stores a SIMD register of values starting at a position.

        Args:
            i: The first position.
            value: The register to store.

        Parameters:
            width: The register width in elements.
        """
        self.data.values.bitcast[Self.dt]().unsafe_offset(i).unsafe_store(value)

    def slice(self, start: Int, end: Int) -> Self:
        """Returns a copy of a half-open range of the array.

        Args:
            start: The first position, inclusive.
            end: The last position, exclusive.

        Returns:
            A new array of length `end - start`.
        """
        var out = Self(end - start)
        for i in range(start, end):
            out[i - start] = self[i]
        out.data.validity = self.data.validity.slice(start, end)
        return out^

    def to_list(self) -> List[Scalar[Self.dt]]:
        """Returns the values as a list, for tests and small interop paths.

        Returns:
            A list of the array's length, including the values under nulls.
        """
        var out = List[Scalar[Self.dt]](capacity=self.data.length)
        for i in range(self.data.length):
            out.append(self[i])
        return out^


def from_list[dt: DType](values: List[Scalar[dt]]) -> Array[dt]:
    """Builds an array from a list, with every value marked present.

    Args:
        values: The values.

    Parameters:
        dt: The dtype.

    Returns:
        An array of the same length.
    """
    var out = Array[dt](len(values))
    for i in range(len(values)):
        out[i] = values[i]
    return out^
