"""The type-erased column.

A frame holds columns of different types in one list, so at the frame boundary
the dtype has to stop being a parameter and start being a value. `AnyArray` is
that boundary. It holds the same storage `Array[dt]` holds plus the dtype as a
field, and it hands out a typed column through `as_typed` and `as_typed_view`.
The first copies and the second borrows, and anything that only reads should be
using the second.

`as_typed[dt]()` is a checked reinterpretation, not a conversion. It raises if the
requested dtype does not match the one in the field. That check is the only thing
standing between a wrong dispatch and reading an int64 column as float64, so it is
never elided, not even in release; the cost is one comparison against a value that
is in cache and predicted.

A string column is the one thing that does not fit in `ColumnData`, because its
elements are not all the same width. It is carried in `text` alongside, and `data`
holds an empty values buffer, the length, and a copy of the validity.

The copy of the validity is the one piece of duplication in this struct and it is
deliberate. Every kernel that only looks at whether a row is present, which is
`is_null`, `is_not_null` and the all-present mask a join and a group by both build,
reads `data.validity` and does not care what the values are. Keeping the bitmap
where those already look means they need no string case at all, and a column is
immutable once constructed, so the two copies are written from the same source in
the same call and cannot drift. The cost is one bit per row per string column, and
it goes away when columns are refcounted rather than deep copied.

What does need a string case is anything that reads values, because
`LogicalType.STRING` has physical dtype uint8 and would otherwise match the `uint8`
arm of a dispatch and read the first byte of a 16 byte view as the value. Every one
of those call sites asks `is_string()` first.
"""

from std.memory import unsafe_memcpy

from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.lists import dtype_size
from firepanda.dtype.logical import LogicalType, TypeKind, logical_for

from .array import Array
from .data import ColumnData
from .strings import StringArray


comptime ColumnRefs[o: ImmOrigin] = List[Pointer[AnyArray, o]]
"""A borrowed set of columns.

Anything that reads several of a frame's columns wants them borrowed. Taking a
`List[AnyArray]` means the caller either gives up ownership or copies, and for a
group by on six key columns of ten million rows the copy is more work than the
group by itself. This costs a pointer a column and the caller keeps what it has.

The origin is carried rather than erased, and that is not a formality. A borrow
with an untracked origin lets the compiler destroy the frame after the argument
is evaluated and before the callee runs, and what the callee then reads is freed
memory that still looks like a column: `group_ordinals` on a frame built inline
returned one group for eight rows with three distinct keys rather than crashing.
With the origin in the type that program does not compile.
"""


struct AnyArray(Copyable, Movable, Sized):
    """A column whose dtype is a runtime value."""

    var data: ColumnData
    """The values buffer, validity bitmap and length."""

    var type: LogicalType
    """The column type, carried as data rather than as a parameter."""

    var text: Optional[StringArray]
    """The variable width elements, present only for a string column."""

    def __init__(out self, var data: ColumnData, type: LogicalType):
        """Constructs a type-erased column over storage the caller built.

        Args:
            data: The storage.
            type: The column type.
        """
        self.data = data^
        self.type = type
        self.text = None

    def __init__[dt: DType](out self, var typed: Array[dt]):
        """Erases the dtype of a typed array, taking ownership of its buffers.

        Args:
            typed: The array to erase. Consumed, with no copy of the buffers.

        Parameters:
            dt: The dtype being erased.
        """
        self.data = typed^.into_data()
        self.type = logical_for(dt)
        self.text = None

    def __init__(out self, var strings: StringArray):
        """Erases a string column, taking ownership of its buffers.

        Args:
            strings: The column to erase. Consumed.
        """
        var length = len(strings)
        self.data = ColumnData(Buffer(0), Bitmap(copy=strings.validity), length)
        self.type = LogicalType.STRING
        self.text = strings^

    def __init__(out self, *, copy: Self):
        """Deep-copies a column.

        Args:
            copy: The column to copy.
        """
        self.data = ColumnData(copy=copy.data)
        self.type = copy.type
        self.text = Optional[StringArray](copy=copy.text)

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

    def nbytes(self) -> Int:
        """Returns the bytes the column's buffers occupy.

        Every buffer the column actually has, which is not the same set for
        every column: a fixed width column has values and maybe a validity
        bitmap, and a string column has views and a payload instead of values.
        The bitmap is counted at one byte per eight rows rather than at its
        allocated capacity, so the answer is about the data and not about the
        allocator's rounding.

        pandas reports this as `nbytes` and reports it for a `RangeIndex` too,
        where the number it gives is the size of the Python object rather than of
        any labels. `firepanda/py/index.mojo` answers zero there instead.

        Returns:
            The size in bytes.
        """
        var out = self.data.validity.byte_length()
        if self.text:
            ref held = self.text.value()
            return out + len(held.views) + len(held.payload)
        return out + len(self.data.values)

    def is_string(self) -> Bool:
        """Reports whether the column holds variable width elements.

        Every dispatch that reads values has to ask this before it matches on
        `dtype()`, because a string column's physical dtype is uint8 and would
        otherwise select the uint8 arm and read a view byte as a value.

        Returns:
            True for a string or binary column.
        """
        return Bool(self.text)

    def strings(ref self) raises -> ref[self.text.value()] StringArray:
        """Returns the variable width elements without copying them.

        Returns:
            A reference to the string column, valid as long as this one is.

        Raises:
            If the column is not a string column.
        """
        if not self.text:
            raise Error(
                "column is " + String(self.type) + ", not a string column"
            )
        return self.text.value()

    def into_strings(deinit self) raises -> StringArray:
        """Converts to a string column without copying, consuming this one.

        Returns:
            The string column.

        Raises:
            If the column is not a string column.
        """
        if not self.text:
            raise Error(
                "column is " + String(self.type) + ", not a string column"
            )
        var held = self.text^
        return held.take()

    def check_dtype[dt: DType](self) raises:
        """Raises unless the column has a given dtype.

        A string column fails this for every `dt`. Its physical dtype is uint8,
        so without the first check `as_typed[DType.uint8]()` would hand back a
        column over the views buffer, whose values are not the column's values.

        Parameters:
            dt: The expected dtype.

        Raises:
            If the column's dtype differs, or if it is a string column.
        """
        if self.is_string():
            raise Error(
                "column is "
                + String(self.type)
                + " and has no fixed width values; use strings() instead"
            )
        if self.type.physical != dt:
            raise Error(
                "dtype mismatch: column is "
                + String(self.type.physical)
                + ", requested "
                + String(dt)
            )

    def as_typed[dt: DType](self) raises -> Array[dt]:
        """Returns a typed copy of the column.

        Use this only when the caller needs to own the result. Anything that
        reads and hands the column back should take `as_typed_view`, which
        borrows the same storage and copies nothing.

        Parameters:
            dt: The dtype to read the column as.

        Returns:
            A typed array holding the same values.

        Raises:
            If `dt` is not the column's dtype.
        """
        self.check_dtype[dt]()
        return Array[dt](ColumnData(copy=self.data))

    def as_typed_view[dt: DType](ref self) raises -> ref[self.data] Array[dt]:
        """Returns a typed reference to the column, borrowing rather than copying.

        `as_typed` is a deep copy of every byte the column holds, which for a
        forty megabyte key column is a real cost, and a group by pays it once
        per key column on every query because `factorize` takes an `Array[dt]`
        and there was no other way to produce one from a borrowed `AnyArray`.
        This is that other way. It hands back a reference into the column's own
        storage, so it copies nothing and the caller may only read.

        `Array[dt]` holds one field, a `ColumnData`, and a struct of one field
        has that field's layout, so a pointer to the storage is a pointer to the
        array. That is an assumption about layout and it is asserted in
        `tests/test_array.mojo` rather than left to be noticed later.

        Parameters:
            dt: The dtype to read the column as.

        Returns:
            A reference to the column read as a typed array.

        Raises:
            If `dt` is not the column's dtype.
        """
        self.check_dtype[dt]()
        return Pointer(to=self.data).unsafe_bitcast[Array[dt]]()[]

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

    def slice(self, start: Int, end: Int) raises -> Self:
        """Returns a copy of a half-open range of the column.

        For a fixed width column this needs no dispatch at all. A slice moves
        bytes and does not look at them, so the element width coming from
        `dtype_size` as a runtime value is all it needs, and the whole thing
        compiles to a single copy of the loop instead of one per dtype.

        A string column cannot be cut that way, because a run of views refers to
        payload bytes scattered through the block, so it is rebuilt element by
        element. That is the same choice `StringArray.slice` documents.

        Args:
            start: The first position, inclusive.
            end: The last position, exclusive.

        Returns:
            A new column of length `end - start` and the same dtype.

        Raises:
            If the range is outside a string column.
        """
        if self.is_string():
            return Self(self.strings().slice(start, end))
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


def borrow_columns[o: ImmOrigin](ref[o] cols: List[AnyArray]) -> ColumnRefs[o]:
    """Borrows every column in a list.

    Parameters:
        o: The origin of the list, which the references inherit.

    Args:
        cols: The columns.

    Returns:
        One reference per column, in order.
    """
    var out = ColumnRefs[o](capacity=len(cols))
    for i in range(len(cols)):
        out.append(Pointer(to=cols[i]).unsafe_origin_cast[o]())
    return out^
