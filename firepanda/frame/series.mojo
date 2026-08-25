"""A named column.

`Series` is a name and an `AnyArray`, and almost every method on it forwards to a
kernel. That is the whole design and it is deliberate: the frame layer decides
what an operation means and the kernel layer decides how it runs, and the two are
not allowed to leak into each other. `docs/specs/11-package-layout.md` puts it as
`kernel` never seeing a `DataFrame`, and this file is the other half of that rule.

Two divergences from pandas are visible here and both are from
`docs/specs/04-python-dx.md`. There is no index, so a `Series` is positional and
nothing aligns on labels. And nothing mutates: every method returns a new
`Series`, there is no `inplace=`, and the copy is real rather than a view. An
eager API that hands out views has to answer what happens when the parent is
dropped, and firepanda answers it by not having views until the plan layer at M4
makes the lifetime explicit.

The cost of that is a copy per operation and it is worth being honest about the
size of it. `head(10)` on a million row column copies ten values. `cast` copies
everything, and would have even as a view. The one that stings is column access
off a `DataFrame`, which is a full copy today, and it is the reason `at` exists
next to `column`.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.logical import LogicalType
from firepanda.kernel.cast import cast_any
from firepanda.kernel.select import filter_any, take_any
from firepanda.kernel.sort import argsort_any


struct Series(Copyable, Movable, Sized, Writable):
    """A named, positional, immutable column."""

    var name: String
    """The column name. Not unique by construction; a `DataFrame` enforces that."""

    var values: AnyArray
    """The data, with the dtype carried as a field."""

    def __init__(out self, name: String, var values: AnyArray):
        """Constructs a series over a column the caller already built.

        Args:
            name: The column name.
            values: The data. Consumed, with no copy of the buffers.
        """
        self.name = name
        self.values = values^

    def __init__[dt: DType](out self, name: String, var values: Array[dt]):
        """Constructs a series from a typed column.

        Args:
            name: The column name.
            values: The data. Consumed, with no copy of the buffers.

        Parameters:
            dt: The dtype being erased.
        """
        self.name = name
        self.values = AnyArray(values^)

    def __init__(out self, *, copy: Self):
        """Deep-copies a series.

        Args:
            copy: The series to copy.
        """
        self.name = copy.name
        self.values = AnyArray(copy=copy.values)

    def into_values(deinit self) -> AnyArray:
        """Gives up the column without copying it, dropping the name.

        A `DataFrame` keeps names in its schema, so building one out of series
        has to separate the two, and Mojo will not let a caller reach in and move
        `values` out from under a struct that still owns a `String`. `deinit`
        here is the whole point: it says the rest of the series is being torn
        down, which is what makes the move legal.

        Returns:
            The data.
        """
        return self.values^

    def __len__(self) -> Int:
        """Returns the number of rows.

        Returns:
            The length.
        """
        return len(self.values)

    def dtype(self) -> DType:
        """Returns the physical dtype.

        Returns:
            The dtype the values buffer is laid out as.
        """
        return self.values.dtype()

    def logical(self) -> LogicalType:
        """Returns the logical type.

        Returns:
            The physical dtype plus what the values mean.
        """
        return self.values.type

    def is_valid(self, i: Int) -> Bool:
        """Reports whether a row is present.

        Args:
            i: The position. Must be less than the length.

        Returns:
            True if the value is not null.
        """
        return self.values.is_valid(i)

    def null_count(self) -> Int:
        """Returns the number of nulls.

        Returns:
            The count of clear validity bits.
        """
        return self.values.null_count()

    def as_typed[dt: DType](self) raises -> Array[dt]:
        """Returns the data as a typed column.

        Parameters:
            dt: The expected dtype.

        Returns:
            A typed copy of the column.

        Raises:
            If `dt` is not the series' dtype.
        """
        return self.values.as_typed[dt]()

    def rename(self, name: String) raises -> Self:
        """Returns the same data under a different name.

        Args:
            name: The new name.

        Returns:
            A renamed copy.
        """
        return Self(name, AnyArray(copy=self.values))

    def cast(self, to: DType) raises -> Self:
        """Returns the series converted to another dtype.

        No range check. Casting 300 to int8 gives 44, which is what the hardware
        gives and what `firepanda.kernel.cast` documents. pandas raises here and
        the decision to follow it or not belongs to the milestone that settles
        the error model, not to this one.

        Args:
            to: The target dtype.

        Returns:
            A series of dtype `to`, null in the same places.

        Raises:
            If either dtype has no physical layout.
        """
        return Self(self.name, cast_any(self.values, to))

    def take(self, indices: List[Int]) raises -> Self:
        """Returns rows gathered by position.

        Args:
            indices: The positions to gather. A negative index produces a null,
                which is how a left join reports a missing row.

        Returns:
            A series of length `len(indices)`.

        Raises:
            If the dtype has no physical layout.
        """
        return Self(self.name, take_any(self.values, indices))

    def filter(self, mask: Array[DType.bool]) raises -> Self:
        """Returns the rows where the mask is true.

        A null in the mask drops the row. Keeping it would mean `filter(m)` and
        `filter(not m)` both contain it.

        Args:
            mask: The mask. Must be the same length as the series.

        Returns:
            A series holding the kept rows in their original order.

        Raises:
            If the mask length does not match, or the dtype has no physical
            layout.
        """
        if len(mask) != len(self):
            raise Error(
                "filter mask must be the same length as the series; series has "
                + String(len(self))
                + " rows and mask has "
                + String(len(mask))
            )
        return Self(self.name, filter_any(self.values, mask))

    def slice(self, start: Int, end: Int) raises -> Self:
        """Returns a half-open range of rows.

        Args:
            start: The first position, inclusive.
            end: The last position, exclusive.

        Returns:
            A series of length `end - start`.

        Raises:
            If the range is reversed or runs past either end of the series.
        """
        _check_range(start, end, len(self), "series")
        return Self(self.name, self.values.slice(start, end))

    def head(self, n: Int = 5) raises -> Self:
        """Returns the first rows.

        Args:
            n: How many rows. Clamped to the length, and a negative count gives
                an empty series rather than an error, because `head(len - k)` is
                a normal thing to write and going negative there is not worth an
                exception.

        Returns:
            A series of at most `n` rows.
        """
        return self.slice(0, _clamp(n, len(self)))

    def tail(self, n: Int = 5) raises -> Self:
        """Returns the last rows.

        Args:
            n: How many rows, clamped as in `head`.

        Returns:
            A series of at most `n` rows.
        """
        var length = len(self)
        return self.slice(length - _clamp(n, length), length)

    def argsort(
        self, descending: Bool = False, nulls_first: Bool = False
    ) raises -> Array[DType.uint32]:
        """Returns the row order that sorts the series.

        Args:
            descending: Largest first.
            nulls_first: Put the nulls at the front rather than the back.

        Returns:
            A permutation of `[0, len(self))`.

        Raises:
            If the dtype is not sortable.
        """
        return argsort_any(self.values, descending, nulls_first)

    def sort_values(
        self, descending: Bool = False, nulls_first: Bool = False
    ) raises -> Self:
        """Returns the series in sorted order.

        Args:
            descending: Largest first.
            nulls_first: Put the nulls at the front rather than the back.

        Returns:
            A sorted series.

        Raises:
            If the dtype is not sortable.
        """
        var order = self.argsort(descending, nulls_first)
        return self.take(_to_positions(order))

    def write_to(self, mut writer: Some[Writer]):
        """Writes a one line summary.

        This is not the display layer. Rendering the values with the alignment,
        truncation and null spelling that a notebook needs is its own change;
        what this gives is enough to identify a column in a test failure.

        Args:
            writer: The sink.
        """
        writer.write(
            "Series[",
            self.name,
            ": ",
            self.values.type,
            "] ",
            len(self),
            " rows, ",
            self.null_count(),
            " null",
        )


def _clamp(n: Int, length: Int) -> Int:
    """Bounds a row count to `[0, length]`."""
    if n < 0:
        return 0
    if n > length:
        return length
    return n


def _check_range(start: Int, end: Int, length: Int, what: String) raises:
    """Raises if a half-open range is not inside `[0, length]`."""
    if start < 0 or end > length or start > end:
        raise Error(
            "slice ["
            + String(start)
            + ", "
            + String(end)
            + ") is out of range for a "
            + what
            + " of "
            + String(length)
            + " rows"
        )


def _to_positions(order: Array[DType.uint32]) -> List[Int]:
    """Widens a permutation into the signed index list `take` wants.

    The two representations exist for different reasons and neither is going
    away. A permutation is uint32 because it is one per row and the sort moves it
    on every pass. A take list is signed because a negative index means null,
    which is how an outer join reports a row that was not there. Converting costs
    one pass and one allocation, and a `DataFrame` sort does it once for the
    whole frame rather than once per column.
    """
    var n = len(order)
    var out = List[Int](capacity=n)
    var values = order.unsafe_ptr()
    for i in range(n):
        out.append(Int(values.unsafe_offset(i).unsafe_load()))
    return out^
