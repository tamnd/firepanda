"""A named column.

`Series` is a name and an `AnyArray`, and almost every method on it forwards to a
kernel. That is the whole design and it is deliberate: the frame layer decides
what an operation means and the kernel layer decides how it runs, and the two are
not allowed to leak into each other. `docs/specs/11-package-layout.md` puts it as
`kernel` never seeing a `DataFrame`, and this file is the other half of that rule.

One divergence from pandas is visible here and it is from
`docs/specs/04-python-dx.md`: nothing mutates. Every method returns a new
`Series`, there is no `inplace=`, and the copy is real rather than a view. An
eager API that hands out views has to answer what happens when the parent is
dropped, and firepanda answers it by not having views until the plan layer at M4
makes the lifetime explicit.

The arithmetic at the bottom of the file matches rows by label and not by
position, which is the thing about pandas that surprises people who came from
arrays. `firepanda/frame/align.mojo` has the rules and the reasoning. The one to
know before reading anything else is that `+` aligns and `==` refuses to, which
is pandas' rule and not an oversight: a row that is on one side only has a
missing sum and has no comparison at all.

The cost of that is a copy per operation and it is worth being honest about the
size of it. `head(10)` on a million row column copies ten values. `cast` copies
everything, and would have even as a view. The one that stings is column access
off a `DataFrame`, which is a full copy today, and it is the reason `at` exists
next to `column`.
"""

from std.collections import Optional

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray
from firepanda.array.value import Value
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.logical import LogicalType
from firepanda.frame.align import align_pair, fill_one_sided, keep_rows
from firepanda.frame.display import DisplayOptions, render_column
from firepanda.frame.index import Index
from firepanda.kernel.binary import BinaryOp, binary_any, binary_value_any
from firepanda.kernel.cast import cast_any
from firepanda.kernel.nulls import (
    coalesce_any,
    fill_backward_any,
    fill_forward_any,
    is_not_null_any,
    is_null_any,
    missing_count_any,
)
from firepanda.kernel.select import filter_any, take_any
from firepanda.kernel.sort import argsort_any, is_sorted_any
from firepanda.kernel.unary import UnaryOp, unary_any


struct Series(Copyable, Movable, Sized, Writable):
    """A named, positional, immutable column."""

    var name: String
    """The column name. Not unique by construction; a `DataFrame` enforces that."""

    var values: AnyArray
    """The data, with the dtype carried as a field."""

    var index: Index
    """The row labels, defaulting to the range zero to n minus one, which is two
    integers and no memory. See `firepanda/frame/index.mojo`."""

    def __init__(out self, name: String, var values: AnyArray):
        """Constructs a series over a column the caller already built.

        Args:
            name: The column name.
            values: The data. Consumed, with no copy of the buffers.
        """
        self.name = name
        self.values = values^
        self.index = Index(len(self.values))

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
        self.index = Index(len(self.values))

    def __init__(out self, name: String, var values: StringArray):
        """Constructs a series from a string column.

        Args:
            name: The column name.
            values: The data. Consumed, with no copy of the buffers.
        """
        self.name = name
        self.values = AnyArray(values^)
        self.index = Index(len(self.values))

    def __init__(out self, *, copy: Self):
        """Deep-copies a series.

        Args:
            copy: The series to copy.
        """
        self.name = copy.name
        self.values = AnyArray(copy=copy.values)
        self.index = Index(copy=copy.index)

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

    def null_count(self) raises -> Int:
        """Returns the number of rows pandas would call missing.

        On a float column that is the cleared validity bits plus the NaNs, which
        is a scan of the values rather than a number the column already knows.
        `Array.null_count` is the other one and still counts bits and nothing
        else. The split is the line between the two halves of the library: an
        `Array` is Arrow and says what is in the buffers, a `Series` is pandas
        and says what pandas would say. See #170.

        Returns:
            The count of missing rows.

        Raises:
            Error: Only what the morsel runtime raises.
        """
        return missing_count_any(self.values)

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

    def is_string(self) -> Bool:
        """Reports whether the series holds text.

        Returns:
            True for a string column.
        """
        return self.values.is_string()

    def as_strings(self) raises -> StringArray:
        """Returns the data as a string column.

        Returns:
            A copy of the string column.

        Raises:
            If the series is not a string column.
        """
        return StringArray(copy=self.values.strings())

    def text(self, i: Int) raises -> String:
        """Returns one element as a string.

        This copies the element, and a null reads as the empty string, which is
        why it is here for tests and for printing rather than for anything that
        runs per row. `as_strings().unsafe_bytes(i)` is the one that does not
        copy.

        Args:
            i: The row.

        Returns:
            The element's bytes as a string.

        Raises:
            If the series is not a string column.
        """
        return self.values.strings()[i]

    def rename(self, name: String) raises -> Self:
        """Returns the same data under a different name.

        Args:
            name: The new name.

        Returns:
            A renamed copy.
        """
        return self._relabelled(name, AnyArray(copy=self.values))

    def cast(self, to: DType, strict: Bool = True) raises -> Self:
        """Returns the series converted to another dtype.

        No range check between numbers. Casting 300 to int8 gives 44, which is
        what the hardware gives and what `firepanda.kernel.cast` documents.
        pandas raises here and the decision to follow it or not belongs to the
        milestone that settles the error model, not to this one.

        Text is the exception, because a string is not a number until it is read
        as one and the reading can fail. Strict, which is the default, raises and
        names the row. Not strict writes a null.

        Args:
            to: The target dtype.
            strict: Whether a text value that is not a number raises.

        Returns:
            A series of dtype `to`, null in the same places.

        Raises:
            If either dtype has no physical layout, or the series is text and
            strict and some value is not a number.
        """
        return self._relabelled(self.name, cast_any(self.values, to, strict))

    def cast(self, to: LogicalType, strict: Bool = True) raises -> Self:
        """Returns the series converted to another logical type.

        This is the overload that can name text, text having no dtype of its own.
        `series.cast(LogicalType.STRING)` renders a number column as text and
        `series.cast(LogicalType.INT64)` reads a text one back.

        Args:
            to: The target type.
            strict: Whether a text value that is not a number raises.

        Returns:
            A series of type `to`, null in the same places.

        Raises:
            If the type has no conversion from this one, or the series is text
            and strict and some value is not a number.
        """
        return self._relabelled(self.name, cast_any(self.values, to, strict))

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
        var out = Self(self.name, take_any(self.values, indices))
        out.index = self.index.take(indices)
        return out^

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
        var out = Self(self.name, filter_any(self.values, mask))
        out.index = self.index.filter(mask)
        return out^

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
        var out = Self(self.name, self.values.slice(start, end))
        out.index = self.index.slice(start, end)
        return out^

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
        return self.slice(0, _head_end(n, len(self)))

    def tail(self, n: Int = 5) raises -> Self:
        """Returns the last rows.

        Args:
            n: How many rows, clamped as in `head`.

        Returns:
            A series of at most `n` rows.
        """
        var length = len(self)
        return self.slice(_tail_start(n, length), length)

    def argsort(
        self, descending: Bool = False, nulls_first: Bool = False
    ) raises -> Array[DType.int64]:
        """Returns the row order that sorts the series, as pandas spells it.

        The permutation the sort produces is uint32, because it is one value per
        row and the sort rewrites all of it on every pass, so the width of it is
        where a sort's memory goes. pandas returns int64 here, because numpy's
        `argsort` returns the platform index type and pandas uses a negative
        position as the sentinel for a row that does not exist. firepanda has no
        such sentinel, since a null is placed by `nulls_first` rather than
        removed and marked, so the sign is unused and the extra four bytes buy
        nothing internally.

        They are still worth paying for at this one boundary. Somebody who
        writes `s.argsort()` and hands the answer to something expecting a
        signed index has been given a difference they did not ask for and cannot
        see until it bites, and the cast is a single pass over a result the sort
        has already been over many times. So the kernel keeps uint32 and the
        method that carries the pandas name widens on the way out. Every
        internal caller, this file's `sort_values` included, goes to
        `argsort_any` and never pays it.

        Args:
            descending: Largest first.
            nulls_first: Put the nulls at the front rather than the back.

        Returns:
            A permutation of `[0, len(self))`, as int64.

        Raises:
            If the dtype is not sortable.
        """
        return _widen_positions(
            argsort_any(self.values, descending, nulls_first)
        )

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
        var order = argsort_any(self.values, descending, nulls_first)
        return self.take(_to_positions(order))

    def is_monotonic_increasing(self) raises -> Bool:
        """Reports whether the values never decrease, as pandas does.

        A series is one contiguous array with no room to remember an answer, so
        this scans every time it is asked. The frame's version of the same
        question caches, because a column there carries a sortedness flag that a
        sort has usually filled in already.

        A series holding a null is never monotonic, which is what pandas says.

        Returns:
            True if every value is at least the one before it.

        Raises:
            If the dtype is not one firepanda can order.
        """
        if self.null_count() > 0:
            return False
        return is_sorted_any(self.values, descending=False)

    def is_monotonic_decreasing(self) raises -> Bool:
        """Reports whether the values never increase, as pandas does.

        Returns:
            True if every value is at most the one before it.

        Raises:
            If the dtype is not one firepanda can order.
        """
        if self.null_count() > 0:
            return False
        return is_sorted_any(self.values, descending=True)

    def is_null(self) raises -> Array[DType.bool]:
        """Returns a mask that is true where a row is missing.

        The mask is a bare `Array` rather than a `Series`, because that is what
        `filter` takes and the round trip through a named column would be in the
        way of the thing this is almost always used for.

        Returns:
            A bool column, with no nulls of its own, as tall as the series.

        Raises:
            Error: Only what the morsel runtime raises.
        """
        return is_null_any(self.values)

    def is_not_null(self) raises -> Array[DType.bool]:
        """Returns a mask that is true where a row is present.

        Returns:
            A bool column, with no nulls of its own, as tall as the series.

        Raises:
            Error: Only what the morsel runtime raises.
        """
        return is_not_null_any(self.values)

    def drop_nulls(self) raises -> Self:
        """Returns the series with the missing rows removed.

        Returns:
            A series of `len(self) - null_count()` rows, in their original order.

        Raises:
            If the dtype has no physical layout.
        """
        return self.filter(self.is_not_null())

    def fill_null(self, other: Self) raises -> Self:
        """Returns the series with the missing rows taken from another.

        The fallback may be one row, which is used for every missing row. That
        is how filling with a scalar is spelled, and it is the same operation as
        filling from a column rather than a second one.

        Args:
            other: The fallback, of the same dtype and either as tall as the
                series or one row. Its name is ignored; the result keeps this
                series' name.

        Returns:
            A series null only where both were.

        Raises:
            If the dtypes differ, if the fallback is neither one row nor as tall
            as the series, or if the dtype has no physical layout.
        """
        return self._relabelled(
            self.name, coalesce_any(self.values, other.values)
        )

    def fill_forward(self, limit: Int = 0) raises -> Self:
        """Returns the series with each null taking the last present value before it.

        This is one of two methods on `Series` whose answer depends on the row
        order, the other being `fill_backward`. On a series that came out of a
        group by or a join, the order is the one this library promises for that
        operation, and on a series the caller built it is theirs.

        Args:
            limit: The longest run of nulls to fill, or zero for no limit.

        Returns:
            A series of the same height. Nulls before the first present value
            stay null.

        Raises:
            If the dtype has no physical layout.
        """
        return self._relabelled(self.name, fill_forward_any(self.values, limit))

    def fill_backward(self, limit: Int = 0) raises -> Self:
        """Returns the series with each null taking the next present value after it.

        Args:
            limit: The longest run of nulls to fill, or zero for no limit.

        Returns:
            A series of the same height. Nulls after the last present value stay
            null.

        Raises:
            If the dtype has no physical layout.
        """
        return self._relabelled(
            self.name, fill_backward_any(self.values, limit)
        )

    def _relabelled(self, name: String, var values: AnyArray) raises -> Self:
        """Builds a result that has this series' row labels.

        Every operation that answers one row per input row keeps the labels,
        because the labels are what the rows are called and nothing about a cast
        or a fill renames a row. The plain constructor cannot do this, since it
        takes a column with nothing attached and has to invent a range.

        Args:
            name: The result's name.
            values: The result's column. Must be as tall as this series.

        Returns:
            The series.

        Raises:
            Error: If the column is a different height, which would mean the
                labels do not describe it.
        """
        if len(values) != len(self.values):
            raise Error(
                "series: a result of "
                + String(len(values))
                + " rows cannot carry the labels of "
                + String(len(self.values))
            )
        var out = Self(name, values^)
        out.index = Index(copy=self.index)
        return out^

    def binary(
        self,
        other: Self,
        op: BinaryOp,
        fill_value: Optional[Value] = None,
        flip: Bool = False,
    ) raises -> Self:
        """Applies an operation to two series, matching rows by label.

        The alignment is in `firepanda/frame/align.mojo` and the short circuit
        for two equal indexes is the case that runs on almost every call.

        Args:
            other: The right operand.
            op: The operation.
            fill_value: What to put where exactly one of the two sides is
                missing. `None` leaves both sides as they are, which makes a row
                either side is missing from a missing row.
            flip: True for `other op self`, which is what the reflected forms
                need and which alignment cannot express by swapping the
                arguments, since the result keeps this series' name and labels.

        Returns:
            A series on the union of the two indexes.

        Raises:
            Error: If the labels differ and either index has a duplicate, or the
                operation is not defined on the two dtypes.
        """
        var pair = align_pair(
            self.index, self.values, other.index, other.values
        )
        var keep = Bitmap(0)
        if fill_value:
            keep = fill_one_sided(pair.left, pair.right, fill_value.value())

        var out = binary_any(pair.right, pair.left, op) if flip else binary_any(
            pair.left, pair.right, op
        )
        if fill_value:
            keep_rows(out, keep)

        var result = Self(_shared_name(self.name, other.name), out^)
        result.index = pair^.into_index()
        return result^

    def binary(
        self, value: Value, op: BinaryOp, value_on_left: Bool = False
    ) raises -> Self:
        """Applies an operation to every row of a series and one constant.

        Nothing aligns here, because a constant has no labels to align against.

        Args:
            value: The constant.
            op: The operation.
            value_on_left: True for `5 - s` rather than `s - 5`.

        Returns:
            A series of the same height, on the same labels.

        Raises:
            Error: If the operation is not defined on the two types.
        """
        return self._relabelled(
            self.name,
            binary_value_any(self.values, value, op, value_on_left),
        )

    def compare(self, other: Self, op: BinaryOp) raises -> Self:
        """Compares two series row by row, refusing to align them.

        This is what `==` and its five relatives do, and it is deliberately not
        what `+` does. pandas aligns an arithmetic operation and refuses to
        align a comparison, saying it can only compare identically labelled
        series, and the reason is that a comparison has no answer for a row that
        is on one side only. Arithmetic can say the answer is missing there. A
        comparison would have to say False, which reads as "these are different"
        rather than as "one of them is not here". The flexible forms `eq` and
        the rest do align, because they have a `fill_value` to say it with.

        Args:
            other: The right operand.
            op: The comparison.

        Returns:
            A boolean series on this series' labels.

        Raises:
            Error: If the two indexes hold different labels, or the comparison
                is not defined on the two dtypes.
        """
        if not self.index.equals(other.index):
            raise Error(
                "series: can only compare identically-labeled Series objects;"
                " use the eq, ne, lt, le, gt or ge method to compare these two"
                " on the union of their labels instead"
            )
        return self._relabelled(
            _shared_name(self.name, other.name),
            binary_any(self.values, other.values, op),
        )

    def add(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Adds two series, matching rows by label.

        Args:
            other: The right operand.
            fill_value: What to put where one side only is missing.

        Returns:
            The sums, on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned or added.
        """
        return self.binary(other, BinaryOp.ADD, fill_value)

    def radd(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Adds two series the other way round.

        Args:
            other: The left operand.
            fill_value: What to put where one side only is missing.

        Returns:
            The sums, on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned or added.
        """
        return self.binary(other, BinaryOp.ADD, fill_value, flip=True)

    def sub(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Subtracts one series from another, matching rows by label.

        Args:
            other: The right operand.
            fill_value: What to put where one side only is missing.

        Returns:
            The differences, on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned or subtracted.
        """
        return self.binary(other, BinaryOp.SUB, fill_value)

    def rsub(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Subtracts this series from another one.

        Args:
            other: The left operand.
            fill_value: What to put where one side only is missing.

        Returns:
            The differences, on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned or subtracted.
        """
        return self.binary(other, BinaryOp.SUB, fill_value, flip=True)

    def mul(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Multiplies two series, matching rows by label.

        Args:
            other: The right operand.
            fill_value: What to put where one side only is missing.

        Returns:
            The products, on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned or multiplied.
        """
        return self.binary(other, BinaryOp.MUL, fill_value)

    def rmul(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Multiplies two series the other way round.

        Args:
            other: The left operand.
            fill_value: What to put where one side only is missing.

        Returns:
            The products, on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned or multiplied.
        """
        return self.binary(other, BinaryOp.MUL, fill_value, flip=True)

    def truediv(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Divides two series, matching rows by label.

        Args:
            other: The right operand.
            fill_value: What to put where one side only is missing.

        Returns:
            A float64 series on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned or divided.
        """
        return self.binary(other, BinaryOp.DIV, fill_value)

    def rtruediv(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Divides two series the other way round.

        Args:
            other: The left operand.
            fill_value: What to put where one side only is missing.

        Returns:
            A float64 series on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned or divided.
        """
        return self.binary(other, BinaryOp.DIV, fill_value, flip=True)

    def floordiv(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Floor divides two series, matching rows by label.

        Args:
            other: The right operand.
            fill_value: What to put where one side only is missing.

        Returns:
            The quotients, on the union of the two indexes. An integer row whose
            divisor is zero is missing rather than infinite, which is the
            registered divergence explained in `firepanda/kernel/arith.mojo`.

        Raises:
            Error: If the operands cannot be aligned or divided.
        """
        return self.binary(other, BinaryOp.FLOORDIV, fill_value)

    def rfloordiv(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Floor divides two series the other way round.

        Args:
            other: The left operand.
            fill_value: What to put where one side only is missing.

        Returns:
            The quotients, on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned or divided.
        """
        return self.binary(other, BinaryOp.FLOORDIV, fill_value, flip=True)

    def mod(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Takes the remainder of two series, matching rows by label.

        Args:
            other: The right operand.
            fill_value: What to put where one side only is missing.

        Returns:
            The remainders, on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned or divided.
        """
        return self.binary(other, BinaryOp.MOD, fill_value)

    def rmod(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Takes the remainder the other way round.

        Args:
            other: The left operand.
            fill_value: What to put where one side only is missing.

        Returns:
            The remainders, on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned or divided.
        """
        return self.binary(other, BinaryOp.MOD, fill_value, flip=True)

    def pow(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Raises one series to the power of another, matching rows by label.

        Args:
            other: The exponents.
            fill_value: What to put where one side only is missing.

        Returns:
            The powers, on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned, or an integer column meets
                a negative exponent, which numpy refuses and so does this.
        """
        return self.binary(other, BinaryOp.POW, fill_value)

    def rpow(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Raises another series to the power of this one.

        Args:
            other: The bases.
            fill_value: What to put where one side only is missing.

        Returns:
            The powers, on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned, or an integer column meets
                a negative exponent.
        """
        return self.binary(other, BinaryOp.POW, fill_value, flip=True)

    def eq(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Compares two series for equality, matching rows by label.

        The flexible form aligns where the `==` operator refuses to, because
        this one has a `fill_value` to say what a row on one side only should be
        compared against.

        Args:
            other: The right operand.
            fill_value: What to compare against where one side only is missing.

        Returns:
            A boolean series on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned or compared.
        """
        return self.binary(other, BinaryOp.EQ, fill_value)

    def ne(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Compares two series for difference, matching rows by label.

        Args:
            other: The right operand.
            fill_value: What to compare against where one side only is missing.

        Returns:
            A boolean series on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned or compared.
        """
        return self.binary(other, BinaryOp.NE, fill_value)

    def lt(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Compares two series with less than, matching rows by label.

        Args:
            other: The right operand.
            fill_value: What to compare against where one side only is missing.

        Returns:
            A boolean series on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned or compared.
        """
        return self.binary(other, BinaryOp.LT, fill_value)

    def le(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Compares two series with less than or equal, matching rows by label.

        Args:
            other: The right operand.
            fill_value: What to compare against where one side only is missing.

        Returns:
            A boolean series on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned or compared.
        """
        return self.binary(other, BinaryOp.LE, fill_value)

    def gt(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Compares two series with greater than, matching rows by label.

        Args:
            other: The right operand.
            fill_value: What to compare against where one side only is missing.

        Returns:
            A boolean series on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned or compared.
        """
        return self.binary(other, BinaryOp.GT, fill_value)

    def ge(
        self, other: Self, fill_value: Optional[Value] = None
    ) raises -> Self:
        """Compares two series with greater or equal, matching rows by label.

        Args:
            other: The right operand.
            fill_value: What to compare against where one side only is missing.

        Returns:
            A boolean series on the union of the two indexes.

        Raises:
            Error: If the operands cannot be aligned or compared.
        """
        return self.binary(other, BinaryOp.GE, fill_value)

    def __add__(self, other: Self) raises -> Self:
        """Adds two series, matching rows by label.

        Args:
            other: The right operand.

        Returns:
            The sums.

        Raises:
            Error: If the operands cannot be aligned or added.
        """
        return self.binary(other, BinaryOp.ADD)

    def __add__(self, value: Value) raises -> Self:
        """Adds a constant to every row.

        Args:
            value: The constant.

        Returns:
            The sums.

        Raises:
            Error: If the two types cannot be added.
        """
        return self.binary(value, BinaryOp.ADD)

    def __radd__(self, value: Value) raises -> Self:
        """Adds every row to a constant.

        Args:
            value: The constant.

        Returns:
            The sums.

        Raises:
            Error: If the two types cannot be added.
        """
        return self.binary(value, BinaryOp.ADD, value_on_left=True)

    def __sub__(self, other: Self) raises -> Self:
        """Subtracts one series from another, matching rows by label.

        Args:
            other: The right operand.

        Returns:
            The differences.

        Raises:
            Error: If the operands cannot be aligned or subtracted.
        """
        return self.binary(other, BinaryOp.SUB)

    def __sub__(self, value: Value) raises -> Self:
        """Subtracts a constant from every row.

        Args:
            value: The constant.

        Returns:
            The differences.

        Raises:
            Error: If the two types cannot be subtracted.
        """
        return self.binary(value, BinaryOp.SUB)

    def __rsub__(self, value: Value) raises -> Self:
        """Subtracts every row from a constant.

        Args:
            value: The constant.

        Returns:
            The differences.

        Raises:
            Error: If the two types cannot be subtracted.
        """
        return self.binary(value, BinaryOp.SUB, value_on_left=True)

    def __mul__(self, other: Self) raises -> Self:
        """Multiplies two series, matching rows by label.

        Args:
            other: The right operand.

        Returns:
            The products.

        Raises:
            Error: If the operands cannot be aligned or multiplied.
        """
        return self.binary(other, BinaryOp.MUL)

    def __mul__(self, value: Value) raises -> Self:
        """Multiplies every row by a constant.

        Args:
            value: The constant.

        Returns:
            The products.

        Raises:
            Error: If the two types cannot be multiplied.
        """
        return self.binary(value, BinaryOp.MUL)

    def __rmul__(self, value: Value) raises -> Self:
        """Multiplies a constant by every row.

        Args:
            value: The constant.

        Returns:
            The products.

        Raises:
            Error: If the two types cannot be multiplied.
        """
        return self.binary(value, BinaryOp.MUL, value_on_left=True)

    def __truediv__(self, other: Self) raises -> Self:
        """Divides two series, matching rows by label.

        Args:
            other: The divisors.

        Returns:
            A float64 series.

        Raises:
            Error: If the operands cannot be aligned or divided.
        """
        return self.binary(other, BinaryOp.DIV)

    def __truediv__(self, value: Value) raises -> Self:
        """Divides every row by a constant.

        Args:
            value: The divisor.

        Returns:
            A float64 series.

        Raises:
            Error: If the two types cannot be divided.
        """
        return self.binary(value, BinaryOp.DIV)

    def __rtruediv__(self, value: Value) raises -> Self:
        """Divides a constant by every row.

        Args:
            value: The numerator.

        Returns:
            A float64 series.

        Raises:
            Error: If the two types cannot be divided.
        """
        return self.binary(value, BinaryOp.DIV, value_on_left=True)

    def __floordiv__(self, other: Self) raises -> Self:
        """Floor divides two series, matching rows by label.

        Args:
            other: The divisors.

        Returns:
            The quotients, keeping the operand type.

        Raises:
            Error: If the operands cannot be aligned or divided.
        """
        return self.binary(other, BinaryOp.FLOORDIV)

    def __floordiv__(self, value: Value) raises -> Self:
        """Floor divides every row by a constant.

        Args:
            value: The divisor.

        Returns:
            The quotients, keeping the operand type.

        Raises:
            Error: If the two types cannot be divided.
        """
        return self.binary(value, BinaryOp.FLOORDIV)

    def __rfloordiv__(self, value: Value) raises -> Self:
        """Floor divides a constant by every row.

        Args:
            value: The numerator.

        Returns:
            The quotients, keeping the operand type.

        Raises:
            Error: If the two types cannot be divided.
        """
        return self.binary(value, BinaryOp.FLOORDIV, value_on_left=True)

    def __mod__(self, other: Self) raises -> Self:
        """Takes the remainder of two series, matching rows by label.

        Args:
            other: The divisors.

        Returns:
            The remainders, keeping the operand type.

        Raises:
            Error: If the operands cannot be aligned or divided.
        """
        return self.binary(other, BinaryOp.MOD)

    def __mod__(self, value: Value) raises -> Self:
        """Takes the remainder of every row by a constant.

        Args:
            value: The divisor.

        Returns:
            The remainders, keeping the operand type.

        Raises:
            Error: If the two types cannot be divided.
        """
        return self.binary(value, BinaryOp.MOD)

    def __rmod__(self, value: Value) raises -> Self:
        """Takes the remainder of a constant by every row.

        Args:
            value: The numerator.

        Returns:
            The remainders, keeping the operand type.

        Raises:
            Error: If the two types cannot be divided.
        """
        return self.binary(value, BinaryOp.MOD, value_on_left=True)

    def __pow__(self, other: Self) raises -> Self:
        """Raises every row to the matching row of another series.

        Args:
            other: The exponents.

        Returns:
            The powers.

        Raises:
            Error: If the operands cannot be aligned, or an integer column meets
                a negative exponent.
        """
        return self.binary(other, BinaryOp.POW)

    def __pow__(self, value: Value) raises -> Self:
        """Raises every row to a constant power.

        Args:
            value: The exponent.

        Returns:
            The powers.

        Raises:
            Error: If an integer column meets a negative exponent.
        """
        return self.binary(value, BinaryOp.POW)

    def __rpow__(self, value: Value) raises -> Self:
        """Raises a constant to the power of every row.

        Args:
            value: The base.

        Returns:
            The powers.

        Raises:
            Error: If an integer base meets a negative exponent.
        """
        return self.binary(value, BinaryOp.POW, value_on_left=True)

    def __eq__(self, other: Self) raises -> Self:
        """Compares two identically labelled series for equality.

        Args:
            other: The right operand.

        Returns:
            A boolean series.

        Raises:
            Error: If the labels differ, which `eq` allows and this does not.
        """
        return self.compare(other, BinaryOp.EQ)

    def __eq__(self, value: Value) raises -> Self:
        """Compares every row against a constant.

        Args:
            value: The constant.

        Returns:
            A boolean series.

        Raises:
            Error: If the two types cannot be compared.
        """
        return self.binary(value, BinaryOp.EQ)

    def __ne__(self, other: Self) raises -> Self:
        """Compares two identically labelled series for difference.

        Args:
            other: The right operand.

        Returns:
            A boolean series.

        Raises:
            Error: If the labels differ.
        """
        return self.compare(other, BinaryOp.NE)

    def __ne__(self, value: Value) raises -> Self:
        """Compares every row against a constant for difference.

        Args:
            value: The constant.

        Returns:
            A boolean series.

        Raises:
            Error: If the two types cannot be compared.
        """
        return self.binary(value, BinaryOp.NE)

    def __lt__(self, other: Self) raises -> Self:
        """Compares two identically labelled series with less than.

        Args:
            other: The right operand.

        Returns:
            A boolean series.

        Raises:
            Error: If the labels differ.
        """
        return self.compare(other, BinaryOp.LT)

    def __lt__(self, value: Value) raises -> Self:
        """Compares every row against a constant with less than.

        Args:
            value: The constant.

        Returns:
            A boolean series.

        Raises:
            Error: If the two types cannot be compared.
        """
        return self.binary(value, BinaryOp.LT)

    def __le__(self, other: Self) raises -> Self:
        """Compares two identically labelled series with less or equal.

        Args:
            other: The right operand.

        Returns:
            A boolean series.

        Raises:
            Error: If the labels differ.
        """
        return self.compare(other, BinaryOp.LE)

    def __le__(self, value: Value) raises -> Self:
        """Compares every row against a constant with less or equal.

        Args:
            value: The constant.

        Returns:
            A boolean series.

        Raises:
            Error: If the two types cannot be compared.
        """
        return self.binary(value, BinaryOp.LE)

    def __gt__(self, other: Self) raises -> Self:
        """Compares two identically labelled series with greater than.

        Args:
            other: The right operand.

        Returns:
            A boolean series.

        Raises:
            Error: If the labels differ.
        """
        return self.compare(other, BinaryOp.GT)

    def __gt__(self, value: Value) raises -> Self:
        """Compares every row against a constant with greater than.

        Args:
            value: The constant.

        Returns:
            A boolean series.

        Raises:
            Error: If the two types cannot be compared.
        """
        return self.binary(value, BinaryOp.GT)

    def __ge__(self, other: Self) raises -> Self:
        """Compares two identically labelled series with greater or equal.

        Args:
            other: The right operand.

        Returns:
            A boolean series.

        Raises:
            Error: If the labels differ.
        """
        return self.compare(other, BinaryOp.GE)

    def __ge__(self, value: Value) raises -> Self:
        """Compares every row against a constant with greater or equal.

        Args:
            value: The constant.

        Returns:
            A boolean series.

        Raises:
            Error: If the two types cannot be compared.
        """
        return self.binary(value, BinaryOp.GE)

    def unary(self, op: UnaryOp) raises -> Self:
        """Puts every row through a one operand operation.

        Nothing aligns here, because there is only one operand and it brings its
        own labels, so this is the kernel and a relabel and nothing else.

        Args:
            op: Which operation.

        Returns:
            A series of the same type, under the same labels and the same name.

        Raises:
            Error: If the operation has no meaning on this type, which is `-`
                and `abs` on a text column and `~` on a float one.
        """
        return self._relabelled(self.name, unary_any(self.values, op))

    def __neg__(self) raises -> Self:
        """Flips the sign of every row.

        On a bool column this is the logical not rather than an arithmetic
        negation, which is pandas' answer and not numpy's. numpy refuses the
        expression outright.

        Returns:
            The negated series.

        Raises:
            Error: If the type has no negation.
        """
        return self.unary(UnaryOp.NEG)

    def __pos__(self) raises -> Self:
        """Answers the series unchanged.

        pandas has this and it does nothing on every type that accepts it at
        all, so it is a copy rather than a pass over the values.

        Returns:
            A copy of the series.

        Raises:
            Error: If the type has no unary plus.
        """
        return self.unary(UnaryOp.POS)

    def abs(self) raises -> Self:
        """Takes the absolute value of every row.

        pandas spells this both ways, `abs(s)` and `s.abs()`, and Mojo only has
        the second. The builtin `abs` needs the `Absable` trait, and a trait
        cannot be conformed to by a method that raises, so `abs(s)` does not
        compile here even though `s.__abs__()` does. The Python surface gets
        both spellings, since the dunder is still there for it to bind to.

        The most negative integer answers itself, which is what the hardware
        does and what numpy and pandas both report. A guard would make firepanda
        the only one of the three that answers something else.

        Returns:
            The magnitudes.

        Raises:
            Error: If the type has no absolute value.
        """
        return self.unary(UnaryOp.ABS)

    def __abs__(self) raises -> Self:
        """Takes the absolute value of every row.

        Here for the Python binding to reach, since `abs(s)` in Python looks
        this up. Mojo callers want `abs` above, for the reason written there.

        Returns:
            The magnitudes.

        Raises:
            Error: If the type has no absolute value.
        """
        return self.abs()

    def __invert__(self) raises -> Self:
        """Inverts every row.

        Two operations picked by the type: the logical not on a bool column, so
        that a mask can be negated, and the bitwise not on an integer one, so
        that `~Series([1, 2])` is `[-2, -3]`.

        Returns:
            The inverted series.

        Raises:
            Error: On a float column, which has no bitwise not.
        """
        return self.unary(UnaryOp.INVERT)

    def write_to(self, mut writer: Some[Writer]):
        """Writes the values, one per line, with a dtype footer.

        The default limits apply. `render_column` in `firepanda.frame.display`
        takes a `DisplayOptions` if a caller wants more or fewer rows than the
        ten this prints.

        Args:
            writer: The sink.
        """
        writer.write(render_column(self.name, self.values, DisplayOptions()))


def _shared_name(a: String, b: String) -> String:
    """The name a result of two series should carry.

    pandas keeps the name when both operands agree on it and drops it when they
    do not, on the reasoning that a column called `price` plus a column called
    `tax` is neither of those things. An unnamed series is the empty string
    here, so two differently named operands and two unnamed ones land on the
    same answer, which is what pandas does with `None` as well.

    Args:
        a: The left name.
        b: The right name.

    Returns:
        The shared name, or the empty string if they differ.
    """
    return a if a == b else String()


def _head_end(n: Int, length: Int) -> Int:
    """Where `head(n)` stops.

    A negative `n` means all but the last `n` of them, which is the pandas answer
    and the Python slicing one, `s[:-2]`. It used to mean nothing at all here: the
    count was clamped to zero and `head(-2)` returned an empty frame, which is not a
    reading anybody would defend, it is a bound written for a positive number and
    then handed a negative one.

    Args:
        n: How many rows, or how many to leave off the end when negative.
        length: The height.

    Returns:
        The exclusive end of the range, inside `[0, length]`.
    """
    if n < 0:
        return length + n if length + n > 0 else 0
    return n if n < length else length


def _tail_start(n: Int, length: Int) -> Int:
    """Where `tail(n)` starts.

    A negative `n` means all but the first `n` of them, `s[2:]`, which is the mirror
    of `_head_end` and is also what pandas does.

    Args:
        n: How many rows, or how many to skip from the front when negative.
        length: The height.

    Returns:
        The inclusive start of the range, inside `[0, length]`.
    """
    if n < 0:
        return -n if -n < length else length
    return length - n if n < length else 0


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


def _widen_positions(order: Array[DType.uint32]) -> Array[DType.int64]:
    """Widens a permutation into the signed column pandas hands back.

    A permutation holds no nulls, so the validity bitmap the new array is born
    with is already right and nothing here touches it.
    """
    var n = len(order)
    var out = Array[DType.int64](overwritten=n)
    var source = order.unsafe_ptr()
    var target = out.unsafe_ptr()
    for i in range(n):
        target.unsafe_offset(i).unsafe_store(
            Int64(source.unsafe_offset(i).unsafe_load())
        )
    return out^


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
