"""Running totals, products and extremes down a column.

Row `i` of the answer is the whole of the column up to and including row `i`,
folded with one operator. That makes every one of these a scan, so row `i`
depends on row `i - 1` and the naive loop has a dependency chain as long as the
column. It does not follow that there is nothing for the vector unit to do, and
this is where these differ from the fills in `nulls.mojo`. A fill carries a value
and has no arithmetic at all, so the only thing a vector can do for it is skip.
A running total has arithmetic, and a prefix scan over a register is a known
trick: fold the register against itself shifted by one lane, then by two, then by
four, and after log2(width) steps every lane holds the prefix of the lanes at or
below it. Add the carry from the block before, store, take the last lane as the
next carry. The chain is then one step per block rather than one step per row.

The ladder buys that speed by regrouping the folds, and regrouping is only free
when the operator is associative. It is free on every integer width, because
wrapping arithmetic is associative, and it is free for a running maximum or
minimum on any dtype, because those select an input rather than computing a new
number. It is not free for a running total or product of a float column, where
the two groupings round in different places and the answer comes out a bit or
two from the one pandas gives. Those two cases are summed one row at a time on
purpose. `_reassociates` is the one line that decides it and it is the only
place in this file where the dtype changes what the loop does.

## What a missing row does to a running total

pandas skips it in the total and emits it in place, so `[1, None, 3]` sums to
`[1, nan, 4]`. The gap does not restart the total and it does not poison it.

Writing that as a branch per row would undo the paragraph above, so it is not
written that way. A missing lane is replaced by the operator's identity before
the scan, which is zero for a sum, one for a product, the lowest value of the
dtype for a running maximum and the highest for a running minimum. An identity
folded into a total leaves the total alone, which is what identity means, so the
scan then computes the right answer with no knowledge that anything was missing.
The output rows are blanked afterwards from the same bitmap the lanes were
selected with.

That leaves one thing worth being explicit about, because it looks like a bug
and is not. A NaN that arrives in the column is missing and is skipped. A NaN
that the arithmetic produces is a value and is carried, so a running total that
adds positive infinity to negative infinity is NaN from there to the end of the
column even though every row after it is present. pandas does exactly this and
the conformance corpus has a frame that proves it.

## The types

Measured against pandas 3.0.3. A running maximum or minimum answers the column's
own type, every one of them, including bool and including the temporal types.
A running sum or product widens the way a whole column sum does, so every signed
integer width answers int64, every unsigned one answers uint64, and bool answers
int64. Floats are the exception and do not widen: a float32 column has a float32
running total, where `Series.sum` over the same column accumulates in float64.
That is not an inconsistency to fix. A reduction produces one number and can
afford a wider accumulator; a scan produces a column and widening it would double
the answer.

A duration has a running sum and a running extreme and has no running product,
because a product of two elapsed times is an area and pandas refuses it. An
instant has running extremes only, for the reason `unary.mojo` gives: there is no
point in time that is the sum of two other points in time.
"""

from std.math import iota
from std.sys.info import simd_width_of

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL
from firepanda.dtype.logical import LogicalType, TypeKind

from .accum import highest, lowest
from .cast import cast_any
from .nulls import present_bitmap, present_bitmap_any

comptime OP_CUMSUM = 0
"""Operation code for the running total."""

comptime OP_CUMPROD = 1
"""Operation code for the running product."""

comptime OP_CUMMAX = 2
"""Operation code for the running maximum."""

comptime OP_CUMMIN = 3
"""Operation code for the running minimum."""


@fieldwise_init
struct CumulativeOp(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Which fold a column is being run through.

    Held as a code for the same reason `UnaryOp` is, which is that the erased
    entry point takes it as an ordinary argument while the typed loop takes it
    as a parameter the compiler folds away.
    """

    var code: Int
    """The operation, as one of the four values below."""

    comptime SUM = Self(OP_CUMSUM)
    """The running total."""

    comptime PROD = Self(OP_CUMPROD)
    """The running product."""

    comptime MAX = Self(OP_CUMMAX)
    """The running maximum."""

    comptime MIN = Self(OP_CUMMIN)
    """The running minimum."""

    def __eq__(self, other: Self) -> Bool:
        """Compares two operations.

        Args:
            other: The operation to compare against.

        Returns:
            True if they are the same operation.
        """
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        """Compares two operations for difference.

        Args:
            other: The operation to compare against.

        Returns:
            True if they are different operations.
        """
        return self.code != other.code

    def write_to(self, mut writer: Some[Writer]):
        """Writes the operation as the pandas method it answers.

        Args:
            writer: Where to write.
        """
        if self == Self.SUM:
            writer.write("cumsum")
        elif self == Self.PROD:
            writer.write("cumprod")
        elif self == Self.MAX:
            writer.write("cummax")
        else:
            writer.write("cummin")


def cumulative_type(op: CumulativeOp, t: LogicalType) raises -> LogicalType:
    """Returns the type a running fold answers, without touching a value.

    The whole rule is at the top of the file. This is the place a plan finds out
    that a running product of a column of instants has no answer before it starts
    reading rows.

    Args:
        op: The operation.
        t: The column's type.

    Returns:
        The type of the answer.

    Raises:
        Error: If the operation has no answer on that type.
    """
    var extreme = op == CumulativeOp.MAX or op == CumulativeOp.MIN
    if t.is_variable_width() or t.kind == TypeKind.NULL:
        raise Error(
            "cumulative: " + String(op) + " is not defined on " + String(t)
        )
    if extreme:
        if t.is_numeric() or t == LogicalType.BOOL or t.is_temporal():
            return t
        raise Error(
            "cumulative: " + String(op) + " is not defined on " + String(t)
        )

    if t.kind == TypeKind.DURATION:
        if op == CumulativeOp.PROD:
            raise Error(
                "cumulative: a product of two elapsed times is not an elapsed"
                " time, so cumprod is not defined on "
                + String(t)
            )
        return t
    if t.is_temporal():
        raise Error(
            "cumulative: there is no instant that is the sum of two"
            " instants, so "
            + String(op)
            + " is not defined on "
            + String(t)
        )
    if t == LogicalType.BOOL:
        return LogicalType.INT64
    if not t.is_numeric():
        raise Error(
            "cumulative: " + String(op) + " is not defined on " + String(t)
        )
    if t.is_float():
        return t
    if t.physical.is_signed():
        return LogicalType.INT64
    return LogicalType.UINT64


def cumulative_any(col: AnyArray, op: CumulativeOp) raises -> AnyArray:
    """Runs a fold down a column whose dtype is a runtime value.

    The cast to the answer's width happens before the scan rather than inside it,
    so the loop has one dtype to think about and the widening is a pass the cast
    kernel already knows how to vectorise. On the common columns, where the
    answer's width is the column's width, there is no cast and no pass.

    Args:
        col: The column.
        op: Which fold to run.

    Returns:
        A column of the same height, missing exactly where the input was, and of
        the type `cumulative_type` gives.

    Raises:
        Error: If the operation has no answer on the column's type, or the dtype
            has no physical layout.
    """
    var answer = cumulative_type(op, col.type)
    var present = present_bitmap_any(col)
    var wide = AnyArray(
        copy=col
    ) if col.dtype() == answer.physical else cast_any(col, answer.physical)

    comptime for candidate in ALL:
        if answer.physical == candidate:
            var out = _fold[candidate](
                wide.unsafe_ptr[candidate](), present, len(col), op.code
            )
            return AnyArray(out^.into_data(), answer)
    raise Error("cumulative: unsupported dtype " + String(answer.physical))


def cumulative[dt: DType, //, code: Int](col: Array[dt]) raises -> Array[dt]:
    """Runs a fold down a column whose dtype the caller already knows.

    There is no widening here and no type rule at all, so it folds an int8 column
    into an int8 column and wraps on the way. `cumulative_any` is the entry point
    that answers pandas and this is the one that answers the column, and the fuzz
    harness wants the second so that it can run the scan at every dtype rather
    than at the two the widening leaves.

    Args:
        col: The column.

    Parameters:
        dt: The dtype.
        code: The operation, as one of the four codes above.

    Returns:
        A column of the same height and the same dtype.

    Raises:
        Error: Only what allocation raises.
    """
    return _scan[code=code](col.unsafe_ptr(), present_bitmap(col), len(col))


def _fold[
    dt: DType, origin: ImmOrigin
](
    src: Pointer[Scalar[dt], origin], present: Bitmap, rows: Int, code: Int
) raises -> Array[dt]:
    """Sends the scan to the loop written for one operation.

    The operation arrives as a runtime code and leaves as a parameter, which is
    what lets the identity, the fold and the whole unrolled shift ladder be
    constants inside the loop rather than a switch per block.

    Args:
        src: The values, already at the answer's width.
        present: Which rows hold a value.
        rows: How many rows.
        code: The operation.

    Parameters:
        dt: The answer's dtype.
        origin: The origin of the values.

    Returns:
        The folded column.

    Raises:
        Error: If the code is not one of the four.
    """
    comptime if dt != DType.bool:
        if code == OP_CUMSUM:
            return _scan[code=OP_CUMSUM](src, present, rows)
        if code == OP_CUMPROD:
            return _scan[code=OP_CUMPROD](src, present, rows)
    if code == OP_CUMMAX:
        return _scan[code=OP_CUMMAX](src, present, rows)
    if code == OP_CUMMIN:
        return _scan[code=OP_CUMMIN](src, present, rows)
    raise Error("cumulative: unknown operation " + String(code))


def _identity[dt: DType, code: Int]() -> Scalar[dt]:
    """Returns the value that folds into a total without changing it.

    Args:

    Parameters:
        dt: The dtype.
        code: The operation.

    Returns:
        Zero for a sum, one for a product, the lowest value of the dtype for a
        running maximum and the highest for a running minimum.
    """
    comptime if code == OP_CUMSUM:
        return Scalar[dt](0)
    elif code == OP_CUMPROD:
        return Scalar[dt](1)
    elif code == OP_CUMMAX:
        return lowest[dt]()
    else:
        return highest[dt]()


def _blank[dt: DType]() -> Scalar[dt]:
    """Returns what a row with nothing in it holds in the values buffer.

    A NaN on a float dtype, because that is the only missing pandas has there,
    and a zero on every other dtype, because a null holds a zero everywhere in
    this package. The same rule the fills take, for the same reason. See #170.

    Args:

    Parameters:
        dt: The dtype.

    Returns:
        The blank.
    """
    comptime if dt.is_floating_point():
        return Scalar[dt](0) / Scalar[dt](0)
    else:
        return Scalar[dt](0)


def _fold_pair[
    dt: DType, width: SIMDLength, //, code: Int
](a: SIMD[dt, width], b: SIMD[dt, width]) -> SIMD[dt, width]:
    """Folds two vectors of running values together lanewise.

    A bool column reaches this for a running extreme and never for a running
    total, because a running total of a bool column is a running total of an
    int64 column by the time it gets here. The extremes on bool are the two
    logical operators, since a maximum over true and false is whether either was
    true. `_fold` is what makes the other two unreachable rather than wrong.

    Args:
        a: The left operand.
        b: The right operand.

    Parameters:
        dt: The dtype.
        width: How many lanes.
        code: The operation.

    Returns:
        The fold.
    """
    comptime if dt == DType.bool:
        comptime if code == OP_CUMMAX:
            return a | b
        else:
            return a & b
    elif code == OP_CUMSUM:
        return a + b
    elif code == OP_CUMPROD:
        return a * b
    elif code == OP_CUMMAX:
        return max(a, b)
    else:
        return min(a, b)


def _reassociates[dt: DType, code: Int]() -> Bool:
    """Says whether the ladder gives the same answer as the loop it replaces.

    The ladder folds row three as rows two and three folded together, then
    folded with rows zero and one folded together. The loop folds row three as
    rows zero through two folded and then row three. Those are the same answer
    when the operator is associative and they are not the same answer otherwise,
    and a floating point addition is not associative: the two groupings round at
    different places and differ in the last bit or two.

    Integer addition and multiplication wrap, and wrapping is associative, so the
    ladder is exact on every integer width and on bool. A running maximum or
    minimum is associative on every dtype including float, because it selects one
    of its inputs rather than computing a new number, and a NaN never reaches it
    since an arriving NaN is missing and a maximum cannot produce one.

    That leaves a running total or product of a float column as the only pair
    that has to be summed in the order pandas sums it, and it is summed one row
    at a time. Reassociating it would answer within an ulp or so of pandas rather
    than answering pandas, and the whole point of this library is the second one.

    Args:

    Parameters:
        dt: The dtype.
        code: The operation.

    Returns:
        True when the block ladder may be used.
    """
    comptime if not dt.is_floating_point():
        return True
    return code == OP_CUMMAX or code == OP_CUMMIN


def _prefix[
    dt: DType, width: SIMDLength, //, code: Int
](var v: SIMD[dt, width]) -> SIMD[dt, width]:
    """Turns a register of values into a register of running values.

    The ladder is the standard one. After folding against a shift of one lane,
    every lane holds two rows folded. After a shift of two, four rows. After
    log2(width) steps, lane `i` holds rows zero through `i`, which is the prefix.
    The lanes a shift brings in from below the register are filled with the
    identity, so the low lanes fold against something that leaves them alone.

    Args:
        v: The values.

    Parameters:
        dt: The dtype.
        width: How many lanes.
        code: The operation.

    Returns:
        The running values.
    """
    comptime one = SIMD[dt, width](_identity[dt, code]())
    comptime lane = iota[DType.int32, width]()
    comptime for k in range(6):
        comptime step = 1 << k
        comptime if step < Int(width):
            var reach = lane.ge(SIMD[DType.int32, width](step))
            v = _fold_pair[code](v, reach.select(v.shift_right[step](), one))
    return v


def _scan[
    dt: DType, //, code: Int, origin: ImmOrigin
](src: Pointer[Scalar[dt], origin], present: Bitmap, rows: Int) raises -> Array[
    dt
]:
    """The scan itself, over pointers and bitmaps rather than columns.

    One block at a time, and every block does the same four things with no branch
    on the data in any of them. It reads the block's bits out of the validity
    word it lies inside, which it always does lie inside because the block is a
    power of two no wider than sixty four and starts at a multiple of its own
    width. It replaces the missing lanes with the identity. It runs the shift
    ladder and folds in the carry from the block before. And it writes the block
    back with the missing rows blanked.

    Args:
        src: The values, already at the answer's width.
        present: Which rows hold a value.
        rows: How many rows.

    Parameters:
        dt: The dtype.
        code: The operation.
        origin: The origin of the values.

    Returns:
        A column of running values, missing exactly where the input was.

    Raises:
        Error: Only what allocation raises.
    """
    comptime width = simd_width_of[dt]()
    comptime one = _identity[dt, code]()
    comptime ones = SIMD[dt, width](one)
    comptime blanks = SIMD[dt, width](_blank[dt]())
    comptime bit = iota[DType.uint64, width]()
    comptime full = (
        UInt64(1) << UInt64(width)
    ) - 1 if width < 64 else UInt64.MAX

    var out = Array[dt](rows)
    var target = out.unsafe_mut_ptr()
    var carry = one

    var i = 0
    comptime if _reassociates[dt, code]():
        while i + width <= rows:
            var bits = (present.unsafe_word(i // 64) >> UInt64(i % 64)) & full
            var here = ((SIMD[DType.uint64, width](bits) >> bit) & 1).ne(0)
            var block = _prefix[code](
                here.select(
                    src.unsafe_offset(i).unsafe_load[width=width](), ones
                )
            )
            block = _fold_pair[code](SIMD[dt, width](carry), block)
            carry = block[width - 1]
            target.unsafe_offset(i).unsafe_store(here.select(block, blanks))
            i += width

    while i < rows:
        if present.get(i):
            carry = _fold_pair[code](carry, src.unsafe_offset(i).unsafe_load())
            target.unsafe_offset(i).unsafe_store(carry)
        else:
            target.unsafe_offset(i).unsafe_store(_blank[dt]())
        i += 1

    comptime if not dt.is_floating_point():
        out.data.validity = Bitmap(copy=present)
    return out^
