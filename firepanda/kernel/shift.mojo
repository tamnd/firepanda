"""Moving a column along its own rows.

Shifting is the one operation in this package that has no opinion about values at
all. Row `i` of the answer is row `i - periods` of the input, every row that
reaches off the end of the input is missing, and nothing is read, compared or
computed on the way. That makes it a copy with a gap at one end, which is why it
is written as one here rather than as a loop: a block for the gap, a slice for
the overlap, and a concat to put them together. Each of those three already
exists, each is already parallel, and the slice is a byte move rather than a
gather, so the whole operation costs one pass and allocates nothing per row.

The alternative spelling is a `take` with an index array, and it would be
correct. It would also allocate eight bytes a row for indices nobody looks at
twice and turn a memory copy into a gather, which on a tall column is the
difference between reading the input once in order and reading it once out of
order. Shift is the operation people put inside a loop over a hundred lags.

Like the fills in `nulls.mojo` and unlike everything else here, this is
order-dependent by construction. A shift of a column whose rows are in no
particular order is a shift of nothing in particular, which is a fact about what
the caller asked for and not something this can check.

The fill is a `Value` rather than a flag plus a value, so a shift with nothing to
put in the gap and a shift with something to put in the gap are one function with
one loop. A null `Value` means the gap stays missing, which is what pandas does
when `fill_value` is not passed, and it is also the only case where the answer's
type can differ from the input's. See `Series.shift` for that part; it is a
pandas rule and it does not belong down here.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder
from firepanda.array.value import Value
from firepanda.dtype.lists import ALL
from firepanda.dtype.logical import LogicalType

from .binary import all_null
from .concat import concat_two_any


def shift_any(col: AnyArray, periods: Int) raises -> AnyArray:
    """Moves a column's rows along, leaving the gap missing.

    Args:
        col: The column.
        periods: How far to move. Positive moves rows towards the end, so the
            gap is at the start, and negative moves them towards the start.

    Returns:
        A column of the same type and length, with `periods` rows missing at one
        end.

    Raises:
        Error: If the column's type has no physical layout.
    """
    return shift_any(col, periods, Value(null=col.type))


def shift_any(col: AnyArray, periods: Int, fill: Value) raises -> AnyArray:
    """Moves a column's rows along, putting something in the gap.

    The three shapes are the whole function. A shift of nothing is a copy. A
    shift further than the column is tall is a column of nothing but gap, which
    is worth its own line rather than being left to a slice of negative length.
    Anything else is a gap and an overlap in one order or the other.

    Args:
        col: The column.
        periods: How far to move. Positive moves rows towards the end, so the
            gap is at the start, and negative moves them towards the start.
        fill: What to put in the gap. A null value leaves it missing.

    Returns:
        A column of the same type and length.

    Raises:
        Error: If the column's type has no physical layout, or the fill value
            cannot be read as that type.
    """
    var rows = len(col)
    if periods == 0:
        return AnyArray(copy=col)
    var gap = periods if periods > 0 else -periods
    if gap >= rows:
        return _gap_block(col.type, rows, fill)
    if periods > 0:
        return concat_two_any(
            _gap_block(col.type, gap, fill), col.slice(0, rows - gap)
        )
    return concat_two_any(col.slice(gap, rows), _gap_block(col.type, gap, fill))


def _gap_block(type: LogicalType, rows: Int, fill: Value) raises -> AnyArray:
    """Builds the run of rows the shift has nothing to put in.

    Four cases and they are the product of two questions, which is whether the
    gap is missing or filled and whether the column holds bytes or numbers. The
    missing fixed width case is the one that costs nothing, because a null holds
    a zero everywhere in this package and the values buffer starts zeroed, so
    `all_null` only has to install a bitmap.

    Args:
        type: The column's type. The block takes it, so a shifted timestamp is
            still a timestamp and the concat below does not refuse the pair.
        rows: How many rows of gap there are.
        fill: What to put in them. A null value leaves them missing.

    Returns:
        A column of `rows` rows, all of them the same.

    Raises:
        Error: If the type has no physical layout, or the fill value cannot be
            read as that type.
    """
    if type.is_variable_width():
        var builder = StringBuilder(capacity=rows)
        if fill.is_null():
            for _ in range(rows):
                builder.append_null()
        else:
            var text = fill.as_string()
            for _ in range(rows):
                builder.append(text.as_bytes())
        return AnyArray(builder^.finish())

    if fill.is_null():
        return all_null(type, rows)

    comptime for candidate in ALL:
        if type.physical == candidate:
            var out = Array[candidate](rows)
            var one = fill.as_scalar[candidate]()
            for i in range(rows):
                out[i] = one
            return AnyArray(out^.into_data(), type)
    raise Error("shift: unsupported dtype " + String(type))
