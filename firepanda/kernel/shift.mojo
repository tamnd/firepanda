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

The block for the gap is `filled_block` in `binary.mojo`, next to the `all_null`
it is the filled half of. It was written here and moved there when `reindex`
turned out to want the same thing, which is a block of one value repeated in a
type the value did not come with.
"""

from firepanda.array.any import AnyArray
from firepanda.array.value import Value
from firepanda.kernel.dictionary import with_categories

from .binary import filled_block
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
    # The gap block is built from the type alone, and for a category column the
    # type does not carry the categories, so every path below produces codes
    # with nothing behind them until the list is put back on. The gap itself is
    # missing rather than in any category, which is why the block needs no
    # categories of its own and the source's list is the whole answer.
    if gap >= rows:
        return with_categories(filled_block(col.type, rows, fill), col)
    if periods > 0:
        return with_categories(
            concat_two_any(
                filled_block(col.type, gap, fill), col.slice(0, rows - gap)
            ),
            col,
        )
    return with_categories(
        concat_two_any(col.slice(gap, rows), filled_block(col.type, gap, fill)),
        col,
    )
