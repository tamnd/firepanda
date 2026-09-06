"""Putting two labelled columns onto one set of row labels.

This is what makes `a + b` in pandas a different operation from adding two
arrays. The rows are matched by label and not by position, so two series whose
labels are `abc` and `bcd` add to four rows, the two they share holding sums and
the two they do not holding nothing. Nobody writes that on purpose very often.
Everybody gets it by accident the first time they add two columns that came out
of different filters, which is why the rule has to be here rather than in the
caller.

The short circuit is the part that matters for speed. Two frames that came out of
the same file have the same labels, and `Index.equals` answers that by comparing
two integers when both sides are ranges. So the common case pays a comparison and
nothing else: no union, no lookup, no gather, and the two columns go to the kernel
exactly as they arrived.

When the labels do differ, the work is a union of the two indexes and one
`get_indexer` per side against it, which is the pair of primitives #227 and #229
built. A label that is on one side only comes back as `NOT_FOUND`, and `take_any`
turns a negative position into a null row, so the missing side needs no second
pass to be told it is missing.

Duplicated labels are refused for now rather than guessed at, and what pandas
does with them is worth writing down because it is not what the name "align"
suggests. Two indexes that are equal align positionally however many times a
label appears in them, which is the case this file handles. Two that differ take
a cartesian product per label: a series labelled `a a a` plus one labelled `a a`
is six rows in pandas and not three and not two, because every left `a` meets
every right `a` exactly as a join would pair them. `Index.union` cannot express
that, since it takes the larger of the two counts rather than the product, so
this is a join and not a union and it is a piece of work of its own rather than
a special case to bolt on here. It is #244. Until it exists, two differing
indexes with a duplicate in either of them are turned away with a message saying
so.

`fill_value` is the other rule in here and it is not the one people expect. It
fills a row where exactly one of the two sides is missing, and leaves a row where
both are missing alone, so `a.add(b, fill_value=0)` on `abc` and `bcd` answers a
number for all four labels and `a.add(b, fill_value=0)` on two columns that are
both null at `a` answers nothing there. It also fills a null that was in the
column to start with and not only one the alignment introduced, which is measured
against pandas rather than assumed.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.value import Value
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL
from firepanda.dtype.logical import LogicalType
from firepanda.frame.index import NOT_FOUND, Index
from firepanda.kernel.mask import apply_validity
from firepanda.kernel.nulls import coalesce_any, present_bitmap_any
from firepanda.kernel.select import take_any


struct Aligned(Movable):
    """Two columns on one index, ready for a kernel that walks them together."""

    var left: AnyArray
    """The left column, reindexed onto `index`."""

    var right: AnyArray
    """The right column, reindexed onto `index`."""

    var index: Index
    """The labels both columns are now on."""

    def __init__(
        out self, var left: AnyArray, var right: AnyArray, var index: Index
    ):
        """Constructs the triple.

        Args:
            left: The left column.
            right: The right column.
            index: The labels.
        """
        self.left = left^
        self.right = right^
        self.index = index^

    def into_index(deinit self) -> Index:
        """Gives up the labels, dropping the two columns.

        A caller that has already handed the columns to a kernel has no use for
        them and does want the labels, and Mojo will not let it reach in and
        move one field out of a struct that still owns the others. `deinit` says
        the rest is being torn down, which is what makes the move legal.

        Returns:
            The labels.
        """
        return self.index^


def align_pair(
    left_index: Index,
    left: AnyArray,
    right_index: Index,
    right: AnyArray,
) raises -> Aligned:
    """Reindexes two columns onto the union of their labels.

    Args:
        left_index: The left column's labels.
        left: The left column.
        right_index: The right column's labels.
        right: The right column.

    Returns:
        The two columns on one index, in the union's order.

    Raises:
        Error: If the labels differ and either index holds a duplicate, or if
            the two label dtypes cannot be unioned.
    """
    if left_index.equals(right_index):
        return Aligned(
            AnyArray(copy=left), AnyArray(copy=right), Index(copy=left_index)
        )

    _no_duplicates(left_index, "left")
    _no_duplicates(right_index, "right")

    var union = left_index.union(right_index)
    var labels = union.materialize()
    return Aligned(
        _reindex(left, left_index.get_indexer(labels)),
        _reindex(right, right_index.get_indexer(labels)),
        union^,
    )


def fill_one_sided(
    mut left: AnyArray, mut right: AnyArray, fill: Value
) raises -> Bitmap:
    """Fills a row missing from one side only, and reports which rows had both.

    The two columns are edited in place because the caller is about to hand them
    to a kernel and has no other use for what they held. The bitmap comes back
    rather than being applied here, since applying it means knowing the result
    type and the result does not exist yet.

    Args:
        left: The left column, already aligned.
        right: The right column, already aligned.
        fill: The value to put where one side is missing.

    Returns:
        The rows that should survive: everywhere except where both sides were
        missing.

    Raises:
        Error: If the fill value cannot be held by either column's type.
    """
    var keep = present_bitmap_any(left)
    keep.or_with(present_bitmap_any(right))

    var rows = len(left)
    left = coalesce_any(left, _constant(fill, rows, left.type))
    right = coalesce_any(right, _constant(fill, rows, right.type))
    return keep^


def keep_rows(mut col: AnyArray, keep: Bitmap) raises:
    """Nulls the rows a bitmap does not have set, leaving the rest alone.

    This is how a `fill_value` stops short of a row where both sides were
    missing. The fill has already gone in by the time this runs and the kernel
    has already answered, so the row holds an arithmetic result that has to be
    taken back out again.

    Args:
        col: The column to edit.
        keep: The rows to keep. Must be as long as the column.

    Raises:
        Error: If the column's dtype has no physical layout.
    """
    if keep.count_ones() == len(col):
        return
    var wanted = Bitmap(copy=col.data.validity)
    wanted.and_with(keep)
    comptime for target in ALL:
        if col.dtype() == target:
            ref typed = col.as_typed_view[target]()
            apply_validity(typed, wanted^)
            return
    raise Error("align: unsupported dtype")


def _no_duplicates(index: Index, side: String) raises:
    """Refuses an index that cannot be looked up in.

    Args:
        index: The index.
        side: Which operand it belongs to, for the message.

    Raises:
        Error: If the index holds a label more than once.
    """
    if not index.is_unique():
        raise Error(
            "align: the "
            + side
            + " operand has a duplicated row label, and pandas pairs a repeated"
            + " label with every copy of it on the other side, which is a join"
            + " and not the union this does; two operands whose labels are"
            + " identical still align, in position order"
        )


def _reindex(col: AnyArray, positions: Array[DType.int64]) raises -> AnyArray:
    """Gathers a column onto a new set of positions.

    Args:
        col: The column.
        positions: One position per output row, `NOT_FOUND` for a label this
            column does not have.

    Returns:
        A column with as many rows as there were positions.

    Raises:
        Error: If the column's dtype has no physical layout.
    """
    var rows = List[Int](capacity=len(positions))
    for i in range(len(positions)):
        var at = positions[i]
        # `take_any` reads any negative position as a row that is not there,
        # which is what `NOT_FOUND` already is, so the two agree without a
        # translation step.
        rows.append(Int(at) if at != NOT_FOUND else -1)
    return take_any(col, rows)


def _constant(value: Value, rows: Int, type: LogicalType) raises -> AnyArray:
    """Builds a column of one value repeated, in a given type.

    The type is the type of the column the fill is going into rather than the
    fill's own type, so that filling a float column with a zero written as an
    integer does not promote the column on the way past.

    Args:
        value: The value.
        rows: How many rows.
        type: The type the column should have.

    Returns:
        The column.

    Raises:
        Error: If the type is a text one, which has no arithmetic for a fill
            value to take part in, or has no physical layout.
    """
    if type.is_variable_width():
        raise Error(
            "align: fill_value has nothing to do on a text column, which has"
            " no arithmetic to fill a gap in"
        )
    comptime for target in ALL:
        if type.physical == target:
            var out = Array[target](rows)
            var element = value.as_scalar[target]()
            for i in range(rows):
                out.set_valid(i, element)
            return AnyArray(out^)
    raise Error("align: unsupported dtype")
