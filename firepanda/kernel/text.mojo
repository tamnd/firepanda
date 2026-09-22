"""Elementwise comparison of text columns, producing a boolean column.

`compare.mojo` holds the same six operations for the fixed width dtypes and does
them a register at a time. Text cannot be done that way. An element is a run of
bytes of its own length, so the comparison is a loop over bytes and the answer
for one row says nothing about how long the next row takes. That is why this is a
file of its own rather than another arm of the numeric dispatch.

The operation codes are the `CMP_` ones from `compare.mojo`, reused rather than
redefined, because the caller that picks between the two files picks the code
once and does not want to know which family it ended up in.

Equality and ordering take different routes on purpose. Equality can be settled
by the view alone whenever the two strings are short, since a short view holds
the whole string zero padded into sixteen bytes and two of them are equal exactly
when the strings are. Ordering cannot use that trick, because the bytes are
packed into words in an order that makes equality one compare and makes ordering
wrong, so it goes to the byte loop.

The constant form hoists what it can out of the loop. A short constant is turned
into a view once, before any row is read, and then the rows are taken a block at
a time: a view is two 64-bit words, and one exclusive or against a register
holding a copy of the constant per row settles the whole block with nothing
loaded but the views themselves. That is the shape a filter on a status column
or a country code has, and it is the case worth being fast. The tail, which is
shorter than a block, goes one row at a time.

Comparison against a null is null, exactly as it is for numbers, and it is
handled the same way: the loop writes whatever falls out and the repair at the
end of each morsel clears the rows where either side was missing. The value
under a null is false either way, so nothing depends on which branch the loop
took there.

Both mask forms run on every core. A row's cost here depends on its own bytes
and on nothing else, so the column splits over morsels with no more care than a
numeric kernel needs, and the only thing that had to move is the constant's view,
which is now built once above the split rather than once per worker.

There is a third form against a constant, which answers the rows it keeps rather
than a column of bits, and it is serial. A filter is the only caller, the filter
is already on a worker, and the bool column between the comparison and the rows
was a byte a row written and read for nothing. It has an arm that reads the
column through a selection, which is what keeps a text condition sitting behind
other conditions from gathering the column it compares.
"""

from std.collections.span import Span

from firepanda.array.array import Array
from firepanda.array.strings import StringArray
from firepanda.array.strview import (
    EQUAL_BLOCK,
    INLINE_CAPACITY,
    StringView,
    make_inline,
    short_pattern,
    views_equal_short,
)
from firepanda.bitmap.bitmap import Bitmap
from firepanda.exec import parallel_morsels

from .compare import CMP_EQ, CMP_GE, CMP_GT, CMP_LE, CMP_LT, CMP_NE
from .mask import combined_validity, repair_range


def compare_text[
    op: Int
](a: StringArray, b: StringArray) raises -> Array[DType.bool]:
    """Compares two text columns elementwise.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        op: One of the `CMP_` codes from `compare.mojo`.

    Returns:
        A bool column, null wherever either input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    # Every row is written below, so the zeroing allocation is a wasted pass.
    var out = Array[DType.bool](overwritten=n)
    var validity = combined_validity(a.validity, b.validity)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_mut_ptr()
        comptime if op == CMP_EQ or op == CMP_NE:
            for i in range(start, stop):
                # `equals` answers False for a null on the left, and the bytes
                # of a null on the right are empty, so a null row can come out
                # either way here. The repair pass below settles it.
                var same = a.equals(i, b.unsafe_bytes(i))
                comptime if op == CMP_EQ:
                    dst.unsafe_offset(i).unsafe_write(Scalar[DType.bool](same))
                else:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](not same)
                    )
        else:
            for i in range(start, stop):
                var order = a.compare(i, b.unsafe_bytes(i))
                comptime if op == CMP_LT:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](order < 0)
                    )
                elif op == CMP_LE:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](order <= 0)
                    )
                elif op == CMP_GT:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](order > 0)
                    )
                else:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](order >= 0)
                    )

        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def compare_text_const[
    op: Int
](a: StringArray, b: Span[UInt8, _]) raises -> Array[DType.bool]:
    """Compares a text column against one run of bytes.

    There is no flipped form. A constant on the left is the mirrored operation
    on the right, and the caller does that swap, which is the same arrangement
    the numeric constant kernel has.

    Args:
        a: The column.
        b: The constant's bytes. Borrowed for the length of the call and not
            stored.

    Parameters:
        op: One of the `CMP_` codes from `compare.mojo`.

    Returns:
        A bool column, null wherever the column is null. A null constant makes
        the whole answer null and never reaches here.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    # Every row is written below, so the zeroing allocation is a wasted pass.
    var out = Array[DType.bool](overwritten=n)
    var validity = Bitmap(copy=a.validity)

    # A short constant becomes a view once, before any row is read and before
    # the split, so the workers share it rather than each making their own. A
    # long one cannot: the view for a long string points into a payload this
    # constant does not live in, so those rows go to the byte loop.
    var short = len(b) <= INLINE_CAPACITY
    var probe = StringView()
    comptime if op == CMP_EQ or op == CMP_NE:
        if short:
            probe = make_inline(b)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_mut_ptr()
        comptime if op == CMP_EQ or op == CMP_NE:
            # The two cases get a loop each rather than one loop asking which
            # it is on every row. They were one loop with a branch in it, and
            # a 13900K read the long case six percent slower that way once the
            # block compare was sitting above it, on code the block compare
            # does not otherwise touch.
            if short:
                var pattern = short_pattern(probe)
                var i = start
                while i + EQUAL_BLOCK <= stop:
                    var block = a.equal_short_block(i, pattern)
                    comptime if op == CMP_NE:
                        block = ~block
                    dst.unsafe_offset(i).unsafe_store(block)
                    i += EQUAL_BLOCK

                # Fewer rows than a block, so at most `EQUAL_BLOCK - 1`.
                while i < stop:
                    var same = views_equal_short(a.view(i), probe)
                    comptime if op == CMP_NE:
                        same = not same
                    dst.unsafe_offset(i).unsafe_write(Scalar[DType.bool](same))
                    i += 1
            else:
                for i in range(start, stop):
                    var same = a.equals(i, b)
                    comptime if op == CMP_NE:
                        same = not same
                    dst.unsafe_offset(i).unsafe_write(Scalar[DType.bool](same))
        else:
            for i in range(start, stop):
                var order = a.compare(i, b)
                comptime if op == CMP_LT:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](order < 0)
                    )
                elif op == CMP_LE:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](order <= 0)
                    )
                elif op == CMP_GT:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](order > 0)
                    )
                else:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](order >= 0)
                    )

        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def _text_holds[
    op: Int
](
    a: StringArray, i: Int, b: Span[UInt8, _], probe: StringView, short: Bool
) raises -> Bool:
    """Answers one comparison between an element and the constant.

    Args:
        a: The column.
        i: The element.
        b: The constant's bytes.
        probe: The constant as a view, built only when it is short and the
            operation is equality. Not read otherwise.
        short: Whether `probe` holds the constant.

    Parameters:
        op: One of the `CMP_` codes.

    Returns:
        Whether the comparison is true of the pair, taking the element as
        present.

    Raises:
        Never. The signature carries it because the element readers do.
    """
    comptime if op == CMP_EQ or op == CMP_NE:
        var same = views_equal_short(a.view(i), probe) if short else a.equals(
            i, b
        )
        comptime if op == CMP_EQ:
            return same
        else:
            return not same
    else:
        var order = a.compare(i, b)
        comptime if op == CMP_LT:
            return order < 0
        elif op == CMP_LE:
            return order <= 0
        elif op == CMP_GT:
            return order > 0
        else:
            return order >= 0


def compare_text_const_positions[
    op: Int
](a: StringArray, b: Span[UInt8, _]) raises -> List[UInt32]:
    """Compares a text column against a constant and returns the rows it keeps.

    What `compare_text_const` followed by `select_positions` answers, without the
    bool column in the middle, and the text half of what
    `compare_const_positions` does for the fixed width dtypes. A filter over
    `SearchPhrase <> ''` is eleven of the ClickBench statements, and every one of
    them wrote a byte a row and read it back to carry an answer the loop already
    had.

    Serial, unlike `compare_text_const`, because the cursor is the loop and the
    caller is already on a worker of its own. That is the same trade the numeric
    positions kernels make and the reason is written out there.

    A null element drops the row, which is what `select_positions` does to a null
    in a mask and is what makes the two routes agree. It matters more here than
    it does for numbers: a null element's view is the view of the empty string,
    so a null would otherwise answer true to `= ''` rather than dropping out of
    both that and its opposite.

    Args:
        a: The column.
        b: The constant's bytes. Borrowed for the length of the call and not
            stored.

    Parameters:
        op: One of the `CMP_` codes from `compare.mojo`.

    Returns:
        The positions the comparison is true on, in order.

    Raises:
        Never. The signature carries it because the element readers do.
    """
    var n = len(a)
    # Room for every row taken in front and cut back at the end, which is what
    # the numeric positions kernels do and for the reason written there.
    var out = List[UInt32](unsafe_uninit_length=n)
    var target = out.unsafe_ptr()
    var at = 0

    # The same hoist the mask form does, and only for equality, since ordering
    # has no use for the view. There is no block compare here: a block settles
    # `EQUAL_BLOCK` rows at once and the cursor takes them one at a time anyway.
    var short = False
    var probe = StringView()
    comptime if op == CMP_EQ or op == CMP_NE:
        short = len(b) <= INLINE_CAPACITY
        if short:
            probe = make_inline(b)

    # The branchless cursor, from `select_positions`. A row nobody keeps is a
    # store the next row overwrites.
    if a.null_count() == 0:
        for i in range(n):
            target.unsafe_offset(at).unsafe_write(UInt32(i))
            at += Int(_text_holds[op](a, i, b, probe, short))
        out.resize(at, 0)
        return out^

    for i in range(n):
        var valid = Int(a.validity.get(i))
        target.unsafe_offset(at).unsafe_write(UInt32(i))
        at += valid & Int(_text_holds[op](a, i, b, probe, short))
    out.resize(at, 0)
    return out^


def compare_text_const_positions_through[
    op: Int
](a: StringArray, b: Span[UInt8, _], picks: List[UInt32]) raises -> List[
    UInt32
]:
    """The same comparison over a column that is read through a selection.

    The rows are `a[picks[0]]`, `a[picks[1]]` and so on, and what comes back is
    positions into `picks` rather than into `a`, so it composes with the
    selection the chunk arrived under exactly as a mask over those rows would
    have.

    This is the arm the ClickBench page view statements land on. `URL <> ''` is
    the last of six conditions there, so by the time it runs the five in front of
    it have composed a selection holding a small part of the rows, and the only
    way to compare without this was to gather the whole URL column through that
    selection first. A gathered text column is not the cheap kind of copy either:
    it is a payload as well as a view per row.

    Args:
        a: The column the positions point into.
        b: The constant's bytes. Borrowed for the length of the call and not
            stored.
        picks: The selection, one position per row.

    Parameters:
        op: One of the `CMP_` codes from `compare.mojo`.

    Returns:
        The positions into `picks` the comparison is true on, in order.

    Raises:
        Never. The signature carries it because the element readers do.
    """
    var n = len(picks)
    var read = picks.unsafe_ptr()
    var out = List[UInt32](unsafe_uninit_length=n)
    var target = out.unsafe_ptr()
    var at = 0

    var short = False
    var probe = StringView()
    comptime if op == CMP_EQ or op == CMP_NE:
        short = len(b) <= INLINE_CAPACITY
        if short:
            probe = make_inline(b)

    if a.null_count() == 0:
        for j in range(n):
            var i = Int(read.unsafe_offset(j).unsafe_load())
            target.unsafe_offset(at).unsafe_write(UInt32(j))
            at += Int(_text_holds[op](a, i, b, probe, short))
        out.resize(at, 0)
        return out^

    for j in range(n):
        var i = Int(read.unsafe_offset(j).unsafe_load())
        var valid = Int(a.validity.get(i))
        target.unsafe_offset(at).unsafe_write(UInt32(j))
        at += valid & Int(_text_holds[op](a, i, b, probe, short))
    out.resize(at, 0)
    return out^
