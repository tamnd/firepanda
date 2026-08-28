"""Stacking frames on top of each other.

This is a free function rather than a method for the reason it usually is: the
frames are peers. `a.concat(b)` reads as though `a` were doing something to `b`,
and the two argument spelling would then need a third one for the list case,
which is what anybody reading a directory of files actually has.

Two rules make the operation total.

**Names decide, not positions.** A frame whose columns are in a different order
still stacks, because the schema is the authority and the columns are looked up
through it. Stacking by position would mean a reordered frame silently producing
a column with two different meanings in it, and there is nothing on screen at the
call site that would catch it.

**Dtypes must match.** Stacking an int32 column onto a float64 one is refused
rather than promoted, which is the same rule `concat` and `coalesce` follow in
the kernel layer. The cast is one line and belongs where a reader can see it.

Rows are the concatenation of the parts in the order given, so `concat` of a
frame with itself is that frame twice and not a set union. Nothing here dedupes
and nothing here sorts.

There is no horizontal spelling. Putting two frames side by side means deciding
which row of one lines up with which row of the other, and without an index the
only available answer is position, which is a real answer but a sharp one. It
waits until there is something to align on.
"""

from firepanda.array.any import AnyArray
from firepanda.dtype.schema import Field, Schema
from firepanda.kernel.concat import concat_any, concat_two_any

from .frame import DataFrame
from .series import Series


def concat(frames: List[DataFrame]) raises -> DataFrame:
    """Stacks frames on top of each other, matching columns by name.

    Args:
        frames: The frames, in output row order. An empty list gives an empty
            frame, and a single frame is copied.

    Returns:
        A frame as tall as the parts put together, with the first frame's
        columns in the first frame's order.

    Raises:
        If two frames disagree on which columns exist or on a column's dtype.
    """
    if len(frames) == 0:
        return DataFrame()
    if len(frames) == 1:
        return DataFrame(copy=frames[0])

    var names = frames[0].names()
    var width = frames[0].width()

    # Resolve every part's column positions against the first frame's names
    # before copying anything, so a mismatched frame raises before the work.
    var at = List[List[Int]](capacity=len(frames))
    for f in range(len(frames)):
        var part = List[Int](capacity=len(names))
        if frames[f].width() != width:
            raise Error(
                "concat: every frame must have the same columns; frame 0 has "
                + String(width)
                + " and frame "
                + String(f)
                + " has "
                + String(frames[f].width())
            )
        for c in range(len(names)):
            var found = frames[f].index_of(names[c])
            if frames[f][found].dtype() != frames[0][c].dtype():
                raise Error(
                    "concat: column '"
                    + names[c]
                    + "' is "
                    + String(frames[0][c].dtype())
                    + " in frame 0 and "
                    + String(frames[f][found].dtype())
                    + " in frame "
                    + String(f)
                )
            part.append(found)
        at.append(part^)

    var columns = List[AnyArray](capacity=len(names))
    for c in range(len(names)):
        if len(frames) == 2:
            # The common call, and the one worth a branch. Handing the kernel a
            # `List[AnyArray]` means owning the parts, which for columns this
            # function only borrows means deep copying every one of them before
            # the copy that concat itself does. Two arguments borrow.
            columns.append(
                concat_two_any(frames[0][at[0][c]], frames[1][at[1][c]])
            )
            continue
        var parts = List[AnyArray](capacity=len(frames))
        for f in range(len(frames)):
            parts.append(AnyArray(copy=frames[f][at[f][c]]))
        columns.append(concat_any(parts))

    var fields = List[Field](capacity=len(names))
    for c in range(len(names)):
        fields.append(frames[0].schema.fields[c].copy())
    return DataFrame(Schema(fields^), columns^)


def concat_series(parts: List[Series]) raises -> Series:
    """Stacks series on top of each other.

    Args:
        parts: The series, in output row order. All of them must have the same
            dtype. The first one's name is the result's, because a name is not
            something two series can be asked to agree on.

    Returns:
        A series as tall as the parts put together.

    Raises:
        If the list is empty, if two parts disagree on dtype, or if the dtype
        has no physical layout.
    """
    if len(parts) == 0:
        raise Error("concat: at least one series is required")

    var columns = List[AnyArray](capacity=len(parts))
    for p in range(len(parts)):
        columns.append(AnyArray(copy=parts[p].values))
    return Series(parts[0].name, concat_any(columns))
