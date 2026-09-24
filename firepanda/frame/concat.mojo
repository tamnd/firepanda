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

from std.collections import Set
from std.memory import ArcPointer

from firepanda.array.any import AnyArray
from firepanda.array.chunked import ChunkedArray
from firepanda.array.strings import StringBuilder
from firepanda.dtype.logical import LogicalType, TypeKind
from firepanda.dtype.schema import Field, Schema
from firepanda.kernel.binary import all_null
from firepanda.kernel.concat import (
    column_ref,
    concat_any,
    concat_refs_any,
    concat_two_any,
)

from .frame import DataFrame
from .index import Index
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
            # The common call, and the one worth a branch: two arguments say
            # what they mean and skip building a list at all.
            columns.append(
                concat_two_any(frames[0][at[0][c]], frames[1][at[1][c]])
            )
            continue
        # References rather than columns. This function only borrows its
        # frames, so a `List[AnyArray]` would mean deep copying every part
        # before the copy the concat itself does, which is the whole cost of
        # the operation paid twice.
        var parts = List[Pointer[AnyArray, ImmUntrackedOrigin]](
            capacity=len(frames)
        )
        for f in range(len(frames)):
            parts.append(column_ref(frames[f][at[f][c]]))
        columns.append(concat_refs_any(parts))

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

    var columns = List[Pointer[AnyArray, ImmUntrackedOrigin]](
        capacity=len(parts)
    )
    for p in range(len(parts)):
        columns.append(column_ref(parts[p].values))
    return Series(parts[0].name, concat_refs_any(columns))


def stack_rows(
    frames: List[ArcPointer[DataFrame]], names: List[String], labels: Bool
) raises -> DataFrame:
    """Stacks frames over a set of columns some of them may not have.

    This is `pd.concat` down the rows once the Python layer has settled the
    types. `concat` above wants every frame to hold the same columns, and pandas
    does not: a column one frame lacks is missing on that frame's rows. So each
    column here is the frames' own pieces with a block of nulls standing in for
    a frame that has no such column, stacked in one pass.

    The row labels come along when `labels` is set, which is pandas' default and
    the reason `concat([df, df])` repeats its index. The result's index takes the
    name every part agrees on and no name otherwise. With `labels` off the rows
    are numbered from zero, which is `ignore_index=True`.

    Args:
        frames: The frames, in output row order. Only borrowed.
        names: The result's columns, in order. Every one has to be in at least
            one frame, with the same type in every frame that has it.
        labels: Whether to keep the parts' row labels.

    Returns:
        A frame as tall as the parts put together.

    Raises:
        If a column is in no frame, if two frames disagree on its type, or if
        two parts' labels have different types.
    """
    var rows = 0
    for f in range(len(frames)):
        rows += frames[f][].rows

    var fields = List[Field](capacity=len(names))
    var columns = List[AnyArray](capacity=len(names))
    for c in range(len(names)):
        var name = names[c]
        var at = List[Int](capacity=len(frames))
        var found = -1
        for f in range(len(frames)):
            var position = -1
            ref fields_of = frames[f][].schema.fields
            for i in range(len(fields_of)):
                if fields_of[i].name == name:
                    position = i
                    break
            at.append(position)
            if position >= 0 and found < 0:
                found = f
        if found < 0:
            raise Error(
                "concat: column '" + name + "' is in none of the frames"
            )
        var type = frames[found][].columns[at[found]].type.copy()
        var nullable = False

        # The blocks of nulls go into a list of their own before any reference
        # is taken, since a reference into a list that is still growing can be
        # left pointing at memory the list has moved away from.
        var blocks = List[AnyArray]()
        for f in range(len(frames)):
            if at[f] < 0:
                nullable = True
                blocks.append(_null_block(type, frames[f][].rows))
                continue
            ref column = frames[f][].columns[at[f]]
            if column.type != type:
                raise Error(
                    "concat: column '"
                    + name
                    + "' is "
                    + String(type)
                    + " in one frame and "
                    + String(column.type)
                    + " in another"
                )
            if frames[f][].schema.fields[at[f]].nullable:
                nullable = True

        var parts = List[Pointer[AnyArray, ImmUntrackedOrigin]](
            capacity=len(frames)
        )
        var block = 0
        for f in range(len(frames)):
            if at[f] < 0:
                parts.append(column_ref(blocks[block]))
                block += 1
                continue
            ref column = frames[f][].columns[at[f]]
            for k in range(len(column.chunks)):
                parts.append(column_ref(column.chunks[k]))
        if len(parts) == 0:
            columns.append(_null_block(type, 0))
        else:
            columns.append(concat_refs_any(parts))
        # The references above do not keep the blocks alive, and without this
        # the list can be freed as soon as the last reference is taken, before
        # the stack reads through them.
        _ = blocks^
        var field = Field(name, type)
        field.nullable = nullable
        fields.append(field^)

    var out = DataFrame(Schema(fields^), columns^)
    out.rows = rows
    if labels:
        out.index = _stacked_index(frames, rows)
    else:
        out.index = Index(rows)
    return out^


def stack_columns(frames: List[ArcPointer[DataFrame]]) raises -> DataFrame:
    """Puts frames side by side, which is `pd.concat(axis=1)`.

    The rows are lined up by position, so the Python layer brings every part to
    the same labels in the same order first, which is where pandas' alignment
    happens. Nothing is copied here: a column is shared with the frame it came
    from until one of them writes to it. The result keeps the first frame's
    labels.

    Args:
        frames: The frames, in output column order. Only borrowed.

    Returns:
        A frame as wide as the parts put together.

    Raises:
        If the frames differ in height or two of them have a column of the same
        name, which pandas allows and a schema here cannot hold.
    """
    if len(frames) == 0:
        return DataFrame()
    var rows = frames[0][].rows
    var fields = List[Field]()
    var columns = List[ChunkedArray]()
    var seen = Set[String]()
    for f in range(len(frames)):
        ref part = frames[f][]
        if part.rows != rows:
            raise Error(
                "concat: frame 0 has "
                + String(rows)
                + " rows and frame "
                + String(f)
                + " has "
                + String(part.rows)
            )
        for c in range(len(part.columns)):
            var name = part.schema.fields[c].name
            if name in seen:
                raise Error(
                    "concat: two parts have a column named '"
                    + name
                    + "', and a frame here holds one column per name"
                )
            seen.add(name)
            fields.append(part.schema.fields[c].copy())
            columns.append(part.columns[c].copy())
    var out = DataFrame(Schema(fields^), columns^)
    out.rows = rows
    out.index = frames[0][].index.copy()
    return out^


def _null_block(type: LogicalType, rows: Int) raises -> AnyArray:
    """A column of a given type with every row missing.

    Args:
        type: The column's type.
        rows: How many rows.

    Returns:
        The column.

    Raises:
        If the type has no layout this can build.
    """
    if type.kind == TypeKind.STRING:
        var builder = StringBuilder(rows)
        for _ in range(rows):
            builder.append_null()
        return AnyArray(builder^.finish())
    return all_null(type, rows)


def _stacked_index(
    frames: List[ArcPointer[DataFrame]], rows: Int
) raises -> Index:
    """The parts' row labels one after another.

    Args:
        frames: The frames, in output row order.
        rows: Their total height, for the default labels of no frames at all.

    Returns:
        The labels, named for the name every part shares.

    Raises:
        If two parts' labels have different types.
    """
    if len(frames) == 0:
        return Index(rows)
    var name = frames[0][].index.name.copy()
    var labels = List[AnyArray](capacity=len(frames))
    for f in range(len(frames)):
        ref index = frames[f][].index
        if name and (not index.name or index.name.value() != name.value()):
            name = None
        labels.append(index.materialize())
    return Index(concat_any(labels), name^)
