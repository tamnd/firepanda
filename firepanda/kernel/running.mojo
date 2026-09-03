"""Folding a chunk's rows straight into a running table.

Every other aggregation kernel in the package takes a column and gives back one
row per group. These take a column and a table that already has a row per group,
and add the one to the other. That difference is what a streaming group by needs.

An operator holding a running table over a hundred thousand groups and reading a
chunk of a hundred thousand rows can do it two ways. It can reduce the chunk into
a second table and then reduce the two tables into a third, which is an
allocation the size of the table and three passes over the groups, every chunk.
Or it can add the chunk's rows to the slots it already has, which is one pass
over the chunk and no allocation at all. The first way costs the number of chunks
times the number of groups, and the number of chunks grows with the input, so the
part of the work that is not proportional to the input grows with the square of
it. On a thousand groups that is invisible. On a hundred thousand it is the whole
cost of the operator.

## Why the table is longer than the group count

A chunk can introduce groups, so the table has to grow, and growing it by exactly
the number of new groups is a copy of the whole table per chunk, which is the
cost this file exists to remove. So the caller keeps a capacity as well as a
count, `state_capacity` doubles it, and `settle_any` cuts the table down to its
real height once at the end. That is the ordinary growable array argument and it
is here for the ordinary reason.

## Identities and which slots have been reached

A slot no row has reached yet has to hold something, and what it holds is the
reduction's identity: a zero for the sums and the counts, the largest value of
the dtype for a minimum and the smallest for a maximum. That is what lets the
inner loops have no branch on whether a group has been seen before, which is the
same trade `_extreme_core` makes in `group.mojo` and it is made here for the same
reason.

An identity is not an answer, though, so the extremes and the edges carry the
reached flags in the validity bitmap they had to maintain anyway, and
`settle_any` turns a slot that was never reached into a null holding a zero,
which is what a null holds everywhere else in the package. The sums and the
counts need none of that: a group whose rows were all null has a sum of zero and
a count of zero, and both of those are the answer rather than a stand in for one.

## What is not here

Text. A minimum over a column of names is a comparison over bytes and a running
slot would have to own a copy of the bytes it currently holds, which is a
different kind of state from a number in a slot and wants its own file. The
caller checks for it and keeps the older route, which is correct and merely
slower, and slower on a shape that is rare.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.lists import ALL

from .accum import accumulator, highest, lowest
from .group import AggKind


def state_capacity(needed: Int, have: Int) -> Int:
    """Returns how long a running state column should be to hold a group count.

    Args:
        needed: The group count the table has to index.
        have: How long the column is now.

    Returns:
        `have` if that is already enough, and otherwise at least twice it, so
        that a table which grows a little on every chunk is copied a logarithmic
        number of times rather than every chunk.
    """
    if needed <= have:
        return have
    var next = have * 2
    if next < needed:
        next = needed
    if next < MIN_SLOTS:
        next = MIN_SLOTS
    return next


comptime MIN_SLOTS = 64
"""Slots in the smallest running state column.

Small enough that a group by on four groups wastes nothing worth naming, large
enough that the first few chunks of a real one do not each copy the table.
"""


def tracks_reach(kind: AggKind) -> Bool:
    """Reports whether a reduction needs to know which slots it has reached.

    Args:
        kind: The reduction.

    Returns:
        True for the extremes and the edges, whose identity is not an answer,
        and False for the sums and the counts, whose identity is.
    """
    return (
        kind == AggKind.MIN
        or kind == AggKind.MAX
        or kind == AggKind.FIRST
        or kind == AggKind.LAST
    )


def widen_any(mut state: AnyArray, capacity: Int, kind: AggKind) raises:
    """Grows a running state column and puts every unreached slot at the identity.

    Both halves of that matter. The new slots are unreached because they have
    just been made, and some of the old ones are unreached because the reduction
    that built the column left a group that saw nothing but nulls holding a zero,
    which is the right thing for an answer and the wrong thing for an
    accumulator: a minimum against a zero is not a minimum against negative
    numbers. So this walks the whole column rather than the tail of it, and it is
    the reason a caller can hand over a column `aggregate_group_any` produced and
    then keep folding into it.

    Args:
        state: The column, replaced by the grown one.
        capacity: How many slots the result should have. Must be at least the
            column's current length.
        kind: The reduction the column is accumulating.

    Raises:
        If the column's dtype has no physical layout.
    """
    var reached = tracks_reach(kind)
    comptime for slot in ALL:
        if state.dtype() == slot:
            var identity = Scalar[slot](0)
            if kind == AggKind.MIN:
                identity = highest[slot]()
            elif kind == AggKind.MAX:
                identity = lowest[slot]()
            _widen[slot](state, capacity, identity, reached)
            return
    raise Error(
        "running: state dtype "
        + String(state.dtype())
        + " has no physical layout"
    )


def _widen[
    dt: DType
](
    mut state: AnyArray, capacity: Int, identity: Scalar[dt], reached: Bool
) raises:
    """Grows one column of a known dtype."""
    ref old = state.as_typed_view[dt]()
    var have = len(old)
    var out = Array[dt](capacity)
    var src = old.unsafe_ptr()
    var dst = out.unsafe_ptr()

    if not reached:
        for i in range(have):
            dst.unsafe_offset(i).unsafe_write(
                src.unsafe_offset(i).unsafe_load()
            )
        for i in range(have, capacity):
            dst.unsafe_offset(i).unsafe_write(identity)
        state = AnyArray(out^)
        return

    # A fresh array is all present, and every slot past the ones carried over is
    # unreached, so the flags start empty and the carried ones are put back.
    out.data.validity.clear_all()
    for i in range(have):
        if old.data.validity.get(i):
            dst.unsafe_offset(i).unsafe_write(
                src.unsafe_offset(i).unsafe_load()
            )
            out.data.validity.set(i, True)
        else:
            dst.unsafe_offset(i).unsafe_write(identity)
    for i in range(have, capacity):
        dst.unsafe_offset(i).unsafe_write(identity)
    state = AnyArray(out^)


def settle_any(
    var state: AnyArray, kind: AggKind, groups: Int
) raises -> AnyArray:
    """Cuts a running state column down to its groups and blanks what it missed.

    Args:
        state: The column, as long as its capacity. Consumed.
        kind: The reduction it was accumulating.
        groups: How many groups it really has.

    Returns:
        The first `groups` slots, with every one the reduction never reached
        holding a null over a zero rather than the identity it was accumulating
        against.

    Raises:
        If the column's dtype has no physical layout.
    """
    if tracks_reach(kind):
        _blank_unreached(state, groups)
    return state.slice(0, groups)


def _blank_unreached(mut state: AnyArray, groups: Int) raises:
    """Puts a zero in every slot whose validity bit is clear."""
    comptime for slot in ALL:
        if state.dtype() == slot:
            ref view = state.as_typed_view[slot]()
            var at = view.unsafe_ptr()
            for g in range(groups):
                if not view.data.validity.get(g):
                    at.unsafe_offset(g).unsafe_write(Scalar[slot](0))
            return
    raise Error(
        "running: state dtype "
        + String(state.dtype())
        + " has no physical layout"
    )


def accumulate_any(
    mut state: AnyArray,
    values: AnyArray,
    kind: AggKind,
    codes: Array[DType.uint32],
    rows: Int,
) raises:
    """Folds a chunk's rows into the slots their groups already own.

    Args:
        state: The running table for one reduction, at least as long as the
            largest ordinal in `codes` plus one. Written in place.
        values: The chunk's column to reduce.
        kind: The reduction.
        codes: The group ordinal of every row, in the running table's numbering
            rather than the chunk's.
        rows: How many rows of `values` to fold.

    Raises:
        If the dtype has no physical layout, or if the reduction is one this does
        not fold in place.
    """
    comptime for source in ALL:
        if values.dtype() == source:
            _accumulate[source](state, values, kind, codes, rows)
            return
    raise Error(
        "running: dtype " + String(values.dtype()) + " has no physical layout"
    )


def _accumulate[
    dt: DType
](
    mut state: AnyArray,
    values: AnyArray,
    kind: AggKind,
    codes: Array[DType.uint32],
    rows: Int,
) raises:
    """Folds a chunk of a known dtype, one loop per reduction.

    The reductions get a loop each rather than one loop with the kind tested
    inside it, because the test would be per row and it would be the only branch
    in a body that is otherwise a load, an add and a store.
    """
    ref column = values.as_typed_view[dt]()
    var src = column.unsafe_ptr()
    var at = codes.unsafe_ptr()
    var has_null = column.null_count() > 0

    if kind == AggKind.SUM:
        comptime acc = accumulator(dt)
        ref into = state.as_typed_view[acc]()
        var total = into.unsafe_ptr()
        # Validity is ignored here on purpose, the same way `_sum_core` ignores
        # it: a null holds a zero and adding a zero is what skipping it would
        # have done, without the branch.
        for i in range(rows):
            var g = Int(at.unsafe_offset(i).unsafe_load())
            total.unsafe_offset(g).unsafe_store(
                total.unsafe_offset(g).unsafe_load()
                + src.unsafe_offset(i).unsafe_load().cast[acc]()
            )
        return

    if kind == AggKind.SIZE:
        ref into = state.as_typed_view[DType.int64]()
        var tally = into.unsafe_ptr()
        for i in range(rows):
            var g = Int(at.unsafe_offset(i).unsafe_load())
            tally.unsafe_offset(g).unsafe_store(
                tally.unsafe_offset(g).unsafe_load() + 1
            )
        return

    if kind == AggKind.COUNT:
        ref into = state.as_typed_view[DType.int64]()
        var tally = into.unsafe_ptr()
        for i in range(rows):
            if has_null and not column.data.validity.get(i):
                continue
            var g = Int(at.unsafe_offset(i).unsafe_load())
            tally.unsafe_offset(g).unsafe_store(
                tally.unsafe_offset(g).unsafe_load() + 1
            )
        return

    if kind == AggKind.MIN:
        _extreme_into[dt, True](state, column, codes, rows, has_null)
        return

    if kind == AggKind.MAX:
        _extreme_into[dt, False](state, column, codes, rows, has_null)
        return

    if kind == AggKind.FIRST:
        ref into = state.as_typed_view[dt]()
        var best = into.unsafe_ptr()
        for i in range(rows):
            if has_null and not column.data.validity.get(i):
                continue
            var g = Int(at.unsafe_offset(i).unsafe_load())
            if into.data.validity.get(g):
                continue
            best.unsafe_offset(g).unsafe_store(
                src.unsafe_offset(i).unsafe_load()
            )
            into.data.validity.set(g, True)
        return

    if kind == AggKind.LAST:
        ref into = state.as_typed_view[dt]()
        var best = into.unsafe_ptr()
        for i in range(rows):
            if has_null and not column.data.validity.get(i):
                continue
            var g = Int(at.unsafe_offset(i).unsafe_load())
            best.unsafe_offset(g).unsafe_store(
                src.unsafe_offset(i).unsafe_load()
            )
            into.data.validity.set(g, True)
        return

    raise Error(
        "running: " + String(kind) + " does not fold into a running slot"
    )


def _extreme_into[
    dt: DType, want_min: Bool
](
    mut state: AnyArray,
    column: Array[dt],
    codes: Array[DType.uint32],
    rows: Int,
    has_null: Bool,
) raises:
    """Folds a chunk into a running minimum or maximum.

    The comparison is against the slot's current value with no test for whether
    the group has been reached, because an unreached slot holds the identity and
    every value beats it. The reached flag is written unconditionally for the
    same reason: a branch to avoid a bit write is not worth a branch.
    """
    ref into = state.as_typed_view[dt]()
    var best = into.unsafe_ptr()
    var src = column.unsafe_ptr()
    var at = codes.unsafe_ptr()
    for i in range(rows):
        if has_null and not column.data.validity.get(i):
            continue
        var g = Int(at.unsafe_offset(i).unsafe_load())
        var value = src.unsafe_offset(i).unsafe_load()
        var current = best.unsafe_offset(g).unsafe_load()
        comptime if want_min:
            if value < current:
                best.unsafe_offset(g).unsafe_store(value)
        else:
            if value > current:
                best.unsafe_offset(g).unsafe_store(value)
        into.data.validity.set(g, True)
