"""Reducing a whole column to one value, with no grouping in the way.

Tier: unstable, documented. docs/specs/11-package-layout.md.

`group.mojo` can already answer this question. Hand it a code per row that is
always zero and a group count of one and it reduces the whole column, which is
exactly what a frame wanting a total wants. It is also the slowest possible way
to ask. Ten million rows means allocating and zeroing forty megabytes of codes,
then walking them beside the values and scattering into a table with one entry
in it. The scatter is a dependent store per row and none of it vectorizes.

`agg.mojo` has had the fast answer the whole time. A sum with nothing to group by
is a vectorized add over the values buffer and a minimum is a walk of the
validity a word at a time. On ten million float64 rows on gamingpc the group
route takes 85 ms and this one takes 5.

So this file is a dispatch and not an implementation. The reductions that
`agg.mojo` has get the fast route. The rest, meaning the variance, the order
statistics and the distinct count, go to `aggregate_group_any` with a single
group, because they have no whole column spelling yet and a correct slow answer
is better than a missing one. The dividing line is written down in
`_takes_fast_route` rather than being spread through the branches, so that adding
a whole column variance later is a change in two places and not a hunt.

Every result is a column of one row, not a scalar. That is what the frame layer
wants back, because `DataFrame.agg` builds a one row frame out of these and a
scalar would have to be widened into a column anyway. It also means the dtype
rules are the same rules the group by uses, which they have to be: a sum over an
int32 column widens to int64 whether or not anybody grouped it.
"""

from std.math import nan

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL
from firepanda.dtype.logical import LogicalType, TypeKind
from firepanda.hash.factorize import (
    DIRECT_LIMIT,
    direct_plan,
    distinct_hashed,
    distinct_strings,
)

from .accum import accumulator
from .agg import (
    extreme_over,
    mean_over,
    sum_over,
    text_edge_row,
    text_extreme_row,
)
from .group import (
    AggKind,
    aggregate_group_any,
    retag_temporal,
    temporal_agg_type,
)
from .nulls import missing_count_any


def _takes_fast_route(kind: AggKind) -> Bool:
    """Reports whether this reduction has a whole column implementation here.

    Args:
        kind: Which reduction.

    Returns:
        True for the five that `agg.mojo` covers plus the two that are counts.
    """
    return (
        kind == AggKind.SUM
        or kind == AggKind.MEAN
        or kind == AggKind.MIN
        or kind == AggKind.MAX
        or kind == AggKind.COUNT
        or kind == AggKind.SIZE
    )


def _reports_a_row(kind: AggKind) -> Bool:
    """Reports whether this reduction answers with a value the column held.

    Args:
        kind: Which reduction.

    Returns:
        True for the four that pick a row rather than computing something out of
        several of them.
    """
    return (
        kind == AggKind.MIN
        or kind == AggKind.MAX
        or kind == AggKind.FIRST
        or kind == AggKind.LAST
    )


def _reduce_text(col: StringArray, kind: AggKind) raises -> AnyArray:
    """Reduces a text column to the one element the reduction names.

    All four of these answer with a row of the column, so all four are the same
    two steps: find the row, then copy that one element out. The finding is in
    `agg.mojo` beside the numeric extremes, because it is the numeric extreme
    with a row number where the accumulator used to be.

    What this replaces is a trip through `aggregate_group_any` with a code per
    row that is always zero. On the hits table that is four hundred megabytes of
    codes allocated and zeroed so that a scan can read them and scatter into a
    table of one entry, and ClickBench q21, q22 and q28 all ask for exactly this.

    Args:
        col: The column.
        kind: One of MIN, MAX, FIRST or LAST.

    Returns:
        A text column of exactly one row, null if every input row was null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var row: Int
    if kind == AggKind.MIN:
        row = text_extreme_row[want_min=True](col)
    elif kind == AggKind.MAX:
        row = text_extreme_row[want_min=False](col)
    else:
        row = text_edge_row(col, kind == AggKind.FIRST)

    var builder = StringBuilder(capacity=1)
    if row < 0:
        builder.append_null()
    else:
        builder.append(col.unsafe_bytes(row))
    return AnyArray(builder^.finish())


def reduce_any(col: AnyArray, kind: AggKind) raises -> AnyArray:
    """Reduces a column to a single row.

    Args:
        col: The column.
        kind: Which reduction.

    Returns:
        A column of exactly one row, in the dtype the grouped reduction would
        have produced for the same kind.

    Raises:
        If the dtype has no physical layout, or if the reduction is one no path
        here implements for this column.
    """
    if kind == AggKind.SIZE:
        var sized = Array[DType.int64](1)
        sized[0] = Int64(len(col))
        return AnyArray(sized^)

    if kind == AggKind.COUNT:
        # The one reduction that works on a column of strings on the same line as
        # a column of numbers, since counting what is there needs no order and no
        # arithmetic. It used to need no values either and was a subtraction of
        # two numbers the column already knew. On a float column it is a scan
        # now, because a NaN is missing and the validity bitmap does not know
        # that. See #170.
        var counted = Array[DType.int64](1)
        counted[0] = Int64(len(col) - missing_count_any(col))
        return AnyArray(counted^)

    if kind == AggKind.NUNIQUE:
        var distinct = Array[DType.int64](1)
        distinct[0] = Int64(distinct_count_any(col))
        return AnyArray(distinct^)

    if col.type.is_temporal():
        return _reduce_temporal(col, kind)

    if col.is_string() and _reports_a_row(kind):
        return _reduce_text(col.strings(), kind)

    # As in `aggregate_group_any`: uint8 is in ALL, so a string column would
    # match it and a sum over a column of names would return a number taken from
    # the first byte of every view rather than an error. The grouped path already
    # knows what to do with strings, so send them there.
    if not col.is_string() and _takes_fast_route(kind):
        comptime for candidate in ALL:
            if col.dtype() == candidate:
                return _reduce_core(
                    col.unsafe_ptr[candidate](),
                    col.data.validity,
                    len(col) - col.null_count(),
                    len(col),
                    kind,
                )

    var codes = Array[DType.uint32](len(col))
    return aggregate_group_any(col, kind, codes^, 1, trusted=True)


comptime DISTINCT_SHARE = 8
"""Slots per row of the widest bit set `distinct_count` will take.

`factorize` accepts a direct table of a quarter of a slot per row, and says why
in `DIRECT_SHARE`: a slot there is four bytes, so that bound is one byte a row,
which is a quarter of what the ordinals it hands back already cost and cannot be
a surprise in anybody's memory budget.

A slot here is one bit, because counting distinct values only needs to know a
value has been seen and never needs to name it. So eight slots a row is the same
one byte a row, and the bound is the same promise in different units, which lets
the direct route cover thirty two times the range.

What it is being compared against is the hash route rather than a factorize.
That route allocates nothing per row either, so this is no longer the difference
between a bit set and four bytes a row; it is a bit set against a table, and the
bit set wins on every column whose range this bound accepts.
"""


def distinct_count_any(col: AnyArray) raises -> Int:
    """Counts the distinct values in a column, skipping the nulls.

    The whole column spelling of `nunique`. Asking the group by for it means
    allocating a code per row, sorting a copy of every value into a slab and
    walking the runs, which is `_nunique_core` doing exactly the right thing for
    a question nobody asked: it is counting distinct values in each of a million
    groups, and here there is one.

    What is left when there is one group is a question the hash layer answers on
    the way past. A factorize hands out an ordinal per distinct value, and the
    number it handed out is the answer, so the count is the part of a factorize
    that gets thrown away everywhere else. `distinct_hashed` is that part on its
    own, with the ordinals never written and the table the only allocation. An
    integer column with a bounded range does not need even the table, because a
    bit per possible value and a population count is the answer.

    Nulls are not a value, which is pandas' rule for `nunique` and the rule the
    grouped form here already follows. An empty string is a value. Those two
    sentences are one line apart in the code and a dataset that spells its
    missing text as an empty string, which the ClickBench hits table does, turns
    the difference into a wrong answer rather than a debate.

    Args:
        col: The column.

    Returns:
        How many distinct non-null values it holds.

    Raises:
        If the dtype has no physical layout.
    """
    if len(col) == 0:
        return 0
    # Before the numeric dispatch, because uint8 is in ALL and a string column
    # would match it and count distinct first bytes.
    if col.is_string():
        return distinct_strings(col.strings())

    comptime for candidate in ALL:
        if col.dtype() == candidate:
            return distinct_count(col.as_typed_view[candidate]())
    raise Error("nunique: unsupported dtype")


def distinct_count[dt: DType](col: Array[dt]) raises -> Int:
    """Counts the distinct non-null values in a typed column.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.

    Returns:
        How many distinct non-null values it holds.

    Raises:
        If one of the workers the count starts cannot be run.
    """
    comptime if dt.is_integral():
        var ceiling = len(col) * DISTINCT_SHARE
        if ceiling < DIRECT_LIMIT:
            ceiling = DIRECT_LIMIT
        var plan = direct_plan[dt](col, ceiling)
        if plan.span >= 0:
            return _distinct_direct(col, plan.span, plan.base)
    return distinct_hashed(col)


def _distinct_direct[
    dt: DType
](col: Array[dt], span: Int, base: Scalar[dt]) raises -> Int:
    """Counts distinct integers by setting a bit per value and counting them.

    No table, no ordinals and no second pass over anything the size of the
    column. What it costs is `span` bits, which `DISTINCT_SHARE` bounds at one
    byte a row, and one random bit write per row, which is the same random
    access the hash route was going to make anyway with a quarter of the
    footprint and none of the probing.

    It is serial. The parallel shape is a bit set per worker and an `or` at the
    end, and it was not worth writing yet: the write is one instruction into a
    region that fits in cache for every column this route accepts, so what bounds
    the loop is reading the values, and reading them on more cores is a change to
    make when a measurement asks for it.

    Args:
        col: The column.
        span: How many slots the plan says the range needs.
        base: The value that indexes slot zero, which is the column's minimum.

    Parameters:
        dt: The column's dtype.

    Returns:
        How many distinct non-null values it holds.
    """
    var seen = Bitmap(span, all_valid=False)
    var values = col.unsafe_ptr()
    if col.null_count() == 0:
        for i in range(len(col)):
            seen.set(Int(values.unsafe_offset(i).unsafe_load() - base), True)
    else:
        ref validity = col.data.validity
        for i in range(len(col)):
            if validity.get(i):
                seen.set(
                    Int(values.unsafe_offset(i).unsafe_load() - base), True
                )
    return seen.count_ones()


def _reduce_temporal(col: AnyArray, kind: AggKind) raises -> AnyArray:
    """Reduces a column of instants, dates or elapsed times.

    The arithmetic is the arithmetic the loops below already do, because all
    three types are an integer count underneath and the smallest of a set of
    counts is the smallest of the instants they stand for. What is different is
    the answer type. `_reduce_core` builds its result out of the physical dtype
    and would hand back an int64, so the extreme of a timestamp column would
    stop being a time on the way out. Every branch here therefore ends by
    putting the column's own type back on the result.

    Which reductions exist at all, and what each of them answers, is
    `temporal_agg_type` and is pandas' answer rather than a choice. The table
    lives beside the grouped path rather than here, because the two have to
    agree and reading the same thing is the only way to be sure they do.

    The mean is the one that costs something. pandas computes it in float64 and
    then truncates toward zero at both signs, so the mean of one and two seconds
    is one second and the mean of minus two and minus one is minus one second,
    and above 2**53 in the column's own unit the float has run out of mantissa
    and the answer drifts. The same route is taken here on purpose. Doing the
    division in int64 would be more accurate and would disagree with pandas on
    columns nobody has, which is the trade `binary.mojo` makes at the top of its
    own file and the same trade is right here.

    The reductions with no whole column spelling fall through to the grouped
    path with one group, exactly as a column of numbers does, and get their
    label there. Nothing about a column of times changes that route.

    Args:
        col: The column.
        kind: Which reduction.

    Returns:
        A column of one row, carrying whatever type the table says the answer
        has, which is usually but not always the column's own.

    Raises:
        If pandas has no answer for this reduction on this type.
    """
    # Asking the table first is what makes a refusal a refusal rather than a
    # missing case. `temporal_agg_type` raises here for the reductions pandas
    # has no answer for, so the fall through below never reaches the grouped
    # path carrying one of them, which matters because the grouped path reads
    # the same table with `whole_column=False` and would answer a standard error
    # that pandas refuses for a whole column.
    var wanted = temporal_agg_type(col.type, kind, whole_column=True)

    if _takes_fast_route(kind):
        comptime for candidate in ALL:
            if col.dtype() == candidate:
                var raw = _reduce_core(
                    col.unsafe_ptr[candidate](),
                    col.data.validity,
                    len(col) - col.null_count(),
                    len(col),
                    kind,
                )
                if not wanted.is_temporal():
                    return raw^
                return retag_temporal(raw^, wanted)
        raise Error("reduce: unsupported dtype")

    var codes = Array[DType.uint32](len(col))
    return aggregate_group_any(col, kind, codes^, 1, trusted=True)


def _reduce_core[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    present: Int,
    n: Int,
    kind: AggKind,
) raises -> AnyArray:
    """Runs one whole column reduction. One instantiation per dtype.

    Args:
        source: The values.
        validity: Which of them are present.
        present: How many of them are present.
        n: How many there are.
        kind: Which reduction.

    Parameters:
        dt: The value dtype.
        origin: Where the values live.

    Returns:
        A column of one row.

    Raises:
        If the kind is not one this function was told it handles, which would
        mean `_takes_fast_route` and this disagree.
    """
    if kind == AggKind.SUM:
        comptime acc = accumulator(dt)
        var summed = Array[acc](1)
        # A sum is always valid, including over an empty column and over a column
        # that is entirely null, where it is zero. `agg.mojo` explains why that
        # is pandas' answer and why disagreeing with it would be worse.
        summed[0] = sum_over(source, n).value
        return AnyArray(summed^)

    if kind == AggKind.MEAN:
        var averaged = Array[DType.float64](1)
        var mean = mean_over(source, present, n)
        # A mean always answers in float64, so this one is always the NaN branch
        # of `_place`. The two extremes below keep the column's own dtype and get
        # whichever branch that dtype has.
        _place(averaged, Float64(mean.value), mean.valid)
        return AnyArray(averaged^)

    if kind == AggKind.MIN:
        var smallest = Array[dt](1)
        var low = extreme_over[want_min=True](source, validity, n)
        _place(smallest, low.value, low.valid)
        return AnyArray(smallest^)

    if kind == AggKind.MAX:
        var largest = Array[dt](1)
        var high = extreme_over[want_min=False](source, validity, n)
        _place(largest, high.value, high.valid)
        return AnyArray(largest^)

    raise Error("reduce: unsupported aggregation")


def _place[dt: DType](mut out: Array[dt], value: Scalar[dt], valid: Bool):
    """Writes the one row of a result, or says it found nothing.

    How it says so depends on the dtype, because pandas has two spellings and
    only one of them is available in each. A float column takes a NaN and stays
    valid, since that is the only missing a pandas float column has. Every other
    dtype takes a zero behind a cleared validity bit, since there is no NaN to
    write. See #170.

    A missing row of the second kind still holds a zero rather than whatever the
    reduction was carrying when it gave up, because every null in the package
    holds a zero and a minimum that left negative infinity behind the validity
    bit would be the one exception.

    Args:
        out: The one row column.
        value: The reduced value.
        valid: Whether the reduction found anything.
    """
    if valid:
        out[0] = value
        return
    comptime if dt.is_floating_point():
        out[0] = nan[dt]()
        return
    out[0] = Scalar[dt](0)
    out.data.validity.set(0, False)
