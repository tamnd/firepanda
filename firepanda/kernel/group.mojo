"""Reductions that produce one value per group instead of one per column.

`firepanda/kernel/agg.mojo` reduces a column to a scalar. This reduces a column
to an array, indexed by a group ordinal that something else assigned. The
something else is `firepanda.hash.factorize`, and the split matters: a group by on
three columns factorizes once and aggregates six times, so fusing the two would
repeat the expensive half five times over.

Every function here takes the same three things. A values column, a `codes`
column holding one ordinal per row, and the number of distinct ordinals.

## Why these are scatter loops and not gathers

The obvious implementation of a grouped sum is a loop per group over the rows
belonging to it, which reads each group's rows contiguously. Getting there needs
the rows sorted by group first, and a sort costs more than the aggregation it
would be speeding up. So these go the other way: one pass over the rows in their
original order, each accumulating into `out[code]`. The working set is the
accumulator array, which is `groups` wide rather than `rows` wide, and for the
cardinalities a group by actually has it stays in cache. When it does not, the
fix is radix partitioning the rows first, which is what `firepanda/hash/
partition.mojo` exists for and which belongs to the executor rather than here.

## What a group with nothing in it produces

pandas is not consistent here and it has reasons, so this copies it rather than
inventing something tidier. A sum over a group with no non-null values is zero
and not null, because `min_count` defaults to zero. A count is zero. A minimum, a
maximum, a mean, a first and a last are all null, because there is no value to
report and zero would be a lie. `size` is the odd one out and counts rows rather
than values, nulls included, which is the only way to ask how big a group is.

## The null-is-zero invariant pays off twice here

A null holds a zero in the values buffer, so `group_sum` never reads the validity
bitmap at all: adding the nulls in adds zero. Count, min, max, first and last do
read it, because for those a null is not a neutral element. That is the same
split as in `agg.mojo` and for the same reason.

## Two spellings, one body

Same arrangement as `select.mojo` and for the same reason. A `DataFrame`
aggregates columns whose dtypes are runtime values, so it needs
`aggregate_group_any`; a caller who already knows the dtype should not pay a
dispatch chain for it. Both spellings resolve to the same pointer level core, so
there is one loop per reduction rather than two, and the twin in `scalar.mojo`
covers both.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL
from firepanda.kernel.accum import accumulator, highest, lowest
from firepanda.kernel.agg import max_of


@fieldwise_init
struct AggKind(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Which reduction a grouped aggregation should run.

    This is a runtime tag rather than eight separate erased entry points because
    the erased path instantiates its body once per dtype. Eight entry points over
    twelve dtypes would be ninety six instantiations of one loop; one entry point
    carrying a tag is twelve instantiations of eight loops, which is the same
    code with one dispatch chain instead of eight.
    """

    var code: UInt8

    comptime SUM = Self(0)
    """Adds the non-null values. Zero for a group with none."""

    comptime MEAN = Self(1)
    """Sum over non-null count. Null for a group with no non-null values."""

    comptime MIN = Self(2)
    """The smallest non-null value, or null."""

    comptime MAX = Self(3)
    """The largest non-null value, or null."""

    comptime COUNT = Self(4)
    """How many non-null values the group has."""

    comptime FIRST = Self(5)
    """The first non-null value in row order, or null."""

    comptime LAST = Self(6)
    """The last non-null value in row order, or null."""

    comptime SIZE = Self(7)
    """How many rows the group has, nulls included."""

    def __eq__(self, other: Self) -> Bool:
        """Compares two kinds.

        Args:
            other: The kind to compare against.

        Returns:
            True if they are the same reduction.
        """
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        """Compares two kinds for inequality.

        Args:
            other: The kind to compare against.

        Returns:
            True if they are different reductions.
        """
        return self.code != other.code

    def counts_rows(self) -> Bool:
        """Reports whether this reduction ignores the values entirely.

        Returns:
            True for `SIZE`, which needs only the codes.
        """
        return self == Self.SIZE

    def result_dtype(self, dt: DType) -> DType:
        """Returns the dtype this reduction produces over a column of `dt`.

        Args:
            dt: The input column's dtype.

        Returns:
            The dtype the reduction produces, which is int64 for the two
            counts, float64 for a mean, the accumulator dtype for a sum, and
            `dt` itself for the four that report a value the column held.
        """
        if self == Self.COUNT or self == Self.SIZE:
            return DType.int64
        if self == Self.MEAN:
            return DType.float64
        if self == Self.SUM:
            return accumulator(dt)
        return dt

    def write_to(self, mut writer: Some[Writer]):
        """Writes the name a user would recognise.

        Args:
            writer: The sink.
        """
        if self == Self.SUM:
            writer.write("sum")
        elif self == Self.MEAN:
            writer.write("mean")
        elif self == Self.MIN:
            writer.write("min")
        elif self == Self.MAX:
            writer.write("max")
        elif self == Self.COUNT:
            writer.write("count")
        elif self == Self.FIRST:
            writer.write("first")
        elif self == Self.LAST:
            writer.write("last")
        else:
            writer.write("size")


def group_size(codes: Array[DType.uint32], groups: Int) -> Array[DType.int64]:
    """Counts the rows in each group, nulls included.

    The only reduction that does not look at a values column at all, which is why
    it takes none.

    Args:
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Returns:
        A column of `groups` counts, every one of them present.
    """
    var out = Array[DType.int64](groups)
    var totals = out.unsafe_ptr()
    var at = codes.unsafe_ptr()
    for i in range(len(codes)):
        var g = Int(at.unsafe_offset(i).unsafe_load())
        totals.unsafe_offset(g).unsafe_store(
            totals.unsafe_offset(g).unsafe_load() + 1
        )
    return out^


def group_count[
    dt: DType
](values: Array[dt], codes: Array[DType.uint32], groups: Int) -> Array[
    DType.int64
]:
    """Counts the non-null values in each group.

    Args:
        values: The column being aggregated.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Parameters:
        dt: The column's dtype.

    Returns:
        A column of `groups` counts, every one of them present. A group whose
        values are all null counts zero rather than reporting null.
    """
    return _count_core(
        values.data.validity, values.null_count() > 0, codes, groups
    )


def _count_core(
    validity: Bitmap,
    has_null: Bool,
    codes: Array[DType.uint32],
    groups: Int,
) -> Array[DType.int64]:
    """Counts the present rows per group. Needs no dtype and reads no values."""
    if not has_null:
        return group_size(codes, groups)

    var out = Array[DType.int64](groups)
    var totals = out.unsafe_ptr()
    var at = codes.unsafe_ptr()
    for i in range(len(codes)):
        if validity.get(i):
            var g = Int(at.unsafe_offset(i).unsafe_load())
            totals.unsafe_offset(g).unsafe_store(
                totals.unsafe_offset(g).unsafe_load() + 1
            )
    return out^


def group_sum[
    dt: DType
](values: Array[dt], codes: Array[DType.uint32], groups: Int) -> Array[
    accumulator(dt)
]:
    """Adds up the non-null values in each group.

    The nulls are added too. They hold zero, so it makes no difference to the
    answer and it saves a bitmap read per row. That is the invariant stated in
    `firepanda/kernel/__init__.mojo` being spent.

    Args:
        values: The column being aggregated.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Parameters:
        dt: The column's dtype.

    Returns:
        A column of `groups` sums in the accumulator dtype, every one present. A
        group with no non-null values sums to zero, matching pandas at its
        default `min_count` of zero.
    """
    return _sum_core(values.unsafe_ptr(), codes, groups)


def _sum_core[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    codes: Array[DType.uint32],
    groups: Int,
) -> Array[accumulator(dt)]:
    """Accumulates every row into its group, validity ignored on purpose."""
    comptime acc = accumulator(dt)
    var out = Array[acc](groups)
    var totals = out.unsafe_ptr()
    var at = codes.unsafe_ptr()
    for i in range(len(codes)):
        var g = Int(at.unsafe_offset(i).unsafe_load())
        totals.unsafe_offset(g).unsafe_store(
            totals.unsafe_offset(g).unsafe_load()
            + source.unsafe_offset(i).unsafe_load().cast[acc]()
        )
    return out^


def group_mean[
    dt: DType
](values: Array[dt], codes: Array[DType.uint32], groups: Int) -> Array[
    DType.float64
]:
    """Averages the non-null values in each group.

    The divisor is the non-null count and not the group size, which is what
    pandas does and is why a group holding `[1, null, 3]` means 2 rather than
    1.33.

    Args:
        values: The column being aggregated.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Parameters:
        dt: The column's dtype.

    Returns:
        A column of `groups` means. A group with no non-null values is null,
        because zero would be a value the group does not have.
    """
    return _mean_core(
        values.unsafe_ptr(),
        values.data.validity,
        values.null_count() > 0,
        codes,
        groups,
    )


def _mean_core[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_null: Bool,
    codes: Array[DType.uint32],
    groups: Int,
) -> Array[DType.float64]:
    """Divides a grouped sum by a grouped count, one group at a time."""
    var sums = _sum_core(source, codes, groups)
    var counts = _count_core(validity, has_null, codes, groups)

    var out = Array[DType.float64](groups)
    var target = out.unsafe_ptr()
    var total = sums.unsafe_ptr()
    var n = counts.unsafe_ptr()
    for g in range(groups):
        var count = n.unsafe_offset(g).unsafe_load()
        if count == 0:
            out.data.validity.set(g, False)
            continue
        target.unsafe_offset(g).unsafe_store(
            total.unsafe_offset(g).unsafe_load().cast[DType.float64]()
            / Float64(count)
        )
    return out^


def group_min[
    dt: DType
](values: Array[dt], codes: Array[DType.uint32], groups: Int) -> Array[dt]:
    """Returns the smallest non-null value in each group.

    Args:
        values: The column being aggregated.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Parameters:
        dt: The column's dtype.

    Returns:
        A column of `groups` minima, null for any group with no non-null values.
    """
    return _extreme_core[want_min=True](
        values.unsafe_ptr(),
        values.data.validity,
        values.null_count() > 0,
        codes,
        groups,
    )


def group_max[
    dt: DType
](values: Array[dt], codes: Array[DType.uint32], groups: Int) -> Array[dt]:
    """Returns the largest non-null value in each group.

    Args:
        values: The column being aggregated.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Parameters:
        dt: The column's dtype.

    Returns:
        A column of `groups` maxima, null for any group with no non-null values.
    """
    return _extreme_core[want_min=False](
        values.unsafe_ptr(),
        values.data.validity,
        values.null_count() > 0,
        codes,
        groups,
    )


def _extreme_core[
    dt: DType, //, origin: ImmOrigin, want_min: Bool
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_null: Bool,
    codes: Array[DType.uint32],
    groups: Int,
) -> Array[dt]:
    """Reduces each group to its smallest or largest non-null value.

    Min and max differ by one comparison, so they share a body on the same terms
    as `_extreme` in `agg.mojo`.

    Every accumulator starts at the reduction's identity, so the loop has no
    per-row branch on whether the group has been seen yet. Which groups were seen
    is carried in the output's validity bitmap, which had to be maintained
    anyway, so the seen flags cost nothing beyond the bit that was already there.
    """
    comptime identity = highest[dt]() if want_min else lowest[dt]()

    var out = Array[dt](groups)
    var best = out.unsafe_ptr()
    out.data.validity.clear_all()
    for g in range(groups):
        best.unsafe_offset(g).unsafe_store(identity)

    var at = codes.unsafe_ptr()
    for i in range(len(codes)):
        if has_null and not validity.get(i):
            continue
        var g = Int(at.unsafe_offset(i).unsafe_load())
        var value = source.unsafe_offset(i).unsafe_load()
        var current = best.unsafe_offset(g).unsafe_load()
        comptime if want_min:
            if value < current:
                best.unsafe_offset(g).unsafe_store(value)
        else:
            if value > current:
                best.unsafe_offset(g).unsafe_store(value)
        out.data.validity.set(g, True)

    # A group that was never seen still holds the identity, which is a real value
    # of the dtype and would read as a maximum of negative infinity. Zero it, so
    # that a null holds a zero the way every other null in the package does.
    for g in range(groups):
        if not out.data.validity.get(g):
            best.unsafe_offset(g).unsafe_store(Scalar[dt](0))
    return out^


def group_first[
    dt: DType
](values: Array[dt], codes: Array[DType.uint32], groups: Int) -> Array[dt]:
    """Returns each group's first non-null value in row order.

    pandas skips nulls here rather than reporting the literal first row, so a
    group whose first row is null reports its second. That is `first()` and not
    `nth(0)`, and the two differ only on data with nulls in it, which is exactly
    the data where it matters.

    Args:
        values: The column being aggregated.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Parameters:
        dt: The column's dtype.

    Returns:
        A column of `groups` values, null for any group with no non-null values.
    """
    return _edge_core[want_first=True](
        values.unsafe_ptr(),
        values.data.validity,
        values.null_count() > 0,
        codes,
        groups,
    )


def group_last[
    dt: DType
](values: Array[dt], codes: Array[DType.uint32], groups: Int) -> Array[dt]:
    """Returns each group's last non-null value in row order.

    Args:
        values: The column being aggregated.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Parameters:
        dt: The column's dtype.

    Returns:
        A column of `groups` values, null for any group with no non-null values.
    """
    return _edge_core[want_first=False](
        values.unsafe_ptr(),
        values.data.validity,
        values.null_count() > 0,
        codes,
        groups,
    )


def _edge_core[
    dt: DType, //, origin: ImmOrigin, want_first: Bool
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_null: Bool,
    codes: Array[DType.uint32],
    groups: Int,
) -> Array[dt]:
    """Takes each group's first or last non-null value.

    First walks the rows forwards and writes only into groups it has not filled;
    last walks them backwards and does the same. Both are one loop with the range
    reversed, which beats last's other option of writing on every row and letting
    the final write win, because that writes once per row where this writes once
    per group. The early exit once every group is filled is what makes a `first`
    over a low cardinality column cost almost nothing.
    """
    var out = Array[dt](groups)
    var target = out.unsafe_ptr()
    out.data.validity.clear_all()

    var at = codes.unsafe_ptr()
    var rows = len(codes)
    var filled = 0
    for step in range(rows):
        var i = step if want_first else rows - 1 - step
        if has_null and not validity.get(i):
            continue
        var g = Int(at.unsafe_offset(i).unsafe_load())
        if out.data.validity.get(g):
            continue
        target.unsafe_offset(g).unsafe_store(
            source.unsafe_offset(i).unsafe_load()
        )
        out.data.validity.set(g, True)
        filled += 1
        if filled == groups:
            break
    return out^


def aggregate_group[
    dt: DType
](
    values: Array[dt], kind: AggKind, codes: Array[DType.uint32], groups: Int
) raises -> AnyArray:
    """Runs one grouped reduction, chosen at runtime, over a typed column.

    Args:
        values: The column being aggregated.
        kind: Which reduction.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Parameters:
        dt: The column's dtype.

    Returns:
        A column of `groups` values in whatever dtype the reduction produces.

    Raises:
        If the reduction is not one of the eight.
    """
    return _dispatch_core(
        values.unsafe_ptr(),
        values.data.validity,
        values.null_count() > 0,
        kind,
        codes,
        groups,
    )


def _dispatch_core[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_null: Bool,
    kind: AggKind,
    codes: Array[DType.uint32],
    groups: Int,
) raises -> AnyArray:
    """Picks the reduction. One instantiation per dtype, eight loops inside."""
    if kind == AggKind.SIZE:
        return AnyArray(group_size(codes, groups))
    if kind == AggKind.COUNT:
        return AnyArray(_count_core(validity, has_null, codes, groups))
    if kind == AggKind.SUM:
        return AnyArray(_sum_core(source, codes, groups))
    if kind == AggKind.MEAN:
        return AnyArray(_mean_core(source, validity, has_null, codes, groups))
    if kind == AggKind.MIN:
        return AnyArray(
            _extreme_core[want_min=True](
                source, validity, has_null, codes, groups
            )
        )
    if kind == AggKind.MAX:
        return AnyArray(
            _extreme_core[want_min=False](
                source, validity, has_null, codes, groups
            )
        )
    if kind == AggKind.FIRST:
        return AnyArray(
            _edge_core[want_first=True](
                source, validity, has_null, codes, groups
            )
        )
    if kind == AggKind.LAST:
        return AnyArray(
            _edge_core[want_first=False](
                source, validity, has_null, codes, groups
            )
        )
    raise Error("group by: unsupported aggregation")


def aggregate_group_any(
    col: AnyArray, kind: AggKind, codes: Array[DType.uint32], groups: Int
) raises -> AnyArray:
    """Runs one grouped reduction over a column whose dtype is a runtime value.

    This is the entry point a `DataFrame` calls, and it is the one that checks
    the codes rather than trusting them. The check is once per call and not once
    per row: it takes the maximum ordinal, which is one pass over a column the
    group by has already touched, and compares it against the group count. The
    typed spellings above skip it, because their codes come from `factorize` in
    the same expression that produced them.

    Args:
        col: The column being aggregated.
        kind: Which reduction.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Returns:
        A column of `groups` values.

    Raises:
        If the dtype has no physical layout, if the codes and the column are
        different lengths, or if a code names a group that does not exist.
    """
    if groups < 0:
        raise Error("group by: the group count cannot be negative")
    if not kind.counts_rows() and len(codes) != len(col):
        raise Error(
            "group by: the codes and the column must be the same length; codes"
            " has "
            + String(len(codes))
            + " rows and the column has "
            + String(len(col))
        )
    _check_codes(codes, groups)

    if kind == AggKind.SIZE:
        return AnyArray(group_size(codes, groups))

    comptime for candidate in ALL:
        if col.dtype() == candidate:
            return _dispatch_core(
                col.unsafe_ptr[candidate](),
                col.data.validity,
                col.null_count() > 0,
                kind,
                codes,
                groups,
            )
    raise Error("group by: unsupported dtype")


def _check_codes(codes: Array[DType.uint32], groups: Int) raises:
    """Raises if any ordinal names a group that was never allocated.

    `max_of` rather than a loop over the codes. This runs on every erased call and
    a scalar scan of a million of them measured at roughly 350 us on the reference
    machine, which was most of the gap between `group/sum` and
    `group/sum_dispatched` and was more than the reduction it was guarding.
    """
    if len(codes) == 0:
        return
    var top = max_of(codes)
    if top.valid and Int(top.value) >= groups:
        raise Error(
            "group by: code "
            + String(top.value)
            + " names a group outside the "
            + String(groups)
            + " that exist"
        )
