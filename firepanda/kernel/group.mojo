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

## Why a big one runs on every core and a small one does not

A scatter cannot be split by handing each worker a slice of the rows and letting
them all write to the same table, because two rows in two slices can belong to
the same group and the increments would be lost. So each worker gets its own
table and the tables are added together afterwards, which is the same shape
`firepanda/hash/factorize.mojo` uses for the same reason.

That trade has a ceiling in it, and the ceiling is why `PRIVATE_BYTES` exists.
The merge costs `groups * workers` rather than `rows`, so at a hundred groups it
is nothing and at a million it is most of the work. Past the ceiling the
reduction stays on one core, and the answer for that case is partitioning the
rows by code so that each worker owns a range of groups outright and there is no
merge at all. That is a change to `firepanda/hash/partition.mojo` and the
executor rather than to this file.

Ten million float64 rows on an i9-13900K, medians of five: a sum over a hundred
groups is 5.4 ms on one core and 1.9 ms on all of them, a mean is 8.5 ms against
2.8 ms, a maximum is 8.1 ms against 2.5 ms. At a million groups, where the
ceiling holds it to four workers, a sum is 16.4 ms against 10.0 ms.

## What a group with nothing in it produces

pandas is not consistent here and it has reasons, so this copies it rather than
inventing something tidier. A sum over a group with no non-null values is zero
and not null, because `min_count` defaults to zero. A count is zero. `size` is
the odd one out and counts rows rather than values, nulls included, which is the
only way to ask how big a group is.

Everything else reports that it found nothing, and the spelling depends on the
dtype it answers in rather than on which reduction it is. A float column says it
with a NaN in a row that stays valid. Any other dtype says it with a null, which
holds a zero the way every null in the package does.

That is not two rules, it is one rule and a dtype that does not have both
options. pandas has no separate presence bitmap for a float column and NaN is the
only missing it has there, so NaN is not a second spelling invented for variety,
it is what missing looks like. An int64 column has no NaN available to write, so
a null is the only spelling there is. firepanda used to report a null in both
cases and the difference was visible at the API boundary, which is #170.

Sorting the reductions by which branch they take: a mean, a variance, a
deviation, a standard error, a skewness, a median, a quantile, a correlation and
a covariance always answer in float64 and so always take the NaN. A minimum, a
maximum, a first and a last keep the column's own dtype, so they take the NaN on
a float column and the null on every other one.

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

from std.collections.span import Span
from std.math import nan, sqrt
from std.sys.info import simd_width_of, size_of

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.lists import ALL
from firepanda.exec.parallel import parallel_for, worker_count
from firepanda.hash.factorize import factorize_strings
from firepanda.kernel.accum import accumulator, highest, lowest
from firepanda.kernel.agg import max_of
from firepanda.kernel.cast import cast_any

comptime PRIVATE_BYTES = 32 * 1024 * 1024
"""How much memory the private accumulator tables may take, in total.

The tables are the whole cost of the parallel route and the whole reason it has
a ceiling. One per worker means the memory is `groups * workers * width`, so at
thirty two workers and eight byte accumulators a hundred thousand groups is
twenty five megabytes and a million groups would be two hundred and fifty six.
Past a point the allocation and the merge cost more than the scatter they are
speeding up, and the answer for that case is partitioning the rows by code
rather than replicating the table, which is a separate change.

Thirty two megabytes is chosen so the tables are at worst a couple of times an
L3 cache rather than a fraction of main memory.
"""

comptime PRIVATE_ROWS = 1 << 16
"""Rows below which a grouped reduction stays on one core.

Sixty five thousand, which is half a morsel. A scatter of that many rows is a
couple of hundred microseconds, and starting a task per worker and merging their
tables afterwards is a fair fraction of that, so below here the parallel route is
break even at best and the serial one is the honest answer.
"""

comptime GROUP_BLOCK_ALIGN = 8
"""Groups a per group loop's pieces are cut on a multiple of.

Eight presence bits are a byte and eight float64 results are a cache line, so
this is the smallest number that keeps two workers from writing the same byte of
validity or fighting over the same line of output.
"""


def _private_workers[dt: DType](rows: Int, groups: Int) -> Int:
    """Decides how many private tables a grouped reduction should build.

    Args:
        rows: How many rows are being scattered.
        groups: How wide one table is.

    Parameters:
        dt: The accumulator dtype, which sets how wide a table entry is.

    Returns:
        The worker count, or one to say the reduction should stay serial.
    """
    if rows < PRIVATE_ROWS or groups <= 0:
        return 1

    var workers = worker_count()
    if workers <= 1:
        return 1

    var affordable = PRIVATE_BYTES // (groups * size_of[dt]())
    if affordable < 2:
        return 1
    return workers if workers < affordable else affordable


def _row_bounds(rows: Int, workers: Int) -> List[Int]:
    """Cuts a row range into one contiguous piece per worker.

    A static split rather than a morsel queue, because every row of a scatter
    costs the same and there is nothing for a queue to balance. What the split
    does buy is that each worker's piece is contiguous, so the values and the
    codes are read forwards.

    Args:
        rows: How many rows there are.
        workers: How many pieces to cut them into.

    Returns:
        `workers + 1` offsets, the first zero and the last `rows`.
    """
    var bounds = List[Int](capacity=workers + 1)
    for w in range(workers):
        bounds.append(rows * w // workers)
    bounds.append(rows)
    return bounds^


comptime PARTITION_BYTES = 256 * 1024
"""How much of an accumulator table one partition of a scatter may cover.

The partitioned route is what happens when `PRIVATE_BYTES` says a table per
worker will not fit. Instead of replicating the table it cuts the group range:
each partition owns a run of ordinals no other partition holds, so there is one
table rather than one per worker and there is no merge at all. What the partition
width has to satisfy is that its slice of the table stays in a core's private
cache while the partition is being folded, because every row of the partition is
a random write inside that slice. A quarter of a megabyte is around the L2 on the
machines this was measured on, which leaves the codes and the values being read
alongside to the rest of it.
"""

comptime PARTITION_ROWS = 1 << 18
"""Rows below which the partitioned route is not worth its two extra passes.

Partitioning reads the column and writes a copy of it before the fold reads that
copy back, so it is three passes over the data where the replicated route is one.
It wins anyway on a high cardinality key, because the scatter it replaces misses
cache on every single row and a miss moves a whole line in each direction. It
does not win on a small column, where the scatter fits in cache and there was
nothing to fix.
"""


def _partition_shift[dt: DType](groups: Int) -> Int:
    """Returns how many low bits of a group ordinal one partition covers.

    Args:
        groups: How many groups there are.

    Parameters:
        dt: The accumulator dtype, which sets how wide a table entry is.

    Returns:
        The shift, so that partition `p` owns ordinals `p << shift` upwards.
    """
    var span = PARTITION_BYTES // size_of[dt]()
    var shift = 0
    while (1 << (shift + 1)) <= span and (1 << (shift + 1)) <= groups:
        shift += 1
    return shift


def _partition_parts[dt: DType](rows: Int, groups: Int) -> Int:
    """Decides whether a scatter should be partitioned, and into how many pieces.

    The partitioned route is taken only where the replicated one has given up
    entirely, which is to say where the group count is large enough that a table
    per worker does not fit inside `PRIVATE_BYTES` even twice over. Between the
    two there is a band where replication still fits but only for a handful of
    workers, and which route wins there is a measurement nobody has taken, so
    this does not guess at it.

    Args:
        rows: How many rows are being scattered.
        groups: How many groups they land in.

    Parameters:
        dt: The accumulator dtype.

    Returns:
        The partition count, or zero to say the scatter should not be partitioned.
    """
    if rows < PARTITION_ROWS or groups <= 0:
        return 0
    if worker_count() <= 1:
        return 0
    if _private_workers[dt](rows, groups) > 1:
        return 0

    var shift = _partition_shift[dt](groups)
    var parts = (groups + (1 << shift) - 1) >> shift
    return parts if parts >= 2 else 0


def _partition_starts(
    codes: Array[DType.uint32],
    validity: Bitmap,
    skip_null: Bool,
    bounds: List[Int],
    workers: Int,
    parts: Int,
    shift: Int,
) raises -> List[Int]:
    """Counts each worker's rows per partition and turns them into write offsets.

    The counting table is worker major, `workers * parts` entries with worker
    `w`'s run contiguous, so that two workers counting at once are not writing
    into one cache line. The offsets that come out of it are partition major,
    because that is the order the rows have to end up in for a partition to be a
    contiguous run that one task can fold.

    Args:
        codes: One group ordinal per row.
        validity: Which rows are present.
        skip_null: True to leave null rows out of the partitioning entirely.
        bounds: `workers + 1` row offsets, one piece per worker.
        workers: How many pieces the rows are cut into.
        parts: How many partitions the groups are cut into.
        shift: How many low bits of an ordinal one partition covers.

    Returns:
        `parts * workers + 1` offsets. Worker `w`'s run of partition `p` starts
        at entry `p * workers + w`, and the last entry is the total row count
        that will be placed.

    Raises:
        If one of the counting workers cannot be run.
    """
    var tally = Buffer(workers * parts * 8)

    def count(w: Int) raises {mut tally, imm}:
        var mine = tally.bitcast[DType.int64]().unsafe_offset(w * parts)
        var at = codes.unsafe_ptr()
        for i in range(bounds[w], bounds[w + 1]):
            if skip_null and not validity.get(i):
                continue
            var p = Int(at.unsafe_offset(i).unsafe_load()) >> shift
            mine.unsafe_offset(p).unsafe_store(
                mine.unsafe_offset(p).unsafe_load() + 1
            )

    parallel_for(count, workers)

    var counts = tally.bitcast[DType.int64]()
    var starts = List[Int](capacity=parts * workers + 1)
    var running = 0
    for p in range(parts):
        for w in range(workers):
            starts.append(running)
            running += Int(counts.unsafe_offset(w * parts + p).unsafe_load())
    starts.append(running)
    return starts^


def _partitioned_sums[
    dt: DType, //, origin: ImmOrigin, acc: DType = accumulator(dt)
](
    source: Pointer[Scalar[dt], origin],
    codes: Array[DType.uint32],
    groups: Int,
    parts: Int,
) raises -> Array[acc]:
    """Adds every row into its group by cutting the group range, not the table.

    Three passes rather than one. The first counts how many rows each worker has
    for each partition, the second copies the ordinals and the values into
    partition order, and the third folds one partition at a time into the single
    output table. The rows of a partition can only touch that partition's run of
    the output, so the fold needs no locks, no private copies and no merge, and
    the run is small enough to sit in a core's cache while it is being written.

    Args:
        source: The values being added.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.
        parts: How many partitions to cut the groups into.

    Parameters:
        dt: The column's dtype.
        origin: The origin of the values pointer.
        acc: What to accumulate in. Defaults to int64, uint64 or float64.

    Returns:
        A column of `groups` sums, every one of them present.

    Raises:
        If one of the workers cannot be run.
    """
    var rows = len(codes)
    var workers = worker_count()
    var bounds = _row_bounds(rows, workers)
    var shift = _partition_shift[acc](groups)

    # Every row is placed, the null ones included, because this reduction adds
    # every row and a null holds a zero. That is the same invariant the serial
    # loop spends and it saves the placement a bitmap read per row.
    var starts = _partition_starts(
        codes, codes.data.validity, False, bounds, workers, parts, shift
    )

    var held = Buffer(overwritten=rows * 4)
    var carried = Buffer(overwritten=rows * size_of[acc]())

    def place(w: Int) raises {mut held, mut carried, imm}:
        var at = codes.unsafe_ptr()
        var ordinals = held.bitcast[DType.uint32]()
        var values = carried.bitcast[acc]()
        var cursor = List[Int](capacity=parts)
        for p in range(parts):
            cursor.append(starts[p * workers + w])
        for i in range(bounds[w], bounds[w + 1]):
            var g = at.unsafe_offset(i).unsafe_load()
            var p = Int(g) >> shift
            var slot = cursor[p]
            ordinals.unsafe_offset(slot).unsafe_write(g)
            values.unsafe_offset(slot).unsafe_write(
                source.unsafe_offset(i).unsafe_load().cast[acc]()
            )
            cursor[p] = slot + 1

    parallel_for(place, workers)

    var out = Array[acc](groups)

    def fold(p: Int) raises {mut out, imm}:
        var totals = out.unsafe_ptr()
        var ordinals = held.bitcast[DType.uint32]()
        var values = carried.bitcast[acc]()
        for i in range(starts[p * workers], starts[(p + 1) * workers]):
            var g = Int(ordinals.unsafe_offset(i).unsafe_load())
            totals.unsafe_offset(g).unsafe_store(
                totals.unsafe_offset(g).unsafe_load()
                + values.unsafe_offset(i).unsafe_load()
            )

    parallel_for(fold, parts)
    return out^


def _partitioned_tally(
    validity: Bitmap,
    skip_null: Bool,
    codes: Array[DType.uint32],
    groups: Int,
    parts: Int,
) raises -> Array[DType.int64]:
    """Counts rows per group by cutting the group range, not the table.

    The same three passes `_partitioned_sums` takes, without the values, because
    a count reads none. A null row is left out of the partitioning rather than
    skipped in the fold, so the copy is only as long as the answer needs.

    Args:
        validity: Which rows are present.
        skip_null: True to count only the present rows.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.
        parts: How many partitions to cut the groups into.

    Returns:
        A column of `groups` counts, every one of them present.

    Raises:
        If one of the workers cannot be run.
    """
    var rows = len(codes)
    var workers = worker_count()
    var bounds = _row_bounds(rows, workers)
    var shift = _partition_shift[DType.int64](groups)
    var starts = _partition_starts(
        codes, validity, skip_null, bounds, workers, parts, shift
    )

    var held = Buffer(overwritten=starts[parts * workers] * 4)

    def place(w: Int) raises {mut held, imm}:
        var at = codes.unsafe_ptr()
        var ordinals = held.bitcast[DType.uint32]()
        var cursor = List[Int](capacity=parts)
        for p in range(parts):
            cursor.append(starts[p * workers + w])
        for i in range(bounds[w], bounds[w + 1]):
            if skip_null and not validity.get(i):
                continue
            var g = at.unsafe_offset(i).unsafe_load()
            var slot = cursor[Int(g) >> shift]
            ordinals.unsafe_offset(slot).unsafe_write(g)
            cursor[Int(g) >> shift] = slot + 1

    parallel_for(place, workers)

    var out = Array[DType.int64](groups)

    def fold(p: Int) raises {mut out, imm}:
        var totals = out.unsafe_ptr()
        var ordinals = held.bitcast[DType.uint32]()
        for i in range(starts[p * workers], starts[(p + 1) * workers]):
            var g = Int(ordinals.unsafe_offset(i).unsafe_load())
            totals.unsafe_offset(g).unsafe_store(
                totals.unsafe_offset(g).unsafe_load() + 1
            )

    parallel_for(fold, parts)
    return out^


def _group_blocks(rows: Int, groups: Int) -> Int:
    """Decides how many pieces a per group loop should be cut into.

    A per group loop is not a scatter. Each group's values are already in their
    own run of the slab, so the pieces touch nothing in common and there are no
    private tables to pay for. What there is instead is imbalance, because the
    groups are not the same size, so this asks for four pieces per worker rather
    than one and lets the short ones finish early.

    Args:
        rows: How many rows went into the slab.
        groups: How many groups there are to walk.

    Returns:
        The number of pieces, or one to say the loop should stay serial.
    """
    if rows < PRIVATE_ROWS or groups < GROUP_BLOCK_ALIGN:
        return 1
    var blocks = worker_count() * 4
    var most = groups // GROUP_BLOCK_ALIGN
    return blocks if blocks < most else most


def _group_bounds(groups: Int, blocks: Int) -> List[Int]:
    """Cuts a group range into pieces that share no byte of the output.

    Every cut is rounded up to a multiple of eight. Eight groups' presence bits
    are one byte of the output validity, and clearing a bit is a read, a mask and
    a write, so two workers on the same byte would lose one of the two clears.
    Eight float64 results are also a cache line, so the same rounding keeps the
    value writes off each other's lines as well.

    Args:
        groups: How many groups there are.
        blocks: How many pieces to cut them into.

    Returns:
        `blocks + 1` offsets, the first zero and the last `groups`.
    """
    var bounds = List[Int](capacity=blocks + 1)
    for b in range(blocks):
        var cut = groups * b // blocks
        cut = ((cut + GROUP_BLOCK_ALIGN - 1) // GROUP_BLOCK_ALIGN) * (
            GROUP_BLOCK_ALIGN
        )
        bounds.append(cut if cut < groups else groups)
    bounds.append(groups)
    return bounds^


def _merge_sums[
    dt: DType
](partials: Array[dt], groups: Int, workers: Int) -> Array[dt]:
    """Adds the private tables together into one.

    This is the half of the parallel route that is not parallel, and it is
    vectorized because it can be: the tables are laid out one after another, so
    adding table `w` into the answer is a walk of two contiguous arrays and the
    scatter's dependent stores are gone. The cost is `groups * workers` adds
    against the `rows` the scatter did, which is why `PRIVATE_BYTES` is the
    thing that decides whether any of this was worth doing.

    Args:
        partials: One table per worker, laid end to end.
        groups: How wide one table is.
        workers: How many tables there are.

    Parameters:
        dt: The accumulator dtype.

    Returns:
        A column of `groups` totals, every one of them present.
    """
    comptime width = simd_width_of[dt]()

    var out = Array[dt](groups)
    var totals = out.unsafe_ptr()
    var tables = partials.unsafe_ptr()
    for w in range(workers):
        var table = tables.unsafe_offset(w * groups)
        var g = 0
        while g + width <= groups:
            totals.unsafe_offset(g).unsafe_store(
                totals.unsafe_offset(g).unsafe_load[width=width]()
                + table.unsafe_offset(g).unsafe_load[width=width]()
            )
            g += width
        while g < groups:
            totals.unsafe_offset(g).unsafe_store(
                totals.unsafe_offset(g).unsafe_load()
                + table.unsafe_offset(g).unsafe_load()
            )
            g += 1
    return out^


struct AggKind(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Which reduction a grouped aggregation should run.

    This is a runtime tag rather than a parameter because the erased path
    instantiates its body once per dtype. Thirteen single column entry points
    over twelve dtypes would be a hundred and fifty six instantiations of one
    loop; one entry point carrying a tag is twelve instantiations of thirteen
    loops, which is the same code with one dispatch chain instead of thirteen.
    `CORR` and `COV` read a pair of columns and so are not in that chain at all,
    for a reason `aggregate_group_pair_any` gives.

    Four of the reductions take a number as well as a name. `VAR` and `STD` take
    a delta degrees of freedom, `COV` takes one too, and `QUANTILE` takes the
    quantile itself, so the tag carries a `Float64` beside the code. Anything
    that does not use it leaves it at zero.

    Two kinds are equal when their codes are, and `param` is deliberately not
    part of that. `kind == AggKind.QUANTILE` has to be true for the ninetieth
    percentile as well as for the median, because that comparison is how the
    dispatch chain picks a branch and it is asking which reduction this is, not
    which arguments it was given.
    """

    var code: UInt8
    """Which reduction."""

    var param: Float64
    """The delta degrees of freedom for `VAR` and `STD`, the quantile for
    `QUANTILE` and `MEDIAN`, and zero for everything else."""

    def __init__(out self, code: UInt8):
        """Constructs a kind with the reduction's own default parameter.

        Args:
            code: Which reduction.
        """
        self.code = code
        self.param = Self._default_param(code)

    def __init__(out self, code: UInt8, param: Float64):
        """Constructs a kind with an explicit parameter.

        Args:
            code: Which reduction.
            param: The degrees of freedom or the quantile.
        """
        self.code = code
        self.param = param

    @staticmethod
    def _default_param(code: UInt8) -> Float64:
        """Returns the parameter a bare code should carry.

        Constructing from a code alone has to give the reduction its documented
        default rather than zero, because zero is a legal delta degrees of freedom
        and a legal quantile. A caller that writes `AggKind(UInt8(11))` and gets
        the minimum instead of the median has been handed a bug rather than an
        argument, so the default lives here rather than in a signature.

        Args:
            code: Which reduction.

        Returns:
            One for the two dispersions, the standard error and the
            covariance, a half for the two order statistics, and zero for the
            rest.
        """
        if code == 8 or code == 9 or code == 14 or code == 15:
            return 1.0
        if code == 10 or code == 11:
            return 0.5
        return 0.0

    @staticmethod
    def var_with(ddof: Int) -> Self:
        """Returns a variance with an explicit delta degrees of freedom.

        Args:
            ddof: Subtracted from the count to give the divisor.

        Returns:
            The kind.
        """
        return Self(8, Float64(ddof))

    @staticmethod
    def std_with(ddof: Int) -> Self:
        """Returns a standard deviation with an explicit delta degrees of freedom.

        Args:
            ddof: Subtracted from the count to give the divisor.

        Returns:
            The kind.
        """
        return Self(9, Float64(ddof))

    @staticmethod
    def sem_with(ddof: Int) -> Self:
        """Returns a standard error with an explicit delta degrees of freedom.

        Args:
            ddof: Subtracted from the count to give the variance divisor. The
                count under the square root is not adjusted, which is what
                pandas does.

        Returns:
            The kind.
        """
        return Self(15, Float64(ddof))

    @staticmethod
    def cov_with(ddof: Int) -> Self:
        """Returns a covariance with an explicit delta degrees of freedom.

        Args:
            ddof: Subtracted from the pairwise count to give the divisor.

        Returns:
            The kind.
        """
        return Self(14, Float64(ddof))

    @staticmethod
    def quantile_at(q: Float64) -> Self:
        """Returns a quantile at a given position.

        Args:
            q: Where in the sorted values to land, from zero to one.

        Returns:
            The kind.
        """
        return Self(11, q)

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

    comptime VAR = Self(8)
    """The variance of the non-null values, dividing by count minus `param`."""

    comptime STD = Self(9)
    """The square root of `VAR`, with the same divisor."""

    comptime MEDIAN = Self(10)
    """The middle of the non-null values, interpolating between two."""

    comptime QUANTILE = Self(11)
    """The value at position `param` of the sorted non-null values."""

    comptime NUNIQUE = Self(12)
    """How many distinct non-null values the group has."""

    comptime CORR = Self(13)
    """The Pearson correlation of two columns, over the rows where both are
    present. The only pair of kinds that reads a second column."""

    comptime COV = Self(14)
    """The covariance of two columns, dividing by the pairwise count minus
    `param`."""

    comptime SEM = Self(15)
    """The standard error of the mean: `STD` with the same divisor, over the
    square root of the non-null count."""

    comptime SKEW = Self(16)
    """The sample skewness of the non-null values, the adjusted Fisher Pearson
    coefficient pandas uses. Null for a group of fewer than three, and zero
    rather than null for a group whose values are all the same."""

    def __eq__(self, other: Self) -> Bool:
        """Compares two kinds, by reduction and not by parameter.

        Args:
            other: The kind to compare against.

        Returns:
            True if they are the same reduction, whatever each was asked for.
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

    def reads_two_columns(self) -> Bool:
        """Reports whether this reduction needs a second column.

        Returns:
            True for `CORR` and `COV`, which are statements about a pair of
            columns rather than about one, and which therefore travel a different
            entry point from every other kind here.
        """
        return self == Self.CORR or self == Self.COV

    def result_dtype(self, dt: DType) -> DType:
        """Returns the dtype this reduction produces over a column of `dt`.

        Args:
            dt: The input column's dtype.

        Returns:
            The dtype the reduction produces, which is int64 for the two
            counts, float64 for a mean, the accumulator dtype for a sum, and
            `dt` itself for the four that report a value the column held.
        """
        if self == Self.COUNT or self == Self.SIZE or self == Self.NUNIQUE:
            return DType.int64
        if (
            self == Self.MEAN
            or self == Self.VAR
            or self == Self.STD
            or self == Self.SEM
            or self == Self.SKEW
            or self == Self.MEDIAN
            or self == Self.QUANTILE
            or self == Self.CORR
            or self == Self.COV
        ):
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
        elif self == Self.SIZE:
            writer.write("size")
        elif self == Self.VAR:
            writer.write("var")
        elif self == Self.STD:
            writer.write("std")
        elif self == Self.SEM:
            writer.write("sem")
        elif self == Self.SKEW:
            writer.write("skew")
        elif self == Self.MEDIAN:
            writer.write("median")
        elif self == Self.QUANTILE:
            writer.write("quantile")
        elif self == Self.CORR:
            writer.write("corr")
        elif self == Self.COV:
            writer.write("cov")
        else:
            writer.write("nunique")


def group_size(
    codes: Array[DType.uint32], groups: Int
) raises -> Array[DType.int64]:
    """Counts the rows in each group, nulls included.

    The only reduction that does not look at a values column at all, which is why
    it takes none.

    Args:
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Returns:
        A column of `groups` counts, every one of them present.

    Raises:
        If one of the workers the parallel route starts cannot be run. A
        reduction small enough to stay on one core cannot fail.
    """
    # The bitmap is not read, because `check` is false, and it is passed rather
    # than made optional because a `Bitmap` argument is a borrow and costs
    # nothing. The codes' own validity is the one already in hand.
    return _tally_core[check=False](codes.data.validity, codes, groups)


def group_count[
    dt: DType
](values: Array[dt], codes: Array[DType.uint32], groups: Int) raises -> Array[
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

    Raises:
        If one of the workers the parallel route starts cannot be run. A
        reduction small enough to stay on one core cannot fail.
    """
    return _count_core(
        values.data.validity, values.null_count() > 0, codes, groups
    )


def _count_core(
    validity: Bitmap,
    has_null: Bool,
    codes: Array[DType.uint32],
    groups: Int,
) raises -> Array[DType.int64]:
    """Counts the present rows per group. Needs no dtype and reads no values."""
    if not has_null:
        return group_size(codes, groups)
    return _tally_core[check=True](validity, codes, groups)


def _tally_core[
    check: Bool
](validity: Bitmap, codes: Array[DType.uint32], groups: Int) raises -> Array[
    DType.int64
]:
    """Counts rows per group, either all of them or only the present ones.

    Args:
        validity: Which rows are present. Read only when `check`.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Parameters:
        check: True to skip null rows, False to count every row.

    Returns:
        A column of `groups` counts, every one of them present.
    """
    var n = len(codes)
    var workers = _private_workers[DType.int64](n, groups)

    if workers <= 1:
        var parts = _partition_parts[DType.int64](n, groups)
        if parts > 0:
            return _partitioned_tally(validity, check, codes, groups, parts)

        var out = Array[DType.int64](groups)
        var totals = out.unsafe_ptr()
        var at = codes.unsafe_ptr()
        for i in range(n):
            comptime if check:
                if not validity.get(i):
                    continue
            var g = Int(at.unsafe_offset(i).unsafe_load())
            totals.unsafe_offset(g).unsafe_store(
                totals.unsafe_offset(g).unsafe_load() + 1
            )
        return out^

    var bounds = _row_bounds(n, workers)
    var partials = Array[DType.int64](groups * workers)

    def one(w: Int) raises {mut partials, imm}:
        var totals = partials.unsafe_ptr().unsafe_offset(w * groups)
        var at = codes.unsafe_ptr()
        for i in range(bounds[w], bounds[w + 1]):
            comptime if check:
                if not validity.get(i):
                    continue
            var g = Int(at.unsafe_offset(i).unsafe_load())
            totals.unsafe_offset(g).unsafe_store(
                totals.unsafe_offset(g).unsafe_load() + 1
            )

    parallel_for(one, workers)
    return _merge_sums(partials, groups, workers)


def group_sum[
    dt: DType
](values: Array[dt], codes: Array[DType.uint32], groups: Int) raises -> Array[
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

    Raises:
        If one of the workers the parallel route starts cannot be run. A
        reduction small enough to stay on one core cannot fail.
    """
    return _sum_core(values.unsafe_ptr(), codes, groups)


def _sum_core[
    dt: DType, //, origin: ImmOrigin, acc: DType = accumulator(dt)
](
    source: Pointer[Scalar[dt], origin],
    codes: Array[DType.uint32],
    groups: Int,
) raises -> Array[acc]:
    """Accumulates every row into its group, validity ignored on purpose.

    The accumulator is a parameter with the natural widening as its default,
    because a grouped sum and a grouped mean want different ones. A sum over
    int64 accumulates in int64 and wraps, which is what pandas does. A mean must
    not be computed from that wrapped total, so `_mean_core` asks for float64
    here and divides that.
    """
    var n = len(codes)
    var workers = _private_workers[acc](n, groups)

    if workers <= 1:
        var parts = _partition_parts[acc](n, groups)
        if parts > 0:
            return _partitioned_sums[acc=acc](source, codes, groups, parts)

        var out = Array[acc](groups)
        var totals = out.unsafe_ptr()
        var at = codes.unsafe_ptr()
        for i in range(n):
            var g = Int(at.unsafe_offset(i).unsafe_load())
            totals.unsafe_offset(g).unsafe_store(
                totals.unsafe_offset(g).unsafe_load()
                + source.unsafe_offset(i).unsafe_load().cast[acc]()
            )
        return out^

    var bounds = _row_bounds(n, workers)
    var partials = Array[acc](groups * workers)

    def one(w: Int) raises {mut partials, imm}:
        var totals = partials.unsafe_ptr().unsafe_offset(w * groups)
        var at = codes.unsafe_ptr()
        for i in range(bounds[w], bounds[w + 1]):
            var g = Int(at.unsafe_offset(i).unsafe_load())
            totals.unsafe_offset(g).unsafe_store(
                totals.unsafe_offset(g).unsafe_load()
                + source.unsafe_offset(i).unsafe_load().cast[acc]()
            )

    parallel_for(one, workers)
    return _merge_sums(partials, groups, workers)


def group_mean[
    dt: DType
](values: Array[dt], codes: Array[DType.uint32], groups: Int) raises -> Array[
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
        A column of `groups` means. A group with no non-null values is NaN and
        not null, because zero would be a value the group does not have and a
        pandas float column spells missing with a NaN.

    Raises:
        If one of the workers the parallel route starts cannot be run. A
        reduction small enough to stay on one core cannot fail.
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
) raises -> Array[DType.float64]:
    """Divides a grouped sum by a grouped count, one group at a time.

    The sum is taken in float64 and not in the natural accumulator, which for an
    integer column is the difference between an answer and a wrapped one. pandas
    converts to float64 before dividing, so the mean of a group of large int64
    values is a large float rather than whatever the int64 wrap happened to leave
    behind. `group_sum` still wraps, because there pandas wraps too.
    """
    var sums = _sum_core[acc=DType.float64](source, codes, groups)
    var counts = _count_core(validity, has_null, codes, groups)

    var out = Array[DType.float64](groups)
    var target = out.unsafe_ptr()
    var total = sums.unsafe_ptr()
    var n = counts.unsafe_ptr()
    for g in range(groups):
        var count = n.unsafe_offset(g).unsafe_load()
        if count == 0:
            # A group with nothing in it gets NaN and stays valid, because in a
            # pandas float column NaN is what missing looks like. See #170.
            target.unsafe_offset(g).unsafe_store(nan[DType.float64]())
            continue
        target.unsafe_offset(g).unsafe_store(
            total.unsafe_offset(g).unsafe_load() / Float64(count)
        )
    return out^


def group_min[
    dt: DType
](values: Array[dt], codes: Array[DType.uint32], groups: Int) raises -> Array[
    dt
]:
    """Returns the smallest non-null value in each group.

    Args:
        values: The column being aggregated.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Parameters:
        dt: The column's dtype.

    Returns:
        A column of `groups` minima, null for any group with no non-null values.

    Raises:
        If one of the workers the parallel route starts cannot be run. A
        reduction small enough to stay on one core cannot fail.
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
](values: Array[dt], codes: Array[DType.uint32], groups: Int) raises -> Array[
    dt
]:
    """Returns the largest non-null value in each group.

    Args:
        values: The column being aggregated.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Parameters:
        dt: The column's dtype.

    Returns:
        A column of `groups` maxima, null for any group with no non-null values.

    Raises:
        If one of the workers the parallel route starts cannot be run. A
        reduction small enough to stay on one core cannot fail.
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
) raises -> Array[dt]:
    """Reduces each group to its smallest or largest non-null value.

    Min and max differ by one comparison, so they share a body on the same terms
    as `_extreme` in `agg.mojo`.

    Every accumulator starts at the reduction's identity, so the loop has no
    per-row branch on whether the group has been seen yet. Which groups were seen
    is carried in the output's validity bitmap, which had to be maintained
    anyway, so the seen flags cost nothing beyond the bit that was already there.

    The identity being neutral is also what makes the merge cheap. A worker that
    never reached a group left the identity in its slot, and folding the identity
    in changes nothing, so the merge does not have to ask which groups a worker
    saw before it takes that worker's table. It is then the same vectorized walk
    of two contiguous arrays that `_merge_sums` is, with a comparison and a
    select where the sums have an add. It used to be a scalar loop carrying a
    branch and a bitmap read modify write per group per worker, which on a
    hundred thousand groups over thirty two workers is three million of them and
    was most of what a grouped min or max cost on a wide key.
    """
    comptime identity = highest[dt]() if want_min else lowest[dt]()
    comptime width = simd_width_of[dt]()

    var n = len(codes)
    var workers = _private_workers[dt](n, groups)

    # Every slot is written by the identity fill below, so the zero fill the
    # plain constructor does is a pass over the output for nothing.
    var out = Array[dt](overwritten=groups)
    var best = out.unsafe_ptr()
    out.data.validity.clear_all()
    var head = 0
    while head + width <= groups:
        best.unsafe_offset(head).unsafe_store(SIMD[dt, width](identity))
        head += width
    while head < groups:
        best.unsafe_offset(head).unsafe_store(identity)
        head += 1

    if workers <= 1:
        var at = codes.unsafe_ptr()
        for i in range(n):
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
    else:
        var bounds = _row_bounds(n, workers)
        var slots = groups * workers
        var partials = Array[dt](overwritten=slots)
        var start = partials.unsafe_ptr()
        var filled = 0
        while filled + width <= slots:
            start.unsafe_offset(filled).unsafe_store(SIMD[dt, width](identity))
            filled += width
        while filled < slots:
            start.unsafe_offset(filled).unsafe_store(identity)
            filled += 1

        # Which groups a worker saw is a byte per group rather than a bit,
        # because a bitmap packs sixty four groups into one word and two workers
        # owning neighbouring groups would be writing the same word at once. A
        # byte each is `groups * workers` bytes on top of the tables, which is
        # an eighth of what the tables themselves cost at eight bytes a value.
        var seen = Array[DType.uint8](groups * workers)

        def one(w: Int) raises {mut partials, mut seen, imm}:
            var mine = partials.unsafe_ptr().unsafe_offset(w * groups)
            var hit = seen.unsafe_ptr().unsafe_offset(w * groups)
            var at = codes.unsafe_ptr()
            for i in range(bounds[w], bounds[w + 1]):
                if has_null and not validity.get(i):
                    continue
                var g = Int(at.unsafe_offset(i).unsafe_load())
                var value = source.unsafe_offset(i).unsafe_load()
                var current = mine.unsafe_offset(g).unsafe_load()
                comptime if want_min:
                    if value < current:
                        mine.unsafe_offset(g).unsafe_store(value)
                else:
                    if value > current:
                        mine.unsafe_offset(g).unsafe_store(value)
                hit.unsafe_offset(g).unsafe_store(UInt8(1))

        parallel_for(one, workers)

        comptime bytes = simd_width_of[DType.uint8]()
        var tables = partials.unsafe_ptr()
        var hits = seen.unsafe_ptr()
        for w in range(workers):
            var table = tables.unsafe_offset(w * groups)
            var g = 0
            while g + width <= groups:
                var current = best.unsafe_offset(g).unsafe_load[width=width]()
                var value = table.unsafe_offset(g).unsafe_load[width=width]()
                # `min` and `max` rather than a comparison and a select, and
                # they differ only on a NaN, which cannot be here. The scatter
                # above stores a value only when it compares below or above what
                # the slot holds, and every comparison against a NaN is false, so
                # no NaN is ever written into a table and none can be read out of
                # one.
                comptime if want_min:
                    best.unsafe_offset(g).unsafe_store(min(value, current))
                else:
                    best.unsafe_offset(g).unsafe_store(max(value, current))
                g += width
            while g < groups:
                var current = best.unsafe_offset(g).unsafe_load()
                var value = table.unsafe_offset(g).unsafe_load()
                comptime if want_min:
                    if value < current:
                        best.unsafe_offset(g).unsafe_store(value)
                else:
                    if value > current:
                        best.unsafe_offset(g).unsafe_store(value)
                g += 1

            # The seen flags fold the same way and into the first worker's row,
            # a byte at a time so that thirty two groups settle per instruction.
            # A group is present if any worker reached it, which is an or.
            if w > 0:
                var hit = hits.unsafe_offset(w * groups)
                var b = 0
                while b + bytes <= groups:
                    hits.unsafe_offset(b).unsafe_store(
                        hits.unsafe_offset(b).unsafe_load[width=bytes]()
                        | hit.unsafe_offset(b).unsafe_load[width=bytes]()
                    )
                    b += bytes
                while b < groups:
                    hits.unsafe_offset(b).unsafe_store(
                        hits.unsafe_offset(b).unsafe_load()
                        | hit.unsafe_offset(b).unsafe_load()
                    )
                    b += 1

        for g in range(groups):
            if hits.unsafe_offset(g).unsafe_load() != 0:
                out.data.validity.set(g, True)

    # A group that was never seen still holds the identity, which is a real value
    # of the dtype and would read as a maximum of negative infinity. It has to be
    # overwritten either way, and what goes in depends on whether the dtype has a
    # NaN. A float column takes one and stays valid, because that is what pandas
    # has there and it is the same rule the reductions that answer in float64
    # follow. Every other dtype takes a zero behind a cleared bit, because there
    # is no NaN to write and a null is the only spelling available. See #170.
    comptime if dt.is_floating_point():
        for g in range(groups):
            if not out.data.validity.get(g):
                best.unsafe_offset(g).unsafe_store(nan[dt]())
                out.data.validity.set(g, True)
    else:
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

    # A float column says missing with a NaN and stays valid, the same rule the
    # rest of this file follows. Nothing to do when the walk filled every group,
    # which is the common shape and the one the early exit above is for, and
    # nothing to do for a dtype that has no NaN to write. See #170.
    comptime if dt.is_floating_point():
        if filled < groups:
            for g in range(groups):
                if not out.data.validity.get(g):
                    target.unsafe_offset(g).unsafe_store(nan[dt]())
                    out.data.validity.set(g, True)
    return out^


def group_var[
    dt: DType
](
    values: Array[dt],
    codes: Array[DType.uint32],
    groups: Int,
    ddof: Int = 1,
) raises -> Array[DType.float64]:
    """Returns the variance of the non-null values in each group.

    Args:
        values: The column being aggregated.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.
        ddof: Subtracted from the count to give the divisor. One is the sample
            variance and is what pandas defaults to; zero is the population one.

    Parameters:
        dt: The column's dtype.

    Returns:
        A column of `groups` variances. A group with `ddof` or fewer non-null
        values is NaN, which is what pandas reports there.

    Raises:
        If one of the workers the parallel route starts cannot be run. A
        reduction small enough to stay on one core cannot fail.
    """
    return _var_core[want_std=False](
        values.unsafe_ptr(),
        values.data.validity,
        values.null_count() > 0,
        codes,
        groups,
        ddof,
    )


def group_std[
    dt: DType
](
    values: Array[dt],
    codes: Array[DType.uint32],
    groups: Int,
    ddof: Int = 1,
) raises -> Array[DType.float64]:
    """Returns the standard deviation of the non-null values in each group.

    Args:
        values: The column being aggregated.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.
        ddof: Subtracted from the count to give the divisor.

    Parameters:
        dt: The column's dtype.

    Returns:
        A column of `groups` standard deviations, NaN on the same groups
        `group_var` reports NaN on.

    Raises:
        If one of the workers the parallel route starts cannot be run. A
        reduction small enough to stay on one core cannot fail.
    """
    return _var_core[want_std=True](
        values.unsafe_ptr(),
        values.data.validity,
        values.null_count() > 0,
        codes,
        groups,
        ddof,
    )


def group_sem[
    dt: DType
](
    values: Array[dt],
    codes: Array[DType.uint32],
    groups: Int,
    ddof: Int = 1,
) raises -> Array[DType.float64]:
    """Returns the standard error of the mean of each group.

    Args:
        values: The column being aggregated.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.
        ddof: Subtracted from the count to give the variance divisor. The count
            under the root is the plain one and is not corrected by it.

    Parameters:
        dt: The column's dtype.

    Returns:
        A column of `groups` standard errors, NaN on the same groups
        `group_std` reports NaN on.

    Raises:
        If one of the workers the parallel route starts cannot be run. A
        reduction small enough to stay on one core cannot fail.
    """
    return _sem_core(
        values.unsafe_ptr(),
        values.data.validity,
        values.null_count() > 0,
        codes,
        groups,
        ddof,
    )


def group_skew[
    dt: DType
](values: Array[dt], codes: Array[DType.uint32], groups: Int) raises -> Array[
    DType.float64
]:
    """Returns the sample skewness of the non-null values in each group.

    Args:
        values: The column being aggregated.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Parameters:
        dt: The column's dtype.

    Returns:
        A column of `groups` skewnesses, NaN where a group has fewer than three
        non-null values, which is what pandas reports there. A group whose values
        are all equal is zero rather than NaN.

    Raises:
        If one of the workers the parallel route starts cannot be run. A
        reduction small enough to stay on one core cannot fail.
    """
    return _skew_core(
        values.unsafe_ptr(),
        values.data.validity,
        values.null_count() > 0,
        codes,
        groups,
    )


def _var_core[
    dt: DType, //, want_std: Bool, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_null: Bool,
    codes: Array[DType.uint32],
    groups: Int,
    ddof: Int,
) raises -> Array[DType.float64]:
    """Sums the squared deviations from each group's own mean.

    Two passes rather than one. The single pass version accumulates the sum and
    the sum of squares together and subtracts at the end, which is one loop
    instead of two and is the arrangement most libraries start with. It also
    loses every significant digit when the values are large and the spread is
    small, because it computes a small number as the difference of two large
    ones: a column of timestamps around 1.7e9 with a spread of a few seconds has
    a sum of squares near 3e18, and the subtraction that is supposed to leave the
    variance is cancelling seventeen digits away. Timestamps in a group by are
    not a corner case, so this takes the second pass.

    The second pass is not quite enough on its own either. The centre it
    subtracts is a rounded sum divided by a count, so it is near the mean but it
    is not the mean, and the squared deviations are therefore measured from
    slightly the wrong place. That error does not cancel, it accumulates, and it
    grows with the magnitude of the values: on five values sitting at 4.5e15
    whose true variance is 37.2 the uncorrected form answers 37.25. So this also
    accumulates the plain deviations, which sum to zero when the centre is exact
    and to the size of the miss when it is not, and subtracts the square of that
    residual at the end. The identity is the same one the single pass form uses,
    but applied to numbers that are already small, so there is nothing left to
    cancel. It costs one add per row and makes the answer independent of how
    accurate the mean was.
    """
    var means = _mean_core(source, validity, has_null, codes, groups)
    var counts = _count_core(validity, has_null, codes, groups)

    var rows = len(codes)
    var workers = _private_workers[DType.float64](rows, groups)
    var centre = means.unsafe_ptr()
    var at = codes.unsafe_ptr()

    # The same private table arrangement the sums and the counts already use.
    # The first pass of this reduction goes through `_mean_core` and has been
    # parallel for a while; the second one was still a serial scatter, so a
    # standard deviation cost more than the mean it is built on and used one
    # core to do it.
    var deltas = Array[DType.float64](groups * workers)
    var partials = Array[DType.float64](groups * workers)
    var bounds = _row_bounds(rows, workers)

    def one(w: Int) raises {mut deltas, mut partials, imm}:
        var plain = deltas.unsafe_ptr().unsafe_offset(w * groups)
        var totals = partials.unsafe_ptr().unsafe_offset(w * groups)
        for i in range(bounds[w], bounds[w + 1]):
            if has_null and not validity.get(i):
                continue
            var g = Int(at.unsafe_offset(i).unsafe_load())
            var delta = (
                source.unsafe_offset(i).unsafe_load().cast[DType.float64]()
                - centre.unsafe_offset(g).unsafe_load()
            )
            plain.unsafe_offset(g).unsafe_store(
                plain.unsafe_offset(g).unsafe_load() + delta
            )
            totals.unsafe_offset(g).unsafe_store(
                totals.unsafe_offset(g).unsafe_load() + delta * delta
            )

    parallel_for(one, workers)

    if workers > 1:
        deltas = _merge_sums(deltas, groups, workers)
        partials = _merge_sums(partials, groups, workers)
    var out = partials^
    var target = out.unsafe_ptr()
    var missed = deltas.unsafe_ptr()
    var n = counts.unsafe_ptr()
    for g in range(groups):
        var present = Int(n.unsafe_offset(g).unsafe_load())
        var divisor = present - ddof
        if divisor <= 0:
            # Too few values to have a spread, so NaN and still valid. A pandas
            # float column says missing with NaN and not with a null. See #170.
            target.unsafe_offset(g).unsafe_store(nan[DType.float64]())
            continue
        var residual = missed.unsafe_offset(g).unsafe_load()
        var squares = target.unsafe_offset(
            g
        ).unsafe_load() - residual * residual / Float64(present)
        # The correction is a subtraction and a subtraction can overshoot, so a
        # group whose values are all the same can land a rounding error below
        # zero rather than on it. A negative variance is not a number this
        # returns.
        if squares < 0.0:
            squares = 0.0
        var value = squares / Float64(divisor)
        comptime if want_std:
            value = sqrt(value)
        target.unsafe_offset(g).unsafe_store(value)
    return out^


def _sem_core[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_null: Bool,
    codes: Array[DType.uint32],
    groups: Int,
    ddof: Int,
) raises -> Array[DType.float64]:
    """Divides each group's standard deviation by the root of its count.

    Built on `_var_core` rather than beside it, because the standard error is
    the standard deviation and one division and there is no numerical reason for
    a second traversal of the column. The count under the root is the plain non
    null count and is not adjusted by `ddof`, which is what pandas does: the
    degrees of freedom correction belongs to the variance and applying it twice
    would be a different statistic.

    Args:
        source: The values.
        validity: Which of them are present.
        has_null: Whether any are absent.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.
        ddof: Subtracted from each group's count to give the variance divisor.

    Returns:
        A column of `groups` standard errors, NaN where the variance was NaN,
        which is where a group had `ddof` or fewer values.

    Raises:
        If one of the workers the parallel route starts cannot be run.
    """
    var out = _var_core[want_std=True](
        source, validity, has_null, codes, groups, ddof
    )
    var counts = _count_core(validity, has_null, codes, groups)
    var target = out.unsafe_ptr()
    var n = counts.unsafe_ptr()
    for g in range(groups):
        if not out.data.validity.get(g):
            continue
        var present = Int(n.unsafe_offset(g).unsafe_load())
        target.unsafe_offset(g).unsafe_store(
            target.unsafe_offset(g).unsafe_load() / sqrt(Float64(present))
        )
    return out^


def _skew_core[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_null: Bool,
    codes: Array[DType.uint32],
    groups: Int,
) raises -> Array[DType.float64]:
    """Sums the deviations from each group's own mean, squared and cubed.

    Two passes, and for the reason `_var_core` gives at length: the one pass
    arrangement accumulates the raw moments and subtracts at the end, and the
    cancellation that costs a variance its significant digits costs a skewness
    more, because the third moment is the difference of larger quantities still.

    Two passes is not enough on its own, which is the difference between this
    and `_var_core`. The centre is a rounded sum divided by a count, so it is
    not the mean, and the deviations are taken from something slightly off. A
    variance barely notices, because the error enters squared. A skewness
    notices a great deal, because the correction to the third moment is
    proportional to the second, and the second is very much larger than the
    third whenever the data is anywhere near symmetric. So the sum of the
    deviations is accumulated as well, which is zero in exact arithmetic and is
    the amount the centre missed by in this one, and both moments are corrected
    by it with the shifted data identity. Three multiplications per group, and
    on a column sitting at 4.6e18 with a spread of 4.2e9 it is the difference
    between five correct digits and eight.

    The coefficient is the one pandas reports, the adjusted Fisher Pearson
    standardized moment, which is the plain moment ratio scaled by
    `sqrt(n * (n - 1)) / (n - 2)` so that it is unbiased for a normal sample.
    Two edges come with it and both are pandas' answers rather than choices made
    here. A group of fewer than three has no skewness, because the adjustment
    divides by `n - 2`. A group whose values are all the same has a skewness of
    zero rather than a null, because the shape of a constant is not undefined,
    it is symmetric.

    Args:
        source: The values.
        validity: Which of them are present.
        has_null: Whether any are absent.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Returns:
        A column of `groups` skewnesses, NaN where a group has fewer than three
        non-null values.

    Raises:
        If one of the workers the parallel route starts cannot be run.
    """
    var means = _mean_core(source, validity, has_null, codes, groups)
    var counts = _count_core(validity, has_null, codes, groups)

    var rows = len(codes)
    var workers = _private_workers[DType.float64](rows, groups)
    var centre = means.unsafe_ptr()
    var at = codes.unsafe_ptr()

    var deltas = Array[DType.float64](groups * workers)
    var squares = Array[DType.float64](groups * workers)
    var cubes = Array[DType.float64](groups * workers)
    var bounds = _row_bounds(rows, workers)

    def one(w: Int) raises {mut deltas, mut squares, mut cubes, imm}:
        var first = deltas.unsafe_ptr().unsafe_offset(w * groups)
        var second = squares.unsafe_ptr().unsafe_offset(w * groups)
        var third = cubes.unsafe_ptr().unsafe_offset(w * groups)
        for i in range(bounds[w], bounds[w + 1]):
            if has_null and not validity.get(i):
                continue
            var g = Int(at.unsafe_offset(i).unsafe_load())
            var delta = (
                source.unsafe_offset(i).unsafe_load().cast[DType.float64]()
                - centre.unsafe_offset(g).unsafe_load()
            )
            var squared = delta * delta
            first.unsafe_offset(g).unsafe_store(
                first.unsafe_offset(g).unsafe_load() + delta
            )
            second.unsafe_offset(g).unsafe_store(
                second.unsafe_offset(g).unsafe_load() + squared
            )
            third.unsafe_offset(g).unsafe_store(
                third.unsafe_offset(g).unsafe_load() + squared * delta
            )

    parallel_for(one, workers)

    if workers > 1:
        deltas = _merge_sums(deltas, groups, workers)
        squares = _merge_sums(squares, groups, workers)
        cubes = _merge_sums(cubes, groups, workers)
    var out = squares^
    var target = out.unsafe_ptr()
    var first_total = deltas.unsafe_ptr()
    var third_total = cubes.unsafe_ptr()
    var n = counts.unsafe_ptr()
    for g in range(groups):
        var present = Int(n.unsafe_offset(g).unsafe_load())
        if present < 3:
            # A skewness needs three values and this group has fewer, so NaN and
            # still valid, the way a pandas float column spells missing. See #170.
            target.unsafe_offset(g).unsafe_store(nan[DType.float64]())
            continue
        var size = Float64(present)
        # The three sums are taken about the mean, so in exact arithmetic the
        # first of them is zero and the other two are the moments. It is not
        # zero, because the mean is itself a rounded sum, and every value here
        # is a rounded difference from it. Correcting the moments by the amount
        # the centre missed by is the standard shifted data identity and it costs
        # three multiplications per group. It matters because the correction is
        # proportional to the second moment and the third moment is much smaller
        # than the second whenever the data is anywhere near symmetric, so an
        # error too small to see in a variance is the leading term in a skewness.
        # On a column at 4.6e18 with a spread of 4.2e9 this is the difference
        # between five correct digits and eight.
        var residual = first_total.unsafe_offset(g).unsafe_load() / size
        var second_moment = (
            target.unsafe_offset(g).unsafe_load() / size - residual * residual
        )
        if second_moment <= 0.0:
            # Every value in the group is the same, up to the rounding above.
            # The shape of a constant is symmetric rather than undefined, which
            # is the answer pandas gives, and the comparison is against zero
            # rather than equal to it because the correction can leave a
            # negative the size of a rounding error behind.
            target.unsafe_offset(g).unsafe_store(0.0)
            continue
        var third_moment = (
            third_total.unsafe_offset(g).unsafe_load() / size
            - 3.0 * residual * target.unsafe_offset(g).unsafe_load() / size
            + 2.0 * residual * residual * residual
        )
        var adjust = sqrt(size * (size - 1.0)) / (size - 2.0)
        target.unsafe_offset(g).unsafe_store(
            adjust * third_moment / (second_moment * sqrt(second_moment))
        )
    return out^


def group_median[
    dt: DType
](values: Array[dt], codes: Array[DType.uint32], groups: Int) raises -> Array[
    DType.float64
]:
    """Returns the median of the non-null values in each group.

    Args:
        values: The column being aggregated.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Parameters:
        dt: The column's dtype.

    Returns:
        A column of `groups` medians, NaN where a group has no non-null values.
        An even count interpolates between the two middle values, so the median
        of a column of integers is a float and can be a half.

    Raises:
        If one of the workers the parallel route starts cannot be run. A
        reduction small enough to stay on one core cannot fail.
    """
    return _quantile_core(
        values.unsafe_ptr(),
        values.data.validity,
        values.null_count() > 0,
        codes,
        groups,
        0.5,
    )


def group_quantile[
    dt: DType
](
    values: Array[dt], codes: Array[DType.uint32], groups: Int, q: Float64
) raises -> Array[DType.float64]:
    """Returns one quantile of the non-null values in each group.

    Args:
        values: The column being aggregated.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.
        q: Where in the sorted values to land, from zero for the minimum to one
            for the maximum.

    Parameters:
        dt: The column's dtype.

    Returns:
        A column of `groups` quantiles, interpolated linearly between the two
        values the position falls between, which is what pandas does by default.
        NaN where a group has no non-null values.

    Raises:
        If `q` is outside zero to one.
    """
    if not (q >= 0.0 and q <= 1.0):
        raise Error(
            "group by: a quantile must be between 0 and 1, got " + String(q)
        )
    return _quantile_core(
        values.unsafe_ptr(),
        values.data.validity,
        values.null_count() > 0,
        codes,
        groups,
        q,
    )


def group_nunique[
    dt: DType
](values: Array[dt], codes: Array[DType.uint32], groups: Int) raises -> Array[
    DType.int64
]:
    """Counts the distinct non-null values in each group.

    Args:
        values: The column being aggregated.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Parameters:
        dt: The column's dtype.

    Returns:
        A column of `groups` counts, every one of them present. A group with no
        non-null values counts zero rather than reporting null, which is what
        pandas does and is the same choice `group_count` makes.

    Raises:
        If one of the workers the parallel route starts cannot be run. A
        reduction small enough to stay on one core cannot fail.
    """
    return _nunique_core(
        values.unsafe_ptr(),
        values.data.validity,
        values.null_count() > 0,
        codes,
        groups,
    )


def _slab_bounds(counts: Array[DType.int64], groups: Int) -> List[Int]:
    """Turns per group counts into `groups + 1` slab offsets."""
    # A dependent add per group, and it stays that way. The blocked parallel scan
    # was written and measured and it is not faster here: the chain is one add
    # deep and the counts are already streaming out of memory, so what bounds the
    # loop is the read rather than the latency, and a version that reads the
    # counts twice to get a second core onto them reads twice as much for it. At
    # six and a half million groups the two were within the run to run spread.
    var bounds = List[Int](capacity=groups + 1)
    var n = counts.unsafe_ptr()
    var running = 0
    bounds.append(0)
    for g in range(groups):
        running += Int(n.unsafe_offset(g).unsafe_load())
        bounds.append(running)
    return bounds^


comptime SLAB_SERIAL_GROUPS = PARTITION_BYTES // 72
"""Groups below which the slab fill stays on one core.

The fill holds a write cursor per group, eight bytes, and it has an open cache
line at the head of every group's run of the slab, sixty four more. While those
two fit in a core's private cache the fill is a sequential read and a manageable
set of write streams and there is nothing here to fix. Above it every row costs
two misses, and that is where partitioning starts to pay for the extra pass it
takes.
"""


def _slab_shift[dt: DType](rows: Int, groups: Int) -> Int:
    """Returns how many low bits of a group ordinal one slab partition covers.

    Args:
        rows: How many rows are being laid out.
        groups: How many groups they land in.

    Parameters:
        dt: The slab's dtype.

    Returns:
        The shift, so that partition `p` owns ordinals `p << shift` upwards.
    """
    # What a partition touches is its cursor, eight bytes a group, and its run of
    # the slab, which is as many values as its groups hold between them. So the
    # width is set by how many rows a group holds on average and not by the group
    # count alone, and that is the whole difference between this and
    # `_partition_shift`. A slab partition over a key with a hundred rows to the
    # group has to be a hundred times narrower than one over a key with one.
    var per_group = 8 + (size_of[dt]() * rows) // groups
    var span = PARTITION_BYTES // per_group
    var shift = 0
    while (1 << (shift + 1)) <= span and (1 << (shift + 1)) <= groups:
        shift += 1
    return shift


def _fill_slab_serial[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_null: Bool,
    codes: Array[DType.uint32],
    groups: Int,
    bounds: List[Int],
    mut slab: Array[dt],
):
    """Walks the rows once, writing each into its group's run."""
    var cursor = List[Int](capacity=groups)
    for g in range(groups):
        cursor.append(bounds[g])

    var into = slab.unsafe_ptr()
    var at = codes.unsafe_ptr()
    for i in range(len(codes)):
        if has_null and not validity.get(i):
            continue
        var g = Int(at.unsafe_offset(i).unsafe_load())
        into.unsafe_offset(cursor[g]).unsafe_store(
            source.unsafe_offset(i).unsafe_load()
        )
        cursor[g] = cursor[g] + 1


def _fill_slab[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_null: Bool,
    codes: Array[DType.uint32],
    groups: Int,
    bounds: List[Int],
    mut slab: Array[dt],
) raises:
    """Gathers each group's non-null values into its own run of the slab.

    The three reductions below all need a group's values next to each other, and
    two of them need them sorted. Sorting the whole column by group and then by
    value would do it in one radix pass, but the sort kernel here argsorts into a
    permutation and applying that is another pass over the column plus an
    indirection per row. Scattering into per group runs is the same single pass
    and leaves each run short enough that sorting it is cheap: the runs sum to
    the row count, so sorting all of them is n log(n over groups) rather than
    n log n, and on the thousand group shape that is a third of the comparisons.

    That single pass is a random write per row, and on a high cardinality key it
    is the whole cost of the reduction. Measured on ten million rows on an
    i9-13900K, a median over a thousand groups was three nanoseconds a row and
    the same median over six and a half million groups was seventeen, while the
    sorts it was paying for got shorter at every step. So the pass is cut the
    same way `_partitioned_sums` cuts its fold: the rows are copied once into
    partition order, where a partition is a run of ordinals, and then each
    partition is filled by one worker into a slice of the slab small enough to
    stay in that core's cache. Three passes instead of one, all of them parallel,
    and none of them missing.

    A group's values stay in row order either way. The copy lays the workers out
    in order inside a partition and each worker reads its rows forwards, so the
    order a group's values end up in is the order they were in the column. The
    two callers sort, so nothing depends on that, and it costs nothing to keep.

    Args:
        source: The values being laid out.
        validity: Which rows are present.
        has_null: False to skip the presence check per row.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.
        bounds: `groups + 1` slab offsets, from `_slab_bounds`.
        slab: Filled with every present value, grouped.

    Parameters:
        dt: The column's dtype.
        origin: The origin of the values pointer.

    Raises:
        If one of the workers the partitioned route starts cannot be run. The
        serial route cannot fail.
    """
    var rows = len(codes)
    var workers = worker_count()
    if rows < PARTITION_ROWS or groups <= SLAB_SERIAL_GROUPS or workers <= 1:
        _fill_slab_serial(
            source, validity, has_null, codes, groups, bounds, slab
        )
        return

    var shift = _slab_shift[dt](rows, groups)
    var parts = (groups + (1 << shift) - 1) >> shift
    if parts < 2:
        _fill_slab_serial(
            source, validity, has_null, codes, groups, bounds, slab
        )
        return

    var blocks = _row_bounds(rows, workers)
    var starts = _partition_starts(
        codes, validity, has_null, blocks, workers, parts, shift
    )

    # A null row is left out of the partitioning, so these hold the present rows
    # and the tail of each is never read. Sizing them by the row count rather
    # than by the present count saves a pass and costs nothing that is touched.
    var held = Buffer(overwritten=rows * 4)
    var carried = Buffer(overwritten=rows * size_of[dt]())

    def place(w: Int) raises {mut held, mut carried, imm}:
        var at = codes.unsafe_ptr()
        var ordinals = held.bitcast[DType.uint32]()
        var values = carried.bitcast[dt]()
        var cursor = List[Int](capacity=parts)
        for p in range(parts):
            cursor.append(starts[p * workers + w])
        for i in range(blocks[w], blocks[w + 1]):
            if has_null and not validity.get(i):
                continue
            var g = at.unsafe_offset(i).unsafe_load()
            var p = Int(g) >> shift
            var slot = cursor[p]
            ordinals.unsafe_offset(slot).unsafe_write(g)
            values.unsafe_offset(slot).unsafe_write(
                source.unsafe_offset(i).unsafe_load()
            )
            cursor[p] = slot + 1

    parallel_for(place, workers)

    def settle(p: Int) raises {mut slab, imm}:
        var into = slab.unsafe_ptr()
        var ordinals = held.bitcast[DType.uint32]()
        var values = carried.bitcast[dt]()
        var low = p << shift
        var high = min(groups, low + (1 << shift))
        var cursor = List[Int](capacity=high - low)
        for g in range(low, high):
            cursor.append(bounds[g])
        for i in range(starts[p * workers], starts[(p + 1) * workers]):
            var g = Int(ordinals.unsafe_offset(i).unsafe_load()) - low
            var slot = cursor[g]
            into.unsafe_offset(slot).unsafe_store(
                values.unsafe_offset(i).unsafe_load()
            )
            cursor[g] = slot + 1

    parallel_for(settle, parts)


def _quantile_core[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_null: Bool,
    codes: Array[DType.uint32],
    groups: Int,
    q: Float64,
) raises -> Array[DType.float64]:
    """Sorts each group's values and reads the position `q` falls at."""
    var counts = _count_core(validity, has_null, codes, groups)
    var bounds = _slab_bounds(counts, groups)
    # Every element of the slab is a present row and the fill writes all of them,
    # so zeroing it first is a pass over the column for nothing.
    var slab = Array[dt](overwritten=bounds[groups])
    _fill_slab(source, validity, has_null, codes, groups, bounds, slab)

    var out = Array[DType.float64](groups)

    # One group's run of the slab is nobody else's, so this loop needs no
    # private tables and no merge. It needs no care about where the cut falls
    # either, now that an empty group answers NaN in its own float64 slot rather
    # than clearing a bit two workers could share a byte of. It is the whole
    # cost of a median on a high
    # cardinality key, where the row count is spread over so many groups that
    # the sorts are short and there are millions of them.
    var blocks = _group_blocks(len(codes), groups)
    var cuts = _group_bounds(groups, blocks)

    def one(b: Int) raises {mut slab, mut out, imm}:
        var values = slab.unsafe_ptr()
        var target = out.unsafe_ptr()
        for g in range(cuts[b], cuts[b + 1]):
            var start = bounds[g]
            var count = bounds[g + 1] - start
            if count == 0:
                # Nothing to take a quantile of, so NaN and still valid, which is
                # how a pandas float column carries a missing value. See #170.
                target.unsafe_offset(g).unsafe_store(nan[DType.float64]())
                continue
            sort(
                Span[Scalar[dt], origin_of(slab)](
                    unsafe_ptr=values.unsafe_offset(start), length=count
                )
            )
            # pandas' default interpolation. The position is on the sorted
            # values rather than between them, so q of zero is the minimum and
            # q of one is the maximum exactly, and everything in between is a
            # weighted pair.
            var position = q * Float64(count - 1)
            var lower = Int(position)
            var upper = lower + 1 if lower + 1 < count else lower
            var low = (
                values.unsafe_offset(start + lower)
                .unsafe_load()
                .cast[DType.float64]()
            )
            var high = (
                values.unsafe_offset(start + upper)
                .unsafe_load()
                .cast[DType.float64]()
            )
            target.unsafe_offset(g).unsafe_store(
                low + (high - low) * (position - Float64(lower))
            )

    parallel_for(one, blocks)
    return out^


def _nunique_core[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_null: Bool,
    codes: Array[DType.uint32],
    groups: Int,
) raises -> Array[DType.int64]:
    """Sorts each group's values and counts the runs.

    Sorting to count distinct values rather than putting them in a hash set. The
    set is asymptotically better and loses here for two reasons: the slab is
    already being built and sorted for the quantiles, so this is reusing a pass
    rather than adding one, and a set of every distinct value in the column is
    unbounded where the slab is exactly the size of the data. A group by that
    counts distinct values on a high cardinality column is the case where the
    set would be biggest and it is also the case where it would collide most.
    """
    var counts = _count_core(validity, has_null, codes, groups)
    var bounds = _slab_bounds(counts, groups)
    # Every element of the slab is a present row and the fill writes all of them,
    # so zeroing it first is a pass over the column for nothing.
    var slab = Array[dt](overwritten=bounds[groups])
    _fill_slab(source, validity, has_null, codes, groups, bounds, slab)

    var out = Array[DType.int64](groups)
    var blocks = _group_blocks(len(codes), groups)
    var cuts = _group_bounds(groups, blocks)

    def one(b: Int) raises {mut slab, mut out, imm}:
        var values = slab.unsafe_ptr()
        var target = out.unsafe_ptr()
        for g in range(cuts[b], cuts[b + 1]):
            var start = bounds[g]
            var count = bounds[g + 1] - start
            if count == 0:
                continue
            sort(
                Span[Scalar[dt], origin_of(slab)](
                    unsafe_ptr=values.unsafe_offset(start), length=count
                )
            )
            var distinct = 1
            for i in range(start + 1, start + count):
                if (
                    values.unsafe_offset(i).unsafe_load()
                    != values.unsafe_offset(i - 1).unsafe_load()
                ):
                    distinct += 1
            target.unsafe_offset(g).unsafe_store(Int64(distinct))

    parallel_for(one, blocks)
    return out^


def group_corr[
    dx: DType, dy: DType
](
    x: Array[dx], y: Array[dy], codes: Array[DType.uint32], groups: Int
) raises -> Array[DType.float64]:
    """Returns the Pearson correlation of two columns within each group.

    Args:
        x: One of the two columns.
        y: The other.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Parameters:
        dx: The first column's dtype.
        dy: The second column's dtype.

    Returns:
        A column of `groups` correlations between minus one and one. A group with
        fewer than two rows in which both values are present is NaN, and so is a
        group in which either column does not vary, because the correlation is a
        zero over a zero there.

    Raises:
        If one of the workers the parallel route starts cannot be run. A
        reduction small enough to stay on one core cannot fail.
    """
    return _pair_core[want_corr=True](
        x.unsafe_ptr(),
        x.data.validity,
        x.null_count() > 0,
        y.unsafe_ptr(),
        y.data.validity,
        y.null_count() > 0,
        codes,
        groups,
        1,
    )


def group_cov[
    dx: DType, dy: DType
](
    x: Array[dx],
    y: Array[dy],
    codes: Array[DType.uint32],
    groups: Int,
    ddof: Int = 1,
) raises -> Array[DType.float64]:
    """Returns the covariance of two columns within each group.

    Args:
        x: One of the two columns.
        y: The other.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.
        ddof: Subtracted from the pairwise count to give the divisor. One is the
            sample covariance and is what pandas defaults to.

    Parameters:
        dx: The first column's dtype.
        dy: The second column's dtype.

    Returns:
        A column of `groups` covariances. A group with `ddof` or fewer pairwise
        present rows is NaN.

    Raises:
        If one of the workers the parallel route starts cannot be run. A
        reduction small enough to stay on one core cannot fail.
    """
    return _pair_core[want_corr=False](
        x.unsafe_ptr(),
        x.data.validity,
        x.null_count() > 0,
        y.unsafe_ptr(),
        y.data.validity,
        y.null_count() > 0,
        codes,
        groups,
        ddof,
    )


def _pair_core[
    dx: DType,
    dy: DType,
    //,
    want_corr: Bool,
    ox: ImmOrigin,
    oy: ImmOrigin,
](
    left: Pointer[Scalar[dx], ox],
    left_valid: Bitmap,
    left_has_null: Bool,
    right: Pointer[Scalar[dy], oy],
    right_valid: Bitmap,
    right_has_null: Bool,
    codes: Array[DType.uint32],
    groups: Int,
    ddof: Int,
) raises -> Array[DType.float64]:
    """Centres both columns on their pairwise means and accumulates the products.

    Two passes, for the reason `_var_core` takes two. The one pass form keeps the
    five raw sums and subtracts at the end, and the subtraction is where a
    correlation between two columns of timestamps stops having any significant
    digits left. This computes each group's two means first and then accumulates
    deviations, which is the same arithmetic the variance already does here and
    is stable for the same reason.

    Pairwise means rather than per column ones. A row where one of the two values
    is null contributes to neither sum, because a covariance is a statement about
    rows in which both were observed, and centring x on a mean taken over rows y
    was missing from would bias every product afterwards. That is also what
    pandas does, and it is why this cannot reuse `_mean_core`, which knows about
    one column's validity and not two.
    """
    var rows = len(codes)
    # Three tables in the first pass and three in the second, all float64, so a
    # worker's share of the budget is six times a group rather than one.
    var workers = _private_workers[DType.float64](rows, groups * 6)
    var bounds = _row_bounds(rows, workers)
    var at = codes.unsafe_ptr()

    var counts = Array[DType.float64](groups * workers)
    var sum_x = Array[DType.float64](groups * workers)
    var sum_y = Array[DType.float64](groups * workers)

    def totals(w: Int) raises {mut counts, mut sum_x, mut sum_y, imm}:
        var n_at = counts.unsafe_ptr().unsafe_offset(w * groups)
        var mx = sum_x.unsafe_ptr().unsafe_offset(w * groups)
        var my = sum_y.unsafe_ptr().unsafe_offset(w * groups)
        for i in range(bounds[w], bounds[w + 1]):
            if left_has_null and not left_valid.get(i):
                continue
            if right_has_null and not right_valid.get(i):
                continue
            var g = Int(at.unsafe_offset(i).unsafe_load())
            n_at.unsafe_offset(g).unsafe_store(
                n_at.unsafe_offset(g).unsafe_load() + 1.0
            )
            mx.unsafe_offset(g).unsafe_store(
                mx.unsafe_offset(g).unsafe_load()
                + left.unsafe_offset(i).unsafe_load().cast[DType.float64]()
            )
            my.unsafe_offset(g).unsafe_store(
                my.unsafe_offset(g).unsafe_load()
                + right.unsafe_offset(i).unsafe_load().cast[DType.float64]()
            )

    parallel_for(totals, workers)

    if workers > 1:
        counts = _merge_sums(counts, groups, workers)
        sum_x = _merge_sums(sum_x, groups, workers)
        sum_y = _merge_sums(sum_y, groups, workers)

    # The count is carried as a float64 so that it merges with the same
    # vectorized adder the two sums use. Every count here is a row count under
    # two to the fifty three, so it is exact and the divisions below are the
    # ones they would have been in an integer.
    var n_at = counts.unsafe_ptr()
    var mx = sum_x.unsafe_ptr()
    var my = sum_y.unsafe_ptr()

    for g in range(groups):
        var n = n_at.unsafe_offset(g).unsafe_load()
        if n == 0.0:
            continue
        mx.unsafe_offset(g).unsafe_store(mx.unsafe_offset(g).unsafe_load() / n)
        my.unsafe_offset(g).unsafe_store(my.unsafe_offset(g).unsafe_load() / n)

    var prod = Array[DType.float64](groups * workers)
    var square_x = Array[DType.float64](groups * workers if want_corr else 0)
    var square_y = Array[DType.float64](groups * workers if want_corr else 0)

    def spreads(w: Int) raises {mut prod, mut square_x, mut square_y, imm}:
        var sxy = prod.unsafe_ptr().unsafe_offset(w * groups)
        var xx = square_x.unsafe_ptr().unsafe_offset(
            w * groups if want_corr else 0
        )
        var yy = square_y.unsafe_ptr().unsafe_offset(
            w * groups if want_corr else 0
        )
        for i in range(bounds[w], bounds[w + 1]):
            if left_has_null and not left_valid.get(i):
                continue
            if right_has_null and not right_valid.get(i):
                continue
            var g = Int(at.unsafe_offset(i).unsafe_load())
            var a = (
                left.unsafe_offset(i).unsafe_load().cast[DType.float64]()
                - mx.unsafe_offset(g).unsafe_load()
            )
            var b = (
                right.unsafe_offset(i).unsafe_load().cast[DType.float64]()
                - my.unsafe_offset(g).unsafe_load()
            )
            sxy.unsafe_offset(g).unsafe_store(
                sxy.unsafe_offset(g).unsafe_load() + a * b
            )
            comptime if want_corr:
                xx.unsafe_offset(g).unsafe_store(
                    xx.unsafe_offset(g).unsafe_load() + a * a
                )
                yy.unsafe_offset(g).unsafe_store(
                    yy.unsafe_offset(g).unsafe_load() + b * b
                )

    parallel_for(spreads, workers)

    if workers > 1:
        prod = _merge_sums(prod, groups, workers)
        comptime if want_corr:
            square_x = _merge_sums(square_x, groups, workers)
            square_y = _merge_sums(square_y, groups, workers)

    var out = prod^
    var sxy = out.unsafe_ptr()
    var xx = square_x.unsafe_ptr()
    var yy = square_y.unsafe_ptr()

    for g in range(groups):
        var n = Int(n_at.unsafe_offset(g).unsafe_load())
        comptime if want_corr:
            var spread = (
                xx.unsafe_offset(g).unsafe_load()
                * yy.unsafe_offset(g).unsafe_load()
            )
            if n < 2 or not (spread > 0.0):
                # No complete pairs to correlate, or a column that never moves,
                # so there is no correlation to report. NaN and still valid, the
                # way a pandas float column says missing. See #170.
                sxy.unsafe_offset(g).unsafe_store(nan[DType.float64]())
                continue
            # The quotient is one in exact arithmetic when the two columns are
            # the same column, and floating point reaches 1.0000000000000002
            # often enough that a caller squaring it or taking its arccosine
            # would notice. Clamping is cheaper than pretending it cannot
            # happen.
            var r = sxy.unsafe_offset(g).unsafe_load() / sqrt(spread)
            if r > 1.0:
                r = 1.0
            elif r < -1.0:
                r = -1.0
            sxy.unsafe_offset(g).unsafe_store(r)
        else:
            var divisor = n - ddof
            if divisor <= 0:
                # Too few complete pairs to have a covariance. NaN and still
                # valid, which is what missing looks like here. See #170.
                sxy.unsafe_offset(g).unsafe_store(nan[DType.float64]())
                continue
            sxy.unsafe_offset(g).unsafe_store(
                sxy.unsafe_offset(g).unsafe_load() / Float64(divisor)
            )
    return out^


def aggregate_group_pair_any(
    x: AnyArray,
    y: AnyArray,
    kind: AggKind,
    codes: Array[DType.uint32],
    groups: Int,
) raises -> AnyArray:
    """Runs a two column reduction over columns whose dtypes are runtime values.

    The single column reductions dispatch on the dtype and instantiate one loop
    per dtype. A two column reduction would have to dispatch on both, which is a
    hundred and forty four instantiations of a loop that reads its inputs as
    float64 in either case.

    A hundred and forty four is only the count if the two dtypes are allowed to
    differ, and in nearly every call they do not, because the two columns are two
    measurements out of the same table. So the matching case is dispatched on,
    twelve instantiations, and it converts nothing: `_pair_core` reads a value
    and casts it to float64 in the same expression, so an instantiation on the
    column's own dtype does the conversion in a register as it goes. The mixed
    case still casts both columns to float64 and calls the one instantiation,
    which is two passes over the input and two allocations the size of it, and is
    the price of not compiling the other hundred and thirty two.

    Args:
        x: The first column.
        y: The second.
        kind: `CORR` or `COV`.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Returns:
        A float64 column of `groups` values.

    Raises:
        If the reduction is not a two column one, if either column has no
        physical layout as float64, or if the lengths disagree.
    """
    if len(x) != len(y):
        raise Error(
            "group by: a two column reduction needs two columns of the same"
            " length, got "
            + String(len(x))
            + " and "
            + String(len(y))
        )
    if x.is_string() or y.is_string():
        raise Error("group by: a two column reduction needs numeric columns")

    # The two columns are the same dtype in nearly every call, because they are
    # two measurements out of the same table, and that case can be answered
    # without converting either of them. `_pair_core` reads a value and casts it
    # to float64 in the same expression, so an instantiation on the column's own
    # dtype does the conversion in a register as it goes, one instruction per
    # value, instead of materialising two float64 copies of the input first.
    # Twelve instantiations rather than a hundred and forty four, which is what
    # made the cast worth taking in the first place.
    if x.dtype() == y.dtype():
        comptime for dt in ALL:
            if x.dtype() == dt:
                ref a = x.as_typed_view[dt]()
                ref b = y.as_typed_view[dt]()
                if kind == AggKind.CORR:
                    return AnyArray(group_corr(a, b, codes, groups))
                if kind == AggKind.COV:
                    return AnyArray(
                        group_cov(a, b, codes, groups, Int(kind.param))
                    )
                raise Error("group by: unsupported two column aggregation")

    var a = cast_any(x, DType.float64).into_typed[DType.float64]()
    var b = cast_any(y, DType.float64).into_typed[DType.float64]()
    if kind == AggKind.CORR:
        return AnyArray(group_corr(a, b, codes, groups))
    if kind == AggKind.COV:
        return AnyArray(group_cov(a, b, codes, groups, Int(kind.param)))
    raise Error("group by: unsupported two column aggregation")


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
        If the reduction is not one of the thirteen single column ones.
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
    """Picks the reduction. One instantiation per dtype, thirteen loops inside.
    """
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
    if kind == AggKind.VAR:
        return AnyArray(
            _var_core[want_std=False](
                source, validity, has_null, codes, groups, Int(kind.param)
            )
        )
    if kind == AggKind.STD:
        return AnyArray(
            _var_core[want_std=True](
                source, validity, has_null, codes, groups, Int(kind.param)
            )
        )
    if kind == AggKind.SEM:
        return AnyArray(
            _sem_core(
                source, validity, has_null, codes, groups, Int(kind.param)
            )
        )
    if kind == AggKind.SKEW:
        return AnyArray(_skew_core(source, validity, has_null, codes, groups))
    if kind == AggKind.MEDIAN or kind == AggKind.QUANTILE:
        if not (kind.param >= 0.0 and kind.param <= 1.0):
            raise Error(
                "group by: a quantile must be between 0 and 1, got "
                + String(kind.param)
            )
        return AnyArray(
            _quantile_core(
                source, validity, has_null, codes, groups, kind.param
            )
        )
    if kind == AggKind.NUNIQUE:
        return AnyArray(
            _nunique_core(source, validity, has_null, codes, groups)
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

    # As in `cast_any` and `argsort_any_into`: uint8 is in ALL and a string
    # column would match it, so a sum over a column of names would return a
    # number taken from the first byte of every view rather than an error.
    if col.is_string():
        return aggregate_group_strings(col.strings(), kind, codes, groups)

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


def aggregate_group_strings(
    col: StringArray, kind: AggKind, codes: Array[DType.uint32], groups: Int
) raises -> AnyArray:
    """Runs one grouped reduction over a column of text.

    Six of the thirteen reductions mean something over bytes and the rest do not.
    `SIZE` and `COUNT` never look at a value, `FIRST`, `LAST`, `MIN` and `MAX`
    report a value the column held, and `NUNIQUE` counts values without ordering
    them. A sum of names is not a slow operation, it is not an operation, and the
    other six raise saying so rather than returning something defensible.

    The four that report a value do it by keeping a row number per group and
    gathering at the end. Nothing is copied while the scan runs, which matters
    because the alternative is holding a `String` per group and rewriting it
    every time a smaller one turns up.

    `NUNIQUE` is the one that reuses the number path outright. Factorizing the
    column gives every distinct value an ordinal, two rows hold the same bytes
    exactly when they got the same ordinal, so counting distinct values in a
    group is counting distinct ordinals and the existing uint32 core does that.

    Args:
        col: The text column being aggregated.
        kind: Which reduction.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Returns:
        A column of `groups` values, text for the four that report a value and
        int64 for the three that count.

    Raises:
        If the reduction is one that text has no meaning for.
    """
    if kind == AggKind.SIZE:
        return AnyArray(group_size(codes, groups))
    if kind == AggKind.COUNT:
        return AnyArray(
            _count_core(col.validity, col.null_count() > 0, codes, groups)
        )
    if kind == AggKind.NUNIQUE:
        var ordinals = factorize_strings(StringArray(copy=col)).into_codes()
        # The ordinals carry no nulls of their own, because a null row is a group
        # like any other to the factorize. The count has to skip those rows, so
        # the column's own validity is what the core is handed.
        ordinals.data.validity = Bitmap(copy=col.validity)
        return _dispatch_core(
            ordinals.unsafe_ptr(),
            ordinals.data.validity,
            col.null_count() > 0,
            AggKind.NUNIQUE,
            codes,
            groups,
        )

    var wants_edge = kind == AggKind.FIRST or kind == AggKind.LAST
    var wants_extreme = kind == AggKind.MIN or kind == AggKind.MAX
    if not (wants_edge or wants_extreme):
        raise Error(
            "group by: " + String(kind) + " is not defined for a string column"
        )

    var at = List[Int](capacity=groups)
    for _ in range(groups):
        at.append(-1)

    var at_ptr = at.unsafe_ptr()
    var group_of = codes.unsafe_ptr()
    if wants_edge:
        var first = kind == AggKind.FIRST
        for i in range(len(codes)):
            if not col.is_valid(i):
                continue
            var g = Int(group_of.unsafe_offset(i).unsafe_load())
            if first and at_ptr.unsafe_offset(g).unsafe_load() >= 0:
                continue
            at_ptr.unsafe_offset(g).unsafe_store(i)
    else:
        var want_min = kind == AggKind.MIN
        for i in range(len(codes)):
            if not col.is_valid(i):
                continue
            var g = Int(group_of.unsafe_offset(i).unsafe_load())
            var held = at_ptr.unsafe_offset(g).unsafe_load()
            if held < 0:
                at_ptr.unsafe_offset(g).unsafe_store(i)
                continue
            var order = col.compare_elements(i, held)
            if order < 0 if want_min else order > 0:
                at_ptr.unsafe_offset(g).unsafe_store(i)

    var builder = StringBuilder(capacity=groups)
    for g in range(groups):
        var row = at_ptr.unsafe_offset(g).unsafe_load()
        if row < 0:
            builder.append_null()
        else:
            builder.append(col.unsafe_bytes(row))
    return AnyArray(builder^.finish())
