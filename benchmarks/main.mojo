"""The microbenchmark suite that runs on every pull request.

docs/specs/10-benchmarks.md splits the measurement work in two. Anything that
compares firepanda against pandas, Polars or DuckDB lives in `tamnd/firepanda-bench`,
because those comparisons need Python installed, Docker images, datasets and a
machine nobody wants in a pull request. What lives here is the other half: the
kernels, measured against themselves, on every commit, with no dependencies at all.

The two halves answer different questions. The other repository answers "are we
fast", this one answers "did that commit make something slower", and the second
question is the one that has to be answered while the author still remembers what
they changed.

Methodology follows the same document. Ten repetitions, the median reported, the
interquartile range reported next to it, because a single number with no spread is
not a measurement. Each repetition is itself a `std.benchmark` run, so the number
being taken ten medians of is already an average over however many iterations fit
in the minimum runtime.

What is measured is the storage layer, the validity bitmap, the buffer allocator
and its pool, the typed array and the dtype dispatch, and then the kernels that
sit on top of it. Several of the kernel rows are paired with the scalar twin that
`tests/fuzz/kernel.mojo` checks them against, so the table says what the
vectorized version is worth as well as what it costs. The twins are not a
strawman, they are the implementation the library would have if nobody had
bothered, which is the comparison worth printing.

Usage:
    mojo run -I . benchmarks/main.mojo [--rows=N] [--repetitions=N]
        [--min-time=MILLISECONDS] [--max-time=MILLISECONDS] [--json=PATH]
        [--label=NAME] [--filter=SUBSTRING]

`--label` goes in the JSON so a result file says which machine produced it, which
docs/specs/10-benchmarks.md requires and which is easy to forget until two result
files disagree and nobody can say why.
"""

from std.benchmark import Unit, keep, run
from std.collections.span import Span
from std.sys import argv
from std.sys.info import (
    CompilationTarget,
    num_logical_cores,
    num_physical_cores,
    simd_width_of,
)
from std.time import perf_counter_ns

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import (
    StringArray,
    StringBuilder,
    strings_from_list,
)
from firepanda.array.strview import PREFIX_LENGTH
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.buffer.pool import BufferPool
from firepanda.dtype.dispatch import dispatch
from firepanda.dtype.lists import NUMERIC
from firepanda.frame.display import DisplayOptions, render_column
from firepanda.frame.frame import DataFrame
from firepanda.frame.groupby import AggSpec
from firepanda.frame.series import Series
from firepanda.hash import (
    DEFAULT_SEED,
    DIRECT_LIMIT,
    HashTable,
    factorize,
    factorize_dict,
    factorize_strings,
    group_ordinals,
    hash_into,
    mix,
    radix_partition,
)
from firepanda.io.parse import parse_float, parse_int
from firepanda.io.scalar import scan_csv_scalar
from firepanda.io.scan import default_dialect, field_bytes, scan_csv
from firepanda.join import JoinKind, join_indices
from firepanda.frame.concat import concat
from firepanda.kernel import (
    AggKind,
    add,
    aggregate_group_any,
    argsort,
    argsort_any,
    argsort_multi,
    cast_to,
    coalesce,
    concat_arrays,
    concat_two_any,
    divide,
    fill_forward,
    filter_any,
    filter_rows,
    group_first,
    group_mean,
    group_median,
    group_min,
    group_nunique,
    group_size,
    group_std,
    group_sum,
    group_var,
    less,
    mean_of,
    is_null,
    min_of,
    multiply,
    sum_of,
    take_any,
    take_rows,
)
from firepanda.testing.rng import Rng
from firepanda.kernel.scalar import (
    add_scalar,
    filter_scalar,
    min_scalar,
    sum_scalar,
)
from firepanda.version import version

comptime DEFAULT_ROWS = 1 << 20
"""A million rows. Large enough to leave cache, small enough to run in a pull request."""

comptime DEFAULT_REPETITIONS = 10
"""The repetition count from docs/specs/10-benchmarks.md."""

comptime BENCH_DTYPE = DType.int64
"""The dtype the array benchmarks use. Eight bytes is the common case for a key column."""


struct Options(Copyable, Movable):
    """What the suite was asked to do."""

    var rows: Int
    """The number of rows, and the number of bits, per benchmark."""

    var repetitions: Int
    """How many times each benchmark is measured before the median is taken."""

    var min_seconds: Float64
    """The minimum wall clock time of a single repetition."""

    var max_seconds: Float64
    """The most wall clock time a single repetition may spend. Zero means the
    default, which is three times the minimum plus a quarter of a second."""

    var json_path: String
    """Where to write the machine readable results. Empty means nowhere."""

    var label: String
    """A name for the machine, recorded in the results."""

    var filter: String
    """Only run benchmarks whose name contains this. Empty means all of them."""

    def __init__(out self):
        """Constructs the defaults."""
        self.rows = DEFAULT_ROWS
        self.repetitions = DEFAULT_REPETITIONS
        self.min_seconds = 0.1
        self.max_seconds = 0.0
        self.json_path = String("")
        self.label = String("")
        self.filter = String("")


def parse_options() raises -> Options:
    """Reads the command line.

    Returns:
        The options, with anything unspecified left at its default.

    Raises:
        If a flag is not recognized or its value is not a number.
    """
    var options = Options()
    var args = argv()
    for i in range(1, len(args)):
        var arg = args[i]
        if arg.startswith("--rows="):
            options.rows = Int(arg[byte=7:])
        elif arg.startswith("--repetitions="):
            options.repetitions = Int(arg[byte=14:])
        elif arg.startswith("--min-time="):
            options.min_seconds = Float64(Int(arg[byte=11:])) / 1000.0
        elif arg.startswith("--max-time="):
            options.max_seconds = Float64(Int(arg[byte=11:])) / 1000.0
        elif arg.startswith("--json="):
            options.json_path = String(arg[byte=7:])
        elif arg.startswith("--label="):
            options.label = String(arg[byte=8:])
        elif arg.startswith("--filter="):
            options.filter = String(arg[byte=9:])
        else:
            raise Error(String("unrecognized argument: ", arg))
    if options.rows < 64:
        raise Error("--rows must be at least 64")
    if options.repetitions < 1:
        raise Error("--repetitions must be at least 1")
    return options^


def fixed(value: Float64, decimals: Int = 2) -> String:
    """Formats a number with a fixed number of decimal places.

    Mojo prints a `Float64` at full precision, which is right for the JSON and
    unreadable in a table.

    Args:
        value: The number. Expected to be finite and not negative.
        decimals: How many digits after the point.

    Returns:
        The formatted number.
    """
    if value != value:
        return String("nan")
    var scale = 1
    for _ in range(decimals):
        scale *= 10
    var units = Int(value * Float64(scale) + 0.5)
    var whole = units // scale
    if decimals == 0:
        return String(whole)
    var text = String(units % scale)
    while text.byte_length() < decimals:
        text = String("0", text)
    return String(whole, ".", text)


def bytes_of(var text: String) -> List[UInt8]:
    """Copies a string's bytes into a list the CSV scanner can span.

    Args:
        text: The string.

    Returns:
        The bytes, without a terminator.
    """
    var out = List[UInt8](capacity=text.byte_length())
    var ptr = text.unsafe_ptr()
    for i in range(text.byte_length()):
        out.append(ptr.unsafe_offset(i).unsafe_load())
    return out^


def pad(text: String, width: Int) -> String:
    """Pads a string on the right to a width.

    Args:
        text: The string.
        width: The width in bytes.

    Returns:
        The padded string, or the original if it is already wider.
    """
    var out = text
    while out.byte_length() < width:
        out += " "
    return out^


def lpad(text: String, width: Int) -> String:
    """Pads a string on the left to a width, so a column of numbers lines up.

    Args:
        text: The string.
        width: The width in bytes.

    Returns:
        The padded string, or the original if it is already wider.
    """
    var out = text
    while out.byte_length() < width:
        out = String(" ", out)
    return out^


def duration_text(seconds: Float64) -> String:
    """Formats a duration in whichever unit keeps it readable.

    Args:
        seconds: The duration.

    Returns:
        The duration with its unit.
    """
    if seconds >= 1.0:
        return String(fixed(seconds, 3), " s")
    if seconds >= 1e-3:
        return String(fixed(seconds * 1e3, 3), " ms")
    if seconds >= 1e-6:
        return String(fixed(seconds * 1e6, 3), " us")
    return String(fixed(seconds * 1e9, 1), " ns")


def rate_text(per_second: Float64, unit: String) -> String:
    """Formats a throughput with an SI prefix.

    Args:
        per_second: The rate.
        unit: What is being counted, such as "bits" or "rows".

    Returns:
        The rate with its prefix and unit.
    """
    if per_second >= 1e9:
        return String(fixed(per_second / 1e9), " G", unit, "/s")
    if per_second >= 1e6:
        return String(fixed(per_second / 1e6), " M", unit, "/s")
    if per_second >= 1e3:
        return String(fixed(per_second / 1e3), " K", unit, "/s")
    return String(fixed(per_second), " ", unit, "/s")


def quantile(values: List[Float64], q: Float64) -> Float64:
    """Returns a quantile of an already sorted list, interpolating between ranks.

    Args:
        values: The samples, sorted ascending. Must not be empty.
        q: The quantile, between zero and one.

    Returns:
        The interpolated quantile.
    """
    if len(values) == 1:
        return values[0]
    var position = q * Float64(len(values) - 1)
    var lower = Int(position)
    var upper = lower + 1
    if upper > len(values) - 1:
        upper = len(values) - 1
    var fraction = position - Float64(lower)
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


struct Measurement(Copyable, Movable):
    """One benchmark and its repetitions."""

    var name: String
    """The benchmark's name, in `area/operation` form."""

    var unit: String
    """What one item is, such as "bits" or "rows"."""

    var items: Int
    """How many items one iteration processes."""

    var samples: List[Float64]
    """Seconds per iteration, one entry per repetition, sorted ascending."""

    def __init__(
        out self,
        name: String,
        unit: String,
        items: Int,
        var samples: List[Float64],
    ):
        """Constructs a measurement and sorts its samples.

        Args:
            name: The benchmark's name.
            unit: What one item is.
            items: How many items one iteration processes.
            samples: Seconds per iteration, one entry per repetition.
        """
        self.name = name
        self.unit = unit
        self.items = items
        sort(samples)
        self.samples = samples^

    def median(self) -> Float64:
        """Returns the median seconds per iteration.

        Returns:
            The median.
        """
        return quantile(self.samples, 0.5)

    def q1(self) -> Float64:
        """Returns the first quartile of the seconds per iteration.

        Returns:
            The first quartile.
        """
        return quantile(self.samples, 0.25)

    def q3(self) -> Float64:
        """Returns the third quartile of the seconds per iteration.

        Returns:
            The third quartile.
        """
        return quantile(self.samples, 0.75)

    def spread(self) -> Float64:
        """Returns the interquartile range as a fraction of the median.

        This is the number that says whether the median can be trusted. A run
        with a spread over a few percent was measured on a busy machine and its
        comparison against a previous run means very little.

        Returns:
            The interquartile range over the median, or zero if the median is zero.
        """
        var middle = self.median()
        if middle <= 0.0:
            return 0.0
        return (self.q3() - self.q1()) / middle

    def per_item_ns(self) -> Float64:
        """Returns the median nanoseconds per item.

        Returns:
            The nanoseconds, or zero if the benchmark processes no items.
        """
        if self.items <= 0:
            return 0.0
        return self.median() * 1e9 / Float64(self.items)

    def items_per_second(self) -> Float64:
        """Returns the median throughput.

        Returns:
            Items per second, or zero if the median is zero.
        """
        var middle = self.median()
        if middle <= 0.0:
            return 0.0
        return Float64(self.items) / middle


struct Harness(Movable):
    """Runs the benchmarks, prints the table as it goes and keeps the results.
    """

    var options: Options
    """What the suite was asked to do."""

    var results: List[Measurement]
    """Every benchmark that has run so far, in the order it ran."""

    var skipped: Int
    """How many benchmarks the filter excluded."""

    def __init__(out self, var options: Options):
        """Constructs a harness.

        Args:
            options: What the suite was asked to do.
        """
        self.options = options^
        self.results = List[Measurement]()
        self.skipped = 0

    def wanted(self, name: String) -> Bool:
        """Returns whether a benchmark passes the filter.

        Args:
            name: The benchmark's name.

        Returns:
            True if it should run.
        """
        if self.options.filter.byte_length() == 0:
            return True
        return self.options.filter in name

    def max_seconds(self) -> Float64:
        """Returns the wall clock ceiling on one repetition.

        Returns:
            The `--max-time` value, or three times the minimum plus a quarter of
            a second when it was not given.
        """
        if self.options.max_seconds > 0.0:
            return self.options.max_seconds
        return self.options.min_seconds * 3.0 + 0.25

    def record(
        mut self,
        name: String,
        unit: String,
        items: Int,
        body: Some[def() raises],
    ) raises:
        """Measures one benchmark and prints its row.

        Args:
            name: The benchmark's name, in `area/operation` form.
            unit: What one item is.
            items: How many items one iteration of `body` processes.
            body: The work, called repeatedly.

        Raises:
            If the benchmarked function raises.
        """
        if not self.wanted(name):
            self.skipped += 1
            return

        var samples = List[Float64](capacity=self.options.repetitions)
        for _ in range(self.options.repetitions):
            # The upper bound matters more than it looks. A benchmark whose body
            # takes a nanosecond needs hundreds of millions of iterations to fill
            # the minimum runtime, and without a cap a single row of this table
            # can own the pull request for half a minute.
            #
            # It is also what the suite's total runtime is actually made of, which
            # is not obvious and was worth measuring. `run` keeps sampling until
            # this bound rather than stopping at the minimum, so nearly every row
            # costs the bound, and the suite costs the bound times the number of
            # benchmarks times the repetitions almost regardless of the row count.
            # Halving the rows moved a full run from 310 s to 276 s; halving this
            # halves it. That is why it is a flag.
            var report = run(
                body,
                num_warmup_iters=1,
                min_runtime_secs=self.options.min_seconds,
                max_runtime_secs=self.max_seconds(),
            )
            samples.append(report.mean(Unit.s))

        var measurement = Measurement(name, unit, items, samples^)
        print(
            pad(measurement.name, 26),
            lpad(duration_text(measurement.median()), 12),
            lpad(String(fixed(measurement.spread() * 100.0, 1), "%"), 8),
            lpad(String(fixed(measurement.per_item_ns(), 3), " ns"), 12),
            lpad(
                rate_text(measurement.items_per_second(), measurement.unit), 16
            ),
        )
        self.results.append(measurement^)


def sum_erased[dt: DType](col: AnyArray) raises -> Float64:
    """Adds up an erased column at its own dtype.

    This is the operation the dispatch benchmark dispatches. It is the same shape
    as `kernel.sum_of` on purpose, so the overhead measured here is the overhead
    the real kernel pays when it is reached through an erased column. It is not
    the real kernel and it is not named like it, because a benchmark that
    measures the thing it is standing in for is a benchmark nobody can read.

    Args:
        col: The column, whose dtype dispatch has already proved is `dt`.

    Parameters:
        dt: The dtype.

    Returns:
        The sum as a double.
    """
    var total = Float64(0)
    var ptr = col.unsafe_ptr[dt]()
    for i in range(len(col)):
        total += Float64(ptr.unsafe_offset(i).unsafe_load())
    return total


def bench_bitmap(mut harness: Harness) raises:
    """Measures the validity bitmap.

    Every row of every column carries one of these bits, so a null aware kernel
    reads the bitmap as often as it reads the values. The operations here are the
    ones a filter, a join and a null count are built out of.

    Args:
        harness: The harness.

    Raises:
        If a benchmark raises.
    """
    var rows = harness.options.rows
    var bits = Bitmap(rows, all_valid=False)
    for i in range(0, rows, 3):
        bits.set(i, True)
    var other = Bitmap(rows, all_valid=True)
    for i in range(0, rows, 5):
        other.set(i, False)

    def set_bits() raises {mut bits, imm rows}:
        for i in range(rows):
            bits.set(i, i & 3 != 0)

    harness.record("bitmap/set", "bits", rows, set_bits)

    # `keep` on the input as well as the output, in every read only benchmark
    # below. Without it the body is loop invariant, and since `std.benchmark`
    # calls it from an inlined loop the compiler is free to run it once and reuse
    # the answer. That is not a theoretical worry: the first version of the one
    # row dispatch benchmark reported a fifth of a nanosecond per call, which is
    # less than a cycle, because the whole thing had been hoisted.
    def get_bits() raises {imm bits, imm rows}:
        keep(bits)
        var ones = 0
        for i in range(rows):
            if bits.get(i):
                ones += 1
        keep(ones)

    harness.record("bitmap/get", "bits", rows, get_bits)

    def count_ones() raises {imm bits}:
        keep(bits)
        var ones = bits.count_ones()
        keep(ones)

    harness.record("bitmap/count_ones", "bits", rows, count_ones)

    def set_range() raises {mut bits, imm rows}:
        bits.set_range(1, rows - 1, True)

    harness.record("bitmap/set_range", "bits", rows, set_range)

    def and_with() raises {mut bits, imm other}:
        bits.and_with(other)

    harness.record("bitmap/and_with", "bits", rows, and_with)

    def or_with() raises {mut bits, imm other}:
        bits.or_with(other)

    harness.record("bitmap/or_with", "bits", rows, or_with)

    def invert() raises {mut bits}:
        bits.invert()

    harness.record("bitmap/invert", "bits", rows, invert)

    def slice_aligned() raises {imm bits, imm rows}:
        keep(bits)
        var part = bits.slice(64, rows - 64)
        keep(part)

    harness.record("bitmap/slice_aligned", "bits", rows - 128, slice_aligned)

    # The unaligned case is the one that matters, because every slice of a sliced
    # column lands here and the implementation has to shift every byte.
    def slice_unaligned() raises {imm bits, imm rows}:
        keep(bits)
        var part = bits.slice(3, rows - 64)
        keep(part)

    harness.record("bitmap/slice_unaligned", "bits", rows - 67, slice_unaligned)


def bench_buffer(mut harness: Harness) raises:
    """Measures allocation, with and without the pool.

    The pool exists on the theory that an engine allocates the same handful of
    sizes over and over and that going to the system allocator each time is
    wasteful. That theory should be visible as a number, and if it ever stops
    being visible the pool should be deleted rather than defended.

    Args:
        harness: The harness.

    Raises:
        If a benchmark raises.
    """
    var size = harness.options.rows * 8
    var pool = BufferPool()

    def fresh() raises {imm size}:
        var buffer = Buffer(size)
        keep(buffer)

    harness.record("buffer/alloc_fresh", "allocs", 1, fresh)

    # Warm the pool so the benchmark measures the hit path rather than the first miss.
    pool.give(Buffer(size))

    def pooled() raises {mut pool, imm size}:
        var buffer = pool.take(size)
        keep(buffer)
        pool.give(buffer^)

    harness.record("buffer/pool_cycle", "allocs", 1, pooled)

    var small = harness.options.rows // 16

    def fresh_small() raises {imm small}:
        var buffer = Buffer(small)
        keep(buffer)

    harness.record("buffer/alloc_fresh_small", "allocs", 1, fresh_small)

    pool.give(Buffer(small))

    def pooled_small() raises {mut pool, imm small}:
        var buffer = pool.take(small)
        keep(buffer)
        pool.give(buffer^)

    harness.record("buffer/pool_cycle_small", "allocs", 1, pooled_small)


def bench_array(mut harness: Harness) raises:
    """Measures the typed array.

    The three sums are the point of the whole design. The scalar loop is what a
    row at a time engine does, the SIMD loop is what the kernels will do, and the
    null aware loop is what they will do when the column actually has nulls. The
    gap between the first two is the argument for the layout and the gap between
    the last two is the argument for keeping a fast path when there are no nulls.

    Args:
        harness: The harness.

    Raises:
        If a benchmark raises.
    """
    var rows = harness.options.rows
    comptime width = simd_width_of[BENCH_DTYPE]()

    var values = Array[BENCH_DTYPE](rows)
    for i in range(rows):
        values[i] = Scalar[BENCH_DTYPE](i % 1000)

    def fill() raises {mut values, imm rows}:
        for i in range(rows):
            values[i] = Scalar[BENCH_DTYPE](i % 1000)

    harness.record("array/fill", "rows", rows, fill)

    def sum_scalar() raises {imm values, imm rows}:
        keep(values)
        var total = Scalar[BENCH_DTYPE](0)
        for i in range(rows):
            total += values[i]
        keep(total)

    harness.record("array/sum_scalar", "rows", rows, sum_scalar)

    def sum_simd() raises {imm values, imm rows}:
        keep(values)
        var total = SIMD[BENCH_DTYPE, width](0)
        for i in range(0, rows, width):
            total += values.load[width](i)
        var scalar = total.reduce_add()
        keep(scalar)

    harness.record("array/sum_simd", "rows", rows, sum_simd)

    var sparse = Array[BENCH_DTYPE](rows)
    for i in range(rows):
        sparse[i] = Scalar[BENCH_DTYPE](i % 1000)
    for i in range(0, rows, 7):
        sparse.set_null(i)

    def sum_null_aware() raises {imm sparse, imm rows}:
        keep(sparse)
        var total = Scalar[BENCH_DTYPE](0)
        for i in range(rows):
            if sparse.is_valid(i):
                total += sparse[i]
        keep(total)

    harness.record("array/sum_null_aware", "rows", rows, sum_null_aware)

    def null_count() raises {imm sparse}:
        keep(sparse)
        var nulls = sparse.null_count()
        keep(nulls)

    harness.record("array/null_count", "rows", rows, null_count)

    # Element by element today, which is why the number is worth recording. M1
    # replaces this with a memcpy of the values and a shift of the validity.
    def slice_copy() raises {imm values, imm rows}:
        keep(values)
        var part = values.slice(1, rows - 1)
        keep(part)

    harness.record("array/slice_copy", "rows", rows - 2, slice_copy)

    def deep_copy() raises {imm values}:
        keep(values)
        var other = Array[BENCH_DTYPE](copy=values)
        keep(other)

    harness.record("array/copy", "rows", rows, deep_copy)


def bench_kernel(mut harness: Harness) raises:
    """Measures the kernel layer.

    Three of these rows are there to be read together. `kernel/sum_dense` and
    `kernel/sum_sparse` should be the same number, because `sum_of` never looks at
    the validity bitmap and a null is a zero, and `kernel/sum_twin` is what the
    same work costs written the obvious way with a validity check per row. If the
    first two ever drift apart, the fast path has grown a branch.

    Every other row includes the cost of allocating the output column, because
    every one of these kernels returns a new column and pretending otherwise would
    flatter the numbers. The allocator rows in `bench_buffer` are how to subtract
    it back out.

    Args:
        harness: The harness.

    Raises:
        If a benchmark raises.
    """
    var rows = harness.options.rows

    var dense = Array[BENCH_DTYPE](rows)
    var other = Array[BENCH_DTYPE](rows)
    for i in range(rows):
        dense[i] = Scalar[BENCH_DTYPE](i % 1000)
        other[i] = Scalar[BENCH_DTYPE](i % 97 + 1)

    var sparse = Array[BENCH_DTYPE](copy=dense)
    for i in range(0, rows, 7):
        sparse.set_null(i)

    def sum_dense() raises {imm dense}:
        keep(dense)
        var total = sum_of(dense)
        keep(total.value)

    harness.record("kernel/sum_dense", "rows", rows, sum_dense)

    def sum_sparse() raises {imm sparse}:
        keep(sparse)
        var total = sum_of(sparse)
        keep(total.value)

    harness.record("kernel/sum_sparse", "rows", rows, sum_sparse)

    def sum_twin() raises {imm sparse}:
        keep(sparse)
        var total = sum_scalar(sparse)
        keep(total)

    harness.record("kernel/sum_twin", "rows", rows, sum_twin)

    def min_dense() raises {imm dense}:
        keep(dense)
        var low = min_of(dense)
        keep(low.value)

    harness.record("kernel/min_dense", "rows", rows, min_dense)

    # min cannot skip the bitmap the way sum can, so unlike the pair above these
    # two are expected to differ. The gap is what the three-way word split buys.
    def min_sparse() raises {imm sparse}:
        keep(sparse)
        var low = min_of(sparse)
        keep(low.value)

    harness.record("kernel/min_sparse", "rows", rows, min_sparse)

    def min_twin() raises {imm sparse}:
        keep(sparse)
        var low = min_scalar(sparse)
        keep(low[0])

    harness.record("kernel/min_twin", "rows", rows, min_twin)

    def mean_sparse() raises {imm sparse}:
        keep(sparse)
        var avg = mean_of(sparse)
        keep(avg.value)

    harness.record("kernel/mean_sparse", "rows", rows, mean_sparse)

    def add_dense() raises {imm dense, imm other}:
        keep(dense)
        var out = add(dense, other)
        keep(out)

    harness.record("kernel/add_dense", "rows", rows, add_dense)

    def add_sparse() raises {imm sparse, imm other}:
        keep(sparse)
        var out = add(sparse, other)
        keep(out)

    harness.record("kernel/add_sparse", "rows", rows, add_sparse)

    def add_twin() raises {imm sparse, imm other}:
        keep(sparse)
        var out = add_scalar(sparse, other)
        keep(out)

    harness.record("kernel/add_twin", "rows", rows, add_twin)

    def multiply_dense() raises {imm dense, imm other}:
        keep(dense)
        var out = multiply(dense, other)
        keep(out)

    harness.record("kernel/multiply_dense", "rows", rows, multiply_dense)

    def divide_dense() raises {imm dense, imm other}:
        keep(dense)
        var out = divide(dense, other)
        keep(out)

    harness.record("kernel/divide_dense", "rows", rows, divide_dense)

    def compare_dense() raises {imm dense, imm other}:
        keep(dense)
        var out = less(dense, other)
        keep(out)

    harness.record("kernel/less_dense", "rows", rows, compare_dense)

    def cast_widen() raises {imm dense}:
        keep(dense)
        var out = cast_to[BENCH_DTYPE, DType.float64](dense)
        keep(out)

    harness.record("kernel/cast_i64_f64", "rows", rows, cast_widen)

    def cast_narrow() raises {imm dense}:
        keep(dense)
        var out = cast_to[BENCH_DTYPE, DType.int16](dense)
        keep(out)

    harness.record("kernel/cast_i64_i16", "rows", rows, cast_narrow)

    # A stride that is coprime with the row count, so the gather walks the whole
    # column and misses cache the way a real join's take does. A sequential index
    # list would measure the prefetcher instead.
    var scattered = List[Int](capacity=rows)
    for i in range(rows):
        scattered.append((i * 7919) % rows)

    def take_scattered() raises {imm dense, imm scattered}:
        keep(dense)
        var out = take_rows(dense, scattered)
        keep(out)

    harness.record("kernel/take_scattered", "rows", rows, take_scattered)

    var mask = less(other, dense)

    def filter_half() raises {imm dense, imm mask}:
        keep(dense)
        var out = filter_rows(dense, mask)
        keep(out)

    harness.record("kernel/filter", "rows", rows, filter_half)

    # The branchless copy only applies when the filtered column has no nulls, so
    # the version that does have them is measured too. Reporting only the fast
    # path would be reporting the best case and calling it the number.
    def filter_sparse() raises {imm sparse, imm mask}:
        keep(sparse)
        var out = filter_rows(sparse, mask)
        keep(out)

    harness.record("kernel/filter_sparse", "rows", rows, filter_sparse)

    def filter_twin() raises {imm dense, imm mask}:
        keep(dense)
        var out = filter_scalar(dense, mask)
        keep(out)

    harness.record("kernel/filter_twin", "rows", rows, filter_twin)


def bench_sort(mut harness: Harness) raises:
    """Measures the radix sort, on the shapes that change how many passes it does.

    The rows to read together are `int64_random` and `int64_small`. They are the
    same dtype and the same row count and differ only in the range of the values,
    which is what decides how many of the eight digits have more than one bucket
    occupied. If the skip is working, the second is several times faster than the
    first, and if it is not then the two are the same number and the check in
    `_radix_sort` is dead code.

    `uint32_random` is there for the other half of the pass count question. An
    unsigned dtype narrower than eight bytes leaves the high bytes of the key
    zero by construction rather than by luck, so it should cost four passes on
    any data at all.

    `int64_presorted` measures a column that arrives already in order, which a
    query plan hits constantly. It is the slowest row in the group and that is
    not a typo. It runs three passes, the same three that a random column of the
    same value range runs, and it takes 2.6 times as long: 22.8 ms against 8.7 ms
    on the reference machine, verified by counting the passes both ways. The
    cause is the scatter. Sequential input sends consecutive rows to consecutive
    buckets, so the 256 write cursors are visited in strict round robin and every
    one of them has been evicted by the time it comes round again, where random
    input revisits a cursor early often enough to keep some of them live. Fixing
    it means staging the writes, and that is a change to make with the number in
    front of us rather than on the way past.

    `std_sort_int64` sorts a plain `List` with the standard library. It is not a
    like for like comparison, because it moves values rather than producing a
    permutation and it has no nulls to carry, so it is reported as a reference
    row rather than gated on. What it is good for is answering whether writing a
    sort was worth it at all.

    Args:
        harness: The harness.

    Raises:
        If a benchmark raises.
    """
    var rows = harness.options.rows
    var rng = Rng(0x51AB1E)

    var wide = Array[DType.int64](rows)
    var small = Array[DType.int64](rows)
    var narrow = Array[DType.uint32](rows)
    var real = Array[DType.float64](rows)
    var presorted = Array[DType.int64](rows)
    var sparse = Array[DType.int64](rows)
    for i in range(rows):
        var draw = rng.next_u64()
        wide[i] = Int64(draw)
        small[i] = Int64(draw % 1000)
        narrow[i] = UInt32(draw & 0xFFFFFFFF)
        real[i] = Float64(Int64(draw % 1000000)) / Float64(64.0)
        presorted[i] = Int64(i)
        sparse[i] = Int64(draw % 1000)

    for i in range(rows):
        if i % 8 == 0:
            sparse.set_null(i)

    def sort_wide() raises {imm wide}:
        keep(wide)
        var order = argsort(wide)
        keep(order)

    harness.record("sort/int64_random", "rows", rows, sort_wide)

    def sort_small() raises {imm small}:
        keep(small)
        var order = argsort(small)
        keep(order)

    harness.record("sort/int64_small", "rows", rows, sort_small)

    def sort_narrow() raises {imm narrow}:
        keep(narrow)
        var order = argsort(narrow)
        keep(order)

    harness.record("sort/uint32_random", "rows", rows, sort_narrow)

    def sort_real() raises {imm real}:
        keep(real)
        var order = argsort(real)
        keep(order)

    harness.record("sort/float64_random", "rows", rows, sort_real)

    def sort_presorted() raises {imm presorted}:
        keep(presorted)
        var order = argsort(presorted)
        keep(order)

    harness.record("sort/int64_presorted", "rows", rows, sort_presorted)

    def sort_sparse() raises {imm sparse}:
        keep(sparse)
        var order = argsort(sparse, descending=True, nulls_first=True)
        keep(order)

    harness.record("sort/int64_nulls_descending", "rows", rows, sort_sparse)

    def sort_two_keys() raises {imm small, imm narrow}:
        keep(small)
        keep(narrow)
        var cols = List[AnyArray]()
        cols.append(AnyArray(Array[DType.int64](copy=small)))
        cols.append(AnyArray(Array[DType.uint32](copy=narrow)))
        var order = argsort_multi(cols, [False, False], [False, False])
        keep(order)

    harness.record("sort/multi_two_keys", "rows", rows, sort_two_keys)

    var reference = List[Int64]()
    for i in range(rows):
        reference.append(wide[i])

    def sort_reference() raises {imm reference, imm rows}:
        var copy = List[Int64]()
        for i in range(rows):
            copy.append(reference[i])
        sort(copy)
        keep(copy[0])

    harness.record("sort/std_sort_int64", "rows", rows, sort_reference)


def bench_frame(mut harness: Harness) raises:
    """Measures what the frame layer adds on top of the kernels it calls.

    Every row here has a kernel row it should be compared against, because the
    frame is supposed to be a thin thing and the way to check that is to see
    whether it costs what the kernel underneath it costs. `frame/filter` runs
    the same mask over three columns that `kernel/filter_rows` runs over one, so
    it should land near three times it plus the erased dispatch, which is a
    comparison chain over a value in a register and should not be visible at all.

    The pair worth reading first is `frame/column_by_name` against
    `frame/column_by_position`. They fetch the same column. The first returns a
    `Series`, which copies, and the second returns a borrowed reference, which
    does not, and the ratio between them is the price of the eager no-views rule
    stated in `firepanda/frame/frame.mojo`. It is a large ratio and it is meant
    to be visible, because when the plan layer arrives at M4 and column
    references stop being resolved by string lookup, this is the row that should
    move.

    `frame/sort_one_key` against `sort/int64_small` is the other pair. The
    difference between them is the three gathers that apply the permutation,
    which is what a frame sort costs over a column sort and is why `argsort` and
    `take` are separate functions in the first place.

    Args:
        harness: The harness.

    Raises:
        If a benchmark raises.
    """
    var rows = harness.options.rows
    var rng = Rng(0xF2A3E)

    var key = Array[DType.int64](rows)
    var score = Array[DType.float64](rows)
    var flag = Array[DType.bool](rows)
    var mask = Array[DType.bool](rows)
    var scatter = List[Int](capacity=rows)
    for i in range(rows):
        var draw = rng.next_u64()
        key[i] = Int64(draw % 1000)
        score[i] = Float64(Int64(draw % 1000000)) / Float64(64.0)
        flag[i] = draw & 1 == 1
        mask[i] = draw & 2 == 2
        scatter.append(Int(draw % UInt64(rows)))

    var columns = List[Series]()
    columns.append(Series("key", key^))
    columns.append(Series("score", score^))
    columns.append(Series("flag", flag^))
    var df = DataFrame.from_series(columns^)

    def frame_filter() raises {imm df, imm mask}:
        keep(df.rows)
        var out = df.filter(mask)
        keep(out.rows)

    harness.record("frame/filter", "rows", rows, frame_filter)

    def frame_take() raises {imm df, imm scatter}:
        keep(df.rows)
        var out = df.take(scatter)
        keep(out.rows)

    harness.record("frame/take", "rows", rows, frame_take)

    def frame_slice() raises {imm df, imm rows}:
        keep(df.rows)
        var out = df.slice(1, rows // 2)
        keep(out.rows)

    harness.record("frame/slice_half", "rows", rows, frame_slice)

    def frame_select() raises {imm df}:
        keep(df.rows)
        var out = df.select(["score", "key"])
        keep(out.rows)

    harness.record("frame/select_two", "rows", rows, frame_select)

    def frame_cast() raises {imm df}:
        keep(df.rows)
        var out = df.cast("key", DType.float64)
        keep(out.rows)

    harness.record("frame/cast_one", "rows", rows, frame_cast)

    def frame_sort_one() raises {imm df}:
        keep(df.rows)
        var out = df.sort_by("key")
        keep(out.rows)

    harness.record("frame/sort_one_key", "rows", rows, frame_sort_one)

    def frame_sort_two() raises {imm df}:
        keep(df.rows)
        var out = df.sort_values(
            ["key", "score"], [False, True], [False, False]
        )
        keep(out.rows)

    harness.record("frame/sort_two_keys", "rows", rows, frame_sort_two)

    def frame_by_name() raises {imm df}:
        keep(df.rows)
        var got = df.column("score")
        keep(len(got))

    harness.record("frame/column_by_name", "rows", rows, frame_by_name)

    def frame_by_position() raises {imm df}:
        keep(df.rows)
        keep(len(df[1]))

    harness.record("frame/column_by_position", "rows", rows, frame_by_position)

    # These two are per render rather than per row on purpose. Rendering builds
    # only the cells it prints, so the cost should not move when the frame gets
    # taller, and a per-row figure would hide that by dividing a constant by the
    # row count. Read them against each other instead: a frame at three columns
    # against a single column, both over a million rows.
    def frame_render() raises {imm df}:
        keep(df.rows)
        keep(String(df).byte_length())

    harness.record("frame/render", "renders", 1, frame_render)

    def series_render() raises {imm df}:
        keep(df.rows)
        keep(render_column("score", df[1], DisplayOptions()).byte_length())

    harness.record("frame/render_column", "renders", 1, series_render)


def bench_hash(mut harness: Harness) raises:
    """Measures the hash table, and measures whether it was worth writing.

    The rows to read together are the three `factorize_*` numbers and the three
    `dict_*` numbers beside them. They run over the same columns and differ only
    in which map is underneath. M1's exit criteria ask for that comparison rather
    than for a target, because if our table is not meaningfully faster than the
    language's `Dict` then the premise this package rests on is wrong and the
    cheap time to find that out is now.

    Cardinality is varied across three orders of magnitude because that is the
    axis the answer moves along. At a hundred groups the table fits in L1 and
    almost any implementation looks fine; at one group per row every probe is a
    cache miss and the layout is the whole story.

    `factorize_direct` is not part of that comparison. It is the route that does
    not hash at all, and it is here to show what the branch in `factorize` is
    buying on the shape of column that a real group by usually gets.

    Args:
        harness: The harness.

    Raises:
        If a benchmark raises.
    """
    var rows = harness.options.rows

    # Small dense integers, which is what a year, a category code or a previous
    # factorize looks like. Takes the direct route.
    var narrow = Array[DType.int64](rows)
    for i in range(rows):
        narrow[i] = Int64(i % 1000)

    def factorize_direct() raises {imm narrow}:
        keep(narrow)
        var out = factorize(narrow)
        keep(out.codes)

    harness.record("hash/factorize_direct", "rows", rows, factorize_direct)

    # The same shape of data spread far enough apart that the direct table would
    # be larger than the column, so these go through the hash table. The stride
    # is what makes the comparison against `Dict` a comparison of two hash maps
    # rather than of a hash map against an array index.
    comptime stride = Int64(DIRECT_LIMIT + 1)

    var low = Array[DType.int64](rows)
    var mid = Array[DType.int64](rows)
    var high = Array[DType.int64](rows)
    for i in range(rows):
        low[i] = Int64(i % 100) * stride
        mid[i] = Int64(i % 10000) * stride
        high[i] = Int64(i) * stride

    def factorize_low() raises {imm low}:
        keep(low)
        var out = factorize(low)
        keep(out.codes)

    harness.record("hash/factorize_100", "rows", rows, factorize_low)

    def factorize_mid() raises {imm mid}:
        keep(mid)
        var out = factorize(mid)
        keep(out.codes)

    harness.record("hash/factorize_10k", "rows", rows, factorize_mid)

    def factorize_high() raises {imm high}:
        keep(high)
        var out = factorize(high)
        keep(out.codes)

    harness.record("hash/factorize_all_distinct", "rows", rows, factorize_high)

    def dict_low() raises {imm low}:
        keep(low)
        var out = factorize_dict(low)
        keep(out)

    harness.record("hash/dict_100", "rows", rows, dict_low)

    def dict_mid() raises {imm mid}:
        keep(mid)
        var out = factorize_dict(mid)
        keep(out)

    harness.record("hash/dict_10k", "rows", rows, dict_mid)

    def dict_high() raises {imm high}:
        keep(high)
        var out = factorize_dict(high)
        keep(out)

    harness.record("hash/dict_all_distinct", "rows", rows, dict_high)

    # A fifth of the rows null, which is the shape that decides whether the null
    # group cost a branch per row or a branch per column.
    var sparse = Array[DType.int64](copy=mid)
    for i in range(0, rows, 5):
        sparse.set_null(i)

    def factorize_nulls() raises {imm sparse}:
        keep(sparse)
        var out = factorize(sparse)
        keep(out.codes)

    harness.record("hash/factorize_nulls", "rows", rows, factorize_nulls)

    # The pieces on their own, so that a regression in the whole can be attributed
    # to one of them rather than guessed at.
    def hash_column() raises {imm high, imm rows}:
        keep(high)
        var hashes = Buffer(rows * 8)
        hash_into(high, DEFAULT_SEED, hashes)
        keep(hashes)

    harness.record("hash/hash_into", "rows", rows, hash_column)

    var keys = Buffer(rows * 8)
    var key_ptr = keys.bitcast[DType.uint64]()
    for i in range(rows):
        key_ptr.unsafe_offset(i).unsafe_write(mix(UInt64(i), DEFAULT_SEED))

    # Deliberately the row-at-a-time API and deliberately not presized, so that
    # this row and `hash/factorize_all_distinct` above it bracket what the batch
    # build and the sizing hint are together worth. Nothing in the library builds
    # a table this way.
    def table_build() raises {imm keys, imm rows}:
        var bits = keys.bitcast[DType.uint64]()
        var table = HashTable()
        for i in range(rows):
            var k = bits.unsafe_offset(i).unsafe_load()
            _ = table.insert(mix(k, table.seed()))
        keep(len(table))

    harness.record("hash/table_insert_loop", "keys", rows, table_build)

    # Built once outside the timer so this row is probes and nothing else. Every
    # probe hits, which is the group by case; a join has misses too and will get
    # its own row at M8.
    var built = HashTable(rows)
    for i in range(rows):
        var k = key_ptr.unsafe_offset(i).unsafe_load()
        _ = built.insert(mix(k, built.seed()))

    def table_probe() raises {imm keys, imm built, imm rows}:
        var bits = keys.bitcast[DType.uint64]()
        var found = 0
        for i in range(rows):
            var k = bits.unsafe_offset(i).unsafe_load()
            found += built.find(mix(k, built.seed()))
        keep(found)

    harness.record("hash/table_probe", "keys", rows, table_probe)

    var hashes = Buffer(rows * 8)
    var hash_ptr = hashes.bitcast[DType.uint64]()
    for i in range(rows):
        hash_ptr.unsafe_offset(i).unsafe_write(mix(UInt64(i), DEFAULT_SEED))

    def partition() raises {imm hashes, imm rows}:
        var parts = radix_partition(hashes, rows, 8)
        keep(parts.order)

    harness.record("hash/radix_partition_256", "rows", rows, partition)


def bench_nulls(mut harness: Harness) raises:
    """Times the null-handling kernels and `concat`.

    Null density is the variable that matters in this section and every row here
    comes in two densities for that reason. A column with no nulls skips the
    repair pass entirely, because the output of a fresh `Array` is already all
    present and there is nothing to fix. A column that is one in eight null pays
    per row. The gap between the two is the whole design of these kernels and it
    should be visible in the numbers rather than only in the comments.

    `nulls/is_null` is the one operation here that never reads a value. Its cost
    is the expansion from a bit per row to a byte per row, so it should run at
    close to the speed of a memory write and should not care about density at
    all, since an all-present word and an all-null word are both a block store.

    `nulls/ffill` is the only scan. Row `i` depends on row `i - 1`, so there is
    nothing to vectorize and the only saving available is skipping the words with
    no nulls in them. On a dense column that saving is everything and on a sparse
    one it is nothing, which makes the pair of rows a direct reading of how much
    the word at a time shortcut is worth.

    `concat/*` is pure memory traffic. Two parts and eight parts of the same
    total height do the same amount of copying, so the difference between them is
    per-part overhead, and the frame row adds a schema walk and three columns.

    Args:
        harness: The harness.

    Raises:
        If a benchmark raises.
    """
    var rows = harness.options.rows
    var rng = Rng(0x4E0115)

    var dense = Array[BENCH_DTYPE](rows)
    var sparse = Array[BENCH_DTYPE](rows)
    var fallback = Array[BENCH_DTYPE](rows)
    for i in range(rows):
        var draw = rng.next_u64()
        dense[i] = Scalar[BENCH_DTYPE](Int(draw % 1000))
        fallback[i] = Scalar[BENCH_DTYPE](Int(draw % 7))
        if draw % 8 == 0:
            sparse.set_null(i)
        else:
            sparse.set_valid(i, Scalar[BENCH_DTYPE](Int(draw % 1000)))

    def null_mask_dense() raises {imm dense, imm rows}:
        keep(rows)
        var mask = is_null(dense)
        keep(len(mask))

    harness.record("nulls/is_null", "rows", rows, null_mask_dense)

    def null_mask_sparse() raises {imm sparse, imm rows}:
        keep(rows)
        var mask = is_null(sparse)
        keep(len(mask))

    harness.record("nulls/is_null_sparse", "rows", rows, null_mask_sparse)

    def coalesce_dense() raises {imm dense, imm fallback, imm rows}:
        keep(rows)
        var out = coalesce(dense, fallback)
        keep(len(out))

    harness.record("nulls/coalesce", "rows", rows, coalesce_dense)

    def coalesce_sparse() raises {imm sparse, imm fallback, imm rows}:
        keep(rows)
        var out = coalesce(sparse, fallback)
        keep(len(out))

    harness.record("nulls/coalesce_sparse", "rows", rows, coalesce_sparse)

    def ffill_dense() raises {imm dense, imm rows}:
        keep(rows)
        var out = fill_forward(dense)
        keep(len(out))

    harness.record("nulls/ffill", "rows", rows, ffill_dense)

    def ffill_sparse() raises {imm sparse, imm rows}:
        keep(rows)
        var out = fill_forward(sparse)
        keep(len(out))

    harness.record("nulls/ffill_sparse", "rows", rows, ffill_sparse)

    # Two lists of parts covering the same total height, so the difference
    # between the two rows is per-part overhead and nothing else.
    var halves = List[Array[BENCH_DTYPE]]()
    halves.append(sparse.slice(0, rows // 2))
    halves.append(sparse.slice(rows // 2, rows))

    var eighths = List[Array[BENCH_DTYPE]]()
    for p in range(8):
        eighths.append(sparse.slice(p * rows // 8, (p + 1) * rows // 8))

    def concat_two() raises {imm halves, imm rows}:
        keep(rows)
        var out = concat_arrays(halves)
        keep(len(out))

    harness.record("concat/two_parts", "rows", rows, concat_two)

    def concat_eight() raises {imm eighths, imm rows}:
        keep(rows)
        var out = concat_arrays(eighths)
        keep(len(out))

    harness.record("concat/eight_parts", "rows", rows, concat_eight)

    var half_series = List[Series]()
    half_series.append(Series("a", sparse.slice(0, rows // 2)))
    half_series.append(Series("b", dense.slice(0, rows // 2)))
    half_series.append(Series("c", fallback.slice(0, rows // 2)))
    var top = DataFrame.from_series(half_series^)

    var rest_series = List[Series]()
    rest_series.append(Series("a", sparse.slice(rows // 2, rows)))
    rest_series.append(Series("b", dense.slice(rows // 2, rows)))
    rest_series.append(Series("c", fallback.slice(rows // 2, rows)))
    var bottom = DataFrame.from_series(rest_series^)

    var pair = List[DataFrame]()
    pair.append(top^)
    pair.append(bottom^)

    def concat_frames() raises {imm pair, imm rows}:
        keep(rows)
        var out = concat(pair)
        keep(out.rows)

    harness.record("concat/frame_three_columns", "rows", rows, concat_frames)

    var frame_series = List[Series]()
    frame_series.append(Series("a", Array[BENCH_DTYPE](copy=sparse)))
    frame_series.append(Series("b", Array[BENCH_DTYPE](copy=dense)))
    var frame = DataFrame.from_series(frame_series^)

    def drop_nulls() raises {imm frame}:
        keep(frame.rows)
        var out = frame.drop_nulls()
        keep(out.rows)

    harness.record("nulls/drop_nulls", "rows", rows, drop_nulls)


def bench_csv(mut harness: Harness) raises:
    """Times the CSV field scanner and the field parsers.

    The scanner rows are all the same total number of bytes and differ only in
    shape, because the thing being measured is how often the block loop finds a
    boundary rather than how much text there is. A narrow file has a delimiter
    every few bytes and a wide one has a row ending every few, so the two should
    land close together and both should be well short of the memory bandwidth a
    scan with no boundaries at all would reach.

    The quoted row is the one that should hurt. A quoted field cannot use the
    block loop at all, because the closing quote has to be found by looking at
    every byte in case the next one doubles it, so a file where everything is
    quoted is the worst case by construction.

    The two twin rows are here for the ratio and not for their own sake. Each is
    the same algorithm with the register skipping removed, and the pair of gaps
    is the whole argument for the block loop: against `csv/scan_narrow` it
    should be a wash, because a five byte field is found before the vector path
    turns on at all, and against `csv/scan_long_text` it should be several times
    slower, because that is where there is something to skip.

    The parser rows are per field rather than per row. An integer field is a
    handful of multiply-adds and a float field is that plus one scaling multiply,
    so the two should be within a small factor, and both should be far cheaper
    than the cache miss that fetching the field cost.

    Args:
        harness: The harness.

    Raises:
        If a benchmark raises.
    """
    # A quarter of the usual height, because a row here is twenty five bytes of
    # text rather than eight bytes of int64 and the setup would otherwise build
    # a hundred megabytes of string before measuring anything.
    var rows = harness.options.rows // 4
    var rng = Rng(0x0C5D)

    # Four narrow columns, which is the shape of almost every file anybody
    # actually reads: an identifier, a couple of measurements and a label.
    var narrow_text = String()
    for _ in range(rows):
        narrow_text += String(rng.next_below(1000000))
        narrow_text += ","
        narrow_text += String(rng.next_below(100))
        narrow_text += ".25,"
        narrow_text += String(rng.next_below(10000))
        narrow_text += ",ab\n"
    var narrow = bytes_of(narrow_text)

    # The same bytes cut into shorter rows, so the scan meets a row ending far
    # more often for the same amount of text.
    var wide_rows = rows // 4
    var wide_text = String()
    for _ in range(wide_rows):
        for c in range(16):
            if c > 0:
                wide_text += ","
            wide_text += String(rng.next_below(1000))
        wide_text += "\n"
    var wide = bytes_of(wide_text)

    var quoted_text = String()
    for _ in range(rows):
        quoted_text += '"abcdef","ghijkl","mnopqr","stuvwx"\n'
    var quoted = bytes_of(quoted_text)

    # Two columns of sentence length text, which is where a field is longer than
    # a register and the block loop has something to skip.
    var long_rows = rows // 4
    var long_text = String()
    for _ in range(long_rows):
        for c in range(2):
            if c > 0:
                long_text += ","
            for _ in range(8):
                long_text += "the quick brown fox "[
                    byte = 0 : Int(4 + rng.next_below(16))
                ]
        long_text += "\n"
    var long_fields = bytes_of(long_text)

    def scan_narrow() raises {imm narrow, imm rows}:
        keep(rows)
        var scan = scan_csv(narrow, default_dialect())
        keep(len(scan))

    harness.record("csv/scan_narrow", "rows", rows, scan_narrow)

    def scan_wide() raises {imm wide, imm wide_rows}:
        keep(wide_rows)
        var scan = scan_csv(wide, default_dialect())
        keep(len(scan))

    harness.record("csv/scan_wide", "rows", wide_rows, scan_wide)

    def scan_quoted() raises {imm quoted, imm rows}:
        keep(rows)
        var scan = scan_csv(quoted, default_dialect())
        keep(len(scan))

    harness.record("csv/scan_quoted", "rows", rows, scan_quoted)

    def scan_long() raises {imm long_fields, imm long_rows}:
        keep(long_rows)
        var scan = scan_csv(long_fields, default_dialect())
        keep(len(scan))

    harness.record("csv/scan_long_text", "rows", long_rows, scan_long)

    def scan_twin() raises {imm narrow, imm rows}:
        keep(rows)
        var scan = scan_csv_scalar(narrow, default_dialect())
        keep(len(scan))

    harness.record("csv/scan_scalar_twin", "rows", rows, scan_twin)

    def scan_long_twin() raises {imm long_fields, imm long_rows}:
        keep(long_rows)
        var scan = scan_csv_scalar(long_fields, default_dialect())
        keep(len(scan))

    harness.record("csv/scan_long_twin", "rows", long_rows, scan_long_twin)

    # The parsers are timed over the fields the scan just found, so the field
    # boundaries are realistic rather than a loop over one string.
    var found = scan_csv(narrow, default_dialect())
    var integers = List[Int](capacity=rows * 2)
    var floats = List[Int](capacity=rows)
    for r in range(len(found)):
        integers.append(found.starts[r] + 0)
        integers.append(found.starts[r] + 2)
        floats.append(found.starts[r] + 1)

    def parse_integers() raises {imm narrow, imm found, imm integers}:
        var total = Int64(0)
        for i in range(len(integers)):
            var span = found.fields[integers[i]]
            total += parse_int[DType.int64](field_bytes(narrow, span)).value
        keep(Int(total))

    harness.record("csv/parse_int", "fields", len(integers), parse_integers)

    def parse_floats() raises {imm narrow, imm found, imm floats}:
        var total = Float64(0.0)
        for i in range(len(floats)):
            var span = found.fields[floats[i]]
            total += parse_float[DType.float64](field_bytes(narrow, span)).value
        keep(Int(total))

    harness.record("csv/parse_float", "fields", len(floats), parse_floats)


def _string_column(
    count: Int, width: Int, vary_prefix: Bool
) raises -> StringArray:
    """Builds a column of fixed width elements for the string benchmarks.

    Every element differs from its neighbour in its last byte, so a comparison
    between two adjacent elements always has to run to the end unless something
    earlier settles it. What settles it earlier is the prefix, which is what
    `vary_prefix` turns on.

    Args:
        count: How many elements.
        width: How many bytes in each.
        vary_prefix: Whether the first four bytes differ between neighbours.

    Returns:
        The column.

    Raises:
        If the builder raises.
    """
    var pool = List[UInt8](capacity=count * width)
    for i in range(count):
        for j in range(width):
            if vary_prefix and j < PREFIX_LENGTH:
                pool.append(UInt8(97 + (i + j) % 26))
            elif j == width - 1:
                pool.append(UInt8(97 + i % 26))
            else:
                pool.append(UInt8(97 + j % 26))

    var builder = StringBuilder(capacity=count)
    var base = pool.unsafe_ptr()
    for i in range(count):
        builder.append(
            Span[UInt8, origin_of(pool)](
                unsafe_ptr=base.unsafe_offset(i * width), length=width
            )
        )
    return builder^.finish()


def bench_strings(mut harness: Harness) raises:
    """Times the variable width string column on both sides of the inline limit.

    Every row here is measured twice, once at eight bytes and once at thirty two,
    because twelve is where an element stops living inside its own view and moves
    to the payload and that is the only interesting thing about this layout. The
    short column should never touch the payload at all.

    `strings/length_short` and `strings/length_long` should land on top of each
    other. A length is in the view either way, so asking for one costs a single
    load whatever the element is, which is the thing the classic offsets layout
    cannot say.

    `strings/bytes_short` against `strings/bytes_long` does not open the gap that
    it looks like it should. Reading a short element stays inside the views array
    and reading a long one goes to a second buffer, but both walks are in order
    and a prefetcher has no trouble with two sequential streams, so the pair lands
    level. The inline layout is not worth anything to a scan. It is worth
    something to everything that jumps.

    The two equality rows are the argument for the four byte prefix, and they are
    where the layout actually pays. Both compare adjacent long elements that are
    not equal, and both answer false. The only difference is where the difference
    is: in the prefix, which the view settles on its own, or in the last byte,
    which costs a walk into a buffer the view alone would never have touched.

    Args:
        harness: The harness.

    Raises:
        If a benchmark raises.
    """
    # A quarter of the usual row count, because sixteen bytes of view per element
    # plus a payload is a lot of memory to hold three copies of.
    var rows = harness.options.rows // 4
    if rows < 1:
        rows = 1

    var short = _string_column(rows, 8, True)
    var long = _string_column(rows, 32, True)
    var late = _string_column(rows, 32, False)

    def build_short() raises {imm short, imm rows}:
        var builder = StringBuilder(capacity=rows)
        for i in range(rows):
            builder.append(short.unsafe_bytes(i))
        var out = builder^.finish()
        keep(len(out))

    harness.record("strings/build_short", "rows", rows, build_short)

    def build_long() raises {imm long, imm rows}:
        var builder = StringBuilder(capacity=rows)
        for i in range(rows):
            builder.append(long.unsafe_bytes(i))
        var out = builder^.finish()
        keep(len(out))

    harness.record("strings/build_long", "rows", rows, build_long)

    def length_short() raises {imm short, imm rows}:
        var total = 0
        for i in range(rows):
            total += short.byte_length(i)
        keep(total)

    harness.record("strings/length_short", "rows", rows, length_short)

    def length_long() raises {imm long, imm rows}:
        var total = 0
        for i in range(rows):
            total += long.byte_length(i)
        keep(total)

    harness.record("strings/length_long", "rows", rows, length_long)

    def bytes_short() raises {imm short, imm rows}:
        var total = 0
        for i in range(rows):
            var bytes = short.unsafe_bytes(i)
            total += Int(bytes[len(bytes) - 1])
        keep(total)

    harness.record("strings/bytes_short", "rows", rows, bytes_short)

    def bytes_long() raises {imm long, imm rows}:
        var total = 0
        for i in range(rows):
            var bytes = long.unsafe_bytes(i)
            total += Int(bytes[len(bytes) - 1])
        keep(total)

    harness.record("strings/bytes_long", "rows", rows, bytes_long)

    def equals_prefix() raises {imm long, imm rows}:
        var same = 0
        for i in range(rows - 1):
            if long.element_equals(i, i + 1):
                same += 1
        keep(same)

    harness.record("strings/equals_prefix", "pairs", rows - 1, equals_prefix)

    def equals_payload() raises {imm late, imm rows}:
        var same = 0
        for i in range(rows - 1):
            if late.element_equals(i, i + 1):
                same += 1
        keep(same)

    harness.record("strings/equals_payload", "pairs", rows - 1, equals_payload)

    # A stride that is coprime with any plausible row count, so the gather walks
    # the whole column without ever settling into a pattern a prefetcher can pick
    # up. This is the access a join or a sort permutation actually produces.
    var picks = List[Int](capacity=rows)
    var step = 0
    for _ in range(rows):
        step = (step + 7919) % rows
        picks.append(step)

    def take_scattered() raises {imm long, imm picks}:
        var out = long.take(picks)
        keep(len(out))

    harness.record("strings/take", "rows", rows, take_scattered)

    var mask = Bitmap(rows)
    mask.clear_all()
    for i in range(0, rows, 2):
        mask.set(i, True)

    def filter_half() raises {imm long, imm mask}:
        var out = long.filter(mask)
        keep(len(out))

    harness.record("strings/filter", "rows", rows, filter_half)


def bench_text(mut harness: Harness) raises:
    """Times a string column moving through the erased column the frame holds.

    `bench_strings` above measures `StringArray` on its own. This measures the
    same work reached the way a `DataFrame` reaches it, through `AnyArray`, and
    every row is paired with the int64 column of the same height going through
    the same entry point. The pair is the point. A row of text and a row of number
    are not the same amount of work and the table should say by how much rather
    than leave it to be guessed at.

    The gap is expected and is not a defect. A gather of int64 writes eight bytes
    at a computed offset; a gather of text writes a sixteen byte view and, for
    anything past twelve bytes, copies the element into a payload block that grows
    as it goes. What the pair is watching for is the gap changing.

    `text/is_string` is the guard on its own. `AnyArray` grew an
    `Optional[StringArray]` field and every erased kernel now asks `is_string()`
    before it dispatches, so the question of what that costs is worth a row rather
    than an assurance. It is a load and a predictable branch on a field that is
    already in the line the dtype was read from.

    The numeric rows here are not comparable with the `kernel/` rows above. These
    run at a quarter of the row count for the reason given below, and a quarter of
    a million int64 rows sit in a cache the full million does not, so the two
    tables disagree by a factor that is about the machine and not about the code.

    The sort rows and the group rows read the same three text columns and reward
    opposite shapes, which is the most useful thing in the table. A column of
    short repeated labels is the sort's best case and the group's best case for
    different reasons, and a column that shares a nine byte prefix is the sort's
    worst case and costs the group almost nothing.

    Args:
        harness: The harness.

    Raises:
        If a benchmark raises.
    """
    # A quarter of the usual rows, matching `bench_strings`, because the same
    # sixteen bytes of view per element apply and several of these hold two
    # columns and an output at once.
    var rows = harness.options.rows // 4
    if rows < 1:
        rows = 1

    var text = AnyArray(_string_column(rows, 32, True))
    var text_other = AnyArray(_string_column(rows, 32, True))

    var numbers = Array[BENCH_DTYPE](rows)
    for i in range(rows):
        numbers[i] = Scalar[BENCH_DTYPE](i)
    var number = AnyArray(numbers^)

    var more = Array[BENCH_DTYPE](rows)
    for i in range(rows):
        more[i] = Scalar[BENCH_DTYPE](i)
    var number_other = AnyArray(more^)

    # The same coprime stride `bench_strings` uses, so a gather here and a gather
    # there are walking the column the same way and the two tables can be read
    # against each other.
    var picks = List[Int](capacity=rows)
    var step = 0
    for _ in range(rows):
        step = (step + 7919) % rows
        picks.append(step)

    def take_text() raises {imm text, imm picks}:
        var out = take_any(text, picks)
        keep(len(out))

    harness.record("text/take_text", "rows", rows, take_text)

    def take_number() raises {imm number, imm picks}:
        var out = take_any(number, picks)
        keep(len(out))

    harness.record("text/take_number", "rows", rows, take_number)

    var mask = Array[DType.bool](rows)
    var bits = mask.unsafe_ptr()
    for i in range(rows):
        bits.unsafe_offset(i).unsafe_write(i % 2 == 0)

    def filter_text() raises {imm text, imm mask}:
        var out = filter_any(text, mask)
        keep(len(out))

    harness.record("text/filter_text", "rows", rows, filter_text)

    def filter_number() raises {imm number, imm mask}:
        var out = filter_any(number, mask)
        keep(len(out))

    harness.record("text/filter_number", "rows", rows, filter_number)

    # Concat is where the two paths differ most and for a reason worth seeing in
    # the table. The fixed width side memcpys a part into place. The text side
    # cannot, because a part's payload offsets are relative to that part's block,
    # so every element is appended one at a time and the payload is rebuilt.
    def concat_text() raises {imm text, imm text_other}:
        var out = concat_two_any(text, text_other)
        keep(len(out))

    harness.record("text/concat_text", "rows", rows * 2, concat_text)

    def concat_number() raises {imm number, imm number_other}:
        var out = concat_two_any(number, number_other)
        keep(len(out))

    harness.record("text/concat_number", "rows", rows * 2, concat_number)

    var half = rows // 2
    if half < 1:
        half = 1

    def slice_text() raises {imm text, imm half}:
        var out = text.slice(0, half)
        keep(len(out))

    harness.record("text/slice_text", "rows", half, slice_text)

    def slice_number() raises {imm number, imm half}:
        var out = number.slice(0, half)
        keep(len(out))

    harness.record("text/slice_number", "rows", half, slice_number)

    # The guard on its own, with as little else in the frame as possible. It is
    # a load and a branch on a field that is in cache, and the row exists so that
    # the claim it costs nothing is a measurement rather than an assertion.
    def guard() raises {imm text, imm number}:
        keep(text)
        keep(number)
        var found = Int(text.is_string()) + Int(number.is_string())
        keep(found)

    harness.record("text/is_string", "calls", 2, guard)

    # The three rows to read together are the sort rows, and what separates them
    # is how much of the answer the eight byte key can give. `sort_distinct` is
    # the column a dataframe usually sorts, where the key settles almost every
    # pair and the comparison sort is barely entered. `sort_prefixed` shares nine
    # bytes across every element, so the radix passes leave one run as tall as the
    # column and the whole answer comes out of the merge sort. `sort_repeated`
    # has a hundred distinct values, so the runs are large and every comparison
    # inside them returns equal, which is the case where a tie break does the most
    # work for the least benefit.
    var distinct = List[String](capacity=rows)
    var prefixed = List[String](capacity=rows)
    var repeated = List[String](capacity=rows)
    var sort_rng = Rng(0x7E5713)
    for _ in range(rows):
        var n = sort_rng.next_below(1 << 30)
        distinct.append(String("k", n))
        prefixed.append(String("sharedpre", n))
        repeated.append(String("v", sort_rng.next_below(100)))

    var distinct_col = AnyArray(strings_from_list(distinct))
    var prefixed_col = AnyArray(strings_from_list(prefixed))
    var repeated_col = AnyArray(strings_from_list(repeated))

    def sort_distinct() raises {imm distinct_col}:
        var order = argsort_any(distinct_col)
        keep(len(order))

    harness.record("text/sort_distinct", "rows", rows, sort_distinct)

    def sort_prefixed() raises {imm prefixed_col}:
        var order = argsort_any(prefixed_col)
        keep(len(order))

    harness.record("text/sort_prefixed", "rows", rows, sort_prefixed)

    def sort_repeated() raises {imm repeated_col}:
        var order = argsort_any(repeated_col)
        keep(len(order))

    harness.record("text/sort_repeated", "rows", rows, sort_repeated)

    # The numeric column of the same height through the same entry point, so the
    # text sort has something to be a multiple of.
    def sort_number() raises {imm number}:
        var order = argsort_any(number)
        keep(len(order))

    harness.record("text/sort_number", "rows", rows, sort_number)

    # Grouping reads the same three columns and rewards the opposite shapes to
    # the sort. `group_repeated` is the cheap one here because a hundred keys fit
    # in cache and every probe after the first hits. `group_prefixed` is not the
    # pathological case it is for the sort, because the hash reads every byte
    # rather than the first eight, so a shared prefix costs the bytes it adds and
    # nothing more.
    var distinct_keys = StringArray(copy=distinct_col.strings())
    var prefixed_keys = StringArray(copy=prefixed_col.strings())
    var repeated_keys = StringArray(copy=repeated_col.strings())

    def group_distinct() raises {imm distinct_keys}:
        var found = factorize_strings(distinct_keys)
        keep(found.codes)

    harness.record("text/group_distinct", "rows", rows, group_distinct)

    def group_prefixed() raises {imm prefixed_keys}:
        var found = factorize_strings(prefixed_keys)
        keep(found.codes)

    harness.record("text/group_prefixed", "rows", rows, group_prefixed)

    def group_repeated() raises {imm repeated_keys}:
        var found = factorize_strings(repeated_keys)
        keep(found.codes)

    harness.record("text/group_repeated", "rows", rows, group_repeated)

    # The numeric pair, spread far enough apart to be denied the direct route, so
    # both sides of the comparison are the same hash table doing the same job on
    # the same number of groups.
    comptime spread = Int64(DIRECT_LIMIT + 1)
    var all_distinct = Array[DType.int64](rows)
    var hundred = Array[DType.int64](rows)
    for i in range(rows):
        all_distinct[i] = Int64(i) * spread
        hundred[i] = Int64(i % 100) * spread

    def group_number_distinct() raises {imm all_distinct}:
        keep(all_distinct)
        var found = factorize(all_distinct)
        keep(found.codes)

    harness.record(
        "text/group_number_distinct", "rows", rows, group_number_distinct
    )

    def group_number_repeated() raises {imm hundred}:
        keep(hundred)
        var found = factorize(hundred)
        keep(found.codes)

    harness.record(
        "text/group_number_repeated", "rows", rows, group_number_repeated
    )


def bench_dispatch(mut harness: Harness) raises:
    """Measures what the runtime to compile-time bridge costs per call.

    docs/specs/03-dtype-dispatch.md pays for a dtype dispatch once per operation
    rather than once per row, and the only way that argument holds is if the fixed
    cost is small next to the work. So it is measured twice, once over a column of
    one row where the dispatch is nearly the whole cost, and once over the full
    column where it should have disappeared into the noise.

    Args:
        harness: The harness.

    Raises:
        If a benchmark raises.
    """
    var rows = harness.options.rows

    var tiny = Array[BENCH_DTYPE](1)
    tiny[0] = Scalar[BENCH_DTYPE](7)
    var tiny_erased = AnyArray(tiny^)

    def dispatch_tiny() raises {imm tiny_erased}:
        keep(tiny_erased)
        var total = dispatch[NUMERIC](tiny_erased, sum_erased)
        keep(total)

    harness.record("dispatch/call_1_row", "calls", 1, dispatch_tiny)

    var wide = Array[BENCH_DTYPE](rows)
    for i in range(rows):
        wide[i] = Scalar[BENCH_DTYPE](i % 1000)
    var wide_erased = AnyArray(wide^)

    def dispatch_wide() raises {imm wide_erased}:
        keep(wide_erased)
        var total = dispatch[NUMERIC](wide_erased, sum_erased)
        keep(total)

    harness.record("dispatch/sum_full", "rows", rows, dispatch_wide)

    def direct_sum() raises {imm wide_erased}:
        keep(wide_erased)
        var total = sum_erased[BENCH_DTYPE](wide_erased)
        keep(total)

    harness.record("dispatch/sum_full_direct", "rows", rows, direct_sum)


def bench_group(mut harness: Harness) raises:
    """Times grouped reductions from the scatter loop up to `DataFrame.group_by`.

    Three questions, and the row names are arranged so each one is a subtraction.

    How much does grouping cost over the same reduction ungrouped. `group/sum` is
    a scatter into an accumulator per group where `kernel/sum_dense` is a straight
    SIMD accumulate, and the gap between them is what a group by costs before any
    hashing happens at all.

    How much does cardinality cost. `group/sum_cardinality_10` and its 100k twin
    run the identical loop over the identical rows and differ only in how many
    accumulators the scatter is writing into. Ten fits in a cache line and a
    hundred thousand does not, so the pair measures the random write directly.

    How much of a real group by is the grouping rather than the reduction.
    `group/ordinals_one_key` is the factorize and renumber pass on its own, and
    `group/frame_one_key` is the whole operation, so the reduction is what is left
    over. The two key rows say what the second factorize and the repacking add.

    Args:
        harness: The harness.

    Raises:
        If a benchmark raises.
    """
    var rows = harness.options.rows
    var rng = Rng(0x6C0DE5)

    comptime GROUPS = 1000

    var values = Array[DType.int64](rows)
    var sparse = Array[DType.int64](rows)
    var codes = Array[DType.uint32](rows)
    var few = Array[DType.uint32](rows)
    var many = Array[DType.uint32](rows)
    var key = Array[DType.int64](rows)
    var other = Array[DType.int64](rows)
    var wide = 100_000 if rows >= 100_000 else rows
    if wide < 1:
        wide = 1
    for i in range(rows):
        var draw = rng.next_u64()
        values[i] = Int64(draw % 1000)
        sparse[i] = Int64(draw % 1000)
        codes[i] = UInt32(draw % UInt64(GROUPS))
        few[i] = UInt32(draw % 10)
        many[i] = UInt32(draw % UInt64(wide))
        key[i] = Int64(draw % UInt64(GROUPS))
        other[i] = Int64((draw >> 20) % 8)
        if draw & 7 == 0:
            sparse.set_null(i)

    def sum_dense() raises {imm values, imm codes}:
        keep(values)
        var out = group_sum(values, codes, GROUPS)
        keep(out[0])

    harness.record("group/sum", "rows", rows, sum_dense)

    def sum_sparse() raises {imm sparse, imm codes}:
        keep(sparse)
        var out = group_sum(sparse, codes, GROUPS)
        keep(out[0])

    harness.record("group/sum_sparse", "rows", rows, sum_sparse)

    def sum_few() raises {imm values, imm few}:
        keep(values)
        var out = group_sum(values, few, 10)
        keep(out[0])

    harness.record("group/sum_cardinality_10", "rows", rows, sum_few)

    def sum_many() raises {imm values, imm many, imm wide}:
        keep(values)
        var out = group_sum(values, many, wide)
        keep(out[0])

    harness.record("group/sum_cardinality_100k", "rows", rows, sum_many)

    def mean_grouped() raises {imm sparse, imm codes}:
        keep(sparse)
        var out = group_mean(sparse, codes, GROUPS)
        keep(out[0])

    harness.record("group/mean_sparse", "rows", rows, mean_grouped)

    def min_dense() raises {imm values, imm codes}:
        keep(values)
        var out = group_min(values, codes, GROUPS)
        keep(out[0])

    harness.record("group/min", "rows", rows, min_dense)

    def min_grouped() raises {imm sparse, imm codes}:
        keep(sparse)
        var out = group_min(sparse, codes, GROUPS)
        keep(out[0])

    harness.record("group/min_sparse", "rows", rows, min_grouped)

    def first_grouped() raises {imm sparse, imm codes}:
        keep(sparse)
        var out = group_first(sparse, codes, GROUPS)
        keep(out[0])

    harness.record("group/first_sparse", "rows", rows, first_grouped)

    def size_grouped() raises {imm codes}:
        keep(codes)
        var out = group_size(codes, GROUPS)
        keep(out[0])

    harness.record("group/size", "rows", rows, size_grouped)

    # The five that landed after the first eight. `var` is two passes over the
    # column where `mean` is one, and the three below it build a slab of the
    # values and sort it per group, so the gap between these and `group/sum` is
    # the price of a reduction that cannot be done in one accumulator.
    def var_grouped() raises {imm sparse, imm codes}:
        keep(sparse)
        var out = group_var(sparse, codes, GROUPS)
        keep(out[0])

    harness.record("group/var_sparse", "rows", rows, var_grouped)

    def std_grouped() raises {imm sparse, imm codes}:
        keep(sparse)
        var out = group_std(sparse, codes, GROUPS)
        keep(out[0])

    harness.record("group/std_sparse", "rows", rows, std_grouped)

    def median_grouped() raises {imm values, imm codes}:
        keep(values)
        var out = group_median(values, codes, GROUPS)
        keep(out[0])

    harness.record("group/median", "rows", rows, median_grouped)

    def median_sparse_grouped() raises {imm sparse, imm codes}:
        keep(sparse)
        var out = group_median(sparse, codes, GROUPS)
        keep(out[0])

    harness.record("group/median_sparse", "rows", rows, median_sparse_grouped)

    # A thousand groups over a million rows is a thousand values per slab. The
    # cardinality row below it is ten groups, so the slabs are a hundred times
    # longer and the sort inside each one costs more per row, which is the whole
    # difference between the two.
    def median_low_cardinality() raises {imm values, imm few}:
        keep(values)
        var out = group_median(values, few, 10)
        keep(out[0])

    harness.record(
        "group/median_cardinality_10", "rows", rows, median_low_cardinality
    )

    def nunique_grouped() raises {imm values, imm codes}:
        keep(values)
        var out = group_nunique(values, codes, GROUPS)
        keep(out[0])

    harness.record("group/nunique", "rows", rows, nunique_grouped)

    var erased = AnyArray(Array(copy=values))

    def sum_erased_group() raises {imm erased, imm codes}:
        keep(erased)
        var out = aggregate_group_any(erased, AggKind.SUM, codes, GROUPS)
        keep(len(out))

    harness.record("group/sum_dispatched", "rows", rows, sum_erased_group)

    def quantile_erased() raises {imm erased, imm codes}:
        keep(erased)
        var out = aggregate_group_any(
            erased, AggKind.quantile_at(0.9), codes, GROUPS
        )
        keep(len(out))

    harness.record("group/quantile_dispatched", "rows", rows, quantile_erased)

    var columns = List[Series]()
    columns.append(Series("key", key^))
    columns.append(Series("other", other^))
    columns.append(Series("value", values^))
    var df = DataFrame.from_series(columns^)

    var one_key = List[Int]()
    one_key.append(0)
    var two_keys = List[Int]()
    two_keys.append(0)
    two_keys.append(1)

    def ordinals_one() raises {imm df, imm one_key}:
        keep(df.rows)
        var out = group_ordinals(df.columns, one_key, df.rows)
        keep(out.groups)

    harness.record("group/ordinals_one_key", "rows", rows, ordinals_one)

    def ordinals_two() raises {imm df, imm two_keys}:
        keep(df.rows)
        var out = group_ordinals(df.columns, two_keys, df.rows)
        keep(out.groups)

    harness.record("group/ordinals_two_keys", "rows", rows, ordinals_two)

    def frame_one() raises {imm df}:
        keep(df.rows)
        var out = df.group_by(["key"], [AggSpec("value", AggKind.SUM)])
        keep(out.rows)

    harness.record("group/frame_one_key", "rows", rows, frame_one)

    def frame_two() raises {imm df}:
        keep(df.rows)
        var out = df.group_by(["key", "other"], [AggSpec("value", AggKind.SUM)])
        keep(out.rows)

    harness.record("group/frame_two_keys", "rows", rows, frame_two)

    def frame_unsorted() raises {imm df}:
        keep(df.rows)
        var out = df.group_by(
            ["key"], [AggSpec("value", AggKind.SUM)], dropna=False, sort=False
        )
        keep(out.rows)

    harness.record("group/frame_unsorted", "rows", rows, frame_unsorted)

    def frame_three_aggs() raises {imm df}:
        keep(df.rows)
        var out = df.group_by(
            ["key"],
            [
                AggSpec("value", AggKind.SUM),
                AggSpec("value", AggKind.MIN),
                AggSpec("value", AggKind.MAX),
            ],
        )
        keep(out.rows)

    harness.record("group/frame_three_aggs", "rows", rows, frame_three_aggs)


def bench_join(mut harness: Harness) raises:
    """Times joins, from the row pairing up to `DataFrame.join`.

    The shape every row here uses except the last is the one a join actually has
    in a query: a large fact table against a small dimension table, each fact row
    matching exactly one dimension row. That is the case worth being fast at and
    it is also the case where a bad implementation looks fine, because the output
    is the same height as the input and nothing is being duplicated.

    Four questions.

    What does the pairing cost on its own. `join/indices_1000` is
    `join_indices` with no columns gathered afterwards, and `join/inner_1000` is
    the whole operation, so the difference is what building the output costs. On a
    two column frame that difference is two gathers and it should dominate.

    What does dimension size cost. `join/inner_1000` and `join/inner_100k` join
    the same fact rows, using the same thousand keys, against dimensions a
    hundred times apart in size. The result is identical and only the dimension's
    unused rows differ, so the gap is exactly what carrying them costs: the
    counting, prefix and scatter passes are over the whole dimension and the
    bucket array is as wide as it is.

    What do the other kinds cost relative to inner. Semi and anti stop at the
    first match and gather nothing from the right, so they should be cheaper than
    inner on the same inputs. Outer has to track which right rows were hit, which
    is a bitmap write per matched row.

    What does multiplicity cost. `join/many_to_many` is two small frames with
    sixty four keys each, so every key produces a block of output rows and the
    result is far taller than either input. That is the case where a join stops
    being a gather and starts being a product, and it is deliberately small
    because the cost is the output size.

    Args:
        harness: The harness.

    Raises:
        If a benchmark raises.
    """
    var rows = harness.options.rows
    var rng = Rng(0x105E5)

    var dim_rows = 1000 if rows >= 1000 else rows
    if dim_rows < 1:
        dim_rows = 1
    var wide_rows = 100_000 if rows >= 100_000 else dim_rows

    var fact_key = Array[DType.int64](rows)
    var fact_other = Array[DType.int64](rows)
    var fact_value = Array[DType.int64](rows)
    for i in range(rows):
        var draw = rng.next_u64()
        fact_key[i] = Int64(draw % UInt64(dim_rows))
        fact_other[i] = Int64((draw >> 20) % 8)
        fact_value[i] = Int64(draw % 1000)

    var fact_series = List[Series]()
    fact_series.append(Series("key", fact_key^))
    fact_series.append(Series("other", fact_other^))
    fact_series.append(Series("value", fact_value^))
    var fact = DataFrame.from_series(fact_series^)

    var dim = _dimension(dim_rows, 0, "label")
    var wide_dim = _dimension(wide_rows, 0, "label")
    # Half the keys, so that the kinds which report a non-match have something to
    # report. Joining a dimension that covers every key would make anti empty and
    # outer identical to inner, and neither would be measuring anything.
    var partial = _dimension(dim_rows // 2 if dim_rows > 1 else 1, 0, "label")

    var one = List[String]()
    one.append("key")
    var two = List[String]()
    two.append("key")
    two.append("other")

    var fact_keys = List[Int]()
    fact_keys.append(0)
    var dim_keys = List[Int]()
    dim_keys.append(0)

    def indices_only() raises {imm fact, imm dim, imm fact_keys, imm dim_keys}:
        keep(fact.rows)
        var pairs = join_indices(
            fact.columns,
            fact_keys,
            fact.rows,
            dim.columns,
            dim_keys,
            dim.rows,
            JoinKind.INNER,
        )
        keep(len(pairs))

    harness.record("join/indices_1000", "rows", rows, indices_only)

    def inner_small() raises {imm fact, imm dim, imm one}:
        keep(fact.rows)
        var out = fact.join(dim, one)
        keep(out.rows)

    harness.record("join/inner_1000", "rows", rows, inner_small)

    def left_small() raises {imm fact, imm dim, imm one}:
        keep(fact.rows)
        var out = fact.join(dim, one, JoinKind.LEFT)
        keep(out.rows)

    harness.record("join/left_1000", "rows", rows, left_small)

    def inner_wide() raises {imm fact, imm wide_dim, imm one}:
        keep(fact.rows)
        var out = fact.join(wide_dim, one)
        keep(out.rows)

    harness.record("join/inner_100k", "rows", rows, inner_wide)

    def semi_partial() raises {imm fact, imm partial, imm one}:
        keep(fact.rows)
        var out = fact.join(partial, one, JoinKind.SEMI)
        keep(out.rows)

    harness.record("join/semi", "rows", rows, semi_partial)

    def anti_partial() raises {imm fact, imm partial, imm one}:
        keep(fact.rows)
        var out = fact.join(partial, one, JoinKind.ANTI)
        keep(out.rows)

    harness.record("join/anti", "rows", rows, anti_partial)

    def outer_partial() raises {imm fact, imm partial, imm one}:
        keep(fact.rows)
        var out = fact.join(partial, one, JoinKind.OUTER)
        keep(out.rows)

    harness.record("join/outer", "rows", rows, outer_partial)

    # Every distinct pair of the two fact key columns, so the two key join finds
    # a match for every row and the extra cost is the packing rather than a
    # different result.
    var pair_rows = dim_rows * 8
    var pair_key = Array[DType.int64](pair_rows)
    var pair_other = Array[DType.int64](pair_rows)
    var pair_label = Array[DType.int64](pair_rows)
    for i in range(pair_rows):
        pair_key[i] = Int64(i // 8)
        pair_other[i] = Int64(i % 8)
        pair_label[i] = Int64(i)
    var pair_series = List[Series]()
    pair_series.append(Series("key", pair_key^))
    pair_series.append(Series("other", pair_other^))
    pair_series.append(Series("label", pair_label^))
    var pair_dim = DataFrame.from_series(pair_series^)

    def two_key() raises {imm fact, imm pair_dim, imm two}:
        keep(fact.rows)
        var out = fact.join(pair_dim, two)
        keep(out.rows)

    harness.record("join/two_keys", "rows", rows, two_key)

    # Deliberately small. Sixty four keys over four thousand rows a side is
    # sixty four rows per key per side, which is four thousand output rows per
    # key and a quarter of a million altogether.
    var block = 4096 if rows >= 4096 else rows
    if block < 1:
        block = 1
    var left_key = Array[DType.int64](block)
    var left_value = Array[DType.int64](block)
    var right_key = Array[DType.int64](block)
    var right_value = Array[DType.int64](block)
    for i in range(block):
        left_key[i] = Int64(i % 64)
        left_value[i] = Int64(i)
        right_key[i] = Int64(i % 64)
        right_value[i] = Int64(i)
    var left_series = List[Series]()
    left_series.append(Series("key", left_key^))
    left_series.append(Series("a", left_value^))
    var right_series = List[Series]()
    right_series.append(Series("key", right_key^))
    right_series.append(Series("b", right_value^))
    var dense_left = DataFrame.from_series(left_series^)
    var dense_right = DataFrame.from_series(right_series^)

    # The item count is the output height rather than the input height, because
    # that is what the work is proportional to and a per-row number against four
    # thousand inputs would be off by the fan-out.
    var product = block * (block // 64 if block >= 64 else 1)

    def many_to_many() raises {
        imm dense_left, imm dense_right, imm one, imm block
    }:
        keep(block)
        var out = dense_left.join(dense_right, one)
        keep(out.rows)

    harness.record("join/many_to_many", "rows", product, many_to_many)


def _dimension(rows: Int, base: Int, label: String) raises -> DataFrame:
    """Builds a dimension table with one row per key.

    Args:
        rows: The height, which is also the number of distinct keys.
        base: The first key.
        label: The name of the payload column.

    Returns:
        A two column frame keyed by `key`.

    Raises:
        If the frame cannot be built.
    """
    var key = Array[DType.int64](rows)
    var payload = Array[DType.int64](rows)
    for i in range(rows):
        key[i] = Int64(base + i)
        payload[i] = Int64(i * 7)
    var series = List[Series]()
    series.append(Series("key", key^))
    series.append(Series(label, payload^))
    return DataFrame.from_series(series^)


def machine_json(options: Options) -> String:
    """Describes the machine the numbers came from.

    A benchmark result that does not say what produced it is not comparable
    against anything, so this goes in every result file.

    Args:
        options: The options, for the label.

    Returns:
        A JSON object.
    """
    var os_name = String("unknown")
    comptime if CompilationTarget.is_linux():
        os_name = String("linux")
    elif CompilationTarget.is_macos():
        os_name = String("macos")

    var arch = String("aarch64")
    comptime if CompilationTarget.is_x86():
        arch = String("x86_64")

    return String(
        '{"label": "',
        options.label,
        '", "os": "',
        os_name,
        '", "arch": "',
        arch,
        '", "physical_cores": ',
        num_physical_cores(),
        ', "logical_cores": ',
        num_logical_cores(),
        ', "simd_bytes": ',
        simd_width_of[DType.uint8](),
        "}",
    )


def results_json(harness: Harness) -> String:
    """Renders the results as JSON for `tools/bench_compare.py`.

    Args:
        harness: The harness, after running.

    Returns:
        A JSON document, ending in a newline.
    """
    var out = String('{\n  "schema": 1,\n  "version": "', version(), '",\n')
    out += String('  "machine": ', machine_json(harness.options), ",\n")
    out += String(
        '  "config": {"rows": ',
        harness.options.rows,
        ', "repetitions": ',
        harness.options.repetitions,
        ', "min_runtime_secs": ',
        harness.options.min_seconds,
        ', "max_runtime_secs": ',
        harness.max_seconds(),
        "},\n",
    )
    out += String('  "benchmarks": [\n')
    for i in range(len(harness.results)):
        ref result = harness.results[i]
        out += String(
            '    {"name": "',
            result.name,
            '", "unit": "',
            result.unit,
            '", "items": ',
            result.items,
            ', "median_secs": ',
            result.median(),
            ', "min_secs": ',
            result.samples[0],
            ', "q1_secs": ',
            result.q1(),
            ', "q3_secs": ',
            result.q3(),
            ', "per_item_ns": ',
            result.per_item_ns(),
            ', "items_per_second": ',
            result.items_per_second(),
            ', "samples": [',
        )
        for j in range(len(result.samples)):
            if j > 0:
                out += ", "
            out += String(result.samples[j])
        out += "]}"
        if i + 1 < len(harness.results):
            out += ","
        out += "\n"
    out += "  ]\n}\n"
    return out^


def main() raises:
    """Runs the suite and writes the results.

    Raises:
        If an argument is bad, a benchmark raises, or the results cannot be written.
    """
    var options = parse_options()
    var json_path = options.json_path
    print("firepanda microbenchmarks, version", version())
    print(
        "rows",
        options.rows,
        "repetitions",
        options.repetitions,
        "cores",
        num_physical_cores(),
        "physical",
        num_logical_cores(),
        "logical",
    )
    print()
    print(
        pad("benchmark", 26),
        lpad("median", 12),
        lpad("iqr", 8),
        lpad("per item", 12),
        lpad("throughput", 16),
    )

    var harness = Harness(options^)
    var started = perf_counter_ns()
    bench_bitmap(harness)
    bench_buffer(harness)
    bench_array(harness)
    bench_kernel(harness)
    bench_sort(harness)
    bench_frame(harness)
    bench_hash(harness)
    bench_group(harness)
    bench_join(harness)
    bench_nulls(harness)
    bench_csv(harness)
    bench_strings(harness)
    bench_text(harness)
    bench_dispatch(harness)
    var elapsed = Float64(perf_counter_ns() - started) / 1e9

    print()
    print(
        "ran",
        len(harness.results),
        "benchmarks,",
        harness.skipped,
        "filtered out, in",
        duration_text(elapsed),
    )

    if json_path.byte_length() > 0:
        var handle = open(json_path, "w")
        handle.write(results_json(harness))
        handle.close()
        print("wrote", json_path)
