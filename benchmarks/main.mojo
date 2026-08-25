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
        [--min-time=MILLISECONDS] [--json=PATH] [--label=NAME] [--filter=SUBSTRING]

`--label` goes in the JSON so a result file says which machine produced it, which
docs/specs/10-benchmarks.md requires and which is easy to forget until two result
files disagree and nobody can say why.
"""

from std.benchmark import Unit, keep, run
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
    group_ordinals,
    hash_into,
    mix,
    radix_partition,
)
from firepanda.kernel import (
    AggKind,
    add,
    aggregate_group_any,
    argsort,
    argsort_multi,
    cast_to,
    divide,
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
    min_of,
    multiply,
    sum_of,
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
            var report = run(
                body,
                num_warmup_iters=1,
                min_runtime_secs=self.options.min_seconds,
                max_runtime_secs=self.options.min_seconds * 3.0 + 0.25,
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
