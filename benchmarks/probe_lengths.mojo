"""What a skewed key does to the table, and where the time goes when it does.

ClickBench q31 groups a hundred million rows by `ClientIP`. Every number the hash
table has ever been tuned against came from a uniform column, and an address
column is not one: a few addresses account for a large share of the rows, most
are seen once, and all of them sit inside a handful of networks so the top of
every key is drawn from a set of eight. tamnd/firepanda#484 asks what that costs
before anything is changed to make it cheaper, which is the right order, because
the answer might be that the table is fine.

Two measurements, and they answer different questions.

**The probe length distribution.** How many slots a successful lookup reads, over
the keys, for a table built exactly the way `_factorize_hashed` builds one. The
mean is not the interesting part and is printed anyway so it can be dismissed;
what a skewed key would do, if it did anything, is grow a few clusters, and a
cluster only shows up at the far end. p99 and the longest probe are the two
columns to read.

**The time split.** Hashing, probing, and building the answer, as three
subtractions of measurements anyone can repeat: `hash_into` over the column, then
`factorize`, then `DataFrame.group_by` with a sum on it. Hashing is the first,
probing is the second minus the first, and the answer is the third minus the
second. A group by that is slow on this column and fast on a uniform one is slow
in one of those three and this says which.

Four columns, all the same height, so the rows are comparable down the page:

    uniform      every row its own group, keys spread with a stride
    head         a thousand hosts per network taking half the rows
    heavy        a thousand hosts per network taking nine tenths of them
    tail         no head at all, which is the nearly unique case q31 is

The two subtractions are subtractions of separate measurements, so on a loaded
machine at a small row count either of them can come out negative. That is the
noise being larger than the phase, and the answer to it is more rows or more
repetitions rather than a floor on the print, which would hide it.

Usage:
    mojo run -I . benchmarks/probe_lengths.mojo [rows] [repetitions]
"""

from std.sys import argv
from std.time import perf_counter_ns

from firepanda.array.array import Array
from firepanda.array.value import Value
from firepanda.buffer.buffer import Buffer
from firepanda.frame.frame import DataFrame
from firepanda.frame.groupby import AggSpec
from firepanda.frame.series import Series
from firepanda.hash import DEFAULT_SEED, HashTable, factorize, hash_into
from firepanda.hash.factorize import CHUNK_ROWS
from firepanda.hash.function import hash_chunk
from firepanda.kernel import AggKind
from firepanda.testing import skewed_int64

comptime DEFAULT_ROWS = 10_000_000
comptime DEFAULT_REPS = 3


def _median(var xs: List[Int]) -> Int:
    """Returns the middle value.

    Args:
        xs: The samples. Sorted in place.

    Returns:
        The median.
    """
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[j] < xs[i]:
                var t = xs[i]
                xs[i] = xs[j]
                xs[j] = t
    return xs[len(xs) // 2]


def _uniform(rows: Int) -> Array[DType.int64]:
    """Builds the column the table has always been measured on.

    Every row is its own group and the values are strided far enough apart that
    the direct table refuses them, which is what sends the column to the hash.

    Args:
        rows: The height.

    Returns:
        The column.
    """
    var out = Array[DType.int64](rows)
    for i in range(rows):
        out[i] = Int64(i) * 2_654_435_761
    return out^


def _table_over(col: Array[DType.int64]) -> HashTable:
    """Builds a table the way the serial factorize builds one.

    Spelled out rather than reached through `factorize` because what is wanted
    afterwards is the table, and `factorize` keeps it. The loop is the one in
    `_factorize_hashed`: a chunk hashed, that chunk probed, so the sizing
    schedule sees the same checkpoints in the same order and the table ends at
    the capacity a real group by would leave it at.

    Args:
        col: The key column.

    Returns:
        The built table.
    """
    var n = len(col)
    var hashes = Buffer(CHUNK_ROWS * 8)
    var codes = Array[DType.uint32](overwritten=n)
    var firsts = List[Int]()
    var table = HashTable(0, DEFAULT_SEED)
    var base = 0
    while base < n:
        var count = min(CHUNK_ROWS, n - base)
        hash_chunk(col, base, count, DEFAULT_SEED, hashes)
        table.build(
            hashes,
            col.data.validity,
            False,
            base,
            base,
            count,
            n,
            0,
            codes,
            firsts,
        )
        base += count
    return table^


def _report(name: String, var col: Array[DType.int64], reps: Int) raises:
    """Measures one key column and prints its two rows.

    Args:
        name: What to call it in the table.
        col: The keys.
        reps: How many times to time each phase.

    Raises:
        If the group by raises.
    """
    var rows = len(col)

    var table = _table_over(col)
    var stats = table.probe_lengths()
    print(
        "  ",
        name,
        "groups",
        len(table),
        "slots",
        stats.capacity,
        "load",
        stats.load(),
        "mean",
        stats.mean(),
        "p50",
        stats.quantile(0.5),
        "p90",
        stats.quantile(0.9),
        "p99",
        stats.quantile(0.99),
        "max",
        stats.longest(),
    )

    var values = Array[DType.int64](rows)
    for i in range(rows):
        values[i] = Int64(i & 0xFF)
    var series = List[Series]()
    series.append(Series("key", Array[DType.int64](copy=col)))
    series.append(Series("value", values^))
    var frame = DataFrame.from_series(series^)

    var hashing = List[Int]()
    var whole_factorize = List[Int]()
    var whole_group = List[Int]()
    # Allocated once and written into once before the timer starts, because a
    # freshly mapped eight hundred megabytes costs its page faults on the first
    # write and charging those to the hashing pass reads as the hash being ten
    # times its real cost.
    var buffer = Buffer(rows * 8)
    hash_into(col, DEFAULT_SEED, buffer)
    for _ in range(reps):
        var t0 = perf_counter_ns()
        hash_into(col, DEFAULT_SEED, buffer)
        var t1 = perf_counter_ns()
        hashing.append(t1 - t0)

        var t2 = perf_counter_ns()
        var codes = factorize(col)
        var t3 = perf_counter_ns()
        whole_factorize.append(t3 - t2)
        if codes.count() == 0:
            print("   the factorize found nothing")

        var t4 = perf_counter_ns()
        var out = frame.group_by(["key"], [AggSpec("value", AggKind.SUM)])
        var t5 = perf_counter_ns()
        whole_group.append(t5 - t4)
        if out.rows == 0:
            print("   the group by found nothing")

    var hashed = _median(hashing^) // 1000
    var factorized = _median(whole_factorize^) // 1000
    var grouped = _median(whole_group^) // 1000
    print(
        "  ",
        name,
        "hash",
        hashed,
        "us  probe",
        factorized - hashed,
        "us  answer",
        grouped - factorized,
        "us  group_by",
        grouped,
        "us",
    )


def main() raises:
    """Runs the four columns and prints both tables.

    Raises:
        If a group by raises.
    """
    var args = argv()
    var rows = DEFAULT_ROWS
    if len(args) > 1:
        rows = Int(String(args[1]))
    var reps = DEFAULT_REPS
    if len(args) > 2:
        reps = Int(String(args[2]))

    print("-- probe lengths and the time split,", rows, "rows,", reps, "reps")
    _report("uniform", _uniform(rows), reps)
    _report("head   ", skewed_int64(rows, 1024, 0.5), reps)
    _report("heavy  ", skewed_int64(rows, 1024, 0.9), reps)
    _report("tail   ", skewed_int64(rows, 0, 0.0), reps)
