"""Where the time in a group by actually goes, split by phase and by cardinality.

db-benchmark q3, q5 and q7 all group ten million rows on a hundred thousand
distinct keys and firepanda is about twice DuckDB on all three. This asks which
half is responsible: turning the key column into one ordinal per row, or the
scatter that accumulates the values into those ordinals.

The keys are drawn so the distinct count is what the caller asked for and the
rows land on them roughly evenly, which is what the db-benchmark generator does
and is the case a private table per worker is designed for.
"""

from std.time import perf_counter_ns

from firepanda.array.array import Array
from firepanda.hash.grouping import group_ordinals
from firepanda.array.any import AnyArray
from firepanda.kernel.group import (
    group_max,
    group_mean,
    group_sum,
)

comptime ROWS = 10000000
comptime REPS = 5


def _median(mut xs: List[Int]) -> Int:
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[j] < xs[i]:
                var t = xs[i]
                xs[i] = xs[j]
                xs[j] = t
    return xs[len(xs) // 2]


def _report(name: String, mut xs: List[Int]):
    print(name, _median(xs) // 1000, "us")


def run(cardinality: Int) raises:
    print("--", cardinality, "distinct keys over", ROWS, "rows")

    var keys = Array[DType.int32](ROWS)
    var values = Array[DType.float64](ROWS)
    var seed = 987654321
    for i in range(ROWS):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        keys[i] = Int32(seed % cardinality)
        values[i] = Float64(seed % 100)

    var columns = List[AnyArray]()
    columns.append(AnyArray(keys^))
    var at = List[Int]()
    at.append(0)

    var ordinal_times = List[Int]()
    var sum_times = List[Int]()
    var mean_times = List[Int]()
    var max_times = List[Int]()
    var groups = 0

    for _ in range(REPS):
        var t0 = perf_counter_ns()
        var found = group_ordinals(columns, at, ROWS)
        var t1 = perf_counter_ns()
        ordinal_times.append(t1 - t0)
        groups = found.groups

        var t2 = perf_counter_ns()
        var summed = group_sum(values, found.codes, found.groups)
        var t3 = perf_counter_ns()
        sum_times.append(t3 - t2)

        var t4 = perf_counter_ns()
        var averaged = group_mean(values, found.codes, found.groups)
        var t5 = perf_counter_ns()
        mean_times.append(t5 - t4)

        var t6 = perf_counter_ns()
        var largest = group_max(values, found.codes, found.groups)
        var t7 = perf_counter_ns()
        max_times.append(t7 - t6)

        # Keep every answer alive past its timer so nothing is dead code.
        if len(summed) + len(averaged) + len(largest) == 0:
            print("impossible")

    print("   groups found", groups)
    _report("   ordinals   ", ordinal_times)
    _report("   sum        ", sum_times)
    _report("   mean       ", mean_times)
    _report("   max        ", max_times)


def main() raises:
    run(100)
    run(10000)
    run(100000)
    run(1000000)
