"""What the direct route spends, split into the scan, the build and the wrapper.

`group_phases.mojo` puts the ordinals for ten million rows over a hundred
distinct integer keys at about twenty four milliseconds, and a sweep of
`_factorize_direct_serial` on the same column puts the factorize itself at about
six. The rest is somewhere between the two, and the candidates are the min and
max scan `direct_plan` runs before the route is chosen and whatever
`group_ordinals` does around `factorize`. This times all four so the gap is
attributed rather than assumed.

The worker sweep next to it is the same measurement at every worker count. This
one is the split by phase at whatever the machine picks on its own.
"""

from std.time import perf_counter_ns

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.hash.factorize import (
    DIRECT_LIMIT,
    _factorize_direct_parallel,
    _factorize_direct_serial,
    direct_plan,
    factorize,
)
from firepanda.hash.grouping import group_ordinals

comptime ROWS = 10000000
comptime REPS = 7


def _median(mut xs: List[Int]) -> Int:
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[j] < xs[i]:
                var t = xs[i]
                xs[i] = xs[j]
                xs[j] = t
    return xs[len(xs) // 2]


def _numbers(cardinality: Int) -> Array[DType.int32]:
    var out = Array[DType.int32](ROWS)
    var seed = 987654321
    for i in range(ROWS):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        out[i] = Int32(seed % cardinality)
    return out^


def run(cardinality: Int) raises:
    print("--", cardinality, "distinct over", ROWS, "rows")
    var col = _numbers(cardinality)

    # One untimed call of each so that the first one does not pay for pages the
    # rest of them find already mapped.
    var warm = direct_plan[DType.int32](col, DIRECT_LIMIT)
    if warm.span < 0:
        print("impossible")

    var scan = List[Int]()
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        var plan = direct_plan[DType.int32](col, DIRECT_LIMIT)
        var t1 = perf_counter_ns()
        scan.append(t1 - t0)
        if plan.span < 0:
            print("impossible")
    print("   direct_plan   ", _median(scan) // 1000, "us")

    var one = _factorize_direct_serial[DType.int32](col, cardinality, Int32(0))
    if one.count() == 0:
        print("impossible")
    var serial = List[Int]()
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        var out = _factorize_direct_serial[DType.int32](
            col, cardinality, Int32(0)
        )
        var t1 = perf_counter_ns()
        serial.append(t1 - t0)
        if out.count() == 0:
            print("impossible")
    print("   build serial  ", _median(serial) // 1000, "us")

    for workers in [2, 4, 6, 8, 12, 16, 24, 32]:
        var many = List[Int]()
        for _ in range(REPS):
            var t0 = perf_counter_ns()
            var out = _factorize_direct_parallel[DType.int32](
                col, cardinality, Int32(0), workers
            )
            var t1 = perf_counter_ns()
            many.append(t1 - t0)
            if out.count() == 0:
                print("impossible")
        print("   build", workers, "  ", _median(many) // 1000, "us")

    var whole = List[Int]()
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        var out = factorize[DType.int32](col, UInt64(0x9E3779B97F4A7C15))
        var t1 = perf_counter_ns()
        whole.append(t1 - t0)
        if out.count() == 0:
            print("impossible")
    print("   factorize     ", _median(whole) // 1000, "us")

    var columns = List[AnyArray]()
    columns.append(AnyArray(col^))
    var at = List[Int]()
    at.append(0)
    var grouped = List[Int]()
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        var found = group_ordinals(columns, at, ROWS)
        var t1 = perf_counter_ns()
        grouped.append(t1 - t0)
        if found.groups == 0:
            print("impossible")
    print("   group_ordinals", _median(grouped) // 1000, "us")


def main() raises:
    run(100)
    run(1000)
    run(10000)
    run(65536)
