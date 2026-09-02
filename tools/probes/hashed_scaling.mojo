"""How the hashed factorize scales with cores, by group count.

`tools/probes/q10_ordinals.mojo` says the last pass of a six key group by, the
one factorize of the packed value, is a hundred and seventy eight milliseconds of
a three hundred millisecond ordinals, and that pass is a hashed build over ten
million rows where nearly every row is its own group. That is the largest single
number in the query and this is the first question to ask about it: does it get
faster when it is given more cores.

If it does, the build is the cost and the way to make it cheaper is to make the
build cheaper. If it stops improving early, the merge is the cost, and the merge
is what a partitioned build would delete, because partitions are disjoint and
have nothing to reconcile afterwards.

So this calls the serial and the parallel route by name at fixed worker counts,
rather than through `factorize`, which would pick one. The three cases are a
column where every row is its own group, which is the packed value's shape, one
with a million groups, and one with a hundred thousand, so that the answer can be
read as a function of how much table each worker is carrying.
"""

from std.time import perf_counter_ns

from firepanda.array.array import Array
from firepanda.hash.factorize import (
    _factorize_hashed_parallel,
    _factorize_hashed_partitioned,
    _factorize_hashed_serial,
)
from firepanda.hash.function import DEFAULT_SEED

comptime ROWS = 10000000
comptime REPS = 3


def _median(mut xs: List[Int]) -> Int:
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[j] < xs[i]:
                var t = xs[i]
                xs[i] = xs[j]
                xs[j] = t
    return xs[len(xs) // 2]


def _run(cardinality: Int) raises:
    # Values are spread far enough apart that nothing about them is dense, which
    # is not required here since both routes hash, but keeps the column honest
    # against a reader who tries it through `factorize`.
    var col = Array[DType.int64](ROWS)
    var seed = 7
    for i in range(ROWS):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        col[i] = Int64(seed % cardinality) * 1024

    var serial = List[Int]()
    var groups = 0
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        var a = _factorize_hashed_serial[DType.int64](col, DEFAULT_SEED)
        var t1 = perf_counter_ns()
        serial.append(t1 - t0)
        groups = a.count()

    var base = _median(serial) // 1000
    print("   groups", groups, "serial", base, "us")

    for workers in [2, 4, 8, 16, 32]:
        var split = List[Int]()
        var bad = 0
        for _ in range(REPS):
            var t2 = perf_counter_ns()
            var b = _factorize_hashed_parallel[DType.int64](
                col, DEFAULT_SEED, workers
            )
            var t3 = perf_counter_ns()
            split.append(t3 - t2)
            if b.count() != groups:
                bad = b.count()
        var took = _median(split) // 1000
        print(
            "      workers",
            workers,
            "slices",
            took,
            "us  speedup",
            base * 100 // took,
            "MISMATCH" if bad != 0 else "",
        )

        var cut = List[Int]()
        var wrong = 0
        for _ in range(REPS):
            var t4 = perf_counter_ns()
            var c = _factorize_hashed_partitioned[DType.int64](
                col, DEFAULT_SEED, workers
            )
            var t5 = perf_counter_ns()
            cut.append(t5 - t4)
            if c.count() != groups:
                wrong = c.count()
        var spent = _median(cut) // 1000
        print(
            "      workers",
            workers,
            "parts ",
            spent,
            "us  speedup",
            base * 100 // spent,
            "MISMATCH" if wrong != 0 else "",
        )


def main() raises:
    print("-- hashed factorize scaling,", ROWS, "rows of int64")
    print("   speedup is hundredths, so 250 is two and a half times")
    _run(ROWS)
    _run(1000000)
    _run(100000)
    _run(10000)
    _run(1000)
