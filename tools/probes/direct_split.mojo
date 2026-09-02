"""What a direct factorize is worth splitting across cores, by span.

`_factorize_direct` will only split when `DIRECT_MERGE_BYTES // (span * 4)` is
four or more, which is a span of sixteen thousand. Every wider span runs on one
thread however long the column is. That rule was written when the widest span the
route would ever see was `DIRECT_LIMIT`, and it is the thing standing between a
ten million row column with a hundred thousand values and its other fifteen
cores.

This times the two routes against each other directly, at worker counts the rule
would refuse, so that the refusal can be checked rather than assumed. Both are
called by name rather than through `factorize`, because `factorize` would pick
one and the point is to see both.

The columns are int64 with values packed into the span, which is the shape the
direct route is offered, and the cardinality is the span so the table is full.
The second block holds the same spans with far fewer values in them, which is
where the merge walks slots nothing ever wrote.
"""

from std.time import perf_counter_ns

from firepanda.array.array import Array
from firepanda.hash.factorize import (
    _factorize_direct_parallel,
    _factorize_direct_serial,
)

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


def _run(span: Int, cardinality: Int) raises:
    var stride = span // cardinality
    var col = Array[DType.int64](ROWS)
    var seed = 7
    for i in range(ROWS):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        col[i] = Int64((seed % cardinality) * stride)

    var serial = List[Int]()
    var groups = 0
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        var a = _factorize_direct_serial[DType.int64](col, span, 0)
        var t1 = perf_counter_ns()
        serial.append(t1 - t0)
        groups = a.count()

    print("   span", span, "groups", groups, "serial", _median(serial) // 1000)

    for workers in [4, 8, 16, 32]:
        var split = List[Int]()
        var bad = 0
        for _ in range(REPS):
            var t2 = perf_counter_ns()
            var b = _factorize_direct_parallel[DType.int64](
                col, span, 0, workers
            )
            var t3 = perf_counter_ns()
            split.append(t3 - t2)
            if b.count() != groups:
                bad = b.count()
        print(
            "      workers",
            workers,
            "parallel",
            _median(split) // 1000,
            "us",
            "MISMATCH" if bad != 0 else "",
        )


def main() raises:
    print("-- direct serial against direct split,", ROWS, "rows of int64")
    _run(16384, 16384)
    _run(65536, 65536)
    _run(100000, 100000)
    _run(1000000, 1000000)
    _run(2500000, 2500000)
    print("-- the same spans holding far fewer values")
    _run(100000, 1000)
    _run(1000000, 1000)
    _run(2500000, 10000)
