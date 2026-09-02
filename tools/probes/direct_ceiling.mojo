"""How wide a direct table is still worth taking on a long column.

`factorize` used to offer the direct table a ceiling of `min(DIRECT_LIMIT,
len(col))`, so a ten million row column would not take a table wider than sixty
five thousand slots however dense it was. `factorize_dense` has always used a
looser rule, one slot per row, because its caller built the values and knows how
much of the span is occupied. This is the measurement behind `factorize` moving
to the same rule.

Each case builds the same groups twice. One column holds them packed into the
span, which is the shape a direct table is being offered. The other holds them
multiplied far enough apart that no row count would accept the table, which is
what pins that column to the hash and is the number to beat.

The direct side is timed as `direct_plan` and then `factorize_dense`, which is
exactly the pair of calls `factorize` makes when it accepts, scan included. It is
spelled out rather than left to `factorize` so that the spans past the ceiling
are still measured, since those are what the ceiling is chosen against.

The second block is the case the old ceiling was protecting against: a wide span
with hardly anything in it, where the table is mostly holes.
"""

from std.time import perf_counter_ns

from firepanda.array.array import Array
from firepanda.hash.factorize import direct_plan, factorize, factorize_dense

comptime ROWS = 10000000
comptime REPS = 3

comptime SPREAD = 1 << 10
"""What the hashed column's values are multiplied by.

Enough that the widest span here lands well past the row count, which is what
makes the direct table refuse it, and small enough to stay inside the window
`direct_plan` will do arithmetic in.
"""


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
    var packed = Array[DType.int64](ROWS)
    var spread = Array[DType.int64](ROWS)
    var seed = 7
    for i in range(ROWS):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        var value = Int64((seed % cardinality) * stride)
        packed[i] = value
        spread[i] = value * Int64(SPREAD)

    var hashed = List[Int]()
    var direct = List[Int]()
    var groups = 0
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        var a = factorize(spread)
        var t1 = perf_counter_ns()
        hashed.append(t1 - t0)
        groups = a.count()

        var t2 = perf_counter_ns()
        var plan = direct_plan[DType.int64](packed, span)
        var b = factorize_dense(packed, plan.span)
        var t3 = perf_counter_ns()
        direct.append(t3 - t2)
        if b.count() != groups:
            print("   MISMATCH", b.count(), "against", groups)

    print(
        "   span",
        span,
        "groups",
        groups,
        "hashed",
        _median(hashed) // 1000,
        "us  direct",
        _median(direct) // 1000,
        "us",
    )


def main() raises:
    print("-- direct against hashed,", ROWS, "rows of int64, span fully used")
    _run(100000, 100000)
    _run(250000, 250000)
    _run(1000000, 1000000)
    _run(2500000, 2500000)
    _run(5000000, 5000000)
    print("-- the same spans holding far fewer values")
    _run(100000, 1000)
    _run(1000000, 1000)
    _run(1000000, 100000)
    _run(5000000, 100)
    _run(5000000, 100000)
