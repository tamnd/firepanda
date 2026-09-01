"""How long the ordinals take on the key shapes db-benchmark actually uses.

`group_phases.mojo` next door showed that a group by is mostly the factorize,
using an integer key column. The suite does not: `id1`, `id2` and `id3` are
strings of the form `id001`, and `id4`, `id5` and `id6` are integers, so q1 and
q7 hash text and q5 walks a direct table. This times the four shapes the suite
asks for so the routes can be compared against each other rather than against a
synthetic column none of the queries has.

q1 is one string key of a hundred distinct values, q3 and q7 are one string key
of a hundred thousand, q5 is one integer key of a hundred thousand, and q10 is
all six keys at once, which is one group per row.
"""

from std.time import perf_counter_ns

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder
from firepanda.hash.grouping import group_ordinals

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


def _numbers(offset: Int, cardinality: Int) -> Array[DType.int32]:
    """An integer key column, the shape `id4` through `id6` have."""
    var out = Array[DType.int32](ROWS)
    var seed = 987654321 + offset * 7717
    for i in range(ROWS):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        out[i] = Int32(seed % cardinality)
    return out^


def _text(offset: Int, cardinality: Int) raises -> AnyArray:
    """A string key column, the shape `id1` through `id3` have."""
    var builder = StringBuilder(capacity=ROWS)
    var seed = 987654321 + offset * 7717
    for _ in range(ROWS):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        builder.append(String("id", seed % cardinality).as_bytes())
    return AnyArray(builder^.finish())


def time_keys(
    name: String, var columns: List[AnyArray], var at: List[Int]
) raises:
    var times = List[Int]()
    var groups = 0
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        var found = group_ordinals(columns, at, ROWS)
        var t1 = perf_counter_ns()
        times.append(t1 - t0)
        groups = found.groups
    print(name, _median(times) // 1000, "us,", groups, "groups")


def main() raises:
    var one = List[Int]()
    one.append(0)

    var q1 = List[AnyArray]()
    q1.append(_text(0, 100))
    time_keys("q1  one string key, 100    ", q1^, one.copy())

    var q3 = List[AnyArray]()
    q3.append(_text(2, 100000))
    time_keys("q3  one string key, 100k   ", q3^, one.copy())

    var q5 = List[AnyArray]()
    q5.append(AnyArray(_numbers(5, 100000)))
    time_keys("q5  one int key, 100k      ", q5^, one.copy())

    var q9 = List[AnyArray]()
    q9.append(_text(1, 100))
    q9.append(AnyArray(_numbers(3, 100)))
    var two = List[Int]()
    two.append(0)
    two.append(1)
    time_keys("q9  string and int, 100 each", q9^, two^)

    var q10 = List[AnyArray]()
    q10.append(_text(0, 100))
    q10.append(_text(1, 100))
    q10.append(_text(2, 100000))
    q10.append(AnyArray(_numbers(3, 100)))
    q10.append(AnyArray(_numbers(4, 100)))
    q10.append(AnyArray(_numbers(5, 100000)))
    var six = List[Int]()
    for i in range(6):
        six.append(i)
    time_keys("q10 six keys               ", q10^, six^)
