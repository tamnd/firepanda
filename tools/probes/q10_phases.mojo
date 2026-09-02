"""Where db-benchmark q10's time goes, phase by phase.

q10 groups ten million rows on all six key columns and every row turns out to be
its own tuple, so it is the one query in the suite whose output is as tall as its
input. It is also the query firepanda is furthest behind DuckDB on, and the
aggregation work that fixed q6 barely moved it, which says the cost is somewhere
else. This splits the query into the four things it does and times each one.

The keys are drawn the way the db-benchmark generator draws them: id1, id2 and
id3 are labels, id4, id5 and id6 are the same values as integers, and the
cardinalities are a hundred, a hundred and a hundred thousand.
"""

from std.time import perf_counter_ns

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder
from firepanda.hash.grouping import group_ordinals
from firepanda.kernel.group import group_sum, group_size
from firepanda.kernel.select import take_any

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


def _report(name: String, mut xs: List[Int]):
    print(name, _median(xs) // 1000, "us")


def _labels(prefix: String, cardinality: Int, seed_at: Int) raises -> AnyArray:
    var builder = StringBuilder(capacity=ROWS)
    var seed = seed_at
    for _ in range(ROWS):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        builder.append(String(prefix, seed % cardinality).as_bytes())
    return AnyArray(builder^.finish())


def _numbers(cardinality: Int, seed_at: Int) raises -> AnyArray:
    var col = Array[DType.int32](ROWS)
    var seed = seed_at
    for i in range(ROWS):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        col[i] = Int32(seed % cardinality)
    return AnyArray(col^)


def main() raises:
    print("-- q10 shape:", ROWS, "rows, six keys, one sum and one size")

    var columns = List[AnyArray]()
    columns.append(_labels("id", 100, 11))
    columns.append(_labels("id", 100, 22))
    columns.append(_labels("id", 100000, 33))
    columns.append(_numbers(100, 44))
    columns.append(_numbers(100, 55))
    columns.append(_numbers(100000, 66))

    var values = Array[DType.int32](ROWS)
    for i in range(ROWS):
        values[i] = Int32(i % 100)

    var at = List[Int]()
    for k in range(6):
        at.append(k)

    var ordinal_times = List[Int]()
    var gather_text_times = List[Int]()
    var gather_number_times = List[Int]()
    var sum_times = List[Int]()
    var size_times = List[Int]()
    var groups = 0

    for _ in range(REPS):
        var t0 = perf_counter_ns()
        var found = group_ordinals(columns, at, ROWS)
        var t1 = perf_counter_ns()
        ordinal_times.append(t1 - t0)
        groups = found.groups

        var t2 = perf_counter_ns()
        var a = take_any(columns[0], found.rows_at)
        var b = take_any(columns[1], found.rows_at)
        var c = take_any(columns[2], found.rows_at)
        var t3 = perf_counter_ns()
        gather_text_times.append(t3 - t2)

        var t4 = perf_counter_ns()
        var d = take_any(columns[3], found.rows_at)
        var e = take_any(columns[4], found.rows_at)
        var f = take_any(columns[5], found.rows_at)
        var t5 = perf_counter_ns()
        gather_number_times.append(t5 - t4)

        var t6 = perf_counter_ns()
        var summed = group_sum(values, found.codes, found.groups)
        var t7 = perf_counter_ns()
        sum_times.append(t7 - t6)

        var t8 = perf_counter_ns()
        var sized = group_size(found.codes, found.groups)
        var t9 = perf_counter_ns()
        size_times.append(t9 - t8)

        # Keep every answer alive past its timer so nothing is dead code.
        if len(a) + len(b) + len(c) + len(d) + len(e) + len(f) == 0:
            print("impossible")
        if len(summed) + len(sized) == 0:
            print("impossible")

    print("   groups found     ", groups)
    _report("   ordinals         ", ordinal_times)
    _report("   gather 3 text    ", gather_text_times)
    _report("   gather 3 numeric ", gather_number_times)
    _report("   sum              ", sum_times)
    _report("   size             ", size_times)
