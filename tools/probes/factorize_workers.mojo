"""What the parallel factorize costs at every worker count, by cardinality.

`_parallel_workers` picks a worker count from a cost model, and the model had a
merge term in it that grew with the worker count because the merge was serial.
The merge is not serial any more, so the term had to be refitted, and this is
what it was refitted against.

The shape it exposes is that the best worker count is still well below the core
count on a column with many groups, for a reason that has nothing to do with the
merge. Every worker builds a table holding roughly all of the column's groups,
so the tables together are the group count times the worker count, and past the
point where that stops fitting in the shared cache each extra worker slows every
other one down. This prints the whole curve so the crossing can be read off
rather than guessed at.
"""

from std.time import perf_counter_ns

from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder, StringArray
from firepanda.hash.factorize import (
    _factorize_hashed_parallel,
    _factorize_hashed_serial,
    _factorize_strings_parallel,
    _factorize_strings_serial,
)

comptime ROWS = 10000000
comptime REPS = 3
comptime SEED = UInt64(0x9E3779B97F4A7C15)


def _median(mut xs: List[Int]) -> Int:
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[j] < xs[i]:
                var t = xs[i]
                xs[i] = xs[j]
                xs[j] = t
    return xs[len(xs) // 2]


def _numbers(offset: Int, cardinality: Int) -> Array[DType.int32]:
    var out = Array[DType.int32](ROWS)
    var seed = 987654321 + offset * 7717
    for i in range(ROWS):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        out[i] = Int32(seed % cardinality)
    return out^


def _text(offset: Int, cardinality: Int) raises -> StringArray:
    var builder = StringBuilder(capacity=ROWS)
    var seed = 987654321 + offset * 7717
    for _ in range(ROWS):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        builder.append(String("id", seed % cardinality).as_bytes())
    return builder^.finish()


def sweep_numbers(cardinality: Int) raises:
    print("-- int32,", cardinality, "distinct over", ROWS, "rows")
    var col = _numbers(5, cardinality)
    var serial = List[Int]()
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        var out = _factorize_hashed_serial(col, SEED)
        var t1 = perf_counter_ns()
        serial.append(t1 - t0)
        if len(out.firsts) == 0:
            print("impossible")
    print("   serial    ", _median(serial) // 1000, "us")
    for workers in [2, 4, 6, 8, 12, 16, 24, 32]:
        var times = List[Int]()
        for _ in range(REPS):
            var t0 = perf_counter_ns()
            var out = _factorize_hashed_parallel(col, SEED, workers)
            var t1 = perf_counter_ns()
            times.append(t1 - t0)
            if len(out.firsts) == 0:
                print("impossible")
        print("   workers", workers, _median(times) // 1000, "us")


def sweep_text(cardinality: Int) raises:
    print("-- string,", cardinality, "distinct over", ROWS, "rows")
    var col = _text(2, cardinality)
    var serial = List[Int]()
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        var out = _factorize_strings_serial(col, SEED)
        var t1 = perf_counter_ns()
        serial.append(t1 - t0)
        if len(out.firsts) == 0:
            print("impossible")
    print("   serial    ", _median(serial) // 1000, "us")
    for workers in [2, 4, 6, 8, 12, 16, 24, 32]:
        var times = List[Int]()
        for _ in range(REPS):
            var t0 = perf_counter_ns()
            var out = _factorize_strings_parallel(col, SEED, workers)
            var t1 = perf_counter_ns()
            times.append(t1 - t0)
            if len(out.firsts) == 0:
                print("impossible")
        print("   workers", workers, _median(times) // 1000, "us")


def main() raises:
    sweep_numbers(100)
    sweep_numbers(100000)
    sweep_numbers(1000000)
    sweep_text(100)
    sweep_text(100000)
    sweep_text(1000000)
    sweep_numbers(10000000)
