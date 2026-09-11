"""How a limit of ten compares against a full sort as the frame grows.

`benchmarks/main.mojo` runs both at `--rows`, and the row that matters is the
one at a hundred million, which the harness cannot produce because every other
section allocates at `--rows` as well and the machine runs out of memory long
before it gets there. This builds one column and times the two answers.

The block size was measured here too, by calling `_top_rows_core` directly at
sixty five thousand, a million and eight million rows a block. The first two are
within noise of each other at every scale and the third is two to four times
worse, because the first block is sorted in full and eight million rows is a
real sort. So the constant stays where it is.

Run it with `mojo run -I . tools/probes/limitscale.mojo`.
"""

from std.time import perf_counter_ns

from firepanda.array.array import Array
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series


def _median(mut xs: List[Int]) -> Int:
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[j] < xs[i]:
                var t = xs[i]
                xs[i] = xs[j]
                xs[j] = t
    return xs[len(xs) // 2]


def _frame(rows: Int) raises -> DataFrame:
    var key = Array[DType.int64](rows)
    var seed = UInt64(0x2545F4914F6CDD1D)
    for i in range(rows):
        seed = seed * 6364136223846793005 + 1442695040888963407
        key.set_valid(i, Int64((seed >> 33) % 1000000))
    var columns = List[Series]()
    columns.append(Series("key", key^))
    return DataFrame.from_series(columns^)


def main() raises:
    var scales = [1000000, 10000000, 50000000, 100000000]
    var by = List[String]()
    by.append("key")
    var ascending = List[Bool]()
    ascending.append(False)

    for s in range(len(scales)):
        var rows = scales[s]
        var df = _frame(rows)

        var sorted_ms = List[Int]()
        var limit_ms = List[Int]()
        var check = 0
        for _ in range(3):
            var t0 = perf_counter_ns()
            var order = df.argsort(by, ascending, ascending)
            var t1 = perf_counter_ns()
            check += Int(order[0])
            sorted_ms.append(Int(t1 - t0) // 1000000)

            t0 = perf_counter_ns()
            var top = df.argsort_limit(by, ascending, ascending, 10)
            t1 = perf_counter_ns()
            check += Int(top[0])
            limit_ms.append(Int(t1 - t0) // 1000000)

        print(
            rows,
            "rows  sort",
            _median(sorted_ms),
            "ms  limit ten",
            _median(limit_ms),
            "ms  check",
            check,
        )
