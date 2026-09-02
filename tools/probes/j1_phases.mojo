from std.time import perf_counter_ns

from firepanda import DataFrame, JoinKind, Series
from firepanda.array import AnyArray, Array
from firepanda.frame.groupby import AggKind, AggSpec
from firepanda.join.pairs import join_indices
from firepanda.kernel.select import take_any

comptime LEFT = 10000000
comptime RIGHT = 10000


def _median(mut xs: List[Int]) -> Int:
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[j] < xs[i]:
                var t = xs[i]
                xs[i] = xs[j]
                xs[j] = t
    return xs[len(xs) // 2]


def _keys(name: String) -> List[String]:
    var out = List[String]()
    out.append(name)
    return out^


def _left() raises -> DataFrame:
    var id1 = Array[DType.int32](LEFT)
    var id2 = Array[DType.int32](LEFT)
    var id3 = Array[DType.int32](LEFT)
    var v1 = Array[DType.float64](LEFT)
    var seed = 987654321
    for i in range(LEFT):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        id1[i] = Int32(seed % RIGHT)
        id2[i] = Int32(seed % (RIGHT * 10))
        id3[i] = Int32(seed % LEFT)
        v1[i] = Float64(seed % 100)
    var columns: List[Series] = [
        Series("id1", id1^),
        Series("id2", id2^),
        Series("id3", id3^),
        Series("v1", v1^),
    ]
    return DataFrame.from_series(columns^)


def _right() raises -> DataFrame:
    var id1 = Array[DType.int32](RIGHT)
    var v2 = Array[DType.float64](RIGHT)
    for i in range(RIGHT):
        id1[i] = Int32(i)
        v2[i] = Float64(i % 7)
    var columns: List[Series] = [Series("id1", id1^), Series("v2", v2^)]
    return DataFrame.from_series(columns^)


def main() raises:
    var left = _left()
    var right = _right()

    var pairing = List[Int]()
    var whole = List[Int]()
    var two = List[Int]()
    var reduce = List[Int]()
    var reduced = List[Int]()

    for _ in range(5):
        var lk = List[Int]()
        lk.append(0)
        var rk = List[Int]()
        rk.append(0)

        var t0 = perf_counter_ns()
        var pairs = join_indices(
            left.columns, lk, LEFT, right.columns, rk, RIGHT, JoinKind.INNER
        )
        var t1 = perf_counter_ns()
        var got = take_any(left[3], pairs.left_at)
        var got2 = take_any(right[1], pairs.right_at)
        var t2 = perf_counter_ns()
        pairing.append(Int(t1 - t0) // 1000000)
        two.append(Int(t2 - t1) // 1000000)
        print("  rows", len(pairs), Int(len(got)), Int(len(got2)))
        _ = got^
        _ = got2^
        _ = pairs^

        var t3 = perf_counter_ns()
        var joined = left.join(right, _keys("id1"), JoinKind.INNER)
        var t4 = perf_counter_ns()
        whole.append(Int(t4 - t3) // 1000000)

        var height = len(joined)
        var specs: List[AggSpec] = [
            AggSpec("v1", AggKind.SUM, "v1"),
            AggSpec("v2", AggKind.SUM, "v2"),
        ]
        var zeros = Array[DType.int32](height)
        var t5 = perf_counter_ns()
        var one = joined.with_column(Series("all", zeros^))
        var out = one.group_by(_keys("all"), specs^, True, False)
        var t6 = perf_counter_ns()
        reduce.append(Int(t6 - t5) // 1000000)
        _ = len(out)

        var direct: List[AggSpec] = [
            AggSpec("v1", AggKind.SUM, "v1"),
            AggSpec("v2", AggKind.SUM, "v2"),
        ]
        var t7 = perf_counter_ns()
        var straight = joined.agg(direct^)
        var t8 = perf_counter_ns()
        reduced.append(Int(t8 - t7) // 1000000)
        _ = len(straight)
        _ = joined^

    print("pair only          ", _median(pairing), "ms")
    print("gather v1 and v2   ", _median(two), "ms")
    print("whole join 5 cols  ", _median(whole), "ms")
    print("bench reduce       ", _median(reduce), "ms")
    print("frame agg          ", _median(reduced), "ms")
