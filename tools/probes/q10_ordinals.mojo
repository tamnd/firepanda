"""Inside `group_ordinals` on the q10 shape, one phase at a time.

`q10_phases.mojo` says the ordinals are most of q10. This says which part of the
ordinals. The body below is `firepanda/hash/grouping.mojo` copied with a timer
around every pass it makes: one factorize per key, the widen that turns the first
key's codes into the running packed value, one combine per later key, the
condense that fires when the packed space overflows an int64, and the factorize
at the bottom that makes the tuple dense.

The keys are drawn the way the db-benchmark generator draws them: id1, id2 and
id3 are labels, id4, id5 and id6 are the same values as integers, and the
cardinalities are a hundred, a hundred and a hundred thousand.
"""

from std.sys.info import simd_width_of
from std.time import perf_counter_ns

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder
from firepanda.dtype.lists import ALL
from firepanda.exec import parallel_morsels
from firepanda.hash.factorize import factorize, factorize_dense
from firepanda.hash.factorize import factorize_strings

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
    if len(xs) == 0:
        print(name, "never ran")
        return
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


def _key_codes(
    col: AnyArray, mut groups: Int
) raises -> Array[DType.uint32]:
    """One key's ordinals and group count, the way `_factorize_any` gets them."""
    if col.is_string():
        var text = factorize_strings(col.strings())
        groups = text.count()
        return text^.into_codes()
    comptime for candidate in ALL:
        if col.dtype() == candidate:
            ref view = col.as_typed_view[candidate]()
            var found = factorize(view)
            groups = found.count()
            return found^.into_codes()
    raise Error("probe: unsupported key dtype")


def main() raises:
    print("-- q10 ordinals:", ROWS, "rows, six keys")

    var columns = List[AnyArray]()
    columns.append(_labels("id", 100, 11))
    columns.append(_labels("id", 100, 22))
    columns.append(_labels("id", 100000, 33))
    columns.append(_numbers(100, 44))
    columns.append(_numbers(100, 55))
    columns.append(_numbers(100000, 66))

    comptime lanes = simd_width_of[DType.int64]()

    var factor_times = List[List[Int]]()
    for _ in range(6):
        factor_times.append(List[Int]())
    var widen_times = List[Int]()
    var combine_times = List[Int]()
    var condense_times = List[Int]()
    var final_times = List[Int]()
    var whole_times = List[Int]()
    var groups = 0
    var condensed = 0

    for _ in range(REPS):
        var w0 = perf_counter_ns()

        var space = 0
        var t0 = perf_counter_ns()
        var codes = _key_codes(columns[0], space)
        var t1 = perf_counter_ns()
        factor_times[0].append(t1 - t0)

        var running = Array[DType.int64](overwritten=ROWS)

        def widen(begin: Int, stop: Int) raises {mut running, imm}:
            var into = running.unsafe_ptr()
            var from_ = codes.unsafe_ptr()
            var i = begin
            while i + lanes <= stop:
                into.unsafe_offset(i).unsafe_store(
                    from_.unsafe_offset(i)
                    .unsafe_load[width=lanes]()
                    .cast[DType.int64]()
                )
                i += lanes
            while i < stop:
                into.unsafe_offset(i).unsafe_store(
                    Int64(from_.unsafe_offset(i).unsafe_load())
                )
                i += 1

        var t2 = perf_counter_ns()
        parallel_morsels(widen, ROWS)
        var t3 = perf_counter_ns()
        widen_times.append(t3 - t2)

        for k in range(1, 6):
            var next_groups = 0
            var f0 = perf_counter_ns()
            var next = _key_codes(columns[k], next_groups)
            var f1 = perf_counter_ns()
            factor_times[k].append(f1 - f0)

            if next_groups > 0 and space > Int(Int64.MAX) // next_groups:
                var c0 = perf_counter_ns()
                var found = factorize_dense(running, space)
                space = found.count()
                var packed = found^.into_codes()

                def rewrite(begin: Int, stop: Int) raises {mut running, imm}:
                    var into = running.unsafe_ptr()
                    var from_ = packed.unsafe_ptr()
                    for i in range(begin, stop):
                        into.unsafe_offset(i).unsafe_store(
                            Int64(from_.unsafe_offset(i).unsafe_load())
                        )

                parallel_morsels(rewrite, ROWS)
                var c1 = perf_counter_ns()
                condense_times.append(c1 - c0)
                condensed += 1

            def combine(begin: Int, stop: Int) raises {mut running, imm}:
                var pack = running.unsafe_ptr()
                var right = next.unsafe_ptr()
                var i = begin
                while i + lanes <= stop:
                    pack.unsafe_offset(i).unsafe_store(
                        pack.unsafe_offset(i).unsafe_load[width=lanes]()
                        * Int64(next_groups)
                        + right.unsafe_offset(i)
                        .unsafe_load[width=lanes]()
                        .cast[DType.int64]()
                    )
                    i += lanes
                while i < stop:
                    pack.unsafe_offset(i).unsafe_store(
                        pack.unsafe_offset(i).unsafe_load() * Int64(next_groups)
                        + Int64(right.unsafe_offset(i).unsafe_load())
                    )
                    i += 1

            var g0 = perf_counter_ns()
            parallel_morsels(combine, ROWS)
            var g1 = perf_counter_ns()
            combine_times.append(g1 - g0)
            space *= next_groups

        var e0 = perf_counter_ns()
        var combined = factorize_dense(running, space)
        groups = combined.count()
        var out = combined^.into_codes()
        var e1 = perf_counter_ns()
        final_times.append(e1 - e0)

        var w1 = perf_counter_ns()
        whole_times.append(w1 - w0)

        if len(out) == 0:
            print("impossible")

    print("   groups found     ", groups)
    print("   condense ran     ", condensed // REPS, "times per rep")
    for k in range(6):
        _report("   factorize key " + String(k) + " ", factor_times[k])
    _report("   widen            ", widen_times)
    _report("   combine (one)    ", combine_times)
    _report("   condense (one)   ", condense_times)
    _report("   final factorize  ", final_times)
    _report("   whole            ", whole_times)
