"""Times `read_parquet` on a file that already exists.

Same shape as `read_arrow_file.mojo` next door and for the same reason: the
number that matters is a real file on a real disk, because how much of a Parquet
read goes to decompression rather than to decoding depends on how the writer
chunked and compressed it, and we do not get to choose that.

The comparison is run by pointing this, DuckDB, Polars and pyarrow at the same
file on the same machine. Our read goes through DuckDB, so the DuckDB number is
the floor and the interesting quantity is the gap between them, which is what
the Arrow C Data Interface handoff and our own assembly cost.

Usage:
    mojo build -I . benchmarks/read_parquet_file.mojo -o read_parquet_file
    ./read_parquet_file data.parquet 5
"""

from std.sys import argv
from std.time import perf_counter_ns

from firepanda.io.parquet import read_parquet


def main() raises:
    """Reads a file a few times and prints the median wall clock.

    Raises:
        Error: If no path was given, or the file is not readable as Parquet.
    """
    var args = argv()
    if len(args) < 2:
        raise Error("usage: read_parquet_file <path.parquet> [repetitions]")
    var path = String(args[1])
    var repetitions = 5
    if len(args) > 2:
        repetitions = Int(String(args[2]))

    var times = List[Float64](capacity=repetitions)
    var rows = 0
    var width = 0
    for _ in range(repetitions):
        var started = perf_counter_ns()
        var frame = read_parquet(path)
        var elapsed = perf_counter_ns() - started
        rows = len(frame)
        width = frame.width()
        times.append(Float64(elapsed) / 1.0e6)

    sort(times)
    var median = times[len(times) // 2]
    print("rows", rows, "columns", width)
    print("median", median, "ms")
    print("best", times[0], "ms")
