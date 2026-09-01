"""Times `read_arrow` on a file that already exists.

Same reasoning as `read_file.mojo` next door. The microbenchmark in `main.mojo`
hands the importer buffers it built itself, which measures the importer and not
the reader. An Arrow file on disk has framing, a schema message, some number of
record batches and a footer, and how much of the time goes to metadata rather
than to bytes depends entirely on how the producer chunked it. So this takes a
path, and the comparison is run by pointing this and `pyarrow.ipc` at the same
file on the same machine.

Usage:
    mojo build -I . benchmarks/read_arrow_file.mojo -o read_arrow_file
    ./read_arrow_file data.arrow 5
"""

from std.sys import argv
from std.time import perf_counter_ns

from firepanda.io.arrow_ipc import read_arrow


def main() raises:
    """Reads a file a few times and prints the median wall clock.

    Raises:
        Error: If no path was given, or the file is not readable as Arrow.
    """
    var args = argv()
    if len(args) < 2:
        raise Error("usage: read_arrow_file <path.arrow> [repetitions]")
    var path = String(args[1])
    var repetitions = 5
    if len(args) > 2:
        repetitions = Int(String(args[2]))

    var times = List[Float64](capacity=repetitions)
    var rows = 0
    var width = 0
    for _ in range(repetitions):
        var started = perf_counter_ns()
        var frame = read_arrow(path)
        var elapsed = perf_counter_ns() - started
        rows = len(frame)
        width = frame.width()
        times.append(Float64(elapsed) / 1.0e6)

    sort(times)
    var median = times[len(times) // 2]
    print("rows", rows, "columns", width)
    print("median", median, "ms")
    print("best", times[0], "ms")
