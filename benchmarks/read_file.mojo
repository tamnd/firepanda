"""Times `read_csv` on a file that already exists.

The microbenchmark beside this one builds its own bytes in memory, which is the
right shape for measuring the reader against itself over time and the wrong shape
for measuring it against pandas. Two readers given two files of the same
description are not given the same work: the column widths differ, the quoting
differs, and the ratio moves with both.

So this takes a path, and the comparison is run by pointing this and
`pandas.read_csv` at the same bytes on the same machine.

Usage:
    mojo build -I . benchmarks/read_file.mojo -o read_file
    ./read_file data.csv 5
"""

from std.sys import argv
from std.time import perf_counter_ns

from firepanda.io.read import ReadOptions, read_csv


def main() raises:
    """Reads a file a few times and prints the median wall clock.

    Raises:
        Error: If no path was given, or the file is not readable as CSV.
    """
    var args = argv()
    if len(args) < 2:
        raise Error("usage: read_file <path.csv> [repetitions]")
    var path = String(args[1])
    var repetitions = 5
    if len(args) > 2:
        repetitions = Int(String(args[2]))

    var options = ReadOptions()
    var times = List[Float64](capacity=repetitions)
    var rows = 0
    var width = 0
    for _ in range(repetitions):
        var started = perf_counter_ns()
        var frame = read_csv(path, options)
        var elapsed = perf_counter_ns() - started
        rows = len(frame)
        width = frame.width()
        times.append(Float64(elapsed) / 1.0e6)

    sort(times)
    var median = times[len(times) // 2]
    print("rows", rows, "columns", width)
    print("median", median, "ms")
    print("best", times[0], "ms")
