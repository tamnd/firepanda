"""Times `write_arrow` on a frame read off disk.

The mirror of `read_arrow_file.mojo`, and it takes an input path for the same
reason: what a write costs depends on what is in the frame, and a frame built by
a benchmark is made of whatever the benchmark found easy to build. So this reads
a real Arrow file, writes it back out, and the comparison is run by pointing this
and `pyarrow.ipc` at the same input on the same machine.

The read is not timed. It happens once, before the loop, so what is measured is
the write and the frame it writes is the same frame every time.

Usage:
    mojo build -I . benchmarks/write_arrow_file.mojo -o write_arrow_file
    ./write_arrow_file data.arrow /tmp/out.arrow 5 [rows_per_batch]
"""

from std.sys import argv
from std.time import perf_counter_ns

from firepanda.io.arrow_ipc import read_arrow
from firepanda.io.arrow_ipc_write import IpcWriteOptions, write_arrow


def main() raises:
    """Writes a frame a few times and prints the median wall clock.

    Raises:
        Error: If no paths were given, or the input is not readable as Arrow.
    """
    var args = argv()
    if len(args) < 3:
        raise Error(
            "usage: write_arrow_file <in.arrow> <out.arrow> [repetitions]"
            " [rows_per_batch]"
        )
    var source = String(args[1])
    var target = String(args[2])
    var repetitions = 5
    if len(args) > 3:
        repetitions = Int(String(args[3]))
    var options = IpcWriteOptions()
    if len(args) > 4:
        options = IpcWriteOptions(Int(String(args[4])))

    var frame = read_arrow(source)

    var times = List[Float64](capacity=repetitions)
    for _ in range(repetitions):
        var started = perf_counter_ns()
        write_arrow(frame, target, options)
        var elapsed = perf_counter_ns() - started
        times.append(Float64(elapsed) / 1.0e6)

    sort(times)
    var median = times[len(times) // 2]
    print("rows", len(frame), "columns", frame.width())
    print("median", median, "ms")
    print("best", times[0], "ms")
