"""One compiled pattern, run down a text column.

Everything else in this package is about one pattern and one piece of text. This
is the part that makes it a kernel: the pattern is compiled once, before any row
is read, and every row runs the same program.

That order is the whole design. Compiling is parsing, a walk to decide what RE2
would refuse, and a walk that writes instructions, and none of it depends on the
text, so doing it per row would be the usual way a regular expression accessor
becomes slower than the loop a caller would have written. Compiling once also
puts the refusal in one place: a pattern that cannot be answered is refused
before the column is touched, so a caller gets an error rather than a column
that is half filled.

The kernel takes a program rather than a pattern for the same reason. The layer
that holds the call is the one that has to decide what a refusal means, since a
refusal RE2 would make too is an error the caller should see and a refusal of
this library's own is a feature that is missing, and those are two different
exceptions in Python. Handing the kernel a compiled program keeps that decision
out of the kernel and in the one place that can make it.

Comparison against a null is null, the same as every other kernel here, and it is
handled the same way: the loop writes whatever falls out and the repair at the
end of each morsel clears the rows where the input was missing.

### What the twin checks, and what checks the engine

`text_matches_regex_scalar` is the slow twin, and it is honest about being a
narrower check than the usual one. It runs the same engine, so it cannot catch
the engine being wrong. What it checks is everything around the engine: the
morsel split, the null repair, the reused buffers. Those are the parts of this
file that are not the engine, and reusing a stamp array across rows is exactly
the kind of change that works on one row and fails on the second.

What checks the engine is `tests/differential/regex_match.mojo`, which asks
pandas about thirty thousand generated patterns. A twin that was a second engine
would be a backtracking one, which is the thing document 77 section 2 refuses to
have in the repository at all.
"""

from std.collections.span import Span

from firepanda.array.array import Array
from firepanda.array.strings import StringArray
from firepanda.bitmap.bitmap import Bitmap
from firepanda.exec import parallel_morsels
from firepanda.kernel.mask import repair_range
from firepanda.kernel.regex.parse import decode_into
from firepanda.kernel.regex.pike import Machine
from firepanda.kernel.regex.program import Program


def text_matches_regex(
    a: StringArray, program: Program
) raises -> Array[DType.bool]:
    """Whether each element matches a compiled pattern somewhere in it.

    Args:
        a: The column.
        program: The pattern, already compiled. A program that did not compile
            answers False everywhere, which no caller should ever see, because
            the layer holding the call raises on a refusal before reaching here.

    Returns:
        A bool column, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    # Every row is written below, so the zeroing allocation is a wasted pass.
    var out = Array[DType.bool](overwritten=n)
    var validity = Bitmap(copy=a.validity)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_mut_ptr()
        # One machine and one decode buffer for the whole morsel rather than one
        # of each per row.
        var machine = Machine(program)
        var points = List[UInt32]()
        for i in range(start, stop):
            decode_into(a.unsafe_bytes(i), points)
            var found = machine.matches(program, Span(points))
            dst.unsafe_offset(i).unsafe_write(Scalar[DType.bool](found))
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^
