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

Comparison against a null is null, the same as every other kernel here. The two
that answer a number or a flag handle it the usual way, which is to write
whatever falls out and let the repair at the end of each morsel clear the rows
where the input was missing. The one that answers text writes the null itself,
because a builder has to be told what a row is before it can be told what the
next one is.

### What the twin checks, and what checks the engine

`text_matches_regex_scalar` and `text_count_regex_scalar` are the slow twins,
and they are honest about being a narrower check than the usual one. They run
the same engine, so they cannot catch the engine being wrong. What they check is
everything around the engine: the morsel split, the null repair, the reused
buffers. Those are the parts of this file that are not the engine, and reusing a
stamp array across rows is exactly the kind of change that works on one row and
fails on the second.

What checks the engine is `tests/differential/regex_match.mojo`, which asks
pandas about thirty thousand generated patterns. A twin that was a second engine
would be a backtracking one, which is the thing document 77 section 2 refuses to
have in the repository at all.
"""

from std.collections.span import Span

from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.exec import parallel_morsels
from firepanda.kernel.mask import repair_range
from firepanda.kernel.regex.parse import decode_into
from firepanda.kernel.regex.pike import Machine, byte_width
from firepanda.kernel.regex.program import Program
from firepanda.kernel.regex.replace import Rewrite, replaced


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


def text_count_regex(
    a: StringArray, program: Program
) raises -> Array[DType.int64]:
    """How many times a compiled pattern matches in each element.

    The same shape as the kernel above and the same reasons for it, with one
    difference that is worth knowing before reading either: this one runs the
    pattern over a row as many times as the row has matches, plus once more to
    find out that there are no more, where the one above stops at the first.
    So a row that matches nothing costs the same in both and a row full of
    matches costs this one a pass per match. That is what Arrow does as well,
    since RE2 is asked again from after each match, and it is why an empty
    pattern against a long row is the expensive case in both libraries.

    Args:
        a: The column.
        program: The pattern, already compiled. A program that did not compile
            answers zero everywhere, which no caller should ever see, for the
            reason above.

    Returns:
        An int64 column, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    var out = Array[DType.int64](overwritten=n)
    var validity = Bitmap(copy=a.validity)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_mut_ptr()
        var machine = Machine(program)
        var points = List[UInt32]()
        for i in range(start, stop):
            decode_into(a.unsafe_bytes(i), points)
            var seen = machine.counts(program, Span(points))
            dst.unsafe_offset(i).unsafe_write(Int64(seen))
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def text_extract_regex(
    a: StringArray, program: Program
) raises -> List[StringArray]:
    """What each group of the first match held, one column per group.

    The fourth kernel here and the first whose answer is more than one column.
    Everything above answers a column as tall as the one it read, and so does
    this, several times over: one column per capturing group, each as tall as
    the input, and a caller with the group labels puts them side by side into a
    frame.

    Three rules decide what a row gets and all three are pandas', which here
    means Python's, since this is one of the three names that never reach
    Arrow. The match is the leftmost one anywhere in the row rather than one
    anchored at the front, because upstream runs `regex.search`. A row with no
    match is null in every column rather than null in some of them, so the
    columns of one row agree about whether there was a match at all. And a
    group that took no part in the match it was in is null on its own, which is
    the one case where the columns of a row disagree and is what `(a)(x)?`
    answers for a row holding `a`.

    A null row is null everywhere, the same as every other kernel here, and it
    is written rather than repaired at the end because these are builders and a
    builder has to be told what a row is before it can be told the next one.

    This one is serial for the same reason `text_replace_regex` is. The answers
    go into builders, a builder is one buffer with one cursor, and handing four
    threads a share of one is a different design rather than a flag. What is
    paid once per column rather than once per row is everything else: the
    pattern is compiled before the first row, and the machine, the offsets and
    the slots are made here and handed to every row.

    Args:
        a: The column.
        program: The pattern, already compiled with captures. A program that did
            not compile answers nothing, which no caller should ever see,
            because the layer holding the call raises on a refusal before
            reaching here.

    Returns:
        One text column per capturing group, in the order the groups were
        opened. A pattern with no groups answers no columns, which the Python
        layer refuses before reaching here because pandas refuses it too.

    Raises:
        Error: If a builder cannot allocate.
    """
    var n = len(a)
    var groups = program.groups
    var built = List[StringBuilder]()
    for _ in range(groups):
        built.append(StringBuilder(capacity=n))

    var machine = Machine(program)
    var points = List[UInt32]()
    var offsets = List[Int]()
    var found = List[Int32]()
    for i in range(n):
        if not a.is_valid(i):
            for g in range(groups):
                built[g].append_null()
            continue
        var bytes = a.unsafe_bytes(i)
        decode_into(bytes, points)
        offsets.clear()
        var at = 0
        for k in range(len(points)):
            offsets.append(at)
            at += byte_width(points[k])
        offsets.append(at)
        # The whole row is searched from its first position, which is the one
        # place this differs from the replacing scan: that one walks a cursor
        # and this one asks once and stops.
        var end = machine.search(program, Span(points), 0, found)
        if end < 0:
            for g in range(groups):
                built[g].append_null()
            continue
        for g in range(groups):
            var opened = Int(found[(g + 1) * 2])
            var closed = Int(found[(g + 1) * 2 + 1])
            if opened < 0 or closed < opened:
                built[g].append_null()
                continue
            built[g].append(bytes[offsets[opened] : offsets[closed]])

    var out = List[StringArray]()
    for _ in range(groups):
        out.append(built.pop(0).finish())
    return out^


def text_replace_regex(
    a: StringArray, program: Program, rewrite: Rewrite
) raises -> StringArray:
    """Writes every element out with every match of a compiled pattern swapped.

    The third kernel here and the first whose answer is text, which is what
    makes it the odd one of the three. How long a row comes out is not known
    until the scan has run, so the rows go into a builder one at a time rather
    than into a column allocated up front, and that is also why this one is not
    split into morsels: a builder is one buffer with one cursor, and handing
    four threads a share of it is a different design rather than a flag. The
    literal `text_replace` is serial for the same reason and document 80 has
    the note about what closes it, which is a builder per morsel and a join.

    Everything the row costs is still paid once per column rather than once per
    row. The pattern is compiled before the first row, the replacement is read
    before the first row, and the machine, the offsets, the slots and the
    output buffer are made here and handed to every row.

    Args:
        a: The column.
        program: The pattern, already compiled with captures. A program that did
            not compile replaces nothing, which no caller should ever see,
            because the layer holding the call raises on a refusal before
            reaching here.
        rewrite: The replacement, already read, and refused the same way.

    Returns:
        A text column of the same height, null wherever the input is null.

    Raises:
        Error: If the builder cannot allocate.
    """
    var n = len(a)
    var built = StringBuilder(capacity=n)
    var machine = Machine(program)
    var points = List[UInt32]()
    var offsets = List[Int]()
    var found = List[Int32]()
    var out = List[UInt8]()
    for i in range(n):
        if not a.is_valid(i):
            built.append_null()
            continue
        var bytes = a.unsafe_bytes(i)
        decode_into(bytes, points)
        replaced(
            program,
            rewrite,
            bytes,
            Span(points),
            machine,
            offsets,
            found,
            out,
        )
        built.append(Span(out))
    return built^.finish()
