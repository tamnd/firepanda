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
from firepanda.array.strings import StringArray, stack_payloads
from firepanda.array.strview import (
    INLINE_CAPACITY,
    StringView,
    VIEW_SIZE,
    make_inline_at,
    make_long_at,
)
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.exec import MORSEL_ROWS, parallel_morsels
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
    is written rather than repaired at the end, because a row can be null here
    for three reasons and only one of them is the input being null.

    This is the same morsel split `text_replace_regex` runs, done once per
    group. Each group gets a view buffer sized before anything starts, since
    there is one 16 byte view per row whatever the pattern finds, and a payload
    per morsel for the answers too long to live inside a view. `stack_payloads`
    lays those end to end afterwards and moves the long views onto them.

    The validity is the one thing here that no other kernel in this file has to
    build, because everywhere else a row is null exactly when the input was. A
    group's bitmap starts empty and a thread sets the bit of every row its group
    took part in. Two threads never touch the same byte of it: a morsel holds
    131072 rows, which is a whole number of bytes, so the bit a thread writes is
    in a byte no other thread has a row in.

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
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    var groups = program.groups
    var views = List[Buffer](capacity=groups)
    var valid = List[Bitmap](capacity=groups)
    for _ in range(groups):
        views.append(Buffer(n * VIEW_SIZE))
        valid.append(Bitmap(n, all_valid=False))

    var out = List[StringArray](capacity=groups)
    if n == 0 or groups == 0:
        # An empty column answers empty columns and a pattern with no groups
        # answers no columns at all. Neither is worth starting a morsel for.
        for _ in range(groups):
            out.append(StringArray(views.pop(0), Buffer(1), valid.pop(0), 0))
        return out^

    # A payload per morsel per group, laid out so that one group's morsels sit
    # next to each other, which is the order the join below wants them in.
    var morsels = (n + MORSEL_ROWS - 1) // MORSEL_ROWS
    var parts = List[List[UInt8]](capacity=morsels * groups)
    for _ in range(morsels * groups):
        parts.append(List[UInt8]())

    def compute(start: Int, stop: Int) {mut parts, mut valid, mut views, imm}:
        var mine = start // MORSEL_ROWS
        # The pointer to each group's views is taken once for the morsel rather
        # than once per row, which is what every other kernel that writes views
        # does as well.
        var first = views[0].unsafe_mut_ptr().unsafe_bitcast[StringView]()
        var heads = List[type_of(first)](capacity=groups)
        heads.append(first)
        for g in range(1, groups):
            heads.append(views[g].unsafe_mut_ptr().unsafe_bitcast[StringView]())
        # One machine and one of each buffer for the whole morsel rather than
        # one of each per row, which is what the serial version did per column.
        var machine = Machine(program)
        var points = List[UInt32]()
        var offsets = List[Int]()
        var found = List[Int32]()
        for i in range(start, stop):
            # A null's views are written rather than left alone, because a view
            # that was never written is whatever the allocation held and every
            # read of these columns would follow it.
            if not a.is_valid(i):
                for g in range(groups):
                    heads[g].unsafe_offset(i)[] = StringView()
                continue
            var bytes = a.unsafe_bytes(i)
            decode_into(bytes, points)
            offsets.clear()
            var at = 0
            for k in range(len(points)):
                offsets.append(at)
                at += byte_width(points[k])
            offsets.append(at)
            # The whole row is searched from its first position, which is the
            # one place this differs from the replacing scan: that one walks a
            # cursor and this one asks once and stops.
            var end = machine.search(program, Span(points), 0, found)
            if end < 0:
                for g in range(groups):
                    heads[g].unsafe_offset(i)[] = StringView()
                continue
            for g in range(groups):
                var opened = Int(found[(g + 1) * 2])
                var closed = Int(found[(g + 1) * 2 + 1])
                if opened < 0 or closed < opened:
                    heads[g].unsafe_offset(i)[] = StringView()
                    continue
                valid[g].set(i, True)
                var piece = bytes[offsets[opened] : offsets[closed]]
                if len(piece) == 0:
                    heads[g].unsafe_offset(i)[] = StringView()
                elif len(piece) <= INLINE_CAPACITY:
                    heads[g].unsafe_offset(i)[] = make_inline_at(
                        Pointer(to=piece[0]), len(piece)
                    )
                else:
                    # The offset written is inside this morsel's own payload
                    # and is moved onto the real one by `stack_payloads`.
                    ref payload = parts[g * morsels + mine]
                    var spot = len(payload)
                    payload.extend(piece)
                    heads[g].unsafe_offset(i)[] = make_long_at(
                        Pointer(to=piece[0]), len(piece), 0, spot
                    )

    parallel_morsels(compute, n, MORSEL_ROWS)

    for _ in range(groups):
        var held = List[List[UInt8]](capacity=morsels)
        for _ in range(morsels):
            held.append(parts.pop(0))
        var payload = stack_payloads(held^, views[0], n, MORSEL_ROWS)
        out.append(StringArray(views.pop(0), payload^, valid.pop(0), n))
    return out^


def text_replace_regex(
    a: StringArray, program: Program, rewrite: Rewrite, limit: Int = -1
) raises -> StringArray:
    """Writes every element out with matches of a compiled pattern swapped.

    The third kernel here and the first whose answer is text, which used to be
    what kept it on one thread. How long a row comes out is not known until the
    scan has run, so there is no column to allocate up front and fill, and a
    `StringBuilder` is one buffer with one cursor that four threads cannot
    share. Document 80 section 10 named the way out and this is it: a payload
    per morsel, written by the thread that owns that morsel, and one pass at the
    end that puts the pieces end to end and moves the long views onto them. The
    views themselves are 16 bytes each and there is one per row, so they go into
    a buffer sized before anything starts and every thread writes its own rows
    of it.

    It was worth doing because the kernels either side of it were already
    parallel and this one was measured at 0.87 cores where `text_byte_length`
    next door used 3.40, on the same machine in the same run. tamnd/firepanda#830
    has the profile. The two passes are not the same price: the first runs the
    pattern and the second is a `memcpy` per morsel, so the part that was serial
    is now the part that costs nothing.

    Everything the row costs is still paid once per morsel rather than once per
    row. The pattern is compiled and the replacement is read before any of this,
    and the machine, the offsets, the slots and the output buffer are made once
    inside each morsel and handed to every row in it.

    Args:
        a: The column.
        program: The pattern, already compiled with captures. A program that did
            not compile replaces nothing, which no caller should ever see,
            because the layer holding the call raises on a refusal before
            reaching here.
        rewrite: The replacement, already read, and refused the same way.
        limit: How many matches to replace in each element, or a negative
            number for all of them. `str.replace` asks for all of them and
            SQL's `regexp_replace` asks for one unless the call said `g`.

    Returns:
        A text column of the same height, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    var validity = Bitmap(copy=a.validity)
    var views = Buffer(n * VIEW_SIZE)
    if n == 0:
        return StringArray(views^, Buffer(1), validity^, 0)

    var morsels = (n + MORSEL_ROWS - 1) // MORSEL_ROWS
    var parts = List[List[UInt8]](capacity=morsels)
    for _ in range(morsels):
        parts.append(List[UInt8]())

    def compute(start: Int, stop: Int) {mut parts, mut views, imm}:
        ref payload = parts[start // MORSEL_ROWS]
        var dst = views.unsafe_mut_ptr().unsafe_bitcast[StringView]()
        # One machine and one of each buffer for the whole morsel rather than
        # one of each per row, which is what the serial version did per column.
        var machine = Machine(program)
        var points = List[UInt32]()
        var offsets = List[Int]()
        var found = List[Int32]()
        var out = List[UInt8]()
        for i in range(start, stop):
            # A null's view is written rather than left alone, because a view
            # that was never written is whatever the allocation held and every
            # read of this column would follow it.
            if not a.is_valid(i):
                dst.unsafe_offset(i)[] = StringView()
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
                limit,
            )
            if len(out) == 0:
                dst.unsafe_offset(i)[] = StringView()
            elif len(out) <= INLINE_CAPACITY:
                dst.unsafe_offset(i)[] = make_inline_at(
                    Pointer(to=out[0]), len(out)
                )
            else:
                # The offset written is inside this morsel's own payload and is
                # moved onto the real one by `stack_payloads`.
                var at = len(payload)
                payload.extend(Span(out))
                dst.unsafe_offset(i)[] = make_long_at(
                    Pointer(to=out[0]), len(out), 0, at
                )

    parallel_morsels(compute, n, MORSEL_ROWS)

    var payload = stack_payloads(parts^, views, n, MORSEL_ROWS)
    return StringArray(views^, payload^, validity^, n)
