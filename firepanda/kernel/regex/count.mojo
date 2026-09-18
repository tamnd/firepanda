"""The two scans that count how many times a pattern matches a row.

Neither of them is the engine, which is why they are here rather than on it, and
it is the same reason the replacing scan next door is not on it either. A scan
is a loop that runs the pattern, does something with the match and looks again,
and both halves of that are the caller's rules rather than the engine's. They
also run two engines now, the backtracker with the machine underneath it, and a
loop that picks between engines cannot live inside one of them.

There are two loops because pandas has two. `str.count` is answered out of
Arrow, and a call carrying flags or a `case=False` or an empty pattern is
answered out of Python's `re` instead. The two disagree about where to look
after a match in three ways, every one of which is visible in an ordinary
answer, and document 79 has where each of them came from.

Arrow cuts the text. After a match the rest of the row becomes the text, so `^`,
`\\A` and `\\b` are judged against the rest and not against the row, and
`str.count("^a")` on `aaa` is three because each search starts a text of its
own. Python never cuts, so the same call is one.

Arrow moves the cursor in bytes. After a match of no width it moves one byte,
which is a third of the way through a three byte character, so an empty pattern
against a row with an accented letter in it counts the bytes. Python does not
move the cursor after a match of no width at all. It looks again at the same
position with the end of the pattern refused, and the search moves itself along
a character at a time when that second look finds nothing, so it counts the
characters. Document 93 has why that is not the same as stepping.

Arrow moves the cursor to where the match ended unless the match ended where the
cursor was, which is not the same rule as moving on after a match of no width. A
match of no width found further along the row moves the cursor to it rather than
past it, and the same position is counted a second time from there.
`str.count("\\b")` on `  a  ` is two in pandas for that reason, where Python
answers two by finding two different boundaries and Arrow answers two by finding
one of them twice.

A cut through the middle of a character is the thing that makes Arrow's loop
awkward to write. The cursor moves in bytes and the text is code points, so a
cut can land inside one, and the number of bytes left over is carried along as a
lead and handed to the engine, which reads a position in front of the text it
was given as a character nothing matches.
"""

from std.collections.span import Span

from firepanda.kernel.regex.backtrack import Bounded, located, searched
from firepanda.kernel.regex.parse import decoded
from firepanda.kernel.regex.pike import Machine, byte_width
from firepanda.kernel.regex.program import Program


def counted(
    program: Program,
    points: Span[UInt32, _],
    mut machine: Machine,
    mut bounded: Bounded,
    mut found: List[Int32],
) raises -> Int:
    """How many times a compiled pattern matches in the text, Arrow's way.

    The three rules in the module docstring are all in the loop below, and none
    of them is what a reader would guess, so the loop is written out rather than
    described.

    Args:
        program: The compiled pattern.
        points: The text, as code points.
        machine: The machine's buffers, reused across rows.
        bounded: The backtracker's buffers, the same way. It answers most rows
            and the machine answers the ones it hands back.
        found: Scratch for the slots of a match, which this has no use for and
            passes on so that a match does not allocate.

    Returns:
        How many matches, which is zero for a program that did not compile.

    Raises:
        Error: Only what the engines raise, which is a row a backreference ran
            out of steps on.
    """
    if not program.ok:
        return 0
    var bytes = 0
    for i in range(len(points)):
        bytes += byte_width(points[i])
    var seen = 0
    var cursor = 0
    var at = 0
    var lead = 0
    while cursor <= bytes:
        var end = located(program, points[at:], lead, machine, bounded, found)
        if end < 0:
            break
        seen += 1
        var step = end
        if end > lead:
            step = lead
            for i in range(end - lead):
                step += byte_width(points[at + i])
        if step == 0:
            step = 1
        cursor += step
        if step <= lead:
            lead -= step
        else:
            var rest = step - lead
            lead = 0
            # The cursor can be moved one byte past the end by the rule above,
            # which is where a scan that matched nothing at the end of the row
            # stops. There is no character there to walk over.
            while rest > 0 and at < len(points):
                var width = byte_width(points[at])
                at += 1
                if rest >= width:
                    rest -= width
                else:
                    lead = width - rest
                    rest = 0
    return seen


def counted_python(
    program: Program,
    points: Span[UInt32, _],
    mut machine: Machine,
    mut bounded: Bounded,
    mut found: List[Int32],
) raises -> Int:
    """How many times a compiled pattern matches in the text, Python's way.

    Look from the cursor, count the match, put the cursor where the match ended.
    The cursor never steps over a character. After a match of no width it stays
    where it is and the next search is told to refuse the end of the pattern at
    that one position, so a lower priority arm that reads a character gets its
    turn at a place an arm that read nothing has already answered, and only when
    that second look comes back with nothing does the search move along by
    itself. The replacing scan follows the same rule, which is the other half of
    the difference between the two engines, since Arrow's two scans follow two
    rules that are not each other. Document 93 section 10.

    Args:
        program: The compiled pattern, which has to have been compiled with
            captures, since the rule needs to know where a match started and not
            only where it ended.
        points: The text, as code points.
        machine: The machine's buffers, reused across rows.
        bounded: The backtracker's buffers, the same way.
        found: Scratch for the slots of a match, which is where the rule reads
            the start of one.

    Returns:
        How many matches, which is zero for a program that did not compile.

    Raises:
        Error: Only what the engines raise, which is a row a backreference ran
            out of steps on.
    """
    if not program.ok:
        return 0
    var n = len(points)
    var seen = 0
    var p = 0
    var advance = False
    while p <= n:
        var end = searched(program, points, p, machine, bounded, found, advance)
        if end < 0:
            break
        seen += 1
        var start = Int(found[0])
        advance = start == end
        p = end
    return seen


def counted_text(program: Program, text: StringSlice) raises -> Int:
    """How many times a compiled pattern matches in a piece of text.

    The one shot form of `counted`, which is where the rules are.

    Args:
        program: The compiled pattern.
        text: The text.

    Returns:
        How many matches.

    Raises:
        Error: Only what the engines raise.
    """
    var points = decoded(text)
    var machine = Machine(program)
    var bounded = Bounded(program)
    var found = List[Int32]()
    return counted(program, Span(points), machine, bounded, found)


def counted_python_text(program: Program, text: StringSlice) raises -> Int:
    """How many times a compiled pattern matches in a piece of text, Python's
    way.

    The one shot form of `counted_python`, which is where the rules are.

    Args:
        program: The compiled pattern, compiled with captures.
        text: The text.

    Returns:
        How many matches.

    Raises:
        Error: Only what the engines raise.
    """
    var points = decoded(text)
    var machine = Machine(program)
    var bounded = Bounded(program)
    var found = List[Int32]()
    return counted_python(program, Span(points), machine, bounded, found)
