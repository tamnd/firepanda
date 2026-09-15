"""Running a compiled pattern over text, once, without backtracking.

The machine is Pike's: every place the pattern could be after reading the same
number of characters is held at once, as a list of instruction indices, and the
whole list steps forward one character at a time. So the text is read once and
never returned to, and the work is the length of the text times the size of the
program however badly the pattern is written. `(a+)+b` against sixty letters is
the pattern that makes a backtracking engine hang, and here it is sixty steps.

Two details are the whole of the correctness argument, and both are about the
stamp array.

An instruction is added to a list at most once per position in the text. That is
what keeps a list shorter than the program, and it is also what makes a repeat
with a body that can match nothing terminate: `(a*)*` goes round its outer loop,
arrives back at an instruction it has already added at this position, and stops.
Without the stamp that is an infinite loop rather than a slow one.

The stamp holds the position rather than a round number, and a thread queued for
the next character is stamped with the next position. So when the search reaches
that position and tries to start a fresh attempt there, the instructions already
queued are recognised as already present. One array, two lists, no clearing.

A column runs the same program over every row, so the three buffers are a struct
a caller can keep rather than three allocations a row pays for. The stamp is
refilled between rows, which is one write per instruction and is next to nothing
beside the work of a row: a row of twenty characters against a program of thirty
instructions is six hundred steps and a refill of thirty. A generation counter
would avoid even that and it would buy a five hundredth of the run in exchange
for an overflow nobody would ever see fail.

`matches` answers whether, and `find` answers where a match ends. The second is
the first with two rules added and it is the harder of the two to get right,
because a pattern that can match in several places has to end where the caller's
engine says it ends rather than wherever the machine happened to notice first.
The order the threads are held in is the answer to both rules. A thread added
earlier is preferred, the start of a fresh attempt is added last, and a thread
reaching a match ends every thread behind it in the list while leaving the ones
in front of it running. So the earliest attempt that can match wins, and among
the ways that attempt can match the one the pattern prefers wins, which is what
leftmost first means and is what both RE2 and Python's `re` do. That order was
already what `_queue` produced before anything read it, and `_queue` says why it
was written that way while nothing needed it.

`counts` is neither of those two. It is a loop around `find`, and the loop is
Arrow's rather than either engine's: three rules about where to look next after
a match, none of which is what a reader would guess, all of which were measured
out of pandas rather than read anywhere. They are written out on `counts` and
document 79 has where each of them came from.
"""

from std.collections.span import Span

from firepanda.kernel.regex.parse import decoded
from firepanda.kernel.regex.program import (
    IN_ANY,
    IN_ANY_ALL,
    IN_AT,
    IN_CHAR,
    IN_JUMP,
    IN_MATCH,
    IN_NOT_SET,
    IN_SET,
    IN_SPLIT,
    Instruction,
    Program,
    in_set,
    is_word_point,
)
from firepanda.kernel.regex.tokens import (
    AT_BEGINNING,
    AT_BEGINNING_LINE,
    AT_BEGINNING_STRING,
    AT_BOUNDARY,
    AT_END,
    AT_END_LINE,
    AT_END_STRING,
    AT_NON_BOUNDARY,
)


comptime NEWLINE: UInt32 = 0x0A
"""The character both engines treat as a line ending, and the only one. Neither
of them counts a carriage return or any of the Unicode line separators, which
is an agreement rather than an accident: RE2 says so and Python says so."""


comptime UNREADABLE: UInt32 = 0xFFFFFFFF
"""A byte in the middle of a character, standing where a character would.

Arrow scans a column for matches in bytes, so a search can begin in the middle
of a character and the text it then searches begins with a byte that is not a
character at all. RE2 reads such a byte as nothing: no class matches it, the
full stop does not match it, and it is not a word character, so the only thing
that can happen at that position is a match of no width. This value is that
byte, and it behaves that way here because `_accepts` refuses it outright and
`is_word_point` says no to anything above the last code point.

It is a position rather than a character, which is the point of having it. A
search that begins two bytes into a three byte character has one of these in
front of it and a search that begins one byte in has two, and the difference is
visible to `^`, to `\\b` and to anything else that reads the text around a
position rather than the text at it.
"""


def _point(points: Span[UInt32, _], lead: Int, at: Int) -> UInt32:
    """The character at a position, counting the unreadable bytes in front.

    Args:
        points: The characters.
        lead: How many unreadable bytes stand in front of them.
        at: The position, from zero.

    Returns:
        The character, or `UNREADABLE` for one of the bytes in front.
    """
    if at < lead:
        return UNREADABLE
    return points[at - lead]


def _holds(which: Int32, points: Span[UInt32, _], lead: Int, at: Int) -> Bool:
    """Whether a position in the text is the kind of position an anchor wants.

    `AT_END` is the line in this function where the two engines part company.
    Python's `$` matches at the end of the text and also just before a newline
    that ends it, so `re.search("a$", "a\\n")` finds something. RE2's matches
    only at the end, so the same pattern through Arrow finds nothing. This is
    the RE2 reading, and the Python one is a different table for the same
    reason the classes are.

    `AT_END_STRING` is `\\z`, and `\\Z` arrives here as the same thing because
    pandas rewrites a trailing `\\Z` to `\\z` on the way to Arrow. A `\\Z` that
    is not trailing is not rewritten and RE2 refuses it, which is a refusal this
    engine does not reproduce because the rewrite belongs to the layer holding
    the call rather than to the tree.

    Args:
        which: The `AT_` value.
        points: The text.
        lead: How many unreadable bytes stand in front of the text.
        at: How many characters have been read.

    Returns:
        True when the anchor is satisfied here.
    """
    var length = lead + len(points)
    if which == Int32(Int(AT_BEGINNING)) or which == Int32(
        Int(AT_BEGINNING_STRING)
    ):
        return at == 0
    if which == Int32(Int(AT_BEGINNING_LINE)):
        if at == 0:
            return True
        return _point(points, lead, at - 1) == NEWLINE
    if which == Int32(Int(AT_END)) or which == Int32(Int(AT_END_STRING)):
        return at == length
    if which == Int32(Int(AT_END_LINE)):
        if at == length:
            return True
        return _point(points, lead, at) == NEWLINE
    var before = at > 0 and is_word_point(_point(points, lead, at - 1))
    var after = at < length and is_word_point(_point(points, lead, at))
    if which == Int32(Int(AT_BOUNDARY)):
        return before != after
    if which == Int32(Int(AT_NON_BOUNDARY)):
        return before == after
    return False


def _accepts(
    instruction: Instruction, ranges: Span[Int32, _], point: UInt32
) -> Bool:
    """Whether an instruction that reads a character accepts this one.

    Args:
        instruction: The instruction.
        ranges: The program's range table.
        point: The character.

    Returns:
        True when the machine may step over it.
    """
    if point == UNREADABLE:
        return False
    if instruction.op == IN_CHAR:
        return Int32(Int(point)) == instruction.a
    if instruction.op == IN_ANY:
        return point != NEWLINE
    if instruction.op == IN_ANY_ALL:
        return True
    if instruction.op == IN_SET:
        return in_set(ranges, instruction.a, instruction.b, point)
    if instruction.op == IN_NOT_SET:
        return not in_set(ranges, instruction.a, instruction.b, point)
    return False


def _byte_width(point: UInt32) -> Int:
    """How many bytes a character takes when it is written out.

    The scan in `counts` moves in bytes because Arrow's does, and this is the
    whole of what it needs to know about how the row was written. Nothing else
    in this file has any idea that a character is more than one thing.

    Args:
        point: The character.

    Returns:
        One, two, three or four.
    """
    if point < 0x80:
        return 1
    if point < 0x800:
        return 2
    if point < 0x10000:
        return 3
    return 4


def _queue(
    code: List[Instruction],
    mut list: List[Int32],
    mut stamp: List[Int32],
    at: Int32,
    start: Int32,
    points: Span[UInt32, _],
    lead: Int,
    position: Int,
):
    """Adds an instruction and everything reachable from it without reading a
    character.

    The walk is depth first with the first arm of a split taken before the
    second, which puts the instructions into the list in the order the pattern
    prefers them. Nothing here needs that order, since the answer is yes or no,
    and it is written that way because the thing that will need it is captures
    and changing the order later is the kind of change that looks harmless.

    Args:
        code: The program.
        list: The list to add to.
        stamp: One entry per instruction, holding the position it was last added
            at.
        at: The position to stamp with, which is where this list will be read.
        start: The instruction to add.
        points: The text.
        lead: How many unreadable bytes stand in front of the text.
        position: Where in the text the assertions are to be judged.
    """
    var stack = List[Int32]()
    stack.append(start)
    while len(stack) > 0:
        var pc = stack.pop()
        if stamp[Int(pc)] == at:
            continue
        stamp[Int(pc)] = at
        var instruction = code[Int(pc)]
        if instruction.op == IN_JUMP:
            stack.append(instruction.a)
        elif instruction.op == IN_SPLIT:
            stack.append(instruction.b)
            stack.append(instruction.a)
        elif instruction.op == IN_AT:
            if _holds(instruction.a, points, lead, position):
                stack.append(pc + 1)
        else:
            list.append(pc)


struct Machine(Movable):
    """The three buffers a run needs, kept so that a column allocates once.

    Sized for one program and usable on any text, which is the shape a column
    wants: compile the pattern, build one of these, and walk the rows. Handing a
    machine a program of a different size is a bug the stamp will not catch, so
    the program is passed to both the constructor and the run rather than being
    remembered here, which keeps the two visibly the same call away from each
    other.
    """

    var stamp: List[Int32]
    """One entry per instruction, holding the position it was last added at."""

    var here: List[Int32]
    """The threads waiting to read the character at this position."""

    var next: List[Int32]
    """The threads that have read it and are waiting for the next one."""

    def __init__(out self, program: Program):
        """Sizes the buffers for a program.

        Args:
            program: The compiled pattern this machine is going to run.
        """
        self.stamp = List[Int32](length=program.sized(), fill=-1)
        self.here = []
        self.next = []

    def matches(mut self, program: Program, points: Span[UInt32, _]) -> Bool:
        """Whether a compiled pattern matches anywhere in the text.

        Unanchored, because that is the only question pandas asks of the engine.
        `str.match` and `str.fullmatch` are not modes here or upstream: pandas
        rewrites the pattern into `^(pat)` and `^(pat)$` and asks the same
        question, which is a decision worth copying rather than improving on,
        since the rewrite is visible in what the pattern does and not only in
        the answer.

        Args:
            program: The compiled pattern.
            points: The text, as code points.

        Returns:
            True when some part of the text matches.
        """
        if not program.ok:
            return False
        for i in range(len(self.stamp)):
            self.stamp[i] = -1
        self.here.clear()
        self.next.clear()
        var length = len(points)
        var position = 0
        while position <= length:
            _queue(
                program.code,
                self.here,
                self.stamp,
                Int32(position),
                0,
                points,
                0,
                position,
            )
            var i = 0
            while i < len(self.here):
                var pc = self.here[i]
                var instruction = program.code[Int(pc)]
                if instruction.op == IN_MATCH:
                    return True
                if position < length and _accepts(
                    instruction, program.ranges, points[position]
                ):
                    _queue(
                        program.code,
                        self.next,
                        self.stamp,
                        Int32(position + 1),
                        pc + 1,
                        points,
                        0,
                        position + 1,
                    )
                i += 1
            swap(self.here, self.next)
            self.next.clear()
            position += 1
        return False

    def find(
        mut self, program: Program, points: Span[UInt32, _], lead: Int
    ) -> Int:
        """Where the leftmost first match of a compiled pattern ends.

        The end and not the start, because the end is the whole of what a scan
        down a row needs: it is where the next search begins, and a match of no
        width is one that ends where the search began rather than one whose two
        ends agree. Those are the same thing for a machine that only ever starts
        where it was told to, and this one does.

        Two rules turn the yes or no of `matches` into a where. New attempts
        stop being started once anything has matched, which is what makes the
        answer the leftmost one. A thread that matches ends every thread behind
        it in the list and leaves the ones in front of it running, which is what
        makes the answer the one the pattern prefers rather than the longest
        one, so `a|ab` ends after one character and `ab|a` ends after two.

        Args:
            program: The compiled pattern.
            points: The text, as code points.
            lead: How many unreadable bytes stand in front of the text, which
                is how a search that begins in the middle of a character says
                so.

        Returns:
            How many positions from the start of the text the match ends, or
            -1 when there is no match. A position counts an unreadable byte as
            one and a character as one, so a caller that wants bytes turns the
            answer back itself.
        """
        if not program.ok:
            return -1
        for i in range(len(self.stamp)):
            self.stamp[i] = -1
        self.here.clear()
        self.next.clear()
        var length = lead + len(points)
        var end = -1
        var position = 0
        while position <= length:
            if end < 0:
                _queue(
                    program.code,
                    self.here,
                    self.stamp,
                    Int32(position),
                    0,
                    points,
                    lead,
                    position,
                )
            elif len(self.here) == 0:
                break
            var i = 0
            while i < len(self.here):
                var pc = self.here[i]
                var instruction = program.code[Int(pc)]
                if instruction.op == IN_MATCH:
                    end = position
                    break
                if position < length and _accepts(
                    instruction,
                    program.ranges,
                    _point(points, lead, position),
                ):
                    _queue(
                        program.code,
                        self.next,
                        self.stamp,
                        Int32(position + 1),
                        pc + 1,
                        points,
                        lead,
                        position + 1,
                    )
                i += 1
            swap(self.here, self.next)
            self.next.clear()
            position += 1
        return end

    def counts(mut self, program: Program, points: Span[UInt32, _]) -> Int:
        """How many times a compiled pattern matches in the text.

        This is the scan Arrow runs and not the one Python's `re` runs, and the
        two differ in three ways that are all visible in ordinary answers. It is
        written to Arrow's because pandas answers `str.count` out of Arrow.

        The text is cut rather than searched from an offset. After a match, the
        rest of the row becomes the text, so `^`, `\\A` and `\\b` are judged
        against the rest and not against the row: `str.count("^a")` on `aaa` is
        three, because each of the three searches starts a text of its own.

        The cursor moves in bytes. After a match of no width it moves one byte,
        which is a third of the way through a three byte character, so an empty
        pattern against a row with an accented letter in it counts the bytes and
        not the characters. The literal path counts an empty pattern the same
        way and document 66 measured it there first.

        The cursor moves to where the match ended unless the match ended where
        the cursor was. That is not the same rule as moving on after a match of
        no width: a match of no width found further along the row moves the
        cursor to it rather than past it, and the same position is then counted
        a second time from there. `str.count("\\b")` on `  a  ` is two in pandas
        for that reason, where Python answers two by finding two different
        boundaries and Arrow answers two by finding one of them twice.

        Args:
            program: The compiled pattern.
            points: The text, as code points.

        Returns:
            How many matches, which is zero for a program that did not compile.
        """
        if not program.ok:
            return 0
        var bytes = 0
        for i in range(len(points)):
            bytes += _byte_width(points[i])
        var seen = 0
        var cursor = 0
        var at = 0
        var lead = 0
        while cursor <= bytes:
            var end = self.find(program, points[at:], lead)
            if end < 0:
                break
            seen += 1
            var step = end
            if end > lead:
                step = lead
                for i in range(end - lead):
                    step += _byte_width(points[at + i])
            if step == 0:
                step = 1
            cursor += step
            if step <= lead:
                lead -= step
            else:
                var rest = step - lead
                lead = 0
                # The cursor can be moved one byte past the end by the rule
                # above, which is where a scan that matched nothing at the end
                # of the row stops. There is no character there to walk over.
                while rest > 0 and at < len(points):
                    var width = _byte_width(points[at])
                    at += 1
                    if rest >= width:
                        rest -= width
                    else:
                        lead = width - rest
                        rest = 0
        return seen


def runs(program: Program, points: Span[UInt32, _]) -> Bool:
    """Whether a compiled pattern matches anywhere in the text.

    The one shot form, which builds a machine, uses it once and drops it. A
    caller with a column to walk wants `Machine` instead.

    Args:
        program: The compiled pattern.
        points: The text, as code points.

    Returns:
        True when some part of the text matches.
    """
    var machine = Machine(program)
    return machine.matches(program, points)


def matches_text(program: Program, text: StringSlice) -> Bool:
    """Whether a compiled pattern matches somewhere in a piece of text.

    Args:
        program: The compiled pattern.
        text: The text.

    Returns:
        True when some part of it matches.
    """
    var points = decoded(text)
    return runs(program, Span(points))


def counts_text(program: Program, text: StringSlice) -> Int:
    """How many times a compiled pattern matches in a piece of text.

    The one shot form of `Machine.counts`, which is where the rules are.

    Args:
        program: The compiled pattern.
        text: The text.

    Returns:
        How many matches.
    """
    var points = decoded(text)
    var machine = Machine(program)
    return machine.counts(program, Span(points))
