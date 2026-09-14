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


def _holds(which: Int32, points: Span[UInt32, _], at: Int) -> Bool:
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
        at: How many characters have been read.

    Returns:
        True when the anchor is satisfied here.
    """
    var length = len(points)
    if which == Int32(Int(AT_BEGINNING)) or which == Int32(
        Int(AT_BEGINNING_STRING)
    ):
        return at == 0
    if which == Int32(Int(AT_BEGINNING_LINE)):
        if at == 0:
            return True
        return points[at - 1] == NEWLINE
    if which == Int32(Int(AT_END)) or which == Int32(Int(AT_END_STRING)):
        return at == length
    if which == Int32(Int(AT_END_LINE)):
        if at == length:
            return True
        return points[at] == NEWLINE
    var before = at > 0 and is_word_point(points[at - 1])
    var after = at < length and is_word_point(points[at])
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


def _queue(
    code: List[Instruction],
    mut list: List[Int32],
    mut stamp: List[Int32],
    at: Int32,
    start: Int32,
    points: Span[UInt32, _],
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
            if _holds(instruction.a, points, position):
                stack.append(pc + 1)
        else:
            list.append(pc)


def runs(program: Program, points: Span[UInt32, _]) -> Bool:
    """Whether a compiled pattern matches anywhere in the text.

    Unanchored, because that is the only question pandas asks of the engine.
    `str.match` and `str.fullmatch` are not modes here or upstream: pandas
    rewrites the pattern into `^(pat)` and `^(pat)$` and asks the same question,
    which is a decision worth copying rather than improving on, since the
    rewrite is visible in what the pattern does and not only in the answer.

    Args:
        program: The compiled pattern.
        points: The text, as code points.

    Returns:
        True when some part of the text matches.
    """
    if not program.ok:
        return False
    var length = len(points)
    var stamp = List[Int32](length=len(program.code), fill=-1)
    var here = List[Int32]()
    var next = List[Int32]()
    var position = 0
    while position <= length:
        _queue(
            program.code,
            here,
            stamp,
            Int32(position),
            0,
            points,
            position,
        )
        var i = 0
        while i < len(here):
            var pc = here[i]
            var instruction = program.code[Int(pc)]
            if instruction.op == IN_MATCH:
                return True
            if position < length and _accepts(
                instruction, program.ranges, points[position]
            ):
                _queue(
                    program.code,
                    next,
                    stamp,
                    Int32(position + 1),
                    pc + 1,
                    points,
                    position + 1,
                )
            i += 1
        here = next^
        next = List[Int32]()
        position += 1
    return False


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
