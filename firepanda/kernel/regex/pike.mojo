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
leftmost first means and is what both RE2 and Python's `re` do.

`search` is `find` with two differences, and both of them are for the scan that
replaces. It starts its attempts at a cursor rather than at the start of the
text, so the text around a position is the real one and `^` stays the start of
the row. And it answers where every group matched as well as where the whole
match did, which is a slot vector carried by every thread and is the reason a
program is compiled with saves in it or without them.

What is not here is a scan. Counting is a loop around `find` and replacing is a
loop around `search`, and both loops belong to the caller rather than to either
engine: the rules about where to look next after a match were measured out of
pandas and are not what a reader would guess. They also pick between two engines
now, which is a thing a loop cannot do from inside one of them. So they live in
`count.mojo` and `replace.mojo`.

Four things here have no underscore in front of them and are read by the other
two engines rather than by this one. `accepts` says whether an instruction takes
a character and the state cache asks it once per class. `holds` says whether a
position is the kind an assertion wants, `point_at` reads the character at a
position counting the unreadable bytes in front of it, and `first_stop` steps
over the positions no match can begin at, and the backtracker asks all three.
Every one of them is a question about what the pattern means rather than about
how this machine is written, so a copy next door would be a second answer to the
same question and the two would part company the first time one was edited.
"""

from std.collections.span import Span

from firepanda.kernel.regex.parse import decoded
from firepanda.kernel.regex.program import (
    IN_ANY,
    IN_ANY_ALL,
    IN_AT,
    IN_BEHIND,
    IN_CHAR,
    IN_CUT,
    IN_JUMP,
    IN_LOOK,
    IN_MARK,
    IN_MATCH,
    IN_NOT_SET,
    IN_REF,
    IN_SAVE,
    IN_SET,
    IN_SPLIT,
    IN_TEST,
    Instruction,
    Program,
    in_set,
    is_word_point,
    is_word_point_unicode,
    word_ranges_unicode,
)
from firepanda.kernel.regex.tokens import (
    AT_BEGINNING,
    AT_BEGINNING_LINE,
    AT_BEGINNING_STRING,
    AT_BOUNDARY,
    AT_BOUNDARY_UNICODE,
    AT_END,
    AT_END_LINE,
    AT_END_STRING,
    AT_END_TEXT,
    AT_NON_BOUNDARY,
    AT_NON_BOUNDARY_UNICODE,
    AT_TEXT_NOT_EMPTY,
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
byte, and it behaves that way here because `accepts` refuses it outright and
`is_word_point` says no to anything above the last code point.

It is a position rather than a character, which is the point of having it. A
search that begins two bytes into a three byte character has one of these in
front of it and a search that begins one byte in has two, and the difference is
visible to `^`, to `\\b` and to anything else that reads the text around a
position rather than the text at it.
"""


def point_at(points: Span[UInt32, _], lead: Int, at: Int) -> UInt32:
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


def holds(
    which: Int32,
    points: Span[UInt32, _],
    lead: Int,
    at: Int,
    word: Span[Int32, _],
) -> Bool:
    """Whether a position in the text is the kind of position an anchor wants.

    `AT_END` is RE2's dollar sign, which matches only at the end, and
    `AT_END_TEXT` is Python's, which matches at the end and also just before a
    newline that ends the text, so `re.search("a$", "a\\n")` finds something and
    the same pattern through Arrow finds nothing. Which of the two an
    instruction holds was settled while the pattern was being compiled, so this
    function carries both readings and has no idea which engine asked for
    either.

    `AT_BOUNDARY_UNICODE` and `AT_NON_BOUNDARY_UNICODE` are the same story about
    the word boundary. RE2 asks it against 63 characters and Python asks it
    against 138558, and document 81 is where that was measured. Under `(?a)`
    Python asks against the 63 as well, so both halves land on the pair above
    and neither needs a value of its own.

    `AT_TEXT_NOT_EMPTY` is nothing any engine spells. CPython up to 3.13 fails a
    `\\B` on an empty row and 3.14 does not, so the compiler writes one of these
    in front of a `\\B` when it is answering beside one of the older ones.
    Document 90.

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
        word: Python's word characters as ranges, empty when the program never
            asks for one of the two Unicode boundaries.

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
        return point_at(points, lead, at - 1) == NEWLINE
    if which == Int32(Int(AT_END)) or which == Int32(Int(AT_END_STRING)):
        return at == length
    if which == Int32(Int(AT_END_LINE)):
        if at == length:
            return True
        return point_at(points, lead, at) == NEWLINE
    if which == Int32(Int(AT_END_TEXT)):
        if at == length:
            return True
        if at != length - 1:
            return False
        return point_at(points, lead, at) == NEWLINE
    if which == Int32(Int(AT_TEXT_NOT_EMPTY)):
        # Nobody writes this and the compiler puts it in front of a `\\B` when
        # the interpreter beside it is one of the ones that fails a `\\B` on an
        # empty row. It is a question about the row rather than about the
        # position, which is why it reads `length` and not `at`.
        return length != 0
    if which == Int32(Int(AT_BOUNDARY_UNICODE)) or which == Int32(
        Int(AT_NON_BOUNDARY_UNICODE)
    ):
        var was = at > 0 and is_word_point_unicode(
            point_at(points, lead, at - 1), word
        )
        var next = at < length and is_word_point_unicode(
            point_at(points, lead, at), word
        )
        if which == Int32(Int(AT_BOUNDARY_UNICODE)):
            return was != next
        return was == next
    var before = at > 0 and is_word_point(point_at(points, lead, at - 1))
    var after = at < length and is_word_point(point_at(points, lead, at))
    if which == Int32(Int(AT_BOUNDARY)):
        return before != after
    if which == Int32(Int(AT_NON_BOUNDARY)):
        return before == after
    return False


def accepts(
    instruction: Instruction, ranges: Span[Int32, _], point: UInt32
) -> Bool:
    """Whether an instruction that reads a character accepts this one.

    The cache next door asks the same question of the same instructions, once
    per class of the alphabet rather than once per character, and the two have
    to agree about every character or the cache answers a different pattern from
    the machine it is standing in front of. The backtracker asks it too, one
    instruction at a time. So it is one function with three callers rather than
    three functions that look alike.

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


def byte_width(point: UInt32) -> Int:
    """How many bytes a character takes when it is written out.

    The scan in `counts` moves in bytes because Arrow's does, and this is the
    whole of what it needs to know about how the row was written. Nothing else
    in this file has any idea that a character is more than one thing.

    The scan that replaces borrows it for a different reason. That one moves in
    characters, so it never needs a width to step with, and it needs the widths
    to turn a pair of character positions back into a piece of the row it can
    copy out.

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


def fill_byte_offsets(
    bytes: Span[UInt8, _], points: Span[UInt32, _], mut offsets: List[Int]
):
    """Fills a table turning a character position into a byte position.

    The scans that hand a piece of the row back work in characters and the row
    is written in bytes, so a match that ended at character nine has to be told
    which byte that was. The table has one entry per character and one more for
    the end, and it is the caller's list so that a column pays for it once
    rather than once per row.

    A row that is all ASCII is left with an empty table rather than a table
    holding the numbers zero to its length, because those are the same numbers
    and `byte_of` reads an empty table as saying so. That is the common row in
    the columns this runs over, and writing an `Int` per character of it was
    work that answered nothing.

    It is worth less than it looks. The replacing scan over 200000 URL rows
    with q28's pattern went from about 221 ms to about 212 ms when this was
    put in, which is four percent, because the machine step that reads a
    character costs much more than the entry that was being written beside it.
    tamnd/firepanda#830 listed this table as one of three costs and this is the
    measurement that says which of the three it is.

    The entries are appended rather than written into a table grown to its size
    first, which is the opposite of what `decode_into` next door does, and the
    difference was measured rather than chosen. Growing the list costs a fill
    of everything the last row did not use, and the row that empties it is the
    ASCII row, which is most of them. So a column pays the fill on every row
    that needs a table and saves a capacity check per entry, and on the URL
    column in `benchmarks/main.mojo` that came out slower than what is here.
    tamnd/firepanda#889 has the numbers.

    Args:
        bytes: The row as it is written.
        points: The same row as code points.
        offsets: The table, emptied first and filled only when it is needed.
    """
    offsets.clear()
    if len(bytes) == len(points):
        return
    var at = 0
    for i in range(len(points)):
        offsets.append(at)
        at += byte_width(points[i])
    offsets.append(at)


def byte_of(offsets: List[Int], at: Int) -> Int:
    """Which byte a character position is at.

    Args:
        offsets: The table `fill_byte_offsets` filled, which is empty for a row
            whose characters are all one byte.
        at: The character position, which may be one past the last character.

    Returns:
        The byte position.
    """
    if len(offsets) == 0:
        return at
    return offsets[at]


def first_stop(
    program: Program,
    points: Span[UInt32, _],
    lead: Int,
    from_at: Int,
    length: Int,
) -> Int:
    """The first position at or after one that could begin a match.

    Only the characters are asked about. A position holding one the program
    cannot begin with is stepped over without the walk, and so is one of the
    unreadable bytes a search into the middle of a character puts in front,
    since no first step of any program accepts one of those either.

    Args:
        program: The compiled pattern, which has to have a first character set
            for this to be worth calling.
        points: The text, as code points.
        lead: How many unreadable bytes stand in front of them.
        from_at: Where to start looking.
        length: One past the last position.

    Returns:
        The position, or `length` when the rest of the text cannot begin one.
    """
    var at = from_at
    while at < length:
        var point = point_at(points, lead, at)
        if point != UNREADABLE and in_set(
            program.ranges, program.first_at, program.first_count, point
        ):
            return at
        at += 1
    return length


def _queue(
    program: Program,
    mut list: List[Int32],
    mut slots: List[Int32],
    mut carry: List[Int32],
    mut stamp: List[Int32],
    at: Int32,
    start: Int32,
    points: Span[UInt32, _],
    lead: Int,
    position: Int,
    nslots: Int,
    word: Span[Int32, _],
):
    """Adds an instruction and everything reachable from it without reading a
    character.

    The walk is depth first with the first arm of a split taken before the
    second, which puts the instructions into the list in the order the pattern
    prefers them. The yes or no answer does not need that order, but where a
    match ends does and where each group matched does, so it is the order the
    whole of the rest of this file rests on.

    It is written as a recursion rather than as a loop over an explicit stack
    because of the save instruction. A save writes a position into a slot, walks
    on, and then has to put back what the slot held, since the walk it made is
    one path through the program and the next path does not go through this
    save. A stack would have to carry a copy of the slots per entry to say the
    same thing, and the copy is the expensive part. The depth is the length of a
    chain of instructions that read no character, which a pattern can make long
    with nested repeats and cannot make unbounded, because a repeat is unrolled
    at most a thousand times and the stamp stops the walk the second time it
    reaches the same instruction at the same position.

    The slots are the one thing here that is optional. A program compiled
    without captures has none, `nslots` is zero, nothing is appended beside the
    thread, and the save branch is never reached because no save was emitted.

    The lookahead is the one branch here that is not a step in a walk. It runs a
    whole second machine over the rest of the text before deciding whether to go
    on, which is the only place in this file where answering a question about a
    position costs more than reading a character. Document 93.

    Args:
        program: The compiled pattern, which is the instructions and also the
            ranges, since a lookahead runs a piece of the same program and needs
            everything a run needs.
        list: The list to add to.
        slots: The slots of the threads in `list`, `nslots` of them per thread,
            laid out end to end rather than as a list of lists so that a thread
            costs no allocation.
        carry: The slots of the path being walked, which is what a thread that
            is added here takes a copy of.
        stamp: One entry per instruction, holding the position it was last added
            at.
        at: The position to stamp with, which is where this list will be read.
        start: The instruction to add.
        points: The text.
        lead: How many unreadable bytes stand in front of the text.
        position: Where in the text the assertions are to be judged.
        nslots: How many slots a thread carries.
        word: Python's word characters as ranges, which only the two Unicode
            boundaries read and which is empty when the program holds neither.
    """
    if stamp[Int(start)] == at:
        return
    stamp[Int(start)] = at
    var instruction = program.code[Int(start)]
    if instruction.op == IN_JUMP:
        _queue(
            program,
            list,
            slots,
            carry,
            stamp,
            at,
            instruction.a,
            points,
            lead,
            position,
            nslots,
            word,
        )
    elif instruction.op == IN_SPLIT:
        _queue(
            program,
            list,
            slots,
            carry,
            stamp,
            at,
            instruction.a,
            points,
            lead,
            position,
            nslots,
            word,
        )
        _queue(
            program,
            list,
            slots,
            carry,
            stamp,
            at,
            instruction.b,
            points,
            lead,
            position,
            nslots,
            word,
        )
    elif instruction.op == IN_AT:
        if holds(instruction.a, points, lead, position, word):
            _queue(
                program,
                list,
                slots,
                carry,
                stamp,
                at,
                start + 1,
                points,
                lead,
                position,
                nslots,
                word,
            )
    elif instruction.op == IN_LOOK:
        var found = _looks(program, instruction.a, points, lead, position, word)
        if found == (instruction.b == 1):
            _queue(
                program,
                list,
                slots,
                carry,
                stamp,
                at,
                start + 1,
                points,
                lead,
                position,
                nslots,
                word,
            )
    elif instruction.op == IN_BEHIND:
        # The width is packed above the sign because an instruction carries two
        # numbers and this asks for three. A body wider than the text behind the
        # thread has nowhere to start, which is a False without a machine being
        # run, and a negative lookbehind is happy with that.
        var back = position - (Int(instruction.b) >> 1)
        var found = back >= 0 and _looks(
            program, instruction.a, points, lead, back, word
        )
        if found == ((instruction.b & 1) == 1):
            _queue(
                program,
                list,
                slots,
                carry,
                stamp,
                at,
                start + 1,
                points,
                lead,
                position,
                nslots,
                word,
            )
    elif instruction.op == IN_REF:
        # The thread dies, and the reason it dies rather than being answered is
        # the reason this machine is safe. Two threads standing at the same
        # instruction and the same position are merged here, which is what makes
        # the work the length of the row times the size of the program, and a
        # backreference is a question those two threads can answer differently.
        # So the merge would have to go, and with it the bound.
        #
        # No program holding one of these reaches here. The compiler sets a flag
        # on it, the backtracker takes every such program and never hands one
        # back, and the state cache and this machine both turn it down. This
        # branch is the third of those refusals written where a reader of the
        # walk will meet it. Document 95.
        pass
    elif instruction.op == IN_MARK or instruction.op == IN_CUT:
        # The thread dies here too, and again the merge is the reason. A cut
        # says the choices the group could have made instead are thrown away,
        # and there are no choices to throw away in a walk where every choice is
        # being followed at once: the threads that stand for them are the other
        # threads in this same list and they belong to paths that never entered
        # the group. Obeying it would mean knowing which of them did, which is
        # the merge undone.
        #
        # So no program holding one of these reaches here either, by the same
        # three refusals a backreference gets. Document 99.
        pass
    elif instruction.op == IN_TEST:
        # And the third, which is the first one over again. Whether a group took
        # part is a question two merged threads can answer differently, exactly
        # as a backreference is, so the test would have to be taken after the
        # merge and the merge is what there is to keep. Document 100.
        pass
    elif instruction.op == IN_SAVE:
        if nslots == 0:
            # A caller asking a program with saves in it a question that has no
            # use for them, which is `matches` being handed whatever program is
            # to hand. The instruction is then a jump to the next one.
            _queue(
                program,
                list,
                slots,
                carry,
                stamp,
                at,
                start + 1,
                points,
                lead,
                position,
                nslots,
                word,
            )
            return
        var slot = Int(instruction.a)
        var was = carry[slot]
        carry[slot] = Int32(position)
        _queue(
            program,
            list,
            slots,
            carry,
            stamp,
            at,
            start + 1,
            points,
            lead,
            position,
            nslots,
            word,
        )
        carry[slot] = was
    else:
        list.append(start)
        for i in range(nslots):
            slots.append(carry[i])


def _looks(
    program: Program,
    entry: Int32,
    points: Span[UInt32, _],
    lead: Int,
    position: Int,
    word: Span[Int32, _],
) -> Bool:
    """Whether the body of a lookaround matches starting exactly here.

    The same machine as the one outside, with two differences and both of them
    matter.

    It starts one attempt rather than one per position. A lookaround asks
    whether the body matches at one position, not whether it matches somewhere,
    so `(?=b)` on `ab` at position zero is False.

    And it reads the same text the outer run is reading rather than a piece of
    it cut off at the position, which is what keeps every anchor honest.
    `(?=a$)` has to know where the row ends and `(?<=^a)` has to know where it
    begins, so the text is whole and the position is where to begin.

    Both directions come here and the seed is the whole of the difference. A
    lookahead begins the body where the thread is standing. A lookbehind begins
    it that many characters further back, which lands the end of the body on the
    thread because the compiler has already refused every body whose width is
    not always the same number. Nothing in here knows which direction it is
    answering, and there is nothing it would do with knowing.

    Nothing is carried back out. A group inside the body would keep what it
    matched upstream and does not here, which is why the compiler refuses that
    shape when the caller asked for captures rather than answering it wrongly.

    The buffers are allocated per call. A machine outside is built once per
    column for the good reason that a row should not pay for one, and the same
    argument says this should be held too, but a lookaround can hold a
    lookaround and the nesting is what a held buffer would have to be indexed
    by. It is measured work rather than guessed work and it is named in document
    93 as the thing to do next.

    Args:
        program: The compiled pattern, whose instructions hold the body.
        entry: Where the body starts.
        points: The whole text.
        lead: How many unreadable bytes stand in front of it.
        position: Where the body has to start matching.
        word: Python's word characters as ranges.

    Returns:
        True when the body matches.
    """
    var stamp = List[Int32](length=program.sized(), fill=-1)
    var here = List[Int32]()
    var next = List[Int32]()
    var slots = List[Int32]()
    var carry = List[Int32]()
    var length = lead + len(points)
    var at = position
    _queue(
        program,
        here,
        slots,
        carry,
        stamp,
        Int32(at),
        entry,
        points,
        lead,
        at,
        0,
        word,
    )
    while True:
        var i = 0
        while i < len(here):
            var pc = here[i]
            var instruction = program.code[Int(pc)]
            if instruction.op == IN_MATCH:
                return True
            if at < length and accepts(
                instruction, program.ranges, point_at(points, lead, at)
            ):
                _queue(
                    program,
                    next,
                    slots,
                    carry,
                    stamp,
                    Int32(at + 1),
                    pc + 1,
                    points,
                    lead,
                    at + 1,
                    0,
                    word,
                )
            i += 1
        if at >= length or len(next) == 0:
            # The second half of that is worth more than it looks. A lookbehind
            # asks about a body of a known width and nothing survives past it,
            # so without this the machine would walk to the end of the row every
            # time to learn what it already knew a few characters in. A
            # lookahead wins the same way whenever its body is short, which most
            # of them are.
            return False
        swap(here, next)
        next.clear()
        at += 1


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

    var slots_here: List[Int32]
    """The slots of the threads in `here`, `nslots` of them per thread."""

    var slots_next: List[Int32]
    """The slots of the threads in `next`, the same way."""

    var carry: List[Int32]
    """The slots of the path `_queue` is walking, which it lends rather than
    copies. One of these rather than one per call because the walk puts back
    what it changed on the way out."""

    var nslots: Int
    """How many slots a thread carries, which is zero for a program compiled
    without captures and is then the whole of what the slot machinery costs."""

    var word: List[Int32]
    """Python's word characters as ranges, held here rather than read per
    position because the table lives in the compiler's world and coming out of
    it costs a copy of six kilobytes. Empty unless the program holds one of the
    two Unicode boundaries, which is every program RE2 would have run."""

    def __init__(out self, program: Program):
        """Sizes the buffers for a program.

        Args:
            program: The compiled pattern this machine is going to run.
        """
        self.word = []
        for i in range(len(program.code)):
            var instruction = program.code[i]
            if instruction.op != IN_AT:
                continue
            if instruction.a == Int32(Int(AT_BOUNDARY_UNICODE)) or (
                instruction.a == Int32(Int(AT_NON_BOUNDARY_UNICODE))
            ):
                self.word = word_ranges_unicode()
                break
        self.stamp = List[Int32](length=program.sized(), fill=-1)
        self.here = []
        self.next = []
        self.nslots = program.slots
        self.slots_here = []
        self.slots_next = []
        self.carry = List[Int32](length=self.nslots, fill=-1)

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
        var skipping = program.first_count > 0
        while position <= length:
            if not (program.anchored and position > 0):
                if skipping and len(self.here) == 0:
                    # Nothing is running, so the only thing this position can do
                    # is begin a match, and the program says which characters
                    # can begin one. The same rule the other scan gives a
                    # paragraph to, asked once above because the answer does not
                    # change while the row is being read.
                    position = first_stop(program, points, 0, position, length)
                    if position >= length:
                        break
                _queue(
                    program,
                    self.here,
                    self.slots_here,
                    self.carry,
                    self.stamp,
                    Int32(position),
                    0,
                    points,
                    0,
                    position,
                    0,
                    Span(self.word),
                )
            elif len(self.here) == 0:
                # The same rule the other scan gives a paragraph to: an anchored
                # program starts no attempt above position zero, so once the one
                # attempt it does start has died there is nothing left to read
                # the rest of the row for.
                break
            var i = 0
            while i < len(self.here):
                var pc = self.here[i]
                var instruction = program.code[Int(pc)]
                if instruction.op == IN_MATCH:
                    return True
                if position < length and accepts(
                    instruction, program.ranges, points[position]
                ):
                    _queue(
                        program,
                        self.next,
                        self.slots_next,
                        self.carry,
                        self.stamp,
                        Int32(position + 1),
                        pc + 1,
                        points,
                        0,
                        position + 1,
                        0,
                        Span(self.word),
                    )
                i += 1
            swap(self.here, self.next)
            self.next.clear()
            position += 1
        return False

    def _run(
        mut self,
        program: Program,
        points: Span[UInt32, _],
        lead: Int,
        first: Int,
        mut found: List[Int32],
        advance: Bool = False,
    ) -> Int:
        """Where the leftmost first match ends, starting attempts at `first`.

        Two rules turn the yes or no of `matches` into a where. New attempts
        stop being started once anything has matched, which is what makes the
        answer the leftmost one. A thread that matches ends every thread behind
        it in the list and leaves the ones in front of it running, which is what
        makes the answer the one the pattern prefers rather than the longest
        one, so `a|ab` ends after one character and `ab|a` ends after two.

        `first` is where the attempts begin and not where the text begins, and
        the difference is the whole reason this is one function with two names
        on it. A caller that cut the text and handed over what was left passes
        zero, and the assertions then read the cut text as the whole of it. A
        caller that kept the text whole and moved a cursor along it passes the
        cursor, and the assertions read the text as it really is, so `^` is the
        start of the row rather than the start of what is left. Arrow's two
        scans disagree about which of those is right, and document 80 has the
        measurements: counting cuts, replacing does not.

        Args:
            program: The compiled pattern.
            points: The text, as code points.
            lead: How many unreadable bytes stand in front of the text, which
                is how a search that begins in the middle of a character says
                so.
            first: The first position an attempt may start at.
            found: Filled with the slots of the thread that matched, and left
                alone when nothing matched or when the program carries none.
            advance: Whether a match of no width at `first` is allowed. False is
                the plain search. True is the one a scan asks for when the last
                match it found had no width, and it is the whole of Python's
                rule about repeating a match: the same position is searched
                again and only something wider is taken from it. A thread that
                reaches the end at `first` has read nothing, since nothing has
                been read yet at all, so the test is the position and not the
                slots and a program with no slots in it can be asked as well.
                Document 93.

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
        self.slots_here.clear()
        self.slots_next.clear()
        var length = lead + len(points)
        var end = -1
        var position = first
        var skipping = program.first_count > 0
        while position <= length:
            if end < 0 and not (program.anchored and position > 0):
                # A fresh attempt knows nothing about any group, and it is added
                # behind whatever survived the last character, which is what
                # makes an earlier attempt win over this one.
                #
                # An anchored program is the one case where an attempt is known
                # to fail before it is started, since its first step asks
                # whether the position is zero. Skipping those leaves the loop
                # with nothing in `here` as soon as the attempt at zero has
                # died, and the branch below then ends the row. Position zero
                # and not `first`, because the anchor is asked about the text
                # and a caller searching from a cursor is asking about a
                # position the anchor has already ruled out.
                #
                # The set of characters a match can begin with is the general
                # form of the same idea. With nothing running, this position can
                # only begin a match, and a position holding a character no
                # first step accepts cannot. Running out of row that way ends it
                # too, since a program that has a set is a program that has to
                # read something. An anchored program never has a set, so its
                # scan is the scan it was before this was written.
                if skipping and len(self.here) == 0:
                    position = first_stop(
                        program, points, lead, position, length
                    )
                    if position >= length:
                        break
                for k in range(self.nslots):
                    self.carry[k] = -1
                _queue(
                    program,
                    self.here,
                    self.slots_here,
                    self.carry,
                    self.stamp,
                    Int32(position),
                    0,
                    points,
                    lead,
                    position,
                    self.nslots,
                    Span(self.word),
                )
            elif len(self.here) == 0:
                break
            var i = 0
            while i < len(self.here):
                var pc = self.here[i]
                var instruction = program.code[Int(pc)]
                if instruction.op == IN_MATCH:
                    if advance and position == first:
                        # Refused rather than taken, and the threads behind it
                        # left running, which is the part that matters. What
                        # upstream does here is fail the end of the pattern and
                        # carry on backtracking, so the alternative the pattern
                        # liked less gets its turn and can match something
                        # wider at the same place. Ending the row instead would
                        # turn `(?!x)|\\s` into a pattern that never reads a
                        # space, since the arm that reads nothing is the one it
                        # likes better and would be the only one ever asked.
                        i += 1
                        continue
                    end = position
                    found.clear()
                    for k in range(self.nslots):
                        found.append(self.slots_here[i * self.nslots + k])
                    break
                if position < length and accepts(
                    instruction,
                    program.ranges,
                    point_at(points, lead, position),
                ):
                    for k in range(self.nslots):
                        self.carry[k] = self.slots_here[i * self.nslots + k]
                    _queue(
                        program,
                        self.next,
                        self.slots_next,
                        self.carry,
                        self.stamp,
                        Int32(position + 1),
                        pc + 1,
                        points,
                        lead,
                        position + 1,
                        self.nslots,
                        Span(self.word),
                    )
                i += 1
            swap(self.here, self.next)
            swap(self.slots_here, self.slots_next)
            self.next.clear()
            self.slots_next.clear()
            position += 1
        return end

    def find(
        mut self, program: Program, points: Span[UInt32, _], lead: Int
    ) -> Int:
        """Where the leftmost first match of a compiled pattern ends.

        The end and not the start, because the end is the whole of what the
        counting scan needs: it is where the next search begins, and a match of
        no width is one that ends where the search began rather than one whose
        two ends agree. Those are the same thing for a search that starts at the
        start of the text it was given, and this one does.

        Args:
            program: The compiled pattern.
            points: The text, as code points.
            lead: How many unreadable bytes stand in front of the text.

        Returns:
            Where the match ends, or -1 when there is no match.
        """
        var nothing = List[Int32]()
        return self._run(program, points, lead, 0, nothing)

    def search(
        mut self,
        program: Program,
        points: Span[UInt32, _],
        first: Int,
        mut found: List[Int32],
        advance: Bool = False,
    ) -> Int:
        """Where the leftmost first match at or after a position ends, and what
        each group of it held.

        The text stays whole, so a pattern that reads the text around a position
        reads the real one. That is what the replacing scan wants and what the
        counting scan does not, and the two are different because Arrow's two
        kernels are different rather than because either of them is right.

        Args:
            program: The compiled pattern, which has to have been compiled with
                captures for `found` to hold anything.
            points: The whole text, as code points.
            first: The cursor, which is the first position an attempt may start
                at.
            found: Filled with the two ends of the whole match and the two ends
                of every group, as `2k` and `2k + 1` for group `k`, with -1 for
                a group that did not take part.
            advance: Whether a match of no width at the cursor is allowed,
                which is False for the first search of a row and True for the
                one after a match that had no width.

        Returns:
            Where the match ends, or -1 when there is no match at or after the
            cursor.
        """
        return self._run(program, points, 0, first, found, advance)


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
