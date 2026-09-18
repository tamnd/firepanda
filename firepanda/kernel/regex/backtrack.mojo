"""Backtracking over a short row, with a bitmap that makes it safe.

The machine next door holds every position the pattern could be in at once, and
that is what makes it safe against a pattern written to blow up: the work is the
length of the row times the size of the program however the pattern is written.
The price is that it pays that every time. A thread list is walked per character,
a thread carries a copy of its slots, and a row that a plain backtracking engine
would answer in forty steps is answered in four hundred.

This is the plain backtracking engine with the blowing up taken out. It follows
one path through the program at a time, and it writes down every pair of an
instruction and a position it has already been to. Arriving at a pair it has
already been to means the path from there has already been tried and has already
failed, since whether a match can be found from an instruction and a position
does not depend on how the search arrived there. So the pair is dropped. That
gives the same bound the machine has, one visit per instruction per position, by
memoising failure rather than by carrying every path at once. It is RE2's
BitState and the name is the bitmap.

What it buys is the constants. There are no thread lists, no stamp array to
refill, and no slot vector per thread: one slot vector is written as the path
goes forward and put back as it comes out, which is the same trick the machine's
queue walk already uses inside a position. On the ClickBench q28 pattern over a
URL that is most of the work of a row.

What it costs is the bitmap, which is one bit per instruction per position and
so grows with the row. That is the whole of the give up rule. A row long enough
that the bitmap would be larger than a quarter of a million bits is handed back
to the caller, who runs the machine on it, which is the same arrangement the
state cache has and for the same reason: this is an accelerator with an engine
underneath it rather than a second engine a caller has to choose between.

The order matters as much as the bound does. A split pushes its second arm and
then its first, so the first comes off the stack first and the path the pattern
prefers is followed first. Attempts are started at one position after another
from the cursor, and the first position that matches wins. Those two together
are leftmost first, which is what the machine does and what both RE2 and Python
do, and it is the reason a match is returned the moment it is reached rather than
after the rest of the stack has been looked at.

The bitmap is not cleared between attempts at different positions, which looks
wrong and is the thing that keeps the whole scan linear. A pair that failed from
one starting position fails from every other, because an instruction and a
position say everything about what is left to do. Nothing in the program reads
where the attempt began: the assertions read the row and the cursor is not one of
them. The one thing that does read it is the refusal a scan asks for after a
match of no width, and that refusal only ever lands on the attempt that starts
at the cursor and only ever on the position it starts at, which is a position no
later attempt can reach, so it cannot write a cell a later attempt would want.

Captures come out of the same walk. A save writes a position into a slot on the
way in and puts back what it found on the way out, which on an explicit stack is
a second kind of entry pushed under the path it protects. So the slots are right
at the moment a match is reached and nowhere else, which is all anybody needs.

That last paragraph is why this file, and not the one next door, is where a
backreference lives. Reading one means asking what a group matched on the way
here, and a path is the only thing that knows. It also takes most of the bitmap
away, since the reason a pair may be dropped was that an instruction and a
position say everything about what is left to do, and with a backreference in
the program they no longer do. What still does is the pair and the slots, so
such a program keeps its bitmap and forgets everything in it the moment a slot
changes value, and a count of steps is the bound underneath that. It is the one
shape here that is never handed back, because there is nothing underneath to
hand it to. Document 95.
"""

from std.collections.span import Span

from firepanda.kernel.regex.parse import decoded
from firepanda.kernel.regex.pike import (
    Machine,
    accepts,
    first_stop,
    holds,
    matches_text,
    point_at,
)
from firepanda.kernel.regex.program import (
    IN_AT,
    IN_BEHIND,
    IN_JUMP,
    IN_LOOK,
    IN_MATCH,
    IN_REF,
    IN_SAVE,
    IN_SPLIT,
    Program,
    word_ranges_unicode,
)
from firepanda.kernel.regex.tokens import (
    AT_BOUNDARY_UNICODE,
    AT_NON_BOUNDARY_UNICODE,
)


comptime MAX_CELLS: Int = 1 << 18
"""How large a bitmap a row is allowed to need, in bits.

A quarter of a million of them, which is thirty two kilobytes and is RE2's
number. It is a bound on the product of the row and the program rather than on
either of them, so a program of forty instructions takes a row of six thousand
characters and a program of four hundred takes a row of six hundred. Rows longer
than that go to the machine, which needs no bitmap at all.

The number could be larger. It is here because the bitmap is cleared per row and
a bitmap the size of a cache is a clear that costs more than the row it is
clearing, not because anything breaks above it.
"""

comptime MAX_STEPS: Int = 1 << 22
"""How many instructions one search may walk when the bitmap speaks for less.

A pattern holding a backreference keeps its bitmap but is made to forget it
every time a slot changes, so the bound above stops being a bound on the walk
and this one is what is left. Four million, which is sixteen times the number of
visits the bitmap allows, because a program that needs a bound at all is one
nobody should be able to hang the process with and not one anybody should be
able to notice on an ordinary row. Document 95 section 6 has why there is no
third engine to hand such a row to and what a caller is told instead.
"""

comptime GAVE_UP: Int = -2
"""The row was too long for the bitmap. Run the machine on it."""

comptime _TOO_LONG: StaticString = "this pattern is taking too long on this row"
"""What a caller is told when a search runs out of steps.

A sentence about the pair rather than about the pattern, because a backreference
that is cheap on most rows and ruinous on one is the ordinary case rather than
the odd one. Upstream says nothing at all here and keeps going, which on the
same pair is a call that does not come back. Document 95 section 6.
"""

comptime NO_MATCH: Int = -1
"""There is no match at or after the cursor. The same value the machine's own
search returns, so a caller that falls back reads one number."""


def _reads_again(
    points: Span[UInt32, _],
    lead: Int,
    opened: Int,
    at: Int,
    width: Int,
    folding: Bool,
) -> Bool:
    """Whether the text at one position is the text a group matched at another.

    The folding half is the ASCII letters and nothing else, which is what
    upstream compares under `(?ai)`. Under the wide alphabet upstream compares
    the two characters by their simple lowercase, which is a table this library
    has not got yet, and the compiler refuses that spelling rather than guessing
    at it. Document 95 section 8.

    Args:
        points: The text, as code points.
        lead: How many unreadable bytes stand in front of them.
        opened: Where the group started.
        at: Where the reference is being read.
        width: How long the group's text is.
        folding: Whether to drop the case of the twenty six ASCII letters.

    Returns:
        True when the two runs are the same run.
    """
    for i in range(width):
        var want = point_at(points, lead, opened + i)
        var have = point_at(points, lead, at + i)
        if folding:
            if want >= 0x41 and want <= 0x5A:
                want += 32
            if have >= 0x41 and have <= 0x5A:
                have += 32
        if want != have:
            return False
    return True


struct Bounded(Movable):
    """The bitmap, the stack and the slots, kept so that a column allocates once.

    Sized for one program and usable on any row, which is the shape a column
    wants. The program is passed to the constructor and to every search rather
    than being held here, which is the arrangement the machine and the state
    cache both have: handing this a program of a different size is a bug no
    assertion here would catch, so the two calls are kept visibly next to each
    other.
    """

    var ok: Bool
    """Whether this can be asked anything at all, which is whether the program
    compiled and whether it holds a question about the text ahead. The
    assertions and the captures it answers itself, unlike the state cache, and
    the row being too long is its other way out. A lookaround is the one shape it
    hands straight back, because a lookaround is a search inside a search and
    there is one stack and one bitmap here to run it on.

    The traffic runs the other way for a backreference. That is the one shape
    this engine takes and the machine cannot, so a program holding one arrives
    here and is never handed back, which is why the compiler refuses a pattern
    holding a lookaround and a backreference at once. Document 95."""

    var seen: List[UInt64]
    """One bit per instruction per position, `pc * (length + 1) + at`. Cleared
    per row, which is the reason the bound above is a small number, and cleared
    again inside a row whenever `stamped` and a slot changes."""

    var slots: List[Int32]
    """Where each group opened and closed on the path being walked, which is the
    answer when a match is reached and is scaffolding at every other moment."""

    var nslots: Int
    """How many of those there are, which is zero for a program compiled without
    captures and is then the whole of what the slot machinery costs."""

    var word: List[Int32]
    """Python's word characters as ranges, held for the same reason the machine
    holds them: the table lives in the compiler's world and coming out of it
    costs a copy of six kilobytes. Empty unless the program asks for one of the
    two Unicode boundaries."""

    var jobs_pc: List[Int32]
    """The stack, as instructions. A negative entry is not an instruction: it is
    the slot `-pc - 1` waiting to be put back, and the number beside it is what
    to put back into it."""

    var jobs_at: List[Int32]
    """The positions of the entries in `jobs_pc`, or the values to restore."""

    var memo: Bool
    """Whether arriving twice at an instruction and a position may be dropped
    outright.

    True for every program but one holding a backreference. That instruction
    reads what the path that arrived took, so two paths standing in the same
    place are two different questions and dropping the second one is dropping an
    answer. With this off the bitmap is still kept, under the narrower rule
    `stamped` below has, and `MAX_STEPS` is what bounds the walk. Document 95.
    """

    var stamped: Bool
    """Whether the bitmap is in use under that narrower rule, which is the mode a
    program holding a backreference runs in.

    An instruction and a position do not say everything about what is left to do
    when a backreference can read a slot, but an instruction and a position and
    the slots do. So the bitmap is kept and everything in it is forgotten the
    moment a slot changes value, which leaves only the arrivals that really are
    the same state. Set per row rather than per program, because a row too long
    for the bitmap has to run without one and there is no second engine to hand
    such a row to. Document 95 section 3."""

    var marks: List[Int32]
    """The cells set since the last slot changed, so that forgetting them is the
    length of that list rather than the length of the bitmap. Empty unless
    `stamped`."""

    var steps: Int
    """How much of that bound this search has spent, counted only when `memo` is
    off since that is the only case where anything but the bitmap bounds the
    walk."""

    var overrun: Bool
    """Whether the last search ran out of steps rather than finding an answer.

    Read by the caller rather than by anything here, because the answer to an
    overrun is an error and this is several layers below the one that raises.
    """

    def __init__(out self, program: Program):
        """Sizes everything for a program.

        Args:
            program: The compiled pattern this is going to run.
        """
        self.ok = program.ok
        self.seen = []
        self.nslots = program.slots
        self.slots = List[Int32](length=self.nslots, fill=-1)
        self.jobs_pc = []
        self.jobs_at = []
        self.word = []
        self.memo = not program.refs
        self.stamped = False
        self.marks = []
        self.steps = 0
        self.overrun = False
        for i in range(len(program.code)):
            var instruction = program.code[i]
            if instruction.op == IN_LOOK or instruction.op == IN_BEHIND:
                # The machine runs a lookaround by starting a second machine on
                # the text around, with buffers of its own, and comes back with
                # one answer. There is no second stack and no second bitmap
                # here, and giving this one a nested walk would mean a path
                # that is two paths, so the whole program goes to the machine.
                # Documents 93 and 94.
                self.ok = False
                return
            if instruction.op != IN_AT or len(self.word) > 0:
                continue
            if instruction.a == Int32(Int(AT_BOUNDARY_UNICODE)) or (
                instruction.a == Int32(Int(AT_NON_BOUNDARY_UNICODE))
            ):
                self.word = word_ranges_unicode()

    def _push(mut self, pc: Int32, at: Int32):
        """Puts one entry on the stack.

        Args:
            pc: The instruction, or `-slot - 1` for a slot to put back.
            at: The position, or the value to put back.
        """
        self.jobs_pc.append(pc)
        self.jobs_at.append(at)

    def _write(mut self, slot: Int, value: Int32):
        """Puts a value in a slot and forgets the bitmap if that changed it.

        Every write to a slot goes through here, the one that opens a group and
        the one that puts back what a group held before it, because the bitmap
        in `stamped` mode is only allowed to speak for as long as the slots
        stand still.

        The comparison matters rather than being a saving. A loop whose body
        matches nothing writes the same two numbers into the same two slots on
        every turn of it, and a write counted as a change on each turn would
        forget the bitmap on each turn and leave the loop with nothing to stop
        it. Document 95 section 3.

        Args:
            slot: Which one.
            value: What to put in it.
        """
        if self.stamped and self.slots[slot] != value:
            self._forget()
        self.slots[slot] = value

    def _forget(mut self):
        """Clears the cells set since the last slot changed."""
        for i in range(len(self.marks)):
            var cell = Int(self.marks[i])
            self.seen[cell >> 6] &= ~(UInt64(1) << UInt64(cell & 63))
        self.marks.clear()

    def _attempt(
        mut self,
        program: Program,
        points: Span[UInt32, _],
        lead: Int,
        length: Int,
        start: Int,
        mut found: List[Int32],
        advance: Bool = False,
    ) -> Int:
        """Follows every path from one starting position until one matches.

        Args:
            program: The compiled pattern.
            points: The text, as code points.
            lead: How many unreadable bytes stand in front of them.
            length: One past the last position, counting those bytes.
            start: Where this attempt begins.
            found: Filled with the slots of the match, when there is one.
            advance: Whether a match of no width is refused here, which is what
                a scan asks for at the position its last match ended at.

        Returns:
            Where the match ends, or `NO_MATCH`.
        """
        self.jobs_pc.clear()
        self.jobs_at.clear()
        self._push(0, Int32(start))
        while len(self.jobs_pc) > 0:
            var pc = self.jobs_pc.pop()
            var at = self.jobs_at.pop()
            if pc < 0:
                # The way out of a save. Everything the save protected has been
                # walked, so the slot goes back to what it held before it.
                self._write(Int(-pc - 1), at)
                continue
            if self.memo or self.stamped:
                var cell = Int(pc) * (length + 1) + Int(at)
                var word_at = cell >> 6
                var bit = UInt64(1) << UInt64(cell & 63)
                if (self.seen[word_at] & bit) != 0:
                    continue
                self.seen[word_at] |= bit
                if self.stamped:
                    self.marks.append(Int32(cell))
            if not self.memo:
                # The bitmap speaks for less here, so the count is what really
                # bounds this. The step is charged at the pop rather than at the
                # push, because what is being bounded is the walking and a push
                # that is never popped costs nothing.
                self.steps += 1
                if self.steps > MAX_STEPS:
                    self.overrun = True
                    return NO_MATCH
            var instruction = program.code[Int(pc)]
            if instruction.op == IN_MATCH:
                if advance and Int(at) == start:
                    # The end of the pattern refused rather than taken, with
                    # the rest of the stack left standing, which is the part
                    # that matters. The arm the pattern liked less gets its
                    # turn at this position and can read a character here.
                    # Ending the attempt instead would turn `(?!x)|\s` into a
                    # pattern that never reads a space.
                    continue
                found.clear()
                for k in range(self.nslots):
                    found.append(self.slots[k])
                return Int(at)
            elif instruction.op == IN_JUMP:
                self._push(instruction.a, at)
            elif instruction.op == IN_SPLIT:
                # The second arm first, so that the first arm comes off the
                # stack first and the path the pattern prefers is the path that
                # is followed.
                self._push(instruction.b, at)
                self._push(instruction.a, at)
            elif instruction.op == IN_AT:
                if holds(instruction.a, points, lead, Int(at), Span(self.word)):
                    self._push(pc + 1, at)
            elif instruction.op == IN_SAVE:
                if self.nslots == 0:
                    # A program compiled with saves in it being asked a question
                    # that has no use for them, which is the machine's rule as
                    # well. The instruction is then a jump to the next one.
                    self._push(pc + 1, at)
                else:
                    var slot = Int(instruction.a)
                    self._push(Int32(-slot - 1), self.slots[slot])
                    self._write(slot, at)
                    self._push(pc + 1, at)
            elif instruction.op == IN_REF:
                var slot = Int(instruction.a)
                var opened = Int(self.slots[slot])
                var closed = Int(self.slots[slot + 1])
                # A group that never took part, which is a slot pair still at
                # minus one, fails the reference rather than matching nothing.
                # That is upstream's answer: `re.match(r"(a)?\1b", "b")` is None
                # and `re.match(r"(a?)\1b", "b")` matches, because in the second
                # one the group took part and matched nothing.
                if opened >= 0 and closed >= opened:
                    var width = closed - opened
                    if Int(at) + width <= length and _reads_again(
                        points,
                        lead,
                        opened,
                        Int(at),
                        width,
                        instruction.b == 1,
                    ):
                        self._push(pc + 1, at + Int32(width))
            elif Int(at) < length and accepts(
                instruction, program.ranges, point_at(points, lead, Int(at))
            ):
                self._push(pc + 1, at + 1)
        return NO_MATCH

    def search(
        mut self,
        program: Program,
        points: Span[UInt32, _],
        lead: Int,
        first: Int,
        mut found: List[Int32],
        advance: Bool = False,
    ) -> Int:
        """Where the leftmost first match at or after a cursor ends.

        The same answer the machine's own search gives, to the same two rules.
        The earliest starting position that can match wins, because the
        positions are tried in order and the first one that matches returns. And
        among the ways that position can match, the one the pattern prefers
        wins, because the walk from it takes the first arm of every split first
        and returns at the first match it reaches.

        Args:
            program: The compiled pattern.
            points: The text, as code points.
            lead: How many unreadable bytes stand in front of the text, which is
                how a search that begins in the middle of a character says so.
            first: The cursor, which is the first position an attempt may start
                at.
            found: Filled with the two ends of the whole match and the two ends
                of every group, as `2k` and `2k + 1` for group `k`, and left
                alone when nothing matched or when the program carries no slots.
            advance: Whether a match of no width is refused at the cursor, which
                is what a scan asks for at the position its last match ended at.
                Only the attempt that starts at the cursor is affected, since an
                attempt further along the row is a different position and a
                match of no width there is one the scan has not seen yet.

        Returns:
            Where the match ends, `NO_MATCH` when there is none, or `GAVE_UP`
            when the row is too long for the bitmap and the caller has to run
            the machine.
        """
        if not self.ok:
            return GAVE_UP
        self.overrun = False
        self.steps = 0
        var length = lead + len(points)
        var cells = program.sized() * (length + 1)
        self.marks.clear()
        if cells > MAX_CELLS:
            # A row too long for the bitmap goes to the machine, and a program
            # holding a backreference is one the machine cannot be handed at
            # all. So that one runs the row without a bitmap and the step count
            # is the whole of the bound. Document 95 section 3.
            if self.memo:
                return GAVE_UP
            self.stamped = False
        else:
            var words = (cells + 63) // 64
            if len(self.seen) < words:
                self.seen = List[UInt64](length=words, fill=0)
            else:
                for i in range(words):
                    self.seen[i] = 0
            self.stamped = not self.memo
        for k in range(self.nslots):
            self.slots[k] = -1

        var position = first
        var skipping = program.first_count > 0
        while position <= length:
            if program.anchored and position > 0:
                # The anchor is asked about the row rather than about the
                # cursor, so a program whose first step is `^` has one attempt
                # in it and it is the one at zero. A cursor above zero is a
                # scan that has already replaced something, and there is nothing
                # left for it to find.
                break
            if skipping:
                # The set of characters a match can begin with, asked here for
                # the same reason the machine asks it: an attempt at a position
                # holding a character no first step accepts is an attempt that
                # dies on its first instruction, and stepping over it costs a
                # lookup rather than a walk.
                position = first_stop(program, points, lead, position, length)
                if position >= length:
                    break
            var end = self._attempt(
                program,
                points,
                lead,
                length,
                position,
                found,
                advance and position == first,
            )
            if end != NO_MATCH:
                return end
            if self.overrun:
                # Nothing later in the row is going to be cheaper than what has
                # already been given up on, and the caller is about to raise
                # anyway, so the rest of the positions are not tried.
                return NO_MATCH
            position += 1
        return NO_MATCH


def searched(
    program: Program,
    points: Span[UInt32, _],
    first: Int,
    mut machine: Machine,
    mut bounded: Bounded,
    mut found: List[Int32],
    advance: Bool = False,
) raises -> Int:
    """The leftmost first match at or after a cursor, from whichever engine can
    answer.

    The backtracker first, because it is the faster of the two on every row it
    will take, and the machine for the rows it hands back. The two answer the
    same question and a caller that read which of them answered would be reading
    something that is none of its business, so this is the only place either of
    them is chosen and every scan that wants a match with its groups in it comes
    through here.

    Args:
        program: The compiled pattern.
        points: The whole row, as code points.
        first: The cursor, which is the first position an attempt may start at.
        machine: The machine's buffers, which the caller keeps across rows.
        bounded: The backtracker's, the same way.
        found: Filled with the two ends of the whole match and of every group.
        advance: Whether a match of no width is refused at the cursor, which is
            Python's rule for what happens after one and is document 93.

    Returns:
        Where the match ends, or -1 when there is none at or after the cursor.

    Raises:
        Error: If the row ran out of steps, which only a pattern holding a
            backreference can do.
    """
    var end = bounded.search(program, points, 0, first, found, advance)
    if bounded.overrun:
        raise Error(_TOO_LONG)
    if end != GAVE_UP:
        return end
    return machine.search(program, points, first, found, advance)


def held_text(program: Program, text: StringSlice) raises -> Bool:
    """Whether a compiled pattern matches somewhere in a piece of text.

    The same question `matches_text` next door answers and the same answer for
    every program that one can read, and here rather than there because a
    program holding a backreference is one the machine cannot read at all. That
    machine is written on threads that merge, this engine is the one that keeps
    a path, and a backreference is a question about the path. So the choosing
    has to live on this side of the two, since this is the side that can see
    both. Document 95.

    The one shot form, which builds both sets of buffers, uses them once and
    drops them. A caller with a column to walk wants to keep them instead.

    Args:
        program: The compiled pattern.
        text: The text.

    Returns:
        True when some part of it matches.

    Raises:
        Error: If the row ran out of steps, which only a pattern holding a
            backreference can do.
    """
    if not program.refs:
        return matches_text(program, text)
    var points = decoded(text)
    var machine = Machine(program)
    var bounded = Bounded(program)
    var found = List[Int32]()
    return searched(program, Span(points), 0, machine, bounded, found) >= 0


def located(
    program: Program,
    points: Span[UInt32, _],
    lead: Int,
    mut machine: Machine,
    mut bounded: Bounded,
    mut found: List[Int32],
) raises -> Int:
    """Where the leftmost first match ends, from whichever engine can answer.

    The other half of the arrangement above, for the caller that cuts its text
    down after every match rather than moving a cursor along one row. It asks
    about the text it was handed from the start of it, and the unreadable bytes
    in front of that text are how a cut through the middle of a character says
    so. Nothing here needs the groups, but a program compiled with them fills
    them anyway, so the caller passes a list rather than this allocating one per
    match.

    Args:
        program: The compiled pattern.
        points: The text as it stands now, as code points.
        lead: How many unreadable bytes stand in front of it.
        machine: The machine's buffers, which the caller keeps across rows.
        bounded: The backtracker's, the same way.
        found: Scratch, filled when the program carries slots.

    Returns:
        Where the match ends, or -1 when there is none.

    Raises:
        Error: If the row ran out of steps, which only a pattern holding a
            backreference can do and which this door never sees, since a
            backreference is refused on the engine that comes through here.
    """
    var end = bounded.search(program, points, lead, 0, found)
    if bounded.overrun:
        raise Error(_TOO_LONG)
    if end != GAVE_UP:
        return end
    return machine.find(program, points, lead)
