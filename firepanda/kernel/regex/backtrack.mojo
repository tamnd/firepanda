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

One shape of repeat is walked in one step rather than three. A class under a
plus or a star is a split, the class, and a jump back to the split, and on the
q28 pattern over a URL that is most of what the engine does: two hundred and
forty five steps a row against eighty six characters. The split is where the
choice actually is, so the class is read at the split and the arm that goes
round comes straight back to the split one position along, and the body and the
jump behind it are never visited. Which splits those are is worked out once per
program by `run_bodies` next door rather than being written into the program, so
the machine and the state cache read the same instructions they always did and
carry no extra comparison for a note they have no use for. It is worth about 1.8
times on the pattern that asked for it. Issue #897.

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

The atomic group is the second shape that is never handed back, and it is here
for the opposite reason. A backreference is a question the machine cannot
answer. A cut is an answer the machine cannot give, because throwing a choice
away means nothing where every choice is being followed at once, and it is
answered here because an explicit stack of choices is a thing you can throw part
of away. It leaves the bitmap alone: what the group matched from a position is
the same whatever path arrived there, so a pair may still be dropped outright,
and the only thing the cut changes is that a row too long for the bitmap runs
under the step count rather than going to a machine that could not run it.
Document 99.

The conditional group is the third, and it is the first reason again rather than
the second. Asking whether a group took part is asking about the path that
arrived, so the other two engines cannot answer it and the bitmap here cannot
hold the answer, and a program holding one runs under exactly the arrangement a
backreference runs under: the bitmap kept, forgotten the moment a slot changes
value, with the step count underneath. What it does not share is the cost, since
the question is settled by looking at two numbers and going one way or the other
rather than by reading the text again. Document 100.

A lookaround is none of those three. It is the one construct here the machine
next door can also run, and it is here so that it can stand beside one of the
three that the machine cannot. A search inside a search runs on the stack that
is already here, from the height the stack stood at when the body started,
which is the same way an atomic group already knows which choices are its own.
The body's marks are forgotten on the way out, because a body that matched
marked every pair on the way and the next question asked of it at the same
position would read those marks as an answer already found. A program whose
only unusual thing is an assertion is still handed straight back, so nothing
that ran on the machine before runs here now. Document 120.
"""

from std.collections.span import Span

from firepanda.kernel.regex.lowerdata import (
    LOWER_DELTA,
    LOWER_EVEN_ONLY,
    LOWER_HIGH,
    LOWER_LOW,
    LOWER_RUNS,
)
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
    IN_CUT,
    IN_JUMP,
    IN_LOOK,
    IN_MARK,
    IN_MATCH,
    IN_REF,
    IN_SAVE,
    IN_SPLIT,
    IN_TEST,
    REF_NARROW,
    REF_WIDE,
    Program,
    run_bodies,
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

It is also what is left when a row is too long for the bitmap and the program is
one no other engine can run, which is the atomic group's case. There the bitmap
is not narrowed, it is absent, and the count is the only bound there is.
Document 99 section 5.
"""

comptime MARKED: Int32 = -0x40000000
"""The stack entry an atomic group leaves where it began.

The stack already holds two kinds of thing, an instruction to walk and a slot to
put back, and the second is written as a negative number so that one test at the
top of the loop tells them apart. This is a third kind and it is written the same
way, far below any slot a program could have, so that the same test catches it
and the loop costs no second comparison on the path that has no atomic group in
it at all.

Popping one means the group ran out of ways to match and the pattern is leaving
it backwards, so there is nothing to do but drop it. Document 99.
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


def lower_table() -> List[List[Int32]]:
    """The simple lowercase runs brought out of the compiler's world.

    Three lists, the lows, the highs and the deltas, in that order and the same
    length. Like the word ranges next door the table only exists while the
    program is being compiled, so something has to copy it out, and the caller is
    expected to hold on to what it gets rather than ask again per row. It is two
    and a half kilobytes, which is why only a program that reads a backreference
    under the wide alphabet asks at all.

    Returns:
        The lows, the highs and the deltas.
    """
    var lows = materialize[LOWER_LOW]()
    var highs = materialize[LOWER_HIGH]()
    var deltas = materialize[LOWER_DELTA]()
    var out = List[List[Int32]]()
    out.append(_held(Span(lows)))
    out.append(_held(Span(highs)))
    out.append(_held(Span(deltas)))
    return out^


def _held(table: Span[Int32, _]) -> List[Int32]:
    """One of those arrays as a list.

    Args:
        table: The array from `lowerdata.mojo`, already materialized.

    Returns:
        The same numbers as a list.
    """
    var out = List[Int32](capacity=len(table))
    for i in range(len(table)):
        out.append(table[i])
    return out^


def simple_lower(
    lows: Span[Int32, _],
    highs: Span[Int32, _],
    deltas: Span[Int32, _],
    point: UInt32,
) -> UInt32:
    """The simple lowercase of one code point, which is upstream's `tolower`.

    A binary search of the runs in lowerdata.mojo, laid out the same way the fold
    table is and read the same way. A code point in no run is its own lowercase,
    which is every code point but the fourteen hundred odd that move.

    This is the run time half of case insensitivity and the only half of it that
    exists at run time. Everything else the flag touches is settled while the
    pattern is being compiled, by widening a literal or a class into a set. A
    backreference cannot be widened that way because what it will be compared
    against is not known until the row is being walked, so it is compared here
    instead, and upstream compares it by this rather than by the fold that the
    literals beside it were compiled under. Document 97.

    Args:
        lows: Where each run starts.
        highs: Where each run ends.
        deltas: What to add, or `LOWER_EVEN_ONLY`.
        point: The code point.

    Returns:
        Its simple lowercase, or the code point itself when it has none.
    """
    var key = Int32(point)
    var at = 0
    var stop = len(lows)
    while at < stop:
        var middle = (at + stop) // 2
        if highs[middle] < key:
            at = middle + 1
        else:
            stop = middle
    if at >= len(lows) or lows[at] > key:
        return point
    var delta = deltas[at]
    if delta == LOWER_EVEN_ONLY:
        return point + 1 if (point & 1) == 0 else point
    return UInt32(key + delta)


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
    the row being too long is its other way out. A lookaround it hands back
    only when the machine next door could have taken the program anyway, since
    that machine pays for a search inside a search what this engine pays and no
    more, and a program that holds nothing else this engine is needed for is
    better off there.

    The traffic runs the other way for a backreference, an atomic group and a
    conditional. Those are the three shapes this engine takes and the machine
    cannot, so a program holding any of them arrives here and is never handed
    back, and a lookaround standing beside one of them is walked here too.
    Documents 95, 99, 100 and 120."""

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

    var lower: List[List[Int32]]
    """The simple lowercase runs, held for exactly that reason and empty unless
    the program reads a backreference under the wide reading of `(?i)`. Three
    lists, the lows, the highs and the deltas. Document 97."""

    var jobs_pc: List[Int32]
    """The stack, as instructions. A negative entry is not an instruction: it is
    the slot `-pc - 1` waiting to be put back, and the number beside it is what
    to put back into it. The one negative entry that is neither is `MARKED`,
    which is where an atomic group began."""

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

    An atomic group does not turn this off. What the group matches from a
    position is the same whichever path arrived there, so the second arrival is
    the same question as the first and is still worth nothing. Document 99
    section 5."""

    var alone: Bool
    """Whether there is no engine underneath this one for the program in hand.

    True for a backreference, which the machine cannot answer, and for an atomic
    group, which the machine cannot obey. The two reasons have nothing in common
    and the consequence is the same one: the row that is too long for the bitmap
    cannot be handed anywhere, so it is run here without a bitmap and the step
    count is the whole of the bound. Documents 95 and 99."""

    var bitmap: Bool
    """Whether the bitmap is in use for the row in hand at all.

    Set per row rather than per program, since what it answers is whether the
    row fits. False only when `alone` is true and the row did not fit, because
    every other program in that position is handed back instead."""

    var stamped: Bool
    """Whether the bitmap is in use under that narrower rule, which is the mode a
    program holding a backreference runs in.

    An instruction and a position do not say everything about what is left to do
    when a backreference can read a slot, but an instruction and a position and
    the slots do. So the bitmap is kept and everything in it is forgotten the
    moment a slot changes value, which leaves only the arrivals that really are
    the same state. Set per row for the same reason `bitmap` is, since a row
    running without a bitmap is not running under a narrower rule about one.
    Document 95 section 3."""

    var counting: Bool
    """Whether the step count is bounding this row.

    Which is whenever the bitmap is not bounding it outright, so either the
    bitmap is being forgotten under `stamped` or there is no bitmap at all. The
    test is lifted out of the loop and into a field for the same reason the
    others are, which is that the ordinary program pays nothing for it."""

    var marks: List[Int32]
    """The cells set since the last slot changed, so that forgetting them is the
    length of that list rather than the length of the bitmap. Empty unless
    `stamped`."""

    var steps: Int
    """How much of that bound this search has spent, counted only when
    `counting` is on since that is the only case where anything but the bitmap
    bounds the walk."""

    var overrun: Bool
    """Whether the last search ran out of steps rather than finding an answer.

    Read by the caller rather than by anything here, because the answer to an
    overrun is an error and this is several layers below the one that raises.
    """

    var nests: Bool
    """Whether the program holds a lookaround, which is a search inside a
    search and is the one thing here that walks from a height other than
    nothing.

    A program holding one keeps its bitmap and turns the stamp on. The bitmap
    says that arriving twice at an instruction and a position is worth nothing
    the second time, which is as true inside a body as it is outside one and is
    false only across two separate askings of the same body: a body that
    matched marked every pair on the way to matching, and the next question
    asked of it at the same position would read those marks as an answer
    already found. So the body forgets its own marks on the way out, using the
    list the stamp already keeps, and everything outside the body goes on being
    bounded by the bitmap rather than by the step count. Document 120."""

    var runs: List[Int32]
    """Which splits are a repeat of one character and where the body of each
    one is, worked out once per program by `run_bodies`. A split with an entry
    here is walked in one step per character instead of three. Everything else
    holds -1, which is every split in a pattern whose repeat is longer than a
    single class."""

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
        self.lower = []
        self.memo = not (program.refs or program.asks)
        self.alone = program.refs or program.cuts or program.asks
        self.bitmap = False
        self.stamped = False
        self.counting = False
        self.marks = []
        self.steps = 0
        self.overrun = False
        self.runs = run_bodies(Span(program.code))
        self.nests = False
        for i in range(len(program.code)):
            var instruction = program.code[i]
            if instruction.op == IN_LOOK or instruction.op == IN_BEHIND:
                self.nests = True
            if instruction.op == IN_REF and instruction.b == REF_WIDE:
                if len(self.lower) == 0:
                    self.lower = lower_table()
                continue
            if instruction.op != IN_AT or len(self.word) > 0:
                continue
            if instruction.a == Int32(Int(AT_BOUNDARY_UNICODE)) or (
                instruction.a == Int32(Int(AT_NON_BOUNDARY_UNICODE))
            ):
                self.word = word_ranges_unicode()
        if self.nests and not self.alone:
            # A lookaround is run here only when this is the engine that has to
            # run the program. The machine next door runs one by starting a
            # second machine with buffers of its own, it pays nothing here that
            # it does not pay there, and a program holding nothing else this
            # engine is needed for is a program it can have. Documents 93, 94
            # and 120.
            self.ok = False

    def _reads_again(
        self,
        points: Span[UInt32, _],
        lead: Int,
        opened: Int,
        at: Int,
        width: Int,
        folding: Int32,
    ) -> Bool:
        """Whether the text at one position is the text a group matched at
        another.

        Three ways of comparing rather than two, because the two flags that
        touch case do not ask for the same comparison. `(?ai)` is the twenty six
        ASCII letters and nothing else. `(?i)` is the simple lowercase of both
        characters, which is not the fold the literals in the same pattern were
        compiled under, and the difference shows in ordinary text rather than in
        a corner. Document 97.

        A pair that is already equal is left alone rather than lowered twice,
        which is the common case and costs nothing to ask.

        Args:
            points: The text, as code points.
            lead: How many unreadable bytes stand in front of them.
            opened: Where the group started.
            at: Where the reference is being read.
            width: How long the group's text is.
            folding: `REF_EXACT`, `REF_NARROW` or `REF_WIDE`.

        Returns:
            True when the two runs are the same run.
        """
        for i in range(width):
            var want = point_at(points, lead, opened + i)
            var have = point_at(points, lead, at + i)
            if want == have:
                continue
            if folding == REF_NARROW:
                if want >= 0x41 and want <= 0x5A:
                    want += 32
                if have >= 0x41 and have <= 0x5A:
                    have += 32
            elif folding == REF_WIDE:
                want = simple_lower(
                    self.lower[0], self.lower[1], self.lower[2], want
                )
                have = simple_lower(
                    self.lower[0], self.lower[1], self.lower[2], have
                )
            if want != have:
                return False
        return True

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
        self._forget_from(0)

    def _forget_from(mut self, base: Int):
        """Clears the cells marked since the list was this long.

        Forgetting a cell is always safe, because the bitmap only ever says
        that a pair is not worth arriving at twice and forgetting it costs the
        walk a second arrival rather than an answer. So the one thing this has
        to get right is forgetting enough, and a list shorter than the height
        it is asked about means a slot changed in the meantime and took the
        whole list with it, so everything left was written after that and all
        of it goes.

        Args:
            base: The height the list stood at.
        """
        var height = base if base <= len(self.marks) else 0
        for i in range(height, len(self.marks)):
            var cell = Int(self.marks[i])
            self.seen[cell >> 6] &= ~(UInt64(1) << UInt64(cell & 63))
        while len(self.marks) > height:
            _ = self.marks.pop()

    def _cut(mut self):
        """Throws away every choice the group that is closing could have made.

        The group's own part of the stack is everything above the nearest
        `MARKED`, which is always the group's own mark: a group nested inside
        this one either reached its own cut, which took its mark off, or failed,
        which popped its mark off, and either way it is gone before this runs.
        So there is nothing to number and no group identifier to carry.

        What is thrown away is the choices and what is kept is the saves, in the
        order they were made. A choice is a way the group could have matched
        instead and is exactly what an atomic group says there is no going back
        to. A save is not a choice, it is what a slot held before the group
        wrote to it, and it is still owed to the pattern outside: the group as a
        whole can still fail, because what follows it can fail, and the groups
        it captured have to go back to what they were when it does. The stack is
        compacted in place rather than copied, since the saves keep their order
        and only move down. Document 99 section 4.
        """
        var mark = len(self.jobs_pc) - 1
        while mark >= 0 and self.jobs_pc[mark] != MARKED:
            mark -= 1
        var write = mark if mark >= 0 else 0
        for i in range(mark + 1, len(self.jobs_pc)):
            if self.jobs_pc[i] < 0:
                self.jobs_pc[write] = self.jobs_pc[i]
                self.jobs_at[write] = self.jobs_at[i]
                write += 1
        while len(self.jobs_pc) > write:
            _ = self.jobs_pc.pop()
            _ = self.jobs_at.pop()

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
        return self._walk(
            program, points, lead, length, start, 0, found, advance
        )

    def _body(
        mut self,
        program: Program,
        entry: Int32,
        points: Span[UInt32, _],
        lead: Int,
        length: Int,
        at: Int,
        mut inner: List[Int32],
    ) -> Bool:
        """Whether the body of a lookaround matches starting exactly here.

        A search inside a search, run on the stack that is already here rather
        than on a second one. The body's own part of the stack is everything
        above the height the stack stood at when it started, which is the same
        way an atomic group already knows which choices are its own, and the
        walk is the same walk because the body is ordinary instructions.

        What comes back out is the groups, in `inner`, and the caller decides
        whether to keep them. Everything else is put back: the stack is unwound
        to the height it stood at and every save above that is paid, so a body
        that matched and a body that failed leave the slots where the path
        outside left them. Upstream keeps what a group inside a positive
        assertion matched and leaves a group inside a negative one unset, and
        both of those are the caller writing back what is in `inner` or not
        writing it.

        Args:
            program: The compiled pattern, whose instructions hold the body.
            entry: Where the body starts.
            points: The text, as code points.
            lead: How many unreadable bytes stand in front of them.
            length: One past the last position, counting those bytes.
            at: Where the body has to start matching.
            inner: Filled with the slots as they stood at the end of the body,
                when it matched.

        Returns:
            True when the body matches.
        """
        var base = len(self.jobs_pc)
        var marked = len(self.marks)
        self._push(entry, Int32(at))
        var end = self._walk(
            program, points, lead, length, at, base, inner, False
        )
        if self.stamped:
            self._forget_from(marked)
        while len(self.jobs_pc) > base:
            var pc = self.jobs_pc.pop()
            var value = self.jobs_at.pop()
            if pc < 0 and pc != MARKED:
                self._write(Int(-pc - 1), value)
        return end != NO_MATCH

    def _walk(
        mut self,
        program: Program,
        points: Span[UInt32, _],
        lead: Int,
        length: Int,
        start: Int,
        base: Int,
        mut found: List[Int32],
        advance: Bool,
    ) -> Int:
        """Follows every path on the stack above one height until one matches.

        The body of `_attempt`, taken out so that the body of a lookaround can
        be walked by the same code from a different height. Nothing in here
        reads the height except the loop that stops at it.

        Args:
            program: The compiled pattern.
            points: The text, as code points.
            lead: How many unreadable bytes stand in front of them.
            length: One past the last position, counting those bytes.
            start: Where this walk begins, which the refusal below reads.
            base: How high the stack stood before this walk pushed anything.
            found: Filled with the slots of the match, when there is one.
            advance: Whether a match of no width is refused at `start`.

        Returns:
            Where the match ends, or `NO_MATCH`.
        """
        while len(self.jobs_pc) > base:
            var pc = self.jobs_pc.pop()
            var at = self.jobs_at.pop()
            if pc < 0:
                if pc == MARKED:
                    # An atomic group being left backwards, which is the group
                    # failing. Everything it could have tried was above this and
                    # has already been popped, so the mark is dropped and the
                    # pattern goes on failing past it.
                    continue
                # The way out of a save. Everything the save protected has been
                # walked, so the slot goes back to what it held before it.
                self._write(Int(-pc - 1), at)
                continue
            if self.bitmap:
                var cell = Int(pc) * (length + 1) + Int(at)
                var word_at = cell >> 6
                var bit = UInt64(1) << UInt64(cell & 63)
                if (self.seen[word_at] & bit) != 0:
                    continue
                self.seen[word_at] |= bit
                if self.stamped:
                    self.marks.append(Int32(cell))
            if self.counting:
                # The bitmap speaks for less here, or for nothing at all, so the
                # count is what really bounds this. The step is charged at the
                # pop rather than at the push, because what is being bounded is
                # the walking and a push that is never popped costs nothing.
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
                var body = self.runs[Int(pc)]
                if body < 0:
                    # The second arm first, so that the first arm comes off the
                    # stack first and the path the pattern prefers is the path
                    # that is followed.
                    self._push(instruction.b, at)
                    self._push(instruction.a, at)
                elif instruction.a == body:
                    # A repeat of one character, greedy, walked in one step
                    # rather than three. The arm that leaves the repeat is put
                    # on the stack the same way it always was, and the arm that
                    # goes round reads its character here and comes straight
                    # back to this instruction one position along, so the body
                    # and the jump behind it are never visited. Reading it here
                    # is the same test the body would have done and it is done
                    # once, so the one visit per instruction per position the
                    # bitmap gives is untouched.
                    self._push(instruction.b, at)
                    if Int(at) < length and accepts(
                        program.code[Int(body)],
                        program.ranges,
                        point_at(points, lead, Int(at)),
                    ):
                        self._push(pc, at + 1)
                else:
                    # The same repeat written lazily, so the order is the other
                    # way round: the arm that leaves goes on last and comes off
                    # first, which is what the split itself did.
                    if Int(at) < length and accepts(
                        program.code[Int(body)],
                        program.ranges,
                        point_at(points, lead, Int(at)),
                    ):
                        self._push(pc, at + 1)
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
            elif instruction.op == IN_MARK:
                # Where the stack stood when the group opened, so that the cut
                # at the other end of it knows how much of the stack is the
                # group's own. The position beside it is never read and is put
                # there so that the two lists stay the same length.
                self._push(MARKED, at)
                self._push(pc + 1, at)
            elif instruction.op == IN_CUT:
                self._cut()
                self._push(pc + 1, at)
            elif instruction.op == IN_TEST:
                # One arm or the other and nothing pushed, because this is a
                # question with an answer rather than a choice with two ways
                # out. A group that never took part is a slot pair still at
                # minus one, which is the same reading the reference below
                # gives it, so `(a)?(?(1)b|c)` against `c` takes the second arm.
                var slot = Int(instruction.a)
                var took = self.slots[slot] >= 0 and self.slots[slot + 1] >= 0
                self._push(pc + 1 if took else instruction.b, at)
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
                    if Int(at) + width <= length and self._reads_again(
                        points,
                        lead,
                        opened,
                        Int(at),
                        width,
                        instruction.b,
                    ):
                        self._push(pc + 1, at + Int32(width))
            elif instruction.op == IN_LOOK or instruction.op == IN_BEHIND:
                # Where the body has to start, which is the whole of the
                # difference between the two directions: a lookahead starts it
                # where the path is standing and a lookbehind that many
                # characters further back, which lands the end of the body on
                # the path because the compiler has refused every body whose
                # width is not always the same number.
                var behind = instruction.op == IN_BEHIND
                var width = Int(instruction.b) >> 1 if behind else 0
                var want = (
                    (instruction.b & 1) == 1 if behind else instruction.b == 1
                )
                var into = Int(at) - width
                var inner = List[Int32]()
                var got = into >= 0 and self._body(
                    program, instruction.a, points, lead, length, into, inner
                )
                if self.overrun:
                    return NO_MATCH
                if got == want:
                    if got:
                        # The groups the body matched, kept the way upstream
                        # keeps them, and owed back to the path outside the
                        # same way a save is owed back. A body that failed
                        # wrote nothing, and a body that matched under a
                        # negative assertion is a body the pattern threw away,
                        # so neither of those is here.
                        for k in range(self.nslots):
                            if self.slots[k] != inner[k]:
                                self._push(Int32(-k - 1), self.slots[k])
                                self._write(k, inner[k])
                    self._push(pc + 1, at)
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
            # holding a backreference, an atomic group or a conditional is one
            # the machine cannot be handed at all. So that one runs the row
            # without a bitmap and the step count is the whole of the bound.
            # Document 95 section 3 and document 99 section 5.
            if not self.alone:
                return GAVE_UP
            self.bitmap = False
            self.stamped = False
        else:
            var words = (cells + 63) // 64
            if len(self.seen) < words:
                self.seen = List[UInt64](length=words, fill=0)
            else:
                for i in range(words):
                    self.seen[i] = 0
            self.bitmap = True
            self.stamped = not self.memo
        if self.nests and self.bitmap:
            # For the reason on the field, which is that a body that matched
            # marked every pair on the way, and the next question asked of the
            # same body at the same position would read those marks as an
            # answer already found. The stamp is the list of what has been
            # marked and it is already here for the program that changes a
            # slot, so a body forgets its own marks on the way out and the
            # bitmap goes on bounding everything outside it.
            self.stamped = True
        self.counting = not self.bitmap or self.stamped
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
            backreference, an atomic group or a conditional can do.
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
    program holding a backreference, an atomic group or a conditional is one
    the machine cannot read at all. That machine is written on threads that
    merge, this engine is the one that keeps a path, a backreference and a
    conditional are questions about the path and a cut is an answer only a path
    can give. So the choosing has to live on this side of the two, since this is
    the side that can see both. Documents 95, 99 and 100.

    The one shot form, which builds both sets of buffers, uses them once and
    drops them. A caller with a column to walk wants to keep them instead.

    Args:
        program: The compiled pattern.
        text: The text.

    Returns:
        True when some part of it matches.

    Raises:
        Error: If the row ran out of steps, which only a pattern holding a
            backreference, an atomic group or a conditional can do.
    """
    if not program.refs and not program.cuts and not program.asks:
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
            backreference, an atomic group or a conditional can do and which
            this door never sees, since all three of those are refused on the
            engine that comes through here.
    """
    var end = bounded.search(program, points, lead, 0, found)
    if bounded.overrun:
        raise Error(_TOO_LONG)
    if end != GAVE_UP:
        return end
    return machine.find(program, points, lead)
