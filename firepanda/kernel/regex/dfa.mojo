"""Remembering what the machine did, so a character costs one table read.

The machine next door holds a set of instructions and walks the whole set every
time it reads a character. The set it holds is worked out afresh at every
position of every row, and a column of a million rows over a program of forty
instructions asks the same question about the same set a very large number of
times. This is the cache that answers it once.

A state is one of those sets. A transition is a state and one character giving
the next state. There are far too many states in a pattern of any size to build
them all up front, so they are built the first time they are reached and kept,
which is what lazy means here and is the RE2 design.

Two things make that work at all.

The first is the alphabet. The machine runs over code points, so a row of
transitions keyed on the character would be a million numbers wide and nobody
can hold one of those per state. The program carries a class table instead, one
number per set of characters the program cannot tell apart, and a row of
transitions is as wide as that. `abc` has four classes, so a state is four
numbers. Issue #866 has why the alphabet is over code points rather than over
bytes, and #871 wrote the table.

The second is the bound. A pattern like `[ab]*a[ab]{9}` needs five hundred odd
states and there are patterns that need far more, so the cache stops at
`MAX_STATES` and says so. A caller that gets that answer runs the machine
instead and gets the same answer more slowly, which is the only fall back there
is and is the reason none of this can be the only engine.

What is left out is the harder half of the story. This answers whether a row
matched and nothing else: not where the match was, and certainly not where each
group was, because a set of instructions is not a path through the program and
the groups are a property of the path. So `replace` and `extract` are not
callers of this and never will be. Issue #863 has the order the pieces go in and
what carries the capturing scans.

Three shapes of pattern are refused outright rather than run slowly.

A program with no class table is refused, because there is no alphabet to lay
the transitions out on. Nothing is compiled with one unless a caller asks, so
the caller that wants the cache is the caller that passes `alphabet=True`.

A program holding an assertion that reads the text around a position is refused:
both word boundaries, the multiline `^` and `$`, and Python's `$` that also
matches in front of a newline ending the row. All four ask about the character
on the other side of the position, and a state here knows where it is only in
the two ways a state can know it, which is that it is the first position or the
last one. RE2 answers those by folding the neighbouring character into the state
and it is a real design, just a larger one than this box. `^`, `\\A`, `\\z` and
RE2's `$` are all answered, since the first two are the start and the last two
are the end.

A program holding a lookahead is refused for a plainer reason. That instruction
runs a second machine over the rest of the row before deciding whether a thread
goes on, so the answer at a position depends on text this box has not read and
will not read again, and there is nothing to fold into a state. Document 93.

The end of the row is a flag on a state rather than a column in the transition
table. A state is built twice, once asking what it reaches with the end of the
row asserted and once without, and the first answer is kept as a bool. That
costs a second walk per state, which happens once per state ever rather than
once per row, and it keeps the table one column narrower.

An unanchored pattern gets instruction zero folded into every state it builds,
which is the usual way of saying that an attempt may begin at any position. It
means a state here is the set of places every live attempt could be in at once,
all of them together, which is exactly why the answer cannot say where a match
began.
"""

from std.collections.span import Span

from firepanda.kernel.regex.pike import accepts
from firepanda.kernel.regex.program import (
    IN_AT,
    IN_BEHIND,
    IN_JUMP,
    IN_LOOK,
    IN_MATCH,
    IN_SAVE,
    IN_SPLIT,
    Program,
    class_of,
)
from firepanda.kernel.regex.tokens import (
    AT_BEGINNING,
    AT_BEGINNING_STRING,
    AT_END,
    AT_END_STRING,
)


comptime MAX_STATES: Int = 256
"""How many states the cache keeps before it gives up on a program.

Two hundred and fifty six rows of transitions is a few tens of kilobytes for
the alphabets these patterns have, which is small enough to build per column and
throw away. The patterns that need more than this are the ones whose states are
the subsets of something, and those do not stop at a slightly larger number
either, so a bound that is generous rather than tight buys very little.
"""


comptime NO_STATE: Int32 = -1
"""A transition that has not been worked out yet, and what the cache answers
when it has no room left to work one out."""


comptime SCAN_NO: Int = 0
"""The row does not match."""


comptime SCAN_YES: Int = 1
"""The row matches."""


comptime SCAN_GAVE_UP: Int = -1
"""The cache cannot answer for this program or has no room left, and the caller
is to run the machine instead. It is not an error and it is not an answer."""


comptime NO_SAMPLE: UInt32 = 0xFFFFFFFF
"""Stands in the sample table for a class no character has been found for yet,
and is not a code point."""


def _stands(which: Int32, at_begin: Bool, at_end: Bool) -> Bool:
    """Whether an assertion holds, for the four the cache takes.

    The refusal in the constructor is what makes this short. `^` and `\\A` ask
    whether this is the first position and `\\z` and RE2's `$` ask whether it is
    the last one, and every other assertion in the instruction set wants a
    character this function has not got.

    Args:
        which: The `AT_` value.
        at_begin: Whether the position is the start of the row.
        at_end: Whether the position is the end of the row.

    Returns:
        True when the assertion is satisfied.
    """
    if which == Int32(Int(AT_BEGINNING)) or which == Int32(
        Int(AT_BEGINNING_STRING)
    ):
        return at_begin
    return at_end


def _key(kernel: Span[Int32, _], hits: Bool, hits_at_end: Bool) -> UInt64:
    """A number that stands for a state, so two states are compared cheaply.

    Args:
        kernel: The instructions the state holds, sorted.
        hits: Whether the state is a match.
        hits_at_end: Whether the state is a match at the end of the row.

    Returns:
        The hash. Two states with the same one still have to be compared
        properly, and two with different ones are different.
    """
    var value: UInt64 = 0xCBF29CE484222325
    for i in range(len(kernel)):
        value = (value ^ UInt64(Int(kernel[i]) + 1)) * 0x100000001B3
    if hits:
        value = (value ^ 1) * 0x100000001B3
    if hits_at_end:
        value = (value ^ 2) * 0x100000001B3
    return value


struct Cache(Movable):
    """The states of one program, built as they are reached.

    One of these belongs to one compiled program and is worth keeping for as
    long as that program is being run, which is a column rather than a row. The
    first rows of a column pay for the states and the rest of them read the
    table, so a cache that is thrown away per row is a cache that costs more
    than it saves.

    It holds no text and nothing about a row, so the same cache answers every
    row of the column and the order the rows come in does not matter.
    """

    var ok: Bool
    """Whether the cache will run this program at all. False means the caller
    runs the machine for every row and never asks again."""

    var problem: String
    """Why not, in the words a reader would want, and empty when `ok`."""

    var full: Bool
    """Whether the cache has run out of room. Once this is set the answers for
    some rows are `SCAN_GAVE_UP` and the caller runs the machine for those, and
    the rows that were answered before it was set were answered correctly."""

    var anchored: Bool
    """Whether the program may only begin a match at the start of the row, which
    is the one thing about the program the scan keeps a copy of."""

    var classes: Int
    """How wide a row of transitions is, which is the program's alphabet."""

    var samples: List[UInt32]
    """One character per class, which is what a transition is worked out with.

    Any character of a class does, since a class is a set of characters the
    program cannot tell apart, and that is the whole reason the alphabet was
    worked out.
    """

    var kernels: List[Int32]
    """The instructions of every state, end to end."""

    var kernel_at: List[Int32]
    """Where each state's instructions start in `kernels`."""

    var kernel_count: List[Int32]
    """How many instructions each state holds."""

    var keys: List[UInt64]
    """The hash of each state."""

    var hits: List[Bool]
    """Whether reaching each state is a match."""

    var hits_at_end: List[Bool]
    """Whether reaching each state at the end of the row is a match."""

    var next: List[Int32]
    """The transitions, `classes` of them per state, `NO_STATE` until the first
    time a character of that class arrives in that state."""

    var start: Int32
    """The state a row begins in, which is the only one built knowing it is at
    the beginning."""

    var mark: List[Int32]
    """One entry per instruction, holding the walk that last reached it, which
    is how a walk adds an instruction at most once without clearing anything.

    A round number rather than a position, because the walks here happen once
    per state rather than once per position, and there are at most two per state
    and at most `MAX_STATES` states.
    """

    var round: Int32
    """How many walks have been made, which is what `mark` is stamped with."""

    var stack: List[Int32]
    """The walk's own stack, kept rather than allocated per walk."""

    var seed: List[Int32]
    """The instructions a state is to be built from, before the walk."""

    var built: List[Int32]
    """What the last walk arrived at, which is the state being looked up."""

    def __init__(out self, program: Program):
        """Works out whether the program can be cached, and builds the first
        state if it can.

        Args:
            program: The compiled pattern, which has to have been compiled with
                `alphabet=True` for any of this to be possible.
        """
        self.ok = True
        self.problem = String("")
        self.full = False
        self.anchored = program.anchored
        self.classes = 0
        self.samples = []
        self.kernels = []
        self.kernel_at = []
        self.kernel_count = []
        self.keys = []
        self.hits = []
        self.hits_at_end = []
        self.next = []
        self.start = NO_STATE
        self.mark = []
        self.round = 0
        self.stack = []
        self.seed = []
        self.built = []
        if not program.ok:
            self.ok = False
            self.problem = String("the pattern did not compile")
            return
        if program.class_count == 0:
            self.ok = False
            self.problem = String(
                "the program was compiled without an alphabet"
            )
            return
        for i in range(len(program.code)):
            var instruction = program.code[i]
            if instruction.op == IN_LOOK or instruction.op == IN_BEHIND:
                self.ok = False
                self.problem = String("the pattern asks about the text around")
                return
            if instruction.op != IN_AT:
                continue
            if (
                instruction.a == Int32(Int(AT_BEGINNING))
                or instruction.a == Int32(Int(AT_BEGINNING_STRING))
                or instruction.a == Int32(Int(AT_END))
                or instruction.a == Int32(Int(AT_END_STRING))
            ):
                continue
            self.ok = False
            self.problem = String(
                "the pattern asks about the text around a position"
            )
            return
        self.classes = Int(program.class_count)
        self.samples = List[UInt32](length=self.classes, fill=NO_SAMPLE)
        for point in range(128):
            var which = Int(program.class_ascii[point])
            if self.samples[which] == NO_SAMPLE:
                self.samples[which] = UInt32(point)
        for i in range(len(program.class_above) // 2):
            var which = Int(program.class_above[i * 2 + 1])
            if self.samples[which] == NO_SAMPLE:
                self.samples[which] = UInt32(Int(program.class_above[i * 2]))
        for i in range(self.classes):
            if self.samples[i] == NO_SAMPLE:
                # A class the table names and no character is in, which the
                # compiler does not produce and which would silently answer for
                # the wrong characters if it ever did.
                self.ok = False
                self.problem = String(
                    "the alphabet has a class with nothing in"
                )
                return
        self.mark = List[Int32](length=program.sized(), fill=0)
        self.seed.append(0)
        self.start = self._enter(program, True)
        if self.start < 0:
            self.ok = False
            self.problem = String("there was no room for the first state")

    def states(self) -> Int:
        """How many states have been built so far.

        Returns:
            The count, which grows as rows are scanned and stops at
            `MAX_STATES`.
        """
        return len(self.keys)

    def _close(
        mut self, program: Program, at_begin: Bool, at_end: Bool
    ) -> Bool:
        """Walks from `seed` to every instruction reachable without reading a
        character, and leaves the ones that read a character in `built`.

        The machine's walk of the same shape is a recursion in the order the
        pattern prefers its arms, because the machine has to say where a match
        ended and that order is the answer. This one says whether and not where,
        so the order does not matter and a stack is cheaper than a recursion.

        A save is walked through rather than acted on, for the same reason: the
        slots it writes are a property of one path through the program and this
        walk is not following one path.

        Args:
            program: The compiled pattern.
            at_begin: Whether the position is the start of the row.
            at_end: Whether the position is the end of the row.

        Returns:
            True when a match instruction was reached, which is this state
            saying that the row has matched by here.
        """
        self.built.clear()
        self.round += 1
        var round = self.round
        self.stack.clear()
        for i in range(len(self.seed)):
            self.stack.append(self.seed[i])
        var hit = False
        while len(self.stack) > 0:
            var pc = self.stack.pop()
            if self.mark[Int(pc)] == round:
                continue
            self.mark[Int(pc)] = round
            var instruction = program.code[Int(pc)]
            if instruction.op == IN_JUMP:
                self.stack.append(instruction.a)
            elif instruction.op == IN_SPLIT:
                self.stack.append(instruction.a)
                self.stack.append(instruction.b)
            elif instruction.op == IN_SAVE:
                self.stack.append(pc + 1)
            elif instruction.op == IN_AT:
                if _stands(instruction.a, at_begin, at_end):
                    self.stack.append(pc + 1)
            elif instruction.op == IN_MATCH:
                hit = True
            else:
                self.built.append(pc)
        sort(self.built)
        return hit

    def _settle(mut self, hits: Bool, hits_at_end: Bool) -> Int32:
        """Finds the state `built` describes, or adds it.

        Two states that hold the same instructions and answer the same at the
        end of the row are the same state, however they were arrived at, which
        is what keeps a pattern's state count down to the sets it can really be
        in.

        Args:
            hits: Whether reaching the state is a match.
            hits_at_end: Whether reaching it at the end of the row is a match.

        Returns:
            The state, or `NO_STATE` when the cache is full.
        """
        var key = _key(Span(self.built), hits, hits_at_end)
        for which in range(len(self.keys)):
            if self.keys[which] != key:
                continue
            if Int(self.kernel_count[which]) != len(self.built):
                continue
            if self.hits[which] != hits or self.hits_at_end[which] != (
                hits_at_end
            ):
                continue
            var base = Int(self.kernel_at[which])
            var same = True
            for i in range(len(self.built)):
                if self.kernels[base + i] != self.built[i]:
                    same = False
                    break
            if same:
                return Int32(which)
        if len(self.keys) >= MAX_STATES:
            self.full = True
            return NO_STATE
        var made = Int32(len(self.keys))
        self.kernel_at.append(Int32(len(self.kernels)))
        self.kernel_count.append(Int32(len(self.built)))
        for i in range(len(self.built)):
            self.kernels.append(self.built[i])
        self.keys.append(key)
        self.hits.append(hits)
        self.hits_at_end.append(hits_at_end)
        for _ in range(self.classes):
            self.next.append(NO_STATE)
        return made

    def _enter(mut self, program: Program, at_begin: Bool) -> Int32:
        """Builds the state `seed` leads to, both ways round.

        The end of the row is asked first because the walk that answers it is
        the one whose instructions are thrown away, and the walk that answers
        the ordinary case is the one whose instructions are the state.

        Args:
            program: The compiled pattern.
            at_begin: Whether the position is the start of the row.

        Returns:
            The state, or `NO_STATE` when the cache is full.
        """
        var hits_at_end = self._close(program, at_begin, True)
        var hits = self._close(program, at_begin, False)
        return self._settle(hits, hits_at_end)

    def _step(mut self, program: Program, state: Int32, which: Int) -> Int32:
        """Works out what a state does when it reads a character of a class.

        Args:
            program: The compiled pattern.
            state: The state being left.
            which: The class of the character read.

        Returns:
            The state arrived at, or `NO_STATE` when the cache is full.
        """
        var point = self.samples[which]
        var base = Int(self.kernel_at[Int(state)])
        var count = Int(self.kernel_count[Int(state)])
        self.seed.clear()
        for i in range(count):
            var pc = self.kernels[base + i]
            if accepts(program.code[Int(pc)], program.ranges, point):
                self.seed.append(pc + 1)
        if not self.anchored:
            # An attempt may begin here, so every state carries the whole of the
            # program's start along with whatever was already running. It is the
            # `.*?` in front of the pattern, said once rather than compiled in.
            self.seed.append(0)
        return self._enter(program, False)

    def scan(mut self, program: Program, points: Span[UInt32, _]) -> Int:
        """Whether a row matches, or that the caller has to run the machine.

        The text is the whole row and it starts where the row starts, which is
        the one thing this cannot be told otherwise: a state knows it is at the
        beginning because the scan started there, so a caller that cut a row and
        handed over what was left would be asking `^` the wrong question. The
        machine takes a lead for that and this does not.

        Args:
            program: The compiled pattern, which has to be the one the cache was
                built for.
            points: The row, as code points.

        Returns:
            `SCAN_YES`, `SCAN_NO`, or `SCAN_GAVE_UP` when the cache has no room
            left to answer.
        """
        if not self.ok:
            return SCAN_GAVE_UP
        var state = self.start
        var length = len(points)
        var at = 0
        while at <= length:
            var here = Int(state)
            if self.hits[here]:
                return SCAN_YES
            if at == length:
                return SCAN_YES if self.hits_at_end[here] else SCAN_NO
            if self.anchored and self.kernel_count[here] == 0:
                # The one attempt this program is allowed has died, so the rest
                # of the row is not worth reading. An unanchored program never
                # gets here, since instruction zero is in every state it builds.
                return SCAN_NO
            var which = Int(class_of(program, points[at]))
            var slot = here * self.classes + which
            var to = self.next[slot]
            if to == NO_STATE:
                to = self._step(program, state, which)
                if to == NO_STATE:
                    return SCAN_GAVE_UP
                self.next[slot] = to
            state = to
            at += 1
        return SCAN_NO
