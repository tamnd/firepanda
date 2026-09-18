"""A parsed pattern turned into instructions something can run.

The shape is Thompson's, which is to say the program is a list of instructions
with two branching ones and no backtracking anywhere, and the reason for that
choice is the same reason RE2 made it: a pattern is written by a caller and run
against a column, and an engine that can take exponential time on a pattern like
`(a+)+b` is a denial of service with a friendly API in front of it.

Both engines are compiled here and the difference between them is four things.
Python reads the three Perl classes as Unicode where RE2 reads them as ASCII,
Python asks the word boundary question against its own wider class, Python's
dollar sign matches before a newline that ends the text where RE2's does not,
and Python folds the dotted and dotless Turkish I onto the plain one under
`(?i)` where RE2 leaves both of them alone. All four are settled while the
pattern is being compiled, so the machine that runs the program never learns
which engine asked for it. Document 81 is where the first three were measured
and says which methods take which engine, and document 83 is where the fourth
was.

Three of those four are questions about which alphabet rather than about which
engine, and `(?a)` is a caller answering that question for themselves on the
engine that reads it. A Python program compiled under that letter takes the
ASCII classes, folds only the twenty six letters, and asks the word boundary
question against the ASCII class, which puts it beside RE2 on two of the three
and not on the third: Python's ASCII `\\s` holds a vertical tab and RE2's never
did. Document 88 is where that was measured.

What is still RE2 only is most of the constructs. Python has a lookaround, a
backreference, a conditional, an atomic group and a possessive quantifier, and of
those this engine now has the lookahead half of the first one. The rest are
refused for Python as a gap here rather than as something Python cannot do.

The refusals are worth reading as a group, because they are not a list of things
that were too hard. Every one of them is a construct RE2 itself refuses, which
was measured rather than assumed: a lookaround, a backreference, a conditional,
an atomic group, a possessive quantifier, and the four inline flag letters out of
seven that RE2 has never heard of. A pattern this compiler turns down for RE2 is
a pattern pyarrow turns down, and that correspondence is the thing
`tests/differential/regex_match.mojo` checks over the generated corpus. What
changed with the lookahead is that the two engines now refuse different lists,
which they always did for the flags and now do for a construct as well, and the
differential that reads the Python side is a different one.
"""

from std.collections.span import Span

from firepanda.kernel.regex.classdata import (
    DIGIT_RANGES,
    SPACE_RANGES,
    WORD_RANGES,
)
from firepanda.kernel.regex.folddata import (
    FOLD_DELTA,
    FOLD_EVEN_ODD,
    FOLD_HIGH,
    FOLD_LOW,
)
from firepanda.kernel.regex.parse import TYPE_FLAGS, Parsed
from firepanda.kernel.regex.route import ENGINE_PYTHON, ENGINE_RE2
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
    CATEGORY_DIGIT,
    CATEGORY_NOT_DIGIT,
    CATEGORY_NOT_SPACE,
    CATEGORY_NOT_WORD,
    CATEGORY_SPACE,
    CATEGORY_WORD,
    FLAG_ASCII,
    FLAG_DOTALL,
    FLAG_IGNORECASE,
    FLAG_LOCALE,
    FLAG_MULTILINE,
    FLAG_UNICODE,
    FLAG_VERBOSE,
    MAXREPEAT,
    Node,
    OP_ANY,
    OP_AT,
    OP_ATOMIC_GROUP,
    OP_ASSERT,
    OP_ASSERT_NOT,
    OP_BRANCH,
    OP_CATEGORY,
    OP_FAILURE,
    OP_GROUPREF,
    OP_GROUPREF_EXISTS,
    OP_IN,
    OP_LITERAL,
    OP_MAX_REPEAT,
    OP_MIN_REPEAT,
    OP_NEGATE,
    OP_NOT_LITERAL,
    OP_POSSESSIVE_REPEAT,
    OP_RANGE,
    OP_SCOPE,
    OP_SEQ,
    OP_SUBPATTERN,
)


comptime IN_CHAR: UInt8 = 1
"""Match one code point, which is `a`. The commonest instruction there is, kept
separate from a one element set so that the commonest case is a comparison
rather than a search."""

comptime IN_SET: UInt8 = 2
"""Match a code point in a set. `a` is where the set starts in `ranges` and `b`
is how many ranges it has, sorted and with no two of them touching, so
membership is a binary search."""

comptime IN_NOT_SET: UInt8 = 3
"""Match a code point outside a set, read the same way. A separate instruction
rather than a flag because a negated class is common enough that storing the
complement would double the size of the range table for no gain."""

comptime IN_ANY: UInt8 = 4
"""Match anything but a newline, which is what a full stop is."""

comptime IN_ANY_ALL: UInt8 = 5
"""Match anything at all, which is what a full stop is under `(?s)`."""

comptime IN_SPLIT: UInt8 = 6
"""Go to `a` and to `b`, in that order. The whole of the branching in a Thompson
program is this one instruction, and the order matters for a capture even though
it does not for an answer of yes or no."""

comptime IN_JUMP: UInt8 = 7
"""Go to `a`."""

comptime IN_AT: UInt8 = 8
"""Check a position rather than a character. `a` is the `AT_` value, and this is
where the two engines disagree about `$`."""

comptime IN_MATCH: UInt8 = 9
"""The pattern has matched."""

comptime IN_SAVE: UInt8 = 10
"""Write the position into slot `a` of the thread running this.

The one instruction here that a yes or no answer has no use for, and it is only
written into a program whose caller asked for captures. Slot zero and slot one
are the two ends of the whole match, slot `2k` and slot `2k + 1` are the two ends
of group `k`, and a group inside a repeat writes its slots again every time round,
so what a group holds at the end is what it held on the last pass, which is what
both engines say it holds.

A program carrying these is larger and slower to run than the same pattern
without them, which is why `contains` and the two anchored questions and `count`
are compiled without and only `replace` is compiled with. The cost is not the
instruction, it is that every thread then carries a copy of the slots.
"""

comptime IN_LOOK: UInt8 = 11
"""Ask whether another program matches here, without reading anything.

`a` is where that other program starts and `b` is 1 for `(?=...)` and 0 for
`(?!...)`. It is a position test like `IN_AT` is, and the machine treats it as
one: the thread goes on to the next instruction when the answer agrees with `b`
and dies when it does not, and either way the position does not move.

The body is written into the same instruction list as everything else, ending in
its own `IN_MATCH`, with a jump written over it so that nothing walks into it
from in front. Nothing reaches those instructions except through this one.

This is the only instruction here whose cost is more than a comparison. A
thread that arrives at one runs a whole second machine over the rest of the row,
so a row of a thousand characters against a pattern holding a lookahead is a
thousand runs of the body in the worst case. That is polynomial rather than
exponential, which is the property the whole file is built to keep, and it is
why RE2 refuses the construct outright rather than paying for it. This engine
pays for it because pandas answers these patterns with `re` and a refusal here
is a column a caller does not get.

Document 93.
"""


comptime MAX_INSTRUCTIONS: Int = 200000
"""How large a program may get before the compiler gives up.

A counted repeat is compiled by copying its body, which is what a Thompson
program has instead of a counter, so `(?:abcde){1000}` is five thousand
instructions. `MAX_REPEAT_COUNT` is what really bounds this, since it caps the
number of copies at a thousand and leaves only the length of the body free, and
this is the second line rather than the first. Python has no such ceiling and
RE2 has one measured in bytes, so a pattern reaching this is refused here and
run by both of the others, which is why it is set high enough that nothing
anybody writes on purpose arrives.
"""


comptime MAX_REPEAT_COUNT: Int32 = 1000
"""How many times RE2 will repeat anything, counting the whole way down.

Measured rather than read: `a{1000}` is a pattern and `a{1001}` is an error, and
so is `(a{11}){91}`, which is 1001 copies written as two numbers neither of
which is over the limit. So the count is a budget that divides on the way into
a repeat rather than a check on one number, and a repeat with no ceiling spends
its lower bound, which is why `(a*){1000}` is fine and `(a{1000,}){2}` is not.

Python has no limit at all, which makes this a refusal rather than a gap: a
caller writing `a{5000}` gets an Arrow error out of pandas today, and a
firepanda that answered would be answering where pandas raises.
"""


comptime UNICODE_LAST: Int32 = 0x10FFFF
"""The last code point, which is the top of the range a negated class is
complemented against."""


comptime PYTHON_NEWEST: Int = 14
"""The newest CPython this library has been measured against.

What this library copies on one of its two engines is the `re` module, and `re`
is not the same module in every version of Python this project supports.
`pixi.toml` says 3.12 and up, and two rules this file writes down changed inside
that range, so a program is compiled for a version of Python rather than for
Python.

This is the default for every caller with no interpreter to ask, which is every
test and every differential that is not already standing inside one. The Python
door reads the real number off `sys.version_info` and passes it, because the
answer that matters there is the answer pandas would have given in the process
the call arrived in.

A number like this one fails quietly. When a release moves a rule, nothing
raises: the library keeps answering with the rules it was last told about, and
the wrong answers are ordinary looking columns. So `tools/python_version.py`
compares this against the interpreter it is standing in and fails when it is
behind, which is what turns the next release into a red light on purpose rather
than into one by luck. It runs in CI and again in the accessor suite, because
those two stand in different interpreters. Document 92.
"""


comptime PYTHON_PLAIN_NON_BOUNDARY: Int = 14
"""The first CPython where `\\B` is simply the negation of `\\b`.

Up to 3.13 it fails on an empty row instead, whichever alphabet was asked for,
which is a case written into the engine rather than a consequence of any rule
about word characters, and 3.14 took it out. Every other engine, RE2 included,
has always had the plain reading. Document 90.
"""


comptime PYTHON_ZED_ESCAPE: Int = 14
"""The first CPython that reads `\\z` at all.

Before that it is a `bad escape \\z` wherever it appears, and from 3.14 it is
the end of the string, which is the position `\\Z` already was and still is. So
nothing gained a meaning that was not already sayable and a spelling stopped
being an error, which is the whole of the change and is why this refusal is a
refusal rather than a second reading.

The second rule in one release, after the one above, and the reason the builder
carries a version number rather than a flag per rule. Document 91.
"""


@fieldwise_init
struct Instruction(Copyable, ImplicitlyCopyable, Movable):
    """One instruction, which is an op code and two payloads.

    Everything that is not `IN_SPLIT` or `IN_JUMP` falls through to the next
    instruction, so a program is mostly straight line and the two branching ones
    are the only place an index into the program is written down.
    """

    var op: UInt8
    """Which instruction this is."""

    var a: Int32
    """A code point, a target, a range table offset, or an `AT_` value."""

    var b: Int32
    """The second target of a split, or how many ranges a set has."""


struct Program(Movable):
    """A compiled pattern, or the reason there is not one.

    `ok` is the only field a caller may read without checking anything first,
    and when it is False the program is empty rather than partial, because a
    half compiled program is a thing that runs and gives wrong answers.
    """

    var code: List[Instruction]
    """The instructions. The program starts at zero."""

    var ranges: List[Int32]
    """Every set's ranges, laid end to end as low and high pairs. One list for
    the whole program rather than one per instruction, so a program is two
    allocations however many classes the pattern has."""

    var ok: Bool
    """Whether the pattern compiled."""

    var problem: String
    """Why it did not, and empty when it did."""

    var gap: Bool
    """Whether the refusal is firepanda's rather than RE2's.

    The distinction is the whole of the difference between a feature that is
    missing and a feature that is not supposed to exist. RE2 refuses a
    lookaround, a backreference, a conditional, an atomic group, a possessive
    quantifier and four of the seven inline flag letters, and pandas hands it
    those patterns anyway, so refusing them here is agreement rather than a
    shortfall. Refusing a pattern Python answers, for a reason of this library's
    own, is a shortfall.

    A caller deciding what to do next needs the two told apart, and so does the
    differential, which compares the first kind against pandas' own refusal and
    counts the second kind as held out.
    """

    var slots: Int
    """How many capture slots a thread running this program carries.

    Zero for a program compiled without captures, which is every program asking
    whether or how many, and `2 * (groups + 1)` for one compiled with them. A
    machine reads this to size its thread state, so a program and a machine built
    for it are the same call away from each other and neither has to be told
    twice.
    """

    var groups: Int
    """How many capturing groups the pattern opened.

    Kept beside the slots because the replacement string is checked against it
    rather than against the slot count, and because the number in RE2's own
    refusal is this one.
    """

    var python: Bool
    """Which engine this was built for, with True meaning Python's.

    The instructions are the same either way and this is not read while the
    program runs. What reads it is the scan around the program, because the two
    engines walk a row looking for a second match by different rules and the
    rules belong to whoever compiled the pattern rather than to whoever is
    walking. `firepanda/kernel/regex/pike.mojo` has both loops and
    `firepanda/kernel/regex/replace.mojo` has the other two, and a caller that
    has a program has already been told which pair it wants.

    It also picks the grammar the replacement string is read by, which is the
    one place where a fact about the pattern decides something about an argument
    that is not the pattern. That is upstream's arrangement rather than one made
    here: `re.sub` reads its template Python's way and `replace_substring_regex`
    reads its rewrite RE2's way, and which of the two a call reaches is the same
    routing decision that picked the engine.
    """

    var anchored: Bool
    """Whether the pattern can only match at the start of the text.

    True when the first instruction that is not a save is `^` outside multiline
    mode or `\\A`, which is the one shape where a fresh attempt at any position
    but zero is known to die on its first step. A pattern beginning with an
    alternation compiles to a split first and is left alone even when both
    branches are anchored, because the scans read this as a promise about the
    whole program and a split is where a second promise would have to be
    checked.

    What reads it is the scan rather than the machine. `_run` and `matches` in
    `firepanda/kernel/regex/pike.mojo` start a fresh attempt at every position
    of the row, and for a program with this flag set every one of those after the
    first walks from instruction zero to the anchor and stops, which is a walk
    per character of every row that answers nothing. A row of a hundred
    characters pays it a hundred times, and q28 of ClickBench is a column of
    URLs read by an anchored pattern, so document 80 has the number.
    """

    var first_at: Int32
    """Where in `ranges` the characters a match can begin with are, and zero
    when there is no such set worth having."""

    var first_count: Int32
    """How many ranges that set has, and zero when there is not one.

    A scan that has nothing running and has not matched yet is about to start a
    fresh attempt, and an attempt that begins by reading a character the program
    cannot begin with dies on its first step. So a position holding such a
    character can be stepped over without the walk, which is the same trade the
    `anchored` flag above makes and is the general form of it: that one knows
    every position but zero is hopeless, this one knows which characters are.

    Empty for four kinds of program, each for its own reason. A pattern that
    can match nothing has every position begin a match, so there is nothing to
    skip. A pattern that can begin with anything, which is a leading `(?s).`,
    has the same problem. A pattern whose set holds most of ASCII is left out on
    purpose, because the test costs a binary search per position and only pays
    where it rejects, so a set that accepts nearly everything would be paid for
    and never used. The line is drawn at three quarters of ASCII, which is a
    reading of that trade rather than a measured crossing point. And an anchored
    pattern is left out because it has no position to step over, which keeps its
    scan the scan it was before any of this was written.
    """

    var class_count: Int32
    """How many classes the alphabet was cut into, and zero when there is none.

    A class is a set of characters this program cannot tell apart. Two
    characters are in the same one when every question the program asks about a
    character answers the same for both, so `abc` has four classes: the `a`, the
    `b`, the `c` and everything else, wherever in the code points that everything
    else happens to be.

    What wants it is a machine that remembers what it did. A table of what a
    state does next is as wide as the alphabet, and an alphabet of a million
    code points is not a table anybody can hold, while an alphabet of four is a
    row of four numbers. So this is the axis a lazy DFA's transition table is
    laid out on, and until that exists this is a table nothing reads. Issue #863
    has the order the pieces are being written in and why this one is first.

    Zero for a program compiled without `alphabet`, which is every program
    compiled today, since a table nothing reads is not worth several times the
    compile it costs. Zero as well for a program that did not compile, for one
    that asks more questions than `MAX_CLASS_TESTS` and for one that cuts the
    code points into more pieces than `MAX_CLASS_PIECES`. A reader has to have
    the fall back anyway, since the DFA above it has one for the patterns it
    cannot run.
    """

    var class_ascii: List[Int32]
    """The class of each of the 128 ASCII characters, or empty when there is no
    table.

    A direct index rather than a search, because a column of text is mostly
    these and the whole point of the class table is that reading a character
    stops being a binary search.
    """

    var class_above: List[Int32]
    """The classes above ASCII, as a start and a class for each piece.

    In order and read by a binary search, the way every other range table here
    is read. The first piece starts at 128 and the last one runs to the last
    code point, so every character above ASCII falls in exactly one of them and
    a program whose pattern is all ASCII has one piece.
    """

    var labels: List[String]
    """What each group is called, one entry per group and empty for an unnamed
    one.

    This is here rather than being worked out again from the pattern because
    `extract` labels its columns with them, and working them out again would
    mean parsing the pattern a second time in the one place that has already
    parsed it. It is the length of `groups` whether or not the pattern named
    anything, so a caller walking it does not have to know how many names there
    were.
    """

    def __init__(out self):
        """Starts an empty program, which is what a refusal leaves behind."""
        self.code = []
        self.ranges = []
        self.ok = True
        self.problem = String("")
        self.gap = False
        self.slots = 0
        self.groups = 0
        self.python = False
        self.anchored = False
        self.first_at = 0
        self.first_count = 0
        self.class_count = 0
        self.class_ascii = []
        self.class_above = []
        self.labels = []

    def sized(self) -> Int:
        """How many instructions the program has.

        Returns:
            The count.
        """
        return len(self.code)


struct _Builder(Movable):
    """The compiler's state while it walks the tree.

    Carried as one struct for the same reason the parser's cursor is: every
    function below needs the instructions, the range table and the failure, and
    threading three mutable references through a recursive walk is how one of
    them ends up left behind.
    """

    var code: List[Instruction]
    """The instructions so far."""

    var ranges: List[Int32]
    """The range table so far."""

    var failed: Bool
    """Whether the compiler has given up."""

    var problem: String
    """Why, and empty until then."""

    var gap: Bool
    """Whether that reason is a gap here rather than a refusal RE2 also
    makes."""

    var flags: Int32
    """The pattern's global flags, which decide what a full stop and the two
    anchors mean."""

    var captures: Bool
    """Whether the caller wants to know where each group matched.

    A flag rather than two compilers, because the only difference it makes is
    three lines in `_emit_node` and two in `compile_program`, and because a
    second walk over the same tree writing almost the same instructions is a
    thing that drifts.
    """

    var python: Bool
    """Whether this program is being compiled for Python's engine.

    A flag for the same reason `captures` is one, and it earns the comparison
    better than that one does: the two engines differ in three places in this
    file and nowhere else, and all three are decided here rather than while a
    row is being read. The classes become ranges, the word boundary becomes a
    different position code, and the dollar sign becomes a different position
    code. Everything downstream of the compiler is the same machine.
    """

    var narrow: Bool
    """Whether Python's engine was asked for the ASCII alphabet.

    Three of the differences above are questions about which alphabet rather
    than about which engine, and `(?a)` is the caller answering that question
    for themselves. It is kept beside the engine rather than read off the flags
    at each of the three places, because it is only ever true on one engine and
    a field says that once.
    """

    var minor: Int
    """Which CPython this program is being compiled beside.

    A number rather than a flag saying whether one rule is on, because the
    question a reader has here is which interpreter and not which rule, and
    because there is no reason to think `\\B` is the last thing `re` will change
    inside a range of versions this library supports. Only ever read on Python's
    engine, since RE2 has not got a version of Python.
    """

    def __init__(
        out self,
        flags: Int32,
        captures: Bool,
        python: Bool = False,
        minor: Int = PYTHON_NEWEST,
    ):
        """Starts an empty program.

        Args:
            flags: The pattern's global flags.
            captures: Whether to write the save instructions.
            python: Whether the program is for Python's engine.
            minor: Which CPython the program is being compiled beside.
        """
        self.code = []
        self.ranges = []
        self.failed = False
        self.problem = String("")
        self.gap = False
        self.flags = flags
        self.captures = captures
        self.python = python
        self.narrow = python and (flags & FLAG_ASCII) != 0
        self.minor = minor

    def give_up(mut self, problem: String, gap: Bool = False):
        """Records the first reason the pattern cannot be compiled.

        The first rather than the last, because compiling carries on after a
        refusal in the places where stopping would mean unwinding a recursion,
        and the reason a caller wants is the one nearest what they wrote.

        Args:
            problem: The reason.
            gap: Whether the reason is a gap here rather than something RE2
                refuses as well.
        """
        if self.failed:
            return
        self.failed = True
        self.problem = problem.copy()
        self.gap = gap

    def emit(mut self, op: UInt8, a: Int32, b: Int32) -> Int32:
        """Appends an instruction.

        Args:
            op: Which instruction.
            a: The first payload.
            b: The second.

        Returns:
            Where it landed, so that a forward jump to it can be patched later.
        """
        if len(self.code) >= MAX_INSTRUCTIONS:
            self.give_up(String("pattern is too large to compile"), True)
            self.code.append(Instruction(IN_MATCH, 0, 0))
            return Int32(len(self.code) - 1)
        self.code.append(Instruction(op, a, b))
        return Int32(len(self.code) - 1)

    def here(self) -> Int32:
        """Where the next instruction will land.

        Returns:
            The index.
        """
        return Int32(len(self.code))

    def patch_a(mut self, at: Int32, target: Int32):
        """Fills in the first target of a jump or a split written earlier.

        Args:
            at: The instruction.
            target: Where it goes.
        """
        self.code[Int(at)].a = target

    def patch_b(mut self, at: Int32, target: Int32):
        """Fills in the second target of a split written earlier.

        Args:
            at: The instruction.
            target: Where it goes.
        """
        self.code[Int(at)].b = target

    def add_set(mut self, ranges: List[Int32], negated: Bool):
        """Writes a set into the range table and emits the instruction reading
        it.

        Args:
            ranges: The ranges, already sorted and merged, as low and high
                pairs.
            negated: Whether the class was negated.
        """
        var at = Int32(len(self.ranges))
        for i in range(len(ranges)):
            self.ranges.append(ranges[i])
        var count = Int32(len(ranges) // 2)
        _ = self.emit(IN_NOT_SET if negated else IN_SET, at, count)


def _sorted_merged(var ranges: List[Int32]) -> List[Int32]:
    """Puts a set's ranges in order and joins the ones that touch.

    Done once at compile time so that membership at run time is a binary search
    over disjoint ranges rather than a walk over whatever the caller wrote.
    Insertion sort because a character class with more than a handful of pieces
    in it is rare and the ones with hundreds come from a table that is already
    in order.

    Args:
        ranges: The ranges as low and high pairs, in any order, consumed.

    Returns:
        The same set, sorted by low end and with no two ranges overlapping or
        adjacent.
    """
    var count = len(ranges) // 2
    for i in range(1, count):
        var low = ranges[i * 2]
        var high = ranges[i * 2 + 1]
        var j = i - 1
        while j >= 0 and ranges[j * 2] > low:
            ranges[(j + 1) * 2] = ranges[j * 2]
            ranges[(j + 1) * 2 + 1] = ranges[j * 2 + 1]
            j -= 1
        ranges[(j + 1) * 2] = low
        ranges[(j + 1) * 2 + 1] = high

    var out = List[Int32]()
    for i in range(count):
        var low = ranges[i * 2]
        var high = ranges[i * 2 + 1]
        if low > high:
            continue
        if len(out) > 0 and low <= out[len(out) - 1] + 1:
            if high > out[len(out) - 1]:
                out[len(out) - 1] = high
            continue
        out.append(low)
        out.append(high)
    return out^


def _held(table: Span[Int32, _]) -> List[Int32]:
    """Copies one of the generated tables out into the shape the compiler uses.

    A copy rather than a span, because the ranges of a class are merged with
    whatever else is inside the same brackets and the result is sorted in
    place, so the caller owns what it gets back.

    Args:
        table: The table from `classdata.mojo`, already materialized.

    Returns:
        The same low and high pairs as a list.
    """
    var out = List[Int32](capacity=len(table))
    for i in range(len(table)):
        out.append(table[i])
    return out^


def word_ranges_unicode() -> List[Int32]:
    """Python's word characters as low and high pairs, ready to be read.

    The table in `classdata.mojo` only exists while the program is being
    compiled, so something has to bring it out into memory, and that costs a
    copy of six kilobytes. This is the place that does it, once, and the caller
    is expected to hold on to what it gets rather than ask again per row.

    Returns:
        The 749 ranges, in order, both ends inside the class.
    """
    var table = materialize[WORD_RANGES]()
    return _held(Span(table))


def _category_ranges(which: Int32, python: Bool, narrow: Bool) -> List[Int32]:
    """What one of the six Perl classes means to whichever engine is running it.

    These are the measured sets rather than the documented ones. RE2's
    documentation writes `\\s` as `[\\t\\n\\f\\r ]` and RE2 agrees with it, which
    is worth saying because Python's `\\s` holds a vertical tab whichever
    alphabet it is asked about and the documentation is the only place the two
    look alike.

    RE2's three are ASCII. That is the difference document 76 opens with, and
    it is the reason a column of Arabic Indic digits answers False to
    `str.contains(r"\\d")` and True once the same pattern picks up a lookahead.
    Python's three are Unicode and come out of `classdata.mojo`, which is what
    document 81 is about: the same letter in the same accessor covers 63
    characters or 138558 of them depending on which method was called.

    Under `(?a)` Python asks about ASCII, which makes three sets here rather
    than two. Two of the three narrow sets are the same as RE2's and the third
    is not: `\\s` keeps the vertical tab, because that character is in Python's
    ASCII class and was never in RE2's. Writing the third one out is the whole
    reason this cannot be a single flag saying which alphabet to use.

    Args:
        which: The `CATEGORY_` value.
        python: Whether the program is being compiled for Python's engine.
        narrow: Whether that engine was asked for ASCII, which is only ever true
            when it is Python's.

    Returns:
        The ranges as low and high pairs, already in order.
    """
    var digits: Int32 = Int32(Int(CATEGORY_DIGIT))
    var not_digits: Int32 = Int32(Int(CATEGORY_NOT_DIGIT))
    var spaces: Int32 = Int32(Int(CATEGORY_SPACE))
    var not_spaces: Int32 = Int32(Int(CATEGORY_NOT_SPACE))
    var wide = python and not narrow
    if which == digits or which == not_digits:
        if wide:
            var table = materialize[DIGIT_RANGES]()
            return _held(Span(table))
        var out: List[Int32] = [0x30, 0x39]
        return out^
    if which == spaces or which == not_spaces:
        if wide:
            var table = materialize[SPACE_RANGES]()
            return _held(Span(table))
        if narrow:
            var out: List[Int32] = [0x09, 0x0D, 0x20, 0x20]
            return out^
        var out: List[Int32] = [0x09, 0x0A, 0x0C, 0x0D, 0x20, 0x20]
        return out^
    if wide:
        return word_ranges_unicode()
    var out: List[Int32] = [0x30, 0x39, 0x41, 0x5A, 0x5F, 0x5F, 0x61, 0x7A]
    return out^


comptime LATIN_I: Int32 = 0x0049
"""The capital I, which both engines fold onto the small one."""

comptime LATIN_SMALL_I: Int32 = 0x0069
"""The small i, which both engines fold onto the capital one."""

comptime LATIN_I_DOTTED: Int32 = 0x0130
"""The Turkish capital I with a dot, which Python folds onto the other three
and RE2 leaves alone."""

comptime LATIN_I_DOTLESS: Int32 = 0x0131
"""The Turkish small dotless i, which Python folds onto the other three and RE2
leaves alone."""


def _re2_unfolds(point: Int32, other: Int32) -> Bool:
    """Whether the table pairs two code points and RE2 does not.

    The whole of the difference between the two engines' folding, measured over
    every code point either of them considers cased. Python reads these four as
    one letter, so `(?i)i` matches all four of them. RE2 reads the two plain
    ones as one letter and the two Turkish ones as a letter each, so `(?i)i`
    matches two and `(?i)` on either Turkish one matches only itself.

    Nothing outside this family is touched, which is checked by the generator
    rather than assumed here: a code point outside it takes the early return and
    keeps whatever group the table gave it.

    Args:
        point: The code point the group was looked up by.
        other: A member of that group.

    Returns:
        True when Python folds the two together and RE2 does not.
    """
    if point == other:
        return False
    var family = (
        point == LATIN_I
        or point == LATIN_SMALL_I
        or point == LATIN_I_DOTTED
        or point == LATIN_I_DOTLESS
    )
    if not family:
        return False
    var both_plain = (point == LATIN_I or point == LATIN_SMALL_I) and (
        other == LATIN_I or other == LATIN_SMALL_I
    )
    return not both_plain


def _fold_successor(
    lows: Span[Int32, _],
    highs: Span[Int32, _],
    deltas: Span[Int32, _],
    point: Int32,
) -> Int32:
    """The next code point in this one's group, or the code point itself.

    The table is a cycle rather than a list of groups, so a caller reads a whole
    group by starting at a code point and calling this until it comes back to
    where it started. A code point with no other case is its own successor and
    the walk stops at once.

    Args:
        lows: Where each run starts.
        highs: Where each run ends.
        deltas: What to add, or `FOLD_EVEN_ODD`.
        point: The code point.

    Returns:
        The successor, which is the code point itself when it is not cased.
    """
    var at = 0
    var stop = len(lows)
    while at < stop:
        var middle = (at + stop) // 2
        if highs[middle] < point:
            at = middle + 1
        else:
            stop = middle
    if at >= len(lows) or lows[at] > point:
        return point
    var delta = deltas[at]
    if delta == FOLD_EVEN_ODD:
        return point + 1 if (point & 1) == 0 else point - 1
    return point + delta


def _fold_one(
    mut out: List[Int32],
    lows: Span[Int32, _],
    highs: Span[Int32, _],
    deltas: Span[Int32, _],
    point: Int32,
    python: Bool,
    narrow: Bool,
):
    """Adds the other cases of one code point to a set being built.

    The walk stops when the cycle comes back to where it started, which is at
    once for a code point with no other case. RE2's four code point exception is
    spent here rather than in the table, and it is spent on the member rather
    than on the walk: a member RE2 does not fold is stepped over and the walk
    carries on, so the plain `I` and `i` still find each other through a group
    that holds two code points RE2 wants nothing to do with.

    Under `(?a)` the table is not consulted at all. Python folds ASCII letters
    onto ASCII letters and leaves everything else alone, which is the rule
    rather than a narrowing of the rule: the Kelvin sign does not become a `k`,
    the long s does not become an `s`, and the two Turkish letters are two
    letters. So the answer is the thirty two the alphabet is apart, or nothing.

    Args:
        out: The set being built, as low and high pairs.
        lows: Where each run starts.
        highs: Where each run ends.
        deltas: What to add, or `FOLD_EVEN_ODD`.
        point: The code point whose other cases are wanted.
        python: Whether this is Python's engine, which folds all four.
        narrow: Whether that engine was asked for ASCII.
    """
    if narrow:
        if point >= 0x41 and point <= 0x5A:
            out.append(point + 0x20)
            out.append(point + 0x20)
        elif point >= 0x61 and point <= 0x7A:
            out.append(point - 0x20)
            out.append(point - 0x20)
        return
    var other = _fold_successor(lows, highs, deltas, point)
    while other != point:
        if python or not _re2_unfolds(point, other):
            out.append(other)
            out.append(other)
        other = _fold_successor(lows, highs, deltas, other)


def _folded(var ranges: List[Int32], python: Bool, narrow: Bool) -> List[Int32]:
    """The same set of code points with every letter's other cases added.

    This is where `(?i)` is spent. A set that has been through here answers the
    same question with the flag as the original answered without it, so the
    machine that runs the program is never told a flag was set and never pays
    for one. The cost is a walk over the cased code points a class covers, once,
    while the pattern is being compiled.

    The walk is over the table rather than over the set, which matters for
    `(?i)[\\w]`: the class covers 138558 code points and only 2927 of them have
    another case, so the cost is the table's length rather than the class's.

    The walk over the table is still the walk under `(?a)`, even though nothing
    outside ASCII can come back from it. Skipping it would mean a second search
    that knows the same answer, and the table is where the cased code points
    are whichever alphabet is being asked about.

    Args:
        ranges: The set as sorted, merged low and high pairs, consumed.
        python: Whether the program is being compiled for Python's engine, which
            is the one place the two disagree.
        narrow: Whether that engine was asked for ASCII, which folds only the
            twenty six letters onto each other.

    Returns:
        The set with the folds added, sorted and merged again.
    """
    var table_lows = materialize[FOLD_LOW]()
    var table_highs = materialize[FOLD_HIGH]()
    var table_deltas = materialize[FOLD_DELTA]()
    var lows = Span(table_lows)
    var highs = Span(table_highs)
    var deltas = Span(table_deltas)
    var out = ranges.copy()
    for i in range(len(ranges) // 2):
        var low = ranges[i * 2]
        var high = ranges[i * 2 + 1]
        var at = 0
        var stop = len(lows)
        while at < stop:
            var middle = (at + stop) // 2
            if highs[middle] < low:
                at = middle + 1
            else:
                stop = middle
        while at < len(lows) and lows[at] <= high:
            var first = lows[at] if lows[at] > low else low
            var last = highs[at] if highs[at] < high else high
            for step in range(Int(first), Int(last) + 1):
                _fold_one(out, lows, highs, deltas, Int32(step), python, narrow)
            at += 1
    return _sorted_merged(out^)


def _category_is_negated(which: Int32) -> Bool:
    """Whether a category is one of the three capital letters.

    Args:
        which: The `CATEGORY_` value.

    Returns:
        True for `\\D`, `\\S` and `\\W`.
    """
    if which == Int32(Int(CATEGORY_NOT_DIGIT)):
        return True
    if which == Int32(Int(CATEGORY_NOT_SPACE)):
        return True
    return which == Int32(Int(CATEGORY_NOT_WORD))


def _complemented(ranges: List[Int32]) -> List[Int32]:
    """Everything the given ranges leave out.

    Args:
        ranges: Sorted, merged ranges as low and high pairs.

    Returns:
        The complement over every code point, in the same form.
    """
    var out = List[Int32]()
    var next: Int32 = 0
    for i in range(len(ranges) // 2):
        var low = ranges[i * 2]
        var high = ranges[i * 2 + 1]
        if low > next:
            out.append(next)
            out.append(low - 1)
        if high + 1 > next:
            next = high + 1
    if next <= UNICODE_LAST:
        out.append(next)
        out.append(UNICODE_LAST)
    return out^


def _class_ranges(
    mut b: _Builder, nodes: List[Node], node: Int32, mut negated: Bool
) -> List[Int32]:
    """Gathers what the things between square brackets add up to.

    A negated category inside a class is expanded before the union rather than
    after it, which is the only way `[\\d\\D]` comes out as every character and
    `[^\\W]` comes out as a word character. Python's own parser does the same
    thing for the same reason.

    Under `(?i)` each item is folded as it arrives rather than the union being
    folded once at the end, and a negated category is folded and then negated
    rather than the other way round. Both of those are RE2's arrangement and
    neither is an optimisation. `(?i)[\\W]` on RE2 does not match a `k`, and
    folding the complement of an ASCII word class would add one, because the
    Kelvin sign is outside that class and folds onto a letter that is inside
    it. RE2 says as much in a comment where it does this, and a class whose
    items are folded one at a time is the only reading that agrees.

    Args:
        b: The builder, told about anything that cannot be compiled.
        nodes: The arena.
        node: The `OP_IN` node.
        negated: Set to True when the class began with a caret.

    Returns:
        The ranges the class covers, sorted and merged, before negation.
    """
    var folding = (b.flags & FLAG_IGNORECASE) != 0
    var gathered = List[Int32]()
    var child = nodes[Int(node)].first
    while child >= 0:
        var it = nodes[Int(child)]
        if it.op == OP_NEGATE:
            negated = True
        elif it.op == OP_LITERAL or it.op == OP_RANGE:
            var high = it.b if it.op == OP_RANGE else it.a
            var span: List[Int32] = [it.a, high]
            if folding:
                span = _folded(span^, b.python, b.narrow)
            for i in range(len(span)):
                gathered.append(span[i])
        elif it.op == OP_CATEGORY:
            var pieces = _category_ranges(it.a, b.python, b.narrow)
            if folding:
                pieces = _folded(pieces^, b.python, b.narrow)
            if _category_is_negated(it.a):
                pieces = _complemented(_sorted_merged(pieces^))
            for i in range(len(pieces)):
                gathered.append(pieces[i])
        else:
            b.give_up(String("unsupported item in a character class"))
            return List[Int32]()
        child = nodes[Int(child)].next
    return _sorted_merged(gathered^)


def _at_value(b: _Builder, which: Int32) -> Int32:
    """Which position instruction an anchor becomes once the flags are read.

    The parser writes down the character the caller typed and this turns it
    into what it means, because `^` under `(?m)` is a different question from
    `^` without it and the parser does not know the flags when it reads the
    caret.

    It also turns it into what it means to the engine that is going to run it,
    which is the second of the three places the two engines part company in this
    file. `\\b` and `\\B` ask about a word character and the two engines mean
    different sets of characters by that, and `$` outside multiline mode matches
    before a newline that ends the text to Python and only at the end to RE2.
    Both are settled here so that nothing downstream has to know.

    The word boundary is also the anchor `(?a)` moves, and both halves of it
    move onto RE2's, because the ASCII word class is the same set of characters
    for both engines and neither of RE2's two questions asks about anything
    else. `$` is not moved at all, since the alphabet has nothing to say about
    where a line ends.

    `\\B` used to need a value of its own under that letter, because Python up
    to 3.13 fails a `\\B` on an empty row and RE2 matches one, so RE2's value
    brought its answer for the empty row along with its alphabet. That case is
    an instruction of its own now, written in front of this one by the caller of
    this function, which is what lets the two alphabets be two alphabets again.

    Args:
        b: The builder, for its flags and its engine.
        which: The `AT_` value the parser wrote.

    Returns:
        The `AT_` value to check at run time.
    """
    if b.python:
        if which == Int32(Int(AT_BOUNDARY)) and not b.narrow:
            return Int32(Int(AT_BOUNDARY_UNICODE))
        if which == Int32(Int(AT_NON_BOUNDARY)) and not b.narrow:
            return Int32(Int(AT_NON_BOUNDARY_UNICODE))
        if which == Int32(Int(AT_END)):
            if (b.flags & FLAG_MULTILINE) == 0:
                return Int32(Int(AT_END_TEXT))
            return Int32(Int(AT_END_LINE))
    if (b.flags & FLAG_MULTILINE) == 0:
        return which
    if which == Int32(Int(AT_BEGINNING)):
        return Int32(Int(AT_BEGINNING_LINE))
    if which == Int32(Int(AT_END)):
        return Int32(Int(AT_END_LINE))
    return which


def _emit_node(mut b: _Builder, nodes: List[Node], node: Int32):
    """Writes the instructions one node turns into.

    Args:
        b: The builder.
        nodes: The arena.
        node: The node.
    """
    if b.failed:
        return
    if node < 0:
        return
    var it = nodes[Int(node)]

    var folding = (b.flags & FLAG_IGNORECASE) != 0

    if it.op == OP_LITERAL:
        if folding:
            var one: List[Int32] = [it.a, it.a]
            var group = _folded(one^, b.python, b.narrow)
            # A letter with no other case is still one code point after
            # folding, and emitting it as a set of one would make the
            # commonest instruction in the program a binary search.
            if len(group) > 2 or group[0] != group[1]:
                b.add_set(group, False)
                return
        _ = b.emit(IN_CHAR, it.a, 0)
        return
    if it.op == OP_NOT_LITERAL:
        var one: List[Int32] = [it.a, it.a]
        if folding:
            one = _folded(one^, b.python, b.narrow)
        b.add_set(one, True)
        return
    if it.op == OP_ANY:
        if (b.flags & FLAG_DOTALL) != 0:
            _ = b.emit(IN_ANY_ALL, 0, 0)
        else:
            _ = b.emit(IN_ANY, 0, 0)
        return
    if it.op == OP_AT:
        if (
            b.python
            and b.minor < PYTHON_PLAIN_NON_BOUNDARY
            and it.a == Int32(Int(AT_NON_BOUNDARY))
        ):
            # Two instructions for one node, and the first of them is the whole
            # of what the older interpreters have that 3.14 does not. A thread
            # has to satisfy both to go on, so a row with nothing in it dies
            # here and a row with something in it reads the second one and gets
            # the plain answer. Which alphabet was asked for does not come into
            # it, which is the point: `(?a)` narrows which characters count as
            # word characters and says nothing about a row that holds none.
            _ = b.emit(IN_AT, Int32(Int(AT_TEXT_NOT_EMPTY)), 0)
        _ = b.emit(IN_AT, _at_value(b, it.a), 0)
        return
    if it.op == OP_CATEGORY:
        # Folded before the negation is applied and not after it, and folded at
        # all because RE2's three classes are ASCII and an ASCII class is not
        # closed under folding: the Kelvin sign and the long s are outside
        # `[0-9A-Za-z_]` and fold onto letters that are inside it, so `(?i)\\w`
        # matches both on RE2 and `(?i)\\W` matches neither. Python's three are
        # Unicode and are closed, so the same two lines are a no change there
        # and are still worth running rather than branching on the engine.
        var pieces = _category_ranges(it.a, b.python, b.narrow)
        if folding:
            pieces = _folded(pieces^, b.python, b.narrow)
        b.add_set(_sorted_merged(pieces^), _category_is_negated(it.a))
        return
    if it.op == OP_IN:
        var negated = False
        # Each item inside the brackets was folded as it arrived, so what is
        # left here is the caret. It is applied last, which is the only order
        # that answers `(?i)[^a]` on a capital A the way both engines do.
        # Folding the complement instead would add the small a back in through
        # the other case of every letter that is not the one written down.
        var pieces = _class_ranges(b, nodes, node, negated)
        if b.failed:
            return
        b.add_set(pieces, negated)
        return
    if it.op == OP_SUBPATTERN:
        # The parser drops a non capturing group by inlining it, so every one of
        # these opened a bracket somebody can refer to, and `a` is the number
        # they would refer to it by.
        if b.captures and it.a >= 1:
            _ = b.emit(IN_SAVE, it.a * 2, 0)
            _emit_children(b, nodes, node)
            _ = b.emit(IN_SAVE, it.a * 2 + 1, 0)
            return
        _emit_children(b, nodes, node)
        return
    if it.op == OP_SCOPE:
        # The whole of a scoped flag group, and most of it is a save and a put
        # back because every question a flag answers was already being asked of
        # the builder rather than of the tree. Restoring the outer pair is what
        # makes nesting work and is also what makes `(?i:a)b` fold the `a` and
        # not the `b`, since the sibling is emitted after this node returns.
        #
        # The three alphabet letters do not combine the way the other four do.
        # Naming one of them clears all three first, so `(?u:\w)` under a
        # global ascii flag is the wide alphabet rather than both letters at
        # once, and that is upstream's `_combine_flags` rather than a reading of
        # it. The other four letters are independent and are a plain on and off.
        #
        # `narrow` is recomputed rather than saved and set, because it is a
        # reading of the flags and two fields that can disagree are worse than
        # one line that cannot. Verbose mode is in `it.a` and is ignored here,
        # having been spent by the parser.
        var flags = b.flags
        var narrow = b.narrow
        if (it.a & TYPE_FLAGS) != 0:
            b.flags &= ~TYPE_FLAGS
        b.flags = (b.flags | it.a) & ~it.b
        b.narrow = b.python and (b.flags & FLAG_ASCII) != 0
        _emit_children(b, nodes, node)
        b.flags = flags
        b.narrow = narrow
        return
    if it.op == OP_SEQ:
        _emit_children(b, nodes, node)
        return
    if it.op == OP_BRANCH:
        _emit_branch(b, nodes, node)
        return
    if it.op == OP_MAX_REPEAT:
        _emit_repeat(b, nodes, node, True)
        return
    if it.op == OP_MIN_REPEAT:
        _emit_repeat(b, nodes, node, False)
        return
    if it.op == OP_ASSERT or it.op == OP_ASSERT_NOT:
        _emit_lookahead(b, nodes, node, it.op == OP_ASSERT)
        return
    if it.op == OP_FAILURE:
        # A set with no ranges in it, which nothing is a member of. The parser
        # writes this node for `(?!)` and for nothing else, and upstream reads
        # that as a pattern that never matches rather than as an error. An
        # instruction of its own would be a second way to say what an empty set
        # already says.
        b.add_set(List[Int32](), False)
        return

    b.give_up(String("unsupported pattern"))


def _emit_lookahead(
    mut b: _Builder, nodes: List[Node], node: Int32, positive: Bool
):
    """Writes a lookahead and the body it asks about.

    The body goes into the same instruction list as everything else, with a jump
    written over it so that the walk that runs the pattern never falls into it
    from in front. The only way in is the `IN_LOOK`, which does not step there
    but hands the position to a second machine.

    The order is the whole of the trick and it is worth reading once. The look
    instruction is written first so that the thread reaching it is at the right
    position, the jump over the body is written second so that the look falls
    through to it when the answer agrees, and the body and its match come last
    so that the jump has somewhere to land.

    Args:
        b: The builder.
        nodes: The arena.
        node: The assertion.
        positive: Whether the body has to match rather than has to not match.
    """
    var look = b.emit(IN_LOOK, 0, Int32(1) if positive else Int32(0))
    var over = b.emit(IN_JUMP, 0, 0)
    b.patch_a(look, b.here())
    _emit_children(b, nodes, node)
    _ = b.emit(IN_MATCH, 0, 0)
    b.patch_a(over, b.here())


def _refuse_construct(mut b: _Builder, what: String):
    """Says that a construct will not compile, in the voice of whichever engine
    asked for it.

    The six constructs below are the five RE2 has never had plus the collapse
    the parser writes for an empty negative lookaround, and refusing them means
    two different things. For RE2 the refusal is agreement with upstream, since
    pandas hands the pattern to RE2 and RE2 raises. For Python the same pattern
    is one pandas answers perfectly well, so a refusal here is a shortfall of
    this library and is flagged as a gap.

    Args:
        b: The builder.
        what: The construct, as a noun phrase that follows `has no`.
    """
    if b.python:
        b.give_up(String("this engine has no ", what, " yet"), True)
        return
    b.give_up(String("RE2 has no ", what))


def _holds_group(nodes: List[Node], node: Int32) -> Bool:
    """Whether a subtree opens a bracket somebody could refer to.

    Every `OP_SUBPATTERN` the parser leaves behind is a capturing one, because a
    non capturing group is inlined, so this is a search for the node rather than
    a search for a number on it.

    Args:
        nodes: The arena.
        node: Where to start, which may be a whole list of siblings.

    Returns:
        True when there is a capturing group anywhere under it.
    """
    var at = node
    while at >= 0:
        var it = nodes[Int(at)]
        if it.op == OP_SUBPATTERN:
            return True
        if _holds_group(nodes, it.first):
            return True
        at = it.next
    return False


def _check_node(mut b: _Builder, nodes: List[Node], node: Int32, budget: Int32):
    """Looks over the whole tree for anything that cannot be compiled.

    A separate walk from the one that writes instructions, and the reason is a
    pattern the corpus found: `(?P<n>a)(?P=n){0}` holds a backreference and
    compiles to nothing at all, because a repeat with a count of zero writes its
    body zero times and the refusal never happens. RE2 parses the whole pattern
    before it runs any of it and refuses that one, so a compiler that only looks
    at what it emits disagrees with RE2 about every construct sitting under a
    `{0}`.

    So this enters every child of every node, including the bodies that are
    about to be thrown away, which is exactly what a parser does and exactly
    what the emitting walk cannot do.

    The repeat budget is carried down the same walk, because RE2 spends it the
    same way: every repeat divides what is left by how many copies it asks for,
    the children of a sequence each get the whole of what their parent had, and
    a repeat asking for more than is left is the error. Doing it here rather
    than where the copies are written means `(a{11}){91}{0}` is refused too.

    Args:
        b: The builder, told about anything that cannot be compiled.
        nodes: The arena.
        node: Where to start.
        budget: How many copies of this node RE2 has left to spend.
    """
    if node < 0 or b.failed:
        return
    var it = nodes[Int(node)]
    if it.op == OP_ASSERT or it.op == OP_ASSERT_NOT:
        if not b.python:
            _refuse_construct(b, String("lookaround"))
            return
        if it.a < 0:
            # A lookbehind is a different question from a lookahead and not a
            # harder version of the same one. Python reads one by trying the
            # body at a position in front of where the machine has got to, which
            # means the body has to have a width the compiler knows, which means
            # a width analysis over the tree and the refusal Python raises for a
            # body that has not got one. None of that is here yet and none of it
            # is needed for the half that is. Document 93.
            _refuse_construct(b, String("lookbehind"))
            return
        if b.captures and _holds_group(nodes, it.first):
            # A group inside a lookahead keeps what it matched upstream, so
            # `re.match(r"(?=(a))a", "a").group(1)` is `a`. The body here is run
            # by a second machine that carries no slots, so the parent thread
            # would come back with the group empty, which is a wrong answer
            # rather than a refusal. Only a caller who asked for captures can
            # see the difference, which is why the question is asked of the
            # builder rather than of the tree alone.
            b.give_up(
                String("this engine has no capture inside a lookahead yet"),
                True,
            )
            return
    if it.op == OP_GROUPREF:
        _refuse_construct(b, String("backreference"))
        return
    if it.op == OP_GROUPREF_EXISTS:
        _refuse_construct(b, String("conditional group"))
        return
    if it.op == OP_ATOMIC_GROUP:
        _refuse_construct(b, String("atomic group"))
        return
    if it.op == OP_POSSESSIVE_REPEAT:
        _refuse_construct(b, String("possessive quantifier"))
        return
    if it.op == OP_FAILURE:
        if not b.python:
            _refuse_construct(b, String("empty negative lookaround"))
            return
        return
    if it.op == OP_AT and it.a == Int32(Int(AT_NON_BOUNDARY)) and not b.python:
        # RE2 asks the word boundary question between bytes rather than between
        # characters, so `\B` matches in the middle of any character that takes
        # more than one byte to write, and `str.contains(r"\B")` on a column
        # holding one word in Greek is True upstream and False here. `\b` is
        # not affected, since a byte in the middle of a character is not a word
        # byte on either side of it and a boundary needs one. Running this
        # engine over bytes rather than code points is the fix and it is a
        # larger change than this one.
        #
        # Python asks it between characters, which is what this engine already
        # walks, so there is nothing to refuse on that side.
        b.give_up(String("RE2 reads a non boundary between bytes"), True)
        return

    var left = budget
    if it.op == OP_MAX_REPEAT or it.op == OP_MIN_REPEAT:
        var asked = it.a if it.b == MAXREPEAT else it.b
        if asked < 1:
            asked = 1
        if asked > budget:
            if b.python:
                b.give_up(
                    String("this engine will not repeat that many times"), True
                )
                return
            b.give_up(String("RE2 will not repeat that many times"))
            return
        left = budget // asked

    var child = it.first
    while child >= 0 and not b.failed:
        _check_node(b, nodes, child, left)
        child = nodes[Int(child)].next


def _emit_children(mut b: _Builder, nodes: List[Node], node: Int32):
    """Writes a node's children one after another.

    Args:
        b: The builder.
        nodes: The arena.
        node: The parent.
    """
    var child = nodes[Int(node)].first
    while child >= 0 and not b.failed:
        _emit_node(b, nodes, child)
        child = nodes[Int(child)].next


def _emit_branch(mut b: _Builder, nodes: List[Node], node: Int32):
    """Writes an alternation.

    Each alternative but the last is reached by a split whose second arm is the
    next one along, and each alternative but the last ends in a jump to the end
    of the whole thing. The alternatives are tried in the order they are
    written, which both engines do, and which matters for `a|ab` even though it
    does not for `ab|a`.

    Args:
        b: The builder.
        nodes: The arena.
        node: The `OP_BRANCH` node.
    """
    var ends = List[Int32]()
    var child = nodes[Int(node)].first
    while child >= 0 and not b.failed:
        var after = nodes[Int(child)].next
        if after < 0:
            _emit_node(b, nodes, child)
            break
        var split = b.emit(IN_SPLIT, 0, 0)
        b.patch_a(split, b.here())
        _emit_node(b, nodes, child)
        ends.append(b.emit(IN_JUMP, 0, 0))
        b.patch_b(split, b.here())
        child = after
    var done = b.here()
    for i in range(len(ends)):
        b.patch_a(ends[i], done)


def _emit_star(mut b: _Builder, nodes: List[Node], node: Int32, greedy: Bool):
    """Writes a repeat's body with no ceiling on how many times it runs.

    Args:
        b: The builder.
        nodes: The arena.
        node: The repeat.
        greedy: Whether to prefer going round again.
    """
    var split = b.emit(IN_SPLIT, 0, 0)
    var body = b.here()
    _emit_children(b, nodes, node)
    _ = b.emit(IN_JUMP, split, 0)
    var after = b.here()
    if greedy:
        b.patch_a(split, body)
        b.patch_b(split, after)
    else:
        b.patch_a(split, after)
        b.patch_b(split, body)


def _emit_repeat(mut b: _Builder, nodes: List[Node], node: Int32, greedy: Bool):
    """Writes a quantifier, by copying its body as many times as it says.

    A Thompson program has no counter, so a count is a number of copies. That
    is why `MAX_INSTRUCTIONS` exists, and it is also why the copies for the
    optional part are nested rather than laid side by side: `a{0,3}` has to be
    `a(a(a)?)?` and not `a?a?a?`, because the second one matches a gap in the
    middle and the first one does not. For a pattern made only of `a` that is
    the same set of strings, and for `(?:ab){0,3}` it is not.

    Args:
        b: The builder.
        nodes: The arena.
        node: The repeat.
        greedy: Whether the quantifier is greedy.
    """
    var it = nodes[Int(node)]
    var least = it.a
    var most = it.b
    if least > most:
        b.give_up(String("min repeat greater than max repeat"))
        return
    for _ in range(Int(least)):
        _emit_children(b, nodes, node)
        if b.failed:
            return
    if most == MAXREPEAT:
        _emit_star(b, nodes, node, greedy)
        return
    var extra = Int(most - least)
    if extra == 0:
        return
    var splits = List[Int32]()
    for _ in range(extra):
        var split = b.emit(IN_SPLIT, 0, 0)
        splits.append(split)
        if greedy:
            b.patch_a(split, b.here())
        else:
            b.patch_b(split, b.here())
        _emit_children(b, nodes, node)
        if b.failed:
            return
    var after = b.here()
    for i in range(len(splits)):
        if greedy:
            b.patch_b(splits[i], after)
        else:
            b.patch_a(splits[i], after)


def _refused_flags(flags: Int32) -> String:
    """Which of a set of flags RE2 does not have, as its own complaint.

    RE2 reads three inline flag letters and refuses the other four outright,
    with the same message for all four, and the message names the operator
    rather than the letter. Two of the four are the interesting ones. `(?a)`
    asks for exactly the classes RE2 already uses and RE2 will not take the
    request, and `(?u)` asks for what Python does anyway.

    `(?i)` is not here, because RE2 has it and so does this. What RE2 folds is
    in `folddata.mojo` and is spent while the pattern is being compiled.

    The same four letters are refused in either form, which is why this takes a
    bitmask rather than the parse: the caller asks it once about the global
    flags and once about the scoped ones and reads the answer the same way.

    Args:
        flags: The flags, as `FLAG_` bits.

    Returns:
        The reason, or empty when there is none.
    """
    if (flags & FLAG_LOCALE) != 0:
        return String("invalid perl operator: (?L")
    if (flags & FLAG_VERBOSE) != 0:
        return String("invalid perl operator: (?x")
    if (flags & FLAG_ASCII) != 0:
        return String("invalid perl operator: (?a")
    if (flags & FLAG_UNICODE) != 0:
        return String("invalid perl operator: (?u")
    return String("")


def _refused_flags_python(flags: Int32) -> String:
    """The same question asked about Python's engine, which turns down one
    letter of the seven.

    The four RE2 will not hear of are all letters Python reads, and three of
    them are read here now as well. `(?u)` asks for what this engine does
    anyway, `(?x)` is spent in the parser before a tree reaches this, and `(?a)`
    is carried on the builder and spent on the classes, the folding and the word
    boundary. None of the three leaves anything for this to say.

    `(?L)` never reaches this, and is answered anyway. Python has the letter and
    will not take it on a pattern made of text, which is the only kind that
    arrives here, so the parser turns the pattern down before any of this and
    the caller is told that Python's grammar cannot read it. The branch is here
    because a reader looking for the seventh letter should find it rather than
    conclude it was forgotten.

    Args:
        flags: The flags, as `FLAG_` bits.

    Returns:
        The reason, or empty when there is none.
    """
    if (flags & FLAG_LOCALE) != 0:
        return String("a locale flag cannot be used on text")
    return String("")


def _anchored(code: Span[Instruction, _]) -> Bool:
    """Whether a program can only match at the start of the text.

    Reads the instructions rather than the tree, because the tree has the flags
    on one side and the anchor on the other and the compiler has already put the
    two together: multiline turns `^` into `AT_BEGINNING_LINE` while it is being
    emitted, so a program that still holds `AT_BEGINNING` was compiled without
    the flag and there is nothing left to work out here.

    The walk steps over the saves a program compiled with captures opens with
    and then looks at one instruction. Anything else, a split from an
    alternation or a character or a jump, answers False, so this says nothing
    about a pattern that has an anchor somewhere other than in front.

    Args:
        code: The instructions, which start at zero.

    Returns:
        True when every attempt above position zero is known to fail.
    """
    var pc = 0
    while pc < len(code) and code[pc].op == IN_SAVE:
        pc += 1
    if pc >= len(code) or code[pc].op != IN_AT:
        return False
    return code[pc].a == Int32(Int(AT_BEGINNING)) or code[pc].a == Int32(
        Int(AT_BEGINNING_STRING)
    )


def _first_ranges(
    code: Span[Instruction, _], ranges: Span[Int32, _]
) -> List[Int32]:
    """The characters a match can begin with, or nothing when anything can.

    The walk starts at instruction zero and goes wherever the program can go
    without reading a character, collecting what each character reading
    instruction it arrives at would accept. A position holding a character
    outside the union of those cannot begin a match, because whichever way the
    first step went it would have to read one of them.

    An assertion is walked through as though it held, which makes the set larger
    than it has to be and never smaller. `\\bfoo` collects `f` rather than
    working out where a word boundary could be, and that is the right way round:
    a set that is too large skips fewer positions and still skips only positions
    that cannot match.

    Two shapes answer with nothing at all. A program that can reach its match
    instruction without reading a character can match nothing, and then every
    position begins a match and there is nothing to skip. A program that can
    begin with `(?s).` accepts every character, so the set would be everything.

    Args:
        code: The instructions, which start at zero.
        ranges: The program's range table.

    Returns:
        The set as sorted disjoint low and high pairs, empty when there is no
        set worth having.
    """
    var seen = List[Bool](length=len(code), fill=False)
    var stack = List[Int32]()
    stack.append(0)
    var out = List[Int32]()
    while len(stack) > 0:
        var pc = Int(stack.pop())
        if seen[pc]:
            continue
        seen[pc] = True
        var instruction = code[pc]
        if instruction.op == IN_JUMP:
            stack.append(instruction.a)
        elif instruction.op == IN_SPLIT:
            stack.append(instruction.a)
            stack.append(instruction.b)
        elif instruction.op == IN_AT or instruction.op == IN_SAVE:
            stack.append(Int32(pc + 1))
        elif instruction.op == IN_CHAR:
            out.append(instruction.a)
            out.append(instruction.a)
        elif instruction.op == IN_SET:
            for i in range(Int(instruction.b)):
                out.append(ranges[Int(instruction.a) + i * 2])
                out.append(ranges[Int(instruction.a) + i * 2 + 1])
        elif instruction.op == IN_NOT_SET:
            var held = List[Int32]()
            for i in range(Int(instruction.b)):
                held.append(ranges[Int(instruction.a) + i * 2])
                held.append(ranges[Int(instruction.a) + i * 2 + 1])
            var rest = _complemented(held)
            for i in range(len(rest)):
                out.append(rest[i])
        elif instruction.op == IN_ANY:
            out.append(0)
            out.append(0x09)
            out.append(0x0B)
            out.append(UNICODE_LAST)
        else:
            return List[Int32]()
    return _sorted_merged(out^)


def _first_worth_having(ranges: List[Int32]) -> Bool:
    """Whether a set of first characters would pay for the test it costs.

    Args:
        ranges: The set, sorted and disjoint.

    Returns:
        True when it holds three quarters of ASCII or less.
    """
    if len(ranges) == 0:
        return False
    var held = 0
    for i in range(len(ranges) // 2):
        var low = ranges[i * 2]
        var high = ranges[i * 2 + 1]
        if low < 0:
            low = 0
        if high > 127:
            high = 127
        if high >= low:
            held += Int(high - low) + 1
    return held <= 96


comptime MAX_CLASS_TESTS: Int = 256
"""How many different questions about a character a program may ask before it
is left without a class table.

Every distinct question doubles the number of classes the alphabet could be cut
into, and the table is built by giving each piece of the alphabet the answers to
all of them and then grouping the pieces that answered alike. That is a walk per
piece per question, so a ceiling keeps the compiler's work bounded for a pattern
nobody meant to write. A repeat is compiled by copying its body and the copies
ask the same questions, so the count here is the distinct ones: `(?:abcde){1000}`
is five thousand instructions and five questions.
"""


comptime MAX_CLASS_PIECES: Int = 4096
"""How many pieces the alphabet may be cut into before the table is given up on.

A Unicode class has hundreds of ranges and a pattern may hold several, so this
is not a number a real pattern reaches either. It is here for the same reason as
the ceiling above: the grouping compares a piece against the classes found so
far, so the work is the pieces times the classes and both of them need a top.
"""


comptime MAX_CLASS_CODE: Int = 20000
"""How long a program may be before it is left without a table.

`MAX_INSTRUCTIONS` is ten times this, because a counted repeat is compiled by
copying its body and a pattern is allowed to say `{1000}`. Every copy asks a
question already on the list and is thrown away, but it is looked at first, so
the walk is the length of the program and something has to bound it. A program
this long is not one a DFA would be kept for anyway.
"""


def _class_tests(
    code: Span[Instruction, _], ranges: Span[Int32, _], mut too_many: Bool
) -> List[List[Int32]]:
    """Every different question the program asks about a character.

    A question is a set: the characters that answer it yes. `IN_NOT_SET` asks
    the same question as the set it negates, since what the alphabet needs to
    know is which characters the program can tell apart rather than what it does
    with the answer, and the two instructions tell the same pairs apart. A full
    stop asks about the newline, a word boundary asks about the word class, and
    a `^` or `$` that reads a line ending asks about the newline again.

    `IN_ANY_ALL` asks nothing. That is the point of it: `(?s).` reads a
    character and cannot tell any two of them apart.

    The instructions are collected by where their set is in the table before any
    of the sets are read out, so the thousand copies a counted repeat made cost
    a comparison of three numbers each rather than a comparison of their ranges.
    Two sets that are equal and written twice are still caught, on the way out.

    Args:
        code: The instructions.
        ranges: The program's range table.
        too_many: Set when the program is longer than `MAX_CLASS_CODE` or asks
            more questions than `MAX_CLASS_TESTS`, which is how a caller tells a
            program that asks too many from one that asks none.

    Returns:
        One sorted disjoint set per question, deduplicated, and empty when there
        are none or when there are too many.
    """
    var out = List[List[Int32]]()
    if len(code) > MAX_CLASS_CODE:
        too_many = True
        return out^
    var kinds = List[UInt8]()
    var at = List[Int32]()
    var counts = List[Int32]()
    var newline = False
    var word_ascii = False
    var word_wide = False
    for i in range(len(code)):
        var instruction = code[i]
        var kind = instruction.op
        if kind == IN_NOT_SET:
            kind = IN_SET
        elif kind == IN_ANY:
            newline = True
            continue
        elif kind == IN_AT:
            var which = UInt8(Int(instruction.a))
            if which == AT_BOUNDARY or which == AT_NON_BOUNDARY:
                word_ascii = True
            elif (
                which == AT_BOUNDARY_UNICODE or which == AT_NON_BOUNDARY_UNICODE
            ):
                word_wide = True
            elif (
                which == AT_BEGINNING_LINE
                or which == AT_END_LINE
                or which == AT_END
                or which == AT_END_TEXT
            ):
                # RE2's `$` reads nothing and Python's reads the character
                # before the end, and neither is worth telling apart from the
                # other here, since one newline in the alphabet costs one class.
                newline = True
            continue
        elif kind != IN_CHAR and kind != IN_SET:
            continue
        var seen = False
        for k in range(len(kinds)):
            if (
                kinds[k] == kind
                and at[k] == instruction.a
                and counts[k] == instruction.b
            ):
                seen = True
                break
        if seen:
            continue
        kinds.append(kind)
        at.append(instruction.a)
        counts.append(instruction.b)
        if len(kinds) > MAX_CLASS_TESTS:
            too_many = True
            return List[List[Int32]]()
    for i in range(len(kinds)):
        var held = List[Int32]()
        if kinds[i] == IN_CHAR:
            held.append(at[i])
            held.append(at[i])
        else:
            for k in range(Int(counts[i])):
                held.append(ranges[Int(at[i]) + k * 2])
                held.append(ranges[Int(at[i]) + k * 2 + 1])
        var set = _sorted_merged(held^)
        if not _already_asked(out, set):
            out.append(set^)
    if newline:
        var only = List[Int32]()
        only.append(0x0A)
        only.append(0x0A)
        if not _already_asked(out, only):
            out.append(only^)
    if word_ascii:
        var narrow = List[Int32]()
        narrow.append(0x30)
        narrow.append(0x39)
        narrow.append(0x41)
        narrow.append(0x5A)
        narrow.append(0x5F)
        narrow.append(0x5F)
        narrow.append(0x61)
        narrow.append(0x7A)
        if not _already_asked(out, narrow):
            out.append(narrow^)
    if word_wide:
        var wide = word_ranges_unicode()
        if not _already_asked(out, wide):
            out.append(wide^)
    return out^


def _already_asked(asked: List[List[Int32]], set: List[Int32]) -> Bool:
    """Whether a question is one of the ones already on the list.

    Args:
        asked: The questions so far.
        set: The question, sorted and disjoint.

    Returns:
        True when some entry holds the same ranges.
    """
    for i in range(len(asked)):
        if len(asked[i]) != len(set):
            continue
        var same = True
        for k in range(len(set)):
            if asked[i][k] != set[k]:
                same = False
                break
        if same:
            return True
    return False


def _class_table(tests: List[List[Int32]]) -> List[Int32]:
    """Cuts the code points into classes, as pairs of a start and a class.

    Every question's two ends are a place the alphabet may have to be cut, so
    the starts are the low end of every range and the code point after every
    high end, with zero added and sorted. Between two of those nothing can tell
    one character from another, which is what a class is.

    Two pieces far apart get the same class when they answer every question the
    same way, which is what keeps `abc` down to four classes rather than five:
    the characters below `a` and the ones above `c` are the same thing to that
    program. A transition table is as wide as the class count, so the grouping
    is worth doing rather than leaving the pieces as they are.

    Args:
        tests: The questions, as `_class_tests` returned them.

    Returns:
        A start and a class for each piece, in order, with the first start zero
        and every code point covered by the piece it falls in. Empty when there
        are more pieces than `MAX_CLASS_PIECES`.
    """
    var starts = List[Int32]()
    starts.append(0)
    for i in range(len(tests)):
        for k in range(len(tests[i]) // 2):
            var low = tests[i][k * 2]
            var high = tests[i][k * 2 + 1]
            if low > 0:
                starts.append(low)
            if high < UNICODE_LAST:
                starts.append(high + 1)
    starts = _sorted_unique(starts^)
    if len(starts) > MAX_CLASS_PIECES:
        return List[Int32]()
    # The signature of a piece is its answer to every question, one bit each,
    # and two pieces are the same class when their signatures are equal. The
    # bits are packed into words so that the comparison is a few integers
    # rather than a walk over the questions.
    var out = List[Int32]()
    if len(tests) == 0:
        # A program that cannot tell any two characters apart, which is `(?s).`
        # and nothing else. One class, holding everything.
        out.append(0)
        out.append(0)
        return out^
    var words = (len(tests) + 63) // 64
    var signatures = List[UInt64]()
    var classes = 0
    for i in range(len(starts)):
        var here = List[UInt64](length=words, fill=0)
        for k in range(len(tests)):
            if in_set(
                Span(tests[k]),
                0,
                Int32(len(tests[k]) // 2),
                UInt32(Int(starts[i])),
            ):
                here[k // 64] |= UInt64(1) << UInt64(k % 64)
        var found = -1
        for k in range(classes):
            var same = True
            for w in range(words):
                if signatures[k * words + w] != here[w]:
                    same = False
                    break
            if same:
                found = k
                break
        if found < 0:
            found = classes
            classes += 1
            for w in range(words):
                signatures.append(here[w])
        out.append(starts[i])
        out.append(Int32(found))
    return out^


def _sorted_unique(var values: List[Int32]) -> List[Int32]:
    """Puts numbers in order and drops the repeats.

    Args:
        values: The numbers, in any order, consumed.

    Returns:
        The same numbers, ascending, each appearing once.
    """
    sort(values)
    var out = List[Int32]()
    for i in range(len(values)):
        if i == 0 or values[i] != values[i - 1]:
            out.append(values[i])
    return out^


def _fill_classes(mut program: Program):
    """Works out the program's alphabet and writes it onto the program.

    The ASCII half is one entry per character and the rest is the pieces above
    it, which is the shape a column of text asks for: mostly ASCII, read by an
    index, with a search kept for the characters that need one.

    A program that asks too much is left without a table rather than given a
    wrong one, and `class_count` staying zero is how it says so.

    Args:
        program: The compiled program, with its code and ranges already on it.
    """
    var too_many = False
    var tests = _class_tests(Span(program.code), Span(program.ranges), too_many)
    if too_many:
        return
    var table = _class_table(tests)
    if len(table) == 0:
        return
    var count = 0
    for i in range(len(table) // 2):
        if Int(table[i * 2 + 1]) + 1 > count:
            count = Int(table[i * 2 + 1]) + 1
    program.class_count = Int32(count)
    var at = 0
    for point in range(128):
        while at + 1 < len(table) // 2 and Int(table[(at + 1) * 2]) <= point:
            at += 1
        program.class_ascii.append(table[at * 2 + 1])
    # The piece ASCII ended inside is written again with 128 as its start, so
    # that the first entry up here is always the one a character just above
    # ASCII falls in and the search has nothing to say about an empty list.
    while at + 1 < len(table) // 2 and Int(table[(at + 1) * 2]) <= 128:
        at += 1
    program.class_above.append(128)
    program.class_above.append(table[at * 2 + 1])
    for i in range(at + 1, len(table) // 2):
        program.class_above.append(table[i * 2])
        program.class_above.append(table[i * 2 + 1])


def class_of(program: Program, point: UInt32) -> Int32:
    """Which class of the program's alphabet a character is in.

    Args:
        program: The compiled program, which has to have a class table for this
            to mean anything.
        point: The code point.

    Returns:
        The class, from zero, and zero for a program with no table at all.
    """
    if program.class_count == 0:
        return 0
    if point < 128:
        return program.class_ascii[Int(point)]
    var value = Int32(Int(point))
    var low = 0
    var high = len(program.class_above) // 2 - 1
    var found = program.class_above[1]
    while low <= high:
        var middle = (low + high) // 2
        if program.class_above[middle * 2] <= value:
            found = program.class_above[middle * 2 + 1]
            low = middle + 1
        else:
            high = middle - 1
    return found


def compile_program(
    tree: Parsed,
    engine: UInt8,
    captures: Bool = False,
    minor: Int = PYTHON_NEWEST,
    alphabet: Bool = False,
) -> Program:
    """Turns a parsed pattern into a program for one of the two engines.

    Args:
        tree: The pattern, as `parse_pattern` read it.
        engine: Which engine is to run it.
        captures: Whether the caller needs to know where each group matched as
            well as whether the pattern did. A program built this way carries a
            save instruction around every group and one around the whole match,
            and every thread running it carries a copy of the slots, so it is
            the slower of the two and only the callers that need the text of a
            match ask for it.
        minor: Which CPython this program is being compiled beside, as the
            minor number alone. Ignored on RE2, which has not got one. On
            Python's engine it decides one rule, which is whether `\\B` matches
            a row with nothing in it, and the default is the newest version
            this library has been measured against.
        alphabet: Whether to work out which characters the program can tell
            apart and write the class table on it. Off by default, because
            nothing reads the table yet and building one costs several times
            what compiling a short pattern costs and a millisecond or two for a
            pattern holding a Unicode class. Issue #863 is what turns it on.

    Returns:
        The program, or the reason there is not one. A refusal is a value here
        rather than a raise for the same reason a parse failure is: the caller
        is deciding what to do about a pattern, and the two engines refuse
        different things.
    """
    var out = Program()
    var python = engine == ENGINE_PYTHON
    out.python = python
    if not tree.ok:
        # pandas gives Arrow the pattern as written, and RE2 has syntax Python
        # does not, so `\p{L}` is a working pattern upstream and unreachable
        # here. An RE2 front end is what closes this, and document 77 section 8
        # has it as the largest single gap in the component.
        #
        # For Python's engine this is not a gap at all. The parser is Python's
        # grammar, so a pattern it cannot read is a pattern Python cannot read
        # either, and the refusal is upstream's rather than this library's.
        out.ok = False
        out.problem = String("Python's grammar cannot read this pattern")
        out.gap = not python
        return out^
    if tree.re2_refuses and not python:
        # A comment group, a `\\u` escape, a `\\Z` that is not trailing and the
        # three others the parser records. Every one of them is syntax Python
        # reads and RE2 has never had, and the parser read all of them into the
        # nodes Python would have read them into, so there is nothing here to
        # refuse on Python's side. The flag is a statement about the other
        # engine and is only consulted when the other engine is the one asking.
        out.ok = False
        out.problem = String("RE2 has no such syntax")
        return out^
    if tree.re2_differs and not python:
        # `a{,2}` and `[[:alpha:]]`, which both engines read and read
        # differently. The tree is Python's reading, so again there is only one
        # engine with a problem and it is not the one this branch is skipped
        # for.
        out.ok = False
        out.problem = String("RE2 reads this syntax differently")
        out.gap = True
        return out^
    if tree.zed and python and minor < PYTHON_ZED_ESCAPE:
        # A spelling RE2 has always had and Python did not have until 3.14, so
        # this is the one refusal here that is neither about an engine nor
        # about a feature. The pattern is fine on Arrow, fine on a new enough
        # interpreter, and a `bad escape \\z` on one this project still
        # supports, and the only thing that can tell those apart is the number
        # the builder carries. Not a gap, because upstream refuses it too.
        out.ok = False
        out.problem = String("this Python has no \\z escape")
        return out^
    if tree.approximate:
        out.ok = False
        out.problem = String("a named character is not resolved yet")
        out.gap = True
        return out^

    var theirs = FLAG_LOCALE | FLAG_VERBOSE | FLAG_ASCII | FLAG_UNICODE
    var refused = _refused_flags_python(
        tree.flags
    ) if python else _refused_flags(tree.flags)
    if refused.byte_length() > 0:
        out.ok = False
        out.problem = refused^
        # For RE2 the four letters it has never had are a refusal and `(?i)` is
        # a gap. For Python the locale letter is the only refusal left and it is
        # one Python makes too, so nothing that reaches here on that engine is a
        # shortfall of this library any more. The expression is kept in the shape
        # that says so rather than written as a constant.
        out.gap = ((tree.flags & FLAG_LOCALE) == 0) if python else (
            (tree.flags & theirs) == 0
        )
        return out^

    # The same four letters are refused in a scoped group on RE2, so `(?x:a)` is
    # an error upstream exactly as `(?x)a` is. The three RE2 does have are
    # carried now, on an `OP_SCOPE` node, so nothing else about a scoped group
    # is refused on either engine.
    #
    # The Python half of this cannot fire and is kept anyway. `L` is the only
    # letter `_refused_flags_python` turns down and the parser turns it down
    # first, with Python's own sentence about a `str` pattern, so a locale flag
    # never reaches a tree at all. Writing the line as a constant would say
    # something different from the line above it about a question that is the
    # same question, and the day a sixth letter is refused here it would be the
    # line somebody forgot.
    var scoped_refused = _refused_flags_python(
        tree.scoped
    ) if python else _refused_flags(tree.scoped)
    if scoped_refused.byte_length() > 0:
        out.ok = False
        out.problem = scoped_refused^
        out.gap = ((tree.scoped & FLAG_LOCALE) == 0) if python else (
            (tree.scoped & theirs) == 0
        )
        return out^

    var b = _Builder(tree.flags, captures, python, minor)
    _check_node(b, tree.nodes, tree.root, MAX_REPEAT_COUNT)
    if b.failed:
        out.ok = False
        out.problem = b.problem.copy()
        out.gap = b.gap
        return out^
    if captures:
        _ = b.emit(IN_SAVE, 0, 0)
    _emit_node(b, tree.nodes, tree.root)
    if captures:
        _ = b.emit(IN_SAVE, 1, 0)
    _ = b.emit(IN_MATCH, 0, 0)
    if b.failed:
        out.ok = False
        out.problem = b.problem.copy()
        out.gap = b.gap
        return out^
    out.code = b.code.copy()
    out.ranges = b.ranges.copy()
    out.anchored = _anchored(Span(out.code))
    # After the range table has been copied, because the set is written onto the
    # end of it rather than into a second table. Nothing already in there moves,
    # so the offsets the instructions hold are the offsets they had.
    #
    # An anchored program is left without one. It starts no attempt above
    # position zero, so there is no position for the set to step over, and a
    # scan that has to ask whether it has one runs slower than a scan that
    # reads a count it knows will be zero.
    var first = List[Int32]()
    if not out.anchored:
        first = _first_ranges(Span(out.code), Span(out.ranges))
    if _first_worth_having(first):
        out.first_at = Int32(len(out.ranges))
        out.first_count = Int32(len(first) // 2)
        for i in range(len(first)):
            out.ranges.append(first[i])
    if alphabet:
        _fill_classes(out)
    out.groups = Int(tree.groups)
    out.slots = 2 * (Int(tree.groups) + 1) if captures else 0
    # The parser keeps the names and the numbers as two lists the length of
    # however many groups were named, and what a caller labelling columns wants
    # is one entry per group whether it was named or not.
    for group in range(1, Int(tree.groups) + 1):
        var label = String("")
        for i in range(len(tree.names)):
            if Int(tree.numbers[i]) == group:
                label = tree.names[i].copy()
                break
        out.labels.append(label^)
    return out^


def in_set(
    ranges: Span[Int32, _], at: Int32, count: Int32, point: UInt32
) -> Bool:
    """Whether a code point is in one of a set's ranges.

    A binary search rather than a walk, because the sets a Unicode class table
    will produce have hundreds of ranges in them and the ones a caller writes
    have two.

    Args:
        ranges: The whole program's range table.
        at: Where this set starts in it.
        count: How many ranges it has.
        point: The code point.

    Returns:
        True when the point is in the set.
    """
    var value = Int32(Int(point))
    var low = 0
    var high = Int(count) - 1
    while low <= high:
        var middle = (low + high) // 2
        var start = ranges[Int(at) + middle * 2]
        var stop = ranges[Int(at) + middle * 2 + 1]
        if value < start:
            high = middle - 1
        elif value > stop:
            low = middle + 1
        else:
            return True
    return False


def is_word_point(point: UInt32) -> Bool:
    """Whether a code point counts as a word character for `\\b`.

    RE2's word boundary is written against RE2's `\\w`, which is ASCII, so a
    boundary in a column of Greek is not where a reader of the pattern would
    put it. Measured rather than assumed, like the class itself.

    Args:
        point: The code point.

    Returns:
        True for a digit, a letter or an underscore, all ASCII.
    """
    if point >= 0x30 and point <= 0x39:
        return True
    if point >= 0x41 and point <= 0x5A:
        return True
    if point == 0x5F:
        return True
    return point >= 0x61 and point <= 0x7A


def is_word_point_unicode(point: UInt32, edges: Span[Int32, _]) -> Bool:
    """Whether a code point counts as a word character to Python's `\\b`.

    The same question as the one above and a different answer for 138495 code
    points, which is what document 81 is about. Python's boundary is written
    against Python's `\\w`, so this reads the generated table rather than four
    ranges of literals, and a boundary in a column of Greek is where a reader of
    the pattern would put it.

    The ranges arrive as a span rather than being read out of `classdata.mojo`
    here, because bringing that table out of the compiler's world costs a copy
    and this is asked once per position of every row. The machine holds one copy
    for as long as it is walking a column and lends it out.

    A binary search rather than the ASCII word and mask `charclass.mojo` uses,
    because a boundary is only asked about where a pattern wrote one, while a
    class question is asked of every character of every row.

    Args:
        point: The code point.
        edges: The ranges from `word_ranges_unicode`, low and high pairs.

    Returns:
        True for anything `str.isalnum` says yes to, and for the underscore.
    """
    var value = Int32(Int(point))
    var low = 0
    var high = len(edges) // 2 - 1
    while low <= high:
        var mid = (low + high) // 2
        if value < edges[mid * 2]:
            high = mid - 1
        elif value > edges[mid * 2 + 1]:
            low = mid + 1
        else:
            return True
    return False
