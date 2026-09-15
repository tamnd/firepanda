"""A parsed pattern turned into instructions something can run.

The shape is Thompson's, which is to say the program is a list of instructions
with two branching ones and no backtracking anywhere, and the reason for that
choice is the same reason RE2 made it: a pattern is written by a caller and run
against a column, and an engine that can take exponential time on a pattern like
`(a+)+b` is a denial of service with a friendly API in front of it.

Both engines are compiled here and the difference between them is three lines.
Python reads the three Perl classes as Unicode where RE2 reads them as ASCII,
Python asks the word boundary question against its own wider class, and Python's
dollar sign matches before a newline that ends the text where RE2's does not.
All three are settled while the pattern is being compiled, so the machine that
runs the program never learns which engine asked for it. Document 81 is where
the three were measured and says which methods take which engine.

What is still RE2 only is the constructs. Python has a lookaround, a
backreference, a conditional, an atomic group and a possessive quantifier and
this engine has none of them, so a pattern using one is refused for Python as a
gap here rather than as something Python cannot do.

The refusals are worth reading as a group, because they are not a list of things
that were too hard. Every one of them is a construct RE2 itself refuses, which
was measured rather than assumed: a lookaround, a backreference, a conditional,
an atomic group, a possessive quantifier, and the four inline flag letters out of
seven that RE2 has never heard of. A pattern this compiler turns down is a
pattern pyarrow turns down, and that correspondence is the thing
`tests/differential/regex_match.mojo` checks over the generated corpus.
"""

from std.collections.span import Span

from firepanda.kernel.regex.classdata import (
    DIGIT_RANGES,
    SPACE_RANGES,
    WORD_RANGES,
)
from firepanda.kernel.regex.parse import Parsed
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
    shortfall. Refusing `(?i)` because no case folding table has been written is
    a shortfall.

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

    def __init__(out self, flags: Int32, captures: Bool, python: Bool = False):
        """Starts an empty program.

        Args:
            flags: The pattern's global flags.
            captures: Whether to write the save instructions.
            python: Whether the program is for Python's engine.
        """
        self.code = []
        self.ranges = []
        self.failed = False
        self.problem = String("")
        self.gap = False
        self.flags = flags
        self.captures = captures
        self.python = python

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


def _category_ranges(which: Int32, python: Bool) -> List[Int32]:
    """What one of the six Perl classes means to whichever engine is running it.

    These are the measured sets rather than the documented ones. RE2's
    documentation writes `\\s` as `[\\t\\n\\f\\r ]` and the ASCII half of this
    agrees with it, which is worth saying because Python's `\\s` also holds a
    vertical tab and the documentation is the only place the two look alike.

    RE2's three are ASCII. That is the difference document 76 opens with, and
    it is the reason a column of Arabic Indic digits answers False to
    `str.contains(r"\\d")` and True once the same pattern picks up a lookahead.
    Python's three are Unicode and come out of `classdata.mojo`, which is what
    document 81 is about: the same letter in the same accessor covers 63
    characters or 138558 of them depending on which method was called.

    Args:
        which: The `CATEGORY_` value.
        python: Whether the program is being compiled for Python's engine.

    Returns:
        The ranges as low and high pairs, already in order.
    """
    var digits: Int32 = Int32(Int(CATEGORY_DIGIT))
    var not_digits: Int32 = Int32(Int(CATEGORY_NOT_DIGIT))
    var spaces: Int32 = Int32(Int(CATEGORY_SPACE))
    var not_spaces: Int32 = Int32(Int(CATEGORY_NOT_SPACE))
    if which == digits or which == not_digits:
        if python:
            var table = materialize[DIGIT_RANGES]()
            return _held(Span(table))
        var out: List[Int32] = [0x30, 0x39]
        return out^
    if which == spaces or which == not_spaces:
        if python:
            var table = materialize[SPACE_RANGES]()
            return _held(Span(table))
        var out: List[Int32] = [0x09, 0x0A, 0x0C, 0x0D, 0x20, 0x20]
        return out^
    if python:
        return word_ranges_unicode()
    var out: List[Int32] = [0x30, 0x39, 0x41, 0x5A, 0x5F, 0x5F, 0x61, 0x7A]
    return out^


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

    Args:
        b: The builder, told about anything that cannot be compiled.
        nodes: The arena.
        node: The `OP_IN` node.
        negated: Set to True when the class began with a caret.

    Returns:
        The ranges the class covers, sorted and merged, before negation.
    """
    var gathered = List[Int32]()
    var child = nodes[Int(node)].first
    while child >= 0:
        var it = nodes[Int(child)]
        if it.op == OP_NEGATE:
            negated = True
        elif it.op == OP_LITERAL:
            gathered.append(it.a)
            gathered.append(it.a)
        elif it.op == OP_RANGE:
            gathered.append(it.a)
            gathered.append(it.b)
        elif it.op == OP_CATEGORY:
            var pieces = _category_ranges(it.a, b.python)
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

    Args:
        b: The builder, for its flags and its engine.
        which: The `AT_` value the parser wrote.

    Returns:
        The `AT_` value to check at run time.
    """
    if b.python:
        if which == Int32(Int(AT_BOUNDARY)):
            return Int32(Int(AT_BOUNDARY_UNICODE))
        if which == Int32(Int(AT_NON_BOUNDARY)):
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

    if it.op == OP_LITERAL:
        _ = b.emit(IN_CHAR, it.a, 0)
        return
    if it.op == OP_NOT_LITERAL:
        var one: List[Int32] = [it.a, it.a]
        b.add_set(one, True)
        return
    if it.op == OP_ANY:
        if (b.flags & FLAG_DOTALL) != 0:
            _ = b.emit(IN_ANY_ALL, 0, 0)
        else:
            _ = b.emit(IN_ANY, 0, 0)
        return
    if it.op == OP_AT:
        _ = b.emit(IN_AT, _at_value(b, it.a), 0)
        return
    if it.op == OP_CATEGORY:
        var pieces = _sorted_merged(_category_ranges(it.a, b.python))
        b.add_set(pieces, _category_is_negated(it.a))
        return
    if it.op == OP_IN:
        var negated = False
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

    b.give_up(String("unsupported pattern"))


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
        _refuse_construct(b, String("lookaround"))
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
        _refuse_construct(b, String("empty negative lookaround"))
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

    `(?i)` is refused here and not by RE2, which does have it. Folding needs a
    table of which code points fold onto which, that table is not in this
    repository, and a pattern that silently did not fold would be a wrong
    answer rather than a missing feature.

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
    if (flags & FLAG_IGNORECASE) != 0:
        return String("case folding is not written yet")
    return String("")


def _refused_flags_python(flags: Int32) -> String:
    """The same question asked about Python's engine, which answers differently
    for every letter.

    Three of the four RE2 will not hear of are letters Python reads perfectly
    well, so refusing them here is this library falling short rather than
    agreeing with anybody. One of them is not refused at all: `(?u)` asks for
    what this engine already does, so taking it is more honest than turning it
    down for a reason that does not exist.

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
    if (flags & FLAG_VERBOSE) != 0:
        return String("verbose mode is not read yet")
    if (flags & FLAG_ASCII) != 0:
        return String("the ascii flag is not carried yet")
    if (flags & FLAG_IGNORECASE) != 0:
        return String("case folding is not written yet")
    return String("")


def compile_program(
    tree: Parsed, engine: UInt8, captures: Bool = False
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

    Returns:
        The program, or the reason there is not one. A refusal is a value here
        rather than a raise for the same reason a parse failure is: the caller
        is deciding what to do about a pattern, and the two engines refuse
        different things.
    """
    var out = Program()
    var python = engine == ENGINE_PYTHON
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
        # a gap. For Python only the locale letter is a refusal, since Python
        # turns that one down as well, and the rest are this library falling
        # short of an engine that reads them.
        out.gap = ((tree.flags & FLAG_LOCALE) == 0) if python else (
            (tree.flags & theirs) == 0
        )
        return out^

    # The same four letters are refused in a scoped group, so `(?x:a)` is an
    # error upstream exactly as `(?x)a` is. The other three are a gap and not an
    # error, because the parser reads a scoped group and throws the letters
    # away, and a program built from that tree would answer `(?i:b)` without
    # folding while RE2 folds. Carrying the flags on the node is what closes
    # this, and document 77 section 8 has it.
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
    if tree.scoped != 0:
        out.ok = False
        out.problem = String("a scoped flag group is not carried yet")
        out.gap = True
        return out^

    var b = _Builder(tree.flags, captures, python)
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
