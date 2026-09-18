"""Reading a pattern with Python's grammar.

### Why this grammar and not RE2's

pandas decides which of its two engines answers a call by handing the pattern to
`re._parser` and walking what comes back. So the parse that decides is Python's,
and a pattern Python cannot read goes to RE2 whatever RE2 thinks of it. That
makes Python's grammar the front end for both engines rather than one grammar of
two, and it makes the set of patterns this parser rejects part of the observable
behaviour rather than an implementation detail: every rejection here is a
pattern that must be routed to the RE2 side, and getting one wrong moves a call
to the wrong engine and changes its answer.

That is why a failure to parse is a value on the way out and not a raised error.
The caller above is not a user with a broken pattern, it is a router with a
question, and the answer to the question is one of three things rather than two:
it parsed and holds a lookaround, it parsed and does not, or it did not parse.

### Why a flat arena rather than a tree

The nodes live in one `List[Node]` and refer to each other by index, with a
first child and a next sibling on each. A tree of nested lists is the shape this
wants to be written as, and it is a shape Mojo makes expensive: a node owning a
`List` of children cannot be moved out of that list without copying the subtree
under it, and the parser does exactly that move every time a quantifier takes
the atom in front of it and puts it underneath a repeat.

The arena also happens to be what the compiler behind this wants. A program
counter is an index, so a walk that already holds indices does not have to
invent them.

### What is deliberately not here

No matching. This file answers what a pattern says and nothing about what it
does to a row, which is why it has no idea that the two engines disagree about
`\\d`. That disagreement belongs to whichever of them is handed the tree.

### Which of Python's collapses are reproduced, and which are not

`re._parser` is not a plain transcription of the pattern. It folds a class of
one literal into that literal, folds an alternation of single characters into a
class, drops a non capturing group by inlining its contents, and turns a
negative lookaround with an empty body into a node that never matches.

Those collapses are reproduced here only where they are visible from outside,
and what outside means for this file is narrow: whether a pattern parses, and
where it routes. Inlining a non capturing group is visible, because it lifts
whatever was inside it to a level the router walks. Collapsing `(?!)` is visible
and is the reason that pattern reaches RE2 and raises rather than being answered
as a column of False. Folding `[a]` into `a` is not visible, because no class
can hold a lookaround, so it is not done here, and the tree keeps the shape the
pattern was written in, which the compiler behind this would rather have.

That is also the honest statement of what this file is checked against. It is
not checked token for token against Python's tokens. It is checked on the two
answers a caller can see, over a corpus large enough that agreeing on both by
accident is not available.
"""

from std.collections.span import Span

from firepanda.kernel.regex.tokens import (
    AT_BEGINNING,
    AT_BEGINNING_STRING,
    AT_BOUNDARY,
    AT_END,
    AT_END_STRING,
    AT_NON_BOUNDARY,
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
    OP_ANY,
    OP_ASSERT,
    OP_ASSERT_NOT,
    OP_AT,
    OP_ATOMIC_GROUP,
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
    OP_POSSESSIVE_REPEAT,
    OP_RANGE,
    OP_SCOPE,
    OP_SEQ,
    OP_SUBPATTERN,
    Node,
)


comptime NO_NODE: Int32 = -1
"""What a missing child or a missing sibling holds. Minus one rather than zero
because zero is a real node index, and the root is usually at it."""

comptime NOTHING: Int32 = -2
"""What a reader answers when it read something that leaves no node behind.

Two constructs do that and both matter. A comment produces nothing, and so does
a global flag group, and Python treats the two identically in the two places it
can be noticed: a quantifier after one of them repeats whatever came before it
rather than repeating the comment, so `a(?#c)+` is `a+`, and a global flag group
is still at the start of the pattern after a comment, so `(?#c)(?i)a` is read
and `(?:)(?i)a` is not.

Modelling that as a node with no children would have got both of those wrong,
which is why this is a third answer rather than an empty node.
"""


struct Parsed(Movable):
    """A pattern read, or the reason it could not be.

    `ok` is the only field a caller may read without checking anything first.
    When it is False the arena is whatever the parser had built when it gave up,
    which is not a tree and must not be walked, and `problem` says why in
    Python's own words where Python has words for it.

    `groups` counts the capturing groups, which is what a backreference is
    checked against while the pattern is still being read, since `\\1` in a
    pattern with no group in front of it is a parse error in Python rather than
    a reference that never matches.
    """

    var nodes: List[Node]
    """The arena. Empty when nothing parsed."""

    var root: Int32
    """The node the pattern is, which is a `SEQ` or a `BRANCH`."""

    var groups: Int32
    """How many capturing groups the pattern opened."""

    var names: List[String]
    """The names of the named groups, in the order they were opened, so that
    `(?P=name)` and `(?(name)...)` can be resolved to a number while reading."""

    var numbers: List[Int32]
    """Which group each of those names belongs to, in the same order.

    The two lists are what Python calls `groupindex` written the other way
    round, and they are both here because a name on its own does not say which
    group it is. `(a)(?P<L>b)` opens two groups and names one of them, so
    `names` is one long and the number it holds is two, and anything labelling
    columns has to know that rather than guess from the position in the list."""

    var ok: Bool
    """Whether it parsed at all."""

    var problem: String
    """Why it did not, and empty when it did."""

    var approximate: Bool
    """Whether the tree holds a node this parser could not read exactly.

    One thing sets it, which is `\\N{NAME}`: the name is not resolved because
    there is no Unicode name table here yet, and the node left behind holds a
    placeholder code point rather than the character the caller named. That is
    enough for the router, which only asks whether the pattern parses, and it is
    not enough for anything that matches text, so the flag is here for the
    matching engine to refuse on rather than left for somebody to discover as a
    pattern that matches the wrong character in silence."""

    var re2_refuses: Bool
    """Whether the pattern holds syntax RE2 will not take at all.

    Several things set it, and all of them are Python syntax RE2 has never had,
    so a caller writing one of them has written a pattern pandas hands to Arrow
    and Arrow rejects. A comment group, `\\uXXXX`, `\\UXXXXXXXX`, a `\\Z` that
    is not the last thing in the pattern, which pandas rewrites to RE2's `\\z`
    only when it is trailing, a backslash in front of a character outside ASCII,
    and a one digit octal escape inside a class.

    The last two are the ones worth stating, because both look like nothing. A
    backslash in front of a character Python does not know an escape for is that
    character to Python, so `\\漢` is a perfectly ordinary way to write one, and
    RE2 refuses every backslash it does not recognise whatever follows it. A
    one digit octal escape inside a class is `[\\1]`, which is the character
    with code one to Python, and RE2 will not read a nonzero octal escape of
    fewer than two digits because that is how it tells one from a
    backreference. `[\\01]` and `[\\12]` are the same character to both.

    This is a fact about the text rather than about the tree, which is why it is
    recorded here: the parser is the only thing that sees a comment, and by the
    time the tree exists the comment has left nothing behind.
    """

    var re2_differs: Bool
    """Whether the pattern holds syntax RE2 reads differently rather than
    refuses.

    Two things set it, and the pair is worth reading together because they are
    the reason this flag is not folded into the one above. A count with no lower
    bound, as in `a{,2}`, is `a{0,2}` to Python and the five literal characters
    `a{,2}` to RE2. A POSIX class, as in `[[:alpha:]]`, is a set holding a
    bracket and a colon and four letters to Python and every letter there is to
    RE2.

    Neither raises anywhere. A pattern setting this flag and answered out of
    this tree would give a column of booleans that looks exactly like a right
    one, which is why the compiler refuses instead, and why the refusal is
    counted as a gap rather than as agreement with an RE2 that refuses nothing.
    """

    var flags: Int32
    """The global flags the pattern turned on, as `FLAG_` bits.

    Only the global form reaches here, because only the global form applies to
    the whole pattern. What the scoped form turned on and off is in `scoped`.

    A caller can start this off with flags the pattern never wrote, which is
    what the accessor's `flags` argument and its `case` argument are. Upstream
    does the same thing by a different route, compiling the pattern with the
    argument and handing the compiled object on, and the two arrive at one
    pattern carrying one set of flags either way.
    """

    var scoped: Int32
    """Every flag any scoped group mentioned, whether it turned it on or off.

    The group itself is an `OP_SCOPE` node carrying the same two sets, and that
    is what the compiler acts on. This field is the flat reading of the whole
    pattern, which answers a different question: was this letter written
    anywhere at all. RE2 needs that question answered, because it reads
    `(?i:a)` and has never heard of `(?x:a)` or `(?a:a)` in any position, so a
    letter it refuses has to be found without knowing where it sat.

    Turning a flag off counts the same as turning it on. `(?-x:a)` is a letter
    RE2 refuses just as much as `(?x:a)` is, since the refusal is about the
    letter being in the pattern rather than about what it was asked to do.
    """

    def __init__(out self):
        """Starts an empty parse, which is what a caller gets for a pattern that
        failed on its first character."""
        self.nodes = []
        self.root = NO_NODE
        self.groups = 0
        self.names = []
        self.numbers = []
        self.ok = True
        self.problem = String("")
        self.approximate = False
        self.re2_refuses = False
        self.re2_differs = False
        self.flags = 0
        self.scoped = 0


struct _Cursor(Movable):
    """The parser's whole state: the pattern, where it is, and the arena.

    One struct rather than several arguments passed down because every function
    below needs all of it, and because a parser that carries its arena in a
    parameter list is one refactor away from carrying two arenas.
    """

    var points: List[UInt32]
    """The pattern decoded into code points. Python's parser works in code
    points and so do the classes, so decoding once at the top is both what
    matches and what stops every character class from having to decode again."""

    var at: Int
    """How many code points have been read."""

    var nodes: List[Node]
    """The arena being built."""

    var groups: Int32
    """How many capturing groups have been opened so far, which is also the
    number the next one gets."""

    var names: List[String]
    """The named groups in the order they were opened."""

    var numbers: List[Int32]
    """Which group each of those names belongs to, since a named group and an
    unnamed one can be interleaved and a name's position in the list is
    therefore not its number."""

    var open_groups: List[Int32]
    """The groups whose closing bracket has not been read yet. A backreference
    to one of these is a parse error in Python rather than a reference that
    never matches, so the list is carried rather than inferred from the count."""

    var depth: Int
    """How many brackets deep the cursor is, which only the global flag rule
    reads, since that rule is about being at the top."""

    var produced: Bool
    """Whether anything has been attached at the top level. The other half of
    the global flag rule: `(?i)` is allowed at the start of a pattern and
    nowhere else, and the start means nothing has been read yet rather than
    character zero, so `(?i)(?s)a` is fine and `a(?i)b` is not."""

    var failed: Bool
    """Whether the parse has given up. Checked at every loop head so that a
    failure deep in a class does not have to unwind through six returns."""

    var problem: String
    """The reason, set once and never overwritten, so the message names the
    first thing that went wrong rather than the last."""

    var lookbehind: Int32
    """How many groups had been opened when the outermost lookbehind that the
    cursor is inside was entered, and minus one when it is inside none.

    A lookbehind cannot refer to a group written inside itself, because the
    engine reads a lookbehind by trying it at a position it has not matched
    forwards through yet. The outermost one wins rather than the innermost,
    which is Python's rule and is why a nested lookbehind does not widen what
    the outer one may refer to.
    """

    var pending: List[Int32]
    """The group numbers a conditional asked about, to be checked at the end.

    Python checks a backreference against the groups seen so far and checks a
    conditional against the groups seen by the end of the pattern, so `\\1(a)`
    is refused and `(?(1)a)(b)` is read. The asymmetry looks like an oversight
    and is not something to correct, since correcting it moves patterns between
    engines.
    """

    var guessed: Bool
    """Whether a `\\N{NAME}` was met, which is the one thing read
    approximately. Carried on the cursor rather than worked out afterwards
    because the placeholder node is indistinguishable from a caller writing the
    replacement character on purpose."""

    var re2_refuses: Bool
    """Whether a comment group, a `\\u` escape, a `\\U` escape, a `\\Z` that is
    not trailing, a backslash before a character outside ASCII or a one digit
    octal escape in a class has been read. What `Parsed.re2_refuses` ends up
    holding."""

    var re2_differs: Bool
    """Whether a count with no lower bound or a POSIX class has been read. What
    `Parsed.re2_differs` ends up holding."""

    var scoped: Int32
    """Which flags a scoped group mentioned, on or off. What `Parsed.scoped`
    ends up holding."""

    var flagged: Int32
    """Which global flags the pattern turned on, as a bit per letter.

    Only needed for one thing: turning on both the ASCII flag and the Unicode
    flag is a conflict Python reports at the very end of the parse, and reports
    by raising `ValueError` rather than a parse error, which pandas does not
    catch. Document 76 section 8 has the consequence.
    """

    def __init__(out self, var points: List[UInt32]):
        """Starts a parse over a decoded pattern.

        Args:
            points: The pattern in code points. Consumed.
        """
        self.points = points^
        self.at = 0
        self.nodes = []
        self.groups = 0
        self.names = []
        self.numbers = []
        self.open_groups = []
        self.depth = 0
        self.produced = False
        self.failed = False
        self.problem = String("")
        self.lookbehind = -1
        self.pending = []
        self.re2_refuses = False
        self.re2_differs = False
        self.scoped = 0
        self.flagged = 0
        self.guessed = False

    def done(self) -> Bool:
        """Whether the cursor is at the end of the pattern.

        Returns:
            True when nothing is left to read.
        """
        return self.at >= len(self.points)

    def peek(self) -> UInt32:
        """The next code point without consuming it.

        Returns:
            The code point, or a value no pattern can hold when the cursor is at
            the end, so that a caller comparing against a character does not
            have to check `done` first.
        """
        if self.done():
            return 0xFFFFFFFF
        return self.points[self.at]

    def ahead(self, n: Int) -> UInt32:
        """The code point n places on, without consuming anything.

        Args:
            n: How far ahead, where zero is the same as `peek`.

        Returns:
            The code point, or a value no pattern can hold when that is past the
            end.
        """
        if self.at + n >= len(self.points):
            return 0xFFFFFFFF
        return self.points[self.at + n]

    def take(mut self) -> UInt32:
        """Reads one code point and moves on.

        Returns:
            The code point, or the past the end value.
        """
        var point = self.peek()
        self.at += 1
        return point

    def give_up(mut self, var why: String):
        """Records the first failure and stops the parse.

        Args:
            why: The reason, in Python's words where Python has words for it.
                Consumed.
        """
        if not self.failed:
            self.failed = True
            self.problem = why^

    def add(mut self, op: UInt8, a: Int32, b: Int32) -> Int32:
        """Puts a node in the arena with no children and no sibling.

        Args:
            op: Which node.
            a: Its first payload, whose meaning depends on the op.
            b: Its second.

        Returns:
            Where it went.
        """
        self.nodes.append(Node(op, a, b, NO_NODE, NO_NODE, NO_NODE))
        return Int32(len(self.nodes) - 1)

    def attach(mut self, parent: Int32, child: Int32):
        """Puts a node on the end of a parent's children.

        Kept to one place because the list is singly linked with a tail pointer
        and getting that wrong shows up as a pattern that parses and is missing
        its last alternative, which no error reports.

        Args:
            parent: The node gaining a child.
            child: The node being attached, which must have no sibling.
        """
        var tail = self.nodes[Int(parent)].last
        if tail == NO_NODE:
            self.nodes[Int(parent)].first = child
        else:
            self.nodes[Int(tail)].next = child
        self.nodes[Int(parent)].last = child


def decoded(pattern: StringSlice) -> List[UInt32]:
    """Reads a pattern's bytes as code points.

    Args:
        pattern: The pattern.

    Returns:
        Its code points.
    """
    var out = List[UInt32]()
    decode_into(pattern.as_bytes(), out)
    return out^


def decode_into(bytes: Span[UInt8, _], mut out: List[UInt32]):
    """Reads bytes as code points, into a list the caller keeps.

    Text that is not valid UTF-8 cannot be written in Python source and Arrow
    says a string column holds none, so a bad byte is taken as a single code
    point of its own value rather than reported. That keeps the cursor in step
    with the bytes and lets whatever is wrong with the pattern be reported by
    the grammar instead.

    The list is emptied first, so a column can hand the same one back row after
    row and pay for the allocation once. That matters more than it looks: the
    row is the unit here, a column has millions of them, and a list per row is
    a malloc per row for a few dozen bytes of text.

    Args:
        bytes: The text. Borrowed for the length of the call and not stored.
        out: Where to put the code points. Emptied first.
    """
    out.clear()
    var i = 0
    while i < len(bytes):
        var lead = UInt32(Int(bytes[i]))
        if lead < 0x80:
            out.append(lead)
            i += 1
        elif lead >= 0xF0 and i + 3 < len(bytes):
            out.append(
                ((lead & 0x07) << 18)
                | ((UInt32(Int(bytes[i + 1])) & 0x3F) << 12)
                | ((UInt32(Int(bytes[i + 2])) & 0x3F) << 6)
                | (UInt32(Int(bytes[i + 3])) & 0x3F)
            )
            i += 4
        elif lead >= 0xE0 and i + 2 < len(bytes):
            out.append(
                ((lead & 0x0F) << 12)
                | ((UInt32(Int(bytes[i + 1])) & 0x3F) << 6)
                | (UInt32(Int(bytes[i + 2])) & 0x3F)
            )
            i += 3
        elif lead >= 0xC0 and i + 1 < len(bytes):
            out.append(
                ((lead & 0x1F) << 6) | (UInt32(Int(bytes[i + 1])) & 0x3F)
            )
            i += 2
        else:
            out.append(lead)
            i += 1


def _is_digit(point: UInt32) -> Bool:
    """Whether a code point is an ASCII digit.

    ASCII rather than Unicode on purpose. These are the digits a repeat count
    and a group number are written with, and Python's parser reads those with
    `isdigit` on a `str` character, which for the characters that reach here is
    the same question.

    Args:
        point: The code point.

    Returns:
        True for `0` through `9`.
    """
    return point >= 0x30 and point <= 0x39


def _is_verbose_space(point: UInt32) -> Bool:
    """Whether a code point is one verbose mode throws away.

    The six Python names in `WHITESPACE`, written out rather than asked of any
    general whitespace test, because the set is Python's and not Unicode's and
    the two differ on characters a caller can write. The file separator at
    `0x1c` is the one that shows it: Python's `str.isspace` says yes and verbose
    mode keeps it, so a pattern holding one means something.

    Args:
        point: The code point.

    Returns:
        True for a space, a tab, a newline, a vertical tab, a form feed or a
        carriage return.
    """
    if point == 0x20 or point == 0x09 or point == 0x0A:
        return True
    return point == 0x0B or point == 0x0C or point == 0x0D


def _is_word(point: UInt32) -> Bool:
    """Whether a code point can appear in a group name.

    Python requires a group name to be a valid identifier, which is a wider
    question than this, and the difference only shows up for names nobody
    writes. What this has to get right is where a name ends, which is at the
    first character that is not a letter, a digit or an underscore.

    Args:
        point: The code point.

    Returns:
        True for an ASCII letter, an ASCII digit, an underscore, or anything
        outside ASCII, which Python treats as a letter here.
    """
    if _is_digit(point) or point == 0x5F:
        return True
    if point >= 0x41 and point <= 0x5A:
        return True
    if point >= 0x61 and point <= 0x7A:
        return True
    return point >= 0x80


def _hex_value(point: UInt32) -> Int32:
    """A hex digit's value, or minus one.

    Args:
        point: The code point.

    Returns:
        Zero to fifteen, or minus one when it is not a hex digit.
    """
    if _is_digit(point):
        return Int32(Int(point) - 0x30)
    if point >= 0x41 and point <= 0x46:
        return Int32(Int(point) - 0x41 + 10)
    if point >= 0x61 and point <= 0x66:
        return Int32(Int(point) - 0x61 + 10)
    return -1


def _octal_value(point: UInt32) -> Int32:
    """An octal digit's value, or minus one.

    Args:
        point: The code point.

    Returns:
        Zero to seven, or minus one.
    """
    if point >= 0x30 and point <= 0x37:
        return Int32(Int(point) - 0x30)
    return -1


def _category_for(point: UInt32) -> Int32:
    """Which Perl class a letter after a backslash names, or minus one.

    Args:
        point: The letter.

    Returns:
        The category, or minus one when the letter names none.
    """
    if point == 0x64:
        return Int32(Int(CATEGORY_DIGIT))
    if point == 0x44:
        return Int32(Int(CATEGORY_NOT_DIGIT))
    if point == 0x73:
        return Int32(Int(CATEGORY_SPACE))
    if point == 0x53:
        return Int32(Int(CATEGORY_NOT_SPACE))
    if point == 0x77:
        return Int32(Int(CATEGORY_WORD))
    if point == 0x57:
        return Int32(Int(CATEGORY_NOT_WORD))
    return -1


def _simple_escape(point: UInt32) -> Int32:
    """The character a one letter escape stands for, or minus one.

    These are the six Python shares with C, and they are handled here rather
    than in the escape reader so that a class and a sequence agree about them
    without the rule being written twice.

    Args:
        point: The letter after the backslash.

    Returns:
        The code point it means, or minus one.
    """
    if point == 0x61:
        return 0x07
    if point == 0x62:
        return 0x08
    if point == 0x66:
        return 0x0C
    if point == 0x6E:
        return 0x0A
    if point == 0x72:
        return 0x0D
    if point == 0x74:
        return 0x09
    if point == 0x76:
        return 0x0B
    return -1


def _is_ascii_letter(point: UInt32) -> Bool:
    """Whether a code point is an ASCII letter.

    Args:
        point: The code point.

    Returns:
        True for `A` to `Z` and `a` to `z`.
    """
    if point >= 0x41 and point <= 0x5A:
        return True
    return point >= 0x61 and point <= 0x7A


def _fixed_hex(mut c: _Cursor, width: Int, what: String) -> Int32:
    """Reads exactly so many hex digits after `\\x`, `\\u` or `\\U`.

    Python requires the full width and reports a bad escape rather than reading
    what it can, so a short one is a parse failure and therefore a pattern that
    routes to the other engine.

    Args:
        c: The cursor, moved past the digits when they are all there.
        width: How many digits are required.
        what: The escape's letter, for the message.

    Returns:
        The code point, or minus one when the parse failed.
    """
    var value: Int32 = 0
    for _ in range(width):
        var digit = _hex_value(c.peek())
        if digit < 0:
            c.give_up(String("incomplete escape \\") + what)
            return -1
        value = value * 16 + digit
        c.at += 1
    if value > 0x10FFFF:
        c.give_up(String("bad escape \\") + what)
        return -1
    return value


def _escape(mut c: _Cursor, in_class: Bool) -> Int32:
    """Reads what follows a backslash and builds the node it means.

    The one difference between a class and a sequence is what a digit means.
    Outside a class `\\1` is a backreference, and inside one it can only be an
    octal character, because there is nothing in a class for a group to be
    referred from. Python enforces that by reading `\\1` in a class as octal only
    when the digits make a valid octal escape and refusing it otherwise, which is
    why `[\\1]` is fine and `[\\8]` is a bad escape.

    Args:
        c: The cursor, sitting on the character after the backslash.
        in_class: Whether this is inside square brackets.

    Returns:
        The node, or minus one when the parse failed.
    """
    if c.done():
        c.give_up(String("bad escape (end of pattern)"))
        return -1
    var point = c.take()

    var category = _category_for(point)
    if category >= 0:
        return c.add(OP_CATEGORY, category, 0)

    if not in_class:
        if point == 0x41:
            return c.add(OP_AT, Int32(Int(AT_BEGINNING_STRING)), 0)
        if point == 0x5A or point == 0x7A:
            # pandas rewrites a trailing `\\Z` into RE2's `\\z` on the way to
            # Arrow and leaves one anywhere else alone, and RE2 has no `\\Z`,
            # so `a\\Zb` is a pattern Python reads and Arrow rejects. `\\z`
            # is RE2's own spelling and needs no rewriting wherever it sits.
            if point == 0x5A and not c.done():
                c.re2_refuses = True
            return c.add(OP_AT, Int32(Int(AT_END_STRING)), 0)
        if point == 0x62:
            return c.add(OP_AT, Int32(Int(AT_BOUNDARY)), 0)
        if point == 0x42:
            return c.add(OP_AT, Int32(Int(AT_NON_BOUNDARY)), 0)

    if in_class and point == 0x62:
        # A backspace to Python and an invalid escape to RE2, which reads `\\b`
        # as a word boundary everywhere and will not have one inside brackets.
        c.re2_refuses = True
        return c.add(OP_LITERAL, 0x08, 0)

    var simple = _simple_escape(point)
    if simple >= 0:
        return c.add(OP_LITERAL, simple, 0)

    if point == 0x78:
        var value = _fixed_hex(c, 2, String("x"))
        if value < 0:
            return -1
        return c.add(OP_LITERAL, value, 0)
    if point == 0x75 or point == 0x55:
        # RE2 has `\\x41` and `\\x{41}` and neither of these two, so a
        # pattern naming a character this way reads here and is refused there.
        c.re2_refuses = True
        var width = 4 if point == 0x75 else 8
        var what = String("u") if point == 0x75 else String("U")
        var value = _fixed_hex(c, width, what)
        if value < 0:
            return -1
        return c.add(OP_LITERAL, value, 0)

    if point == 0x4E:
        return _named_character(c)

    if point == 0x30:
        return _octal_escape(c, 0)

    if _is_digit(point):
        return _digit_escape(c, point, in_class)

    if _is_ascii_letter(point):
        c.give_up(String("bad escape"))
        return -1

    if point >= 0x80:
        # Python's rule is that a backslash in front of anything that is not an
        # ASCII letter or digit is that thing, so `\\漢` is one way to write a
        # character and `\\-` is the usual way to write one inside a class. RE2
        # takes the second and refuses the first: a backslash it does not
        # recognise is an error there, and it recognises nothing outside ASCII.
        c.re2_refuses = True

    return c.add(OP_LITERAL, Int32(Int(point)), 0)


def _named_character(mut c: _Cursor) -> Int32:
    """Reads `\\N{NAME}`.

    The name is not resolved. Resolving it would mean carrying the whole Unicode
    name table for the sake of a pattern almost nobody writes, and the router
    does not need the answer: what it needs is whether the pattern parses, and
    what decides that is the braces rather than what is between them. A name
    that does not exist is a Python error this reads as a valid pattern, which
    means such a pattern routes to the Python side here and raises in pandas.
    That is a real difference and it is written down in document 76 rather than
    hidden, because the fix is a table and the table is not worth it yet.

    Args:
        c: The cursor, sitting on the character after the `N`.

    Returns:
        The node, or minus one.
    """
    if c.peek() != 0x7B:
        c.give_up(String("missing {"))
        return -1
    c.at += 1
    var seen = 0
    while not c.done() and c.peek() != 0x7D:
        c.at += 1
        seen += 1
    if c.done():
        c.give_up(String("missing }, unterminated name"))
        return -1
    c.at += 1
    if seen == 0:
        c.give_up(String("missing character name"))
        return -1
    c.guessed = True
    return c.add(OP_LITERAL, 0xFFFD, 0)


def _octal_escape(mut c: _Cursor, var value: Int32) -> Int32:
    """Reads up to two more octal digits after one that has been read.

    Three digits starting from a zero cannot reach the ceiling, so the range
    check that belongs with this is in the two callers that can.

    Args:
        c: The cursor.
        value: The value of the digit already read.

    Returns:
        The literal node.
    """
    for _ in range(2):
        var digit = _octal_value(c.peek())
        if digit < 0:
            break
        value = value * 8 + digit
        c.at += 1
    return c.add(OP_LITERAL, value, 0)


def _digit_escape(mut c: _Cursor, first: UInt32, in_class: Bool) -> Int32:
    """Reads a backslash followed by a digit that is not zero.

    Python's own comment on this rule is "octal escape or decimal group
    reference (sigh)", and the rule is worth stating exactly because guessing at
    it gets several patterns wrong. One more digit is taken if there is one.
    Then, and only if both digits taken so far are octal and the next character
    is an octal digit too, a third is taken and the three are an octal
    character. Anything else is a group reference made of the one or two digits.

    That is why `\\777` is an octal escape and out of range, `\\778` is a
    reference to group 77, and `\\788` is a reference to group 78. A rule of
    "three digits means octal" would read the last two the same as the first and
    would be wrong about both.

    Inside a class there is nothing for a group to be referred from, so the
    digits are octal or the escape is bad.

    Args:
        c: The cursor, past the first digit.
        first: The digit already read.
        in_class: Whether this is inside square brackets.

    Returns:
        The node, or minus one.
    """
    if in_class:
        var digit = _octal_value(first)
        if digit < 0:
            c.give_up(String("bad escape"))
            return -1
        var value = digit
        var taken = 1
        for _ in range(2):
            var more = _octal_value(c.peek())
            if more < 0:
                break
            value = value * 8 + more
            c.at += 1
            taken += 1
        if taken == 1:
            # `[\\1]` is the character with code one to Python and an error to
            # RE2, which will not read a nonzero octal escape shorter than two
            # digits because that is how it tells one from a backreference it
            # does not have. `[\\01]` and `[\\12]` are the same character to
            # both, so the refusal is about the number of digits rather than
            # about the value.
            c.re2_refuses = True
        if value > 0xFF:
            c.give_up(String("octal escape value outside of range 0-0o377"))
            return -1
        return c.add(OP_LITERAL, value, 0)

    var digits = List[UInt32]()
    digits.append(first)
    if _is_digit(c.peek()):
        digits.append(c.take())
        if (
            _octal_value(digits[0]) >= 0
            and _octal_value(digits[1]) >= 0
            and _octal_value(c.peek()) >= 0
        ):
            digits.append(c.take())
            var value: Int32 = 0
            for i in range(3):
                value = value * 8 + _octal_value(digits[i])
            if value > 0xFF:
                c.give_up(String("octal escape value outside of range 0-0o377"))
                return -1
            return c.add(OP_LITERAL, value, 0)

    var number: Int32 = 0
    for i in range(len(digits)):
        number = number * 10 + Int32(Int(digits[i]) - 0x30)
    if number == 0 or number > c.groups:
        c.give_up(String("invalid group reference"))
        return -1
    if _open(c, number):
        c.give_up(String("cannot refer to an open group"))
        return -1
    if not _behind_allows(c, number):
        return -1
    return c.add(OP_GROUPREF, number, 0)


def _open(c: _Cursor, number: Int32) -> Bool:
    """Whether a group has been opened and not yet closed.

    Args:
        c: The cursor.
        number: The group number.

    Returns:
        True when the closing bracket has not been read.
    """
    for i in range(len(c.open_groups)):
        if c.open_groups[i] == number:
            return True
    return False


def _behind_allows(mut c: _Cursor, number: Int32) -> Bool:
    """Whether a lookbehind the cursor is inside may refer to a group.

    A lookbehind is matched by trying it at a position the engine has not read
    forwards through, so a group written inside one has not taken part by the
    time a reference to it is reached. Python refuses that rather than letting
    the reference fail to match, and it refuses it with two different messages
    depending on whether the group exists at all, which is reproduced because
    the second of them fires for a conditional whose group is written later in
    the pattern.

    Args:
        c: The cursor.
        number: The group being referred to.

    Returns:
        True when the reference is allowed, and False with the parse given up
        when it is not.
    """
    if c.lookbehind < 0:
        return True
    if number > c.groups or _open(c, number):
        c.give_up(String("cannot refer to an open group"))
        return False
    if number > c.lookbehind:
        c.give_up(
            String(
                "cannot refer to group defined in the same lookbehind"
                " subpattern"
            )
        )
        return False
    return True


def _class(mut c: _Cursor) -> Int32:
    """Reads a character class, with the cursor past the opening bracket.

    Three rules here are the ones an implementation written from memory gets
    wrong, and all three are about a character that is a metacharacter almost
    everywhere else standing for itself. A `]` immediately after the opening
    bracket, or immediately after the negating caret, is a literal bracket. A
    `-` at either end of the class is a literal hyphen. And a class is the one
    place where nothing terminates except the closing bracket, so a `[` inside
    one is an ordinary character and does not open anything.

    The fourth rule is a refusal rather than a permission: a range whose end is
    a Perl class, as in `[\\d-z]`, is an error and not a hyphen, because Python
    has already decided a range is being written by the time it finds out what
    is on the other side of it.

    Args:
        c: The cursor, just past the `[`.

    Returns:
        The `OP_IN` node, or minus one.
    """
    var node = c.add(OP_IN, 0, 0)
    if c.peek() == 0x5E:
        c.at += 1
        var negate = c.add(OP_NEGATE, 0, 0)
        c.attach(node, negate)

    var first = True
    while True:
        if c.failed:
            return -1
        if c.done():
            c.give_up(String("unterminated character set"))
            return -1
        if c.peek() == 0x5D and not first:
            c.at += 1
            return node
        first = False
        if c.peek() == 0x5B and c.ahead(1) == 0x3A:
            # A POSIX class to RE2 and a bracket, a colon and some letters to
            # Python, which warns about it and reads it anyway. The two readings
            # have nothing in common, and neither of them is an error.
            c.re2_differs = True

        var left = _class_item(c)
        if left < 0:
            return -1

        if c.peek() == 0x2D and c.ahead(1) != 0x5D and c.ahead(1) != 0xFFFFFFFF:
            c.at += 1
            var right = _class_item(c)
            if right < 0:
                return -1
            if (
                c.nodes[Int(left)].op != OP_LITERAL
                or c.nodes[Int(right)].op != OP_LITERAL
            ):
                c.give_up(String("bad character range"))
                return -1
            var low = c.nodes[Int(left)].a
            var high = c.nodes[Int(right)].a
            if low > high:
                c.give_up(String("bad character range"))
                return -1
            var span = c.add(OP_RANGE, low, high)
            c.attach(node, span)
        else:
            c.attach(node, left)


def _class_item(mut c: _Cursor) -> Int32:
    """Reads one thing out of a character class.

    Args:
        c: The cursor.

    Returns:
        The node, or minus one.
    """
    if c.peek() == 0x5C:
        c.at += 1
        return _escape(c, True)
    return c.add(OP_LITERAL, Int32(Int(c.take())), 0)


def _counted(mut c: _Cursor) -> List[Int32]:
    """Reads a `{m,n}` quantifier, or says it is not one.

    A brace that does not open a well formed count is an ordinary character in
    Python rather than an error, so `a{}` is three literals and `a{2` is three
    literals, and this reports that by leaving the cursor where it found it. The
    answer is a list rather than a tuple because taking a Mojo tuple apart copies
    what comes out of it.

    Args:
        c: The cursor, sitting on the `{`.

    Returns:
        An empty list when this is not a quantifier, and otherwise the lower
        bound followed by the upper one.
    """
    var mark = c.at
    c.at += 1
    var low: Int32 = 0
    var saw_low = False
    while _is_digit(c.peek()):
        low = low * 10 + Int32(Int(c.take()) - 0x30)
        saw_low = True
        if low > MAXREPEAT:
            low = MAXREPEAT
    var high = low
    var saw_high = saw_low
    if c.peek() == 0x2C:
        c.at += 1
        high = 0
        saw_high = False
        while _is_digit(c.peek()):
            high = high * 10 + Int32(Int(c.take()) - 0x30)
            saw_high = True
            if high > MAXREPEAT:
                high = MAXREPEAT
        if not saw_high:
            high = MAXREPEAT
            saw_high = True
    if c.peek() != 0x7D or not (saw_low or saw_high):
        c.at = mark
        return List[Int32]()
    c.at += 1
    if not saw_low:
        # Python has read `{,n}` as `{0,n}` since 3.11 and RE2 reads it as the
        # characters it is made of, so `a{,2}` matches everything upstream of
        # here and matches the text `a{,2}` downstream of it.
        c.re2_differs = True
    var out = List[Int32]()
    out.append(low)
    out.append(high)
    return out^


def _atom(mut c: _Cursor) -> Int32:
    """Reads one thing that a quantifier could be attached to.

    The three quantifier characters do not appear here. A quantifier is read by
    the sequence rather than by the atom, because what it repeats is the last
    thing the sequence holds and not the thing just read, and those differ
    whenever something in between produced nothing. A brace does appear here,
    and reaching it means the sequence has already decided it does not open a
    count, so it is an ordinary character.

    Args:
        c: The cursor.

    Returns:
        The node, `NOTHING` when what was read leaves none, or minus one.
    """
    var point = c.take()
    if point == 0x2E:
        return c.add(OP_ANY, 0, 0)
    if point == 0x5E:
        return c.add(OP_AT, Int32(Int(AT_BEGINNING)), 0)
    if point == 0x24:
        return c.add(OP_AT, Int32(Int(AT_END)), 0)
    if point == 0x5B:
        return _class(c)
    if point == 0x5C:
        return _escape(c, False)
    if point == 0x28:
        return _group(c)
    return c.add(OP_LITERAL, Int32(Int(point)), 0)


def _name(mut c: _Cursor, closer: UInt32) -> String:
    """Reads a group name up to a closing character.

    Args:
        c: The cursor, on the first character of the name.
        closer: What ends it, which is `>` for a definition and `)` for a use.

    Returns:
        The name, and the empty string when the parse failed.
    """
    var points = List[UInt32]()
    while not c.done() and c.peek() != closer:
        var point = c.take()
        if not _is_word(point):
            c.give_up(String("bad character in group name"))
            return String("")
        points.append(point)
    if c.done():
        c.give_up(String("missing >, unterminated name"))
        return String("")
    c.at += 1
    if len(points) == 0:
        c.give_up(String("missing group name"))
        return String("")
    if _is_digit(points[0]):
        c.give_up(String("bad character in group name"))
        return String("")
    var bytes = List[UInt8]()
    for i in range(len(points)):
        _put_point(bytes, points[i])
    return String(StringSlice(unsafe_from_utf8=Span(bytes)))


def _put_point(mut into: List[UInt8], point: UInt32):
    """Writes one code point out as UTF-8, for building a group name.

    Args:
        into: The bytes, appended.
        point: The code point.
    """
    if point < 0x80:
        into.append(point.cast[DType.uint8]())
    elif point < 0x800:
        into.append((0xC0 | (point >> 6)).cast[DType.uint8]())
        into.append((0x80 | (point & 0x3F)).cast[DType.uint8]())
    elif point < 0x10000:
        into.append((0xE0 | (point >> 12)).cast[DType.uint8]())
        into.append((0x80 | ((point >> 6) & 0x3F)).cast[DType.uint8]())
        into.append((0x80 | (point & 0x3F)).cast[DType.uint8]())
    else:
        into.append((0xF0 | (point >> 18)).cast[DType.uint8]())
        into.append((0x80 | ((point >> 12) & 0x3F)).cast[DType.uint8]())
        into.append((0x80 | ((point >> 6) & 0x3F)).cast[DType.uint8]())
        into.append((0x80 | (point & 0x3F)).cast[DType.uint8]())


def _numbered(c: _Cursor, name: String) -> Int32:
    """Which group a name belongs to, or minus one.

    Args:
        c: The cursor, holding the names opened so far.
        name: The name being looked up.

    Returns:
        The group number, or minus one when no group has that name.
    """
    for i in range(len(c.names)):
        if c.names[i] == name:
            return c.numbers[i]
    return -1


def _body(mut c: _Cursor, node: Int32) -> Bool:
    """Parses a group's contents and the bracket that closes it.

    Args:
        c: The cursor.
        node: The node the contents hang under.

    Returns:
        Whether it worked.
    """
    c.depth += 1
    var inner = _branch(c)
    c.depth -= 1
    if inner < 0:
        return False
    if c.peek() != 0x29:
        c.give_up(String("missing ), unterminated subpattern"))
        return False
    c.at += 1
    c.attach(node, inner)
    return True


def _group(mut c: _Cursor) -> Int32:
    """Reads everything that can follow an opening bracket.

    Python has eleven things that can, and the reason they are all here rather
    than spread out is that the first two characters decide between them and
    getting that decision wrong sends a pattern to the wrong engine rather than
    producing a wrong answer, which is a failure that looks like a passing test.

    Args:
        c: The cursor, just past the `(`.

    Returns:
        The node, or minus one.
    """
    if c.peek() != 0x3F:
        c.groups += 1
        var number = c.groups
        c.open_groups.append(number)
        var node = c.add(OP_SUBPATTERN, number, 0)
        var made = _body(c, node)
        _ = c.open_groups.pop()
        if not made:
            return -1
        return node

    c.at += 1
    if c.done():
        c.give_up(String("unexpected end of pattern"))
        return -1
    var kind = c.take()

    if kind == 0x3A:
        c.depth += 1
        var inner = _branch(c)
        c.depth -= 1
        if inner < 0:
            return -1
        if c.peek() != 0x29:
            c.give_up(String("missing ), unterminated subpattern"))
            return -1
        c.at += 1
        return inner

    if kind == 0x3D or kind == 0x21:
        return _lookaround(c, 1, kind == 0x21)

    if kind == 0x3C:
        var after = c.take()
        if after == 0x3D or after == 0x21:
            return _lookaround(c, -1, after == 0x21)
        c.give_up(String("unknown extension ?<"))
        return -1

    if kind == 0x3E:
        var node = c.add(OP_ATOMIC_GROUP, 0, 0)
        if not _body(c, node):
            return -1
        return node

    if kind == 0x23:
        # RE2 has no comment group at all, so a pattern with one in it is
        # handed to Arrow by pandas and refused there. Recorded now because by
        # the time the tree exists the comment has left nothing behind.
        c.re2_refuses = True
        # A backslash hides the character after it, so `(?#\)` is not closed by
        # that bracket and runs off the end. Nothing else in a comment means
        # anything, and this does not because the reader looked at it: Python
        # reads a comment through the same tokenizer as everything else, and
        # that tokenizer hands back an escape as one token. The rule is an
        # accident of the reader rather than a decision, and it is copied here
        # because a caller writing a Windows path in a comment meets it.
        while not c.done() and c.peek() != 0x29:
            c.at += 2 if c.peek() == 0x5C else 1
        if c.done():
            c.give_up(String("missing ), unterminated comment"))
            return -1
        c.at += 1
        return NOTHING

    if kind == 0x50:
        return _named(c)

    if kind == 0x28:
        return _conditional(c)

    c.at -= 1
    return _flags(c)


def _lookaround(mut c: _Cursor, direction: Int32, negative: Bool) -> Int32:
    """Reads a lookahead or a lookbehind.

    The collapse Python does here is the one that matters most in this file. A
    negative lookaround with nothing in it always fails, so Python writes it out
    as a node that never matches rather than as an assertion, and since the
    router looks for assertions by name the pattern then routes to RE2, which
    has no idea what to do with the brackets it was handed. Reproducing the
    collapse is what makes `(?!)` raise here for the same reason it raises
    there, and leaving it out would have this library answer a column of False
    where pandas raises.

    Args:
        c: The cursor, just past the `=` or the `!`.
        direction: 1 for a lookahead and minus one for a lookbehind.
        negative: Whether it is the negative form.

    Returns:
        The node, or minus one.
    """
    var op = OP_ASSERT_NOT if negative else OP_ASSERT
    var node = c.add(op, direction, 0)
    var was = c.lookbehind
    if direction < 0 and was < 0:
        c.lookbehind = c.groups
    var made = _body(c, node)
    if direction < 0 and was < 0:
        c.lookbehind = -1
    if not made:
        return -1
    if negative:
        var inner = c.nodes[Int(node)].first
        if inner != NO_NODE and c.nodes[Int(inner)].first == NO_NODE:
            if c.nodes[Int(inner)].op == OP_SEQ:
                return c.add(OP_FAILURE, 0, 0)
    return node


def _named(mut c: _Cursor) -> Int32:
    """Reads the three things that can follow `(?P`.

    Args:
        c: The cursor, just past the `P`.

    Returns:
        The node, or minus one.
    """
    var next = c.take()
    if next == 0x3C:
        var name = _name(c, 0x3E)
        if c.failed:
            return -1
        if _numbered(c, name) >= 0:
            c.give_up(String("redefinition of group name"))
            return -1
        c.groups += 1
        var number = c.groups
        c.names.append(name)
        c.numbers.append(number)
        c.open_groups.append(number)
        var node = c.add(OP_SUBPATTERN, number, 0)
        var made = _body(c, node)
        _ = c.open_groups.pop()
        if not made:
            return -1
        return node

    if next == 0x3D:
        var name = _name(c, 0x29)
        if c.failed:
            return -1
        var number = _numbered(c, name)
        if number < 0:
            c.give_up(String("unknown group name"))
            return -1
        if _open(c, number):
            c.give_up(String("cannot refer to an open group"))
            return -1
        if not _behind_allows(c, number):
            return -1
        return c.add(OP_GROUPREF, number, 0)

    c.give_up(String("unknown extension ?P"))
    return -1


def _conditional(mut c: _Cursor) -> Int32:
    """Reads `(?(1)yes|no)`, which asks whether a group took part.

    Args:
        c: The cursor, just past the second `(`.

    Returns:
        The node, or minus one.
    """
    var number: Int32
    if _is_digit(c.peek()):
        var value: Int32 = 0
        while _is_digit(c.peek()):
            value = value * 10 + Int32(Int(c.take()) - 0x30)
        if c.peek() != 0x29:
            c.give_up(String("missing ), unterminated name"))
            return -1
        c.at += 1
        if value == 0:
            c.give_up(String("bad group number"))
            return -1
        c.pending.append(value)
        number = value
    else:
        var name = _name(c, 0x29)
        if c.failed:
            return -1
        number = _numbered(c, name)
        if number < 0:
            c.give_up(String("unknown group name"))
            return -1
    if not _behind_allows(c, number):
        return -1

    var node = c.add(OP_GROUPREF_EXISTS, number, 0)
    c.depth += 1
    var yes = _seq(c)
    if yes < 0:
        c.depth -= 1
        return -1
    c.attach(node, yes)
    if c.peek() == 0x7C:
        c.at += 1
        var no = _seq(c)
        if no < 0:
            c.depth -= 1
            return -1
        c.attach(node, no)
        if c.peek() == 0x7C:
            c.depth -= 1
            c.give_up(String("conditional backref with more than two branches"))
            return -1
    c.depth -= 1
    if c.peek() != 0x29:
        c.give_up(String("missing ), unterminated subpattern"))
        return -1
    c.at += 1
    return node


def _flag_bit(point: UInt32) -> Int32:
    """Which flag a letter is, as a bit, or minus one.

    Args:
        point: The letter.

    Returns:
        The bit, or minus one when the letter is not a flag.
    """
    if point == 0x69:
        return FLAG_IGNORECASE
    if point == 0x4C:
        return FLAG_LOCALE
    if point == 0x6D:
        return FLAG_MULTILINE
    if point == 0x73:
        return FLAG_DOTALL
    if point == 0x78:
        return FLAG_VERBOSE
    if point == 0x61:
        return FLAG_ASCII
    if point == 0x75:
        return FLAG_UNICODE
    return -1


comptime TYPE_FLAGS: Int32 = FLAG_LOCALE | FLAG_ASCII | FLAG_UNICODE
"""The three flags that say which alphabet the pattern is written against.

They are the ones that cannot be combined with each other and cannot be turned
off, and they are grouped here rather than checked one at a time because Python
checks them as a set and the messages say so.
"""


def _flag_letters(mut c: _Cursor, mut add: Int32) -> UInt32:
    """Reads the flags being turned on and answers what ended them.

    Args:
        c: The cursor, on the first letter.
        add: The flags so far, added to.

    Returns:
        The character that ended the run, which is one of `)`, `-` and `:`, and
        the past the end value when the parse failed.
    """
    while True:
        var point = c.peek()
        if point == 0x4C:
            c.give_up(
                String(
                    "bad inline flags: cannot use 'L' flag with a str pattern"
                )
            )
            return 0xFFFFFFFF
        var flag = _flag_bit(point)
        add |= flag
        if (flag & TYPE_FLAGS) != 0 and (add & TYPE_FLAGS) != flag:
            c.give_up(
                String(
                    "bad inline flags: flags 'a', 'u' and 'L' are incompatible"
                )
            )
            return 0xFFFFFFFF
        c.at += 1
        if c.done():
            c.give_up(String("missing -, : or )"))
            return 0xFFFFFFFF
        var next = c.peek()
        if next == 0x29 or next == 0x2D or next == 0x3A:
            c.at += 1
            return next
        if _flag_bit(next) < 0:
            if _is_ascii_letter(next):
                c.give_up(String("unknown flag"))
            else:
                c.give_up(String("missing -, : or )"))
            return 0xFFFFFFFF


def _flags_off(mut c: _Cursor, mut off: Int32) -> Bool:
    """Reads the flags being turned off, which must end at a colon.

    Args:
        c: The cursor, just past the `-`.
        off: The flags so far, added to.

    Returns:
        Whether it worked.
    """
    if c.done():
        c.give_up(String("missing flag"))
        return False
    if _flag_bit(c.peek()) < 0:
        if _is_ascii_letter(c.peek()):
            c.give_up(String("unknown flag"))
        else:
            c.give_up(String("missing flag"))
        return False
    while True:
        var flag = _flag_bit(c.peek())
        if (flag & TYPE_FLAGS) != 0:
            c.give_up(
                String(
                    "bad inline flags: cannot turn off flags 'a', 'u' and 'L'"
                )
            )
            return False
        off |= flag
        c.at += 1
        if c.done():
            c.give_up(String("missing :"))
            return False
        var next = c.peek()
        if next == 0x3A:
            c.at += 1
            return True
        if _flag_bit(next) < 0:
            if _is_ascii_letter(next):
                c.give_up(String("unknown flag"))
            else:
                c.give_up(String("missing :"))
            return False


def _flags(mut c: _Cursor) -> Int32:
    """Reads `(?i)` and `(?i:...)` and refuses everything else.

    The two forms are not the same thing wearing different brackets. The scoped
    one is a group and the router walks into it, which is why `(?i:(?=a))`
    routes to Python. The global one produces nothing at all and is only allowed
    at the very start of a pattern, which is a rule Python added in 3.11 and
    which makes `a(?i)b` a parse failure and therefore an RE2 pattern.

    The third difference is the one that is easy to miss. Flags can only be
    turned off in the scoped form, so `(?-i)` and `(?i-s)` are both parse
    failures and neither of them is a global flag group with a minus sign in
    it. That means a caller writing either of those has written an RE2 pattern
    without meaning to.

    Args:
        c: The cursor, on the first flag letter or on the minus sign.

    Returns:
        The node, `NOTHING` for the global form, or minus one.
    """
    var add: Int32 = 0
    var off: Int32 = 0
    var closer = c.peek()
    if closer != 0x2D and _flag_bit(closer) < 0:
        c.give_up(String("unknown extension ?"))
        return -1
    if closer == 0x2D:
        c.at += 1
    else:
        closer = _flag_letters(c, add)
        if c.failed:
            return -1

    if closer == 0x29:
        if c.depth != 0 or c.produced:
            c.give_up(String("global flags not at the start of the expression"))
            return -1
        c.flagged |= add
        return NOTHING

    if closer == 0x2D:
        if not _flags_off(c, off):
            return -1

    if (add & off) != 0:
        c.give_up(String("bad inline flags: flag turned on and off"))
        return -1
    # The letters are still recorded on the parse as well as on the node, and
    # the two are for different readers. The node is what the compiler acts on.
    # The set is what says a letter was written anywhere in the pattern, which
    # is what RE2 has to be told, since RE2 reads `(?i:a)` and has never heard
    # of `(?x:a)` or `(?a:a)` in any position.
    c.scoped |= add | off

    # Verbose mode is spent here rather than passed on, because it decides what
    # the characters inside the group mean and the reading of them happens
    # below this line. Everything else on the node is a question about a
    # character that the compiler asks later. Saving the outer value rather
    # than clearing the bits afterwards is what makes nesting work, since the
    # group this one sits inside may have turned the same letter the other way.
    #
    # The three alphabet letters go into the set too and nothing here reads
    # them, since which alphabet a class comes out of is settled while the
    # program is built. So the rule that naming one of the three clears all
    # three is in the compiler and only there, rather than written twice in two
    # places where one of the two could never be seen to be wrong.
    var outer = c.flagged
    c.flagged = (c.flagged | add) & ~off
    c.depth += 1
    var inner = _branch(c)
    c.depth -= 1
    c.flagged = outer
    if inner < 0:
        return -1
    if c.peek() != 0x29:
        c.give_up(String("missing ), unterminated subpattern"))
        return -1
    c.at += 1
    var node = c.add(OP_SCOPE, add, off)
    c.attach(node, inner)
    return node


def _seq(mut c: _Cursor) -> Int32:
    """Reads items until an alternation bar, a closing bracket or the end.

    The quantifiers are read here rather than next to the atom, and the reason
    is one measured rule: what a quantifier repeats is the last item this
    sequence holds. A comment leaves no item, so `a(?#c)+` repeats the `a`, and
    a quantifier with an empty sequence in front of it has nothing to repeat
    even though something was read. Reading the quantifier next to the atom
    gets the first of those wrong and cannot express the second.

    Two refusals come out of the same place. A quantifier on an anchor is
    nothing to repeat, which is why `^*` is refused and `(?=a)*` is not, and
    that pair is the whole reason the second one reaches RE2 and raises there. A
    quantifier on a quantifier is multiple repeat, except for the `?` and the
    `+` that make the one in front lazy or possessive.

    Verbose mode lives here too, for the same reason and in Python's own place
    for it, which is the top of this loop and nowhere below it. Document 88 is
    about what follows from that.

    Args:
        c: The cursor.

    Returns:
        The `OP_SEQ` node, or minus one.
    """
    var node = c.add(OP_SEQ, 0, 0)
    var before_last = NO_NODE
    while True:
        if c.failed:
            return -1
        if c.done():
            return node

        if (c.flagged & FLAG_VERBOSE) != 0:
            # The whole of verbose mode, and it is here rather than anywhere
            # else because this is where Python puts it: the skip happens at the
            # top of the item loop and nowhere the item loop calls into. That is
            # why `[a b]` still matches a space, why `a{1, 2}` is not a repeat,
            # and why `a * ?` is a multiple repeat rather than a lazy one. The
            # class loop, the counted scan and the peek for a lazy marker are
            # all separate reads and none of them skips anything.
            #
            # The flag is read off the cursor each time round rather than once
            # before the loop, because `(?x)` is itself an item and turns it on
            # partway through the pass. Python restarts the whole parse when it
            # meets one, which reaches the same answer for the only patterns
            # that can hold one: a global flag group has to come before anything
            # that produced a node, so the only thing that can sit behind it is
            # another group of the same kind or a comment.
            var here = c.peek()
            if _is_verbose_space(here):
                c.at += 1
                continue
            if here == 0x23:
                while not c.done():
                    var got = c.peek()
                    c.at += 1
                    if got == 0x0A:
                        break
                continue

        var point = c.peek()
        if point == 0x7C or point == 0x29:
            return node

        var low: Int32 = 0
        var high: Int32 = 0
        var repeating = False
        if point == 0x2A:
            c.at += 1
            high = MAXREPEAT
            repeating = True
        elif point == 0x2B:
            c.at += 1
            low = 1
            high = MAXREPEAT
            repeating = True
        elif point == 0x3F:
            c.at += 1
            high = 1
            repeating = True
        elif point == 0x7B:
            var bounds = _counted(c)
            if len(bounds) != 0:
                low = bounds[0]
                high = bounds[1]
                repeating = True
                if low > high:
                    c.give_up(String("min repeat greater than max repeat"))
                    return -1

        if repeating:
            var last = c.nodes[Int(node)].last
            if last == NO_NODE:
                c.give_up(String("nothing to repeat"))
                return -1
            var was = c.nodes[Int(last)].op
            if was == OP_AT:
                c.give_up(String("nothing to repeat"))
                return -1
            if (
                was == OP_MAX_REPEAT
                or was == OP_MIN_REPEAT
                or was == OP_POSSESSIVE_REPEAT
            ):
                c.give_up(String("multiple repeat"))
                return -1
            var kind = OP_MAX_REPEAT
            if c.peek() == 0x3F:
                c.at += 1
                kind = OP_MIN_REPEAT
            elif c.peek() == 0x2B:
                c.at += 1
                kind = OP_POSSESSIVE_REPEAT
            var repeat = c.add(kind, low, high)
            c.nodes[Int(last)].next = NO_NODE
            c.attach(repeat, last)
            if before_last == NO_NODE:
                c.nodes[Int(node)].first = repeat
            else:
                c.nodes[Int(before_last)].next = repeat
            c.nodes[Int(node)].last = repeat
            continue

        var item = _atom(c)
        if item == NOTHING:
            continue
        if item < 0:
            return -1
        before_last = c.nodes[Int(node)].last
        c.attach(node, item)
        if c.depth == 0:
            c.produced = True


def _branch(mut c: _Cursor) -> Int32:
    """Reads a sequence and any alternatives after it.

    A pattern with no bar in it comes back as the sequence itself rather than as
    a branch of one, which is what Python does and is worth keeping because the
    router walks branches and a branch of one would be a node that exists only
    to be walked through.

    Args:
        c: The cursor.

    Returns:
        The node, or minus one.
    """
    var first = _seq(c)
    if first < 0:
        return -1
    if c.peek() != 0x7C:
        return first
    if c.depth == 0:
        # An alternative that produced nothing is still an alternative, which
        # is why `|(?i)a` is refused where `(?#c)(?i)a` is read.
        c.produced = True
    var node = c.add(OP_BRANCH, 0, 0)
    c.attach(node, first)
    while c.peek() == 0x7C:
        c.at += 1
        var other = _seq(c)
        if other < 0:
            return -1
        c.attach(node, other)
    return node


def _harvested(var c: _Cursor, root: Int32) -> Parsed:
    """Turns a finished cursor into the answer, taking its arena rather than
    copying it.

    A free function taking the cursor by value rather than a method on it,
    because the arena and the names move out of it here and a value being taken
    apart has to be one nothing else can still be holding.

    Args:
        c: The cursor, consumed.
        root: The node the parse ended up at.

    Returns:
        The tree, or the reason there is not one.
    """
    var gave_up = c.failed or root < 0
    var finished = c.done()
    var groups = c.groups
    var guessed = c.guessed
    var flagged = c.flagged
    var refuses = c.re2_refuses
    var differs = c.re2_differs
    var scoped = c.scoped
    var nodes = c.nodes.copy()
    var names = c.names.copy()
    var numbers = c.numbers.copy()
    var problem = c.problem.copy()
    var out = Parsed()
    if gave_up:
        out.ok = False
        out.problem = problem^
        return out^
    if not finished:
        out.ok = False
        out.problem = String("unbalanced parenthesis")
        return out^
    out.root = root
    out.groups = groups
    out.approximate = guessed
    out.flags = flagged
    out.re2_refuses = refuses
    out.re2_differs = differs
    out.scoped = scoped
    out.nodes = nodes^
    out.names = names^
    out.numbers = numbers^
    return out^


def _settle(mut c: _Cursor):
    """The two checks Python leaves until the whole pattern has been read.

    A conditional's group number is checked here rather than where it was
    written, which is why `(?(1)a)(b)` parses and `\\1(a)` does not, and the two
    being different is Python's and not a choice made here.

    The flag conflict is checked here too, and it is the one place in this file
    where reproducing Python exactly is not possible. Python reports it by
    raising `ValueError` rather than a parse error, and pandas catches only
    parse errors, so a pattern turning on both alphabets takes the whole call
    down with an error naming Python's parser. This reads it as a pattern that
    did not parse, which routes it to Arrow. Document 76 section 8 says why that
    is the better of the two wrong answers available.

    Args:
        c: The cursor, after the pattern has been read.
    """
    if c.failed:
        return
    for i in range(len(c.pending)):
        if c.pending[i] > c.groups:
            c.give_up(String("invalid group reference"))
            return
    if (c.flagged & FLAG_ASCII) != 0 and (c.flagged & FLAG_UNICODE) != 0:
        c.give_up(String("ASCII and UNICODE flags are incompatible"))


def parse_pattern(pattern: StringSlice, flags: Int32 = 0) -> Parsed:
    """Reads a pattern with Python's grammar.

    The flags are seeded rather than merged afterwards, so that a letter passed
    as an argument and the same letter written `(?i)` at the front of the
    pattern are the same fact by the time anything reads them. Seeding is safe
    because nothing in this file reads the field before the pattern is walked
    and the only write to it is an or, so a flag turned on here stays on.

    It also means the two checks that look at flags see the argument. A pattern
    passed both alphabets, one written and one argued, is refused here the same
    way a pattern writing both is, which is the answer upstream gives as well
    even though it gives it in a different exception.

    Args:
        pattern: The pattern as the caller wrote it.
        flags: Flags the caller passed beside the pattern rather than inside it,
            as `FLAG_` bits. Zero is the ordinary call.

    Returns:
        The tree, or the reason there is not one. A failure is a value here
        rather than a raise because the caller above is a router deciding which
        engine answers, and a pattern Python cannot read is one RE2 is given.
    """
    var c = _Cursor(decoded(pattern))
    c.flagged = flags
    var root = _branch(c)
    _settle(c)
    return _harvested(c^, root)
