"""Reading a pattern with RE2's grammar, to find out whether RE2 would.

### Why there has to be a second grammar at all

`firepanda/kernel/regex/parse.mojo` reads Python's grammar and that is the one
that decides which engine a call reaches, because pandas hands every pattern to
`re._parser` first and sends whatever it refuses to Arrow. So Python's refusals
are this library's routing table and none of that changes here.

What changes is what happens next. A pattern Python refuses has been sent to
Arrow, and Arrow answers it with RE2, and RE2 has its own opinion. Of the
thirty thousand generated patterns in the corpus, nineteen thousand six hundred
and fifteen are ones Python's grammar cannot read, and of those Arrow refuses
seventeen thousand nine hundred and seventy six and takes one thousand six
hundred and thirty nine. Until this file existed all nineteen thousand came back
from firepanda as the same `NotImplementedError`, and for nine out of ten of
them that was the wrong answer twice over: the right answer was a refusal, and
the refusal pandas gives is a `pyarrow.lib.ArrowInvalid`, which is a
`ValueError` and which a caller catching `ValueError` therefore catches.

So this file is not a second front end for matching. It is the answer to one
question, asked only about patterns Python has already refused: would RE2 take
this. A no is a refusal a caller can see and is now the same kind of refusal
pandas gives. A yes is still a gap, and is now a gap of one thousand six
hundred rather than nineteen thousand, which is the number that says what a real
RE2 front end would still have to be written to reach.

### Why it says yes when it is not sure

The cost of the two mistakes is not the same. Saying RE2 refuses a pattern it
takes turns a column somebody could have had into an exception, which is a new
wrong answer. Saying RE2 takes a pattern it refuses leaves the caller exactly
where they already were, which is the `NotImplementedError` this file is here to
reduce. So every rule below refuses only what it is sure of.

For a while that meant one construct was not judged at all. The Unicode class
`\\p{...}` was read as something RE2 takes even when it was plainly malformed,
because telling a name RE2 knows from one it does not needs a table of script
and category names and there was none. That table is in `unicodedata.mojo` now,
measured against the same RE2 this file is written against, so the construct is
judged like everything else and there is no longer any way for this file to be
unsure. The bias is still the rule every new rule is written under; it just has
nothing left that it applies to.

### What RE2's grammar actually is, and how that was found out

By asking it. Every rule in this file was measured against the RE2 that Arrow
is built with, through `pyarrow.compute.match_substring_regex`, which is the
same code path pandas reaches. Document 101 has the measurements and the
surprises, of which the ones worth knowing before reading the code are these.

RE2 takes a flag group anywhere in a pattern, so `a(?i)b` is a pattern it reads
and Python refuses. It takes `(?)`, a flag group naming no flags. It has four
flag letters and one of them, `U`, is not a letter Python has ever had. It lets
an assertion be repeated, so `^*` and `\\b*` are patterns it reads. It reads
`a{,2}` as five literal characters, and reads any `{` that does not open a well
formed count as a literal. It allows a `]` as the first thing in a class. It
has `\\C`, which matches one byte, and `\\Q...\\E`, which quotes a run.

And it refuses a great deal Python allows: every backreference, every
lookaround, every conditional, the comment group, `\\Z`, `\\u`, `\\U`, a
backslash in front of a character outside ASCII, and a one digit octal escape.

### The eleven things it says

RE2 has eleven ways of refusing a pattern and every one of them turns up in the
corpus. They are, with how many of the thirty thousand each accounts for:
invalid perl operator 12818, invalid escape sequence 4286, no argument for
repetition operator 1501, bad repetition operator 634, missing ] 542, invalid
named capture group 523, unexpected ) 442, missing ) 392, invalid repetition
size 242, invalid character class range 224 and trailing backslash 109. Each of
them is written here as a sentence of this library's own, in the voice the
other refusals in this component already use, rather than as a copy of RE2's
wording. A caller reading the message is a caller of firepanda.
"""

from firepanda.kernel.regex.parse import _put_point, decoded
from firepanda.kernel.regex.unicodedata import unicode_index


comptime RE2_MAX_REPEAT: Int = 1000
"""The largest count RE2 will take, and the budget a nest of counts shares.

Two rules rather than one, and the second is the one nobody expects. A single
count may not name a number over a thousand, so `a{1001}` is refused. And a
count inside a count multiplies, so `(a{100}){11}` is refused as well even
though neither number is anywhere near the limit, because eleven hundred copies
of `a` is what it would take to write the pattern out. `(a{100}){10}` is exactly
a thousand and is taken, and `(a{31}){32}` is nine hundred and ninety two and is
taken, which is how the rule was pinned down to a product rather than to a
depth.
"""


comptime _KIND_CHAR: Int = 0
"""What was last read inside a class is one character, so a `-` after it opens a
range."""

comptime _KIND_CLASS: Int = 1
"""What was last read inside a class is a set of characters, so a `-` after it
is a literal hyphen. `[\\d-a]` is three items and `[a-\\d]` is a refusal."""


struct Re2Read(Movable):
    """Whether RE2 would read a pattern, and why not when it would not."""

    var ok: Bool
    """Whether RE2 takes it."""

    var problem: String
    """Why not, in this library's own words, and empty when it would."""

    def __init__(out self):
        """Starts at a pattern RE2 takes, which is what a caller gets for the
        empty pattern."""
        self.ok = True
        self.problem = String("")


struct _Cursor(Movable):
    """Where the reader is and what it has noticed.

    The same shape the Python grammar's cursor has and much smaller, because
    this reader builds nothing. There is no arena, no group numbering and no
    flag state, since none of that is needed to answer whether a pattern reads.
    """

    var points: List[UInt32]
    """The pattern decoded into code points, because every rule below is about
    characters and a byte of a longer one must not be mistaken for one."""

    var at: Int
    """How many code points have been read."""

    var failed: Bool
    """Whether a refusal has been found. Checked at every loop head so that a
    refusal deep in a class does not have to unwind through five returns."""

    var problem: String
    """The refusal, set once and never overwritten, so the message names the
    first thing RE2 would have stopped at rather than the last."""

    var repeatable: Bool
    """Whether there is anything here for a repeat to repeat.

    RE2 builds a pattern on a stack and a repeat takes whatever is on top of it,
    so the question is not what was written last but whether anything has been
    left there. Three things leave nothing: the start of the pattern, the bar,
    and the opening of a group, all of which put a marker on the stack that a
    repeat will not take. A flag group leaves nothing either, and that is the
    surprise: rather than making a repeat after it illegal it makes the repeat
    reach past it, so `(?i)*` is a refusal at the front of a pattern and
    `a(?i)*` repeats the `a`. An assertion does leave something, so `^*` and
    `\\b*` are ordinary repeats that Python calls nothing to repeat.
    """

    var kind: Int
    """Whether the last item read inside a class was one character or a set."""

    var point: UInt32
    """Which character it was, when it was one, for checking a range runs the
    right way."""

    def __init__(out self, var points: List[UInt32]):
        """Starts a read over a decoded pattern.

        Args:
            points: The pattern in code points. Consumed.
        """
        self.points = points^
        self.at = 0
        self.failed = False
        self.problem = String("")
        self.repeatable = False
        self.kind = _KIND_CHAR
        self.point = 0

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

    def ahead(self, by: Int) -> UInt32:
        """A code point further on without consuming anything.

        Args:
            by: How many past the next one to look.

        Returns:
            The code point, or a value no pattern can hold when that is past the
            end.
        """
        if self.at + by >= len(self.points):
            return 0xFFFFFFFF
        return self.points[self.at + by]

    def take(mut self) -> UInt32:
        """The next code point, consumed.

        Returns:
            The code point, or a value no pattern can hold at the end.
        """
        var out = self.peek()
        self.at += 1
        return out

    def give_up(mut self, said: String):
        """Records a refusal, keeping the first one.

        Args:
            said: Why RE2 would not read it.
        """
        if self.failed:
            return
        self.failed = True
        self.problem = said.copy()


def _is_digit(point: UInt32) -> Bool:
    """Whether a code point is an ASCII decimal digit.

    Args:
        point: The code point.

    Returns:
        True for zero through nine.
    """
    return point >= UInt32(ord("0")) and point <= UInt32(ord("9"))


def _is_octal(point: UInt32) -> Bool:
    """Whether a code point is an ASCII octal digit.

    Args:
        point: The code point.

    Returns:
        True for zero through seven.
    """
    return point >= UInt32(ord("0")) and point <= UInt32(ord("7"))


def _is_hex(point: UInt32) -> Bool:
    """Whether a code point is an ASCII hexadecimal digit.

    Args:
        point: The code point.

    Returns:
        True for a digit and for either case of the first six letters.
    """
    if _is_digit(point):
        return True
    if point >= UInt32(ord("a")) and point <= UInt32(ord("f")):
        return True
    return point >= UInt32(ord("A")) and point <= UInt32(ord("F"))


def _is_name_point(point: UInt32) -> Bool:
    """Whether a code point may appear in an RE2 group name.

    The ASCII word characters and everything outside ASCII, which is wider than
    it used to be and was measured rather than remembered: `(?P<é>a)` is a group
    RE2 names and `(?P<n n>a)` and `(?P<n->a)` are refusals.

    Args:
        point: The code point.

    Returns:
        True when the character may be in a name.
    """
    if _is_digit(point):
        return True
    if point >= UInt32(ord("a")) and point <= UInt32(ord("z")):
        return True
    if point >= UInt32(ord("A")) and point <= UInt32(ord("Z")):
        return True
    if point == UInt32(ord("_")):
        return True
    return point >= 0x80 and point != 0xFFFFFFFF


def _is_flag_point(point: UInt32) -> Bool:
    """Whether a code point is one of RE2's four flag letters.

    Four rather than Python's seven, and three of the four are letters Python
    has as well and spends the same way, which are `i`, `m` and `s`. The fourth
    is `U`, which is RE2's alone and swaps greedy for ungreedy, and which is a
    capital where every Python flag letter is lower case, so `(?U)` is a group
    Python refuses outright.

    Args:
        point: The code point.

    Returns:
        True for one of the four.
    """
    return (
        point == UInt32(ord("i"))
        or point == UInt32(ord("m"))
        or point == UInt32(ord("s"))
        or point == UInt32(ord("U"))
    )


def _posix_names() -> List[String]:
    """The classes RE2 spells `[:name:]`.

    Returns:
        The fourteen names, without the brackets and without the `^` that may
        stand in front of any of them.
    """
    return [
        String("alnum"),
        String("alpha"),
        String("ascii"),
        String("blank"),
        String("cntrl"),
        String("digit"),
        String("graph"),
        String("lower"),
        String("print"),
        String("punct"),
        String("space"),
        String("upper"),
        String("word"),
        String("xdigit"),
    ]


def _escape_is_class(point: UInt32) -> Bool:
    """Whether an escape letter stands for a set of characters.

    It matters in one place only, which is what a `-` after it means. A set
    cannot be one end of a range, so the hyphen after one is a hyphen.

    Args:
        point: The letter after the backslash.

    Returns:
        True for the six Perl classes.
    """
    return (
        point == UInt32(ord("d"))
        or point == UInt32(ord("s"))
        or point == UInt32(ord("w"))
        or point == UInt32(ord("D"))
        or point == UInt32(ord("S"))
        or point == UInt32(ord("W"))
    )


def _plain_escape(point: UInt32, in_class: Bool) -> Bool:
    """Whether a letter after a backslash is one RE2 knows, outside the forms
    that need reading further.

    The two sets are different and the difference is the whole reason this takes
    an argument. Outside a class RE2 knows the five whitespace spellings, the
    six Perl classes, the four assertions `\\b`, `\\B`, `\\A` and `\\z`, and
    `\\C`, which matches one byte. Inside a class it knows the whitespace
    spellings and the Perl classes and none of the rest, because none of the
    rest is a character: `[\\b]` is a refusal in RE2 where it is a backspace in
    Python, and `[\\z]` is a refusal in both.

    Args:
        point: The letter after the backslash.
        in_class: Whether the escape is inside a bracketed class.

    Returns:
        True when RE2 reads it and nothing more has to be looked at.
    """
    if (
        point == UInt32(ord("a"))
        or point == UInt32(ord("f"))
        or point == UInt32(ord("n"))
        or point == UInt32(ord("r"))
        or point == UInt32(ord("t"))
        or point == UInt32(ord("v"))
    ):
        return True
    if _escape_is_class(point):
        return True
    if in_class:
        return False
    return (
        point == UInt32(ord("b"))
        or point == UInt32(ord("B"))
        or point == UInt32(ord("A"))
        or point == UInt32(ord("z"))
        or point == UInt32(ord("C"))
    )


def _named_control(point: UInt32) -> UInt32:
    """What a named control character is worth.

    Args:
        point: The letter after the backslash, which the caller has already
            checked is one of the six.

    Returns:
        The character it names, and zero for the assertions, which are not
        characters at all and never turn up as the end of a range.
    """
    if point == UInt32(ord("a")):
        return 0x07
    if point == UInt32(ord("f")):
        return 0x0C
    if point == UInt32(ord("n")):
        return 0x0A
    if point == UInt32(ord("r")):
        return 0x0D
    if point == UInt32(ord("t")):
        return 0x09
    if point == UInt32(ord("v")):
        return 0x0B
    return 0


def _is_letter(point: UInt32) -> Bool:
    """Whether a code point is an ASCII letter.

    Args:
        point: The code point.

    Returns:
        True for either case.
    """
    if point >= UInt32(ord("a")) and point <= UInt32(ord("z")):
        return True
    return point >= UInt32(ord("A")) and point <= UInt32(ord("Z"))


def re2_reads(pattern: StringSlice) -> Re2Read:
    """Whether RE2 would read a pattern, and why not when it would not.

    Args:
        pattern: The pattern as the caller wrote it.

    Returns:
        The answer. A refusal is a value rather than a raise for the same reason
        the Python grammar's is: the caller above is deciding what to tell
        somebody rather than failing.
    """
    var c = _Cursor(decoded(pattern))
    _ = _alternation(c)
    if not c.failed and not c.done():
        # The only way to leave the top level with something still to read is a
        # closing bracket, because the two functions under this one stop at one
        # and at nothing else.
        c.give_up(String("a bracket is closed that nothing opened"))
    var out = Re2Read()
    if c.failed:
        out.ok = False
        out.problem = c.problem.copy()
    return out^


def _alternation(mut c: _Cursor) -> Int:
    """Reads a run of branches separated by bars.

    An empty branch is a branch, so `|`, `a|`, `|a` and `(|)` are four patterns
    RE2 reads, which is worth saying because Python reads them too and a reader
    who has been told the two grammars differ starts expecting differences
    everywhere.

    Args:
        c: The cursor.

    Returns:
        How many copies of the widest branch a count in it could ask for, which
        is the largest of the branches rather than their total, because the
        budget a nest of counts shares is spent down one path at a time.
    """
    var worst = _branch(c)
    while not c.failed and c.peek() == UInt32(ord("|")):
        _ = c.take()
        var here = _branch(c)
        if here > worst:
            worst = here
    return worst


def _branch(mut c: _Cursor) -> Int:
    """Reads a run of pieces up to a bar, a closing bracket or the end.

    Args:
        c: The cursor.

    Returns:
        The largest count budget any one piece in it spends.
    """
    var worst = 1
    c.repeatable = False
    while not c.failed:
        if c.done():
            break
        var next = c.peek()
        if next == UInt32(ord("|")) or next == UInt32(ord(")")):
            break
        var here = _piece(c)
        if here > worst:
            worst = here
    return worst


def _piece(mut c: _Cursor) -> Int:
    """Reads one atom and whatever repeat is written after it.

    One repeat and no more, which is the rule that separates RE2's two
    complaints about a repeat. A repeat with nothing in front of it is one
    thing, and a repeat with a repeat in front of it is another, and Python
    calls both of them nothing to repeat.

    The question mark that follows a repeat is the ungreedy marker and is part
    of the repeat rather than a second one, so `a*?` is read and `a*??` is not.
    There is no third marker: `a*+` is where Python has a possessive quantifier
    and RE2 has a complaint.

    Args:
        c: The cursor.

    Returns:
        How many copies of this piece a count could ask for.
    """
    var worst = 1
    if not _at_repeat(c):
        worst = _atom(c)
    var repeated = False
    while not c.failed and not c.done():
        var counted = False
        var low = 0
        # A star and a plus and a question mark have no top, so RE2 spends the
        # smaller number rather than the larger one and none of the three can
        # ask for a second copy of anything.
        var high = 0
        var next = c.peek()
        if (
            next == UInt32(ord("*"))
            or next == UInt32(ord("+"))
            or next == UInt32(ord("?"))
        ):
            _ = c.take()
        else:
            if not _repeat_bounds(c, low, high):
                break
            counted = True
        if c.peek() == UInt32(ord("?")):
            _ = c.take()
        # The order of the next three is not the order a reader would write
        # them in and it is the order RE2 checks them in, which was measured
        # rather than guessed. `({99999}` has both no argument and a number too
        # large and RE2 names the number. `a{2}{3,1}` has both a repeat on a
        # repeat and a range the wrong way round and RE2 names the repeat.
        if repeated:
            c.give_up(String("RE2 will not repeat a repeat"))
            break
        if counted and (
            low > RE2_MAX_REPEAT or high > RE2_MAX_REPEAT or high < low
        ):
            c.give_up(String("RE2 will not repeat that many times"))
            break
        if not c.repeatable:
            c.give_up(String("there is nothing here for that repeat to repeat"))
            break
        repeated = True
        if counted and high > 0:
            worst = worst * high
            if worst > RE2_MAX_REPEAT:
                c.give_up(String("RE2 will not repeat that many times"))
                break
    return worst


def _at_repeat(mut c: _Cursor) -> Bool:
    """Whether the next thing to read is a repeat rather than something to
    repeat.

    Asked before the atom rather than after it, because a repeat with nothing in
    front of it is still a repeat to RE2 and has to reach the same three checks
    in the same order as one that has something in front of it. A brace is only
    a repeat when it opens a well formed count, so this has to try reading one,
    and it puts the cursor back afterwards.

    Args:
        c: The cursor.

    Returns:
        True when a repeat operator is next.
    """
    var next = c.peek()
    if (
        next == UInt32(ord("*"))
        or next == UInt32(ord("+"))
        or next == UInt32(ord("?"))
    ):
        return True
    if next != UInt32(ord("{")):
        return False
    var at = c.at
    var low = 0
    var high = 0
    var found = _repeat_bounds(c, low, high)
    c.at = at
    return found


def _repeat_bounds(mut c: _Cursor, mut low: Int, mut high: Int) -> Bool:
    """Reads a well formed count, leaving the cursor alone when there is not one.

    A `{` that does not open a count is a literal brace to RE2, which is where
    the one hundred and sixty six patterns holding `{,` in the corpus go: `a{,2}`
    is five characters rather than a count with no lower bound, and that is the
    reading Python does not have. `a{}`, `a{a}`, `a{2`, `a{-1}` and `a{1,2,3}`
    are all literal braces for the same reason.

    Args:
        c: The cursor.
        low: Set to the smallest number of copies.
        high: Set to the largest, which is the smallest again when the count
            named one number, and a thousand and one when it named no top, since
            a count with no top spends no budget and any number over the limit
            would do.

    Returns:
        True when a count was read and consumed.
    """
    if c.peek() != UInt32(ord("{")):
        return False
    var at = c.at + 1
    var digits = 0
    var first = 0
    while at < len(c.points) and _is_digit(c.points[at]):
        first = first * 10 + Int(c.points[at] - UInt32(ord("0")))
        if first > 100000:
            first = 100000
        at += 1
        digits += 1
    if digits == 0:
        return False
    var second = first
    if at < len(c.points) and c.points[at] == UInt32(ord(",")):
        at += 1
        var more = 0
        second = 0
        while at < len(c.points) and _is_digit(c.points[at]):
            second = second * 10 + Int(c.points[at] - UInt32(ord("0")))
            if second > 100000:
                second = 100000
            at += 1
            more += 1
        if more == 0:
            # `a{2,}` has no top, so nothing multiplies and the only thing left
            # to check is the bottom.
            second = -1
    if at >= len(c.points) or c.points[at] != UInt32(ord("}")):
        return False
    c.at = at + 1
    low = first
    high = second if second >= 0 else first
    return True


def _atom(mut c: _Cursor) -> Int:
    """Reads one thing a repeat could be written after.

    Args:
        c: The cursor.

    Returns:
        How many copies of it a count inside it could ask for, which is one for
        everything that is not a group.
    """
    var next = c.peek()
    if next == UInt32(ord("(")):
        return _group(c)
    c.repeatable = True
    if next == UInt32(ord("[")):
        _class(c)
        return 1
    if next == UInt32(ord("\\")):
        _escape(c, False)
        return 1
    # A brace that opens no well formed count reaches here and is a literal,
    # because the caller only skips the atom when there is a real repeat in
    # front of it. Everything else that reaches here is one character.
    _ = c.take()
    return 1


def _group(mut c: _Cursor) -> Int:
    """Reads a bracket and whatever the two characters after it turn it into.

    RE2 has five bracket forms and Python has a dozen, and the gap between the
    two counts is most of the twelve thousand eight hundred and eighteen
    patterns in the corpus RE2 turns down for the shape of a bracket. The five
    are a plain capture, `(?:`, a named capture written either `(?P<name>` or
    `(?<name>`, a flag group `(?flags)` and a scoped flag group `(?flags:`.

    Two of the five are worth a sentence. A flag group may stand anywhere in a
    pattern rather than only at the front, so `a(?i)b` is read here and refused
    by Python, and it is what nine hundred and forty eight of the corpus
    patterns that RE2 takes and Python does not are made of. And a flag group
    may name no flags at all, so `(?)` is read, while `(?-)` and `(?i-)` are
    not, because a minus sign has to be turning something off.

    Args:
        c: The cursor.

    Returns:
        How many copies a count inside the group could ask for.
    """
    _ = c.take()
    if c.peek() != UInt32(ord("?")):
        var worst = _alternation(c)
        _close(c)
        c.repeatable = True
        return worst
    _ = c.take()
    var next = c.peek()
    if next == UInt32(ord(":")):
        _ = c.take()
        var worst = _alternation(c)
        _close(c)
        c.repeatable = True
        return worst
    if next == UInt32(ord("P")):
        if c.ahead(1) != UInt32(ord("<")):
            c.give_up(String("RE2 has no group written that way"))
            return 1
        _ = c.take()
        return _named(c)
    if next == UInt32(ord("<")):
        var after = c.ahead(1)
        if after == UInt32(ord("=")) or after == UInt32(ord("!")):
            c.give_up(String("RE2 has no group written that way"))
            return 1
        return _named(c)
    if (
        _is_flag_point(next)
        or next == UInt32(ord("-"))
        or next == UInt32(ord(")"))
    ):
        return _flags(c)
    c.give_up(String("RE2 has no group written that way"))
    return 1


def _named(mut c: _Cursor) -> Int:
    """Reads a named capture, with the cursor on the opening angle bracket.

    Args:
        c: The cursor.

    Returns:
        How many copies a count inside the group could ask for.
    """
    _ = c.take()
    var length = 0
    while not c.done() and c.peek() != UInt32(ord(">")):
        if not _is_name_point(c.peek()):
            c.give_up(String("RE2 will not take that group name"))
            return 1
        _ = c.take()
        length += 1
    if c.done() or length == 0:
        c.give_up(String("RE2 will not take that group name"))
        return 1
    _ = c.take()
    var worst = _alternation(c)
    _close(c)
    c.repeatable = True
    return worst


def _flags(mut c: _Cursor) -> Int:
    """Reads a flag group, with the cursor on the first letter or on the minus.

    Args:
        c: The cursor.

    Returns:
        How many copies a count inside a scoped one could ask for.
    """
    var minus = False
    var since = 0
    while True:
        if c.done():
            c.give_up(String("RE2 has no group written that way"))
            return 1
        var next = c.peek()
        if _is_flag_point(next):
            _ = c.take()
            since += 1
            continue
        if next == UInt32(ord("-")) and not minus:
            _ = c.take()
            minus = True
            since = 0
            continue
        break
    if minus and since == 0:
        c.give_up(String("RE2 has no group written that way"))
        return 1
    var next = c.peek()
    if next == UInt32(ord(")")):
        _ = c.take()
        # A flag group leaves nothing behind, and the thing to notice is what
        # that means for a repeat written after one. It does not make the repeat
        # illegal, it makes the repeat apply to whatever was already there, so
        # `(?i)*` is a refusal at the front of a pattern and `a(?i)*` is a
        # perfectly ordinary repeat of `a`. Which is to say this line is
        # deliberately not setting the flag either way.
        return 1
    if next == UInt32(ord(":")):
        _ = c.take()
        var worst = _alternation(c)
        _close(c)
        c.repeatable = True
        return worst
    c.give_up(String("RE2 has no group written that way"))
    return 1


def _close(mut c: _Cursor):
    """Reads the closing bracket of a group, or records that there is not one.

    Args:
        c: The cursor.
    """
    if c.failed:
        return
    if c.peek() != UInt32(ord(")")):
        c.give_up(String("a bracket is opened that nothing closes"))
        return
    _ = c.take()


def _class(mut c: _Cursor):
    """Reads a bracketed class.

    Three of RE2's rules in here are ones Python has not got. A `]` written
    first is a literal `]` rather than the end of an empty class, so `[]a]` is
    a class of two characters and `[]` is a class nothing closes. A `[:name:]`
    written inside is one of fourteen named sets, and a `[:` that is never
    closed by a `:]` is a literal bracket and a literal colon, which is why
    `[[:alpha]]` is read and `[[:foo:]]` is not. And a `-` after something that
    is a set rather than a character is a literal hyphen, so `[\\d-a]` is read
    and `[a-\\d]` is not.

    Args:
        c: The cursor.
    """
    _ = c.take()
    if c.peek() == UInt32(ord("^")):
        _ = c.take()
    var first = True
    while True:
        if c.failed:
            return
        if c.done():
            c.give_up(String("a class is never closed"))
            return
        var next = c.peek()
        if next == UInt32(ord("]")) and not first:
            _ = c.take()
            return
        first = False
        _item(c)
        if c.failed:
            return
        if c.kind != _KIND_CHAR:
            continue
        if c.peek() != UInt32(ord("-")):
            continue
        if c.ahead(1) == UInt32(ord("]")) or c.ahead(1) == 0xFFFFFFFF:
            continue
        var low = c.point
        _ = c.take()
        _item(c)
        if c.failed:
            return
        if c.kind != _KIND_CHAR:
            # `[a-[:alpha:]]` and `[a-\\p{L}]` are the same refusal as
            # `[a-\\d]`, which RE2 words as the escape being wrong rather than
            # as the range being wrong.
            c.give_up(String("RE2 has no such escape"))
            return
        if c.point < low:
            c.give_up(String("a range in a class runs backwards"))
            return
        c.kind = _KIND_CLASS


def _item(mut c: _Cursor):
    """Reads one thing inside a class and records what kind of thing it was.

    Args:
        c: The cursor.
    """
    var next = c.peek()
    if next == UInt32(ord("\\")):
        _escape(c, True)
        return
    if next == UInt32(ord("[")) and c.ahead(1) == UInt32(ord(":")):
        if _posix(c):
            return
    c.kind = _KIND_CHAR
    c.point = c.take()


def _posix(mut c: _Cursor) -> Bool:
    """Reads a `[:name:]` class, leaving the cursor alone when there is not one.

    Args:
        c: The cursor.

    Returns:
        True when one was read or refused, and False when the `[:` turned out to
        be two ordinary characters.
    """
    var at = c.at + 2
    if at < len(c.points) and c.points[at] == UInt32(ord("^")):
        at += 1
    var name = String("")
    while at < len(c.points) and _is_letter(c.points[at]):
        name += chr(Int(c.points[at]))
        at += 1
    if at + 1 >= len(c.points):
        return False
    if c.points[at] != UInt32(ord(":")) or c.points[at + 1] != UInt32(ord("]")):
        return False
    var known = False
    for spelling in _posix_names():
        if spelling == name:
            known = True
    if not known:
        c.give_up(String("RE2 has no such character class"))
        return True
    c.at = at + 2
    c.kind = _KIND_CLASS
    return True


def _escape(mut c: _Cursor, in_class: Bool):
    """Reads a backslash and whatever it is in front of.

    Args:
        c: The cursor.
        in_class: Whether the escape is inside a bracketed class, which changes
            both which letters RE2 knows and what `\\Q` means.
    """
    _ = c.take()
    if c.done():
        c.give_up(String("a backslash is the last thing in the pattern"))
        return
    var next = c.peek()
    c.kind = _KIND_CHAR
    c.point = 0
    if next == UInt32(ord("p")) or next == UInt32(ord("P")):
        _ = c.take()
        # This was the one construct this file would not judge, because the
        # table of names RE2 knows was not in the library and a name guessed
        # wrong would have refused a pattern pandas answers. The table is in
        # `unicodedata.mojo` now, measured against the same RE2 this file is
        # written against, so the question is answered rather than declined and
        # nothing here is unsure any more.
        c.kind = _KIND_CLASS
        var start: Int
        var stop: Int
        if not c.done() and c.peek() == UInt32(ord("{")):
            _ = c.take()
            if not c.done() and c.peek() == UInt32(ord("^")):
                _ = c.take()
            start = c.at
            while not c.done() and c.peek() != UInt32(ord("}")):
                _ = c.take()
            if c.done():
                c.give_up(String("RE2 has no such character class"))
                return
            stop = c.at
            _ = c.take()
        else:
            # One character and not a name, so `\\pLu` is the letter category
            # and then a literal `u`.
            if c.done():
                c.give_up(String("RE2 has no such character class"))
                return
            start = c.at
            _ = c.take()
            stop = c.at
        var bytes = List[UInt8]()
        for i in range(start, stop):
            _put_point(bytes, c.points[i])
        if unicode_index(StringSlice(unsafe_from_utf8=Span(bytes))) < 0:
            c.give_up(String("RE2 has no such character class"))
        return
    if next == UInt32(ord("Q")):
        if in_class:
            c.give_up(String("RE2 has no such escape"))
            return
        _quoted(c)
        return
    if next == UInt32(ord("E")):
        c.give_up(String("RE2 has no such escape"))
        return
    if next == UInt32(ord("x")):
        _hex(c)
        return
    if _is_octal(next):
        _octal(c)
        return
    if _plain_escape(next, in_class):
        _ = c.take()
        if _escape_is_class(next):
            c.kind = _KIND_CLASS
        else:
            # A named control character may be one end of a range, and `[a-\\n]`
            # runs backwards if this file forgets what the name is worth.
            c.point = _named_control(next)
        return
    if _is_letter(next) or _is_digit(next) or next >= 0x80:
        # Every letter RE2 does not know, every digit that is not octal, and
        # every character outside ASCII. The last of the three is the one that
        # looks like nothing: `\\é` is an ordinary way to write that character
        # in Python and is a refusal here.
        c.give_up(String("RE2 has no such escape"))
        return
    c.point = c.take()


def _quoted(mut c: _Cursor):
    """Reads a `\\Q...\\E` run, with the cursor on the `Q`.

    Everything between the two is a literal, and a run that is never closed
    runs to the end of the pattern rather than being a refusal, so `\\Qa` is a
    pattern RE2 reads. This is the construct behind two hundred and eight of the
    corpus patterns RE2 takes and Python does not, which is the largest of them.

    Args:
        c: The cursor.
    """
    _ = c.take()
    while not c.done():
        if c.peek() == UInt32(ord("\\")) and c.ahead(1) == UInt32(ord("E")):
            _ = c.take()
            _ = c.take()
            return
        _ = c.take()


def _hex(mut c: _Cursor):
    """Reads a `\\xHH` or a `\\x{...}`, with the cursor on the `x`.

    Args:
        c: The cursor.
    """
    _ = c.take()
    if c.peek() == UInt32(ord("{")):
        _ = c.take()
        var digits = 0
        var value = 0
        while not c.done() and _is_hex(c.peek()):
            value = value * 16 + _hex_value(c.take())
            if value > 0x7FFFFF:
                value = 0x7FFFFF
            digits += 1
        if digits == 0 or c.peek() != UInt32(ord("}")):
            c.give_up(String("RE2 has no such escape"))
            return
        _ = c.take()
        if value > 0x10FFFF:
            c.give_up(String("RE2 has no such escape"))
            return
        c.point = UInt32(value)
        return
    if not _is_hex(c.peek()) or not _is_hex(c.ahead(1)):
        c.give_up(String("RE2 has no such escape"))
        return
    var high = _hex_value(c.take())
    var low = _hex_value(c.take())
    c.point = UInt32(high * 16 + low)


def _hex_value(point: UInt32) -> Int:
    """What one hexadecimal digit is worth.

    Args:
        point: The digit, which the caller has already checked is one.

    Returns:
        Its value.
    """
    if _is_digit(point):
        return Int(point - UInt32(ord("0")))
    if point >= UInt32(ord("a")):
        return Int(point - UInt32(ord("a"))) + 10
    return Int(point - UInt32(ord("A"))) + 10


def _octal(mut c: _Cursor):
    """Reads an octal escape, with the cursor on the first digit.

    RE2 tells an octal escape from a backreference by how the digits start. A
    run starting with a zero is octal whatever comes next, so `\\0` and `\\08`
    are both read. A run starting with one through seven has to be at least two
    digits long, so `\\12` is read and `\\1` is a refusal, which is RE2 saying
    it has no backreference in the only words it has for it.

    Args:
        c: The cursor.
    """
    var zero = c.peek() == UInt32(ord("0"))
    var value = 0
    var digits = 0
    while digits < 3 and _is_octal(c.peek()):
        value = value * 8 + Int(c.take() - UInt32(ord("0")))
        digits += 1
    if not zero and digits < 2:
        c.give_up(String("RE2 has no such escape"))
        return
    c.point = UInt32(value)
