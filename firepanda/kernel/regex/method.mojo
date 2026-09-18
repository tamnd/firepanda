"""Which pattern each of the six methods actually runs, and who runs it.

`contains`, `match` and `fullmatch` are one question upstream. pandas asks the
engine whether a pattern matches somewhere and asks the other two by changing
the pattern, so the difference between the three is a rewrite rather than a
mode, and the rewrite is visible in what the pattern does rather than only in
the answer. `str.match("a|b")` asks whether a row starts with `a` or starts with
`b`, because the rewrite puts the alternation in a group before it puts the
anchor on, and a reading that anchored the first arm only would be a different
column.

The order the two halves happen in is the part worth stating, because getting it
backwards is easy and costs patterns. pandas decides which engine a pattern goes
to by looking at the pattern the caller wrote, and only the branch that goes to
Arrow rewrites it. Deciding on the rewritten pattern instead is wrong for a
family: `(?i)(?=a)` is a pattern Python reads and answers, and wrapping it puts
the flag group somewhere Python's grammar will not have it, which would send a
pattern upstream answers to the engine that has never heard of a lookahead.

`count` is the fourth and is not that question at all. It asks how many matches
there are rather than whether there is one, and it runs the pattern the caller
wrote without any anchor, so everything in this file treats it the way it treats
`contains`. Where it parts company with the other three is further down, in the
scan that uses the answer: `firepanda/kernel/regex/pike.mojo` has the three
rules Arrow counts by and none of them is visible here.

`replace` is the fifth and is the first that changes how the pattern is
compiled rather than what the pattern is. It anchors nothing, the way `count`
and `contains` do not, and it asks for a program that records where every group
matched, which is the one thing here that costs something and is the one thing
only it needs.

`extract` is the sixth and is the first that is not asking Arrow anything at
all. pandas answers it, and `extractall` and `findall` with it, by compiling the
pattern with `re` and looping in Python, so it is not routed and the letters in
its pattern mean what Python says they mean rather than what RE2 says. That is
the one line in this file where a method's engine is a property of the method
rather than of the pattern, and document 81 has why.

There is a second way to end up on Python's engine and it belongs to the call
rather than to the pattern or to the method. A caller who passes `flags` to
`contains`, `fullmatch`, `count` or `replace` has moved that call to Python's
`re` upstream, because those four hand any flag straight to the object path and
their Arrow path refuses one. `match` is the exception and keeps its two, and
document 84 has why. A call that moved is anchored differently as well as run
differently, since upstream stops rewriting the pattern the moment it stops
talking to Arrow, and `python_anchored` is that other rewrite.

This file is the layer above the compiler and below the binding. It knows what
pandas does with a pattern before handing it over, and it hands back a program
or a refusal, which is the same pair the compiler deals in. What it deliberately
does not know is what a refusal should become in Python, because that is a fact
about the binding.
"""

from firepanda.kernel.regex.parse import parse_pattern
from firepanda.kernel.regex.program import (
    PYTHON_NEWEST,
    Program,
    compile_program,
)
from firepanda.kernel.regex.route import (
    ENGINE_PYTHON,
    ENGINE_RE2,
    holds_unsupported,
)
from firepanda.kernel.regex.tokens import FLAG_VERBOSE


comptime METHOD_CONTAINS: UInt8 = 0
"""`str.contains`, which is the question the engine answers and needs no
rewrite."""

comptime METHOD_MATCH: UInt8 = 1
"""`str.match`, which is the same question with the pattern anchored at the
front."""

comptime METHOD_FULLMATCH: UInt8 = 2
"""`str.fullmatch`, which is the same question anchored at both ends."""

comptime METHOD_COUNT: UInt8 = 3
"""`str.count`, which asks a different question of the engine and asks it of the
same pattern.

It is here rather than being served by `METHOD_CONTAINS` because the two are the
same by arithmetic rather than by meaning. `count` runs the pattern the caller
wrote for the same reason `contains` does, which is that upstream anchors
nothing for either, and a fourth name costs one branch and says which of the
four a reader is looking at. What `count` does differ in is a line upstream that
is not in this file at all: it never asks whether the pattern is a compiled one
carrying flags, because it has no `case` argument to disagree with.
"""

comptime METHOD_REPLACE: UInt8 = 4
"""`str.replace` with a pattern, which anchors nothing and needs captures.

The fifth name and the first that changes what the compiler is asked for rather
than what it is asked about. A replacement can write `\\1` for what a group
held, so the program has to be built with the instructions that record where
each group matched, which every other method here would only pay for.

The scan it runs is not the scan `count` runs either, and the difference is not
arithmetic this time. Counting cuts the row after each match and replacing does
not, so `^` means the start of what is left to one and the start of the row to
the other, in the same library on the same pattern. Document 80 has the
measurements and `firepanda/kernel/regex/replace.mojo` has the loop.
"""

comptime METHOD_EXTRACT: UInt8 = 5
"""`str.extract`, which is the first name here that does not go to Arrow at all.

The five above are five ways of asking Arrow a question. This one is not: pandas
answers it by compiling the pattern with `re` and looping in Python, so the
engine it runs is Python's and the letters in the pattern mean what Python says
they mean. `\\w` covers 138558 code points for this method and 63 for the five
above, in the same accessor, and document 81 is where that was measured.

Everything else about it is the easy half. Nothing is rewritten, because a
rewrite here would be a rewrite pandas does not do, and nothing is anchored,
because upstream runs `regex.search` and takes the leftmost match wherever it
falls. It wants captures for the obvious reason: the groups are the answer.
"""


def _is_flag_byte(b: UInt8) -> Bool:
    """Whether a byte is one that can appear inside an inline flag group.

    The seven letters plus the minus sign that turns them off, and nothing else,
    which is what keeps `(?:`, `(?=`, `(?#` and `(?P<` out of the scan below:
    none of them has a second character in this set, so the scan stops at once.

    Args:
        b: The byte.

    Returns:
        True for one of the eight.
    """
    return (
        b == UInt8(ord("a"))
        or b == UInt8(ord("i"))
        or b == UInt8(ord("L"))
        or b == UInt8(ord("m"))
        or b == UInt8(ord("s"))
        or b == UInt8(ord("u"))
        or b == UInt8(ord("x"))
        or b == UInt8(ord("-"))
    )


def leading_flags(pattern: String) -> Int:
    """How many bytes at the front of a pattern are a global flag group.

    Args:
        pattern: The pattern.

    Returns:
        The length of the group in bytes, and zero when the pattern does not
        open with one. A `(?)` with no letters in it counts as zero, since it is
        not a flag group to anybody and moving it would only move a syntax
        error.
    """
    if not pattern.startswith("(?"):
        return 0
    var bytes = pattern.as_bytes()
    var i = 2
    while i < len(bytes):
        if bytes[i] == UInt8(ord(")")):
            return i + 1 if i > 2 else 0
        if not _is_flag_byte(bytes[i]):
            return 0
        i += 1
    return 0


def preprocessed(pattern: String) -> String:
    """Rewrites a trailing `\\Z` into `\\z`, which is pandas' own first step.

    RE2 has no `\\Z` and refuses the escape, and pandas rewrites one at the end
    of a pattern into the spelling RE2 does have. It does that before it anchors
    anything, which is the part that matters here: `\\Z` on the end of the
    pattern the caller wrote is in the middle of the pattern the engine runs,
    and a rewrite that happened afterwards would come too late to save it.

    Only a `\\Z` that is a real escape is rewritten, so `\\\\Z`, which is a
    backslash and then a letter, is left alone. Counting the backslashes is how
    upstream tells the two apart and is what is copied here.

    Args:
        pattern: The pattern as the caller wrote it.

    Returns:
        The pattern with a trailing `\\Z` spelled RE2's way.
    """
    if not pattern.endswith("\\Z"):
        return pattern.copy()
    var bytes = pattern.as_bytes()
    var at = len(bytes) - 2
    var slashes = 0
    while at >= 0 and bytes[at] == UInt8(ord("\\")):
        slashes += 1
        at -= 1
    if slashes % 2 == 0:
        return pattern.copy()
    return String(pattern[byte = 0 : pattern.byte_length() - 2], "\\z")


def anchored(method: UInt8, pattern: String) -> String:
    """Rewrites a pattern the way pandas rewrites it for `match` and `fullmatch`.

    `match` strips a leading `^`, wraps what is left in a group and puts the `^`
    back. `fullmatch` does the same at both ends and then hands the result to
    `match`, so a pattern goes through both rewrites and comes out inside two
    groups, except for a pattern already carrying both anchors, which fullmatch
    leaves alone and match then wraps once.

    Two details are copied rather than tidied. A trailing `$` counts as an
    anchor only when it is not written `\\$`, which is how pandas tells an
    anchor from a dollar sign and which it gets wrong for a pattern ending in a
    literal backslash. And one leading `^` is stripped and no more, so `^^a`
    becomes `^(^a)` and still means what it meant. Both are what upstream does
    and neither changes an answer, since `^^` and `$$` are each the same
    assertion twice.

    One thing is not copied, and it is the one place where copying would lose
    patterns this library can answer. A pattern opening with a global flag group
    keeps that group at the front, so `(?s)a.b` becomes `(?s)\\A(a.b)` where
    pandas writes `^((?s)a.b)`. The reason is that pandas hands its rewrite
    straight to Arrow while this library parses it first, and Python's grammar
    wants a global flag group first in the pattern and refuses one anywhere
    else. Without the hoist `str.match("(?s)a.b")` is refused over the rewrite
    rather than over the pattern, while `str.contains` with the same pattern
    answers.

    Moving the group changes what it covers, and the two anchors being added are
    what it would now cover that it did not before. So a hoisted rewrite writes
    them `\\A` and `\\z` rather than `^` and `$`, which are the same two
    positions with no flag able to touch them, and it leaves any `^` the caller
    wrote where it was rather than stripping it, because upstream only strips
    one from the front of the whole pattern and a pattern opening with a flag
    group has none there. Both matter for `m` alone and both are written
    unconditionally, since a hoist that is only sound for six of the seven
    letters is a hoist somebody has to keep checking.

    The slicing is by byte, which is exact because every character being looked
    for is ASCII and a byte of a longer character cannot be mistaken for one.

    Args:
        method: Which of the five asked.
        pattern: The pattern as the caller wrote it, with `\\Z` already seen to.

    Returns:
        The pattern the engine is to be given.
    """
    if (
        method == METHOD_CONTAINS
        or method == METHOD_COUNT
        or method == METHOD_REPLACE
        or method == METHOD_EXTRACT
    ):
        return pattern.copy()
    var cut = leading_flags(pattern)
    var head = String(pattern[byte=0:cut])
    var out = String(pattern[byte=cut:])
    var start = String("^") if cut == 0 else String("\\A")
    var stop = String("$") if cut == 0 else String("\\z")
    var dollar = out.endswith("$") and not out.endswith("\\$")
    var caret = cut == 0 and out.startswith("^")
    if method == METHOD_FULLMATCH and not (dollar and caret):
        if dollar:
            var cropped = String(out[byte = 0 : out.byte_length() - 1])
            out = cropped^
        elif caret:
            var tail = String(out[byte=1:])
            out = tail^
        var wrapped = String("(", out, ")", stop)
        out = wrapped^
    if cut == 0 and out.startswith("^"):
        var rest = String(out[byte=1:])
        out = rest^
    return String(head, start, "(", out, ")")


def python_anchored(
    method: UInt8, pattern: String, verbose: Bool = False
) -> String:
    """Rewrites a pattern the way a call that landed on Python's engine needs it.

    The function above copies a rewrite pandas does. This one copies a rewrite
    pandas does not do, which is why the two are separate rather than one
    function with a branch in it. A call that goes to Python's engine upstream
    is answered by `regex.match` and `regex.fullmatch` rather than by a pattern
    with anchors glued to it, and those two methods anchor from outside the
    pattern where the glued anchors are inside it. The difference is a flag
    away: `re.fullmatch("a", "a\\n")` finds nothing, and `^(a)$` with the
    multiline flag on matches the first line of it.

    So the anchors written here are `\\A` and `\\Z`, which are the two positions
    no flag can move, rather than the `^` and `$` the Arrow rewrite uses. That
    is the whole of the difference for `fullmatch`, and `match` needs only the
    first of the two because `regex.match` says where a match may start and says
    nothing about where it ends.

    The bracket is not a capturing one, and that is worth saying because it used
    to be. A capturing bracket around the whole pattern numbers every group the
    caller wrote one higher, so `(a)\\1` comes out as `\\A((a)\\1)\\Z` and the
    backreference in it now names the wrapper rather than the group beside it.
    That is a wrong answer and not a refusal, since the wrapper is still open
    where the reference stands and a group that is still open reads as one that
    never took part, so the pattern quietly stops matching anything. It never
    showed while a backreference was refused on both engines and it stopped
    being harmless the day one of them could read it. Document 95.

    The closing one used to be `\\z`, which is RE2's spelling of the same
    position and is a `bad escape \\z` to every CPython before 3.14. It never
    showed, because this library's own parser reads both spellings and the
    pattern written here is never handed to an interpreter, and it was still
    wrong: this function writes a pattern for Python's engine and Python's
    engine is the thing that has not always had that spelling. Document 91 is
    where it was found, by the slice that had to tell the two spellings apart
    for the caller's sake and could not while this one was writing one of
    them.

    A pattern opening with a global flag group keeps that group at the front for
    the reason `anchored` gives, which is that Python's grammar will not have
    one anywhere else, and nothing is stripped or cropped on the way past
    because there is no upstream rewrite here to copy the quirks of.

    Nothing reaches this with `match` today, since the only flags upstream lets
    `match` keep are the two that leave it on Arrow. The branch is written all
    the same, because the alternative to a line nobody runs is a wrong anchor
    the day somebody does.

    Under verbose mode a newline goes in ahead of the closing bracket, and it
    is there because a pattern is allowed to end in the middle of a comment. `a
    # c` is a perfectly good verbose pattern, and gluing `)` onto the end of it
    puts the bracket inside the comment, where the grammar never sees it and
    the group is never closed. A newline is the only thing that ends a comment,
    and under verbose mode a newline outside a class is thrown away, so it
    costs nothing anywhere else. Upstream has no such trouble because
    `regex.fullmatch` anchors from outside the pattern and never writes a
    bracket at all. Document 88 is where this was found, by a sweep, on the one
    pattern in it that ends in a comment.

    Args:
        method: Which of the six asked.
        pattern: The pattern as the caller wrote it.
        verbose: Whether the pattern is being read under verbose mode, counting
            both the letter passed beside it and a `(?x)` written into it.

    Returns:
        The pattern the engine is to be given.
    """
    if method != METHOD_MATCH and method != METHOD_FULLMATCH:
        return pattern.copy()
    var cut = leading_flags(pattern)
    var head = String(pattern[byte=0:cut])
    var rest = String(pattern[byte=cut:])
    var end = String("\n") if verbose else String("")
    if method == METHOD_MATCH:
        return String(head, "\\A(?:", rest, end, ")")
    return String(head, "\\A(?:", rest, end, ")\\Z")


comptime ALPHABET_ROWS: Int = 4096
"""How tall a column has to be before a pattern is compiled with an alphabet.

The alphabet is what the state cache lays its transitions out on and it is not
free to work out: four to nine microseconds on top of a compile for a pattern
over ASCII, and up to three hundred for one over Python's Unicode classes, which
are the largest range tables in the library. The cache then saves somewhere
between seventy nanoseconds and a microsecond per row depending on the pattern,
so the compile pays for itself somewhere between a hundred rows and a few
thousand, and a bound at the far end of that is the one that never loses.

It has to be decided by the caller rather than by the compiler, because it is a
trade between what a compile costs once and what a row costs many times and only
the caller knows how many rows there are.
"""


def program_for(
    method: UInt8,
    pattern: String,
    flags: Int32 = 0,
    argued: Bool = False,
    minor: Int = PYTHON_NEWEST,
    alphabet: Bool = False,
) -> Program:
    """Compiles what one of the five methods would run, or refuses it.

    The routing decision is made first and on the pattern as written, which is
    this file's docstring and is pandas' order. A pattern routed to Python is
    then rewritten and compiled exactly as an argued call is, because the two
    are the same situation once the engine is settled and the only difference
    between them is what settled it.

    The `\\Z` rewrite comes next and the anchoring last, which is upstream's
    order as well and is not an order either step is indifferent to.

    `extract` skips all three steps. It is not routed, because upstream never
    routes it and always answers it in Python. It is not preprocessed, because
    the `\\Z` rewrite exists to spell an escape the way RE2 spells it and no
    part of this method goes near RE2. And it is not anchored, because upstream
    runs `regex.search` over the row as written.

    The flags a caller passed beside the pattern ride through both parses and
    change nothing about which of the three steps happen. What they can change
    is the engine, and that is what `argued` says. The bits cannot say it on
    their own: `case=False` reaches here as ignore case and stays on RE2, and
    `flags=re.IGNORECASE` reaches here as the same bit and does not, so the fact
    that is needed is how the caller spelled it rather than what they asked for.
    Upstream draws the line in the same place and for the same reason, which is
    that its routing test asks whether the accessor was handed a flags argument
    rather than what the compiled pattern ended up holding.

    A call that is argued skips the `\\Z` rewrite as well as Arrow's anchoring.
    The rewrite exists to spell an escape the way RE2 spells it, and an argued
    call has already left RE2, so doing it would turn a pattern Python reads one
    way into a pattern Python reads another way for no reason at all.

    Args:
        method: Which of the six asked.
        pattern: The pattern as the caller wrote it.
        flags: Flags passed beside the pattern, as `FLAG_` bits. The `case`
            argument arrives here as ignore case, because upstream turns it
            into exactly that before anything else looks at it.
        argued: Whether those flags came from a `flags` argument, which is what
            moves the call to Python's engine. False is the ordinary call and
            leaves everything below exactly as it was.
        minor: Which CPython the answer is to agree with, as the minor number
            alone. It is passed to every compile rather than only to the ones
            that reach Python's engine, because which engine a call lands on is
            decided in here and a caller would have to read this function to
            know when the number mattered.
        alphabet: Whether the column is tall enough to be worth compiling an
            alphabet for, which is what the state cache runs its transitions on.
            It is acted on only for the three methods that answer whether a row
            matched, since those are the ones the cache can answer today, and
            `ALPHABET_ROWS` is the height a caller is expected to ask about.

    Returns:
        The program, or the reason there is not one, with the flag saying whose
        refusal it is. A refusal is a value here rather than a raise for the
        reason `compile_program` gives.
    """
    # Only the three that answer whether a row matched, because the state cache
    # answers those and nothing else yet, and a table nothing reads is a compile
    # nobody asked for.
    var table = alphabet and (
        method == METHOD_CONTAINS
        or method == METHOD_MATCH
        or method == METHOD_FULLMATCH
    )
    var tree = parse_pattern(pattern, flags)
    if method == METHOD_EXTRACT:
        return compile_program(tree, ENGINE_PYTHON, captures=True, minor=minor)
    if argued or holds_unsupported(tree):
        # One branch for the two ways a call lands on Python's engine, which it
        # has not always been. A routed call used to be compiled here over the
        # caller's own tree and unanchored, because every construct that routes
        # was refused and a refusal does not care where the anchors are. The
        # lookahead ended that: a routed call can now come back with a program
        # in it, and a program for `match` with no `\\A` in front of it answers
        # a different question from the one that was asked. Document 93.
        if not tree.ok:
            # Refused over the pattern the caller wrote rather than over the
            # anchored one, for the reason the `not tree.ok` branch below gives.
            # The engine is the same either way here, so the only thing the
            # choice decides is which pattern the message quotes. Only an argued
            # call reaches this, since a pattern the grammar cannot read is not
            # routed anywhere by a walk over a tree that does not exist.
            return compile_program(tree, ENGINE_PYTHON, minor=minor)
        # `count` asks for captures on this engine and not on the other one,
        # which is the one place the two scans disagree about what they need
        # rather than about what they do. Python's rule for where to look next
        # is written in terms of where the match started, since a match of no
        # width is one whose two ends agree wherever it was found, and Arrow's
        # rule only ever compares the end against a cursor it kept itself.
        var wants = method == METHOD_REPLACE or method == METHOD_COUNT
        if method == METHOD_MATCH or method == METHOD_FULLMATCH:
            # The two methods that rewrite are asked about the pattern as the
            # caller wrote it first, and the rewrite is only reached when that
            # answers yes. So a refusal is worded over what the caller wrote,
            # rather than over a pattern with a bracket round it that the caller
            # never asked for and cannot see in the message.
            #
            # The bracket the rewrite writes is a non capturing one, which is
            # what keeps the two compiles asking the same question now that a
            # backreference is one of the things they can be asked. A capturing
            # bracket numbers every group one higher and the reference follows
            # the numbering, so the second compile would answer a different
            # pattern from the one the first agreed to. `python_anchored` says
            # more. Document 95.
            var own = compile_program(tree, ENGINE_PYTHON, wants, minor)
            if not own.ok:
                return own^
        return compile_program(
            parse_pattern(
                python_anchored(
                    method, pattern, (tree.flags & FLAG_VERBOSE) != 0
                ),
                flags,
            ),
            ENGINE_PYTHON,
            captures=wants,
            minor=minor,
            alphabet=table,
        )
    if not tree.ok:
        # A pattern the grammar cannot read is refused over what the caller
        # wrote rather than over the rewrite, because the rewrite can make a
        # bad pattern good and upstream would still have refused it. `)a` is
        # the shape: a bracket nobody opened, which `^()a)` closes. pandas
        # reads the pattern as written when it picks an engine and never gets
        # past that, so answering here would be answering a question that was
        # already over.
        return compile_program(tree, ENGINE_RE2, minor=minor)
    return compile_program(
        parse_pattern(anchored(method, preprocessed(pattern)), flags),
        ENGINE_RE2,
        captures=method == METHOD_REPLACE,
        minor=minor,
        alphabet=table,
    )
