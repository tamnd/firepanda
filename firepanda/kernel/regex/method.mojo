"""Which pattern each of the four methods actually runs, and who runs it.

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

This file is the layer above the compiler and below the binding. It knows what
pandas does with a pattern before handing it over, and it hands back a program
or a refusal, which is the same pair the compiler deals in. What it deliberately
does not know is what a refusal should become in Python, because that is a fact
about the binding.
"""

from firepanda.kernel.regex.parse import parse_pattern
from firepanda.kernel.regex.program import Program, compile_program
from firepanda.kernel.regex.route import ENGINE_RE2, holds_unsupported


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
        method: Which of the four asked.
        pattern: The pattern as the caller wrote it, with `\\Z` already seen to.

    Returns:
        The pattern the engine is to be given.
    """
    if method == METHOD_CONTAINS or method == METHOD_COUNT:
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


def program_for(method: UInt8, pattern: String) -> Program:
    """Compiles what one of the three methods would run, or refuses it.

    The routing decision is made first and on the pattern as written, which is
    this file's docstring and is pandas' order. A pattern routed to Python is
    refused here rather than compiled, because refusing it through the compiler
    would mean compiling the rewritten pattern to find out something that was
    already known about the original.

    The `\\Z` rewrite comes next and the anchoring last, which is upstream's
    order as well and is not an order either step is indifferent to.

    Args:
        method: Which of the four asked.
        pattern: The pattern as the caller wrote it.

    Returns:
        The program, or the reason there is not one, with the flag saying whose
        refusal it is. A refusal is a value here rather than a raise for the
        reason `compile_program` gives.
    """
    var tree = parse_pattern(pattern)
    var out = Program()
    if holds_unsupported(tree):
        out.ok = False
        out.problem = String("the Python engine is not written yet")
        out.gap = True
        return out^
    if not tree.ok:
        # A pattern the grammar cannot read is refused over what the caller
        # wrote rather than over the rewrite, because the rewrite can make a
        # bad pattern good and upstream would still have refused it. `)a` is
        # the shape: a bracket nobody opened, which `^()a)` closes. pandas
        # reads the pattern as written when it picks an engine and never gets
        # past that, so answering here would be answering a question that was
        # already over.
        return compile_program(tree, ENGINE_RE2)
    return compile_program(
        parse_pattern(anchored(method, preprocessed(pattern))), ENGINE_RE2
    )
