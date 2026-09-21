"""Which engine answers a pattern.

pandas asks one question of every pattern handed to `contains`, `count`,
`match`, `fullmatch` and `replace`: does Python's parser report a lookaround or
a backreference in it. If it does the call goes to Python's `re`, and if it does
not the call goes to Arrow, which is RE2. The two engines disagree about what
`\\d` means, about what `$` matches and about three other things measured in
document 76, so this decision is visible in answers and not only in refusals.

### Why the walk is incomplete, and why it is copied anyway

The walk pandas does looks for three op codes and recurses into two node kinds:
a subpattern and a branch. There are seven node kinds that can hold another node
underneath them. The five it does not recurse into are the three repeats, the
atomic group and the conditional, so a lookaround with a quantifier on it is
invisible to the decision.

That is not a corner. `(?=a)` is answered by Python and gives a column of
booleans, and `(?=a)?` is routed to RE2, which has never heard of `(?=`, and
raises. A caller who makes an assertion optional gets an Arrow error naming a
library they did not call.

This reproduces it. Reproducing an upstream bug is the kind of decision that has
to be made by a person and written down rather than arrived at by accident, and
the argument is that the alternative is worse in a way that is harder to find: a
library that answers where pandas raises is a library whose divergence only
shows up when somebody moves a program back the other way, and it shows up then
as a wrong answer rather than as an error. The refusal is reproduced in kind
rather than in wording, since pandas' wording names Arrow.

### Why a sequence is transparent here

Python's parser inlines a non capturing group, so `(?:(?=a))` has its assertion
at the level the walk reads and is detected. This parser keeps the group as a
sequence node instead, and the walk below steps through a sequence without
counting it as a level. The two arrive at the same answer for every pattern,
which is the only thing that has to be true.
"""

from firepanda.kernel.regex.parse import NO_NODE, Parsed, parse_pattern
from firepanda.kernel.regex.tokens import (
    OP_ASSERT,
    OP_ASSERT_NOT,
    OP_BRANCH,
    OP_GROUPREF,
    OP_SCOPE,
    OP_SEQ,
    OP_SUBPATTERN,
    Node,
)


comptime ENGINE_RE2: UInt8 = 0
"""Arrow's engine, which is what a pattern gets when nothing routes it away."""

comptime ENGINE_PYTHON: UInt8 = 1
"""Python's `re`, which a pattern reaches by holding a lookaround or a
backreference where the walk can see it."""


def _walks(nodes: List[Node], node: Int32) -> Bool:
    """Whether the walk finds one of the three op codes under a node.

    The recursion is deliberately partial and the five node kinds it does not
    enter are the whole subject of this file's docstring.

    Args:
        nodes: The arena.
        node: Where to start, which is a node whose children are the tokens at
            one level.

    Returns:
        True when a lookaround or a backreference is reachable.
    """
    var child = nodes[Int(node)].first
    while child != NO_NODE:
        var op = nodes[Int(child)].op
        if op == OP_ASSERT or op == OP_ASSERT_NOT or op == OP_GROUPREF:
            return True
        if (
            op == OP_SUBPATTERN
            or op == OP_BRANCH
            or op == OP_SEQ
            or op == OP_SCOPE
        ):
            if _walks(nodes, child):
                return True
        child = nodes[Int(child)].next
    return False


def holds_unsupported(tree: Parsed) -> Bool:
    """Whether a parsed pattern routes to Python.

    Args:
        tree: The pattern as this library read it. A pattern that did not parse
            answers False, because pandas catches the parse error and lets Arrow
            have the pattern.

    Returns:
        True when the walk finds a lookaround or a backreference.
    """
    if not tree.ok:
        return False
    if tree.python_refuses:
        # The same rule as the line above, for a pattern this parser reads and
        # Python's does not. A flag group written where only RE2 takes one
        # parses here because the parser reads both grammars now, and pandas
        # would still have got a parse error out of `re` and handed the pattern
        # to Arrow. What the walk finds under it does not matter, because the
        # engine the walk would route to is the engine that refused it.
        # Document 102.
        return False
    var op = tree.nodes[Int(tree.root)].op
    if op == OP_ASSERT or op == OP_ASSERT_NOT or op == OP_GROUPREF:
        return True
    return _walks(tree.nodes, tree.root)


def engine_for(pattern: StringSlice) -> UInt8:
    """Which engine pandas would answer a pattern out of.

    Flags are not read here. pandas routes to Python whenever a caller passed a
    non zero `flags`, or whenever the pattern is a compiled one carrying flags
    beyond ignore case and Unicode, and both of those are facts about the call
    rather than about the pattern, so they belong to the layer that has the
    call.

    Args:
        pattern: The pattern.

    Returns:
        `ENGINE_PYTHON` or `ENGINE_RE2`.
    """
    var tree = parse_pattern(pattern)
    return ENGINE_PYTHON if holds_unsupported(tree) else ENGINE_RE2


def reads_as_python(pattern: StringSlice) -> Bool:
    """Whether Python's grammar can read a pattern at all.

    Separate from the routing answer because the two are different questions
    with the same input and a caller wanting to explain a decision needs both:
    a pattern can reach RE2 either by holding nothing interesting or by being
    unreadable, and those are not the same situation to report.

    The parser reads more than Python's grammar, so parsing is not the whole of
    the answer. A flag group written where only RE2 takes one is read here and
    carries the sentence Python would have refused it with, and this question is
    about Python, so such a pattern answers False whatever the parse did.

    Args:
        pattern: The pattern.

    Returns:
        True when Python's grammar reads it.
    """
    var tree = parse_pattern(pattern)
    return tree.ok and not tree.python_refuses
