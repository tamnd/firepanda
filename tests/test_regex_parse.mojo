"""Tests that a pattern is read the way Python reads it.

The differential next door asks thirty thousand generated patterns whether this
parser and Python's agree about two yes or no questions, and it is the stronger
test by a wide margin. This one exists because agreement on a yes or no answer
is not agreement about a tree: a parser can refuse the same patterns Python
refuses and still build the wrong thing out of the ones it accepts, and the
wrong thing is invisible to the router right up until the matching engine is
wired to it.

So the assertions here are shapes rather than verdicts. Each one is a pattern
and the tree it produces, written out, and the value of writing it out is that a
change to the parser which moves a node somewhere else has to come here and say
so.

The patterns are the corners rather than a sample. Each one was measured against
`re._parser.parse` on this interpreter before it was written down, and several
of them are corners this parser got wrong on the first attempt: what a
quantifier repeats when a comment is in the way, when a backslash and a digit
are an octal number instead of a backreference, and where Python checks that a
group exists.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.kernel.regex.parse import NO_NODE, parse_pattern
from firepanda.kernel.regex.tokens import (
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
    OP_NOT_LITERAL,
    OP_POSSESSIVE_REPEAT,
    OP_RANGE,
    OP_SCOPE,
    OP_SEQ,
    OP_SUBPATTERN,
    Node,
)


def shown(value: Int32) -> String:
    """A code point, as a character when it is one somebody can read.

    Args:
        value: The code point.

    Returns:
        The character itself, or the number after a hash, so that a code point
        of forty eight and the digit zero do not draw the same.
    """
    if value >= 0x21 and value < 0x7F:
        return String(chr(Int(value)))
    return String("#", value)


def counted(value: Int32) -> String:
    """A repeat bound.

    Args:
        value: The bound.

    Returns:
        The number, or `inf` for the one that means no ceiling.
    """
    if value == MAXREPEAT:
        return String("inf")
    return String(value)


def drawn(nodes: List[Node], node: Int32) -> String:
    """One node and everything under it, as text.

    The form is a name, the payloads that node uses in brackets, and the
    children in braces. Nodes whose payloads mean nothing print neither, so a
    shape stays short enough to read in one line.

    Args:
        nodes: The arena.
        node: Where to start.

    Returns:
        The drawing.
    """
    var it = nodes[Int(node)]
    var out = String("")
    if it.op == OP_LITERAL:
        out = String("lit(", shown(it.a), ")")
    elif it.op == OP_NOT_LITERAL:
        out = String("notlit(", shown(it.a), ")")
    elif it.op == OP_ANY:
        out = String("any")
    elif it.op == OP_IN:
        out = String("in")
    elif it.op == OP_RANGE:
        out = String("range(", shown(it.a), ",", shown(it.b), ")")
    elif it.op == OP_CATEGORY:
        out = String("cat(", it.a, ")")
    elif it.op == OP_NEGATE:
        out = String("neg")
    elif it.op == OP_AT:
        out = String("at(", it.a, ")")
    elif it.op == OP_BRANCH:
        out = String("branch")
    elif it.op == OP_SEQ:
        out = String("seq")
    elif it.op == OP_SUBPATTERN:
        out = String("group(", it.a, ")")
    elif it.op == OP_MAX_REPEAT:
        out = String("max(", counted(it.a), ",", counted(it.b), ")")
    elif it.op == OP_MIN_REPEAT:
        out = String("min(", counted(it.a), ",", counted(it.b), ")")
    elif it.op == OP_POSSESSIVE_REPEAT:
        out = String("poss(", counted(it.a), ",", counted(it.b), ")")
    elif it.op == OP_ATOMIC_GROUP:
        out = String("atomic")
    elif it.op == OP_ASSERT:
        out = String("assert(", it.a, ")")
    elif it.op == OP_ASSERT_NOT:
        out = String("assertnot(", it.a, ")")
    elif it.op == OP_GROUPREF:
        out = String("ref(", it.a, ")")
    elif it.op == OP_GROUPREF_EXISTS:
        out = String("cond(", it.a, ")")
    elif it.op == OP_FAILURE:
        out = String("fail")
    elif it.op == OP_SCOPE:
        out = String("scope(", it.a, ",", it.b, ")")
    else:
        out = String("op", it.op)

    var child = it.first
    if child == NO_NODE:
        return out^
    out += "{"
    var first = True
    while child != NO_NODE:
        if not first:
            out += ","
        out += drawn(nodes, child)
        first = False
        child = nodes[Int(child)].next
    out += "}"
    return out^


def shape(pattern: StringSlice) -> String:
    """A pattern's tree, or why there is not one.

    Args:
        pattern: The pattern.

    Returns:
        The drawing, or an exclamation mark and the problem.
    """
    var tree = parse_pattern(pattern)
    if not tree.ok:
        return String("!", tree.problem)
    return drawn(tree.nodes, tree.root)


def reason(pattern: StringSlice) -> String:
    """What Python says about a pattern this parser reads and Python does not.

    Args:
        pattern: The pattern.

    Returns:
        Python's sentence, or empty when Python reads it too.
    """
    var tree = parse_pattern(pattern)
    if not tree.ok or not tree.python_refuses:
        return String("")
    return tree.python_problem.copy()


def test_a_plain_pattern_is_a_sequence_of_literals() raises:
    """The simplest tree there is, which fixes the shape everything else is
    written against."""
    assert_equal(shape("abc"), "seq{lit(a),lit(b),lit(c)}")


def test_an_alternation_is_a_branch_of_sequences() raises:
    """Every alternative is a sequence even when it holds one thing, which is
    what makes the walk in the router uniform."""
    assert_equal(shape("a|bc"), "branch{seq{lit(a)},seq{lit(b),lit(c)}}")


def test_an_empty_alternative_is_an_empty_sequence() raises:
    """A bar with nothing after it is legal and matches the empty text, so the
    tree has to have somewhere to put nothing."""
    assert_equal(shape("a|"), "branch{seq{lit(a)},seq}")


def test_a_class_of_one_thing_stays_a_class() raises:
    """One of Python's collapses that is deliberately not reproduced.

    `re._parser` folds `[a]` into a literal and `[^a]` into a single negated
    one. Neither fold is visible from outside: no character class can hold a
    lookaround or a backreference, so folding one changes nothing the router
    reads and nothing about whether the pattern parses. Keeping the class keeps
    the shape the caller wrote, which is the shape the matching engine would
    rather compile, and the file this tests says so in as many words.
    """
    assert_equal(shape("[a]"), "seq{in{lit(a)}}")
    assert_equal(shape("[^a]"), "seq{in{neg,lit(a)}}")


def test_a_negate_is_a_child_of_the_class_and_not_a_flag_on_it() raises:
    """Where the caret goes in the tree is Python's choice and this copies
    it."""
    assert_equal(shape("[^ab]"), "seq{in{neg,lit(a),lit(b)}}")


def test_a_range_keeps_both_ends() raises:
    """A range is one node rather than the characters spelled out, which is the
    only representation that survives a class covering a large alphabet."""
    assert_equal(shape("[a-f]"), "seq{in{range(a,f)}}")


def test_a_dash_at_the_end_of_a_class_is_a_literal_dash() raises:
    """There is nothing after it to be the top of a range, so it is itself."""
    assert_equal(shape("[a-]"), "seq{in{lit(a),lit(-)}}")


def test_a_class_can_hold_a_category() raises:
    """`\\d` in brackets is the same node as `\\d` outside them, which matters
    because the two engines disagree about what it means and the disagreement
    has to be in one place."""
    assert_equal(shape("[\\da]"), "seq{in{cat(1),lit(a)}}")


def test_a_non_capturing_group_leaves_a_sequence() raises:
    """Python inlines the group into the list around it and this keeps a
    sequence node, which the router steps through without counting. The two
    arrive at the same routing answer, which is the whole requirement."""
    assert_equal(shape("(?:ab)"), "seq{seq{lit(a),lit(b)}}")


def test_a_capturing_group_is_numbered_from_one() raises:
    """Group numbers are what a backreference and a conditional are written
    against, so they are a fact about the tree rather than bookkeeping."""
    assert_equal(
        shape("(a)(b)"), "seq{group(1){seq{lit(a)}},group(2){seq{lit(b)}}}"
    )


def test_a_quantifier_repeats_the_last_item_and_not_the_whole_sequence() raises:
    """The rule that decides what `ab*` means.

    Getting this wrong makes a parser that reads every pattern and matches the
    wrong text, which is exactly the class of mistake a yes or no differential
    cannot see.
    """
    assert_equal(shape("ab*"), "seq{lit(a),max(0,inf){lit(b)}}")


def test_a_comment_leaves_nothing_for_a_quantifier_to_repeat() raises:
    """`a(?#c)+` is `a+`.

    This is the pattern that made the quantifier move out of its own function
    and into the sequence reader. A comment produces no item at all, so the plus
    reaches back past it to the `a`, which a reader holding only the item it
    just read cannot do.
    """
    assert_equal(shape("a(?#c)+"), "seq{max(1,inf){lit(a)}}")


def test_a_backslash_in_a_comment_hides_the_bracket_after_it() raises:
    """`(?#\\)` is not closed.

    Python reads a comment through the same tokenizer as everything else and
    that tokenizer hands back an escape as one token, so the closing bracket is
    never seen as one. It is an accident of the reader rather than a rule
    anybody wrote, and it is copied because a caller writing a path in a comment
    meets it.
    """
    assert_equal(shape("(?#\\)"), "!missing ), unterminated comment")
    assert_equal(shape("(?#a\\\\)b"), "seq{lit(b)}")


def test_the_three_quantifier_kinds_are_three_nodes() raises:
    """Greedy, lazy and possessive, where the third is the one RE2 refuses."""
    assert_equal(shape("a*"), "seq{max(0,inf){lit(a)}}")
    assert_equal(shape("a*?"), "seq{min(0,inf){lit(a)}}")
    assert_equal(shape("a*+"), "seq{poss(0,inf){lit(a)}}")


def test_counted_repeats_carry_both_bounds() raises:
    """All four forms of the braces, including the two with a side missing."""
    assert_equal(shape("a{2,3}"), "seq{max(2,3){lit(a)}}")
    assert_equal(shape("a{2}"), "seq{max(2,2){lit(a)}}")
    assert_equal(shape("a{2,}"), "seq{max(2,inf){lit(a)}}")
    assert_equal(shape("a{,3}"), "seq{max(0,3){lit(a)}}")


def test_braces_that_are_not_a_repeat_are_characters() raises:
    """`a{}` is three literals.

    Python does not refuse a brace it cannot read as a count, it puts it back
    and treats it as text, and a parser that refused here would turn a pattern
    pandas answers into one it sends to Arrow.
    """
    assert_equal(shape("a{}"), "seq{lit(a),lit({),lit(})}")
    assert_equal(shape("a{2"), "seq{lit(a),lit({),lit(2)}")


def test_a_repeat_with_the_bounds_the_wrong_way_round_is_refused() raises:
    """The one thing about the braces Python does refuse."""
    assert_equal(shape("a{3,2}"), "!min repeat greater than max repeat")


def test_nothing_to_repeat_and_multiple_repeat_are_different_refusals() raises:
    """Both are parse failures and both route the pattern to Arrow, so keeping
    them apart is for the caller reading the message rather than for the
    router."""
    assert_equal(shape("*"), "!nothing to repeat")
    assert_equal(shape("a**"), "!multiple repeat")


def test_an_anchor_cannot_be_repeated() raises:
    """`^*` and `\\b*` are refused, because a position is not a thing that can
    happen twice."""
    assert_equal(shape("^*"), "!nothing to repeat")
    assert_equal(shape("\\b*"), "!nothing to repeat")


def test_a_group_can_be_repeated() raises:
    """The other side of the same rule, which is the case that would have made a
    too eager refusal obvious."""
    assert_equal(shape("(?:a)*"), "seq{max(0,inf){seq{lit(a)}}}")


def test_a_lookaround_carries_its_direction() raises:
    """Four nodes for four constructs, and the direction is a payload rather
    than four op codes because the matching engine treats it as a payload."""
    assert_equal(shape("(?=a)"), "seq{assert(1){seq{lit(a)}}}")
    assert_equal(shape("(?!a)"), "seq{assertnot(1){seq{lit(a)}}}")
    assert_equal(shape("(?<=a)"), "seq{assert(-1){seq{lit(a)}}}")
    assert_equal(shape("(?<!a)"), "seq{assertnot(-1){seq{lit(a)}}}")


def test_an_empty_negative_lookaround_collapses_to_a_failure() raises:
    """`(?!)` is not an assertion at all once Python is done with it.

    A body that always matches makes a negative lookaround never match, so the
    parser replaces the whole thing. The consequence is the routing: there is no
    assertion left for the walk to find, so the pattern goes to RE2, which has
    never heard of `(?!` and raises.
    """
    assert_equal(shape("(?!)"), "seq{fail}")
    assert_equal(shape("(?<!)"), "seq{fail}")


def test_an_empty_positive_lookaround_stays_an_assertion() raises:
    """The same pattern with one character changed keeps the node and therefore
    keeps the engine, which is the comparison that makes the collapse above a
    finding rather than a curiosity."""
    assert_equal(shape("(?=)"), "seq{assert(1){seq}}")
    assert_equal(shape("(?<=)"), "seq{assert(-1){seq}}")


def test_a_backreference_is_checked_where_it_is_written() raises:
    """`(a)\\1` reads and `\\1(a)` does not.

    Python checks a backreference against the groups opened so far, right there,
    which means the same two tokens in the other order are a parse error.
    """
    assert_equal(shape("(a)\\1"), "seq{group(1){seq{lit(a)}},ref(1)}")
    assert_equal(shape("\\1(a)"), "!invalid group reference")


def test_a_group_cannot_refer_to_itself_while_it_is_open() raises:
    """A separate refusal from the one above, because the group does exist by
    then and still cannot be used."""
    assert_equal(shape("(a\\1)"), "!cannot refer to an open group")


def test_a_conditional_is_checked_at_the_end_of_the_parse() raises:
    """`(?(1)a)(b)` reads and the group it names is opened afterwards.

    This is the other half of the rule above and it goes the other way, which is
    the sort of asymmetry that only gets copied by someone who read the source
    rather than reasoned about what would be sensible.
    """
    assert_equal(
        shape("(?(1)a)(b)"),
        "seq{cond(1){seq{lit(a)}},group(1){seq{lit(b)}}}",
    )
    assert_equal(shape("(?(2)a)(b)"), "!invalid group reference")


def test_a_conditional_can_have_a_second_branch_and_not_a_third() raises:
    """Two is the whole grammar here."""
    assert_equal(
        shape("(a)(?(1)b|c)"),
        "seq{group(1){seq{lit(a)}},cond(1){seq{lit(b)},seq{lit(c)}}}",
    )
    assert_equal(
        shape("(a)(?(1)b|c|d)"),
        "!conditional backref with more than two branches",
    )


def test_a_named_group_is_numbered_like_any_other() raises:
    """The name is a way of writing the number rather than a separate thing, so
    the tree holds the number."""
    assert_equal(shape("(?P<n>a)(?P=n)"), "seq{group(1){seq{lit(a)}},ref(1)}")
    assert_equal(shape("(?P=n)"), "!unknown group name")
    assert_equal(shape("(?P<n>a)(?P<n>b)"), "!redefinition of group name")


def test_a_name_has_to_look_like_an_identifier() raises:
    """Python's rule and not a wider one, so a name with a dash in it is
    refused rather than read as far as the dash."""
    assert_equal(shape("(?P<1n>a)"), "!bad character in group name")
    assert_equal(shape("(?P<>a)"), "!missing group name")


def test_an_octal_escape_is_read_when_the_digits_allow_it() raises:
    """The rule that decides whether a backslash and some digits are a number or
    a reference.

    Python takes a third digit only when the two it has are both octal and the
    next one is too, and prefers a group reference otherwise. The first version
    of this parser had a rule that was close and not the same, which the
    generated corpus found in a few hundred patterns.
    """
    assert_equal(shape("\\101"), "seq{lit(A)}")
    assert_equal(shape("\\0"), "seq{lit(#0)}")
    assert_equal(shape("\\08"), "seq{lit(#0),lit(8)}")
    assert_equal(shape("\\400"), "!octal escape value outside of range 0-0o377")


def test_a_single_digit_escape_is_a_reference_when_a_group_exists() raises:
    """The same characters mean different things depending on what came
    before."""
    assert_equal(shape("(a)\\1"), "seq{group(1){seq{lit(a)}},ref(1)}")
    assert_equal(shape("\\8"), "!invalid group reference")


def test_the_two_end_anchors_are_different_nodes() raises:
    """`$` and `\\Z` are not the same position, and the difference is one of the
    places the two engines disagree."""
    assert_equal(shape("a$"), "seq{lit(a),at(4)}")
    assert_equal(shape("a\\Z"), "seq{lit(a),at(6)}")
    assert_equal(shape("\\Aa"), "seq{at(3),lit(a)}")


def test_a_global_flag_group_produces_nothing_and_has_to_be_first() raises:
    """Python 3.11 made the position a rule, which turns a pattern that used to
    read into one that routes to Arrow.

    At the front the group applies to the whole pattern and leaves no node at
    all, which is Python's reading and RE2's too, since the enclosing group a
    flag runs to the end of is the pattern itself. Several in a row are fine
    and a comment does not count as something in front.
    """
    assert_equal(shape("(?i)a"), "seq{lit(a)}")
    assert_equal(shape("(?i)(?s)a"), "seq{lit(a)}")


def test_a_flag_group_anywhere_else_scopes_the_rest_of_the_group() raises:
    """Which is RE2's rule and not one Python has, so the tree is read and the
    refusal moves to Python's compile.

    An alternation counts as something in front even when the part before the
    bar is empty, which is why `|(?i)a` is one of these and not a global one.
    """
    assert_equal(shape("a(?i)b"), "seq{lit(a),scope(1,0){seq{lit(b)}}}")
    assert_equal(shape("|(?i)a"), "branch{seq,seq{scope(1,0){seq{lit(a)}}}}")
    assert_equal(
        reason("a(?i)b"), "global flags not at the start of the expression"
    )


def test_a_flag_group_stops_at_the_bracket_and_crosses_the_bar() raises:
    """Both halves measured against the RE2 inside pyarrow, since guessing at
    either would have been wrong.

    `((?i))c` does not fold the `c` and `x(?i)x|c` does fold it, so the scope
    ends at the closing bracket of the group it is in and a bar is not a
    boundary at all. A later alternative gets a scope of its own rather than
    the alternation getting one, because wrapping the alternation would change
    which alternatives there are.
    """
    assert_equal(shape("((?i))c"), "seq{group(1){seq},lit(c)}")
    assert_equal(
        shape("a(?i)b|c"),
        "branch{seq{lit(a),scope(1,0){seq{lit(b)}}},scope(1,0){seq{lit(c)}}}",
    )
    assert_equal(
        shape("c|x(?i)x"),
        "branch{seq{lit(c)},seq{lit(x),scope(1,0){seq{lit(x)}}}}",
    )


def test_a_repeat_written_after_a_flag_group_reaches_back_past_it() raises:
    """`xa(?i)*b` matches `xab` and `xaB` and not `xAb`, so the star repeats
    the `a` and the `a` is outside the scope while the `b` is inside it.

    That is why the scope is built once the sequence has ended rather than as
    the group is read. Reading it as a split would put the `a` out of the
    repeat's reach, and the repeat's rule is that it takes the last item the
    sequence holds.
    """
    assert_equal(
        shape("xa(?i)*b"),
        "seq{lit(x),max(0,inf){lit(a)},scope(1,0){seq{lit(b)}}}",
    )
    assert_equal(shape("a(?i)"), "seq{lit(a)}")


def test_two_flag_groups_in_one_sequence_nest() raises:
    """The second one is inside the first, so what follows it carries both
    letters, which is what a flag that accumulates from a place has to mean."""
    assert_equal(
        shape("a(?i)b(?-s)c"),
        "seq{lit(a),scope(1,0){seq{lit(b),scope(0,8){seq{lit(c)}}}}}",
    )


def test_a_comment_does_not_count_as_something_in_front() raises:
    """A comment produces nothing, so a flag group after one is still at the
    start."""
    assert_equal(shape("(?#c)(?i)a"), "seq{lit(a)}")


def test_flags_can_only_be_turned_off_in_the_scoped_form() raises:
    """`(?-i)` is not a global flag group with a minus sign, because Python has
    no such thing, and it is RE2's flag turned off from here to the end of the
    group, which is the kind of reading a caller would never predict.

    The scoped form keeps its letters on a node of its own, with the ones it
    turns on first and the ones it turns off second, and the two payloads are
    `FLAG_` bits rather than the values `re` gives the same letters."""
    assert_equal(shape("(?-i:a)"), "seq{scope(0,1){seq{lit(a)}}}")
    assert_equal(shape("(?i:a)"), "seq{scope(1,0){seq{lit(a)}}}")
    assert_equal(shape("(?i-s:a)"), "seq{scope(1,8){seq{lit(a)}}}")
    assert_equal(shape("(?-i)a"), "seq{scope(0,1){seq{lit(a)}}}")
    assert_equal(shape("(?i-s)a"), "seq{scope(1,8){seq{lit(a)}}}")
    assert_equal(reason("(?-i)a"), "missing :")
    assert_equal(reason("(?i-s)a"), "missing :")


def test_the_two_alphabets_cannot_both_be_turned_on() raises:
    """In one group Python says so where it is written, and across two it says
    so at the end of the parse by raising something pandas does not catch.

    This reads both as a pattern that did not parse, which routes them to Arrow.
    Document 76 section 8 is the argument for that being the better of the two
    wrong answers, since the alternative is reproducing an exception that comes
    out of a caller's `str.contains` naming a module they never imported.
    """
    assert_equal(
        shape("(?au:a)"),
        "!bad inline flags: flags 'a', 'u' and 'L' are incompatible",
    )
    assert_equal(shape("(?a)(?u)"), "!ASCII and UNICODE flags are incompatible")


def test_an_atomic_group_is_read_and_kept() raises:
    """Python reads it and RE2 refuses it, which is why it is a node here rather
    than a refusal."""
    assert_equal(shape("(?>a)"), "seq{atomic{seq{lit(a)}}}")


def test_a_lookbehind_bounds_what_a_reference_can_name() raises:
    """A group opened inside a lookbehind cannot be named from inside it, and
    one opened before the lookbehind can."""
    assert_equal(
        shape("(a)(?<=\\1)"),
        "seq{group(1){seq{lit(a)}},assert(-1){seq{ref(1)}}}",
    )
    assert_equal(
        shape("(?<=(a)\\1)"),
        "!cannot refer to group defined in the same lookbehind subpattern",
    )


def test_an_unbalanced_bracket_is_reported_from_either_side() raises:
    """Two different messages for two different mistakes, both of which route
    the pattern to Arrow."""
    assert_equal(shape("(a"), "!missing ), unterminated subpattern")
    assert_equal(shape("a)"), "!unbalanced parenthesis")
    assert_equal(shape("[a"), "!unterminated character set")


def test_a_named_character_escape_is_read_but_not_resolved() raises:
    """There is no Unicode name table here yet.

    What decides whether the pattern parses is the braces rather than what is
    between them, so the router gets the right answer for every name that
    exists, and the wrong one for a name that does not: Python refuses an
    unknown name and this accepts it. The node left behind holds a placeholder
    rather than the character, which is why the parse says so through
    `approximate` instead of leaving a pattern that quietly matches the
    replacement character.
    """
    assert_equal(shape("\\N{GREEK SMALL LETTER ALPHA}"), "seq{lit(#65533)}")
    assert_true(parse_pattern("\\N{GREEK SMALL LETTER ALPHA}").approximate)
    assert_false(parse_pattern("a").approximate)
    assert_equal(shape("\\N{}"), "!missing character name")
    assert_equal(shape("\\Na"), "!missing {")


def test_a_parse_that_failed_reports_why_and_carries_no_tree() raises:
    """The contract the router relies on: `ok` is the only field safe to read
    first."""
    var tree = parse_pattern("(a")
    assert_false(tree.ok)
    assert_equal(tree.problem, "missing ), unterminated subpattern")
    var good = parse_pattern("(a)")
    assert_true(good.ok)
    assert_equal(good.problem, "")
    assert_equal(good.groups, 1)


def test_group_names_come_back_in_the_order_they_were_opened() raises:
    """What a caller needs to turn a name into a column label later."""
    var tree = parse_pattern("(?P<first>a)(b)(?P<second>c)")
    assert_true(tree.ok)
    assert_equal(tree.groups, 3)
    assert_equal(len(tree.names), 2)
    assert_equal(tree.names[0], "first")
    assert_equal(tree.names[1], "second")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
