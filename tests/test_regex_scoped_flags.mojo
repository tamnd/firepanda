"""`(?i:a)` and the rest of the scoped flag groups, which are a scope.

Every letter was read before this and none of them could be turned on for part
of a pattern, so `(?i:a)b` was refused with `a scoped flag group is not carried
yet` and 525 corpus patterns were counted under that one sentence. The letters
ride on an `OP_SCOPE` node now and the compiler puts them back when the node is
done, which is what makes the `b` above a plain `b`.

Two of the rows here are the ones that separate a scope from a pattern wide
flag. `(?i:a)b` on `aB` is False, which a global `(?i)` would answer True, and
`(?i:(?-i:a)b)` on `aB` is True, which says the inner group put back what the
outer one had rather than what the pattern started with.

Verbose mode is the odd letter out and is spent in the parser rather than in
the compiler, so `(?x:a b)c d` reads as four items where the last two are a
space and a `d`. The comment row is the same trap document 88 section 5 was
about, seen from the other side: `(?x:a#c)b` is an unterminated subpattern
upstream, because the comment eats the closing bracket.

Every row below was asked of a running CPython 3.13.12 and, for the RE2 rows, a
running pandas, before it was written down.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.kernel.regex.parse import parse_pattern
from firepanda.kernel.regex.pike import matches_text
from firepanda.kernel.regex.program import compile_program
from firepanda.kernel.regex.route import (
    ENGINE_PYTHON,
    ENGINE_RE2,
    holds_unsupported,
    reads_as_python,
)
from firepanda.kernel.regex.tokens import (
    FLAG_ASCII,
    FLAG_IGNORECASE,
    FLAG_UNICODE,
    FLAG_VERBOSE,
)


def ours(
    pattern: StringSlice, text: StringSlice, flags: Int32 = 0
) raises -> Bool:
    """Whether a pattern read Python's way under some letters matches the text.

    Args:
        pattern: The pattern.
        text: The text.
        flags: The letters a caller passed beside the pattern, as `FLAG_` bits.

    Returns:
        True when some part of it matches.

    Raises:
        Error: If the pattern did not compile, which in this file is a mistake
            in the test rather than an answer.
    """
    var program = compile_program(parse_pattern(pattern, flags), ENGINE_PYTHON)
    if not program.ok:
        raise Error(
            String("pattern ", pattern, " did not compile: ", program.problem)
        )
    return matches_text(program, text)


def theirs(
    pattern: StringSlice, text: StringSlice, flags: Int32 = 0
) raises -> Bool:
    """The same question asked of RE2's reading, which pandas answers out of
    Arrow.

    Args:
        pattern: The pattern.
        text: The text.
        flags: The letters, as `FLAG_` bits.

    Returns:
        True when some part of it matches.

    Raises:
        Error: If the pattern did not compile.
    """
    var program = compile_program(parse_pattern(pattern, flags), ENGINE_RE2)
    if not program.ok:
        raise Error(
            String("pattern ", pattern, " did not compile: ", program.problem)
        )
    return matches_text(program, text)


def test_a_scope_ends_where_the_bracket_does() raises:
    """The whole point of the node. A global `(?i)` would answer the second row
    True and a dropped letter would answer the first row False."""
    assert_true(ours("(?i:a)b", "Ab"))
    assert_false(ours("(?i:a)b", "aB"))
    assert_true(ours("(?i:a)b", "ab"))
    assert_false(ours("(?i:a)b", "AB"))
    assert_true(ours("a(?i:b)", "aB"))
    assert_false(ours("a(?i:b)", "Ab"))


def test_a_letter_can_be_turned_off_for_part_of_a_pattern() raises:
    """The other direction, which only the scoped form can write at all, since
    `(?-i)` on its own is a parse error rather than a global group with a minus
    sign in it."""
    assert_false(ours("(?-i:a)", "A", FLAG_IGNORECASE))
    assert_true(ours("(?-i:a)", "a", FLAG_IGNORECASE))
    assert_true(ours("(?-i:a)b", "aB", FLAG_IGNORECASE))


def test_the_inner_group_puts_back_what_the_outer_one_had() raises:
    """Nesting, which is the row that says the flags are saved and restored
    rather than cleared. If the restore wrote the pattern's own flags back, the
    second row here would be False and the third would be True."""
    assert_false(ours("(?i:(?-i:a)b)", "AB"))
    assert_true(ours("(?i:(?-i:a)b)", "aB"))
    assert_false(ours("(?i:(?-i:a)b)", "Ab"))
    assert_false(ours("(?i:a(?-i:b))", "AB"))
    assert_true(ours("(?i:a(?-i:b))", "Ab"))


def test_verbose_mode_is_scoped_by_the_parser_and_not_by_the_node() raises:
    """The one letter that is spent before a program exists. The second row is
    the one worth reading: the space between the `c` and the `d` is outside the
    group and is therefore a character the text has to hold."""
    assert_true(ours("(?x:a b)cd", "abcd"))
    assert_false(ours("(?x:a b)c d", "abcd"))
    assert_true(ours("(?x:a b)c d", "abc d"))
    assert_true(ours("(?-x:a b)", "a b", FLAG_VERBOSE))
    assert_false(ours("(?-x:a b)", "ab", FLAG_VERBOSE))
    assert_true(ours("(?x:a#c\nb)", "ab"))


def test_a_comment_still_eats_the_closing_bracket() raises:
    """The same trap as document 88 section 5, met from the caller's side this
    time rather than from the anchoring. A comment ends at a newline or at the
    end of the pattern and a closing bracket is neither, so the group is never
    closed and upstream says so in these words."""
    var tree = parse_pattern("(?x:a#c)b")
    assert_false(tree.ok)
    assert_equal(tree.problem, "missing ), unterminated subpattern")


def test_the_alphabet_narrows_for_part_of_a_pattern_too() raises:
    """`(?a:...)`, which is the letter document 88 added and the one that needs
    the compiler to recompute what it reads off the flags rather than carry a
    second field that can disagree."""
    assert_false(ours("(?a:\\w)", "\u00e9"))
    assert_true(ours("\\w", "\u00e9"))
    assert_true(ours("(?a:\\w)\\w", "a\u00e9"))
    assert_false(ours("(?a:\\w)\\w", "\u00e9a"))
    assert_true(ours("(?a:\\s)", "\x0b"))
    assert_true(ours("(?i:\\w)", "\u017f"))
    assert_false(ours("(?ia:\\w)", "\u017f"))


def test_the_fold_narrows_inside_the_group_and_widens_outside_it() raises:
    """The Kelvin sign, which is the character that tells the two folds apart,
    asked once with the letter and once without it inside the same kind of
    group."""
    assert_true(ours("(?i:k)", "\u212a"))
    assert_false(ours("(?i:(?a:k))", "\u212a"))
    assert_true(ours("(?i:[k])", "\u212a"))
    assert_false(ours("(?i:[^k])", "\u212a"))


def test_the_other_three_letters_are_scoped_as_well() raises:
    """Dotall and multiline, which are read off the builder in one place each
    and so come along for free, and which are worth a row because free is not
    the same as checked."""
    assert_true(ours("(?s:.)", "\n"))
    assert_true(ours("(?s:.)x", "\nx"))
    assert_false(ours(".x", "\nx"))
    assert_true(ours("(?m:^b)", "a\nb"))
    assert_false(ours("^b", "a\nb"))


def test_a_scope_is_not_a_group_anybody_can_refer_to() raises:
    """The numbering, which is the reason this is its own node rather than a
    subpattern with a group number of zero. A scope holding a capture leaves the
    capture where it was and adds nothing of its own."""
    assert_true(ours("(?i:(a))b", "Ab"))
    assert_true(ours("((?i:a))b", "Ab"))
    assert_true(ours("(?:(?i:a))b", "Ab"))
    assert_true(ours("(?i:a)(?i:b)", "AB"))


def test_a_quantifier_repeats_the_scope_and_not_the_letter() raises:
    """A repeat above a scope, which is where a flags field carried on the
    builder and not restored would show up as the letter leaking past the
    bracket after the last pass."""
    assert_true(ours("(?i:a)+b", "Aab"))
    assert_false(ours("(?i:a)+b", "AaB"))


def test_re2_reads_the_three_letters_it_has() raises:
    """The other engine, which has `(?i:...)` and `(?m:...)` and `(?s:...)` of
    its own and which was refused here for the same missing scope. pandas
    answers these out of Arrow and every row was read off it."""
    assert_true(theirs("(?i:a)b", "Ab"))
    assert_false(theirs("(?i:a)b", "aB"))
    assert_true(theirs("(?s:.)", "\n"))
    assert_true(theirs("(?m:^b)", "a\nb"))
    assert_false(theirs("^b", "a\nb"))


def test_re2_still_refuses_the_four_letters_it_never_had() raises:
    """Unchanged, and the reason the set of letters is still recorded on the
    parse beside the node. RE2 reads `(?i:a)` and has never heard of `(?x:a)`
    or `(?a:a)` in any position, so this is agreement with upstream rather than
    a shortfall here."""
    var verbose = compile_program(parse_pattern("(?x:a b)"), ENGINE_RE2)
    assert_false(verbose.ok)
    assert_false(verbose.gap)
    var narrow = compile_program(parse_pattern("(?a:\\w)"), ENGINE_RE2)
    assert_false(narrow.ok)
    assert_false(narrow.gap)
    var wide = compile_program(parse_pattern("(?u:\\w)"), ENGINE_RE2)
    assert_false(wide.ok)
    assert_false(wide.gap)


def test_the_router_walks_into_a_scope() raises:
    """Upstream's walk enters a scoped flag group, because CPython's parser
    hangs one on a `SUBPATTERN` token and the walk recurses into every one of
    those. A node of our own has to be entered on purpose or `(?i:(?=a))` goes
    to an engine that has never heard of a lookahead."""
    assert_true(holds_unsupported(parse_pattern("(?i:(?=a))")))
    assert_true(holds_unsupported(parse_pattern("(?i:(?:(?=a)))")))
    assert_false(holds_unsupported(parse_pattern("(?i:a)")))


def test_the_letters_are_still_refused_where_python_refuses_them() raises:
    """Unchanged by any of this, and here because a scope that carried its
    letters could have been the moment somebody stopped checking them. All four
    are parse errors upstream and the wording is Python's.

    The last of them is read by this parser now, because RE2 reads a flag group
    with no colon in it and the parser reads RE2's grammar too, so Python's
    sentence moved off `problem` and onto `python_problem` and the pattern goes
    to RE2 rather than being refused outright. The sentence is the same one and
    a caller who asked for Python's engine still sees it. Document 102.
    """
    assert_equal(
        parse_pattern("(?-a:x)").problem,
        "bad inline flags: cannot turn off flags 'a', 'u' and 'L'",
    )
    assert_equal(
        parse_pattern("(?L:x)").problem,
        "bad inline flags: cannot use 'L' flag with a str pattern",
    )
    assert_equal(
        parse_pattern("(?au:x)").problem,
        "bad inline flags: flags 'a', 'u' and 'L' are incompatible",
    )
    assert_equal(parse_pattern("(?-i)x").python_problem, "missing :")
    assert_true(parse_pattern("(?-i)x").python_refuses)
    assert_false(reads_as_python("(?-i)x"))


def test_naming_one_alphabet_clears_the_other_two() raises:
    """The three alphabet letters do not combine the way the other four do.

    Upstream clears all three before putting the named one back, so `(?u:...)`
    under a global ascii flag is the wide alphabet rather than both letters at
    once. The other four letters are independent and a plain on and off, which
    is why this rule needed a line rather than coming out of the same one.
    """
    assert_true(ours("(?u:\\w)", "\u00e9", FLAG_ASCII))
    assert_false(ours("\\w", "\u00e9", FLAG_ASCII))
    assert_false(ours("(?a:\\w)", "\u00e9", FLAG_UNICODE))
    assert_true(ours("(?u:\\w)\\w", "\u00e9a", FLAG_ASCII))
    assert_false(ours("(?u:\\w)\\w", "a\u00e9", FLAG_ASCII))
    assert_true(ours("(?a:(?u:k))", "\u212a", FLAG_IGNORECASE))
    assert_false(ours("(?u:(?a:k))", "\u212a", FLAG_IGNORECASE))


def test_the_two_alphabets_are_only_a_conflict_at_the_top() raises:
    """Measured rather than assumed, and it is the one place a scope is not
    simply the global form with a smaller reach. `(?a)` beside `(?u)` is a
    `ValueError` and `(?a:x)` beside `(?u:x)` is a pattern, because Python
    checks the two letters once over the pattern's own flags and never over a
    group's."""
    assert_true(parse_pattern("(?a:x)(?u:x)").ok)
    assert_true(parse_pattern("(?a:(?u:x))").ok)
    assert_true(ours("(?a:\\w)(?u:\\w)", "a\u00e9"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
