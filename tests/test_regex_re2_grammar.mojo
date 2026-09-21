"""Reading a pattern with RE2's grammar.

Every case in here was measured against the RE2 that Arrow is built with rather
than read out of a manual, which matters because the two disagree in places and
because a manual will not tell you that `(a{100}){11}` is refused while both of
its numbers are legal on their own. Document 101 has the measurements. The
differential `tests/differential/regex_re2.mojo` checks the same reader against
a running Arrow over the whole thirty thousand pattern corpus, and this file is
the part a person can read to find out what the rule is meant to be.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.kernel.regex.re2 import re2_reads


def _takes(pattern: String) raises:
    """Asserts RE2 reads a pattern.

    Args:
        pattern: The pattern.

    Raises:
        Error: If the reader refused it.
    """
    var read = re2_reads(pattern)
    if not read.ok:
        raise Error(
            String("expected RE2 to read ") + pattern + ": " + read.problem
        )


def _refuses(pattern: String, said: String) raises:
    """Asserts RE2 refuses a pattern, for a named reason.

    Args:
        pattern: The pattern.
        said: The refusal expected.

    Raises:
        Error: If it was read, or refused for another reason.
    """
    var read = re2_reads(pattern)
    assert_false(read.ok, String("expected RE2 to refuse ") + pattern)
    assert_equal(read.problem, said)


def test_the_plainest_patterns() raises:
    """A pattern with nothing unusual in it reads."""
    _takes(String(""))
    _takes(String("abc"))
    _takes(String("a|b"))
    _takes(String("a*b+c?"))
    _takes(String("(a)(b)"))
    _takes(String("[a-z]+"))
    _takes(String("^a$"))
    _takes(String("."))


def test_an_empty_branch_is_a_branch() raises:
    """A bar with nothing on one side of it is a pattern both grammars read."""
    _takes(String("|"))
    _takes(String("||"))
    _takes(String("a|"))
    _takes(String("|a"))
    _takes(String("(|)"))
    _takes(String("()"))


def test_the_escape_letters_outside_a_class() raises:
    """The letters RE2 knows after a backslash, and some it does not."""
    _takes(String("\\a"))
    _takes(String("\\f"))
    _takes(String("\\n"))
    _takes(String("\\r"))
    _takes(String("\\t"))
    _takes(String("\\v"))
    _takes(String("\\d"))
    _takes(String("\\s"))
    _takes(String("\\w"))
    _takes(String("\\D"))
    _takes(String("\\S"))
    _takes(String("\\W"))
    _takes(String("\\b"))
    _takes(String("\\B"))
    _takes(String("\\A"))
    _takes(String("\\z"))
    _takes(String("\\C"))
    var said = String("RE2 has no such escape")
    _refuses(String("\\Z"), said)
    _refuses(String("\\e"), said)
    _refuses(String("\\h"), said)
    _refuses(String("\\k"), said)
    _refuses(String("\\u0041"), said)
    _refuses(String("\\U00000041"), said)
    _refuses(String("\\N{BULLET}"), said)


def test_the_escape_letters_inside_a_class() raises:
    """The set inside a class is smaller, because an assertion is not a
    character."""
    _takes(String("[\\n]"))
    _takes(String("[\\t]"))
    _takes(String("[\\d]"))
    _takes(String("[\\W]"))
    var said = String("RE2 has no such escape")
    _refuses(String("[\\b]"), said)
    _refuses(String("[\\z]"), said)
    _refuses(String("[\\A]"), said)
    _refuses(String("[\\B]"), said)
    _refuses(String("[\\C]"), said)


def test_punctuation_after_a_backslash() raises:
    """Any ASCII punctuation may be escaped, and nothing outside ASCII may."""
    _takes(String("\\."))
    _takes(String("\\*"))
    _takes(String("\\\\"))
    _takes(String("\\-"))
    _takes(String("\\{"))
    _takes(String("\\ "))
    _refuses(String("\\é"), String("RE2 has no such escape"))


def test_a_trailing_backslash() raises:
    """A backslash with nothing after it has a complaint of its own."""
    var said = String("a backslash is the last thing in the pattern")
    _refuses(String("\\"), said)
    _refuses(String("ab\\"), said)


def test_octal_needs_two_digits_unless_it_starts_at_zero() raises:
    """Which is how RE2 says it has no backreference."""
    _takes(String("\\0"))
    _takes(String("\\00"))
    _takes(String("\\000"))
    _takes(String("\\07"))
    _takes(String("\\08"))
    _takes(String("\\12"))
    _takes(String("\\101"))
    _takes(String("\\400"))
    _takes(String("\\1000"))
    var said = String("RE2 has no such escape")
    _refuses(String("\\1"), said)
    _refuses(String("\\18"), said)
    _refuses(String("\\8"), said)
    _refuses(String("\\9"), said)
    _refuses(String("(a)\\1"), said)


def test_hexadecimal() raises:
    """Two digits, or any number of them inside braces and inside Unicode."""
    _takes(String("\\x41"))
    _takes(String("\\x{4}"))
    _takes(String("\\x{41}"))
    _takes(String("\\x{10FFFF}"))
    _takes(String("\\x{D800}"))
    var said = String("RE2 has no such escape")
    _refuses(String("\\x"), said)
    _refuses(String("\\x4"), said)
    _refuses(String("\\x{}"), said)
    _refuses(String("\\x{110000}"), said)
    _refuses(String("\\xzz"), said)
    _refuses(String("\\x{4"), said)
    _refuses(String("\\x4g"), said)


def test_quoting_a_run() raises:
    """A quote runs to its close or to the end, and not inside a class."""
    _takes(String("\\Qa\\E"))
    _takes(String("\\Q"))
    _takes(String("\\Qa"))
    _takes(String("\\Q\\E"))
    _takes(String("\\Q*\\E"))
    _takes(String("\\Q[\\E"))
    var said = String("RE2 has no such escape")
    _refuses(String("\\E"), said)
    _refuses(String("a\\E"), said)
    _refuses(String("[\\Qa\\E]"), said)
    # The closer is two literal characters and nothing in the run is an escape,
    # so this one ends at the second backslash and the `E` after it is a letter
    # in the run rather than a second close.
    _takes(String("\\Qa\\\\E"))
    _refuses(String("\\Qa\\E\\E"), said)


def test_an_empty_quote_leaves_nothing_for_a_repeat() raises:
    """The same stack rule the flag group above follows. Document 105."""
    _takes(String("x\\Q\\E*"))
    _takes(String("\\Qab\\E*"))
    _takes(String("\\Qa\\E\\Q\\E*"))
    var said = String("there is nothing here for that repeat to repeat")
    _takes(String("\\Q*"))
    _refuses(String("\\Q\\E*"), said)
    _refuses(String("\\Q\\E?"), said)
    _refuses(String("\\Q\\E{2}"), said)
    _refuses(String("\\Q\\E\\Q\\E*"), said)
    _refuses(String("a|\\Q\\E*"), said)
    _refuses(String("(\\Q\\E*)"), said)


def test_the_bracket_forms_re2_has() raises:
    """Five of them, against the dozen Python has."""
    _takes(String("(a)"))
    _takes(String("(?:a)"))
    _takes(String("(?P<n>a)"))
    _takes(String("(?<n>a)"))
    _takes(String("(?i)"))
    _takes(String("(?i:a)"))


def test_the_bracket_forms_re2_has_not() raises:
    """Every lookaround, every conditional, and the comment group."""
    var said = String("RE2 has no group written that way")
    _refuses(String("(?=a)"), said)
    _refuses(String("(?!a)"), said)
    _refuses(String("(?<=a)"), said)
    _refuses(String("(?<!a)"), said)
    _refuses(String("(?(1)a|b)"), said)
    _refuses(String("(?>a)"), said)
    _refuses(String("(?#a)"), said)
    _refuses(String("(?P=n)"), said)
    _refuses(String("(?P>n)"), said)
    _refuses(String("(?&n)"), said)
    _refuses(String("(?|a)"), said)
    _refuses(String("(?'n'a)"), said)
    _refuses(String("(?x)"), said)
    _refuses(String("(?a)"), said)
    _refuses(String("(?L)"), said)
    _refuses(String("(?u)"), said)


def test_a_flag_group_may_stand_anywhere() raises:
    """Which is the single largest thing RE2 reads and Python does not."""
    _takes(String("a(?i)b"))
    _takes(String(".(?m)"))
    _takes(String("((?i))"))
    _takes(String("(?-i)"))
    _takes(String("(?i-s)"))
    _takes(String("(?imsU)"))
    _takes(String("(?U)"))
    _takes(String("(?)"))
    _takes(String("(?ii)"))
    _takes(String("(?i-i)"))
    _takes(String("(?-i:a)"))


def test_a_flag_group_that_is_not_one() raises:
    """A minus sign has to be turning something off."""
    var said = String("RE2 has no group written that way")
    _refuses(String("(?-)"), said)
    _refuses(String("(?i-)"), said)
    _refuses(String("(?iX)"), said)
    _refuses(String("(?i"), said)


def test_a_repeat_after_a_flag_group_reaches_past_it() raises:
    """Because RE2 takes whatever is on top of its stack and a flag group puts
    nothing there, which makes the same repeat legal or not by what came
    before."""
    _refuses(
        String("(?i)*"),
        String("there is nothing here for that repeat to repeat"),
    )
    _takes(String("a(?i)*"))
    _takes(String("(?i:a)*"))
    _takes(String("(?im:(?s))(?i-s){1,3}?"))


def test_group_names() raises:
    """Wider than Python's in one direction and narrower in another, and a
    duplicate is allowed."""
    _takes(String("(?P<n>a)"))
    _takes(String("(?P<1>a)"))
    _takes(String("(?P<1n>a)"))
    _takes(String("(?P<n_1>a)"))
    _takes(String("(?P<é>a)"))
    _takes(String("(?P<n>a)(?P<n>b)"))
    _takes(String("(?P<x١>a)"))
    _takes(String("(?P<x‿>a)"))
    var said = String("RE2 will not take that group name")
    _refuses(String("(?P<n n>a)"), said)
    _refuses(String("(?P<>a)"), said)
    _refuses(String("(?P<n->a)"), said)
    _refuses(String("(?<>a)"), said)
    _refuses(String("(?P<n"), said)
    _refuses(String("(?<n"), said)
    # Not every character outside ASCII, which is what this file said until
    # document 107 measured the rule one code point at a time. A vulgar
    # fraction, a currency sign and a middle dot are all refused, and the first
    # two of those are the categories `\p{No}` and `\p{Sc}` name.
    _refuses(String("(?P<x½>a)"), said)
    _refuses(String("(?P<x€>a)"), said)
    _refuses(String("(?P<x·>a)"), said)


def test_a_bracket_nothing_closes() raises:
    """And a bracket nothing opened, which is a different complaint."""
    var opened = String("a bracket is opened that nothing closes")
    _refuses(String("("), opened)
    _refuses(String("(a"), opened)
    _refuses(String("(()"), opened)
    _refuses(String("(?:"), opened)
    _refuses(String("(?:a"), opened)
    var closed = String("a bracket is closed that nothing opened")
    _refuses(String(")"), closed)
    _refuses(String("a)"), closed)
    _refuses(String("())"), closed)


def test_a_repeat_with_nothing_to_repeat() raises:
    """At the front of a pattern, after a bar, and inside an empty group."""
    var said = String("there is nothing here for that repeat to repeat")
    _refuses(String("*"), said)
    _refuses(String("+"), said)
    _refuses(String("?"), said)
    _refuses(String("{2}"), said)
    _refuses(String("(*)"), said)
    _refuses(String("|*"), said)


def test_a_repeat_on_a_repeat() raises:
    """One repeat per atom, and the ungreedy marker is not a second one."""
    _takes(String("a*"))
    _takes(String("a*?"))
    _takes(String("a{2}"))
    _takes(String("a{2,}?"))
    var said = String("RE2 will not repeat a repeat")
    _refuses(String("a*+"), said)
    _refuses(String("a*??"), said)
    _refuses(String("a**"), said)
    _refuses(String("a*?*"), said)
    _refuses(String("a{2}{2}"), said)
    _refuses(String("a?{2}"), said)
    _refuses(String("a{2}*"), said)


def test_which_complaint_wins_when_a_repeat_has_two() raises:
    """The order RE2 checks in, which is not the order anybody writes in.

    A repeat on a repeat is named before a number out of range, and a number out
    of range is named before there being nothing to repeat, so each of these
    patterns has two things wrong with it and only one of them is reported.
    """
    var stacked = String("RE2 will not repeat a repeat")
    _refuses(String("a{2}{3,1}"), stacked)
    _refuses(String("a?{99999}"), stacked)
    var sized = String("RE2 will not repeat that many times")
    _refuses(String("({99999}"), sized)
    _refuses(String("|{3,1}"), sized)


def test_an_assertion_may_be_repeated() raises:
    """Which Python calls nothing to repeat and RE2 reads without comment."""
    _takes(String("^*"))
    _takes(String("$*"))
    _takes(String("\\b*"))
    _takes(String("\\A+"))
    _takes(String("\\B{0}"))


def test_the_count_limit() raises:
    """A thousand, and it is a budget rather than a ceiling."""
    _takes(String("a{0}"))
    _takes(String("a{1000}"))
    _takes(String("a{0,1000}"))
    var said = String("RE2 will not repeat that many times")
    _refuses(String("a{1001}"), said)
    _refuses(String("a{0,1001}"), said)


def test_the_count_budget_is_shared_by_a_nest() raises:
    """Every one of these numbers is legal on its own."""
    _takes(String("(a{100}){10}"))
    _takes(String("(a{31}){32}"))
    _takes(String("(a{2}){2}"))
    _takes(String("((a{10}){10}){10}"))
    var said = String("RE2 will not repeat that many times")
    _refuses(String("(a{100}){11}"), said)
    _refuses(String("((a{10}){10}){11}"), said)


def test_the_budget_is_spent_down_one_branch() raises:
    """Two branches of a bar do not add up, because only one of them runs."""
    _takes(String("(a{500}|b{500}){2}"))


def test_a_brace_that_opens_no_count_is_a_literal() raises:
    """Which is where a count with no bottom goes, and Python has no such
    reading."""
    _takes(String("a{,}"))
    _takes(String("a{,2}"))
    _takes(String("a{}"))
    _takes(String("a{a}"))
    _takes(String("a{2,"))
    _takes(String("a{2"))
    _takes(String("a{-1}"))
    _takes(String("a{1,2,3}"))


def test_a_class_that_is_never_closed() raises:
    """And the empty class, which is the same complaint because a closing
    bracket written first is a literal."""
    var said = String("a class is never closed")
    _refuses(String("[]"), said)
    _refuses(String("[^]"), said)
    _refuses(String("[a"), said)
    _refuses(String("[^a"), said)


def test_a_closing_bracket_first_in_a_class_is_a_literal() raises:
    """Which is eleven of the corpus patterns RE2 reads and Python does not."""
    _takes(String("[]a]"))
    _takes(String("[^]a]"))
    _takes(String("[\\]]"))


def test_ranges_in_a_class() raises:
    """A hyphen at either end is a hyphen, and one in the middle is a range."""
    _takes(String("[a]"))
    _takes(String("[-]"))
    _takes(String("[a-]"))
    _takes(String("[-a]"))
    _takes(String("[^-]"))
    _takes(String("[a-b-c]"))
    _takes(String("[a-a]"))
    _takes(String("[--0]"))
    _takes(String("[é-漢]"))
    _takes(String("[\\t-a]"))
    var said = String("a range in a class runs backwards")
    _refuses(String("[b-a]"), said)
    _refuses(String("[漢-é]"), said)


def test_a_set_is_not_one_end_of_a_range() raises:
    """So the hyphen after one is a hyphen, and the one before it is not."""
    _takes(String("[\\d-a]"))
    _takes(String("[\\d-\\w]"))
    _refuses(String("[a-\\d]"), String("RE2 has no such escape"))


def test_the_posix_classes() raises:
    """Fourteen names, and an opening that is never closed is two
    characters."""
    _takes(String("[[:alpha:]]"))
    _takes(String("[[:^alpha:]]"))
    _takes(String("[[:alpha:][:digit:]]"))
    _takes(String("[[:]]"))
    _takes(String("[[:alpha]]"))
    _takes(String("[:alpha:]"))
    _refuses(String("[[:foo:]]"), String("RE2 has no such character class"))


def test_a_unicode_class_is_judged_like_everything_else() raises:
    """This was the one construct the reader would not judge, for want of a
    name table. The table is in `unicodedata.mojo` now, so a name RE2 has comes
    back read and a name it has not comes back refused.

    `pl` and `Latin` are the pair worth having, because a name is case
    sensitive to RE2 and the lower case one is not a name at all.
    """
    _takes(String("\\pL"))
    _takes(String("\\p{Latin}"))
    _takes(String("\\p{^L}"))
    _takes(String("\\PL"))
    _takes(String("[\\p{L}]"))
    _takes(String("\\p{Any}"))
    _refuses(String("\\p{Foo}"), String("RE2 has no such character class"))
    _refuses(String("\\p{Cn}"), String("RE2 has no such character class"))
    _refuses(String("\\pl"), String("RE2 has no such character class"))
    _refuses(String("\\p{"), String("RE2 has no such character class"))
    _refuses(String("\\p{latin}"), String("RE2 has no such character class"))


def test_a_bad_name_is_refused_beside_something_else_re2_refuses() raises:
    """There is no longer anything this file declines to judge, so a pattern
    holding a bad name and a lookahead is two refusals rather than one refusal
    and one shrug, and the first one is the one reported."""
    var read = re2_reads(String("\\p{Foo}(?=a)"))
    assert_false(read.ok)
    assert_equal(read.problem, String("RE2 has no such character class"))


def test_the_first_refusal_is_the_one_reported() raises:
    """So the message names where RE2 would have stopped."""
    var read = re2_reads(String("(?=a)\\1"))
    assert_false(read.ok)
    assert_equal(read.problem, String("RE2 has no group written that way"))


def main() raises:
    """Runs the suite.

    Raises:
        Error: If a case fails.
    """
    TestSuite.discover_tests[__functions_in_module()]().run()
