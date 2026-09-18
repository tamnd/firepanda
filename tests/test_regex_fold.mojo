"""Tests the one flag that is spent while the pattern is being compiled.

`(?i)` never reaches the machine. A literal becomes the set of everything that
folds onto it, a class has the folds of everything in it added before the caret
is applied, and what runs afterwards is an ordinary program that has never heard
of a flag. So the tests here are compiler tests wearing a match: every one of
them asks whether a pattern matches a row, and what it is really asking is what
the compiler put in the set.

Three of them are the ones a table alone would get wrong. A negated class has to
be folded before it is negated, since folding the complement puts the small `a`
back in through the other case of every letter that is not `a`. A letter with no
other case has to stay one code point, since a set of one turns the commonest
instruction in the program into a binary search. And the four Turkish I code
points are the whole of the difference between the two engines, so every one of
them is asserted twice, once per engine, with different answers expected.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.kernel.regex.method import (
    METHOD_CONTAINS,
    METHOD_MATCH,
    program_for,
)
from firepanda.kernel.regex.parse import parse_pattern
from firepanda.kernel.regex.pike import matches_text
from firepanda.kernel.regex.program import (
    IN_CHAR,
    IN_SET,
    Program,
    compile_program,
)
from firepanda.kernel.regex.route import ENGINE_PYTHON, ENGINE_RE2


def built(pattern: String, engine: UInt8) raises -> Program:
    """Compiles a pattern for one engine, or fails the test.

    Args:
        pattern: The pattern.
        engine: Which engine is to run it.

    Returns:
        The program.

    Raises:
        Error: If it did not compile, which in this file is a mistake in the
            test rather than an answer.
    """
    var program = compile_program(parse_pattern(pattern), engine)
    if not program.ok:
        raise Error(
            String("pattern ", pattern, " did not compile: ", program.problem)
        )
    return program^


def hits(pattern: String, engine: UInt8, row: String) raises -> Bool:
    """Whether a pattern finds anything in a row, on one engine.

    Args:
        pattern: The pattern.
        engine: Which engine is to run it.
        row: The text.

    Returns:
        True when there is a match anywhere in the row.
    """
    return matches_text(built(pattern, engine), row)


def test_a_letter_matches_both_of_its_cases() raises:
    """The whole of what a caller writing the flag is asking for."""
    for engine in [ENGINE_RE2, ENGINE_PYTHON]:
        assert_true(hits("(?i)a", engine, "a"))
        assert_true(hits("(?i)a", engine, "A"))
        assert_false(hits("(?i)a", engine, "b"))
        assert_true(hits("(?i)ABC", engine, "abc"))


def test_a_pattern_without_the_flag_is_left_alone() raises:
    """The flag is read off the tree rather than assumed, so the case that has
    to keep working is the one nobody wrote a flag on."""
    for engine in [ENGINE_RE2, ENGINE_PYTHON]:
        assert_false(hits("a", engine, "A"))
        assert_false(hits("[a-z]", engine, "K"))


def test_a_character_with_no_other_case_stays_one_instruction() raises:
    """A digit is not cased, so folding has nothing to add and the program
    should hold the comparison rather than a set of one. Reading the
    instruction rather than the answer, because both spellings answer the same
    and only one of them is the reason the fast path exists."""
    var program = built("(?i)1a", ENGINE_RE2)
    assert_equal(program.code[0].op, IN_CHAR)
    assert_equal(program.code[1].op, IN_SET)


def test_a_class_folds_every_letter_in_its_range() raises:
    """The fold runs over the code points the class covers rather than over the
    ranges it was written with, which is why the Kelvin sign and the long s
    arrive from a range that names neither."""
    for engine in [ENGINE_RE2, ENGINE_PYTHON]:
        assert_true(hits("(?i)[a-z]", engine, "K"))
        assert_true(hits("(?i)[a-z]", engine, "\u212a"))
        assert_true(hits("(?i)[A-Z]", engine, "\u017f"))
        assert_false(hits("(?i)[a-z]", engine, "0"))


def test_a_negated_class_is_folded_before_it_is_negated() raises:
    """The order is the whole of this one. Negating first and folding after
    would put the small `a` back in, because every letter other than `a` is in
    the complement and every one of them brings its own other case with it."""
    for engine in [ENGINE_RE2, ENGINE_PYTHON]:
        assert_false(hits("(?i)[^a]", engine, "A"))
        assert_false(hits("(?i)[^a]", engine, "a"))
        assert_true(hits("(?i)[^a]", engine, "b"))


def test_a_negated_literal_folds_the_same_way() raises:
    """Python's parser writes a one character negated class as its own node
    rather than as a class, so it is a second place the fold has to be spent
    and would be a silent hole if it were not."""
    for engine in [ENGINE_RE2, ENGINE_PYTHON]:
        assert_false(hits("(?i)[^a]b", engine, "Ab"))
        assert_true(hits("(?i)[^a]b", engine, "cb"))


def test_the_two_engines_part_company_over_the_turkish_i() raises:
    """The one code point family where the two disagree, measured over every
    cased code point rather than assumed. Python reads all four as one letter
    and RE2 reads the two plain ones as one letter and the other two as
    themselves."""
    assert_true(hits("(?i)i", ENGINE_PYTHON, "\u0130"))
    assert_true(hits("(?i)i", ENGINE_PYTHON, "\u0131"))
    assert_false(hits("(?i)i", ENGINE_RE2, "\u0130"))
    assert_false(hits("(?i)i", ENGINE_RE2, "\u0131"))


def test_the_plain_pair_folds_on_both_engines() raises:
    """The half of that family the two engines agree about, which is what makes
    the other half a difference rather than a missing table."""
    for engine in [ENGINE_RE2, ENGINE_PYTHON]:
        assert_true(hits("(?i)i", engine, "I"))
        assert_true(hits("(?i)I", engine, "i"))


def test_a_turkish_letter_matches_itself_on_the_engine_that_leaves_it_alone() raises:
    """The pattern written the other way round, which is the case that would
    pass by accident if the exception dropped the code point out of its own
    group rather than out of everybody else's."""
    assert_true(hits("(?i)\u0130", ENGINE_RE2, "\u0130"))
    assert_false(hits("(?i)\u0130", ENGINE_RE2, "i"))
    assert_true(hits("(?i)\u0130", ENGINE_PYTHON, "i"))


def test_a_turkish_letter_inside_a_class_takes_the_same_exception() raises:
    """The class path reaches the table through a different function from the
    literal path, so the exception has to hold on both or one of them is
    quietly reading Python's answer for RE2."""
    assert_false(hits("(?i)[i]", ENGINE_RE2, "\u0131"))
    assert_true(hits("(?i)[i]", ENGINE_PYTHON, "\u0131"))
    assert_true(hits("(?i)[^i]", ENGINE_RE2, "\u0131"))
    assert_false(hits("(?i)[^i]", ENGINE_PYTHON, "\u0131"))


def test_pythons_classes_are_already_closed_under_folding() raises:
    """Python's three Perl classes are Unicode, so `\\w` holds every letter of
    both cases already and the flag has nothing to add to it and nothing to take
    away. The same goes for the complement, which is what makes the order the
    negation is applied in invisible on this engine."""
    assert_true(hits("(?i)\\w", ENGINE_PYTHON, "A"))
    assert_true(hits("(?i)\\w", ENGINE_PYTHON, "\u212a"))
    assert_false(hits("(?i)\\W", ENGINE_PYTHON, "\u212a"))
    assert_false(hits("(?i)[^\\w]", ENGINE_PYTHON, "A"))
    assert_false(hits("(?i)\\d", ENGINE_PYTHON, "a"))


def test_re2s_classes_are_ascii_and_are_not_closed_under_folding() raises:
    """The one that a table alone gets wrong. RE2's `\\w` is `[0-9A-Za-z_]`, and
    the Kelvin sign and the long s are outside it and fold onto letters that are
    inside it, so folding widens a class that is spelled in ASCII to two code
    points that are not. `(?i)\\W` has to be the negation of that wider class and
    not the fold of the narrower one's complement, which would hold a plain
    `k`."""
    assert_true(hits("(?i)\\w", ENGINE_RE2, "\u212a"))
    assert_true(hits("(?i)\\w", ENGINE_RE2, "\u017f"))
    assert_false(hits("(?i)\\w", ENGINE_RE2, "\u00e9"))
    assert_false(hits("(?i)\\W", ENGINE_RE2, "\u212a"))
    assert_false(hits("(?i)\\W", ENGINE_RE2, "k"))
    assert_true(hits("(?i)\\W", ENGINE_RE2, "\u00e9"))


def test_a_negated_class_inside_brackets_is_folded_before_it_is_negated() raises:
    """The same rule reached through the other door. `[\\W]` is a class holding
    one item and the item is a negation, so the fold has to be spent inside the
    item rather than on the union the brackets build. Folding the union would
    put a `k` in `(?i)[\\W]` on RE2, which neither engine does."""
    assert_false(hits("(?i)[\\W]", ENGINE_RE2, "k"))
    assert_false(hits("(?i)[\\W]", ENGINE_RE2, "\u212a"))
    assert_true(hits("(?i)[^\\W]", ENGINE_RE2, "\u212a"))
    assert_true(hits("(?i)[\\Wq]", ENGINE_RE2, "Q"))
    assert_false(hits("(?i)[\\W]", ENGINE_PYTHON, "k"))
    assert_true(hits("(?i)[^\\W]", ENGINE_PYTHON, "\u212a"))


def test_the_word_boundary_is_not_folded_on_either_engine() raises:
    """The one question `(?i)` leaves alone, and it was measured rather than
    assumed because the class it asks about is the one the flag does widen.

    `\\b` asks whether a word character meets a non word one, and each engine
    asks it against its own word class, unfolded. So on RE2 the Kelvin sign is
    not a word character even under the flag, and `(?i)\\bk` misses it at the
    start of a row and finds it after a letter. On Python it is a word
    character, and the same pattern over the same two rows answers the other way
    round. Both of those are pandas' answers today."""
    assert_false(hits("(?i)\\bk", ENGINE_RE2, "\u212a"))
    assert_true(hits("(?i)\\bk", ENGINE_RE2, "a\u212a"))
    assert_true(hits("(?i)\\bk", ENGINE_PYTHON, "\u212a"))
    assert_false(hits("(?i)\\bk", ENGINE_PYTHON, "a\u212a"))


def test_the_digit_class_is_closed_on_both_engines() raises:
    """`\\d` has no cased code point in it on either engine, so it is the class
    the flag really does leave alone and the control on the two above."""
    for engine in [ENGINE_RE2, ENGINE_PYTHON]:
        assert_false(hits("(?i)\\d", engine, "a"))
        assert_true(hits("(?i)\\d", engine, "5"))
        assert_true(hits("(?i)\\D", engine, "a"))


def test_a_letter_whose_other_case_is_two_letters_folds_to_the_one_that_exists() raises:
    """The German sharp s has a capital of its own, which both engines fold it
    onto, and a two letter uppercase form, which neither of them reaches because
    folding is a question about single code points."""
    for engine in [ENGINE_RE2, ENGINE_PYTHON]:
        assert_true(hits("(?i)\u00df", engine, "\u1e9e"))
        assert_false(hits("(?i)ss", engine, "\u00df"))


def test_a_letter_with_three_cases_folds_onto_all_of_them() raises:
    """Greek sigma is the group of three, and a final sigma is the member a
    table built from lowercasing alone would leave out."""
    for engine in [ENGINE_RE2, ENGINE_PYTHON]:
        assert_true(hits("(?i)\u03c3", engine, "\u03c2"))
        assert_true(hits("(?i)\u03c2", engine, "\u03a3"))


def test_the_flag_is_no_longer_a_refusal() raises:
    """It used to be refused by both engines with a sentence saying no table had
    been written, which is the sentence document 83 removed."""
    var program = program_for(METHOD_CONTAINS, "(?i)a")
    assert_true(program.ok)
    assert_equal(program.problem, "")


def test_the_flag_still_reaches_the_pattern_the_anchoring_wrote() raises:
    """`match` hoists the flag group in front of its own rewrite, so the tree
    the compiler reads has the flag on it and the anchors outside where no flag
    can touch them. The fold has to survive that hoist."""
    var program = program_for(METHOD_MATCH, "(?i)b")
    assert_true(program.ok)
    assert_true(matches_text(program, "B"))
    assert_false(matches_text(program, "aB"))


def test_a_scoped_flag_group_folds_inside_the_bracket_and_not_past_it() raises:
    """This was the refusal `a scoped flag group is not carried yet` until the
    letters got a node to ride on. The second row is what the sentence was
    protecting against, since a tree with the letters thrown away answers it
    False and a pattern wide `(?i)` answers it True."""
    var program = program_for(METHOD_CONTAINS, "(?i:b)c")
    assert_true(program.ok)
    assert_true(matches_text(program, "Bc"))
    assert_false(matches_text(program, "bC"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
