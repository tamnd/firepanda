"""Tests for the four Unicode normalization forms.

Every expected answer here was read off CPython's `unicodedata`, because that is
what pandas calls for this name on every backend it has. The generator that
writes the tables has already checked the whole algorithm against CPython over
every code point and a hundred thousand awkward sequences, so what this file is
for is different: it checks that the Mojo kernel does what the Python model in
the generator did, and it pins the handful of rows where a plausible
implementation gives a plausible wrong answer.

Rows go in and come out as lists of code points rather than as string literals.
Nothing here is readable as text anyway, since half the rows are a letter
followed by two combining marks, and a list of numbers cannot be quietly
normalized by an editor on the way into the file. That last one is a real hazard
for a test suite about normalization: a file saved in NFC has no NFD rows left in
it.

The rows were chosen so that each one fails differently.

A letter with one accent, written both ways, is the smallest thing that
distinguishes the four forms from doing nothing.

Two marks in the wrong order catch a missing canonical sort, and two marks of the
same class catch a sort that is not stable.

A letter with a mark of the higher class first catches a missing blocking rule in
the composition, which is the failure that would otherwise show up only as an
answer that depends on the order the marks were written in.

A Hangul syllable catches the arithmetic in both directions, and it is the only
part of this kernel with no table behind it.

The angstrom sign and the Devanagari qa are the two ways a decomposition does not
come back. The first is a singleton, which composes to a different character than
the one it came from. The second is on the composition exclusion list, so it
comes apart and stays apart. Both would pass a test suite that only asked whether
composing after decomposing gives the input back.

The long s with two marks is the example UAX 15 uses to show that the four forms
are not three forms and a shorthand, and it is the row most likely to be wrong in
an implementation that is otherwise right. Its NFC and its NFKC are different
characters, and neither is reachable by composing the other's decomposition.
"""

from std.collections.string import Codepoint
from std.testing import TestSuite, assert_equal, assert_true

from firepanda.array.strings import StringArray, StringBuilder
from firepanda.kernel.normalize import text_normalize


def spelled(points: List[UInt32]) -> String:
    """A run of code points as a string, built without a literal.

    Args:
        points: The code points.

    Returns:
        The text.
    """
    var out = String()
    for i in range(len(points)):
        out += String(Codepoint(unsafe_unchecked_codepoint=points[i]))
    return out^


def one_row(points: List[UInt32]) raises -> StringArray:
    """A column of exactly one element, spelled out.

    Args:
        points: The code points of the element.

    Returns:
        The column.
    """
    var built = StringBuilder(capacity=1)
    var text = spelled(points)
    built.append(text.as_bytes())
    return built^.finish()


def read_row(column: StringArray, i: Int) raises -> List[UInt32]:
    """The code points of one element, read back out.

    Args:
        column: The column.
        i: Which element.

    Returns:
        Its code points, or an empty list when the element is null.
    """
    var out = List[UInt32]()
    if not column.is_valid(i):
        return out^
    var text = StringSlice(unsafe_from_utf8=column.unsafe_bytes(i))
    for point in text.codepoints():
        out.append(point.to_u32())
    return out^


def check(
    input: List[UInt32],
    nfd: List[UInt32],
    nfc: List[UInt32],
    nfkd: List[UInt32],
    nfkc: List[UInt32],
) raises:
    """Runs one row through all four forms and checks all four answers.

    Doing the four together rather than one per test is what makes the rows
    readable as a table, and it is also the only way the difference between the
    forms is visible in the file rather than only in the name of the test.

    Args:
        input: The row.
        nfd: What NFD should give.
        nfc: What NFC should give.
        nfkd: What NFKD should give.
        nfkc: What NFKC should give.
    """
    var column = one_row(input)
    assert_equal(read_row(text_normalize(column, False, False), 0), nfd)
    assert_equal(read_row(text_normalize(column, False, True), 0), nfc)
    assert_equal(read_row(text_normalize(column, True, False), 0), nfkd)
    assert_equal(read_row(text_normalize(column, True, True), 0), nfkc)


def test_a_precomposed_letter() raises:
    """The e with an acute, written as one character."""
    var e: List[UInt32] = [0x00E9]
    var apart: List[UInt32] = [0x0065, 0x0301]
    check(e, apart, e, apart, e)


def test_a_decomposed_letter() raises:
    """The same letter written as two, which normalizes to the same answers."""
    var apart: List[UInt32] = [0x0065, 0x0301]
    var e: List[UInt32] = [0x00E9]
    check(apart, apart, e, apart, e)


def test_a_ligature_is_only_a_compatibility_difference() raises:
    """The fi ligature is left alone by NFD and NFC and split by the K forms."""
    var ligature: List[UInt32] = [0xFB01]
    var letters: List[UInt32] = [0x0066, 0x0069]
    check(ligature, ligature, ligature, letters, letters)


def test_a_circled_digit_loses_its_circle_only_under_k() raises:
    """Compatibility equivalence is content with the formatting thrown away."""
    var circled: List[UInt32] = [0x2460]
    var one: List[UInt32] = [0x0031]
    check(circled, circled, circled, one, one)


def test_a_title_case_digraph_takes_two_steps() raises:
    """Its NFKD is three characters and its NFKC is two, not one.

    The compatibility decomposition of the digraph is a D, a z and a caron, and
    composing that gives a D and a z with caron. There is no single character
    for the pair, so the K forms are the only ones that move it and they do not
    move it to the same place.
    """
    var digraph: List[UInt32] = [0x01C5]
    var apart: List[UInt32] = [0x0044, 0x007A, 0x030C]
    var together: List[UInt32] = [0x0044, 0x017E]
    check(digraph, digraph, digraph, apart, together)


def test_two_marks_are_sorted_by_combining_class() raises:
    """A dot above written before a dot below comes out the other way round.

    The dot below is class 220 and the dot above is 230, so the ordering swaps
    them, and only then can the dot below reach the letter. An implementation
    that skipped the sort would compose nothing here and would answer the input.
    """
    var written: List[UInt32] = [0x0061, 0x0307, 0x0323]
    var ordered: List[UInt32] = [0x0061, 0x0323, 0x0307]
    var composed: List[UInt32] = [0x1EA1, 0x0307]
    check(written, ordered, composed, ordered, composed)


def test_a_mark_of_a_higher_class_blocks_the_one_behind_it() raises:
    """The ogonek is class 202 and the acute is 230, so the acute is blocked.

    The a and the ogonek compose. The acute cannot then reach the composite,
    because the ogonek sits between them with a class that is not lower than the
    acute's, so the answer keeps the acute as a separate character. Without the
    blocking rule the acute would reach the a instead and the answer would
    depend on which mark was written first.
    """
    var written: List[UInt32] = [0x0061, 0x0328, 0x0301]
    var composed: List[UInt32] = [0x0105, 0x0301]
    check(written, written, composed, written, composed)


def test_a_hangul_syllable_comes_apart_and_goes_back() raises:
    """Three jamo out and one syllable back, all of it arithmetic."""
    var syllable: List[UInt32] = [0xAC01]
    var jamo: List[UInt32] = [0x1100, 0x1161, 0x11A8]
    check(syllable, jamo, syllable, jamo, syllable)


def test_hangul_jamo_written_out_compose() raises:
    """The same three going the other way, which is the second formula."""
    var jamo: List[UInt32] = [0x1100, 0x1161, 0x11A8]
    var syllable: List[UInt32] = [0xAC01]
    check(jamo, jamo, syllable, jamo, syllable)


def test_a_hangul_syllable_with_no_trailing_consonant() raises:
    """Two jamo rather than three, which is the trail index of zero."""
    var syllable: List[UInt32] = [0xAC00]
    var jamo: List[UInt32] = [0x1100, 0x1161]
    check(syllable, jamo, syllable, jamo, syllable)


def test_a_singleton_composes_to_a_different_character() raises:
    """The angstrom sign decomposes to an A with a ring and comes back as the
    ordinary letter, not as the sign it started as."""
    var sign: List[UInt32] = [0x212B]
    var apart: List[UInt32] = [0x0041, 0x030A]
    var letter: List[UInt32] = [0x00C5]
    check(sign, apart, letter, apart, letter)


def test_an_excluded_pair_comes_apart_and_stays_apart() raises:
    """Devanagari qa is on the composition exclusion list.

    It has a canonical decomposition, so it comes apart under all four forms,
    and the pair is excluded from composition, so nothing puts it back. A
    composition table built by reading every two character decomposition without
    applying the exclusions would answer the input here.
    """
    var qa: List[UInt32] = [0x0958]
    var apart: List[UInt32] = [0x0915, 0x093C]
    check(qa, apart, apart, apart, apart)


def test_the_example_where_nfc_and_nfkc_are_different_characters() raises:
    """The long s with a dot above, followed by a dot below.

    UAX 15 uses this to show the four forms are four. Under NFC the long s keeps
    its dot above and the dot below stays loose. Under NFKC the long s has
    already become an ordinary s, so both dots land on it and the answer is one
    character that the NFC answer has no route to.
    """
    var written: List[UInt32] = [0x1E9B, 0x0323]
    var nfd: List[UInt32] = [0x017F, 0x0323, 0x0307]
    var nfc: List[UInt32] = [0x1E9B, 0x0323]
    var nfkd: List[UInt32] = [0x0073, 0x0323, 0x0307]
    var nfkc: List[UInt32] = [0x1E69]
    check(written, nfd, nfc, nfkd, nfkc)


def test_a_dotted_capital_i_does_not_recompose_further() raises:
    """It comes apart into an I and a dot and goes back to itself."""
    var dotted: List[UInt32] = [0x0130]
    var apart: List[UInt32] = [0x0049, 0x0307]
    check(dotted, apart, dotted, apart, dotted)


def test_two_marks_of_the_same_class_keep_their_order() raises:
    """Which is what stable means, and the reason the sort is an insertion sort.

    Both marks are class 230, so neither is sorted before the other and the
    answer is the input. A sort that swapped equal elements would give the two
    marks the other way round, which is a different string.
    """
    var written: List[UInt32] = [0x0071, 0x0301, 0x0300]
    var other_way: List[UInt32] = [0x0071, 0x0300, 0x0301]
    var column = one_row(written)
    assert_equal(read_row(text_normalize(column, False, False), 0), written)
    assert_true(read_row(text_normalize(column, False, False), 0) != other_way)


def test_a_null_element_stays_null() raises:
    """Under all four forms."""
    var built = StringBuilder(capacity=2)
    built.append_null()
    built.append("a".as_bytes())
    var column = built^.finish()
    var forms: List[Bool] = [False, True]
    for full in forms:
        for compose in forms:
            var out = text_normalize(column, full, compose)
            assert_true(not out.is_valid(0))
            assert_true(out.is_valid(1))


def test_an_empty_element_stays_empty() raises:
    """Normalization never removes the last character and never invents one."""
    var empty: List[UInt32] = []
    check(empty, empty, empty, empty, empty)


def test_an_ascii_element_is_its_own_answer() raises:
    """The fast path, which is only sound because no ASCII character moves."""
    var text: List[UInt32] = [0x0041, 0x007A, 0x0030, 0x005F, 0x0020, 0x000A]
    check(text, text, text, text, text)


def test_a_column_with_no_rows() raises:
    """Nothing to walk and a column of the same height comes back."""
    var built = StringBuilder(capacity=0)
    var column = built^.finish()
    assert_equal(len(text_normalize(column, False, True)), 0)


def test_every_form_is_idempotent() raises:
    """Normalizing an answer again gives the same answer.

    That is the property the whole thing exists for. Anything that failed it
    would mean two strings a reader calls the same could still come out of here
    as different bytes, which is the problem normalization is supposed to solve.
    """
    var rows: List[List[UInt32]] = [
        [0x00E9],
        [0x0065, 0x0301],
        [0xFB01],
        [0x01C5],
        [0x0061, 0x0307, 0x0323],
        [0x0061, 0x0328, 0x0301],
        [0xAC01],
        [0x212B],
        [0x0958],
        [0x1E9B, 0x0323],
    ]
    var forms: List[Bool] = [False, True]
    for row in rows:
        for full in forms:
            for compose in forms:
                var once = text_normalize(one_row(row), full, compose)
                var twice = text_normalize(once, full, compose)
                assert_equal(read_row(twice, 0), read_row(once, 0))


def test_a_column_of_many_rows_keeps_them_apart() raises:
    """Several rows in one column, because the payload is one buffer.

    Each row here decomposes to a different length, so a kernel that wrote the
    answers out without keeping the offsets straight would run them together and
    every row after the first would be wrong.
    """
    var built = StringBuilder(capacity=5)
    var rows: List[List[UInt32]] = [
        [0x00E9],
        [0xAC01],
        [0x0041],
        [0x1E9B, 0x0323],
        [0x0061, 0x0307, 0x0323],
    ]
    for row in rows:
        var text = spelled(row)
        built.append(text.as_bytes())
    var column = built^.finish()
    var out = text_normalize(column, False, False)
    assert_equal(len(out), 5)
    var first: List[UInt32] = [0x0065, 0x0301]
    var second: List[UInt32] = [0x1100, 0x1161, 0x11A8]
    var third: List[UInt32] = [0x0041]
    var fourth: List[UInt32] = [0x017F, 0x0323, 0x0307]
    var fifth: List[UInt32] = [0x0061, 0x0323, 0x0307]
    assert_equal(read_row(out, 0), first)
    assert_equal(read_row(out, 1), second)
    assert_equal(read_row(out, 2), third)
    assert_equal(read_row(out, 3), fourth)
    assert_equal(read_row(out, 4), fifth)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
