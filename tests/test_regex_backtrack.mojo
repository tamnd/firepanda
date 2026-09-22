"""Tests that the backtracker finds the match the machine finds.

The backtracker is a third engine for the same programs and the only property
worth asserting about it is that it never disagrees with the machine standing
behind it. Not only about whether there is a match, which is the easy half, but
about where it ends and about where every group of it opened and closed, which
is the half a replacement reads and the half a wrong answer hides in.

So most of this file is one loop over pairs of a pattern and a piece of text,
both engines asked, both answers compared down to the last slot. The rest is
what that loop cannot see: the row that is too long for the bitmap, which has to
be an admission rather than a guess, the preference order between two ways of
matching the same text, and the cursor, which is where a replacement scan and a
counting scan disagree about what the text even is.
"""

from std.collections.span import Span
from std.testing import (
    TestSuite,
    assert_equal,
    assert_not_equal,
    assert_true,
)

from firepanda.kernel.regex.backtrack import (
    GAVE_UP,
    MAX_CELLS,
    NO_MATCH,
    Bounded,
)
from firepanda.kernel.regex.parse import decoded, parse_pattern
from firepanda.kernel.regex.pike import Machine
from firepanda.kernel.regex.program import Program, compile_program
from firepanda.kernel.regex.route import ENGINE_PYTHON, ENGINE_RE2


def grouped(pattern: StringSlice, engine: UInt8 = ENGINE_RE2) raises -> Program:
    """A program compiled with the slots a replacement needs.

    Args:
        pattern: The pattern.
        engine: Which engine's reading of it.

    Returns:
        The program.

    Raises:
        Error: If the pattern did not compile, which in this file is a mistake
            in the test rather than an answer.
    """
    var program = compile_program(parse_pattern(pattern), engine, captures=True)
    if not program.ok:
        raise Error(
            String("pattern ", pattern, " did not compile: ", program.problem)
        )
    return program^


def compared(
    program: Program,
    text: StringSlice,
    first: Int = 0,
    advance: Bool = False,
) raises:
    """Asks both engines the same question and fails if they differ.

    Args:
        program: The compiled pattern.
        text: The row.
        first: The cursor.
        advance: Whether a match of no width is refused at the cursor, which is
            what a scan asks for after one and is a question both engines have
            to answer the same way.

    Raises:
        Error: If the two answers differ anywhere, including in one slot.
    """
    var points = decoded(text)
    var machine = Machine(program)
    var bounded = Bounded(program)
    var theirs = List[Int32]()
    var ours = List[Int32]()
    var want = machine.search(program, Span(points), first, theirs, advance)
    var got = bounded.search(program, Span(points), 0, first, ours, advance)
    if got == GAVE_UP:
        return
    assert_equal(
        got,
        want,
        String("pattern against ", repr(String(text)), " from ", first),
    )
    if want < 0:
        return
    assert_equal(len(ours), len(theirs), String("slot count on ", repr(text)))
    for k in range(len(theirs)):
        assert_equal(
            ours[k],
            theirs[k],
            String("slot ", k, " on ", repr(String(text)), " from ", first),
        )


def patterns() -> List[String]:
    """The patterns the agreement loop asks about.

    Returns:
        The list, all of them with a group in them where a group makes sense,
        since the groups are half of what is being compared.
    """
    var out = List[String]()
    out.append(String(""))
    out.append(String("a"))
    out.append(String("abc"))
    out.append(String("a*"))
    out.append(String("(a*)"))
    out.append(String("a+b"))
    out.append(String("ab?c"))
    out.append(String("^abc"))
    out.append(String("abc$"))
    out.append(String("^(a)(b)c$"))
    out.append(String("\\Aab"))
    out.append(String("ab\\z"))
    out.append(String("$"))
    out.append(String("^"))
    out.append(String("a|bc"))
    out.append(String("(a|b)*c"))
    out.append(String("[a-z]+"))
    out.append(String("[^a-z]+"))
    out.append(String("."))
    out.append(String("(?s)(.)"))
    out.append(String("a.c"))
    out.append(String("[0-9]{2,4}"))
    out.append(String("^https?://(?:www\\.)?([^/]+)/.*$"))
    out.append(String("x(y|z)*w"))
    out.append(String("(ab)+"))
    out.append(String("a{3}"))
    out.append(String("^$"))
    out.append(String("^a*$"))
    out.append(String("(a*)*"))
    out.append(String("(|a)*b"))
    out.append(String("((a)|(b))+"))
    out.append(String("é"))
    out.append(String("\\d+"))
    out.append(String("\\bfoo\\b"))
    out.append(String("(\\w+)\\s(\\w+)"))
    out.append(String("(a)(b)?(c)?"))
    out.append(String("(?i)ABC"))
    return out^


def texts() -> List[String]:
    """The pieces of text the agreement loop asks about.

    Returns:
        The list.
    """
    var out = List[String]()
    out.append(String(""))
    out.append(String("a"))
    out.append(String("b"))
    out.append(String("abc"))
    out.append(String("abcdef"))
    out.append(String("xabcdef"))
    out.append(String("aaa"))
    out.append(String("aaaaaaab"))
    out.append(String("ac"))
    out.append(String("abab"))
    out.append(String("cab"))
    out.append(String("http://www.example.com/page/two"))
    out.append(String("ftp://example.com/page"))
    out.append(String("héllo"))
    out.append(String("a foo b"))
    out.append(String("xfoo"))
    out.append(String("one two"))
    out.append(String("ΑΒΓ"))
    out.append(String(" "))
    out.append(String("a\nb"))
    out.append(String("a\n"))
    out.append(String("123456"))
    out.append(String("xyzw"))
    out.append(String("_"))
    return out^


def test_the_backtracker_finds_what_the_machine_finds() raises:
    """Every pattern against every piece of text, both engines, compared down to
    the slots. A disagreement is a bug wherever it is."""
    var all_patterns = patterns()
    var all_texts = texts()
    for p in range(len(all_patterns)):
        var program = grouped(all_patterns[p])
        for t in range(len(all_texts)):
            compared(program, all_texts[t])


def test_they_agree_from_a_cursor_in_the_middle_of_a_row() raises:
    """The replacing scan asks the same row again from where the last match
    ended, so every position of every row is a cursor somebody will pass. The
    row stays whole while the cursor moves, which is what keeps `^` the start of
    the row rather than the start of what is left, and it is the one place where
    the two engines could part company without either of them being wrong about
    a plain search."""
    var all_patterns = patterns()
    var all_texts = texts()
    for p in range(len(all_patterns)):
        var program = grouped(all_patterns[p])
        for t in range(len(all_texts)):
            var length = len(decoded(all_texts[t]))
            for first in range(length + 1):
                compared(program, all_texts[t], first)


def _rolled(mut seed: Int, sides: Int) -> Int:
    """A number below `sides`, from a generator small enough to read.

    Args:
        seed: The state, which this moves on.
        sides: How many values are wanted.

    Returns:
        A number from zero to `sides` less one.
    """
    seed = (seed * 1103515245 + 12345) % 2147483648
    return (seed // 65536) % sides


def _pattern(mut seed: Int) -> String:
    """A pattern built out of pieces, which may or may not compile.

    Args:
        seed: The generator state.

    Returns:
        The pattern.
    """
    var atoms = List[String]()
    atoms.append(String("a"))
    atoms.append(String("b"))
    atoms.append(String("c"))
    atoms.append(String("."))
    atoms.append(String("[ab]"))
    atoms.append(String("[^a]"))
    atoms.append(String("[a-c]"))
    atoms.append(String("\\d"))
    atoms.append(String("\\w"))
    atoms.append(String("\\s"))
    atoms.append(String("\\b"))
    atoms.append(String("x"))
    atoms.append(String("é"))
    var repeats = List[String]()
    repeats.append(String(""))
    repeats.append(String(""))
    repeats.append(String("*"))
    repeats.append(String("+"))
    repeats.append(String("?"))
    repeats.append(String("{2}"))
    repeats.append(String("{1,3}"))
    var out = String("")
    if _rolled(seed, 4) == 0:
        out += "^"
    var pieces = 1 + _rolled(seed, 4)
    for piece in range(pieces):
        if piece > 0 and _rolled(seed, 3) == 0:
            out += "|"
        var body = atoms[_rolled(seed, len(atoms))]
        if _rolled(seed, 3) == 0:
            # A group, because the groups are what this engine has that the
            # state cache does not and they are the part worth generating.
            body = String("(", body, "|", atoms[_rolled(seed, len(atoms))], ")")
        out += body + repeats[_rolled(seed, len(repeats))]
    if _rolled(seed, 4) == 0:
        out += "$"
    return out^


def _text(mut seed: Int) -> String:
    """A piece of text out of the characters the patterns talk about.

    Args:
        seed: The generator state.

    Returns:
        The text.
    """
    var letters = List[String]()
    letters.append(String("a"))
    letters.append(String("a"))
    letters.append(String("b"))
    letters.append(String("b"))
    letters.append(String("c"))
    letters.append(String("x"))
    letters.append(String("1"))
    letters.append(String("_"))
    letters.append(String(" "))
    letters.append(String("\n"))
    letters.append(String("é"))
    letters.append(String("Α"))
    var out = String("")
    var length = _rolled(seed, 13)
    for _ in range(length):
        out += letters[_rolled(seed, len(letters))]
    return out^


def test_they_agree_on_a_generated_corpus() raises:
    """The list above holds the shapes somebody thought of. This is thousands of
    pairs nobody thought of, with a repeated group in a good many of them, which
    is where a walk that takes the arms of a split in the wrong order or a slot
    that is not put back on the way out of a save would show up."""
    var seed = 20260918
    var compares = 0
    for _ in range(500):
        var pattern = _pattern(seed)
        var program = compile_program(
            parse_pattern(pattern), ENGINE_RE2, captures=True
        )
        if not program.ok:
            continue
        for _ in range(8):
            var text = _text(seed)
            compared(program, text)
            compares += 1
    assert_true(compares > 3000, String("only ", compares, " pairs compared"))


def test_the_match_is_the_one_the_pattern_prefers() raises:
    """Leftmost first and not leftmost longest, which is the rule both RE2 and
    Python follow and the one thing a backtracker gets wrong if it searches the
    arms of an alternation in the wrong order. `a|ab` ends after one character
    and `ab|a` ends after two, over the same text."""
    var points = decoded("ab")
    var found = List[Int32]()

    var first = grouped("a|ab")
    var one = Bounded(first)
    assert_equal(one.search(first, Span(points), 0, 0, found), 1)

    var second = grouped("ab|a")
    var two = Bounded(second)
    assert_equal(two.search(second, Span(points), 0, 0, found), 2)


def test_the_leftmost_attempt_wins() raises:
    """The other half of the same rule. An attempt that starts earlier beats one
    that starts later even when the later one matches more, so the answer is
    where the first attempt that can match ends rather than the longest thing in
    the row."""
    var program = grouped("(a+)")
    var points = decoded("ab aaa")
    var bounded = Bounded(program)
    var found = List[Int32]()
    assert_equal(bounded.search(program, Span(points), 0, 0, found), 1)
    assert_equal(found[0], 0)
    assert_equal(found[1], 1)


def test_a_row_too_long_for_the_bitmap_says_so() raises:
    """The bitmap is one bit per instruction per position, so a long enough row
    is one this engine will not take. It has to say so rather than answer
    slowly, because the caller has a machine to run instead and the whole
    arrangement rests on the two never disagreeing."""
    var program = grouped("(a+)b")
    var room = MAX_CELLS // program.sized()
    var short = String("")
    for _ in range(room // 2):
        short += "a"
    var long = String("")
    for _ in range(room + 2):
        long += "a"

    var bounded = Bounded(program)
    var found = List[Int32]()
    assert_equal(
        bounded.search(program, Span(decoded(short)), 0, 0, found),
        NO_MATCH,
    )
    assert_equal(
        bounded.search(program, Span(decoded(long)), 0, 0, found),
        GAVE_UP,
    )


def test_a_pattern_that_can_match_nothing_terminates() raises:
    """A repeat whose body can match nothing is the loop a backtracking engine
    hangs in, and the bitmap is what stops it: the second visit to the same
    instruction at the same position is dropped. The assertion here is that the
    call returns at all, and the value it returns is the machine's."""
    var program = grouped("(a*)*b")
    var points = decoded("aaaaaaaaaaaaaaaaaaaab")
    var bounded = Bounded(program)
    var found = List[Int32]()
    assert_equal(bounded.search(program, Span(points), 0, 0, found), 21)


def test_the_bomb_that_hangs_a_plain_backtracker() raises:
    """`(a+)+b` against a row of letters with no `b` in it is the pattern every
    backtracking engine is measured against, and it is the reason this one is
    not the only engine in the library. The bitmap holds the work to one visit
    per instruction per position, so the answer comes back rather than the run
    hanging."""
    var program = grouped("(a+)+b")
    var row = String("")
    for _ in range(64):
        row += "a"
    var bounded = Bounded(program)
    var found = List[Int32]()
    assert_equal(
        bounded.search(program, Span(decoded(row)), 0, 0, found), NO_MATCH
    )


def test_an_anchored_pattern_starts_one_attempt() raises:
    """A program whose first step asks whether the position is zero has one
    attempt in it. Asked from a cursor above zero it answers nothing, which is
    what the replacing scan reads to stop asking."""
    var program = grouped("^a")
    var points = decoded("aaa")
    var bounded = Bounded(program)
    var found = List[Int32]()
    assert_equal(bounded.search(program, Span(points), 0, 0, found), 1)
    assert_equal(bounded.search(program, Span(points), 0, 1, found), NO_MATCH)


def test_the_row_around_the_cursor_is_the_real_one() raises:
    """The cursor says where an attempt may start and not where the row begins,
    so a word boundary at the cursor reads the character in front of it. Asked
    at position three of `ab cd` there is a boundary and asked at position one
    there is not, and a copy of this engine that cut the row would answer both
    the same way."""
    var program = grouped("\\bcd")
    var points = decoded("ab cd")
    var bounded = Bounded(program)
    var found = List[Int32]()
    assert_equal(bounded.search(program, Span(points), 0, 3, found), 5)

    var inside = grouped("\\bb")
    var other = Bounded(inside)
    assert_equal(
        other.search(inside, Span(points), 0, 1, found),
        NO_MATCH,
    )


def test_a_program_with_no_slots_answers_where_it_ends() raises:
    """A program compiled without captures carries no slots, and the whole of
    what the slot machinery then costs is nothing: the saves are not there to
    walk and the caller's list is left alone. The end of the match is still the
    end of the match."""
    var program = compile_program(parse_pattern("a+b"), ENGINE_RE2)
    assert_true(program.ok)
    assert_equal(program.slots, 0)
    var bounded = Bounded(program)
    var found = List[Int32]()
    assert_equal(
        bounded.search(program, Span(decoded("xxaab")), 0, 0, found), 5
    )
    # Empty rather than untouched, which is what the machine does with the same
    # program: a match empties the list and then writes as many slots as the
    # program carries, and this one carries none.
    assert_equal(len(found), 0)


def test_the_two_engines_agree_on_the_other_interpreter() raises:
    """Everything above is RE2's reading of the patterns. Python's engine reads
    a handful of them differently, most of all the word boundary and the dollar
    sign, and the backtracker asks the same function about those as the machine
    does, so the agreement should survive the change of engine. This is the
    check that says it does rather than the argument that it must."""
    var all_texts = texts()
    var cases = List[String]()
    cases.append(String("(a)$"))
    cases.append(String("\\b(foo)\\b"))
    cases.append(String("\\B(foo)"))
    cases.append(String("(\\w)+"))
    cases.append(String("^(a|b)*$"))
    for c in range(len(cases)):
        var program = grouped(cases[c], ENGINE_PYTHON)
        for t in range(len(all_texts)):
            compared(program, all_texts[t])


def test_they_agree_when_a_match_of_no_width_is_refused() raises:
    """The other thing a scan asks for, which is the position it has already
    matched nothing at being asked again with the end of the pattern refused.
    Both engines have to leave the rest of the search standing when they refuse
    it, so that the arm the pattern liked less gets its turn, and a difference
    between them here is a difference in a count or in a replacement rather than
    in a search anybody calls directly. Document 93 section 10."""
    var all_texts = texts()
    var cases = List[String]()
    cases.append(String("(a*?)"))
    cases.append(String("(b*)|(a)"))
    cases.append(String("\\B|(a)"))
    cases.append(String("(a*)"))
    cases.append(String("()"))
    for c in range(len(cases)):
        var program = grouped(cases[c], ENGINE_PYTHON)
        for t in range(len(all_texts)):
            var length = len(decoded(all_texts[t]))
            for first in range(length + 1):
                compared(program, all_texts[t], first, advance=True)


def test_a_lookahead_alone_is_handed_straight_back() raises:
    """A program this engine can run and does not have to. A lookahead is a
    search inside a search and this engine can walk one, on the stack it
    already has, but the machine next door starts a second machine for it and
    pays nothing here that it does not pay there. So a program whose only
    unusual thing is an assertion goes there, and this engine says so at the
    moment it is sized rather than answering a row the machine would have
    answered the same way for less. Put a backreference beside it and there is
    nowhere else for it to go, and then it runs here. Documents 93 and 120."""
    var program = grouped("a(?=b)", ENGINE_PYTHON)
    var bounded = Bounded(program)
    var found = List[Int32]()
    assert_equal(
        bounded.search(program, Span(decoded("ab")), 0, 0, found), GAVE_UP
    )
    # And the plain pattern next to it, so that the handing back is read as
    # being about the lookahead rather than about the engine.
    var plain = grouped("ab", ENGINE_PYTHON)
    var other = Bounded(plain)
    assert_equal(other.search(plain, Span(decoded("ab")), 0, 0, found), 2)
    # And the same lookahead with a backreference beside it, which is the
    # program this engine is the only one that can run.
    var paired = grouped("(a)\\1(?=b)", ENGINE_PYTHON)
    var third = Bounded(paired)
    assert_true(third.ok)
    assert_equal(third.search(paired, Span(decoded("aab")), 0, 0, found), 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
