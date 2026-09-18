"""Tests that the state cache answers what the machine answers.

The cache is a second engine for the same programs, so the test that matters is
not that it gets any particular pattern right but that it never disagrees with
the machine standing behind it. So most of this file is one loop: a few dozen
patterns, a few dozen pieces of text, every pair asked of both and the two
answers compared. A disagreement is a bug wherever it is.

The rest of it is the three things that loop cannot see. Which patterns the
cache refuses, since a refusal is not a wrong answer and a silently wrong answer
would look like one. That a cache built once and used for many rows keeps
answering and stops growing, which is the whole reason it exists. And what
happens when a pattern needs more states than the bound allows, which has to be
an admission rather than a guess.
"""

from std.collections.span import Span
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)

from firepanda.kernel.regex.dfa import (
    MAX_STATES,
    SCAN_GAVE_UP,
    SCAN_NO,
    SCAN_YES,
    Cache,
)
from firepanda.kernel.regex.parse import decoded, parse_pattern
from firepanda.kernel.regex.pike import matches_text
from firepanda.kernel.regex.program import Program, compile_program
from firepanda.kernel.regex.route import ENGINE_PYTHON, ENGINE_RE2


def cached(pattern: StringSlice, engine: UInt8 = ENGINE_RE2) raises -> Program:
    """A program compiled with the alphabet the cache needs.

    Args:
        pattern: The pattern.
        engine: Which engine's reading of it.

    Returns:
        The program.

    Raises:
        Error: If the pattern did not compile, which in this file is a mistake
            in the test rather than an answer.
    """
    var program = compile_program(parse_pattern(pattern), engine, alphabet=True)
    if not program.ok:
        raise Error(
            String("pattern ", pattern, " did not compile: ", program.problem)
        )
    return program^


def patterns() -> List[String]:
    """The patterns the agreement loop asks about.

    Returns:
        The list, all of which the cache takes.
    """
    var out = List[String]()
    out.append(String(""))
    out.append(String("a"))
    out.append(String("abc"))
    out.append(String("a*"))
    out.append(String("a+b"))
    out.append(String("ab?c"))
    out.append(String("^abc"))
    out.append(String("abc$"))
    out.append(String("^abc$"))
    out.append(String("\\Aab"))
    out.append(String("ab\\z"))
    out.append(String("$"))
    out.append(String("\\z"))
    out.append(String("^"))
    out.append(String("a|bc"))
    out.append(String("(a|b)*c"))
    out.append(String("[a-z]+"))
    out.append(String("[^a-z]+"))
    out.append(String("."))
    out.append(String("(?s)."))
    out.append(String("a.c"))
    out.append(String("[0-9]{2,4}"))
    out.append(String("^https?://([^/]+)/"))
    out.append(String("x(y|z)*w"))
    out.append(String("(ab)+"))
    out.append(String("a{3}"))
    out.append(String("^$"))
    out.append(String("^a*$"))
    out.append(String("(a*)*"))
    out.append(String("^(a+)+b$"))
    out.append(String("(|a)*b"))
    out.append(String("é"))
    out.append(String("^h.llo$"))
    out.append(String("^...$"))
    out.append(String("\\d+"))
    out.append(String("\\w"))
    out.append(String("\\s"))
    out.append(String("[^/]+"))
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
    out.append(String("http://example.com/page"))
    out.append(String("ftp://example.com/page"))
    out.append(String("héllo"))
    out.append(String("hello"))
    out.append(String("ΑΒΓ"))
    out.append(String(" "))
    out.append(String("a\nb"))
    out.append(String("a\n"))
    out.append(String("123456"))
    out.append(String("007"))
    out.append(String("xyzw"))
    out.append(String("zzz"))
    out.append(String("_"))
    return out^


def test_the_cache_answers_what_the_machine_answers() raises:
    """Every pattern against every piece of text, both ways, compared. The
    cache is a second engine for the same programs, so the only property worth
    asserting is that the two of them never disagree."""
    var all_patterns = patterns()
    var all_texts = texts()
    for p in range(len(all_patterns)):
        var pattern = all_patterns[p]
        var program = cached(pattern)
        var cache = Cache(program)
        assert_true(
            cache.ok,
            String("cache refused ", pattern, ": ", cache.problem),
        )
        for t in range(len(all_texts)):
            var text = all_texts[t]
            var points = decoded(text)
            var got = cache.scan(program, Span(points))
            var want = SCAN_YES if matches_text(program, text) else SCAN_NO
            assert_equal(
                got,
                want,
                String("pattern ", pattern, " against ", repr(text)),
            )


def _rolled(mut seed: Int, sides: Int) -> Int:
    """A number below `sides`, from a generator small enough to read.

    The differentials next door take their randomness seriously because they
    are comparing against another library and a seed has to be reportable. This
    one is comparing two engines in the same build, so all it has to do is walk
    over a lot of shapes the hand written list above does not have.

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
        if _rolled(seed, 4) == 0:
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


def test_the_cache_and_the_machine_agree_on_a_generated_corpus() raises:
    """The list above is a list somebody wrote down, so it holds the shapes
    somebody thought of. This is six thousand pairs nobody thought of, over the
    same alphabet, and it is what would catch a closure that walks a split arm
    in the wrong order or a state that is confused with another one because the
    end of the row was left out of what makes two states the same."""
    var seed = 20260918
    var compared = 0
    for _ in range(600):
        var pattern = _pattern(seed)
        var program = compile_program(
            parse_pattern(pattern), ENGINE_RE2, alphabet=True
        )
        if not program.ok:
            continue
        var cache = Cache(program)
        if not cache.ok:
            continue
        for _ in range(10):
            var text = _text(seed)
            var points = decoded(text)
            var got = cache.scan(program, Span(points))
            if got == SCAN_GAVE_UP:
                assert_true(cache.full)
                continue
            var want = SCAN_YES if matches_text(program, text) else SCAN_NO
            assert_equal(
                got,
                want,
                String("pattern ", pattern, " against ", repr(text)),
            )
            compared += 1
    assert_true(compared > 5000)


def test_a_pattern_that_reads_around_a_position_is_refused() raises:
    """A word boundary and a multiline anchor both ask about the character on
    the other side of the position, which a state here has no way of knowing.
    Python's dollar sign is the same question wearing RE2's clothes, since it
    matches in front of a newline that ends the row as well as at the end."""
    var boundary = cached("\\bfoo")
    assert_false(Cache(boundary).ok)
    var non_boundary = cached("\\Bfoo", ENGINE_PYTHON)
    assert_false(Cache(non_boundary).ok)
    var line_start = cached("(?m)^a")
    assert_false(Cache(line_start).ok)
    var line_end = cached("(?m)a$")
    assert_false(Cache(line_end).ok)
    var python_end = cached("a$", ENGINE_PYTHON)
    assert_false(Cache(python_end).ok)

    var taken = cached("^a$")
    var cache = Cache(taken)
    assert_true(cache.ok)
    assert_equal(cache.problem, String(""))


def test_a_program_compiled_without_an_alphabet_is_refused() raises:
    """Nothing is compiled with one unless the caller asks, so a caller that
    wants the cache and forgets to ask has to be told rather than handed a
    table of one class that answers the same for every character."""
    var program = compile_program(parse_pattern("abc"), ENGINE_RE2)
    assert_equal(program.class_count, 0)
    var cache = Cache(program)
    assert_false(cache.ok)
    var points = decoded("abc")
    assert_equal(cache.scan(program, Span(points)), SCAN_GAVE_UP)


def test_a_cache_belongs_to_a_column_rather_than_to_a_row() raises:
    """The first rows pay for the states and the rest read the table, which is
    the whole argument for keeping one of these. So the count has to stop going
    up while the answers go on being right."""
    var program = cached("^https?://([^/]+)/")
    var cache = Cache(program)
    var rows = List[String]()
    rows.append(String("http://example.com/one"))
    rows.append(String("https://shop.example.org/catalog"))
    rows.append(String("ftp://example.com/one"))
    rows.append(String("http://a.b/c"))
    rows.append(String("news.site.com/story"))
    for row in range(len(rows)):
        var points = decoded(rows[row])
        _ = cache.scan(program, Span(points))
    var settled = cache.states()
    assert_true(settled > 1)
    for _ in range(20):
        for row in range(len(rows)):
            var points = decoded(rows[row])
            var got = cache.scan(program, Span(points))
            var want = SCAN_YES if matches_text(program, rows[row]) else SCAN_NO
            assert_equal(got, want)
    assert_equal(cache.states(), settled)


def test_a_pattern_with_more_states_than_the_cache_holds_says_so() raises:
    """The tenth character back has to be remembered to answer this one, so the
    states are the subsets of a window and there are five hundred odd of them.
    The cache stops at its bound and admits it, and a caller that gets that
    answer runs the machine. An admission is the only safe thing here, because
    the alternative is throwing a state away and answering from a table that no
    longer says what it used to."""
    var program = cached("^[ab]*a[ab]{9}$")
    var cache = Cache(program)
    assert_true(cache.ok)
    var text = String("")
    var seed = 12345
    for _ in range(600):
        seed = (seed * 1103515245 + 12345) % 2147483648
        text += "a" if (seed // 65536) % 2 == 0 else "b"
    var points = decoded(text)
    assert_equal(cache.scan(program, Span(points)), SCAN_GAVE_UP)
    assert_true(cache.full)
    assert_true(cache.states() <= MAX_STATES)


def test_two_characters_in_a_class_take_one_transition() raises:
    """The alphabet is what makes a row of transitions small enough to hold, so
    a pattern over three letters has a handful of states and each of them is
    four numbers wide rather than a million. The count here is small because
    the sets a three letter pattern can be in are few, and it does not go up
    when the text brings in characters the pattern never mentions."""
    var program = cached("abc")
    var cache = Cache(program)
    assert_equal(program.class_count, 4)
    var points = decoded("xxabcxx")
    assert_equal(cache.scan(program, Span(points)), SCAN_YES)
    var settled = cache.states()
    assert_true(settled <= 6)
    var wider = decoded("ΑΒΓ zq9 ΑΒΓ")
    assert_equal(cache.scan(program, Span(wider)), SCAN_NO)
    assert_equal(cache.states(), settled)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
