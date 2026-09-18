"""`\\z`, which is a spelling rather than a meaning, and which Python was late to.

RE2 has always had `\\z` for the end of the string. Python has always had `\\Z`
for the same position and read `\\z` as a `bad escape \\z` until 3.14, which
added it. So nothing became sayable in 3.14 that was not sayable before, and a
spelling stopped being an error. `pixi.toml` says this project supports 3.12 and
up, so on the same pattern one supported interpreter raises and the next
answers.

That makes this the second rule in one CPython release that a library copying
`re` has to know the version for, after the `\\B` one in document 90, and the
two together are why the compiler carries a version number rather than a flag
per rule.

Inside a class it is a bad escape in every version including 3.14, which is the
ordinary bad escape path and is a row here so that the version question is kept
to the one place it belongs.

Document 91 is the whole of it.
"""

from std.testing import TestSuite, assert_equal

from firepanda.kernel.regex.method import (
    METHOD_CONTAINS,
    METHOD_COUNT,
    METHOD_EXTRACT,
    METHOD_FULLMATCH,
    METHOD_MATCH,
    program_for,
)
from firepanda.kernel.regex.parse import parse_pattern
from firepanda.kernel.regex.pike import matches_text
from firepanda.kernel.regex.program import compile_program
from firepanda.kernel.regex.route import ENGINE_PYTHON, ENGINE_RE2


def said(pattern: StringSlice, engine: UInt8, minor: Int = 14) -> String:
    """Whether a pattern compiles, and the reason when it does not.

    Args:
        pattern: The pattern.
        engine: Which engine to compile it for.
        minor: Which CPython to compile it beside.

    Returns:
        The word ok, or an exclamation mark and the reason.
    """
    var program = compile_program(parse_pattern(pattern), engine, minor=minor)
    if not program.ok:
        return String("!", program.problem)
    return String("ok")


def asked(
    method: UInt8, pattern: StringSlice, argued: Bool, minor: Int
) -> String:
    """The same, for a whole call rather than for a compile.

    This is the one that matters, because the pattern a call compiles is not
    the pattern a caller wrote. `match` and `fullmatch` on Python's engine are
    answered here by writing anchors around the caller's text, and the question
    the version rule asks is about the caller's text alone.

    Args:
        method: Which of the six asked.
        pattern: The pattern as a caller wrote it.
        argued: Whether flags were passed beside it, which is what moves a call
            to Python's engine.
        minor: Which CPython to agree with.

    Returns:
        The word ok, or an exclamation mark and the reason.
    """
    var program = program_for(
        method, String(pattern), 0, argued=argued, minor=minor
    )
    if not program.ok:
        return String("!", program.problem)
    return String("ok")


def matches(pattern: StringSlice, text: StringSlice, minor: Int) raises -> Bool:
    """Whether a pattern read Python's way matches, beside one interpreter.

    Args:
        pattern: The pattern.
        text: The text.
        minor: Which CPython to agree with.

    Returns:
        True when some part of it matches.

    Raises:
        Error: If the pattern did not compile, which in this file is a mistake
            in the test rather than an answer.
    """
    var program = compile_program(
        parse_pattern(pattern), ENGINE_PYTHON, minor=minor
    )
    if not program.ok:
        raise Error(
            String("pattern ", pattern, " did not compile: ", program.problem)
        )
    return matches_text(program, text)


def test_the_spelling_is_refused_beside_an_interpreter_that_has_not_got_it() raises:
    """The whole rule, on the smallest pattern that carries it."""
    assert_equal(
        said("\\z", ENGINE_PYTHON, 13), "!this Python has no \\z escape"
    )
    assert_equal(said("\\z", ENGINE_PYTHON, 14), "ok")
    assert_equal(
        said("a\\z", ENGINE_PYTHON, 12), "!this Python has no \\z escape"
    )
    assert_equal(said("a\\z", ENGINE_PYTHON, 14), "ok")


def test_the_other_spelling_was_never_late() raises:
    """`\\Z` is the same position and every version has it, which is what makes
    the change a spelling rather than a meaning."""
    for minor in [12, 13, 14]:
        assert_equal(said("\\Z", ENGINE_PYTHON, minor), "ok")
        assert_equal(said("a\\Z", ENGINE_PYTHON, minor), "ok")


def test_the_two_spellings_are_one_position() raises:
    """Beside an interpreter that has both, they answer alike on every row,
    which is the claim that makes the refusal a refusal rather than a second
    reading of the pattern."""
    for text in ["", "a", "ab", "a\n", "\n", "ba"]:
        assert_equal(matches("a\\z", text, 14), matches("a\\Z", text, 14), text)
        assert_equal(matches("\\z", text, 14), matches("\\Z", text, 14), text)


def test_re2_has_always_had_it_and_has_no_version_of_python() raises:
    """It is RE2's own spelling, so the engine pandas hands an unflagged call to
    reads it whatever interpreter the call arrived in."""
    for minor in [12, 13, 14]:
        assert_equal(said("a\\z", ENGINE_RE2, minor), "ok")
        assert_equal(said("\\z", ENGINE_RE2, minor), "ok")


def test_inside_a_class_it_is_a_bad_escape_in_every_version() raises:
    """3.14 added an anchor and an anchor is not a thing a class can hold, so
    `[\\z]` is refused by the grammar rather than by the version, which is why
    no number moves it."""
    for minor in [12, 13, 14]:
        for engine in [ENGINE_PYTHON, ENGINE_RE2]:
            assert_equal(
                said("[\\z]", engine, minor),
                "!Python's grammar cannot read this pattern",
            )
            assert_equal(
                said("[a\\z]", engine, minor),
                "!Python's grammar cannot read this pattern",
            )


def test_the_anchor_this_library_writes_is_not_the_callers() raises:
    """The row the slice turns on. `fullmatch` on Python's engine is answered by
    writing anchors around the caller's text, and those anchors have to be
    spelled the way the engine they are written for spells them, or every
    anchored call is refused beside an older interpreter for a `\\z` the caller
    never wrote."""
    for minor in [12, 13, 14]:
        assert_equal(asked(METHOD_FULLMATCH, "a", True, minor), "ok")
        assert_equal(asked(METHOD_MATCH, "a", True, minor), "ok")
        assert_equal(asked(METHOD_FULLMATCH, "a(b)c", True, minor), "ok")


def test_a_caller_who_wrote_one_is_refused_through_the_anchoring() raises:
    """And the caller's own `\\z` survives the rewrite, which it has to, since
    the text being compiled is the caller's text with something wrapped round
    it."""
    assert_equal(
        asked(METHOD_FULLMATCH, "a\\z", True, 13),
        "!this Python has no \\z escape",
    )
    assert_equal(asked(METHOD_FULLMATCH, "a\\z", True, 14), "ok")
    assert_equal(
        asked(METHOD_COUNT, "a\\z", True, 13), "!this Python has no \\z escape"
    )
    assert_equal(asked(METHOD_COUNT, "a\\z", True, 14), "ok")


def test_an_unflagged_call_never_asks_the_question() raises:
    """With no flags beside it the call goes to Arrow, which has the spelling,
    so a version of Python has nothing to do with it. That is upstream's answer
    as well: `str.count("a\\\\z")` is a column on 3.13 and `str.count("a\\\\z",
    flags=re.M)` is a `bad escape` on the same interpreter."""
    for minor in [12, 13, 14]:
        assert_equal(asked(METHOD_CONTAINS, "a\\z", False, minor), "ok")
        assert_equal(asked(METHOD_COUNT, "a\\z", False, minor), "ok")


def test_extract_asks_it_with_no_flags_at_all() raises:
    """`extract` is the one method upstream never routes, so it is answered in
    Python whether or not a flag was passed and the version rule reaches it
    through the front door. `str.extract("(a)\\\\z")` raises on 3.13 with no
    keyword in sight."""
    assert_equal(
        asked(METHOD_EXTRACT, "(a)\\z", False, 13),
        "!this Python has no \\z escape",
    )
    assert_equal(asked(METHOD_EXTRACT, "(a)\\z", False, 14), "ok")
    assert_equal(asked(METHOD_EXTRACT, "(a)\\Z", False, 13), "ok")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
