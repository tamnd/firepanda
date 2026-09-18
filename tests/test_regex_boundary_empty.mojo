"""`\\B` on a row with nothing in it, which is a question about which Python.

Up to 3.13 CPython fails a `\\B` on an empty row. 3.14 took the case out and
made `\\B` the plain negation of `\\b`, which is what every other engine has
always had and what RE2 has. `pixi.toml` says this project supports 3.12 and up,
so both answers are live and neither of them is the answer, and a program is
compiled for a version of Python rather than for Python.

That is the whole of the difference. Every row below that is not empty answers
the same under both, and `\\b` answers the same on the empty row too, since the
case was only ever attached to the negative half.

The case used to be two position codes, one per alphabet, which said that the
answer for an empty row is a fact about which characters count as word
characters. It is not. It is now one instruction of its own, written in front of
the boundary when the interpreter is one of the older ones, and that is what
lets `\\b` and `\\B` be a pair again under both alphabets.

Every row was asked of a running CPython 3.13.12 and a running CPython 3.14.7
before it was written down. RE2 has no version of Python and no `\\B` either,
since this library refuses that one for RE2 on the separate ground that RE2 asks
the boundary question between bytes.

Document 90 is the whole of it.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.kernel.regex.parse import parse_pattern
from firepanda.kernel.regex.pike import matches_text
from firepanda.kernel.regex.program import (
    IN_AT,
    IN_MATCH,
    Program,
    compile_program,
)
from firepanda.kernel.regex.route import ENGINE_PYTHON, ENGINE_RE2
from firepanda.kernel.regex.tokens import FLAG_ASCII


def drawn(pattern: StringSlice, engine: UInt8, minor: Int = 14) -> String:
    """The position checks a compiled pattern holds, in order.

    The programs in this file are two and three instructions long and every one
    of those instructions is a position check or the end, so a listing that
    named the characters would be a listing of nothing.

    Args:
        pattern: The pattern.
        engine: Which engine to compile it for.
        minor: Which CPython to compile it beside.

    Returns:
        The line, or an exclamation mark and the reason there is not one.
    """
    var program = compile_program(parse_pattern(pattern), engine, minor=minor)
    return _drawn(program)


def drawn_narrow(minor: Int) -> String:
    """The same, for `\\B` under the ascii flag on Python's engine.

    Args:
        minor: Which CPython to compile it beside.

    Returns:
        The line.
    """
    return _drawn(
        compile_program(
            parse_pattern("\\B", FLAG_ASCII), ENGINE_PYTHON, minor=minor
        )
    )


def _drawn(program: Program) -> String:
    """One compiled program, as one line.

    Args:
        program: The compiled pattern.

    Returns:
        The line.
    """
    if not program.ok:
        return String("!", program.problem)
    var out = String("")
    for at in range(len(program.code)):
        if at != 0:
            out += "; "
        var it = program.code[at]
        if it.op == IN_AT:
            out += String("at(", it.a, ")")
        elif it.op == IN_MATCH:
            out += "match"
        else:
            out += String("op", it.op)
    return out^


def ours(
    pattern: StringSlice, text: StringSlice, minor: Int, flags: Int32 = 0
) raises -> Bool:
    """Whether a pattern read Python's way matches, beside one interpreter.

    Args:
        pattern: The pattern.
        text: The text.
        minor: Which CPython to agree with, as the minor number alone.
        flags: The letters a caller passed beside the pattern.

    Returns:
        True when some part of it matches.

    Raises:
        Error: If the pattern did not compile, which in this file is a mistake
            in the test rather than an answer.
    """
    var program = compile_program(
        parse_pattern(pattern, flags), ENGINE_PYTHON, minor=minor
    )
    if not program.ok:
        raise Error(
            String("pattern ", pattern, " did not compile: ", program.problem)
        )
    return matches_text(program, text)


def test_the_empty_row_is_the_whole_of_the_difference() raises:
    """One row of the sixteen the differential uses answers differently, and it
    is the one with nothing in it."""
    assert_false(ours("\\B", "", 13))
    assert_true(ours("\\B", "", 14))
    for text in ["a", "b", "ab", "ba", "aab", "abc", "a\n", "\na", "a\nb"]:
        assert_equal(ours("\\B", text, 13), ours("\\B", text, 14), text)


def test_the_positive_half_never_had_the_case() raises:
    """`\\b` fails on an empty row in every version, because there is nothing
    there for a boundary to be between. The case that changed was only ever on
    the negative half, which is why the two were not a pair."""
    assert_false(ours("\\b", "", 13))
    assert_false(ours("\\b", "", 14))
    assert_true(ours("\\b", "a", 13))
    assert_true(ours("\\b", "a", 14))


def test_the_alphabet_has_nothing_to_say_about_an_empty_row() raises:
    """`(?a)` narrows which characters count as word characters, and a row that
    holds none of any kind is not a question about characters at all. So the
    letter moves neither answer."""
    assert_false(ours("\\B", "", 13, FLAG_ASCII))
    assert_true(ours("\\B", "", 14, FLAG_ASCII))
    assert_false(ours("(?a)\\B", "", 13))
    assert_true(ours("(?a)\\B", "", 14))


def test_the_alphabet_still_moves_the_rows_that_hold_something() raises:
    """The two alphabets are a real difference and writing the empty row as its
    own instruction has not flattened it. An e-acute is a word character to the
    wide reading and is not to the narrow one, so the position between an `a`
    and one is a non boundary under the first and a boundary under the second,
    and that holds under both interpreters."""
    for minor in [13, 14]:
        assert_true(ours("a\\B", "a\u00e9", minor))
        assert_false(ours("(?a)a\\B", "a\u00e9", minor))
        assert_false(ours("a\\b", "a\u00e9", minor))
        assert_true(ours("(?a)a\\b", "a\u00e9", minor))


def test_re2_has_not_got_a_version_of_python_and_has_no_non_boundary_either() raises:
    """RE2 asks the boundary question between bytes rather than between
    characters, so this library refuses a `\\B` for that engine and has since
    long before any of this. It is worth a row anyway, because it says the
    version question is Python's alone: the value the compiler now reuses for
    the narrow reading is a value no RE2 program ever holds."""
    assert_equal(
        drawn("\\B", ENGINE_RE2), "!RE2 reads a non boundary between bytes"
    )
    assert_equal(drawn("\\b", ENGINE_RE2), "at(7); match")


def test_the_case_is_an_instruction_and_not_an_alphabet() raises:
    """The listing is where the shape shows with nothing running. An older
    interpreter gets two position checks for the one `\\B` and the newer one
    gets the plain code by itself, and the code is the same code under both
    alphabets."""
    assert_equal(drawn("\\B", ENGINE_PYTHON, 13), "at(12); at(11); match")
    assert_equal(drawn("\\B", ENGINE_PYTHON, 14), "at(11); match")
    assert_equal(drawn_narrow(13), "at(12); at(8); match")
    assert_equal(drawn_narrow(14), "at(8); match")


def test_the_narrow_reading_has_no_position_code_of_its_own_any_more() raises:
    """`(?a)\\B` used to need a code of its own, because the plain code carried
    RE2's answer for an empty row along with RE2's alphabet. With the empty row
    written as its own instruction there is nothing left to carry, so the narrow
    reading is the plain code and the code that was invented for it is gone. The
    two alphabets are two codes again, the way `\\b`'s two are."""
    assert_equal(drawn_narrow(14), "at(8); match")
    assert_equal(drawn("\\B", ENGINE_PYTHON, 14), "at(11); match")
    assert_equal(drawn("(?a)\\b", ENGINE_PYTHON, 14), "at(7); match")
    assert_equal(drawn("\\b", ENGINE_PYTHON, 14), "at(10); match")


def test_a_pattern_with_more_in_it_than_the_boundary() raises:
    """The extra instruction is in front of the boundary and not in front of the
    pattern, so a `\\B` that is not the first thing read still only asks about
    the row it is in."""
    assert_false(ours("a|\\B", "", 13))
    assert_true(ours("a|\\B", "", 14))
    assert_true(ours("a|\\B", "a", 13))
    assert_false(ours("\\Ba", "", 13))
    assert_false(ours("\\Ba", "", 14))
    assert_true(ours("\\Bb", "ab", 13))
    assert_true(ours("\\Bb", "ab", 14))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
