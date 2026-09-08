"""Adversarial input, generated rather than collected.

A PEG parser's failure mode is exponential blowup on input that is short, legal
and easy to write. It is a denial of service the moment anything takes SQL it did
not write itself, and the only defence that works is a test that fails loudly, so
these are here from the first week the matcher existed rather than from the first
week somebody hit one.

Every case is built from a size, so the same shape can be asked twice and the
answer compared. That is the point: an absolute ceiling on a shared runner is a
blunt instrument, but a shape that goes from n to 2n and takes sixty four times
as long is exponential no matter how slow the machine was.

The ceilings are loose on purpose. They are guarding against a regression of the
kind that shows up as a factor and not as a percentage, and this file runs
unoptimized because `mojo run` is what the test runner has.

See docs/specs/sql/04-the-parser.md sections 5 and 6.
"""

from std.testing import TestSuite, assert_true
from std.time import perf_counter_ns

from firepanda.sql import Grammar
from firepanda.sql.matcher import parse


def _repeat(unit: StringSlice, count: Int) -> String:
    """Writes a fragment out some number of times.

    Args:
        unit: The fragment.
        count: How many copies.

    Returns:
        The run.
    """
    var out = String()
    for _ in range(count):
        out += unit
    return out^


def _took(sql: StringSlice, g: Grammar, expected: Bool) raises -> Int:
    """Parses once and says how long it took.

    Args:
        sql: The query.
        g: A loaded grammar.
        expected: Whether it should parse.

    Returns:
        The nanoseconds spent.

    Raises:
        Error: If the answer was not the expected one, because a case that
            stopped being parsed at all is no longer measuring what it says.
    """
    var start = perf_counter_ns()
    var got = True
    try:
        _ = parse(sql, g)
    except:
        got = False
    var spent = perf_counter_ns() - start
    if got != expected:
        raise Error(
            String(
                "a ",
                sql.byte_length(),
                " byte case was ",
                "accepted" if got else "rejected",
                " and should have been the other way",
            )
        )
    return spent


def _under(
    what: StringSlice, sql: StringSlice, g: Grammar, expected: Bool, limit: Int
) raises:
    """Checks that one case finishes inside its ceiling.

    Args:
        what: The shape's name, for the failure message.
        sql: The query.
        g: A loaded grammar.
        expected: Whether it should parse.
        limit: The ceiling in microseconds.

    Raises:
        Error: If it was slower than that, or parsed the wrong way.
    """
    var spent = _took(sql, g, expected) // 1000
    assert_true(
        spent < limit,
        String(what, " took ", spent, " us against a ceiling of ", limit),
    )


def _grows_slowly(
    what: StringSlice, small: StringSlice, big: StringSlice, g: Grammar
) raises:
    """Checks that doubling a shape does not blow the time up.

    Doubling the size of an exponential shape multiplies the time by the size,
    so anything that is really exponential misses this by orders of magnitude
    and no runner is noisy enough to hide that.

    Args:
        what: The shape's name, for the failure message.
        small: The query at size n.
        big: The query at size 2n.
        g: A loaded grammar.

    Raises:
        Error: If the big one took more than sixteen times the small one.
    """
    # Once each to get the grammar and the allocator warm, then once each for
    # the number, because the first parse in a process pays for both.
    _ = _took(small, g, True)
    _ = _took(big, g, True)
    var short = _took(small, g, True)
    var long = _took(big, g, True)
    assert_true(
        long < short * 16,
        String(
            what,
            " went superlinear: ",
            short // 1000,
            " us at half the size and ",
            long // 1000,
            " us at full size",
        ),
    )


# ---------------------------------------------------------------------------
# The shapes that backtrack
# ---------------------------------------------------------------------------


def test_unmatched_parentheses_do_not_blow_up() raises:
    # DuckDB measured this one and published the number: nineteen unmatched
    # parentheses took 10.640 seconds without memoization and 0.001 with it.
    # It is the reason there is a memo table at all.
    var g = Grammar()
    _under(
        "nineteen unmatched parentheses",
        "SELECT " + _repeat("(", 19) + "1",
        g,
        False,
        100_000,
    )


def test_nested_function_calls_do_not_blow_up() raises:
    # This one is worse than the parentheses, because it is valid SQL that a
    # person would write. `TypeModifiers <- Parens(List(Expression)?)` means
    # `f(x)` parses as a type before it parses as a call, so `SingleExpression`
    # walks the argument twice at every level and the whole thing doubles.
    # Failure memoization cannot see it, because both walks succeed.
    var g = Grammar()
    _under(
        "twelve nested function calls",
        "SELECT " + _repeat("f(", 12) + "1" + _repeat(")", 12),
        g,
        True,
        100_000,
    )
    _grows_slowly(
        "nested function calls",
        "SELECT " + _repeat("f(", 6) + "1" + _repeat(")", 6),
        "SELECT " + _repeat("f(", 12) + "1" + _repeat(")", 12),
        g,
    )


def test_nested_list_literals_do_not_blow_up() raises:
    # The same shape as the calls, reached through `ListExpression` instead.
    var g = Grammar()
    _grows_slowly(
        "nested list literals",
        "SELECT " + _repeat("[", 5) + "1" + _repeat("]", 5),
        "SELECT " + _repeat("[", 10) + "1" + _repeat("]", 10),
        g,
    )


def test_nested_parentheses_stay_linear() raises:
    var g = Grammar()
    _grows_slowly(
        "nested parentheses",
        "SELECT " + _repeat("(", 10) + "1" + _repeat(")", 10),
        "SELECT " + _repeat("(", 20) + "1" + _repeat(")", 20),
        g,
    )


def test_nested_case_expressions_stay_linear() raises:
    var g = Grammar()
    _grows_slowly(
        "nested CASE",
        "SELECT "
        + _repeat("CASE WHEN a THEN ", 10)
        + "1"
        + _repeat(" END", 10),
        "SELECT "
        + _repeat("CASE WHEN a THEN ", 20)
        + "1"
        + _repeat(" END", 20),
        g,
    )


def test_an_unterminated_case_chain_fails_quickly() raises:
    var g = Grammar()
    _under(
        "twenty unterminated CASE",
        "SELECT " + _repeat("CASE WHEN a THEN ", 20) + "1",
        g,
        False,
        100_000,
    )


def test_nested_subqueries_stay_linear() raises:
    var g = Grammar()
    _grows_slowly(
        "nested subqueries",
        "SELECT * FROM "
        + _repeat("(SELECT * FROM ", 12)
        + "t"
        + _repeat(")", 12),
        "SELECT * FROM "
        + _repeat("(SELECT * FROM ", 24)
        + "t"
        + _repeat(")", 24),
        g,
    )


# ---------------------------------------------------------------------------
# The shapes that are just long
# ---------------------------------------------------------------------------


def test_a_long_in_list_stays_linear() raises:
    var g = Grammar()
    _grows_slowly(
        "an IN list",
        "SELECT a FROM t WHERE a IN (" + _repeat("0,", 200) + "1)",
        "SELECT a FROM t WHERE a IN (" + _repeat("0,", 400) + "1)",
        g,
    )


def test_a_long_operator_chain_stays_linear() raises:
    # The expression precedence chain is fifteen rules deep and this walks all
    # fifteen once per operand, so it is the shape that says whether the chain
    # itself is the problem.
    var g = Grammar()
    _grows_slowly(
        "an operator chain",
        "SELECT 1" + _repeat(" + 1", 200),
        "SELECT 1" + _repeat(" + 1", 400),
        g,
    )


def test_a_wide_select_list_stays_linear() raises:
    var g = Grammar()
    _grows_slowly(
        "a wide select list",
        "SELECT a" + _repeat(", a", 200) + " FROM t",
        "SELECT a" + _repeat(", a", 400) + " FROM t",
        g,
    )


def test_a_long_qualified_name_stays_linear() raises:
    var g = Grammar()
    _grows_slowly(
        "a qualified name",
        "SELECT a" + _repeat(".b", 200) + " FROM t",
        "SELECT a" + _repeat(".b", 400) + " FROM t",
        g,
    )


def test_many_statements_stay_linear() raises:
    var g = Grammar()
    _grows_slowly(
        "a script of statements",
        _repeat("SELECT 1; ", 200),
        _repeat("SELECT 1; ", 400),
        g,
    )


def test_a_run_of_semicolons_stays_linear() raises:
    var g = Grammar()
    _grows_slowly(
        "a run of semicolons",
        "SELECT 1" + _repeat(";", 400),
        "SELECT 1" + _repeat(";", 800),
        g,
    )


# ---------------------------------------------------------------------------
# The floor
# ---------------------------------------------------------------------------


def test_the_cheapest_statement_stays_cheap() raises:
    # A REPL loop over small statements is the workload the whole budget is
    # about, so the floor gets a ceiling of its own.
    var g = Grammar()
    var rounds = 20
    var start = perf_counter_ns()
    for _ in range(rounds):
        _ = parse("SELECT 1", g)
    var each = (perf_counter_ns() - start) // rounds
    assert_true(
        each < 5_000_000,
        String("SELECT 1 took ", each // 1000, " us"),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
