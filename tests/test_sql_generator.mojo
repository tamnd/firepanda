"""The grammar read backwards, as a writer of SQL.

These do not check that the generated statements are good SQL. The oracle for
that is DuckDB and it lives in `tests/differential/sql_generated.mojo`, which
needs Python and a DuckDB build and so cannot run here. What is left is the
part that has to hold before that comparison means anything: the walk stops, it
stops near its budget, the same seed writes the same thing twice, and different
seeds do not all write the same thing.

The one claim about content that is checked here is that most of what comes out
parses. The generator only follows rules the grammar spells out and cannot
satisfy negative lookahead, so some of its output is legal by the letter of the
walk and refused by the grammar as a whole. A majority is the honest bar. A
floor rather than an exact number, because the number moves whenever the
vendored grammar does.

See docs/specs/sql/11-conformance.md section 4.
"""

from std.testing import TestSuite, assert_equal, assert_true

from firepanda.sql import Grammar
from firepanda.sql.generator import MAX_TOKENS, Generator
from firepanda.sql.matcher import parse, parse_rule


def _words(text: StringSlice) -> Int:
    """Counts the space separated words in a string.

    Args:
        text: The string.

    Returns:
        How many words, which for generated output is how many tokens.
    """
    var count = 0
    var inside = False
    for byte in text.as_bytes():
        if byte == UInt8(32):
            inside = False
        elif not inside:
            inside = True
            count += 1
    return count


def test_a_statement_comes_out_at_all() raises:
    var g = Grammar()
    var generator = Generator(g, 1)
    var sql = generator.statement()
    assert_true(sql.byte_length() > 0, "the generator wrote nothing")


def test_the_same_seed_writes_the_same_run() raises:
    # The whole reproducibility story is that a harness prints a seed and a case
    # number and somebody else gets the same statement back.
    var g = Grammar()
    var first = Generator(g, 0xC0FFEE)
    var second = Generator(g, 0xC0FFEE)
    for _ in range(25):
        assert_equal(first.statement(), second.statement())


def test_different_seeds_write_different_runs() raises:
    # A generator that ignores its seed passes every other test in this file.
    var g = Grammar()
    var one = Generator(g, 1)
    var two = Generator(g, 2)
    var same = 0
    for _ in range(25):
        if one.statement() == two.statement():
            same += 1
    assert_true(same < 12, String(same, " of 25 statements were identical"))


def test_a_run_of_statements_stays_near_the_budget() raises:
    # The budget is where the walk stops spending, not where it stops writing,
    # so a statement is allowed to run past it by whatever it costs to close the
    # frames that were already open. That overshoot is the thing being pinned
    # here. It has to be bounded, because an unbounded one means a choice went
    # on affording alternatives after the budget was gone, and that is the door
    # a runaway comes through. Eight times the budget is generous against a
    # measured worst case of 182 in twenty thousand statements.
    var g = Grammar()
    var generator = Generator(g, 7)
    for _ in range(200):
        var sql = generator.statement()
        var words = _words(sql)
        assert_true(
            words <= MAX_TOKENS * 8,
            String("a statement spent ", words, " tokens: ", sql),
        )


def test_every_statement_is_one_line() raises:
    # One space between every token and nothing else. A newline in the output
    # would break every harness that prints a failing case on one line.
    var g = Grammar()
    var generator = Generator(g, 11)
    for _ in range(100):
        var sql = generator.statement()
        assert_true("\n" not in sql, String("a statement had a newline: ", sql))
        assert_true("  " not in sql, String("a statement had a gap: ", sql))


def test_most_of_what_it_writes_parses() raises:
    # Not all of it, and the docstring at the top says why. This is a floor on
    # how far the walk can drift from the grammar it walks before the harness
    # that uses it stops being worth running.
    var g = Grammar()
    var generator = Generator(g, 0x5EED)
    var rounds = 300
    var accepted = 0
    for _ in range(rounds):
        var sql = generator.statement()
        try:
            _ = parse(sql, g)
            accepted += 1
        except:
            pass
    assert_true(
        accepted * 2 > rounds,
        String(
            "only ", accepted, " of ", rounds, " generated statements parsed"
        ),
    )


def test_one_rule_can_be_aimed_at() raises:
    # Asking for a single rule is how a harness narrows onto one corner of the
    # grammar. What makes it a real test rather than a smoke test is that the
    # matcher can be aimed at the same rule, so the two halves check each other:
    # what the generator writes for a rule is what the matcher reads for it.
    var g = Grammar()
    var wanted = -1
    for i in range(len(g.names)):
        if g.names[i] == "SelectStatement":
            wanted = i
            break
    assert_true(wanted >= 0, "the grammar has no SelectStatement rule")

    var generator = Generator(g, 3)
    var rounds = 200
    var accepted = 0
    for _ in range(rounds):
        var sql = generator.rule_text(wanted)
        try:
            _ = parse_rule(sql, g, wanted)
            accepted += 1
        except:
            pass
    assert_true(
        accepted * 2 > rounds,
        String("only ", accepted, " of ", rounds, " read back as a select"),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
