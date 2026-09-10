"""Bind contexts, and the four rules that make a name mean something.

Every case here was measured against DuckDB 1.5 rather than read from
documentation, because each one is a silently different answer rather than an
error if it is wrong: a bare name that should have been ambiguous and was not,
a `USING` column that resolved twice, a subquery that was correlated and was
not recorded as such.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.sql.bind import (
    AMBIGUOUS,
    NO_BINDING,
    SOURCE_FRAME,
    SOURCE_SUBQUERY,
    Scopes,
)
from firepanda.sql.catalog import NOT_FOUND


def _table(
    mut scopes: Scopes, level: Int, name: StringSlice, columns: List[String]
) -> Int:
    """Adds a binding with those columns, all of them integers.

    Args:
        scopes: The chain.
        level: The level to add at.
        name: What the query calls the binding.
        columns: The column names.

    Returns:
        The binding's position.
    """
    var at = scopes.add(level, name, SOURCE_FRAME, 0)
    for column in columns:
        scopes.levels[level].bindings[at].add(column, LogicalType.INT32)
    return at


def test_a_bare_name_resolves_to_a_position() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "a", [String("x"), String("y")])
    var found = scopes.resolve(top, "y")
    assert_true(found.found())
    assert_equal(found.binding, 0)
    assert_equal(found.column, 1)
    assert_equal(found.depth, 0)
    assert_false(found.correlated())


def test_a_name_that_nothing_has_resolves_to_nothing() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "a", [String("x")])
    assert_equal(scopes.resolve(top, "nocol").binding, NO_BINDING)
    assert_false(scopes.resolve(top, "nocol").found())


def test_names_fold_however_they_were_written() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "MyTable", [String("MyColumn")])
    # The transformer folds a bare identifier and leaves a quoted one alone, so
    # all three of these arrive here spelled differently and all three resolve.
    assert_true(scopes.resolve(top, "mycolumn").found())
    assert_true(scopes.resolve(top, "MyColumn").found())
    assert_true(scopes.resolve(top, "MYCOLUMN").found())
    assert_true(scopes.resolve_qualified(top, "MYTABLE", "mycolumn").found())


def test_a_bare_name_two_bindings_have_is_ambiguous() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "a", [String("x"), String("y")])
    _ = _table(scopes, top, "b", [String("x"), String("z")])
    assert_equal(scopes.resolve(top, "x").binding, AMBIGUOUS)
    assert_false(scopes.resolve(top, "x").found())
    # Not ambiguous, because only one of them has it.
    assert_equal(scopes.resolve(top, "z").binding, 1)


def test_the_ambiguity_error_says_both_ways_to_write_it() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "a", [String("x")])
    _ = _table(scopes, top, "b", [String("x")])
    assert_equal(
        scopes.ambiguity(top, "x"),
        (
            'Binder Error: Ambiguous reference to column name "x" (use: "a.x"'
            ' or "b.x")'
        ),
    )


def test_a_merged_column_is_one_column_with_two_homes() raises:
    # SELECT x FROM a JOIN b USING (x) is not ambiguous, and b.x still means
    # something. Both halves of that matter and a hidden flag is what gives
    # them to us without copying the column anywhere.
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "a", [String("x"), String("y")])
    var right = _table(scopes, top, "b", [String("x"), String("z")])
    scopes.merge(top, right, "x")
    var bare = scopes.resolve(top, "x")
    assert_true(bare.found())
    assert_equal(bare.binding, 0)
    assert_true(scopes.resolve_qualified(top, "b", "x").found())
    assert_equal(scopes.resolve_qualified(top, "b", "x").binding, 1)


def test_a_merged_column_is_expanded_once_by_a_star() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "a", [String("x"), String("y")])
    var right = _table(scopes, top, "b", [String("x"), String("z")])
    scopes.merge(top, right, "x")
    # SELECT * FROM a NATURAL JOIN b is x, y, z and not x, y, x, z.
    var expanded = scopes.visible_columns(top)
    assert_equal(len(expanded), 3)
    assert_equal(expanded[0].binding, 0)
    assert_equal(expanded[0].column, 0)
    assert_equal(expanded[2].binding, 1)
    assert_equal(expanded[2].column, 1)


def test_a_star_expands_in_the_order_the_from_was_written() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "a", [String("x"), String("y")])
    _ = _table(scopes, top, "b", [String("z")])
    var expanded = scopes.visible_columns(top)
    assert_equal(len(expanded), 3)
    assert_equal(expanded[0].binding, 0)
    assert_equal(expanded[1].binding, 0)
    assert_equal(expanded[2].binding, 1)


def test_a_qualified_name_needs_both_halves_to_match() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "a", [String("x")])
    assert_true(scopes.resolve_qualified(top, "a", "x").found())
    assert_false(scopes.resolve_qualified(top, "a", "q").found())
    assert_false(scopes.resolve_qualified(top, "q", "x").found())


def test_whether_a_qualifier_is_a_table_decides_what_a_dot_means() raises:
    # a.b is a column of table a if a is a binding, and a field of struct column
    # a if it is not. The longest interpretation is tried first, and this is the
    # question that decides which one that is.
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "a", [String("b")])
    assert_true(scopes.knows_table(top, "a"))
    assert_false(scopes.knows_table(top, "b"))


def test_a_name_resolved_at_an_outer_level_is_correlated() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "a", [String("x"), String("y")])
    var inner = scopes.open(top)
    _ = _table(scopes, inner, "b", [String("z")])
    var inside = scopes.resolve(inner, "z")
    assert_true(inside.found())
    assert_equal(inside.depth, 0)
    assert_false(inside.correlated())
    var outside = scopes.resolve(inner, "y")
    assert_true(outside.found())
    assert_equal(outside.depth, 1)
    assert_true(outside.correlated())


def test_a_correlation_is_recorded_where_it_happened() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "a", [String("x")])
    var inner = scopes.open(top)
    _ = _table(scopes, inner, "b", [String("z")])
    assert_equal(len(scopes.levels[inner].correlations), 0)
    _ = scopes.resolve(inner, "z")
    assert_equal(len(scopes.levels[inner].correlations), 0)
    _ = scopes.resolve(inner, "x")
    _ = scopes.resolve_qualified(inner, "a", "x")
    assert_equal(len(scopes.levels[inner].correlations), 2)
    assert_equal(scopes.levels[inner].correlations[0].depth, 1)
    # The outer level did not reach for anything, so it records nothing.
    assert_equal(len(scopes.levels[top].correlations), 0)


def test_an_inner_binding_shadows_an_outer_one_of_the_same_name() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("x")])
    var inner = scopes.open(top)
    var shadow = scopes.add(inner, "t", SOURCE_SUBQUERY, 0)
    scopes.levels[inner].bindings[shadow].add("x", LogicalType.INT32)
    var found = scopes.resolve_qualified(inner, "t", "x")
    assert_equal(found.depth, 0)
    assert_false(found.correlated())


def test_resolution_walks_more_than_one_level_out() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "a", [String("x")])
    var middle = scopes.open(top)
    _ = _table(scopes, middle, "b", [String("y")])
    var inner = scopes.open(middle)
    _ = _table(scopes, inner, "c", [String("z")])
    assert_equal(scopes.resolve(inner, "x").depth, 2)
    assert_equal(scopes.resolve(inner, "y").depth, 1)
    assert_equal(scopes.resolve(inner, "z").depth, 0)


def test_a_binding_takes_a_whole_schema_at_once() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    var at = scopes.add(top, "t", SOURCE_FRAME, 0)
    var schema = Schema()
    schema.fields.append(Field("a", LogicalType.INT32))
    schema.fields.append(Field("b", LogicalType.INT64))
    scopes.levels[top].bindings[at].add_schema(schema)
    assert_equal(len(scopes.levels[top].bindings[at].columns), 2)
    assert_equal(scopes.resolve(top, "b").column, 1)
    assert_equal(
        scopes.levels[top].bindings[at].columns[1].dtype, LogicalType.INT64
    )


def test_the_missing_column_error_offers_the_near_misses() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(
        scopes,
        top,
        "a",
        [String("alpha"), String("beta"), String("gamma"), String("delta")],
    )
    var message = scopes.no_such_column(top, "alpah")
    assert_true(
        'Binder Error: Referenced column "alpah" not found in FROM clause!'
        in message
    )
    assert_true("Candidate bindings: " in message)
    assert_true('"alpha"' in message)
    # Three at most, which is DuckDB's number, so a wide table does not print
    # itself into the error.
    assert_equal(message.count('", "'), 2)


def test_a_candidate_is_offered_once_however_many_tables_have_it() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "a", [String("x")])
    _ = _table(scopes, top, "b", [String("x")])
    assert_equal(scopes.no_such_column(top, "y").count('"x"'), 1)


def test_the_missing_table_error_lists_the_tables_in_scope() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "a", [String("x")])
    _ = _table(scopes, top, "b", [String("x")])
    assert_equal(
        scopes.no_such_table(top, "q"),
        (
            'Binder Error: Referenced table "q" not found!\nCandidate tables:'
            ' "a", "b"'
        ),
    )


def test_the_wrong_column_error_lists_that_tables_columns() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "a", [String("x"), String("y")])
    _ = _table(scopes, top, "b", [String("z")])
    var message = scopes.no_such_column_in(top, "a", "q")
    assert_true(
        'Binder Error: Table "a" does not have a column named "q"' in message
    )
    assert_true('"x", "y"' in message)
    assert_false('"z"' in message)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
