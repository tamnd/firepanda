"""Star expansion, and the three modifiers that can be hung off a star.

Every case here was measured against DuckDB 1.5 rather than reasoned about,
because the three modifiers do not behave alike and the differences are not
the ones a reader would guess: `EXCLUDE` and `REPLACE` refuse a name that is
not there and `RENAME` ignores it, `REPLACE` cannot be qualified while the
other two can, and a `REPLACE` that matches twice drops a column rather than
replacing two.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.dtype.logical import LogicalType
from firepanda.sql.bind import SOURCE_FRAME, Scopes
from firepanda.sql.catalog import NOT_FOUND
from firepanda.sql.star import (
    NOT_REPLACED,
    Renaming,
    Replacement,
    Selected,
    Target,
    expand,
)


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


def _two_tables(mut scopes: Scopes) -> Int:
    """Opens a level holding `t(a, b)` and `u(a, d)`.

    Args:
        scopes: The chain.

    Returns:
        The level.
    """
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a"), String("b")])
    _ = _table(scopes, top, "u", [String("a"), String("d")])
    return top


def _names(selected: List[Selected]) -> List[String]:
    """The output names, in order.

    Args:
        selected: What a star expanded to.

    Returns:
        One name per column.
    """
    var out = List[String]()
    for entry in selected:
        out.append(entry.name)
    return out^


def _none() -> List[Target]:
    """An empty `EXCLUDE` list.

    Returns:
        The list.
    """
    return List[Target]()


def _no_replace() -> List[Replacement]:
    """An empty `REPLACE` list.

    Returns:
        The list.
    """
    return List[Replacement]()


def _no_rename() -> List[Renaming]:
    """An empty `RENAME` list.

    Returns:
        The list.
    """
    return List[Renaming]()


def test_a_bare_star_takes_every_column_in_from_order() raises:
    var scopes = Scopes()
    var top = _two_tables(scopes)
    var out = expand(scopes, top, "", _none(), _no_replace(), _no_rename())
    assert_equal(_names(out), [String("a"), "b", "a", "d"])
    assert_equal(out[2].reference.binding, 1)
    assert_equal(out[2].reference.column, 0)


def test_a_qualified_star_takes_one_bindings_columns() raises:
    var scopes = Scopes()
    var top = _two_tables(scopes)
    var out = expand(scopes, top, "u", _none(), _no_replace(), _no_rename())
    assert_equal(_names(out), [String("a"), "d"])


def test_a_qualified_star_does_not_reach_an_outer_level() raises:
    # A qualified column does reach out, and this does not, which is measured
    # rather than assumed: `select (select o.* from u) from t o` is refused.
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "o", [String("a")])
    var inner = scopes.open(top)
    _ = _table(scopes, inner, "u", [String("d")])
    with assert_raises(
        contains='Binder Error: Referenced table "o" not found!'
    ):
        _ = expand(scopes, inner, "o", _none(), _no_replace(), _no_rename())


def test_a_column_merged_by_using_is_only_taken_once() raises:
    var scopes = Scopes()
    var top = _two_tables(scopes)
    scopes.merge(top, 1, "a")
    var out = expand(scopes, top, "", _none(), _no_replace(), _no_rename())
    assert_equal(_names(out), [String("a"), "b", "d"])


def test_exclude_drops_the_column_it_names() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a"), String("b")])
    var out = expand(
        scopes, top, "", [Target("", "b")], _no_replace(), _no_rename()
    )
    assert_equal(_names(out), [String("a")])


def test_exclude_matches_without_regard_to_case() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a"), String("b")])
    var out = expand(
        scopes, top, "", [Target("", "B")], _no_replace(), _no_rename()
    )
    assert_equal(_names(out), [String("a")])


def test_a_bare_exclude_drops_every_column_with_that_name() raises:
    var scopes = Scopes()
    var top = _two_tables(scopes)
    var out = expand(
        scopes, top, "", [Target("", "a")], _no_replace(), _no_rename()
    )
    assert_equal(_names(out), [String("b"), "d"])


def test_a_qualified_exclude_drops_one_of_them() raises:
    var scopes = Scopes()
    var top = _two_tables(scopes)
    var out = expand(
        scopes, top, "", [Target("t", "a")], _no_replace(), _no_rename()
    )
    assert_equal(_names(out), [String("b"), "a", "d"])


def test_exclude_refuses_a_column_that_is_not_there() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a")])
    with assert_raises(
        contains=(
            'Binder Error: Column "nope" in EXCLUDE list not found in FROM'
            " clause"
        )
    ):
        _ = expand(
            scopes, top, "", [Target("", "nope")], _no_replace(), _no_rename()
        )


def test_exclude_refuses_a_qualifier_that_has_the_column_elsewhere() raises:
    var scopes = Scopes()
    var top = _two_tables(scopes)
    with assert_raises(contains='Column "t.d" in EXCLUDE list'):
        _ = expand(
            scopes, top, "", [Target("t", "d")], _no_replace(), _no_rename()
        )


def test_replace_stands_in_for_the_column_and_keeps_its_place() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a"), String("b")])
    var out = expand(
        scopes,
        top,
        "",
        _none(),
        [Replacement(Target("", "a"), 7)],
        _no_rename(),
    )
    assert_equal(_names(out), [String("a"), "b"])
    assert_true(out[0].replaced())
    assert_equal(out[0].node, 7)
    assert_false(out[1].replaced())
    assert_equal(out[1].node, NOT_REPLACED)


def test_a_replace_matching_twice_loses_a_column() raises:
    # DuckDB's, and it is wrong: `select * replace (99 as a) from t, u` gives
    # three columns where `select *` gives four, and nothing says one went
    # missing. Reproduced rather than fixed.
    var scopes = Scopes()
    var top = _two_tables(scopes)
    var out = expand(
        scopes,
        top,
        "",
        _none(),
        [Replacement(Target("", "a"), 7)],
        _no_rename(),
    )
    assert_equal(_names(out), [String("a"), "b", "d"])
    assert_true(out[0].replaced())


def test_replace_refuses_a_column_that_is_not_there() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a")])
    with assert_raises(
        contains=(
            'Binder Error: Column "nope" in REPLACE list not found in FROM'
            " clause"
        )
    ):
        _ = expand(
            scopes,
            top,
            "",
            _none(),
            [Replacement(Target("", "nope"), 7)],
            _no_rename(),
        )


def test_rename_changes_the_output_name_and_nothing_else() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a"), String("b")])
    var out = expand(
        scopes,
        top,
        "",
        _none(),
        _no_replace(),
        [Renaming(Target("", "a"), "z")],
    )
    assert_equal(_names(out), [String("z"), "b"])
    assert_equal(out[0].reference.column, 0)


def test_a_bare_rename_gives_two_columns_the_same_name() raises:
    var scopes = Scopes()
    var top = _two_tables(scopes)
    var out = expand(
        scopes,
        top,
        "",
        _none(),
        _no_replace(),
        [Renaming(Target("", "a"), "z")],
    )
    assert_equal(_names(out), [String("z"), "b", "z", "d"])


def test_a_qualified_rename_renames_one_of_them() raises:
    var scopes = Scopes()
    var top = _two_tables(scopes)
    var out = expand(
        scopes,
        top,
        "",
        _none(),
        _no_replace(),
        [Renaming(Target("u", "a"), "z")],
    )
    assert_equal(_names(out), [String("a"), "b", "z", "d"])


def test_a_rename_that_names_nothing_is_let_through() raises:
    # The one modifier that lets a typo past. EXCLUDE and REPLACE both refuse
    # it, and this is DuckDB's inconsistency rather than a shortcut here.
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a")])
    var out = expand(
        scopes,
        top,
        "",
        _none(),
        _no_replace(),
        [Renaming(Target("", "nope"), "z")],
    )
    assert_equal(_names(out), [String("a")])


def test_the_modifiers_apply_in_the_order_the_grammar_writes_them() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a"), String("b"), String("c")])
    var out = expand(
        scopes,
        top,
        "",
        [Target("", "c")],
        [Replacement(Target("", "a"), 7)],
        [Renaming(Target("", "b"), "z")],
    )
    assert_equal(_names(out), [String("a"), "z"])
    assert_true(out[0].replaced())


def test_a_star_can_expand_to_nothing() raises:
    # Refused at the select list rather than here, since a star is not the
    # only thing a select list can hold.
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a")])
    var out = expand(
        scopes, top, "", [Target("", "a")], _no_replace(), _no_rename()
    )
    assert_equal(len(out), 0)


def test_the_same_name_twice_in_one_list_is_refused() raises:
    # Reported the way the second one was written rather than folded, so the
    # message quotes back a name the query actually has in it.
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a")])
    with assert_raises(
        contains='Parser Error: Duplicate entry "A" in EXCLUDE list'
    ):
        _ = expand(
            scopes,
            top,
            "",
            [Target("", "a"), Target("", "A")],
            _no_replace(),
            _no_rename(),
        )


def test_a_bare_name_counts_against_a_qualified_one() raises:
    var scopes = Scopes()
    var top = _two_tables(scopes)
    with assert_raises(
        contains='Parser Error: Duplicate entry "a" in EXCLUDE list'
    ):
        _ = expand(
            scopes,
            top,
            "",
            [Target("t", "a"), Target("", "a")],
            _no_replace(),
            _no_rename(),
        )


def test_two_qualified_names_are_two_different_columns() raises:
    var scopes = Scopes()
    var top = _two_tables(scopes)
    var out = expand(
        scopes,
        top,
        "",
        [Target("t", "a"), Target("u", "a")],
        _no_replace(),
        _no_rename(),
    )
    assert_equal(_names(out), [String("b"), "d"])


def test_a_duplicate_rename_is_blamed_on_the_exclude_list() raises:
    # DuckDB names a list the query did not write. Reproduced, because a
    # query that is refused there and accepted here is worse than one that is
    # refused the same way in both.
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a")])
    with assert_raises(
        contains='Parser Error: Duplicate entry "a" in EXCLUDE list'
    ):
        _ = expand(
            scopes,
            top,
            "",
            _none(),
            _no_replace(),
            [Renaming(Target("", "a"), "y"), Renaming(Target("", "a"), "z")],
        )


def test_a_name_in_two_lists_is_refused() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a"), String("b")])
    with assert_raises(
        contains=(
            'Parser Error: Column "a" cannot occur in both EXCLUDE and RENAME'
            " list"
        )
    ):
        _ = expand(
            scopes,
            top,
            "",
            [Target("", "A")],
            _no_replace(),
            [Renaming(Target("", "a"), "z")],
        )


def test_a_name_in_two_lists_beats_the_same_name_twice_in_one() raises:
    # The lists are taken in the order they are written and each entry in the
    # order it appears, so the REPLACE entry collides with the EXCLUDE list
    # before the second one collides with the first.
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a"), String("b")])
    with assert_raises(contains="cannot occur in both EXCLUDE and REPLACE"):
        _ = expand(
            scopes,
            top,
            "",
            [Target("", "a")],
            [Replacement(Target("", "a"), 7), Replacement(Target("", "a"), 8)],
            _no_rename(),
        )


def test_a_rename_is_checked_against_the_earlier_lists_first() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a"), String("b")])
    with assert_raises(contains="cannot occur in both REPLACE and RENAME"):
        _ = expand(
            scopes,
            top,
            "",
            _none(),
            [Replacement(Target("", "a"), 7)],
            [Renaming(Target("", "a"), "z"), Renaming(Target("", "a"), "y")],
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
