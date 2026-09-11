"""The catalog.

The interesting cases are the ones where a plausible implementation is wrong
rather than the ones where it works. Folding, because DuckDB folds a quoted name
at the catalog and does not fold it anywhere else. Replacement, because a
notebook reruns a cell. Dropping, because the payload lists are packed and a
drop moves everything after it. And the suggestion, because a suggestion that
fires on any name at all is worse than no suggestion.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.frame import DataFrame
from firepanda.sql.catalog import (
    KIND_FRAME,
    KIND_VIEW,
    NOT_FOUND,
    Catalog,
    View,
    edit_distance,
    fold,
)


def _frame(rows: Int) -> DataFrame:
    """A frame with a row count and nothing else, to tell two apart.

    Args:
        rows: What its `rows` field will read.

    Returns:
        The frame.
    """
    var out = DataFrame()
    out.rows = rows
    return out^


def test_an_empty_catalog_knows_nothing() raises:
    var catalog = Catalog()
    assert_equal(len(catalog), 0)
    assert_equal(catalog.generation(), 0)
    assert_equal(catalog.find("t"), NOT_FOUND)
    assert_false(catalog.contains("t"))


def test_a_registered_frame_comes_back_by_any_spelling() raises:
    var catalog = Catalog()
    catalog.register("MyTable", _frame(7))
    # DuckDB is case insensitive and case preserving, and quoting a name does
    # not make it case sensitive at the catalog. All three of these resolve.
    assert_true(catalog.contains("mytable"))
    assert_true(catalog.contains("MYTABLE"))
    assert_true(catalog.contains("MyTable"))
    assert_equal(catalog.kind_at(catalog.find("mytable")), KIND_FRAME)
    assert_equal(catalog.frame_at(catalog.find("mytable")).rows, 7)


def test_the_spelling_that_comes_back_is_the_one_that_went_in() raises:
    var catalog = Catalog()
    catalog.register("MyTable", _frame(1))
    assert_equal(catalog.name_at(0), "MyTable")
    assert_equal(catalog.names()[0], "MyTable")


def test_registering_the_same_name_twice_replaces_it() raises:
    var catalog = Catalog()
    catalog.register("t", _frame(1))
    catalog.register("T", _frame(2))
    assert_equal(len(catalog), 1)
    assert_equal(catalog.frame_at(0).rows, 2)
    # The second spelling wins, because it is the one the caller most recently
    # said and the one they will recognize in an error.
    assert_equal(catalog.name_at(0), "T")


def test_a_view_and_a_frame_cannot_share_a_name() raises:
    var catalog = Catalog()
    catalog.register("t", _frame(1))
    catalog.define("t", View("SELECT 1", List[String]()))
    assert_equal(len(catalog), 1)
    assert_equal(catalog.kind_at(0), KIND_VIEW)
    assert_equal(catalog.view_at(0).sql, "SELECT 1")
    catalog.register("t", _frame(3))
    assert_equal(len(catalog), 1)
    assert_equal(catalog.kind_at(0), KIND_FRAME)
    assert_equal(catalog.frame_at(0).rows, 3)


def test_a_view_keeps_the_column_names_its_definer_gave_it() raises:
    var catalog = Catalog()
    var columns: List[String] = [String("a"), String("b")]
    catalog.define("v", View("SELECT 1, 2", columns^))
    assert_equal(len(catalog.view_at(0).columns), 2)
    assert_equal(catalog.view_at(0).columns[1], "b")


def test_dropping_leaves_everything_else_where_it_was() raises:
    # The payload lists are packed, so dropping the first frame moves the second
    # one down a slot and every entry pointing past it has to be told. This is
    # the test that catches it if it is not.
    var catalog = Catalog()
    catalog.register("a", _frame(1))
    catalog.define("v", View("SELECT 1", List[String]()))
    catalog.register("b", _frame(2))
    catalog.register("c", _frame(3))
    assert_true(catalog.drop("a"))
    assert_equal(len(catalog), 3)
    assert_equal(catalog.frame_at(catalog.find("b")).rows, 2)
    assert_equal(catalog.frame_at(catalog.find("c")).rows, 3)
    assert_equal(catalog.view_at(catalog.find("v")).sql, "SELECT 1")


def test_dropping_says_whether_there_was_anything_there() raises:
    var catalog = Catalog()
    catalog.register("a", _frame(1))
    assert_true(catalog.drop("A"))
    assert_false(catalog.drop("a"))
    assert_equal(len(catalog), 0)


def test_the_generation_moves_when_the_namespace_does() raises:
    var catalog = Catalog()
    var start = catalog.generation()
    catalog.register("a", _frame(1))
    assert_true(catalog.generation() > start)
    var after_register = catalog.generation()
    catalog.register("a", _frame(2))
    assert_true(catalog.generation() > after_register)
    var after_replace = catalog.generation()
    _ = catalog.drop("missing")
    assert_equal(catalog.generation(), after_replace)
    _ = catalog.drop("a")
    assert_true(catalog.generation() > after_replace)


def test_an_empty_name_is_not_a_name() raises:
    var catalog = Catalog()
    var raised = False
    try:
        catalog.register("", _frame(1))
    except error:
        raised = True
        assert_true("has to have something in it" in String(error))
    assert_true(raised)
    assert_equal(len(catalog), 0)


def test_a_missing_name_raises_duckdbs_error() raises:
    var catalog = Catalog()
    catalog.register("MyTable", _frame(1))
    var raised = False
    try:
        _ = catalog.resolve("mytabel")
    except error:
        raised = True
        var text = String(error)
        assert_true(
            "Catalog Error: Table with name mytabel does not exist!" in text
        )
        assert_true('Did you mean "MyTable"?' in text)
    assert_true(raised)


def test_a_name_that_resembles_nothing_gets_no_suggestion() raises:
    # DuckDB suggests out of a catalog that includes the Postgres compatibility
    # tables, so it can answer with a name the user has never heard of. We have
    # no such tables, and a guess with nothing behind it reads as a bug.
    var catalog = Catalog()
    catalog.register("lineitem", _frame(1))
    assert_equal(catalog.nearest("zzzzzzzzzzzzzzzz"), NOT_FOUND)
    assert_false("Did you mean" in catalog.missing("zzzzzzzzzzzzzzzz"))
    assert_true("Did you mean" in catalog.missing("lineitm"))


def test_the_nearest_name_is_the_nearest_one() raises:
    var catalog = Catalog()
    catalog.register("orders", _frame(1))
    catalog.register("lineitem", _frame(2))
    assert_equal(catalog.name_at(catalog.nearest("lineitm")), "lineitem")
    assert_equal(catalog.name_at(catalog.nearest("order")), "orders")


def test_resolve_gives_back_the_index_find_would() raises:
    var catalog = Catalog()
    catalog.register("a", _frame(1))
    catalog.register("b", _frame(2))
    assert_equal(catalog.resolve("B"), catalog.find("b"))


def test_folding_is_ascii_and_downward() raises:
    assert_equal(fold("MyTable"), "mytable")
    assert_equal(fold("ORDERS"), "orders")
    # Not ASCII, so it is left alone rather than folded by a table we do not
    # have. Two names that differ only outside ASCII stay two names.
    assert_equal(fold("Ä"), "Ä")


def test_the_edit_distance_is_the_edit_distance() raises:
    assert_equal(edit_distance("alpha", "alpha", 10), 0)
    assert_equal(edit_distance("alpha", "alpah", 10), 2)
    assert_equal(edit_distance("alpha", "alph", 10), 1)
    assert_equal(edit_distance("", "abc", 10), 3)
    assert_equal(edit_distance("abc", "", 10), 3)


def test_the_edit_distance_gives_up_at_the_limit() raises:
    # The point of the limit is that the answer above it is never needed, so the
    # function is allowed to stop and say so rather than pay for it.
    assert_equal(edit_distance("alpha", "zzzzzzzzzz", 3), 3)
    assert_equal(edit_distance("alpha", "beta", 2), 2)
    assert_equal(edit_distance("alpha", "alpha", 1), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
