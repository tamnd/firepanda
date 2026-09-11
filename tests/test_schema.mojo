"""Tests for the frame shape.

A schema is an ordered list of named, typed columns. Order is part of the value,
so two schemas with the same fields in a different order are different schemas;
that is what pandas does and reordering silently would be worse than raising.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema


def sample_schema() -> Schema:
    """Builds the schema the tests below work against.

    Returns:
        Three fields covering a nullable column, a non-nullable one and a string.
    """
    var schema = Schema()
    schema.append(Field("id", LogicalType.INT64, False))
    schema.append(Field("price", LogicalType.FLOAT64))
    schema.append(Field("name", LogicalType.STRING))
    return schema^


def test_fields_default_to_nullable() raises:
    # Arrow and pandas both default this way. A column that cannot hold a null is
    # the special case and should have to say so.
    var field = Field("x", LogicalType.INT32)
    assert_true(field.nullable)


def test_length_and_indexing() raises:
    var schema = sample_schema()
    assert_equal(len(schema), 3)
    assert_equal(schema[0].name, "id")
    assert_false(schema[0].nullable)
    assert_equal(schema[2].dtype, LogicalType.STRING)


def test_index_of() raises:
    var schema = sample_schema()
    assert_equal(schema.index_of("price"), 1)
    assert_true(schema.has("name"))
    assert_false(schema.has("missing"))


def test_index_of_unknown_column_raises() raises:
    var schema = sample_schema()
    with assert_raises(contains="missing"):
        _ = schema.index_of("missing")


def test_index_of_all_agrees_with_index_of() raises:
    # The bulk lookup exists so that a projection of a wide frame resolves its
    # names once instead of twice. It is only worth having if it gives the same
    # answer, so it is checked against the scan rather than against a list
    # somebody typed.
    var schema = sample_schema()
    var names: List[String] = ["name", "id", "price", "id"]
    var at = schema.index_of_all(names)
    assert_equal(len(at), 4)
    for i in range(len(names)):
        assert_equal(at[i], schema.index_of(names[i]))


def test_index_of_all_unknown_column_raises() raises:
    var schema = sample_schema()
    with assert_raises(contains="missing"):
        _ = schema.index_of_all(["id", "missing"])


def test_index_of_all_asks_for_nothing() raises:
    var schema = sample_schema()
    assert_equal(len(schema.index_of_all(List[String]())), 0)


def test_index_of_all_takes_the_first_of_a_duplicate_name() raises:
    # Duplicate column names are legal in a schema because pandas allows them,
    # and `index_of` answers with the first. The bulk lookup has to say the same
    # thing, and `select` leans on it: it decides that a name was asked for twice
    # by seeing the same position twice, which is only sound if a repeated name
    # always comes back as the same position.
    var schema = Schema(
        [
            Field("a", LogicalType.INT64),
            Field("b", LogicalType.INT64),
            Field("a", LogicalType.STRING),
        ]
    )
    var at = schema.index_of_all(["a", "b", "a"])
    assert_equal(at[0], 0)
    assert_equal(at[1], 1)
    assert_equal(at[2], 0)
    assert_equal(schema.index_of_all(["a"])[0], schema.index_of("a"))


def test_select_at_projects_by_position() raises:
    # The positional half of `select`, which exists so the frame does not resolve
    # the same names a second time on its way through here.
    var schema = sample_schema()
    var out = schema.select_at([2, 0])
    assert_equal(len(out), 2)
    assert_equal(out[0].name, "name")
    assert_equal(out[1].name, "id")


def test_select_at_off_the_end_raises() raises:
    var schema = sample_schema()
    with assert_raises(contains="position"):
        _ = schema.select_at([0, 9])


def test_a_hundred_and_five_columns_project_by_name() raises:
    # The width of the ClickBench hits table. Nothing here is a timing, it is the
    # shape the lookup was changed for, and a bulk lookup that only works on
    # three columns would pass every other test in this file.
    var schema = Schema()
    for i in range(105):
        schema.append(Field("c" + String(i), LogicalType.INT64))

    var names = List[String]()
    for i in range(105):
        names.append("c" + String(104 - i))
    var at = schema.index_of_all(names)
    for i in range(105):
        assert_equal(at[i], 104 - i)

    var reversed = schema.select(names)
    assert_equal(len(reversed), 105)
    assert_equal(reversed[0].name, "c104")
    assert_equal(reversed[104].name, "c0")


def test_select_keeps_the_requested_order() raises:
    var schema = sample_schema()
    var picked = schema.select(["name", "id"])
    assert_equal(len(picked), 2)
    assert_equal(picked[0].name, "name")
    assert_equal(picked[1].name, "id")
    assert_false(picked[1].nullable)


def test_select_unknown_column_raises() raises:
    var schema = sample_schema()
    with assert_raises():
        _ = schema.select(["id", "nope"])


def test_equality_is_order_sensitive() raises:
    var schema = sample_schema()
    var same = sample_schema()
    assert_true(schema == same)

    var reordered = schema.select(["price", "id", "name"])
    assert_true(schema != reordered)


def test_equality_covers_nullability() raises:
    var left = Schema([Field("a", LogicalType.INT8, True)])
    var right = Schema([Field("a", LogicalType.INT8, False)])
    assert_true(left != right)


def test_printing() raises:
    var schema = sample_schema()
    assert_equal(
        String(schema), "id: int64 not null\nprice: float64\nname: string"
    )


def test_empty_schema() raises:
    var schema = Schema()
    assert_equal(len(schema), 0)
    assert_equal(String(schema), "")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
