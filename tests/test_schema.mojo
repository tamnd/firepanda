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
