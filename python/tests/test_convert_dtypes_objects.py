"""convert_dtypes reads a column of objects by the kind pandas infers for it."""

import pytest

import firepanda as fp


@pytest.mark.parametrize(
    ("values", "options", "expected"),
    [
        ([1, None], {}, "Int64"),
        ([1, None], {"convert_integer": False}, "Float64"),
        (["x", None], {}, "string"),
        (["x", None], {"convert_string": False}, "str"),
        (["x", 1], {}, "object"),
        ([1.0, None], {}, "Int64"),
        ([1.5, None, 2], {}, "Float64"),
        ([True, None], {}, "boolean"),
    ],
)
def test_objects_convert_by_their_kind(values, options, expected):
    column = fp.Series(values, dtype="object").convert_dtypes(**options)
    assert str(column.dtype) == expected


def test_a_frame_converts_its_object_columns():
    frame = fp.DataFrame({"a": fp.Series(["a", "b"], dtype="object"), "b": [1.0, 2.0]})
    assert [str(dtype) for dtype in frame.convert_dtypes().dtypes] == ["string", "Int64"]
