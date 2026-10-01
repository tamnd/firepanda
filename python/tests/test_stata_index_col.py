"""read_stata builds an index_col of narrow integers as int64, as pandas does."""

import io

import firepanda as fp


def test_narrow_integer_labels_are_int64():
    frame = fp.DataFrame({"a": [1, 2], "k": fp.Series([5, 6], dtype="int8"), "u": [1.5, 2.5]})
    for name, kind in (("index", "int64"), ("k", "int64"), ("a", "int64"), ("u", "float64")):
        buffer = io.BytesIO()
        frame.to_stata(buffer)
        buffer.seek(0)
        assert str(fp.read_stata(buffer, index_col=name).index.dtype) == kind
