"""read_json with lines=True and a chunksize reads the lines a chunk at a time."""

import io

import pytest

import firepanda as fp

TEXT = "\n".join(f'{{"a": {k}, "b": "x{k}"}}' for k in range(7))


def test_chunks_carry_labels_on():
    with fp.read_json(io.StringIO(TEXT), lines=True, chunksize=3) as reader:
        chunks = list(reader)
    assert [c.index.tolist() for c in chunks] == [[0, 1, 2], [3, 4, 5], [6]]
    assert chunks[1]["a"].tolist() == [3, 4, 5]
    assert str(chunks[2]["a"].dtype) == "int64"


def test_nrows_is_looked_at_before_each_chunk():
    reader = fp.read_json(io.StringIO(TEXT), lines=True, chunksize=3, nrows=5)
    assert [len(c) for c in reader] == [3, 3]


def test_read_joins_the_chunks():
    got = fp.read_json(io.StringIO(TEXT), lines=True, chunksize=2).read()
    assert got.index.tolist() == list(range(7))


def test_series_chunks():
    reader = fp.read_json(io.StringIO("1\n2\n3"), lines=True, chunksize=2, typ="series")
    assert [c.tolist() for c in reader] == [[1, 2], [3]]


@pytest.mark.parametrize("bad", [0, -1, 1.5, "a"])
def test_bad_chunksize(bad):
    with pytest.raises(ValueError, match="must be an integer"):
        fp.read_json(io.StringIO(TEXT), lines=True, chunksize=bad)
