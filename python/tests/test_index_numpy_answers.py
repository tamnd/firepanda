"""Flags and positions on an index answer numpy arrays, as pandas answers them.

`isin`, `isna`, `notna`, `duplicated`, `argsort`, the `get_indexer` family,
`asof_locs`, and the positions of `sort_values`, `join` and `reindex` were
lists here, and pandas answers each with a numpy array of bools or of
positions. An index also takes a numpy array as a key, so the answers can be
handed straight back.
"""

from __future__ import annotations

import math

import numpy as np

import firepanda as fp


def kind(answer):
    """The numpy kind of an answer, which fails loudly for a list."""
    assert type(answer).__name__ == "ndarray"
    return answer.dtype.kind


def test_flags_are_numpy_bools():
    index = fp.Index([1.0, 2.0, math.nan])
    assert kind(index.isna()) == kind(index.notna()) == "b"
    assert index.isnull().tolist() == [False, False, True]
    assert kind(fp.Index([1, 2, 3]).isin([1, 4])) == "b"
    assert fp.Index(["a", "b", "a"]).duplicated().tolist() == [False, False, True]


def test_positions_are_numpy_integers():
    index = fp.Index(["c", "a", "b"])
    assert kind(index.argsort()) == "i"
    assert index.get_indexer(["a", "x"]).tolist() == [1, -1]
    assert kind(index.get_indexer_for(["b"])) == "i"
    found, missing = fp.Index(["b", "a", "b"]).get_indexer_non_unique(["b", "z"])
    assert found.tolist() == [0, 2, -1]
    assert missing.tolist() == [1]


def test_the_positions_beside_an_index_are_numpy_integers():
    made, order = fp.Index([10, 100, 1]).sort_values(return_indexer=True)
    assert made.tolist() == [1, 10, 100]
    assert order.tolist() == [2, 0, 1]
    joined, mine, theirs = fp.Index([1, 2]).join(fp.Index([3]), how="outer", return_indexers=True)
    assert joined.tolist() == [1, 2, 3]
    assert mine.tolist() == [0, 1, -1]
    assert theirs.tolist() == [-1, -1, 0]
    labels, where = fp.Index(["car", "bike"]).reindex(["bike", "boat"])
    assert labels.tolist() == ["bike", "boat"]
    assert where.tolist() == [1, -1]


def test_an_index_takes_a_numpy_key():
    index = fp.Index(["c", "a", "b"])
    assert index[index.argsort()].tolist() == ["a", "b", "c"]
    assert index[np.array([True, False, True])].tolist() == ["c", "b"]
    assert index[np.int64(1)] == "a"
