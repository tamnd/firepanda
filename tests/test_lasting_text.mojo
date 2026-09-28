"""Tests for the lasting map on text keys.

The same check as `test_lasting`: every chunk's ordinals against a dictionary
that hands out the next ordinal the first time it sees a key, and the stored
keys against the dictionary at the end.
"""

from std.collections import Dict
from std.testing import TestSuite, assert_equal

from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder
from firepanda.hash.lasting import LastingText


def _check(chunks: List[List[String]]) raises:
    var map = LastingText()
    var want = Dict[String, Int]()
    for c in range(len(chunks)):
        var rows = len(chunks[c])
        var builder = StringBuilder()
        for i in range(rows):
            builder.append(chunks[c][i].as_bytes())
        var col = builder^.finish()
        var codes = Array[DType.uint32](rows)
        map.ordinals(col, rows, codes)
        for i in range(rows):
            var key = chunks[c][i]
            if key not in want:
                want[key] = len(want)
            assert_equal(Int(codes[i]), want[key])
    assert_equal(map.__len__(), len(want))
    var keys = map.take_keys()
    assert_equal(len(keys), len(want))
    for entry in want.items():
        assert_equal(keys[entry.value], entry.key)


def _spread(n: Int, groups: Int, seed: Int) -> List[String]:
    # Short keys stay inline in their view and long ones do not, and both
    # kinds are compared differently, so the keys are a mix of the two.
    var out = List[String](capacity=n)
    var x = UInt64(seed)
    for _ in range(n):
        x = x * 6364136223846793005 + 1442695040888963407
        var g = Int((x >> 33) % UInt64(groups))
        if g % 3 == 0:
            out.append(String("k", g))
        else:
            out.append(String("https://example.com/page/", g, "/index.html"))
    return out^


def test_small_chunks() raises:
    var chunks = List[List[String]]()
    chunks.append(_spread(300, 50, 1))
    chunks.append(_spread(7, 50, 2))
    chunks.append(_spread(20_000, 900, 3))
    _check(chunks)


def test_a_tall_chunk_then_another() raises:
    # A chunk tall enough that its table is grown well past the first size,
    # then a chunk that brings as many new keys as it repeats old ones.
    var chunks = List[List[String]]()
    chunks.append(_spread(300_000, 120_000, 4))
    chunks.append(_spread(140_000, 160_000, 5))
    _check(chunks)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
