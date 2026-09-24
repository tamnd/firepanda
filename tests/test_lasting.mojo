"""Tests for the lasting map on fixed width keys.

Every chunk's ordinals are checked against a dictionary that hands out the next
ordinal the first time it sees a key, which is the numbering the map promises:
consecutive, in the order the keys first arrived, and the same across chunks.
"""

from std.collections import Dict
from std.testing import TestSuite, assert_equal

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.hash.lasting import LastingKeys


def _check[dt: DType](chunks: List[List[Int]]) raises:
    var map = LastingKeys()
    var want = Dict[Int, Int]()
    for c in range(len(chunks)):
        var rows = len(chunks[c])
        var col = Array[dt](rows)
        for i in range(rows):
            col[i] = Scalar[dt](chunks[c][i])
        var codes = Array[DType.uint32](rows)
        map.ordinals(AnyArray(col^), rows, codes)
        for i in range(rows):
            var key = chunks[c][i]
            if key not in want:
                want[key] = len(want)
            assert_equal(Int(codes[i]), want[key])
    assert_equal(map.__len__(), len(want))
    var keys = map.take_keys()
    assert_equal(len(keys), len(want))
    var got = keys.unsafe_ptr[dt]()
    for entry in want.items():
        assert_equal(
            Int(got.unsafe_offset(entry.value).unsafe_load()), entry.key
        )


def _spread(n: Int, groups: Int, seed: Int) -> List[Int]:
    var out = List[Int](capacity=n)
    var x = UInt64(seed)
    for _ in range(n):
        x = x * 6364136223846793005 + 1442695040888963407
        out.append(Int((x >> 33) % UInt64(groups)) * 1_000_003 - 7)
    return out^


def test_wide_keys_over_many_chunks() raises:
    var chunks = List[List[Int]]()
    for c in range(5):
        chunks.append(_spread(100_000 + c * 999, 60_000, c + 1))
    _check[DType.int64](chunks)


def test_a_small_chunk_runs_on_one_task() raises:
    var chunks = List[List[Int]]()
    chunks.append(_spread(300, 50, 3))
    chunks.append(_spread(7, 50, 4))
    chunks.append(_spread(20_000, 900, 5))
    _check[DType.int64](chunks)


def test_the_direct_table_hands_over_mid_chunk() raises:
    var first = List[Int]()
    for i in range(5000):
        first.append((i * 31) % 700)
    var second = List[Int]()
    for i in range(50_000):
        if i == 1234:
            second.append(1 << 40)
        else:
            second.append((i * 17) % 3000 - 1500)
    var chunks = List[List[Int]]()
    chunks.append(first^)
    chunks.append(second^)
    chunks.append(_spread(40_000, 20_000, 9))
    _check[DType.int64](chunks)


def test_narrow_signed_keys() raises:
    var chunks = List[List[Int]]()
    for c in range(3):
        var one = List[Int]()
        for i in range(30_000):
            one.append(((i * 7 + c * 13) % 256) - 128)
        chunks.append(one^)
    _check[DType.int8](chunks)


def test_unsigned_keys_past_the_signed_range() raises:
    var chunks = List[List[Int]]()
    var one = List[Int]()
    for i in range(70_000):
        one.append(Int(UInt32(4_000_000_000) + UInt32((i * 7919) % 40_000)))
    chunks.append(one^)
    _check[DType.uint32](chunks)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
