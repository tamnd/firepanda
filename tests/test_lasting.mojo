"""Tests for the lasting map on fixed width keys.

Every chunk's ordinals are checked against a dictionary that hands out the next
ordinal the first time it sees a key, which is the numbering the map promises:
consecutive, in the order the keys first arrived, and the same across chunks.
"""

from std.collections import Dict
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import strings_from_list
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


def test_a_tall_chunk_then_another() raises:
    # A chunk tall enough that its table is grown well past the first size,
    # then a chunk that brings as many new keys as it repeats old ones.
    var chunks = List[List[Int]]()
    chunks.append(_spread(300_000, 150_000, 11))
    chunks.append(_spread(140_000, 200_000, 12))
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




def _coded(picked: List[Int], var categories: List[String]) raises -> AnyArray:
    var codes = Array[DType.int32](len(picked))
    for i in range(len(picked)):
        codes[i] = Int32(picked[i])
    return AnyArray.dictionary_encoded(codes^, strings_from_list(categories))


def test_a_coded_key_is_grouped_on_its_codes() raises:
    """Two chunks over one set of categories go through the codes, and the
    keys come back as text over those categories."""
    var col = _coded([2, 0, 2, 1, 0], ["x", "y", "z"])
    var map = LastingKeys()
    var codes = Array[DType.uint32](3)
    map.ordinals(col.slice(0, 3), 3, codes)
    assert_equal(Int(codes[0]), 0)
    assert_equal(Int(codes[1]), 1)
    assert_equal(Int(codes[2]), 0)
    var more = Array[DType.uint32](2)
    map.ordinals(col.slice(3, 5), 2, more)
    assert_equal(Int(more[0]), 2)
    assert_equal(Int(more[1]), 1)
    assert_true(map.coded, "still on the codes")
    var keys = map.take_keys()
    assert_true(keys.is_coded(), "the keys keep the categories")
    var flat = keys.decoded()
    assert_equal(len(flat), 3)
    assert_equal(String(flat.strings()[0]), "z")
    assert_equal(String(flat.strings()[1]), "x")
    assert_equal(String(flat.strings()[2]), "y")


def test_a_chunk_coded_apart_moves_the_map_to_text() raises:
    """A chunk over other categories, and then a flat one, keep the ordinals
    the first chunk handed out."""
    var map = LastingKeys()
    var first = Array[DType.uint32](3)
    map.ordinals(_coded([1, 0, 1], ["a", "b"]), 3, first)
    assert_equal(Int(first[0]), 0)
    assert_equal(Int(first[1]), 1)
    var second = Array[DType.uint32](3)
    map.ordinals(_coded([0, 1, 2], ["b", "c", "a"]), 3, second)
    assert_false(map.coded, "off the codes")
    assert_equal(Int(second[0]), 0, "b was first")
    assert_equal(Int(second[1]), 2, "c is new")
    assert_equal(Int(second[2]), 1, "a was second")
    var third = Array[DType.uint32](2)
    map.ordinals(AnyArray(strings_from_list(["c", "d"])), 2, third)
    assert_equal(Int(third[0]), 2)
    assert_equal(Int(third[1]), 3)
    var keys = map.take_keys()
    assert_equal(len(keys), 4)
    assert_equal(String(keys.strings()[0]), "b")
    assert_equal(String(keys.strings()[3]), "d")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
