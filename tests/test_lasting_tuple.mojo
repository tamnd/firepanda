"""Tests for the lasting map on tuples of keys.

Every chunk's ordinals are checked against a dictionary keyed by the tuple
written out as a string, and the key columns read back at the end against the
tuples the dictionary saw first, nulls included.
"""

from std.collections import Dict
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder
from firepanda.hash.lasting import LastingTuple


@fieldwise_init
struct _Row(Copyable, Movable):
    var id: Int
    var id_null: Bool
    var text: String
    var text_null: Bool
    var small: Int


def _name(row: _Row) -> String:
    var id = String("-") if row.id_null else String(row.id)
    var text = String("-") if row.text_null else String("+", row.text)
    return String(id, "|", text, "|", row.small)


def _check(chunks: List[List[_Row]]) raises:
    var map = LastingTuple()
    var want = Dict[String, Int]()
    var seen = List[_Row]()
    var at = List[Int]()
    at.append(2)
    at.append(0)
    at.append(1)
    for c in range(len(chunks)):
        var rows = len(chunks[c])
        var ids = Array[DType.int64](rows)
        var smalls = Array[DType.int32](rows)
        var builder = StringBuilder()
        for i in range(rows):
            ref row = chunks[c][i]
            ids[i] = Int64(row.id)
            if row.id_null:
                ids.set_null(i)
            smalls[i] = Int32(row.small)
            if row.text_null:
                builder.append_null()
            else:
                builder.append(row.text.as_bytes())
        var columns = List[AnyArray]()
        columns.append(AnyArray(ids^))
        columns.append(AnyArray(smalls^))
        columns.append(AnyArray(builder^.finish()))
        var codes = Array[DType.uint32](rows)
        map.ordinals(columns, at, rows, codes)
        for i in range(rows):
            var name = _name(chunks[c][i])
            if name not in want:
                want[name] = len(want)
                seen.append(chunks[c][i].copy())
            assert_equal(Int(codes[i]), want[name])
    assert_equal(map.__len__(), len(want))
    var keys = map.take_keys()
    assert_equal(len(keys), 3)
    var texts = keys[0].strings().copy()
    var ids = keys[1].unsafe_ptr[DType.int64]()
    var smalls = keys[2].unsafe_ptr[DType.int32]()
    for g in range(len(seen)):
        ref row = seen[g]
        assert_equal(keys[0].is_valid(g), not row.text_null)
        if not row.text_null:
            assert_equal(texts[g], row.text)
        assert_equal(keys[1].is_valid(g), not row.id_null)
        if not row.id_null:
            assert_equal(Int(ids[g]), row.id)
        assert_true(keys[2].is_valid(g))
        assert_equal(Int(smalls[g]), row.small)


def _spread(n: Int, groups: Int, seed: Int) -> List[_Row]:
    # The text is a mix of short keys, held in the view, and long ones, held
    # in the payload, and the ids and the text are sometimes missing.
    var out = List[_Row](capacity=n)
    var x = UInt64(seed)
    for _ in range(n):
        x = x * 6364136223846793005 + 1442695040888963407
        var g = Int((x >> 33) % UInt64(groups))
        var text = String("t", g % 7) if g % 3 == 0 else String(
            "https://example.com/page/", g, "/index.html"
        )
        out.append(
            _Row(g * 1_000_003, g % 11 == 0, text, g % 13 == 0, g % 5 - 2)
        )
    return out^


def test_small_chunks() raises:
    var chunks = List[List[_Row]]()
    chunks.append(_spread(300, 50, 1))
    chunks.append(_spread(7, 50, 2))
    chunks.append(_spread(20_000, 900, 3))
    _check(chunks)


def test_many_groups_over_many_chunks() raises:
    var chunks = List[List[_Row]]()
    for c in range(4):
        chunks.append(_spread(100_000 + c * 999, 90_000, c + 5))
    _check(chunks)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
