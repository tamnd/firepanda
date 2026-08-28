"""Tests for ordering a text column.

The sort is two things stacked and each one can be right while the other is
wrong, so the tests are written to separate them. The first eight bytes of an
element become a `UInt64` and go through the same radix passes an int64 column
does, and then the runs of rows whose first eight bytes were identical are
finished by a comparison sort. A test whose elements all differ in their first
byte exercises only the radix half and would pass with the tie break deleted, so
most of what is here is elements chosen to collide.

The collisions worth writing are the ones the boundary produces. Two elements
that agree on exactly eight bytes and differ on the ninth are the case the radix
pass cannot see at all. An element that is a prefix of another is the case where
the key is padded with zeros and the tie break has to fall back on the length.
And an element longer than twelve bytes is the case where the comparison leaves
the view and reads the payload, which is a different code path from a short one.

Stability is checked rather than assumed, because a multi-key sort is built out of
stable single-key sorts and nothing else, and an unstable tie break would make
`sort_values` on two columns quietly wrong in a way no single-column test sees.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder, strings_from_list
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series
from firepanda.kernel.sort import argsort_any
from firepanda.testing.rng import Rng


def text_series(values: List[String]) raises -> Series:
    """Builds a named text column with no nulls.

    Args:
        values: The elements, in order.

    Returns:
        A series called `s`.
    """
    return Series("s", strings_from_list(values))


def with_nulls(values: List[String], present: List[Bool]) raises -> Series:
    """Builds a named text column with nulls where asked.

    Args:
        values: The elements. The entry under a false flag is ignored.
        present: One flag per element.

    Returns:
        A series called `s`.
    """
    var builder = StringBuilder(capacity=len(values))
    for i in range(len(values)):
        if present[i]:
            builder.append(values[i].as_bytes())
        else:
            builder.append_null()
    return Series("s", builder^.finish())


def texts(s: Series) raises -> List[String]:
    """Reads a text column out as a list.

    Args:
        s: The column.

    Returns:
        Its elements in order.
    """
    var out = List[String](capacity=len(s))
    for i in range(len(s)):
        out.append(s.text(i))
    return out^


def assert_texts(s: Series, want: List[String]) raises:
    """Asserts a text column holds exactly the given elements in order.

    Args:
        s: The column.
        want: What it should hold.

    Raises:
        If the lengths differ or any element differs.
    """
    var got = texts(s)
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i], String("at ", i))


def test_the_radix_half_on_its_own() raises:
    """Every element differs in its first byte, so no tie break is entered."""
    var s = text_series(["delta", "alpha", "charlie", "bravo"])
    assert_texts(s.sort_values(), ["alpha", "bravo", "charlie", "delta"])


def test_elements_agreeing_on_exactly_eight_bytes() raises:
    """The case the key cannot see, since the key is those eight bytes."""
    var s = text_series(["abcdefghz", "abcdefgha", "abcdefghm"])
    assert_texts(s.sort_values(), ["abcdefgha", "abcdefghm", "abcdefghz"])


def test_a_prefix_sorts_before_what_extends_it() raises:
    var s = text_series(["abcdefghij", "abcdefgh", "abcdefghi", "abcdef"])
    assert_texts(
        s.sort_values(),
        ["abcdef", "abcdefgh", "abcdefghi", "abcdefghij"],
    )


def test_ordering_reads_the_payload_when_it_has_to() raises:
    """Both are past the inline limit and agree well beyond the eight byte key.
    """
    var s = text_series(
        [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy cat",
            "the quick brown fox jumps over the lazy dogs",
        ]
    )
    assert_texts(
        s.sort_values(),
        [
            "the quick brown fox jumps over the lazy cat",
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy dogs",
        ],
    )


def test_an_empty_element_sorts_first() raises:
    var s = text_series(["b", "", "a"])
    assert_texts(s.sort_values(), ["", "a", "b"])


def test_bytes_are_compared_as_unsigned() raises:
    """A high byte sorts after every ASCII one rather than before it.

    This is the test that would fail if the comparison used a signed byte, which
    is what a naive `Int8` subtraction gives, and it would fail only on non-ASCII
    input.
    """
    var s = text_series(["~tilde", "ábove", "Above"])
    assert_texts(s.sort_values(), ["Above", "~tilde", "ábove"])


def test_descending_reverses_the_tie_break_too() raises:
    """The complemented key orders the runs; only the tie break sees direction.

    Every element here shares its first eight bytes, so the radix passes put them
    all in one run and the whole answer comes out of the comparison sort. If that
    sort did not know about the direction the result would be ascending.
    """
    var s = text_series(["prefix00b", "prefix00c", "prefix00a"])
    assert_texts(
        s.sort_values(descending=True), ["prefix00c", "prefix00b", "prefix00a"]
    )


def test_nulls_go_last_by_default_and_first_when_asked() raises:
    var s = with_nulls(["b", "x", "a", "y"], [True, False, True, False])
    var last = s.sort_values()
    assert_equal(last.text(0), "a")
    assert_equal(last.text(1), "b")
    assert_false(last.is_valid(2))
    assert_false(last.is_valid(3))

    var first = s.sort_values(nulls_first=True)
    assert_false(first.is_valid(0))
    assert_false(first.is_valid(1))
    assert_equal(first.text(2), "a")
    assert_equal(first.text(3), "b")


def test_the_tie_break_is_stable() raises:
    """Rows the comparison cannot separate keep the order they arrived in.

    Every element is identical, so the sort has nothing to order by and the only
    correct answer is the input order. The payload column rides along on the
    permutation and is what says whether that happened.
    """
    var keys = Series("k", strings_from_list(["same", "same", "same", "same"]))
    var tags = Array[DType.int64](4)
    var values = tags.unsafe_ptr()
    for i in range(4):
        values.unsafe_offset(i).unsafe_write(Int64(i))
    var frame = DataFrame.from_series([keys^, Series("t", tags^)])

    var sorted = frame.sort_by("k")
    var out = sorted.column("t").as_typed[DType.int64]()
    var got = out.unsafe_ptr()
    for i in range(4):
        assert_equal(Int(got.unsafe_offset(i).unsafe_load()), i)


def test_a_long_tied_run_goes_through_the_merge() raises:
    """More rows in one run than the insertion sort threshold.

    Every element shares a nine byte prefix, so the radix passes leave a single
    run of two hundred rows and `_merge_sort_run` is what orders it. The elements
    are built so that the correct answer is a known permutation rather than the
    input order.
    """
    var count = 200
    var values = List[String](capacity=count)
    for i in range(count):
        # Descending in the part that decides, so a sort that returned its input
        # unchanged would fail rather than accidentally pass.
        var n = count - 1 - i
        values.append(String("sharedpre", _padded(n)))
    var s = text_series(values)

    var sorted = texts(s.sort_values())
    for i in range(count):
        assert_equal(sorted[i], String("sharedpre", _padded(i)))


def _padded(n: Int) -> String:
    """Formats a number to three digits so that text order matches number order.

    Args:
        n: The number, below a thousand.

    Returns:
        The zero padded digits.
    """
    var out = String(n)
    while out.byte_length() < 3:
        out = String("0", out)
    return out^


def test_against_a_reference_sort_on_random_text() raises:
    """A thousand random elements against an insertion sort of the same list.

    The reference is deliberately the slowest correct thing that can be written,
    an insertion sort over `String` comparison, because the point of a reference
    is that it shares no code with what it is checking. The lengths are drawn to
    straddle both boundaries that matter, the eight byte key and the twelve byte
    inline limit.
    """
    var rng = Rng(0xC0FFEE)
    var count = 1000
    var values = List[String](capacity=count)
    for _ in range(count):
        # Mostly short, because that is the shape of real text, with a quarter
        # long enough to be in the payload and a fifth sharing a prefix so that
        # the tie break is entered often.
        var length = 1 + rng.next_below(20)
        var text = String()
        if rng.next_below(5) == 0:
            text += "common__"
        for _ in range(length):
            text += _letter(rng.next_below(4))
        values.append(text^)

    var want = List[String](capacity=count)
    for i in range(count):
        want.append(values[i])
    _reference_sort(want)

    var got = texts(text_series(values).sort_values())
    assert_equal(len(got), count)
    for i in range(count):
        assert_equal(got[i], want[i], String("at ", i))


def _letter(k: Int) -> String:
    """Returns one of four letters.

    A small alphabet on purpose. Four letters over twenty positions collide on
    the first eight far more often than twenty six would, which is what puts the
    tie break under load.

    Args:
        k: Which letter, 0 to 3.

    Returns:
        The letter.
    """
    if k == 0:
        return String("a")
    if k == 1:
        return String("b")
    if k == 2:
        return String("c")
    return String("d")


def _reference_sort(mut values: List[String]):
    """Insertion sorts a list of strings, sharing no code with the kernel.

    Args:
        values: The list, sorted in place.
    """
    for i in range(1, len(values)):
        var value = values[i]
        var j = i - 1
        while j >= 0 and values[j] > value:
            values[j + 1] = values[j]
            j -= 1
        values[j + 1] = value^


def test_a_frame_sorts_on_text_then_number() raises:
    """A two key sort where the text key is the dominant one.

    This is what the stability of the tie break is for. The second pass sorts by
    the text and has to leave the rows it considers equal in the order the first
    pass gave them.
    """
    var city = Series(
        "city",
        strings_from_list(["oslo", "lima", "oslo", "lima", "oslo"]),
    )
    var year = Array[DType.int64](5)
    var values = year.unsafe_ptr()
    values.unsafe_offset(0).unsafe_write(2021)
    values.unsafe_offset(1).unsafe_write(2020)
    values.unsafe_offset(2).unsafe_write(2019)
    values.unsafe_offset(3).unsafe_write(2022)
    values.unsafe_offset(4).unsafe_write(2020)
    var frame = DataFrame.from_series([city^, Series("year", year^)])

    var sorted = frame.sort_values(
        ["city", "year"], [False, False], [False, False]
    )
    var out = sorted.column("year").as_typed[DType.int64]()
    var got = out.unsafe_ptr()

    assert_equal(sorted.column("city").text(0), "lima")
    assert_equal(Int(got.unsafe_offset(0).unsafe_load()), 2020)
    assert_equal(sorted.column("city").text(1), "lima")
    assert_equal(Int(got.unsafe_offset(1).unsafe_load()), 2022)
    assert_equal(sorted.column("city").text(2), "oslo")
    assert_equal(Int(got.unsafe_offset(2).unsafe_load()), 2019)
    assert_equal(sorted.column("city").text(3), "oslo")
    assert_equal(Int(got.unsafe_offset(3).unsafe_load()), 2020)
    assert_equal(sorted.column("city").text(4), "oslo")
    assert_equal(Int(got.unsafe_offset(4).unsafe_load()), 2021)


def test_argsort_of_one_row_and_of_none() raises:
    var one = AnyArray(strings_from_list(["only"]))
    assert_equal(len(argsort_any(one)), 1)

    var none = AnyArray(strings_from_list(List[String]()))
    assert_equal(len(argsort_any(none)), 0)


def test_every_element_identical_is_the_input_order() raises:
    var s = text_series(["x", "x", "x"])
    var order = argsort_any(AnyArray(strings_from_list(["x", "x", "x"])))
    var got = order.unsafe_ptr()
    for i in range(3):
        assert_equal(Int(got.unsafe_offset(i).unsafe_load()), i)
    assert_equal(len(s), 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
