"""Tests for the JSON scanner.

Everything here works on a byte span and checks offsets, which is the whole
point of the file being separate from the reader: a scanner that reports the
wrong span is a reader that reads the wrong bytes, and finding that out from a
frame two layers up is much harder than finding it out here.

The malformed cases are half the file on purpose. A JSON reader that repairs
what it was given turns a corrupt document into a plausible table, and every
error below is a document some other reader accepts by inventing something.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.io.jsonscan import (
    JSON_ARRAY,
    JSON_FALSE,
    JSON_NULL,
    JSON_NUMBER,
    JSON_OBJECT,
    JSON_STRING,
    JSON_TRUE,
    Member,
    scan_object,
    scan_value,
    skip_space,
    text_of,
)


def _bytes(text: StringSlice) -> List[UInt8]:
    var out = List[UInt8](capacity=text.byte_length())
    for byte in text.as_bytes():
        out.append(byte)
    return out^


def _members(text: StringSlice) raises -> List[Member]:
    var raw = _bytes(text)
    var out = List[Member]()
    _ = scan_object(Span(raw), 0, out)
    return out^


def test_an_object_reports_a_member_per_key() raises:
    var raw = _bytes('{"a": 1, "b": "two", "c": true}')
    var members = List[Member]()
    var after = scan_object(Span(raw), 0, members)
    assert_equal(after, len(raw))
    assert_equal(len(members), 3)
    assert_equal(text_of(Span(raw), members[0].key), "a")
    assert_equal(text_of(Span(raw), members[1].key), "b")
    assert_equal(text_of(Span(raw), members[2].key), "c")
    assert_equal(members[0].value.kind, JSON_NUMBER)
    assert_equal(members[1].value.kind, JSON_STRING)
    assert_equal(members[2].value.kind, JSON_TRUE)


def test_a_string_span_is_inside_its_quotes_and_after_is_outside() raises:
    # The three offsets are the reason this struct has three offsets. A parser
    # wants the content and the scan wants to continue past the quote.
    var raw = _bytes('{"k":"hi"}')
    var members = List[Member]()
    _ = scan_object(Span(raw), 0, members)
    var value = members[0].value
    assert_equal(value.start, 6)
    assert_equal(value.end, 8)
    assert_equal(value.after, 9)
    assert_false(value.escaped)


def test_a_number_keeps_the_bytes_it_was_written_as() raises:
    var members = _members('{"a": -12.5e+3, "b": 0, "c": 7}')
    assert_equal(len(members), 3)
    for i in range(3):
        assert_equal(members[i].value.kind, JSON_NUMBER)
    var raw = _bytes('{"a": -12.5e+3, "b": 0, "c": 7}')
    var first = members[0].value
    assert_equal(
        String(
            StringSlice(unsafe_from_utf8=Span(raw)[first.start : first.end])
        ),
        "-12.5e+3",
    )


def test_null_true_and_false_are_three_different_things() raises:
    var members = _members('{"a": null, "b": true, "c": false}')
    assert_equal(members[0].value.kind, JSON_NULL)
    assert_equal(members[1].value.kind, JSON_TRUE)
    assert_equal(members[2].value.kind, JSON_FALSE)


def test_whitespace_anywhere_it_is_allowed_is_ignored() raises:
    var members = _members('{ \n\t "a" \n : \r 1 , "b" : 2 \n }')
    assert_equal(len(members), 2)
    assert_equal(
        text_of(
            Span(_bytes('{ \n\t "a" \n : \r 1 , "b" : 2 \n }')), members[0].key
        ),
        "a",
    )
    assert_equal(members[1].value.kind, JSON_NUMBER)


def test_an_empty_object_has_no_members() raises:
    var members = _members("{}")
    assert_equal(len(members), 0)
    var spaced = _members("{   }")
    assert_equal(len(spaced), 0)


def test_a_nested_value_is_one_span_and_is_not_descended_into() raises:
    # firepanda has no nested column type yet, so a reader that flattened this
    # would be inventing a schema. The span is the right span for the day it
    # does have one.
    var text = '{"a": {"b": {"c": 1}}, "d": [1, [2, 3], 4]}'
    var members = _members(text)
    assert_equal(len(members), 2)
    assert_equal(members[0].value.kind, JSON_OBJECT)
    assert_equal(members[1].value.kind, JSON_ARRAY)
    var raw = _bytes(text)
    var nested = members[0].value
    assert_equal(
        String(
            StringSlice(unsafe_from_utf8=Span(raw)[nested.start : nested.end])
        ),
        '{"b": {"c": 1}}',
    )


def test_a_brace_inside_a_string_is_a_brace() raises:
    # The one thing a nesting counter has to understand is a string.
    var text = '{"a": {"b": "}}}}"}, "c": 1}'
    var members = _members(text)
    assert_equal(len(members), 2)
    assert_equal(members[0].value.kind, JSON_OBJECT)
    assert_equal(members[1].value.kind, JSON_NUMBER)


def test_a_repeated_key_is_reported_twice() raises:
    # Which one wins is the reader's decision. Deduplicating here would take it
    # away from the only layer that knows what the column is for.
    var members = _members('{"a": 1, "a": 2}')
    assert_equal(len(members), 2)
    assert_equal(text_of(Span(_bytes('{"a": 1, "a": 2}')), members[0].key), "a")
    assert_equal(text_of(Span(_bytes('{"a": 1, "a": 2}')), members[1].key), "a")


def test_a_string_with_no_backslash_says_it_has_none() raises:
    var members = _members('{"a": "plain", "b": "with\\ttab"}')
    assert_false(members[0].value.escaped)
    assert_true(members[1].value.escaped)


def test_the_eight_escapes_turn_into_what_they_mean() raises:
    var text = '{"a": "q\\"b\\\\s\\/f\\bl\\ff\\nr\\rt\\t"}'
    var raw = _bytes(text)
    var members = List[Member]()
    _ = scan_object(Span(raw), 0, members)
    var got = text_of(Span(raw), members[0].value)
    var want = String('q"b\\s/f', chr(8), "l", chr(12), "f\nr\rt\t")
    assert_equal(got, want)


def test_a_unicode_escape_becomes_utf8() raises:
    var text = '{"a": "\\u00e9\\u20ac"}'
    var raw = _bytes(text)
    var members = List[Member]()
    _ = scan_object(Span(raw), 0, members)
    assert_equal(text_of(Span(raw), members[0].value), "é€")


def test_a_surrogate_pair_becomes_one_character() raises:
    # Two broken halves is the bug that surfaces a long way from here.
    var text = '{"a": "\\ud83d\\ude00"}'
    var raw = _bytes(text)
    var members = List[Member]()
    _ = scan_object(Span(raw), 0, members)
    assert_equal(text_of(Span(raw), members[0].value), "😀")


def test_a_string_that_never_closes_is_an_error() raises:
    with assert_raises(contains="never closes"):
        _ = _members('{"a": "open}')


def test_an_escape_that_is_not_one_is_an_error() raises:
    with assert_raises(contains="is not an escape"):
        var raw = _bytes('{"a": "\\x"}')
        var members = List[Member]()
        _ = scan_object(Span(raw), 0, members)
        _ = text_of(Span(raw), members[0].value)


def test_a_short_unicode_escape_is_an_error() raises:
    with assert_raises(contains="hex"):
        var raw = _bytes('{"a": "\\u00"}')
        var members = List[Member]()
        _ = scan_object(Span(raw), 0, members)
        _ = text_of(Span(raw), members[0].value)


def test_a_number_with_a_leading_zero_is_an_error() raises:
    with assert_raises(contains="leading zero"):
        _ = _members('{"a": 007}')


def test_a_number_with_nothing_after_its_point_is_an_error() raises:
    with assert_raises(contains="decimal point"):
        _ = _members('{"a": 1.}')


def test_a_number_with_nothing_after_its_exponent_is_an_error() raises:
    with assert_raises(contains="exponent"):
        _ = _members('{"a": 1e}')


def test_a_missing_colon_is_an_error() raises:
    with assert_raises(contains="colon"):
        _ = _members('{"a" 1}')


def test_a_key_that_is_not_a_string_is_an_error() raises:
    with assert_raises(contains="key"):
        _ = _members("{a: 1}")


def test_a_missing_comma_is_an_error() raises:
    with assert_raises(contains="comma"):
        _ = _members('{"a": 1 "b": 2}')


def test_an_object_that_never_closes_is_an_error() raises:
    with assert_raises(contains="closes"):
        _ = _members('{"a": 1')


def test_a_nested_value_that_never_closes_is_an_error() raises:
    with assert_raises(contains="closes"):
        _ = _members('{"a": [1, 2')


def test_something_that_is_not_a_value_is_an_error() raises:
    with assert_raises(contains="does not begin a value"):
        _ = _members('{"a": nul}')


def test_scanning_a_value_on_its_own_works_too() raises:
    # The reader calls this directly on the values inside a line, so it is worth
    # one test that does not go through an object.
    var raw = _bytes("  42  ")
    var at = skip_space(Span(raw), 0)
    assert_equal(at, 2)
    var value = scan_value(Span(raw), at)
    assert_equal(value.kind, JSON_NUMBER)
    assert_equal(value.start, 2)
    assert_equal(value.end, 4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
