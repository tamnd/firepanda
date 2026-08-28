"""Tests for the CSV field scanner.

The interesting cases are all about where a field ends: quotes containing the
delimiter, quotes containing a newline, a doubled quote, a trailing delimiter,
a line ending that is two bytes, and a file that stops without one. Those are
the six ways a CSV reader is usually wrong.

Every test also runs the same input through the byte at a time twin and asserts
the two agree, so a SIMD block boundary landing in the middle of a field cannot
pass unnoticed. That is what the long line test is for: it puts the boundaries
somewhere other than where a short example puts them.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.io.scalar import scan_csv_scalar
from firepanda.io.scan import (
    Dialect,
    Scan,
    default_dialect,
    scan_csv,
    unescape,
)
from firepanda.testing.rng import Rng


def bytes_of(var text: String) -> List[UInt8]:
    """Copies a string's bytes into a list the scanner can span.

    Args:
        text: The string.

    Returns:
        The bytes, without a terminator.
    """
    var out = List[UInt8](capacity=text.byte_length())
    var ptr = text.unsafe_ptr()
    for i in range(text.byte_length()):
        out.append(ptr.unsafe_offset(i).unsafe_load())
    return out^


def text_at(data: List[UInt8], scan: Scan, row: Int, column: Int) -> String:
    """Returns one field's content as a string, unescaped if it needs it.

    Args:
        data: The buffer that was scanned.
        scan: The scan.
        row: The row number.
        column: The field's position within the row.

    Returns:
        The field's literal text.
    """
    var span = scan.at(row, column)
    var out = String()
    if span.escaped:
        var literal = unescape(data, span, default_dialect().quote)
        for i in range(len(literal)):
            out += chr(Int(literal[i]))
        return out^
    for i in range(span.start, span.end):
        out += chr(Int(data[i]))
    return out^


def check_twins(data: List[UInt8]) raises:
    """Asserts the SIMD scan and the byte at a time scan agree exactly.

    Args:
        data: The buffer to scan.

    Raises:
        If either scan raises or the two disagree.
    """
    var fast = scan_csv(data, default_dialect())
    var slow = scan_csv_scalar(data, default_dialect())
    assert_equal(len(fast), len(slow), "row count")
    for r in range(len(fast)):
        assert_equal(fast.width(r), slow.width(r), "width of row " + String(r))
        for c in range(fast.width(r)):
            var at_field = "row " + String(r) + " field " + String(c)
            assert_equal(fast.at(r, c).start, slow.at(r, c).start, at_field)
            assert_equal(fast.at(r, c).end, slow.at(r, c).end, at_field)
            assert_equal(fast.at(r, c).escaped, slow.at(r, c).escaped, at_field)


def test_a_plain_file_splits_into_rows_and_fields() raises:
    var data = bytes_of("a,b,c\n1,2,3\n")
    var scan = scan_csv(data, default_dialect())
    assert_equal(len(scan), 2)
    assert_equal(scan.width(0), 3)
    assert_equal(text_at(data, scan, 0, 1), "b")
    assert_equal(text_at(data, scan, 1, 2), "3")
    check_twins(data)


def test_a_file_with_no_trailing_newline_is_the_same() raises:
    var with_it = bytes_of("a,b\n1,2\n")
    var without = bytes_of("a,b\n1,2")
    assert_equal(
        len(scan_csv(with_it, default_dialect())),
        len(scan_csv(without, default_dialect())),
    )
    check_twins(without)


def test_carriage_returns_are_not_data() raises:
    var data = bytes_of("a,b\r\n1,2\r\n")
    var scan = scan_csv(data, default_dialect())
    assert_equal(len(scan), 2)
    assert_equal(text_at(data, scan, 0, 1), "b")
    assert_equal(text_at(data, scan, 1, 1), "2")
    check_twins(data)


def test_blank_lines_are_skipped() raises:
    var data = bytes_of("a,b\n\n\n1,2\n\n")
    var scan = scan_csv(data, default_dialect())
    assert_equal(len(scan), 2)
    check_twins(data)


def test_an_empty_buffer_has_no_rows() raises:
    var data = bytes_of("")
    assert_equal(len(scan_csv(data, default_dialect())), 0)
    check_twins(data)


def test_empty_fields_are_fields() raises:
    var data = bytes_of("a,,c\n,,\n")
    var scan = scan_csv(data, default_dialect())
    assert_equal(scan.width(0), 3)
    assert_equal(scan.width(1), 3)
    assert_equal(text_at(data, scan, 0, 1), "")
    assert_equal(text_at(data, scan, 1, 2), "")
    check_twins(data)


def test_a_trailing_delimiter_opens_one_more_field() raises:
    var data = bytes_of("a,b,\n")
    var scan = scan_csv(data, default_dialect())
    assert_equal(scan.width(0), 3)
    assert_equal(text_at(data, scan, 0, 2), "")
    check_twins(data)


def test_a_single_column_file_scans() raises:
    var data = bytes_of("x\ny\nz\n")
    var scan = scan_csv(data, default_dialect())
    assert_equal(len(scan), 3)
    assert_equal(scan.width(1), 1)
    assert_equal(text_at(data, scan, 2, 0), "z")
    check_twins(data)


def test_a_quoted_field_loses_its_quotes() raises:
    var data = bytes_of('"a",b\n')
    var scan = scan_csv(data, default_dialect())
    assert_equal(text_at(data, scan, 0, 0), "a")
    assert_false(scan.at(0, 0).escaped)
    check_twins(data)


def test_a_quoted_field_can_hold_the_delimiter() raises:
    var data = bytes_of('"a,b",c\n')
    var scan = scan_csv(data, default_dialect())
    assert_equal(scan.width(0), 2)
    assert_equal(text_at(data, scan, 0, 0), "a,b")
    check_twins(data)


def test_a_quoted_field_can_hold_a_newline() raises:
    var data = bytes_of('"a\nb",c\nd,e\n')
    var scan = scan_csv(data, default_dialect())
    assert_equal(len(scan), 2)
    assert_equal(text_at(data, scan, 0, 0), "a\nb")
    assert_equal(text_at(data, scan, 1, 0), "d")
    check_twins(data)


def test_a_doubled_quote_is_one_quote() raises:
    var data = bytes_of('"say ""hi""",b\n')
    var scan = scan_csv(data, default_dialect())
    assert_true(scan.at(0, 0).escaped)
    assert_equal(text_at(data, scan, 0, 0), 'say "hi"')
    assert_equal(text_at(data, scan, 0, 1), "b")
    check_twins(data)


def test_an_empty_quoted_field_is_empty() raises:
    var data = bytes_of('"",""\n')
    var scan = scan_csv(data, default_dialect())
    assert_equal(scan.width(0), 2)
    assert_equal(text_at(data, scan, 0, 0), "")
    check_twins(data)


def test_a_quoted_field_at_the_end_of_the_buffer_closes() raises:
    var data = bytes_of('a,"b"')
    var scan = scan_csv(data, default_dialect())
    assert_equal(scan.width(0), 2)
    assert_equal(text_at(data, scan, 0, 1), "b")
    check_twins(data)


def test_an_unterminated_quote_is_refused() raises:
    var raised = False
    try:
        _ = scan_csv(bytes_of('a,"b\nc,d\n'), default_dialect())
    except:
        raised = True
    assert_true(raised, "the quote never closes")

    raised = False
    try:
        _ = scan_csv_scalar(bytes_of('a,"b\nc,d\n'), default_dialect())
    except:
        raised = True
    assert_true(raised, "the twin agrees")


def test_a_value_after_a_closing_quote_is_refused() raises:
    var raised = False
    try:
        _ = scan_csv(bytes_of('"a"b,c\n'), default_dialect())
    except:
        raised = True
    assert_true(raised, "junk after the closing quote")

    raised = False
    try:
        _ = scan_csv_scalar(bytes_of('"a"b,c\n'), default_dialect())
    except:
        raised = True
    assert_true(raised, "the twin agrees")


def test_ragged_rows_are_reported_not_refused() raises:
    var even = bytes_of("a,b\nc,d\n")
    assert_false(scan_csv(even, default_dialect()).is_ragged())

    var odd = bytes_of("a,b\nc\n")
    assert_true(scan_csv(odd, default_dialect()).is_ragged())
    check_twins(odd)


def test_a_field_longer_than_a_simd_register() raises:
    # The block loop skips whole registers, so a field has to be long enough to
    # contain one before the skipping is exercised at all.
    var wide = String()
    for _ in range(200):
        wide += "x"
    var data = bytes_of(wide + ",b\n" + wide + ",c\n")
    var scan = scan_csv(data, default_dialect())
    assert_equal(len(scan), 2)
    assert_equal(scan.at(0, 0).end - scan.at(0, 0).start, 200)
    assert_equal(text_at(data, scan, 1, 1), "c")
    check_twins(data)


def test_a_delimiter_at_every_offset_in_a_register() raises:
    # Walks the boundary across a whole register width and then some, which is
    # where an off by one in the block loop shows up.
    for pad in range(1, 80):
        var line = String()
        for _ in range(pad):
            line += "x"
        var data = bytes_of(line + ",b\n")
        var scan = scan_csv(data, default_dialect())
        assert_equal(scan.width(0), 2, "pad " + String(pad))
        assert_equal(
            scan.at(0, 0).end - scan.at(0, 0).start, pad, "pad " + String(pad)
        )
        check_twins(data)


def test_a_custom_dialect() raises:
    var pipe = Dialect(UInt8(124), UInt8(39))
    var data = bytes_of("a|'b|c'|d\n")
    var scan = scan_csv(data, pipe)
    assert_equal(scan.width(0), 3)
    assert_equal(scan.at(0, 1).end - scan.at(0, 1).start, 3)


def test_the_two_scanners_agree_on_random_input() raises:
    # The alphabet is deliberately loaded with the bytes that mean something, so
    # a random buffer is mostly boundaries rather than mostly text.
    var rng = Rng(0xC5C5C5)
    for trial in range(400):
        var text = String()
        var length = rng.next_below(120)
        var quoted = False
        for _ in range(length):
            var pick = rng.next_below(10)
            if pick == 0:
                text += ","
            elif pick == 1:
                text += "\n"
            elif pick == 2:
                if quoted:
                    text += '"'
                    quoted = False
                else:
                    text += '"'
                    quoted = True
            elif pick == 3:
                text += "\r"
            else:
                text += chr(97 + Int(rng.next_below(26)))
        if quoted:
            text += '"'
        var data = bytes_of(text)
        var fast_failed = False
        var slow_failed = False
        try:
            _ = scan_csv(data, default_dialect())
        except:
            fast_failed = True
        try:
            _ = scan_csv_scalar(data, default_dialect())
        except:
            slow_failed = True
        assert_equal(fast_failed, slow_failed, "trial " + String(trial))
        if not fast_failed:
            check_twins(data)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
