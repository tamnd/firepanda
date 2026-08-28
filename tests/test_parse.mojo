"""Tests for the field parsers.

Most of these are about what does not parse rather than what does. A parser that
accepts `12abc` as twelve turns a corrupt file into a clean one with wrong
numbers in it, and no later stage can detect that, so the rejections are the
part worth being thorough about.

The boundary values get their own tests because a range check written from
memory is wrong at exactly one value on exactly one side, every time.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)

from firepanda.io.parse import is_missing, parse_bool, parse_float, parse_int


def bytes_of(var text: String) -> List[UInt8]:
    """Copies a string's bytes into a list the parsers can span.

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


def test_a_plain_integer_parses() raises:
    var field = bytes_of("12345")
    var got = parse_int[DType.int64](field)
    assert_true(got.ok)
    assert_equal(got.value, Int64(12345))


def test_a_signed_integer_parses() raises:
    var minus = bytes_of("-42")
    var got = parse_int[DType.int64](minus)
    assert_true(got.ok)
    assert_equal(got.value, Int64(-42))

    var plus = bytes_of("+42")
    var also = parse_int[DType.int64](plus)
    assert_true(also.ok)
    assert_equal(also.value, Int64(42))


def test_zero_parses_in_every_spelling() raises:
    for spelling in ["0", "-0", "+0", "0000"]:
        var got = parse_int[DType.int64](bytes_of(spelling))
        assert_true(got.ok, spelling)
        assert_equal(got.value, Int64(0), spelling)


def test_an_integer_with_trailing_junk_fails() raises:
    for text in ["12abc", "12 ", " 12", "1,2", "12.0", "1_000"]:
        assert_false(parse_int[DType.int64](bytes_of(text)).ok, text)


def test_an_empty_or_sign_only_field_is_not_an_integer() raises:
    for text in ["", "-", "+"]:
        assert_false(parse_int[DType.int64](bytes_of(text)).ok, text)


def test_the_extreme_signed_values_parse() raises:
    var most = parse_int[DType.int64](bytes_of("9223372036854775807"))
    assert_true(most.ok)
    assert_equal(most.value, Int64.MAX)

    var least = parse_int[DType.int64](bytes_of("-9223372036854775808"))
    assert_true(least.ok)
    assert_equal(least.value, Int64.MIN)


def test_one_past_the_extreme_signed_values_fails() raises:
    assert_false(parse_int[DType.int64](bytes_of("9223372036854775808")).ok)
    assert_false(parse_int[DType.int64](bytes_of("-9223372036854775809")).ok)


def test_a_narrow_dtype_range_checks() raises:
    assert_true(parse_int[DType.int8](bytes_of("127")).ok)
    assert_false(parse_int[DType.int8](bytes_of("128")).ok)
    assert_equal(parse_int[DType.int8](bytes_of("-128")).value, Int8.MIN)
    assert_false(parse_int[DType.int8](bytes_of("-129")).ok)


def test_an_unsigned_dtype_refuses_a_negative() raises:
    assert_true(parse_int[DType.uint32](bytes_of("4294967295")).ok)
    assert_false(parse_int[DType.uint32](bytes_of("4294967296")).ok)
    assert_false(parse_int[DType.uint32](bytes_of("-1")).ok)
    # Negative zero is still zero, and a file that writes it means zero.
    assert_true(parse_int[DType.uint32](bytes_of("-0")).ok)


def test_a_number_too_long_for_the_accumulator_fails() raises:
    var absurd = bytes_of("123456789012345678901234567890")
    assert_false(parse_int[DType.uint64](absurd).ok)


def test_a_plain_float_parses() raises:
    var got = parse_float[DType.float64](bytes_of("3.25"))
    assert_true(got.ok)
    assert_equal(got.value, Float64(3.25))


def test_a_float_with_no_point_parses() raises:
    var got = parse_float[DType.float64](bytes_of("-17"))
    assert_true(got.ok)
    assert_equal(got.value, Float64(-17.0))


def test_a_float_with_an_exponent_parses() raises:
    assert_equal(
        parse_float[DType.float64](bytes_of("1.5e3")).value, Float64(1500.0)
    )
    assert_equal(
        parse_float[DType.float64](bytes_of("1.5E+3")).value, Float64(1500.0)
    )
    assert_equal(
        parse_float[DType.float64](bytes_of("15e-1")).value, Float64(1.5)
    )


def test_a_float_with_a_leading_or_trailing_point_parses() raises:
    assert_equal(parse_float[DType.float64](bytes_of(".5")).value, Float64(0.5))
    assert_equal(parse_float[DType.float64](bytes_of("5.")).value, Float64(5.0))


def test_a_float_with_no_digits_fails() raises:
    for text in ["", ".", "-", "e5", "-.", "+e"]:
        assert_false(parse_float[DType.float64](bytes_of(text)).ok, text)


def test_a_float_with_trailing_junk_fails() raises:
    for text in ["1.2.3", "1.2x", "1e", "1e+", "1e2e3", "1 "]:
        assert_false(parse_float[DType.float64](bytes_of(text)).ok, text)


def test_the_specials_parse() raises:
    var quiet = parse_float[DType.float64](bytes_of("nan"))
    assert_true(quiet.ok)
    assert_true(quiet.value != quiet.value)

    var big = parse_float[DType.float64](bytes_of("inf"))
    assert_true(big.ok)
    assert_equal(big.value, Float64(1.0) / Float64(0.0))

    var small = parse_float[DType.float64](bytes_of("-Infinity"))
    assert_true(small.ok)
    assert_equal(small.value, Float64(-1.0) / Float64(0.0))

    assert_true(parse_float[DType.float64](bytes_of("NaN")).ok)
    assert_true(parse_float[DType.float64](bytes_of("INF")).ok)


def test_a_float_is_exact_where_it_promises_to_be() raises:
    # A mantissa under 2^53 with an exponent inside the table is one rounding,
    # so these are equalities rather than tolerances.
    assert_equal(
        parse_float[DType.float64](bytes_of("0.1")).value, Float64(0.1)
    )
    assert_equal(
        parse_float[DType.float64](bytes_of("123456789.125")).value,
        Float64(123456789.125),
    )
    assert_equal(
        parse_float[DType.float64](bytes_of("1e22")).value, Float64(1e22)
    )
    assert_equal(
        parse_float[DType.float64](bytes_of("1e-22")).value, Float64(1e-22)
    )


def test_a_float_outside_the_exact_range_is_exact_too() raises:
    # Past the table the stepped scaling would compound a rounding per step and
    # land an ulp away, so these go to strtod instead. Equalities, because an
    # ulp is exactly what a round trip cannot afford to lose.
    assert_equal(
        parse_float[DType.float64](bytes_of("1.7976931348623157e308")).value,
        Float64(1.7976931348623157e308),
    )
    assert_equal(
        parse_float[DType.float64](bytes_of("2.2250738585072014e-308")).value,
        Float64(2.2250738585072014e-308),
    )
    assert_equal(
        parse_float[DType.float64](bytes_of("1.2345678901234567e100")).value,
        Float64(1.2345678901234567e100),
    )
    assert_equal(
        parse_float[DType.float64](bytes_of("-9.87654321e-99")).value,
        Float64(-9.87654321e-99),
    )


def test_a_float_with_more_digits_than_fit_is_still_correctly_rounded() raises:
    # Twenty five significant digits, so the accumulator truncates and the fast
    # path is refused on those grounds rather than on the exponent's.
    assert_equal(
        parse_float[DType.float64](
            bytes_of("1.234567890123456789012345")
        ).value,
        Float64(1.234567890123456789012345),
    )


def test_the_fallback_does_not_widen_what_is_accepted() raises:
    # strtod would take all four of these and the reader takes none of them. The
    # grammar is checked before the value is asked for, so the fallback answers
    # the question and does not get to change it.
    assert_false(parse_float[DType.float64](bytes_of(" 1e100")).ok)
    assert_false(parse_float[DType.float64](bytes_of("1e100 ")).ok)
    assert_false(parse_float[DType.float64](bytes_of("0x1p3")).ok)
    assert_false(parse_float[DType.float64](bytes_of("1e100abc")).ok)


def test_an_absurd_exponent_saturates() raises:
    var huge = parse_float[DType.float64](bytes_of("1e999999"))
    assert_true(huge.ok)
    assert_equal(huge.value, Float64(1.0) / Float64(0.0))

    var tiny = parse_float[DType.float64](bytes_of("1e-999999"))
    assert_true(tiny.ok)
    assert_equal(tiny.value, Float64(0.0))


def test_a_float_narrows_to_float32() raises:
    var got = parse_float[DType.float32](bytes_of("1.5"))
    assert_true(got.ok)
    assert_equal(got.value, Float32(1.5))


def test_booleans_parse_in_the_three_cases() raises:
    for text in ["true", "True", "TRUE"]:
        var got = parse_bool(bytes_of(text))
        assert_true(got.ok, text)
        assert_true(got.value, text)
    for text in ["false", "False", "FALSE"]:
        var got = parse_bool(bytes_of(text))
        assert_true(got.ok, text)
        assert_false(got.value, text)


def test_a_digit_is_not_a_boolean() raises:
    for text in ["1", "0", "t", "f", "yes", "no", ""]:
        assert_false(parse_bool(bytes_of(text)).ok, text)


def test_the_missing_sentinels() raises:
    for text in [
        "",
        "NA",
        "na",
        "N/A",
        "n/a",
        "NULL",
        "null",
        "None",
        "nan",
        "NaN",
        "-",
    ]:
        assert_true(is_missing(bytes_of(text)), text)


def test_a_value_is_not_missing() raises:
    for text in ["0", "x", "nane", "n", "--", "NAN1", "false"]:
        assert_false(is_missing(bytes_of(text)), text)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
