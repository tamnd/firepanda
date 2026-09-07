"""Tests for the type lattice and the compile-time dtype lists.

The promotion table is the part worth being careful about. It is copied from
NumPy rather than from SQL, including the two rules that surprise people: signed
mixed with unsigned of the same width goes up a width rather than reinterpreting,
and uint64 mixed with any signed type goes to float64 and loses precision above
2^53. Both are checked against the real numpy in tests/differential/main.mojo;
what is checked here is that the answer does not depend on argument order and
that the raising cases raise.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.dtype.lists import (
    ALL,
    FLOAT,
    INTEGER,
    NUMERIC,
    SIGNED,
    UNSIGNED,
    contains,
    dtype_size,
)
from firepanda.dtype.logical import LogicalType, TypeKind, logical_for, promote
from firepanda.dtype.temporal import TimeUnit, TimeZone, unit_for_code


def test_list_membership() raises:
    assert_true(contains[SIGNED](DType.int32))
    assert_false(contains[SIGNED](DType.uint32))
    assert_true(contains[UNSIGNED](DType.uint32))
    assert_true(contains[INTEGER](DType.uint8))
    assert_false(contains[INTEGER](DType.float32))
    assert_true(contains[FLOAT](DType.float16))
    assert_true(contains[NUMERIC](DType.float64))
    assert_false(contains[NUMERIC](DType.bool))
    assert_true(contains[ALL](DType.bool))


def test_list_sizes() raises:
    # These counts are the multiplier on every dispatched kernel's code size, so
    # a change to them is a change worth noticing in review.
    assert_equal(comptime (len(SIGNED)), 4)
    assert_equal(comptime (len(UNSIGNED)), 4)
    assert_equal(comptime (len(INTEGER)), 8)
    assert_equal(comptime (len(FLOAT)), 3)
    assert_equal(comptime (len(NUMERIC)), 11)
    assert_equal(comptime (len(ALL)), 12)


def test_dtype_size() raises:
    assert_equal(dtype_size(DType.bool), 1)
    assert_equal(dtype_size(DType.int8), 1)
    assert_equal(dtype_size(DType.int16), 2)
    assert_equal(dtype_size(DType.int32), 4)
    assert_equal(dtype_size(DType.int64), 8)
    assert_equal(dtype_size(DType.uint64), 8)
    assert_equal(dtype_size(DType.float16), 2)
    assert_equal(dtype_size(DType.float32), 4)
    assert_equal(dtype_size(DType.float64), 8)


def test_logical_for() raises:
    assert_equal(logical_for(DType.bool), LogicalType.BOOL)
    assert_equal(logical_for(DType.int32), LogicalType.INT32)
    assert_equal(logical_for(DType.uint16), LogicalType.UINT16)
    assert_equal(logical_for(DType.float64), LogicalType.FLOAT64)


def test_predicates() raises:
    assert_true(LogicalType.INT64.is_numeric())
    assert_true(LogicalType.INT64.is_integer())
    assert_false(LogicalType.INT64.is_float())
    assert_true(LogicalType.INT64.is_signed())
    assert_false(LogicalType.UINT64.is_signed())
    assert_true(LogicalType.FLOAT32.is_float())
    assert_false(LogicalType.BOOL.is_numeric())
    assert_true(LogicalType.STRING.is_variable_width())
    assert_false(LogicalType.INT8.is_variable_width())


def test_bit_width() raises:
    assert_equal(LogicalType.INT8.bit_width(), 8)
    assert_equal(LogicalType.INT16.bit_width(), 16)
    assert_equal(LogicalType.UINT32.bit_width(), 32)
    assert_equal(LogicalType.FLOAT64.bit_width(), 64)


def test_promote_identical() raises:
    assert_equal(
        promote(LogicalType.INT32, LogicalType.INT32), LogicalType.INT32
    )
    assert_equal(
        promote(LogicalType.STRING, LogicalType.STRING), LogicalType.STRING
    )


def test_promote_null() raises:
    assert_equal(
        promote(LogicalType.NULL, LogicalType.INT16), LogicalType.INT16
    )
    assert_equal(
        promote(LogicalType.STRING, LogicalType.NULL), LogicalType.STRING
    )


def test_promote_same_signedness_takes_the_wider() raises:
    assert_equal(
        promote(LogicalType.INT8, LogicalType.INT64), LogicalType.INT64
    )
    assert_equal(
        promote(LogicalType.UINT16, LogicalType.UINT8), LogicalType.UINT16
    )


def test_promote_mixed_signedness_goes_up_a_width() raises:
    assert_equal(
        promote(LogicalType.UINT8, LogicalType.INT8), LogicalType.INT16
    )
    assert_equal(
        promote(LogicalType.UINT16, LogicalType.INT8), LogicalType.INT32
    )
    assert_equal(
        promote(LogicalType.UINT32, LogicalType.INT16), LogicalType.INT64
    )
    assert_equal(
        promote(LogicalType.UINT8, LogicalType.INT64), LogicalType.INT64
    )


def test_promote_uint64_with_signed_goes_to_float64() raises:
    assert_equal(
        promote(LogicalType.UINT64, LogicalType.INT8), LogicalType.FLOAT64
    )
    assert_equal(
        promote(LogicalType.UINT64, LogicalType.INT64), LogicalType.FLOAT64
    )


def test_promote_float_beats_integer() raises:
    assert_equal(
        promote(LogicalType.INT8, LogicalType.FLOAT16), LogicalType.FLOAT16
    )
    assert_equal(
        promote(LogicalType.INT16, LogicalType.FLOAT16), LogicalType.FLOAT32
    )
    assert_equal(
        promote(LogicalType.INT32, LogicalType.FLOAT16), LogicalType.FLOAT64
    )
    assert_equal(
        promote(LogicalType.INT16, LogicalType.FLOAT32), LogicalType.FLOAT32
    )
    assert_equal(
        promote(LogicalType.INT32, LogicalType.FLOAT32), LogicalType.FLOAT64
    )
    assert_equal(
        promote(LogicalType.INT64, LogicalType.FLOAT32), LogicalType.FLOAT64
    )
    assert_equal(
        promote(LogicalType.FLOAT32, LogicalType.FLOAT64), LogicalType.FLOAT64
    )


def test_promote_bool_with_numeric() raises:
    assert_equal(promote(LogicalType.BOOL, LogicalType.INT8), LogicalType.INT8)
    assert_equal(
        promote(LogicalType.FLOAT32, LogicalType.BOOL), LogicalType.FLOAT32
    )


def test_promote_is_symmetric() raises:
    var types = [
        LogicalType.BOOL,
        LogicalType.INT8,
        LogicalType.INT16,
        LogicalType.INT32,
        LogicalType.INT64,
        LogicalType.UINT8,
        LogicalType.UINT16,
        LogicalType.UINT32,
        LogicalType.UINT64,
        LogicalType.FLOAT16,
        LogicalType.FLOAT32,
        LogicalType.FLOAT64,
    ]
    for i in range(len(types)):
        for j in range(len(types)):
            assert_equal(
                promote(types[i], types[j]), promote(types[j], types[i])
            )


def test_promote_string_with_number_raises() raises:
    with assert_raises(contains="no common type"):
        _ = promote(LogicalType.STRING, LogicalType.INT64)


def test_type_kind_equality() raises:
    assert_true(TypeKind.INT == TypeKind.INT)
    assert_true(TypeKind.INT != TypeKind.FLOAT_KIND)
    assert_true(TypeKind.DICTIONARY != TypeKind.INT)
    # The last two kinds, and the ones a new kind added without a branch in
    # `write_to` would silently be printed as.
    assert_equal(String(TypeKind.DURATION), "duration")
    assert_equal(String(TypeKind.DICTIONARY), "dictionary")


def test_types_print() raises:
    assert_equal(String(LogicalType.INT64), "int64")
    assert_equal(String(LogicalType.STRING), "string")
    assert_equal(String(LogicalType.NULL), "null")


def test_a_timestamp_prints_the_way_pandas_spells_its_dtype() raises:
    # This string is what `dtype` hands a user, and a user comparing it is
    # comparing it against pandas, so the spelling is pandas' and not Arrow's.
    assert_equal(
        String(LogicalType.timestamp(TimeUnit.SECOND)), "datetime64[s]"
    )
    assert_equal(
        String(LogicalType.timestamp(TimeUnit.MILLI)), "datetime64[ms]"
    )
    assert_equal(
        String(LogicalType.timestamp(TimeUnit.MICRO)), "datetime64[us]"
    )
    assert_equal(String(LogicalType.timestamp(TimeUnit.NANO)), "datetime64[ns]")
    assert_equal(
        String(
            LogicalType.timestamp(
                TimeUnit.MICRO, TimeZone("Australia/Lord_Howe")
            )
        ),
        "datetime64[us, Australia/Lord_Howe]",
    )
    assert_equal(String(LogicalType.DATE32), "date32[day]")


def test_two_timestamps_differing_only_in_unit_are_not_the_same_type() raises:
    # The buffers are identical and the meanings are a thousand apart, so this
    # is the comparison that stops a microsecond column being handed to a kernel
    # that was told it had nanoseconds.
    var micro = LogicalType.timestamp(TimeUnit.MICRO)
    var nano = LogicalType.timestamp(TimeUnit.NANO)
    assert_true(micro != nano)
    assert_true(micro == LogicalType.timestamp(TimeUnit.MICRO))


def test_a_naive_timestamp_is_not_a_zoned_one() raises:
    # A naive column has no answer to which wall clock it is on, which is a
    # different statement from UTC and has to compare unequal to every zone.
    var naive = LogicalType.timestamp(TimeUnit.MICRO)
    var utc = LogicalType.timestamp(TimeUnit.MICRO, TimeZone("UTC"))
    var york = LogicalType.timestamp(
        TimeUnit.MICRO, TimeZone("America/New_York")
    )
    assert_true(naive != utc)
    assert_true(utc != york)
    assert_true(naive.zone.is_naive())
    assert_false(utc.zone.is_naive())


def test_the_longest_zone_name_there_is_fits_and_a_longer_one_does_not() raises:
    # Thirty two characters, which is the longest name in the IANA database, so
    # the capacity is a fact about the world rather than a guess about it.
    var longest = "America/Argentina/ComodRivadavia"
    assert_equal(String(TimeZone(longest)), longest)
    with assert_raises(contains="no zone name is longer"):
        _ = TimeZone("America/Argentina/ComodRivadavia_and_then_some_more")


def test_a_temporal_type_promotes_with_nothing_but_itself() raises:
    var micro = LogicalType.timestamp(TimeUnit.MICRO)
    assert_true(promote(micro, micro) == micro)
    with assert_raises(contains="not the same kind of thing"):
        _ = promote(micro, LogicalType.INT64)
    with assert_raises(contains="differ in kind, in unit or in time zone"):
        _ = promote(micro, LogicalType.timestamp(TimeUnit.NANO))
    with assert_raises(contains="differ in kind, in unit or in time zone"):
        _ = promote(micro, LogicalType.DATE32)
    # A duration against an instant of the same unit is the mixture that looks
    # most like it should work and is the one pandas answers with a timestamp,
    # so it is worth pinning that firepanda refuses it rather than promoting
    # one to the other on the strength of both being an int64.
    with assert_raises(contains="differ in kind, in unit or in time zone"):
        _ = promote(micro, LogicalType.duration(TimeUnit.MICRO))
    with assert_raises(contains="scales an elapsed time by a number"):
        _ = promote(LogicalType.duration(TimeUnit.MICRO), LogicalType.INT64)


def test_a_temporal_type_promotes_with_null_the_way_everything_does() raises:
    # The null type is the one exception, because a column of nothing but nulls
    # has no meaning to disagree with.
    var micro = LogicalType.timestamp(TimeUnit.MICRO)
    assert_true(promote(micro, LogicalType.NULL) == micro)
    assert_true(promote(LogicalType.NULL, micro) == micro)


def test_a_temporal_type_is_not_numeric_even_though_it_holds_integers() raises:
    var micro = LogicalType.timestamp(TimeUnit.MICRO)
    assert_true(micro.is_temporal())
    assert_true(LogicalType.DATE32.is_temporal())
    assert_true(LogicalType.duration(TimeUnit.MICRO).is_temporal())
    assert_false(micro.is_numeric())
    assert_false(micro.is_integer())
    assert_false(LogicalType.INT64.is_temporal())
    assert_equal(micro.bit_width(), 64)
    assert_equal(LogicalType.DATE32.bit_width(), 32)


def test_a_dictionary_is_spelled_the_way_pandas_spells_a_categorical() raises:
    var label = LogicalType.dictionary(DType.int32)
    assert_equal(String(label), "category")
    # The index width and the categories are both absent from the spelling,
    # which is what pandas prints too, so two categorical columns holding
    # nothing in common print the same dtype in both libraries.
    assert_equal(String(LogicalType.dictionary(DType.int8, True)), "category")
    assert_true(label.is_dictionary())
    assert_false(label.is_numeric())
    assert_false(label.is_integer())
    assert_false(label.is_temporal())
    assert_false(label.is_variable_width())
    # The physical dtype is the index and is a perfectly ordinary int32, which
    # is exactly why nothing is allowed to reach the buffer through it.
    assert_equal(label.physical, DType.int32)
    assert_equal(label.bit_width(), 32)
    assert_false(LogicalType.INT32.is_dictionary())


def test_two_dictionaries_differing_in_index_or_in_order_are_not_one_type() raises:
    assert_true(
        LogicalType.dictionary(DType.int32)
        != LogicalType.dictionary(DType.int8)
    )
    assert_true(
        LogicalType.dictionary(DType.int32, True)
        != LogicalType.dictionary(DType.int32, False)
    )
    assert_true(
        LogicalType.dictionary(DType.int32)
        == LogicalType.dictionary(DType.int32)
    )
    # And the one that is not a type question at all. Two columns over different
    # categories carry the same type, because the categories are held by the
    # column, so anything that needs to know has to ask the arrays.
    assert_true(LogicalType.dictionary(DType.int32) != LogicalType.INT32)


def test_a_dictionary_promotes_with_nothing_including_itself() raises:
    var label = LogicalType.dictionary(DType.int32)
    # Itself included, and that is the unusual one. Every other type in this
    # file promotes with a copy of itself, and this refuses because two equal
    # dictionary types say nothing about whether the two columns share their
    # categories, which is the question pandas actually answers.
    with assert_raises(contains="held by the column rather than by the type"):
        _ = promote(label, label)
    with assert_raises(contains="held by the column rather than by the type"):
        _ = promote(label, LogicalType.INT32)
    with assert_raises(contains="held by the column rather than by the type"):
        _ = promote(LogicalType.STRING, label)
    with assert_raises(contains="held by the column rather than by the type"):
        _ = promote(label, LogicalType.NULL)


def test_a_bare_int64_array_is_never_a_timestamp() raises:
    # `logical_for` is what an array built from raw values gets, and eight bytes
    # of integer is a number until somebody says otherwise.
    assert_true(logical_for(DType.int64) == LogicalType.INT64)
    assert_true(logical_for(DType.int32) == LogicalType.INT32)


def test_a_time_unit_knows_how_many_of_it_make_a_second() raises:
    assert_equal(TimeUnit.SECOND.per_second(), Int64(1))
    assert_equal(TimeUnit.MILLI.per_second(), Int64(1_000))
    assert_equal(TimeUnit.MICRO.per_second(), Int64(1_000_000))
    assert_equal(TimeUnit.NANO.per_second(), Int64(1_000_000_000))


def test_a_time_unit_is_spelled_one_way_in_a_dtype_and_another_in_a_format() raises:
    # `ms` and `m` are the same unit, and using either spelling where the other
    # belongs produces a format string nothing can read.
    assert_equal(String(TimeUnit.MILLI), "ms")
    assert_equal(TimeUnit.MILLI.code_letter(), "m")
    assert_equal(String(TimeUnit.SECOND), "s")
    assert_equal(TimeUnit.SECOND.code_letter(), "s")


def test_an_unknown_time_unit_code_is_refused() raises:
    assert_true(unit_for_code(3) == TimeUnit.NANO)
    with assert_raises(contains="is not one of the four"):
        _ = unit_for_code(4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
