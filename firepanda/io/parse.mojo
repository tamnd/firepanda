"""Turning a field's bytes into a value.

These are the leaves of the CSV reader and the first kernels in document 05's
order of work, for the reason given there: ingestion dominates every first
impression benchmark, and `read_csv` is the first line of code almost everybody
writes.

Three properties shape every function here.

**Failure is a value, not an exception.** A field that is not the thing being
asked about returns `Parsed` with `ok` false. Raising per field would be correct
and would also cost more than the parse on a file where one column in six is
text, and the reader needs the answer as data anyway: type inference is exactly
the question "did this parse", asked a few thousand times.

**Nothing is guessed.** Trailing bytes after a number are a failure rather than
something to stop at, so `12abc` is not the integer 12 and `1.2.3` is not 1.2.
A parser that stops early turns a corrupt file into a clean one with wrong
numbers in it, which is the worst available outcome.

**Both paths are exact.** Integers detect overflow of the accumulator and then
range check against the target dtype, so nothing wraps silently. Floats take a
single rounding when the mantissa fits in 53 bits and the decimal exponent is
within the range where the power of ten is itself exact, which covers
essentially every number a person types into a spreadsheet and is one multiply.
Outside that range the scaling would compound a rounding per step and land an
ulp or two from the correctly rounded value, so the field goes to the platform's
`strtod` instead. That is slower and it is also the only way a value written at
seventeen digits reads back as the value it was written from. An Eisel-Lemire
implementation would replace the fallback and keep the fast path exactly as it
is.
"""

from std.collections.span import Span


comptime ZERO = UInt8(48)
"""ASCII `0`."""

comptime NINE = UInt8(57)
"""ASCII `9`."""

comptime PLUS = UInt8(43)
"""ASCII `+`."""

comptime MINUS = UInt8(45)
"""ASCII `-`."""

comptime PERIOD = UInt8(46)
"""ASCII `.`."""

comptime UPPER_E = UInt8(69)
"""ASCII `E`."""

comptime LOWER_E = UInt8(101)
"""ASCII `e`."""

comptime CASE_BIT = UInt8(32)
"""The bit that separates an ASCII letter's two cases."""

comptime EXACT_POWER = 22
"""Largest decimal exponent whose power of ten is exact in float64.

`1e22` is the last power of ten representable with no error. Multiplying a
mantissa under 2^53 by one of these, or dividing by it, is a single rounding and
therefore correctly rounded.
"""

comptime MANTISSA_LIMIT = UInt64(1) << 53
"""Largest integer above which float64 can no longer hold every value."""


@fieldwise_init
struct Parsed[dt: DType](ImplicitlyCopyable, Movable):
    """A parse result.

    Parameters:
        dt: The dtype the bytes were being read as.
    """

    var value: Scalar[Self.dt]
    """The value, or zero if the parse failed."""

    var ok: Bool
    """Whether the bytes were a complete, in range value of this dtype."""


def parse_int[dt: DType](field: Span[UInt8, _]) -> Parsed[dt]:
    """Reads a decimal integer.

    Accepts an optional leading sign followed by one or more digits, and nothing
    else. No underscores, no thousands separators, no leading or trailing space,
    because a field that needs any of those needs the caller to say so rather
    than the parser to assume it.

    A value too large for `dt` fails rather than wrapping. That includes the case
    where the digits are a perfectly good integer and merely do not fit, which is
    what makes type inference able to widen int32 to int64 by asking.

    Args:
        field: The field's bytes, already stripped of quotes.

    Returns:
        The value and whether it parsed.

    Parameters:
        dt: An integer dtype.
    """
    var n = len(field)
    if n == 0:
        return Parsed[dt](0, False)

    var ptr = field.unsafe_ptr()
    var at = 0
    var negative = False
    var lead = ptr.unsafe_load()
    if lead == PLUS or lead == MINUS:
        negative = lead == MINUS
        at = 1
    if at == n:
        return Parsed[dt](0, False)

    comptime signed = Scalar[dt].MIN < 0
    comptime positive_limit = Scalar[dt].MAX.cast[DType.uint64]()

    var acc = UInt64(0)
    while at < n:
        var c = ptr.unsafe_offset(at).unsafe_load()
        if c < ZERO or c > NINE:
            return Parsed[dt](0, False)
        var digit = UInt64(c - ZERO)
        # Checked before the multiply rather than after, because after is a
        # wrapped value that cannot be distinguished from a legitimate one.
        if acc > (UInt64.MAX - digit) // 10:
            return Parsed[dt](0, False)
        acc = acc * 10 + digit
        at += 1

    if not negative:
        if acc > positive_limit:
            return Parsed[dt](0, False)
        return Parsed[dt](acc.cast[dt](), True)

    comptime if not signed:
        # `-0` is zero and every other negative is out of range for unsigned.
        if acc != 0:
            return Parsed[dt](0, False)
        return Parsed[dt](0, True)

    if acc > positive_limit + 1:
        return Parsed[dt](0, False)
    if acc == positive_limit + 1:
        # The one asymmetric value. Negating it through the accumulator would
        # need the magnitude to fit in the signed type, and it does not.
        return Parsed[dt](Scalar[dt].MIN, True)
    return Parsed[dt](Scalar[dt](0) - acc.cast[dt](), True)


def parse_float[dt: DType](field: Span[UInt8, _]) -> Parsed[dt]:
    """Reads a decimal floating point number.

    Accepts an optional sign, digits with an optional single period, and an
    optional `e` or `E` exponent with its own optional sign. Also accepts `nan`,
    `inf` and `infinity` in any case, with a sign, because those are what a
    writer round trips a special value as.

    At least one digit is required somewhere in the significand, so `.` and `e5`
    and an empty field all fail.

    Args:
        field: The field's bytes, already stripped of quotes.

    Returns:
        The value and whether it parsed.

    Parameters:
        dt: A floating point dtype.
    """
    var n = len(field)
    if n == 0:
        return Parsed[dt](0, False)

    var ptr = field.unsafe_ptr()
    var at = 0
    var negative = False
    var lead = ptr.unsafe_load()
    if lead == PLUS or lead == MINUS:
        negative = lead == MINUS
        at = 1
    if at == n:
        return Parsed[dt](0, False)

    if _word_at(field, at, "nan"):
        var quiet = Float64(0.0) / Float64(0.0)
        return Parsed[dt](quiet.cast[dt](), True)
    if _word_at(field, at, "inf") or _word_at(field, at, "infinity"):
        var huge = Float64(1.0) / Float64(0.0)
        return Parsed[dt]((-huge if negative else huge).cast[dt](), True)

    # The significand is accumulated as an integer with a decimal exponent
    # alongside it, which is what makes a single scaling multiply possible.
    var mantissa = UInt64(0)
    var exponent = 0
    var digits = 0
    var truncated = False
    var seen_point = False

    while at < n:
        var c = ptr.unsafe_offset(at).unsafe_load()
        if c >= ZERO and c <= NINE:
            digits += 1
            if mantissa <= (UInt64.MAX - 9) // 10:
                mantissa = mantissa * 10 + UInt64(c - ZERO)
                if seen_point:
                    exponent -= 1
            else:
                # Past nineteen significant digits the rest cannot change a
                # float64 by more than half an ulp, so they are counted and
                # dropped rather than accumulated into an overflow.
                truncated = True
                if not seen_point:
                    exponent += 1
            at += 1
            continue
        if c == PERIOD and not seen_point:
            seen_point = True
            at += 1
            continue
        break

    if digits == 0:
        return Parsed[dt](0, False)

    if at < n:
        var c = ptr.unsafe_offset(at).unsafe_load()
        if c != UPPER_E and c != LOWER_E:
            return Parsed[dt](0, False)
        at += 1
        if at == n:
            return Parsed[dt](0, False)
        var exp_negative = False
        var marker = ptr.unsafe_offset(at).unsafe_load()
        if marker == PLUS or marker == MINUS:
            exp_negative = marker == MINUS
            at += 1
        if at == n:
            return Parsed[dt](0, False)
        var magnitude = 0
        while at < n:
            var d = ptr.unsafe_offset(at).unsafe_load()
            if d < ZERO or d > NINE:
                return Parsed[dt](0, False)
            # Clamped rather than checked. Anything past five digits is already
            # an overflow or an underflow by a wide margin, and the clamp keeps
            # the scaling below from wrapping on a pathological field.
            if magnitude < 100000:
                magnitude = magnitude * 10 + Int(d - ZERO)
            at += 1
        exponent += -magnitude if exp_negative else magnitude

    # The fast path is a single rounding and is therefore the correctly rounded
    # answer. Everything else is not, and the error is small but real: scaling
    # 1.2345678901234567e100 in steps of 1e22 rounds five times and lands one ulp
    # away, so a value written at seventeen digits does not read back as itself.
    # The field is handed to the platform's `strtod` in that case, which is
    # correctly rounded at every exponent and is slow enough to be worth avoiding
    # and rare enough not to matter. The grammar has already been checked here,
    # so this is being asked for a value and not for an opinion on the syntax.
    if not _scaling_is_exact(mantissa, exponent, truncated):
        try:
            return Parsed[dt](
                atof(StringSlice(unsafe_from_utf8=field)).cast[dt](), True
            )
        except:
            pass

    var value = _scale(mantissa, exponent, truncated)
    return Parsed[dt]((-value if negative else value).cast[dt](), True)


def parse_bool(field: Span[UInt8, _]) -> Parsed[DType.bool]:
    """Reads a boolean.

    Accepts `true` and `false` in lower, upper and title case, which is the set
    pandas accepts. `1` and `0` are deliberately not booleans here: a column of
    ones and zeros is an integer column, and inferring it as a boolean would be
    an irreversible narrowing decided by the data.

    Args:
        field: The field's bytes, already stripped of quotes.

    Returns:
        The value and whether it parsed.
    """
    if _word_is(field, "true"):
        return Parsed[DType.bool](True, True)
    if _word_is(field, "false"):
        return Parsed[DType.bool](False, True)
    return Parsed[DType.bool](False, False)


def is_missing(field: Span[UInt8, _]) -> Bool:
    """Returns whether a field is one of the strings that means no value.

    The set is the common part of what pandas and Polars accept: the empty
    field, `NA`, `N/A`, `NULL`, `None`, `NaN` and a bare `-`, each in the cases
    people actually write them in.

    `NaN` being missing rather than a float is pandas' rule and is the one place
    this list is a real choice. In a frame with a validity bitmap the two are
    different things, and a reader that produced a float NaN here would leave no
    way to say the value was absent. A file that means the floating point NaN
    can say `nan` and be read into a float column that was already declared.

    Args:
        field: The field's bytes, already stripped of quotes.

    Returns:
        True if the field means missing.
    """
    var n = len(field)
    if n == 0:
        return True
    if n == 1:
        return field.unsafe_ptr().unsafe_load() == MINUS
    if n == 2:
        return _word_is(field, "na")
    if n == 3:
        return (
            _word_is(field, "n/a")
            or _word_is(field, "nan")
            or _word_is(field, "nil")
        )
    if n == 4:
        return _word_is(field, "null") or _word_is(field, "none")
    return False


def _word_is(field: Span[UInt8, _], var word: String) -> Bool:
    """Returns whether a field equals an ASCII word, ignoring case.

    Args:
        field: The field's bytes.
        word: The word, in lower case.

    Returns:
        True if they match.
    """
    if len(field) != word.byte_length():
        return False
    return _word_at(field, 0, word^)


def _word_at(field: Span[UInt8, _], start: Int, var word: String) -> Bool:
    """Returns whether a field's tail from `start` is exactly an ASCII word.

    Args:
        field: The field's bytes.
        start: Where to begin.
        word: The word, in lower case.

    Returns:
        True if the rest of the field is the word, ignoring case.
    """
    if len(field) - start != word.byte_length():
        return False
    var want = word.unsafe_ptr()
    var have = field.unsafe_ptr()
    for i in range(word.byte_length()):
        # Setting the case bit folds a letter to lower case and leaves `/` and
        # the digits alone, which is all this needs to compare.
        var c = have.unsafe_offset(start + i).unsafe_load() | CASE_BIT
        if c != want.unsafe_offset(i).unsafe_load():
            return False
    return True


def _pow10(exponent: Int) -> Float64:
    """Returns ten raised to a small non-negative power, exactly.

    Binary exponentiation rather than a lookup table, because a comptime list
    cannot be indexed by a runtime value without materializing the whole thing
    on every call. Five multiplies is cheaper than that copy and the result is
    the same: every power of ten up to `1e22` is exactly representable, and each
    partial product along the way is too, so nothing rounds.

    Args:
        exponent: The power. Must be at most 22 for the result to be exact.

    Returns:
        Ten to that power.
    """
    var result = Float64(1.0)
    var base = Float64(10.0)
    var left = exponent
    while left > 0:
        if left & 1 == 1:
            result *= base
        left >>= 1
        if left > 0:
            base *= base
    return result


def _scaling_is_exact(mantissa: UInt64, exponent: Int, truncated: Bool) -> Bool:
    """Reports whether one multiply or divide gives the correctly rounded value.

    That needs the significand to be exact in a float64 and the power of ten to
    be exact in one too, which together make the scaling a single rounding.

    Args:
        mantissa: The significant digits as an integer.
        exponent: The power of ten to multiply them by.
        truncated: Whether digits were dropped from the significand.

    Returns:
        True if `_scale` is correctly rounded for these arguments.
    """
    if truncated or mantissa >= MANTISSA_LIMIT:
        return False
    return exponent >= -EXACT_POWER and exponent <= EXACT_POWER


def _scale(mantissa: UInt64, exponent: Int, truncated: Bool) -> Float64:
    """Turns a significand and a decimal exponent into a float64.

    Args:
        mantissa: The significant digits as an integer.
        exponent: The power of ten to multiply them by.
        truncated: Whether digits were dropped from the significand.

    Returns:
        The value.
    """
    if mantissa == 0:
        return 0.0

    var base = Float64(mantissa)
    # One multiply or one divide, each a single rounding, whenever the mantissa
    # is exact in float64 and the power of ten is too. This is the path almost
    # every real field takes.
    if not truncated and mantissa < MANTISSA_LIMIT:
        if exponent >= 0 and exponent <= EXACT_POWER:
            return base * _pow10(exponent)
        if exponent < 0 and -exponent <= EXACT_POWER:
            return base / _pow10(-exponent)

    var left = exponent
    var value = base
    # Anything else is scaled in exact steps of 1e22, which keeps each multiply
    # correctly rounded on its own and lets the error compound only across the
    # handful of steps a wide exponent needs.
    while left > 0:
        var step = EXACT_POWER if left > EXACT_POWER else left
        value *= _pow10(step)
        left -= step
        if value == Float64(1.0) / Float64(0.0):
            return value
    while left < 0:
        var step = EXACT_POWER if -left > EXACT_POWER else -left
        value /= _pow10(step)
        left += step
        if value == 0.0:
            return value
    return value
