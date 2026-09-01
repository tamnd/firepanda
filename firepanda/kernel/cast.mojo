"""Converting a column from one dtype to another.

This is the shortest kernel in the package and the reason is the null-is-zero
invariant. A null holds a zero, zero converts to zero in every direction, so the
loop can run straight over the values buffer and the validity bitmap comes across
unchanged with no repair pass afterwards.

What this does not do is range checking. Casting 300 to int8 gives whatever the
hardware gives, which is 44, and casting a NaN to an integer is undefined in the
same way it is in C. pandas raises on the first and warns on the second. Deciding
which of those firepanda should copy is a question for the milestone that builds
the user-facing cast, and when it is decided the check goes in the layer above
this one. The kernel stays the raw conversion.

A cast that crosses between text and a number is not that loop and is not free.
Text to number reads bytes and can fail, number to text allocates and cannot.
They are `cast_strings_to` and `cast_to_strings` below, and the reason they take
a `strict` flag while the numeric loop does not is that a number always converts
to some number, correctly or not, and a string does not always convert to
anything at all.

`cast_any` is the erased entry point a `DataFrame` calls, and it is the most
expensive function in the package to compile: it dispatches on both ends, so it
instantiates the loop once per ordered pair of the twelve physical dtypes, which
is a hundred and forty four copies. `tools/probes/cast_matrix.mojo` measures what
that is worth in a binary and the answer is 72 KB and 1.3 seconds of compile time
over a probe that links the package and dispatches over nothing, which is about
500 bytes per pair. It is affordable because the loop being copied is nine lines
with no branch in it, and the alternative, routing every cast through a common
widest type, would silently lose precision on exactly the int64 and uint64 round
trips a join key most needs to survive.
"""

from std.sys.info import simd_width_of

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL, INTEGER, contains
from firepanda.dtype.logical import LogicalType, TypeKind

# `firepanda.io.parse` imports nothing from firepanda. It is text to scalar and
# nothing else, and it lives under io because that is where it was needed first,
# not because it belongs to a file reader. A cast from text asks the same
# question and gets the same answer, rather than a second one that rounds
# differently from the reader.
from firepanda.io.parse import parse_bool, parse_float, parse_int


def cast_to[src: DType, dst: DType](col: Array[src]) -> Array[dst]:
    """Converts a column to another dtype.

    Args:
        col: The column to convert.

    Parameters:
        src: The input dtype.
        dst: The output dtype.

    Returns:
        A column of the target dtype, null in the same places as the input.
    """
    # The narrower of the two register widths. int8 to int64 reads four bytes and
    # writes a full register; int64 to int8 does the reverse. Stepping by the
    # smaller lane count keeps both sides inside one register per iteration,
    # which is what the buffer padding guarantees is in bounds.
    comptime width = min(simd_width_of[src](), simd_width_of[dst]())

    var n = len(col)
    var out = Array[dst](n)
    var source = col.unsafe_ptr()
    var target = out.unsafe_ptr()

    var i = 0
    while i < n:
        target.unsafe_offset(i).unsafe_store(
            source.unsafe_offset(i).unsafe_load[width=width]().cast[dst]()
        )
        i += width

    out.data.validity = Bitmap(copy=col.data.validity)
    return out^


def cast_strings_to[
    dst: DType
](col: StringArray, strict: Bool) raises -> Array[dst]:
    """Reads a text column as numbers.

    A null stays null. Everything else has to parse, in the sense the CSV reader
    means: an optional sign and digits for an integer, the usual forms plus the
    named infinities for a float, and one of a short list of words for a bool. No
    surrounding space, no thousands separators, no currency.

    What happens to a value that does not parse is the caller's to decide, and
    both answers are defensible, so both are here. Strict raises and names the
    row and the value, which is what someone converting a column they believe is
    numeric wants: a column half converted is worse than one not converted.
    Non-strict writes a null, which is what someone cleaning a column scraped out
    of a spreadsheet wants, and matches `to_numeric(errors="coerce")`.

    The empty string is not a number and is not special-cased into one. It is a
    value the column holds, distinct from the null beside it, and coercing it
    silently would throw away the distinction the string column went to some
    trouble to keep.

    Args:
        col: The text column.
        strict: Whether a value that does not parse raises rather than nulls.

    Parameters:
        dst: The target dtype.

    Returns:
        A column of `dst`, null where the input was null and, if not strict,
        also where a value did not parse.

    Raises:
        Error: If strict and some value does not parse.
    """
    var n = len(col)
    var out = Array[dst](n)
    for i in range(n):
        if not col.validity.get(i):
            out.set_null(i)
            continue
        var bytes = col.unsafe_bytes(i)

        comptime if dst == DType.bool:
            var read = parse_bool(bytes)
            if read.ok:
                # `dst` is bool inside this branch and the cast is a no-op, but
                # the parser is not parameterised on it and the compiler will not
                # take the branch condition as proof.
                out.set_valid(i, read.value.cast[dst]())
                continue
        elif contains[INTEGER](dst):
            var read = parse_int[dst](bytes)
            if read.ok:
                out.set_valid(i, read.value)
                continue
        else:
            var read = parse_float[dst](bytes)
            if read.ok:
                out.set_valid(i, read.value)
                continue

        if strict:
            raise Error(
                String(
                    "cast: row ",
                    i,
                    " holds ",
                    col[i],
                    ", which is not a ",
                    dst,
                )
            )
        out.set_null(i)
    return out^


def cast_to_strings[src: DType](col: Array[src]) raises -> StringArray:
    """Writes a number column as text.

    A null stays null, and is not the empty string. Every other value is spelled
    the way the CSV writer spells it, which for a float means enough digits to
    read back as the same float rather than the rounded form the display layer
    shows. A column that survives `cast(STRING).cast(FLOAT64)` unchanged is the
    property worth having, and it is not the property a person reading a screen
    wants, so the two spellings stay separate.

    Args:
        col: The number column.

    Parameters:
        src: The source dtype.

    Returns:
        A text column, null in the same places.

    Raises:
        Error: If the builder cannot grow.
    """
    var n = len(col)
    var builder = StringBuilder(n)
    for i in range(n):
        if not col.is_valid(i):
            builder.append_null()
            continue
        var value = col.unsafe_ptr().unsafe_offset(i).unsafe_load()

        comptime if src == DType.bool:
            var text = String("true") if value else String("false")
            builder.append(text.as_bytes())
        else:
            var text = String(value)
            builder.append(text.as_bytes())
    return builder^.finish()


def cast_any(col: AnyArray, to: DType, strict: Bool = True) raises -> AnyArray:
    """Converts a column whose dtype is a runtime value to another runtime dtype.

    A cast to the dtype the column already has still copies. Returning the input
    would be faster and would make the result share a buffer with something the
    caller still holds, which is the one thing an eager API cannot afford.

    Args:
        col: The column to convert.
        to: The target dtype.
        strict: Whether a text value that is not a number raises rather than
            becoming a null. Ignored when the column is not text, because a
            number always converts to some number.

    Returns:
        A column of dtype `to`, null in the same places as the input.

    Raises:
        If either dtype is not one firepanda has a physical layout for, or the
        column is text holding a value that is not a number.
    """
    # A string column must not fall through to the number path: its physical
    # dtype is uint8, so `_cast_erased` would find the uint8 source arm and
    # convert the first byte of every 16 byte view.
    if col.is_string():
        comptime for target in ALL:
            if to == target:
                return AnyArray(cast_strings_to[target](col.strings(), strict))
        raise Error("cast: unsupported target dtype")
    comptime for target in ALL:
        if to == target:
            return AnyArray(_cast_erased[target](col))
    raise Error("cast: unsupported target dtype")


def cast_any(
    col: AnyArray, to: LogicalType, strict: Bool = True
) raises -> AnyArray:
    """Converts a column to a logical type, which text is one of.

    The overload above takes a `DType` and so cannot name text at all, text
    having no dtype of its own. This one can, and it is what a frame calls when
    the user wrote the type rather than the layout.

    A cast to the type the column already has still copies, for the reason the
    numeric one does: returning the input would share a buffer with something the
    caller still holds.

    Args:
        col: The column to convert.
        to: The target type.
        strict: Whether a text value that is not a number raises rather than
            becoming a null.

    Returns:
        A column of type `to`, null where the input was null.

    Raises:
        Error: If the type has no conversion from this column's type, or the
            column is text holding a value that is not a number.
    """
    if to.kind == TypeKind.STRING or to.kind == TypeKind.BINARY:
        if col.is_string():
            return AnyArray(StringArray(copy=col.strings()))
        comptime for source in ALL:
            if col.dtype() == source:
                ref view = col.as_typed_view[source]()
                return AnyArray(cast_to_strings(view))
        raise Error("cast: unsupported source dtype")
    if to.kind == TypeKind.NULL:
        raise Error("cast: nothing converts to the null type")
    return cast_any(col, to.physical, strict)


def _cast_erased[dst: DType](col: AnyArray) raises -> Array[dst]:
    """Resolves the source dtype, the target having already been resolved."""
    comptime for source in ALL:
        if col.dtype() == source:
            comptime width = min(simd_width_of[source](), simd_width_of[dst]())
            var n = len(col)
            var out = Array[dst](n)
            var values = col.unsafe_ptr[source]()
            var target = out.unsafe_ptr()

            var i = 0
            while i < n:
                target.unsafe_offset(i).unsafe_store(
                    values.unsafe_offset(i)
                    .unsafe_load[width=width]()
                    .cast[dst]()
                )
                i += width

            out.data.validity = Bitmap(copy=col.data.validity)
            return out^
    raise Error("cast: unsupported source dtype")
