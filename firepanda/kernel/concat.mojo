"""Stacking columns end to end.

`concat` is the one operation in this package whose cost is entirely memory
traffic. There is no arithmetic in it and no branch that depends on a value, so
the only thing worth designing is how few passes it makes over the bytes and how
much of the validity work it can skip.

It skips a lot. A fresh `Array` is zeroed and marked present everywhere, so a
part with no nulls needs no validity work at all: the bits it wants are already
there. On the columns most people actually concatenate, that is the whole bitmap
loop gone, and what remains is a vectorized copy of the values.

The parts that do have nulls go through `Bitmap.paste`, which copies a word at a
time and shifts across the boundary when the destination offset is not a multiple
of sixty four, which it almost never is.

A string column is copied the same way as a fixed width one: the views and the
payload of a part are each one memcpy, and the only per element work is adding
the part's payload base to the offset field of the views that have one. That
matters because it is the whole of `read_csv`. A ten million row file is read on
every core and the per core pieces are stacked here, and stacking them element by
element through a builder was three quarters of a second on a file Polars reads
in fifty milliseconds.

Dtypes must match exactly. Promoting them here would mean stacking an int32
column onto a float64 one and silently changing values on the way, and the cast
is cheap to write and belongs where a reader can see it.
"""

from std.memory import unsafe_memcpy
from std.sys.info import simd_width_of

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray
from firepanda.array.strview import StringView, VIEW_SIZE
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.lists import ALL


def concat_arrays[dt: DType](parts: List[Array[dt]]) -> Array[dt]:
    """Stacks columns end to end, in the order given.

    Args:
        parts: The columns. An empty list gives an empty column, which is the
            answer a caller folding over a list of frames wants.

    Parameters:
        dt: The dtype.

    Returns:
        A column as tall as the parts put together.
    """
    var total = 0
    for p in range(len(parts)):
        total += len(parts[p])

    var out = Array[dt](total)
    var at = 0
    for p in range(len(parts)):
        _copy_into(out, at, parts[p].unsafe_ptr(), parts[p].data.validity)
        at += len(parts[p])
    return out^


def concat_any(parts: List[AnyArray]) raises -> AnyArray:
    """Stacks columns whose dtype is a runtime value.

    Args:
        parts: The columns. All of them must have the same dtype.

    Returns:
        A column as tall as the parts put together.

    Raises:
        If the list is empty, if two parts disagree on dtype, or if the dtype
        has no physical layout.
    """
    if len(parts) == 0:
        raise Error("concat: at least one column is required")

    var dt = parts[0].dtype()
    var total = 0
    for p in range(len(parts)):
        # The `is_string` half is not redundant. A string column's physical
        # dtype is uint8, so a string column and a column of bytes agree on
        # `dtype()` and are not the same column at all.
        if (
            parts[p].dtype() != dt
            or parts[p].is_string() != parts[0].is_string()
        ):
            raise Error(
                "concat: every column must have the same dtype; got "
                + String(parts[0].type)
                + " and "
                + String(parts[p].type)
            )
        total += len(parts[p])

    if parts[0].is_string():
        var payload_bytes = 0
        for p in range(len(parts)):
            payload_bytes += len(parts[p].strings().payload)
        var out = _StringStack(total, payload_bytes)
        for p in range(len(parts)):
            out.paste(parts[p].strings())
        return AnyArray(out^.finish())

    comptime for candidate in ALL:
        if dt == candidate:
            var out = Array[candidate](total)
            var at = 0
            for p in range(len(parts)):
                _copy_into(
                    out,
                    at,
                    parts[p].unsafe_ptr[candidate](),
                    parts[p].data.validity,
                )
                at += len(parts[p])
            return AnyArray(out^)
    raise Error("concat: unsupported dtype " + String(dt))


def concat_two_any(a: AnyArray, b: AnyArray) raises -> AnyArray:
    """Stacks exactly two columns whose dtype is a runtime value.

    The list spelling would be the same operation, and it is not the one a join
    can use: building a `List[AnyArray]` out of two columns it only borrows means
    deep copying both, which on the key alignment path is the entire cost of the
    operation paid a second time. Two arguments borrow.

    Args:
        a: The first column.
        b: The second, of the same dtype.

    Returns:
        A column of `len(a) + len(b)` rows.

    Raises:
        If the dtypes differ or have no physical layout.
    """
    if a.dtype() != b.dtype():
        raise Error(
            "concat: every column must have the same dtype; got "
            + String(a.dtype())
            + " and "
            + String(b.dtype())
        )
    if a.is_string() != b.is_string():
        raise Error(
            "concat: every column must have the same dtype; got "
            + String(a.type)
            + " and "
            + String(b.type)
        )
    if a.is_string():
        var out = _StringStack(
            len(a) + len(b),
            len(a.strings().payload) + len(b.strings().payload),
        )
        out.paste(a.strings())
        out.paste(b.strings())
        return AnyArray(out^.finish())

    comptime for candidate in ALL:
        if a.dtype() == candidate:
            var out = Array[candidate](len(a) + len(b))
            _copy_into(out, 0, a.unsafe_ptr[candidate](), a.data.validity)
            _copy_into(out, len(a), b.unsafe_ptr[candidate](), b.data.validity)
            return AnyArray(out^)
    raise Error("concat: unsupported dtype " + String(a.dtype()))


def concat_strings(parts: List[StringArray]) raises -> StringArray:
    """Stacks string columns end to end, in the order given.

    Args:
        parts: The columns. An empty list gives an empty column.

    Returns:
        A column as tall as the parts put together.

    Raises:
        If the parts hold more payload than a view's offset field can address.
    """
    var rows = 0
    var payload_bytes = 0
    for p in range(len(parts)):
        rows += len(parts[p])
        payload_bytes += len(parts[p].payload)

    var out = _StringStack(rows, payload_bytes)
    for p in range(len(parts)):
        out.paste(parts[p])
    return out^.finish()


struct _StringStack(Movable):
    """The output of a string concat, filled one part at a time.

    It exists so that the list form and the two argument form of the concat
    share the copying rather than each spelling it out, and so that the two
    running offsets, the row and the payload byte, cannot get out of step with
    each other.
    """

    var _views: Buffer
    var _payload: Buffer
    var _validity: Bitmap
    var _rows: Int
    var _at: Int
    var _base: Int

    def __init__(out self, rows: Int, payload_bytes: Int) raises:
        """Allocates the output.

        Args:
            rows: The total number of elements.
            payload_bytes: The total payload of every part put together.

        Raises:
            Error: If the payload is larger than a view can address.
        """
        # A view holds its payload offset in 32 bits, so a column can carry four
        # gigabytes of long strings and not a byte more. The parts are each under
        # that or they could not have been built, and their sum need not be.
        if payload_bytes > Int(UInt32.MAX):
            raise Error(
                String(
                    "concat: ",
                    payload_bytes,
                    " bytes of string payload exceeds the ",
                    Int(UInt32.MAX),
                    " a column can address",
                )
            )
        self._views = Buffer(rows * VIEW_SIZE)
        self._payload = Buffer(payload_bytes)
        self._validity = Bitmap(rows)
        self._rows = rows
        self._at = 0
        self._base = 0

    def paste(mut self, col: StringArray):
        """Copies one part in at the current row and payload offset.

        Args:
            col: The part. Its elements land at rows `[at, at + len(col))`.
        """
        var n = len(col)
        if n == 0:
            return

        unsafe_memcpy(
            dest=self._views.unsafe_ptr().unsafe_offset(self._at * VIEW_SIZE),
            src=col.views.unsafe_ptr(),
            count=n * VIEW_SIZE,
        )

        var bytes = len(col.payload)
        if bytes > 0:
            unsafe_memcpy(
                dest=self._payload.unsafe_ptr().unsafe_offset(self._base),
                src=col.payload.unsafe_ptr(),
                count=bytes,
            )
            if self._base > 0:
                self._rebase(n)

        # The output starts all present, so a part with no nulls is already
        # right and the check is a popcount against a memcpy.
        if not col.validity.all_valid():
            self._validity.paste(self._at, col.validity, n)

        self._at += n
        self._base += bytes

    def _rebase(mut self, n: Int):
        """Moves the just-copied views onto the stacked payload.

        A part's offsets are relative to that part's payload, and the payloads
        are laid end to end, so every view that has an offset needs the part's
        base added to it. Short elements carry their bytes inside the view and
        are left alone, which is most of them on the columns a dataframe is
        actually full of.

        Args:
            n: How many views were just copied.
        """
        var views = (
            self._views.unsafe_ptr()
            .unsafe_bitcast[StringView]()
            .unsafe_offset(self._at)
        )
        var base = UInt32(self._base)
        for i in range(n):
            var slot = views.unsafe_offset(i)
            var view = slot[]
            if not view.is_inline():
                view.shift_offset(base)
                slot[] = view

    def finish(deinit self) raises -> StringArray:
        """Hands over the finished column.

        Returns:
            The column.

        Raises:
            Error: If fewer rows were pasted than were allocated for.
        """
        if self._at != self._rows:
            raise Error(
                String(
                    "concat: ",
                    self._at,
                    " rows were pasted into room for ",
                    self._rows,
                )
            )
        return StringArray(
            self._views^, self._payload^, self._validity^, self._rows
        )


def _copy_into[
    dt: DType, //, origin: ImmOrigin
](
    mut out: Array[dt],
    at: Int,
    src: Pointer[Scalar[dt], origin],
    valid: Bitmap,
):
    """Writes one part into the output at a row offset.

    The values go across unconditionally, nulls included, because a null holds a
    zero and copying it preserves the invariant the rest of the kernel layer
    rests on. Only the bits need a decision, and only when there are nulls.

    Args:
        out: The output column. Must have room for the part at `at`.
        at: The row the part starts on.
        src: The part's values.
        valid: The part's validity. Its length is the part's height.

    Parameters:
        dt: The dtype.
        origin: The part's origin.
    """
    comptime width = simd_width_of[dt]()
    var n = len(valid)
    var target = out.unsafe_ptr()

    var i = 0
    while i + width <= n:
        target.unsafe_offset(at + i).unsafe_store(
            src.unsafe_offset(i).unsafe_load[width=width]()
        )
        i += width
    while i < n:
        target.unsafe_offset(at + i).unsafe_store(
            src.unsafe_offset(i).unsafe_load()
        )
        i += 1

    # The output starts all present, so a part with no nulls is already right.
    if valid.all_valid():
        return
    for w in range(valid.word_count()):
        var word = valid.unsafe_word(w)
        if word == UInt64.MAX:
            continue
        var base = w * 64
        var last = base + 64
        if last > n:
            last = n
        for r in range(base, last):
            if (word >> UInt64(r - base)) & 1 == 0:
                out.data.validity.set(at + r, False)
