"""Stacking columns end to end.

`concat` is the one operation in this package whose cost is entirely memory
traffic. There is no arithmetic in it and no branch that depends on a value, so
the only thing worth designing is how few passes it makes over the bytes and how
much of the validity work it can skip.

It skips a lot. The output is allocated unzeroed, because between them the parts
write every row of it, and it starts marked present everywhere, so a part with no
nulls needs no validity work at all: the bits it wants are already there. On the
columns most people actually concatenate, that is a pass over the values and the
whole bitmap loop gone, and what remains is a vectorized copy.

The parts are independent once the offset each one lands on is known, and those
offsets are a prefix sum taken before anything is copied, so a concat over the
parallel threshold hands one part to each core. That is worth doing for the same
reason the copy is the only thing here worth designing: one thread of memcpy
reaches about a sixth of what the machine can move.

There are two spellings of the erased concat and the difference between them is
ownership rather than behaviour. `concat_any` takes the columns, and
`concat_refs_any` takes references to them. A caller whose columns live inside
frames, or inside a list of record batches, only borrows them, and building a
`List[AnyArray]` to hand over would deep copy every one first. That copy is the
same size as the concat, so it doubles an operation that is already nothing but
memory traffic.

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
from firepanda.array.data import ColumnData
from firepanda.array.strings import StringArray
from firepanda.array.strview import StringView, VIEW_SIZE
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.lists import ALL
from firepanda.exec import parallel_for

comptime PARALLEL_ROWS = 1 << 16
"""Rows below which a concat stays on one thread.

Handing a part to a worker costs something and copying a short column costs
almost nothing, so a concat of a few thousand rows is faster left alone. Sixty
five thousand rows of views is a megabyte, which is where the copy stops fitting
in a core's own cache and starts being worth spreading. A fixed width column of
the same height is half that at eight bytes a row, and the threshold is shared
rather than tuned per type because the cost either side of it is small.
"""


def concat_arrays[dt: DType](parts: List[Array[dt]]) raises -> Array[dt]:
    """Stacks columns end to end, in the order given.

    Args:
        parts: The columns. An empty list gives an empty column, which is the
            answer a caller folding over a list of frames wants.

    Parameters:
        dt: The dtype.

    Returns:
        A column as tall as the parts put together.

    Raises:
        Error: If a worker fails. Nothing here can fail on its own, and the
            signature says so only because the copies run on every core.
    """
    var total = 0
    var fixed = List[_Part](capacity=len(parts))
    for p in range(len(parts)):
        fixed.append(_part_of(parts[p].data, total))
        total += len(parts[p])
    return _stack_fixed[dt](fixed, total)


def column_ref(imm col: AnyArray) -> Pointer[AnyArray, ImmUntrackedOrigin]:
    """Takes a reference to a column, for handing to `concat_refs_any`.

    Args:
        col: The column. Must outlive the reference.

    Returns:
        The reference.
    """
    return Pointer(to=col).unsafe_origin_cast[ImmUntrackedOrigin]()


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
    var refs = List[Pointer[AnyArray, ImmUntrackedOrigin]](capacity=len(parts))
    for p in range(len(parts)):
        refs.append(column_ref(parts[p]))
    return concat_refs_any(refs)


def concat_refs_any(
    parts: List[Pointer[AnyArray, ImmUntrackedOrigin]],
) raises -> AnyArray:
    """Stacks columns the caller only borrows.

    This is the spelling anything holding its parts inside something else wants,
    and the list of columns is the one that has to be built from it rather than
    the other way round. A caller with the columns inside frames, or inside a
    list of batches, cannot hand over a `List[AnyArray]` without deep copying
    every column into it first, and that copy is the same size as the concat, so
    the operation costs twice what it should. Reading an Arrow file of a hundred
    and fifty record batches was paying exactly that.

    Args:
        parts: References to the columns, in output row order. All of them must
            have the same dtype, and all must outlive the call.

    Returns:
        A column as tall as the parts put together.

    Raises:
        If the list is empty, if two parts disagree on dtype, or if the dtype
        has no physical layout.
    """
    if len(parts) == 0:
        raise Error("concat: at least one column is required")

    var dt = parts[0][].dtype()
    var total = 0
    for p in range(len(parts)):
        # The `is_string` half is not redundant. A string column's physical
        # dtype is uint8, so a string column and a column of bytes agree on
        # `dtype()` and are not the same column at all.
        if (
            parts[p][].dtype() != dt
            or parts[p][].is_string() != parts[0][].is_string()
        ):
            raise Error(
                "concat: every column must have the same dtype; got "
                + String(parts[0][].type)
                + " and "
                + String(parts[p][].type)
            )
        total += len(parts[p][])

    if parts[0][].is_string():
        var payload_bytes = 0
        var pieces = List[_Piece](capacity=len(parts))
        var row = 0
        for p in range(len(parts)):
            ref col = parts[p][].strings()
            pieces.append(_piece_of(col, row, payload_bytes))
            row += len(col)
            payload_bytes += len(col.payload)
        var out = _StringStack(total, payload_bytes)
        # A read reaches this spelling and not the typed one, so this is the
        # path that has to be the fast one.
        if len(parts) > 1 and total >= PARALLEL_ROWS:
            out.paste_all(pieces, total, payload_bytes)
        else:
            for p in range(len(parts)):
                out.paste(parts[p][].strings())
        return AnyArray(out^.finish())

    var fixed = List[_Part](capacity=len(parts))
    var at = 0
    for p in range(len(parts)):
        fixed.append(_part_of(parts[p][].data, at))
        at += len(parts[p][])

    comptime for candidate in ALL:
        if dt == candidate:
            return AnyArray(_stack_fixed[candidate](fixed, total))
    raise Error("concat: unsupported dtype " + String(dt))


def concat_two_any(a: AnyArray, b: AnyArray) raises -> AnyArray:
    """Stacks exactly two columns whose dtype is a runtime value.

    `concat_refs_any` would give the same answer, and this is still the spelling
    a join wants: two arguments say what they mean, and there is no list to
    build for a case that is known to have exactly two parts.

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

    var fixed = List[_Part](capacity=2)
    fixed.append(_part_of(a.data, 0))
    fixed.append(_part_of(b.data, len(a)))

    comptime for candidate in ALL:
        if a.dtype() == candidate:
            return AnyArray(_stack_fixed[candidate](fixed, len(a) + len(b)))
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
    var at = List[Int](capacity=len(parts))
    var base = List[Int](capacity=len(parts))
    for p in range(len(parts)):
        at.append(rows)
        base.append(payload_bytes)
        rows += len(parts[p])
        payload_bytes += len(parts[p].payload)

    var out = _StringStack(rows, payload_bytes)
    if len(parts) > 1 and rows >= PARALLEL_ROWS:
        var pieces = List[_Piece](capacity=len(parts))
        for p in range(len(parts)):
            pieces.append(_piece_of(parts[p], at[p], base[p]))
        out.paste_all(pieces, rows, payload_bytes)
    else:
        for p in range(len(parts)):
            out.paste(parts[p])
    return out^.finish()


@fieldwise_init
struct _Piece(ImplicitlyCopyable, Movable):
    """Where one part of a string concat is and where it lands.

    Pointers rather than the column itself, because the erased spelling of the
    concat only ever holds a reference to its parts and putting them in a list
    to hand over would deep copy every buffer in them, which is the whole cost
    of the operation paid twice over to avoid writing this struct.
    """

    var views: Pointer[UInt8, ImmUntrackedOrigin]
    var payload: Pointer[UInt8, ImmUntrackedOrigin]
    var validity: Pointer[Bitmap, ImmUntrackedOrigin]
    var rows: Int
    var bytes: Int
    var at: Int
    var base: Int


def _piece_of(imm col: StringArray, at: Int, base: Int) -> _Piece:
    """Describes one part of a concat without copying any of it.

    Args:
        col: The part. Must outlive the piece.
        at: The part's first row in the output.
        base: The part's payload offset in the output.

    Returns:
        The description.
    """
    return _Piece(
        col.views.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
        col.payload.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
        Pointer(to=col.validity).unsafe_origin_cast[ImmUntrackedOrigin](),
        len(col),
        len(col.payload),
        at,
        base,
    )


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
        # Every view and every payload byte is written by the pastes, and
        # `finish` refuses a stack that was not filled to the row it was
        # allocated for, so neither of these needs zeroing first.
        self._views = Buffer(overwritten=rows * VIEW_SIZE)
        self._payload = Buffer(overwritten=payload_bytes)
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

    def paste_all(
        mut self,
        pieces: List[_Piece],
        rows: Int,
        payload_bytes: Int,
    ) raises:
        """Copies every part in at once, one worker to a part.

        Where each part lands is known before any of them is copied, from the
        two prefix sums the caller has already taken, so the parts do not have
        to go in one at a time to keep the offsets in step. That is what lets
        this run on every core, and it is worth running on every core: this is
        two memcpys and a walk over the views per part, all of it memory
        traffic, and one thread of it reaches about a sixth of what the machine
        can move.

        Validity stays sequential. It is addressed in bits, so two parts whose
        boundary falls inside a byte would each have to read, modify and write
        that byte, and it is one bit a row against sixteen bytes a row for the
        views, so there is nothing to win by racing for it.

        Args:
            pieces: Where each part is and where it lands, in order.
            rows: The total row count.
            payload_bytes: The total payload size.
        """
        var views = self._views.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        var payload = self._payload.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()

        def one(p: Int) raises {imm}:
            var piece = pieces[p]
            if piece.rows == 0:
                return
            var slots = views.unsafe_offset(piece.at * VIEW_SIZE)
            unsafe_memcpy(
                dest=slots, src=piece.views, count=piece.rows * VIEW_SIZE
            )
            if piece.bytes == 0:
                return
            unsafe_memcpy(
                dest=payload.unsafe_offset(piece.base),
                src=piece.payload,
                count=piece.bytes,
            )
            if piece.base == 0:
                return
            var typed = slots.unsafe_bitcast[StringView]()
            var shift = UInt32(piece.base)
            for i in range(piece.rows):
                var slot = typed.unsafe_offset(i)
                var view = slot[]
                if not view.is_inline():
                    view.shift_offset(shift)
                    slot[] = view

        parallel_for(one, len(pieces))

        for p in range(len(pieces)):
            var piece = pieces[p]
            if piece.rows > 0 and not piece.validity[].all_valid():
                self._validity.paste(piece.at, piece.validity[], piece.rows)

        self._at = rows
        self._base = payload_bytes

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


@fieldwise_init
struct _Part(ImplicitlyCopyable, Movable):
    """Where one part of a fixed width concat is and where it lands.

    The same shape as `_Piece` and there for the same reason: the erased concat
    borrows its parts, so a worker has to be handed addresses rather than
    columns. The values are typed once the dtype is known, which is why they are
    bytes here.
    """

    var values: Pointer[UInt8, ImmUntrackedOrigin]
    var validity: Pointer[Bitmap, ImmUntrackedOrigin]
    var rows: Int
    var at: Int


def _part_of(imm col: ColumnData, at: Int) -> _Part:
    """Describes one part of a fixed width concat without copying any of it.

    It takes the storage rather than the column so that the typed spelling and
    the erased one reach the same description, which is the point of having one.

    Args:
        col: The part's storage. Must outlive the description.
        at: The part's first row in the output.

    Returns:
        The description.
    """
    return _Part(
        col.values.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
        Pointer(to=col.validity).unsafe_origin_cast[ImmUntrackedOrigin](),
        col.length,
        at,
    )


def _stack_fixed[dt: DType](parts: List[_Part], total: Int) raises -> Array[dt]:
    """Copies every part into a fresh column, on every core when it pays.

    The output is allocated unzeroed, because between them the parts write every
    row of it. That is a whole pass over the column saved, and on a column of a
    hundred and sixty megabytes the pass is not free.

    Validity stays sequential for the same reason it does in the string stack.
    It is one bit a row against at least eight bytes a row for the values, and
    two parts whose boundary falls inside a byte would each have to read, modify
    and write that byte, so there is nothing to win by racing for it.

    Args:
        parts: Where each part is and where it lands, in order.
        total: The output row count, which the parts add up to.

    Parameters:
        dt: The dtype.

    Returns:
        The stacked column.

    Raises:
        Error: If a worker fails.
    """
    var out = Array[dt](overwritten=total)
    var target = out.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()

    def one(p: Int) raises {imm}:
        ref part = parts[p]
        _copy_values(
            target,
            part.at,
            part.values.unsafe_bitcast[Scalar[dt]](),
            part.rows,
        )

    if len(parts) > 1 and total >= PARALLEL_ROWS:
        parallel_for(one, len(parts))
    else:
        for p in range(len(parts)):
            one(p)

    for p in range(len(parts)):
        ref part = parts[p]
        if part.rows > 0 and not part.validity[].all_valid():
            out.data.validity.paste(part.at, part.validity[], part.rows)
    return out^


def _copy_values[
    dt: DType, //, origin: ImmOrigin
](
    target: Pointer[Scalar[dt], MutUntrackedOrigin],
    at: Int,
    src: Pointer[Scalar[dt], origin],
    count: Int,
):
    """Writes one part's values into the output at a row offset.

    Nulls go across with everything else, because a null holds a zero and
    copying it preserves the invariant the rest of the kernel layer rests on.

    Args:
        target: The output values.
        at: The row the part starts on.
        src: The part's values.
        count: How many values to copy.

    Parameters:
        dt: The dtype.
        origin: The part's origin.
    """
    comptime width = simd_width_of[dt]()
    var i = 0
    while i + width <= count:
        target.unsafe_offset(at + i).unsafe_store(
            src.unsafe_offset(i).unsafe_load[width=width]()
        )
        i += width
    while i < count:
        target.unsafe_offset(at + i).unsafe_store(
            src.unsafe_offset(i).unsafe_load()
        )
        i += 1
