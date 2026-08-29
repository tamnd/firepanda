"""The obvious byte at a time versions of the io kernels.

Same contract as `firepanda/kernel/scalar`: these are never called in
production, they are what the fast paths are checked against, and they are
written for a reader rather than for a machine. If one of these and its
counterpart disagree on any input, the fast one is wrong until proven otherwise,
because this one is short enough to read in full and be sure of.

Only the scan has a twin here. The parsers in `parse.mojo` are already a
character at a time and a second copy of them would be the same code with a
different name, which tests nothing.
"""

from std.collections.span import Span

from .scan import NEWLINE, RETURN, Dialect, Scan


def scan_csv_scalar(data: Span[UInt8, _], dialect: Dialect) raises -> Scan:
    """Cuts a buffer into rows of fields, one byte at a time.

    Args:
        data: The whole file.
        dialect: The delimiter and quote character.

    Returns:
        The fields and the row offsets, identical to what `scan_csv` returns.

    Raises:
        If a quoted field is never closed, or if a closing quote is followed by
        anything other than a delimiter or a line ending.
    """
    var out = Scan()
    var n = len(data)
    var ptr = data.unsafe_ptr()
    var at = 0

    while at < n:
        var lead = ptr.unsafe_offset(at).unsafe_load()
        if lead == NEWLINE:
            at += 1
            continue
        if lead == RETURN:
            at += 1
            if at < n and ptr.unsafe_offset(at).unsafe_load() == NEWLINE:
                at += 1
            continue

        var row = len(out)
        var done = False
        while not done:
            var start = at
            var quoted = ptr.unsafe_offset(at).unsafe_load() == dialect.quote
            var escaped = False
            var stop = at

            if quoted:
                at += 1
                start = at
                var closed = False
                while at < n and not closed:
                    var c = ptr.unsafe_offset(at).unsafe_load()
                    if c != dialect.quote:
                        at += 1
                        continue
                    if (
                        at + 1 < n
                        and ptr.unsafe_offset(at + 1).unsafe_load()
                        == dialect.quote
                    ):
                        escaped = True
                        at += 2
                        continue
                    stop = at
                    at += 1
                    closed = True
                if not closed:
                    raise Error(
                        "csv: row "
                        + String(row)
                        + " opens a quoted field at byte "
                        + String(start - 1)
                        + " that is never closed"
                    )
                if at < n:
                    var after = ptr.unsafe_offset(at).unsafe_load()
                    if (
                        after != dialect.delimiter
                        and after != NEWLINE
                        and after != RETURN
                    ):
                        raise Error(
                            "csv: row "
                            + String(row)
                            + " has a closing quote at byte "
                            + String(stop)
                            + " followed by a value instead of a delimiter"
                        )
            else:
                while at < n:
                    var c = ptr.unsafe_offset(at).unsafe_load()
                    if c == dialect.delimiter or c == NEWLINE or c == RETURN:
                        break
                    at += 1
                stop = at

            out.push(start, stop, escaped, quoted)

            if at >= n:
                done = True
                continue
            var here = ptr.unsafe_offset(at).unsafe_load()
            if here == dialect.delimiter:
                at += 1
                if at >= n:
                    out.push(at, at, False, False)
                    done = True
                    continue
                var next = ptr.unsafe_offset(at).unsafe_load()
                if next == NEWLINE or next == RETURN:
                    out.push(at, at, False, False)
                    if next == RETURN:
                        at += 1
                    if (
                        at < n
                        and ptr.unsafe_offset(at).unsafe_load() == NEWLINE
                    ):
                        at += 1
                    done = True
                continue
            if here == RETURN:
                at += 1
                if at < n and ptr.unsafe_offset(at).unsafe_load() == NEWLINE:
                    at += 1
                done = True
                continue
            at += 1
            done = True

        out.end_row()

    return out^
