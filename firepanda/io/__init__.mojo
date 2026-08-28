"""Reading and writing files.

The bottom two layers have nothing to do with a frame: `scan.mojo` finds where
the fields are and `parse.mojo` turns a field's bytes into a value. Both take a
span of bytes and neither knows what a `Schema` is, which is what lets them be
tested and fuzzed on their own and reused by the NDJSON reader when it arrives.

On top of those, `read.mojo` infers the schema and fills the columns, and
`write.mojo` renders a frame back out. Those two are the only files here that
know what a `DataFrame` is.

`split.mojo` sits beside the scanner and cuts a buffer into blocks that each
begin on a row boundary, which is what lets the whole read run on every core. It
knows about quoting and nothing else.

`mapped.mojo` is how a file on disk becomes bytes in the first place. It maps
the file instead of copying it, which is worth a paragraph of libc and four
constants because the copy was the largest single part of a read.
"""

from .mapped import MappedFile, map_file
from .parse import (
    Parsed,
    is_missing,
    parse_bool,
    parse_float,
    parse_int,
)
from .read import (
    INFER_ALL,
    MIN_BLOCK,
    ReadOptions,
    TypeGuess,
    block_count,
    guess_column,
    infer_column,
    infer_schema,
    read_csv,
    read_csv_bytes,
    read_csv_bytes_as,
    scan_blocks,
)
from .scalar import scan_csv_scalar
from .scan import (
    Dialect,
    FieldSpan,
    Scan,
    default_dialect,
    field_bytes,
    scan_block,
    scan_csv,
    unescape,
)
from .split import (
    Split,
    count_bytes,
    row_start_at_or_after,
    split_buffer,
)
from .write import (
    WriteOptions,
    needs_quoting,
    write_csv,
    write_csv_bytes,
)
