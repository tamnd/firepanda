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

`flatbuf.mojo`, `arrow_c.mojo`, `arrow_export.mojo`, `arrow_import.mojo`,
`arrow_ipc.mojo` and `arrow_ipc_write.mojo` are the Arrow side. The first three
are boundaries rather than readers and are imported by their full path when
something needs them; what a caller wants from here is `read_arrow` and
`write_arrow`, which read and write an IPC file the same way `read_csv` and
`write_csv` do a CSV.

`mapped.mojo` is how a file on disk becomes bytes in the first place. It maps
the file instead of copying it, which is worth a paragraph of libc and four
constants because the copy was the largest single part of a read.

`parquet.mojo`, `duckdb.mojo` and `duckvector.mojo` are the Parquet side, and
the shape is different from the rest of this directory because Parquet is
decoded by DuckDB rather than by us. The last two are a binding and a buffer
reader that nothing outside should need; what a caller wants from here is
`read_parquet`, which takes a path, or a path and the columns to read, or a path
and a `ParquetOptions` when the path is a partitioned directory rather than a
file.
"""

from .arrow_ipc import (
    read_arrow,
    read_arrow_bytes,
    read_ipc_file,
    read_ipc_stream,
)
from .arrow_ipc_write import (
    IpcWriteOptions,
    write_arrow,
    write_ipc_file_bytes,
    write_ipc_stream,
    write_ipc_stream_bytes,
)
from .mapped import MappedFile, map_file
from .parquet import ParquetOptions, quote, read_parquet
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
    SPECULATE_ROWS,
    TILE_BYTES,
    ReadOptions,
    TypeGuess,
    block_count,
    guess_column,
    infer_column,
    infer_schema,
    read_csv,
    read_csv_as,
    read_csv_bytes,
    read_csv_bytes_as,
    rung_of,
    sample_columns,
    scan_blocks,
    sweep_fixed,
    wanted_of,
)
from .scalar import scan_csv_scalar
from .scan import (
    Dialect,
    FieldSpan,
    LongField,
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
