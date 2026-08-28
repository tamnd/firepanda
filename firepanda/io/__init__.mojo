"""Reading and writing files.

The bottom two layers have nothing to do with a frame: `scan.mojo` finds where
the fields are and `parse.mojo` turns a field's bytes into a value. Both take a
span of bytes and neither knows what a `Schema` is, which is what lets them be
tested and fuzzed on their own and reused by the NDJSON reader when it arrives.

On top of those, `read.mojo` infers the schema and fills the columns, and
`write.mojo` renders a frame back out. Those two are the only files here that
know what a `DataFrame` is.
"""

from .parse import (
    Parsed,
    is_missing,
    parse_bool,
    parse_float,
    parse_int,
)
from .read import (
    INFER_ALL,
    ReadOptions,
    infer_column,
    infer_schema,
    read_csv,
    read_csv_bytes,
    read_csv_bytes_as,
)
from .scalar import scan_csv_scalar
from .scan import (
    Dialect,
    FieldSpan,
    Scan,
    default_dialect,
    field_bytes,
    scan_csv,
    unescape,
)
from .write import (
    WriteOptions,
    needs_quoting,
    write_csv,
    write_csv_bytes,
)
