"""Reading and writing files.

Right now this is the two halves of a CSV reader that have nothing to do with a
frame: `scan.mojo` finds where the fields are and `parse.mojo` turns a field's
bytes into a value. Both take a span of bytes and neither knows what a `Schema`
is, which is what lets them be tested and fuzzed on their own and reused by the
NDJSON reader when it arrives.

The parts that do know about a frame, schema inference and the column building
loop and `read_csv` itself, land on top of these.
"""

from .parse import (
    Parsed,
    is_missing,
    parse_bool,
    parse_float,
    parse_int,
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
