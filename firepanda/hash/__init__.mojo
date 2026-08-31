"""The hash table, which we have to write ourselves.

Group by and join are the two operations a dataframe is judged on, and both are a
hash table with extra steps. The MojoFrame authors named Mojo's `Dict` as the
reason their high cardinality aggregation results fall behind, so this package
exists from M1 rather than later. Whether that was worth doing is a measurement
and not an opinion; `benchmarks/main.mojo` runs this against a `Dict` doing the
same job and prints both numbers.

Three ideas hold the package together.

**The table stores key bits, not key values.** `key_bits` turns a value of any
numeric dtype into the 64 bits the table compares. Floats go through it too, and
that is where the normalization lives: NaN becomes one canonical pattern so that
all NaNs land in one group instead of each being its own, and negative zero
becomes positive zero so that `-0.0` and `0.0` group together. Comparing values
with `==` would put every NaN in a group of one and make the probe loop insert
forever, and it is the kind of thing that only shows up on real data.

**Not every group by needs a hash table.** `factorize` looks at the column first.
A column of integers whose range is small enough indexes a table directly and
never hashes anything, which is the case a group by on a year, an identifier or a
category code actually is. The hash path is the fallback, not the default.

**Ordinals are assigned on insertion.** A group gets a dense integer the first
time it is seen, so an aggregation writes into a flat array indexed by ordinal
rather than back into the table. That keeps the table read-mostly during the
aggregate and it is what makes radix partitioning worth anything.

**Text is the exception to the first idea.** A string does not fit in 64 bits, so
its hash is a real hash and two different strings can land on the same one. The
string route through the table compares the bytes on a hash match rather than
taking it as proof, which is the one place in this package where a probe does
more than compare two integers.

What is not here yet: the dictionary encoded bypass and the parallel build. The
bypass needs dictionary encoded columns to exist, and the parallel build needs
the executor. `partition.mojo` is the piece the parallel build will stand on and
it is written and tested now because the layout it produces constrains the rest.

Every function here has a twin in `scalar.mojo`, on the same terms as
`firepanda/kernel`: the twin is the specification, and when the two disagree the
fast one is wrong.
"""

from .factorize import (
    CHUNK_ROWS,
    DIRECT_LIMIT,
    Factorized,
    FactorizedStrings,
    factorize,
    factorize_dense,
    factorize_strings,
)
from .function import (
    DEFAULT_SEED,
    hash_bytes,
    hash_chunk,
    hash_into,
    hash_of,
    hash_strings_chunk,
    key_bits,
    mix,
)
from .grouping import Grouping, group_ordinals
from .partition import Partitioning, radix_partition
from .scalar import (
    factorize_dict,
    factorize_linear,
    factorize_strings_linear,
)
from .table import HashTable
