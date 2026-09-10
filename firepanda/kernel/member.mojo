"""Asking whether each row's value is one of a set of values.

This is pandas' `isin` and SQL's `IN`. TPC-H asks for it twice, once against
seven country codes in q22 and once against four container names in q19, and
both of those are the shape this is tuned for: a column of a million rows and a
set of a handful.

That shape is why there are two routes and not one. A small enough set never
builds anything: the needles are compared against one at a time, and above the
threshold a hash table is built once and probed by every row. Where the threshold
sits depends entirely on what a single comparison costs, and that is completely
different for a number and for a string, which is why there are two thresholds
and not one. See `LINEAR_MAX` and `TEXT_LINEAR_MAX`.

The table route is exact and not probabilistic, which is worth spelling out
because it stores hashes rather than keys. For a fixed width dtype `mix` is a
bijection on sixty four bits, so two keys land on the same stored value exactly
when they are the same key and the comparison is already exact; `hash/table.mojo`
makes that argument at length. For text no function can do that, sixteen bytes of
name do not fit in eight bytes of hash, so the text route keeps the needles and
settles a hash match by comparing bytes. A row that collides with a needle it is
not equal to is rejected there.

There is one case the text table cannot settle that way, and it is checked for
rather than assumed away: two needles that are different strings and hash the
same would take one slot between them, and the second one would then be missing
from the set. The build notices when an insert hands back an ordinal that is
already spoken for by a different string, and falls back to comparing against
every needle. That branch has never been taken on real data and will not be, but
a set that silently lost a member is not the kind of wrong that shows up in a
test.

Null follows the comparison kernels: null in, null out. That is SQL's answer for
`NULL IN (...)` and it is what `equal` does here, so a caller building `IN` out
of a chain of equalities and a caller using this get the same column. It is not
what pandas' `isin` does, which reports False for a missing value, and the
binding layer is where that difference is paid because that is where the pandas
contract is.
"""

from std.sys.info import simd_width_of

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray
from firepanda.array.strview import StringView
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL
from firepanda.exec import parallel_morsels
from firepanda.hash.function import DEFAULT_SEED, hash_bytes, hash_of
from firepanda.hash.table import HashTable

from .mask import repair_range

comptime LINEAR_MAX = 32
"""The largest set of numbers answered by comparing against every member.

A number comparison is one SIMD equal against a block that is already loaded, so
each extra member costs about seven hundredths of a nanosecond a row. The table
costs much more than that per row, because it hashes and then probes, and the
probe gets longer as the table fills. That leaves the crossover a long way up.

To find it, both routes were built at every set size, by moving this constant, and
run against each other in one session on a million int64 rows. Nanoseconds a row,
linear against table: nine, 1.30 against 2.99; sixteen, 2.03 against 4.84; thirty
two, 3.82 against 5.03; sixty four, 7.06 against 5.15. So it turns over between
thirty two and sixty four, and thirty two is the last size where comparing
everything still wins with room to spare.
"""

comptime TEXT_LINEAR_MAX = 2
"""The largest set of strings answered by comparing against every member.

Sixteen times lower than the number, and for a reason worth spelling out because
one threshold for both looked obvious and was wrong by more than an order of
magnitude. A string comparison is not one instruction: two members of a set
usually share a length and a prefix, which is what makes them a set, so the
prefix settles nothing and the comparison runs to the end.

Measured the same way, on a million thirty two byte rows, the linear route runs
2.24, 3.89, 5.32, 6.84 and 14.43 nanoseconds a row for sets of one, two, three,
four and eight, against a flat 4.33 and 4.26 for the table at nine and sixty four.
That is about a nanosecond and a half for each extra member against a table that
does not care how many there are, and it puts the crossover between two and three.

Short elements move the crossover down rather than up, to between one and two: an
eight byte column runs 1.32, 1.79 and 2.73 for one, two and four against 1.61 for
the table. Two is the compromise between the two widths, and it is on the right
side of the line for the case that actually matters, which is any set large enough
to be worth writing down. Eight would have cost q19, whose set is four container
names, about sixty percent.

Those linear numbers at three and above cannot be reproduced from the benchmark
as it stands, because at three and above it now takes the table. Moving this
constant back up is what produced them.
"""


def is_in[
    dt: DType
](a: Array[dt], values: Array[dt]) raises -> Array[DType.bool]:
    """Returns a mask that is true where a row's value is in a set.

    Args:
        a: The column.
        values: The set. Duplicates and nulls in it are ignored: a null is not
            equal to anything, including itself, so a null member adds nothing
            that is not already there.

    Parameters:
        dt: The dtype of both.

    Returns:
        A bool column, null wherever the column is null. An empty set gives all
        false, and not all null, because nothing being in nothing is a fact.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    var out = Array[DType.bool](overwritten=n)
    var validity = Bitmap(copy=a.data.validity)

    # A null member is dropped here rather than being compared away row by row,
    # which keeps the loops below free of any mention of the set's validity.
    var wanted = List[Scalar[dt]]()
    for i in range(len(values)):
        if values.is_valid(i):
            wanted.append(values[i])
    var k = len(wanted)

    if k == 0:

        def none(start: Int, stop: Int) {mut out, imm}:
            var dst = out.unsafe_mut_ptr()
            for i in range(start, stop):
                dst.unsafe_offset(i).unsafe_write(False)
            repair_range(out, validity, start, stop)

        parallel_morsels(none, n)
        out.data.validity = validity^
        return out^

    if k <= LINEAR_MAX:
        comptime width = simd_width_of[dt]()

        def linear(start: Int, stop: Int) {mut out, imm}:
            var src = a.unsafe_ptr()
            var dst = out.unsafe_mut_ptr()
            var i = start
            while i < stop:
                var x = src.unsafe_offset(i).unsafe_load[width=width]()
                var hit = x.eq(SIMD[dt, width](wanted[0]))
                for j in range(1, k):
                    hit |= x.eq(SIMD[dt, width](wanted[j]))
                dst.unsafe_offset(i).unsafe_store(hit)
                i += width

            repair_range(out, validity, start, stop)

        parallel_morsels(linear, n)
        out.data.validity = validity^
        return out^

    # Above the threshold, one table built once and read by every worker. The
    # seed is fixed rather than taken from the table so that the probe hashes
    # with what the build hashed with; `find` says nothing matches otherwise.
    var table = HashTable(expected=k, seed=DEFAULT_SEED)
    for j in range(k):
        _ = table.insert(hash_of(wanted[j], DEFAULT_SEED))

    def probed(start: Int, stop: Int) {mut out, imm}:
        var src = a.unsafe_ptr()
        var dst = out.unsafe_mut_ptr()
        for i in range(start, stop):
            var key = hash_of(src.unsafe_offset(i).unsafe_load(), DEFAULT_SEED)
            dst.unsafe_offset(i).unsafe_write(table.find(key) >= 0)

        repair_range(out, validity, start, stop)

    parallel_morsels(probed, n)
    out.data.validity = validity^
    return out^


def text_is_in(a: StringArray, values: StringArray) raises -> Array[DType.bool]:
    """Returns a mask that is true where a row's text is in a set.

    Args:
        a: The column.
        values: The set. Duplicates and nulls in it are ignored.

    Returns:
        A bool column, null wherever the column is null. An empty set gives all
        false.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    var out = Array[DType.bool](overwritten=n)
    var validity = Bitmap(copy=a.validity)

    # The views and where each came from. Both are kept because dropping the
    # nulls renumbers the set, and the table below has to ask `values` about a
    # member by its own index while the loops ask about it by its place here.
    var wanted = List[StringView]()
    var member_at = List[Int]()
    for i in range(len(values)):
        if values.is_valid(i):
            wanted.append(values.view(i))
            member_at.append(i)
    var k = len(wanted)

    def write_false(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_mut_ptr()
        for i in range(start, stop):
            dst.unsafe_offset(i).unsafe_write(False)
        repair_range(out, validity, start, stop)

    if k == 0:
        parallel_morsels(write_false, n)
        out.data.validity = validity^
        return out^

    # A table is only worth building above the threshold, and it is only usable
    # if every needle got a slot of its own. Two different strings sharing a hash
    # would share an ordinal and the set would quietly lose one of them, so that
    # is checked here rather than reasoned about.
    var table = HashTable(expected=k, seed=DEFAULT_SEED)
    var slotted = List[Int]()
    var distinct = k > TEXT_LINEAR_MAX
    if distinct:
        for j in range(k):
            var bytes = values.unsafe_bytes(member_at[j])
            var at = table.insert(hash_bytes(bytes, DEFAULT_SEED))
            if at < len(slotted):
                # The slot was already taken. That is either a duplicate member,
                # which is fine and is dropped, or two different strings sharing
                # a hash, which is not.
                if not values.element_equals(
                    member_at[j], member_at[slotted[at]]
                ):
                    distinct = False
                    break
            else:
                slotted.append(j)

    if not distinct:

        def linear(start: Int, stop: Int) {mut out, imm}:
            var dst = out.unsafe_mut_ptr()
            for i in range(start, stop):
                var hit = False
                for j in range(k):
                    # Settled on the length and the first four bytes when it can
                    # be, which for a set of short names is every row that is not
                    # a member and is most of the column.
                    if a.element_equals_foreign(i, wanted[j], values):
                        hit = True
                        break
                dst.unsafe_offset(i).unsafe_write(hit)
            repair_range(out, validity, start, stop)

        parallel_morsels(linear, n)
        out.data.validity = validity^
        return out^

    def probed(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_mut_ptr()
        for i in range(start, stop):
            var at = table.find(hash_bytes(a.unsafe_bytes(i), DEFAULT_SEED))
            var hit = False
            if at >= 0:
                # A hash match is a candidate. The bytes settle it, which is what
                # keeps a collision from becoming a false positive.
                hit = a.element_equals_foreign(i, wanted[slotted[at]], values)
            dst.unsafe_offset(i).unsafe_write(hit)
        repair_range(out, validity, start, stop)

    parallel_morsels(probed, n)
    out.data.validity = validity^
    return out^


def is_in_any(a: AnyArray, values: AnyArray) raises -> Array[DType.bool]:
    """Returns a mask that is true where a row's value is in a set.

    Args:
        a: The column.
        values: The set. Must be the same type as the column, because promoting
            here would decide silently which side loses precision and a caller
            that meant to compare across types can cast first and say so.

    Returns:
        A bool column, null wherever the column is null.

    Raises:
        If the two are not the same type, or the type has no physical layout.
    """
    if a.is_string() != values.is_string():
        raise Error(
            "is_in: cannot look up "
            + String(a.type)
            + " in a set of "
            + String(values.type)
        )
    if a.is_string():
        return text_is_in(a.strings(), values.strings())
    if a.type.physical != values.type.physical:
        raise Error(
            "is_in: cannot look up "
            + String(a.type)
            + " in a set of "
            + String(values.type)
        )

    comptime for target in ALL:
        if a.type.physical == target:
            ref x = a.as_typed_view[target]()
            ref y = values.as_typed_view[target]()
            return is_in[target](x, y)
    raise Error("is_in: unsupported dtype")
