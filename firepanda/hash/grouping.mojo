"""Turning one or more key columns into a single dense group ordinal per row.

`factorize` does this for one column of one known dtype. A group by needs it for
several columns whose dtypes are runtime values and which have to be combined
into one ordinal that identifies the tuple rather than any single key. That is
what this file is, and it lives in `firepanda/hash` rather than in the frame
layer because it is the same operation `factorize` is and it belongs next to it.

## How several keys become one

The obvious approach is to hash the tuple. This does not, because `factorize`
already exists and composing it is both cheaper and easier to be sure of.

Group on the first key and you have an ordinal per row in `[0, g0)`. Group on the
second and you have another in `[0, g1)`. The pair `(c0, c1)` identifies the tuple
exactly, and `c0 * g1 + c1` packs the pair into one integer without collisions.
That integer is then factorized itself, which makes it dense again, and the
process repeats for the third key and the fourth.

Refactorizing after every combine is what keeps this from overflowing. Without
it, the running product is `g0 * g1 * g2 * ...`, which is the size of the full
cross product and reaches int64 on four columns of a thousand values each. With
it, the running ordinal count is the number of key tuples actually observed,
which can never exceed the row count, so the product being packed is bounded by
`rows * g_next` and stays inside int64 for any table that fits on a machine. The
bound is checked anyway, because "cannot happen" is not a thing to rely on in the
one function every group by goes through.

There is a real cost to this and it should not be hidden: `n` keys means `n`
factorize passes plus `n - 1` more over the combined column. A single fused hash
of the tuple would be one pass. The reason to start here is that the composition
reuses the direct route, so grouping on three small integer columns never hashes
anything at all, and a fused tuple hash would hash all three. Which of those wins
is a measurement, and the benchmark that settles it is `group/ordinals_two_keys`
against `group/ordinals_one_key`.

## The representative row

An aggregation produces one row per group and those rows need their key values
back. Rather than teach every dtype how to reconstruct a key from the hash table,
this records the first row index at which each group was seen, and the frame layer
gathers the key columns by those indices with the take kernel it already has.
That works for every dtype including ones with no key representation at all, and
it gets null keys right for free, because the row it points at holds the null.

For that to be possible every group has to have a row, which is not something
`factorize` promises, so `_densify` renumbers the ordinals into the order they
were first seen and discards the ones nothing carries. Its docstring has the case
that forced it.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.lists import ALL

from firepanda.kernel.agg import max_of

from .factorize import factorize


struct Grouping(Movable):
    """One group ordinal per row, plus what is needed to describe the groups."""

    var codes: Array[DType.uint32]
    """The ordinal each row belongs to, in `[0, groups)`."""

    var groups: Int
    """How many distinct key tuples were seen."""

    var rows_at: List[Int]
    """For each group, the first row index that had it. Gather the key columns by
    this to get the group's key values back."""

    def __init__(
        out self,
        var codes: Array[DType.uint32],
        groups: Int,
        var rows_at: List[Int],
    ):
        """Constructs a grouping.

        Args:
            codes: The per-row ordinals.
            groups: The number of distinct ordinals.
            rows_at: A representative row per group.
        """
        self.codes = codes^
        self.groups = groups
        self.rows_at = rows_at^


def group_ordinals(
    columns: List[AnyArray], at: List[Int], rows: Int
) raises -> Grouping:
    """Assigns a dense ordinal to every distinct tuple of key values.

    Args:
        columns: The frame's columns.
        at: Which of them are keys, in the order they should be combined.
        rows: The frame's height.

    Returns:
        The ordinals, the group count and a representative row per group.

    Raises:
        If no keys were given, if a key dtype has no physical layout, or if the
        combined ordinal space would not fit in an int64.
    """
    if len(at) == 0:
        raise Error("group by: at least one key column is required")

    var codes = _factorize_any(columns[at[0]])
    var rows_at = List[Int]()
    var groups = _densify(codes, rows_at)

    for k in range(1, len(at)):
        var next = _factorize_any(columns[at[k]])
        var spare = List[Int]()
        var next_groups = _densify(next, spare)
        if next_groups > 0 and groups > Int(Int64.MAX) // next_groups:
            raise Error(
                "group by: the combined key space of "
                + String(groups)
                + " by "
                + String(next_groups)
                + " does not fit in an int64"
            )

        # Pack the pair, then make it dense again. The packing is exact and the
        # refactorize is what stops the product from compounding across keys.
        var packed = Array[DType.int64](rows)
        var left = codes.unsafe_ptr()
        var right = next.unsafe_ptr()
        var into = packed.unsafe_ptr()
        for i in range(rows):
            into.unsafe_offset(i).unsafe_store(
                Int64(left.unsafe_offset(i).unsafe_load()) * Int64(next_groups)
                + Int64(right.unsafe_offset(i).unsafe_load())
            )
        codes = factorize(packed).into_codes()
        rows_at = List[Int]()
        groups = _densify(codes, rows_at)

    return Grouping(codes^, groups, rows_at^)


def _factorize_any(col: AnyArray) raises -> Array[DType.uint32]:
    """Factorizes a column whose dtype is a runtime value.

    The typed copy this takes is the one avoidable cost in the whole group by and
    it is here because `factorize` reduces the column with `min_of` and `max_of`
    to decide whether it can skip hashing, and those take an `Array` rather than a
    pointer. Rewriting that path to pointer form is worth doing and is not worth
    doing in the change that introduces group by.

    Args:
        col: The key column.

    Returns:
        One ordinal per row.

    Raises:
        If the dtype has no physical layout.
    """
    comptime for candidate in ALL:
        if col.dtype() == candidate:
            return factorize(col.as_typed[candidate]()).into_codes()
    raise Error("group by: unsupported key dtype")


def _densify(mut codes: Array[DType.uint32], mut rows_at: List[Int]) -> Int:
    """Renumbers ordinals into first-seen order and drops the unused ones.

    `factorize` does not promise that every ordinal it can produce belongs to a
    row. The direct integer route builds its key table from the physical values
    it saw, and a null row still has a physical value, so a column of 10s and 20s
    with one null yields four ordinals where only three are reachable: the null
    group, 10, 20, and a fourth for the zero that `set_null` left behind. Passing
    that count downstream would produce an aggregation row nobody asked for, with
    no row to recover its key from.

    Renumbering by first appearance fixes that and pays for itself twice more. It
    makes the group count exact, which is what bounds the packed key space when a
    second key is combined in. It fills the representative row table in the same
    pass rather than a second one. And it defines what `sort=False` means at the
    frame layer, which is the order the groups were first seen in, matching what
    pandas and Polars both do.

    Args:
        codes: The ordinals, rewritten in place.
        rows_at: Filled with the first row index of each ordinal, in order.

    Returns:
        The number of ordinals that at least one row carries.
    """
    var at = codes.unsafe_ptr()
    var n = len(codes)
    var top = -1
    if n > 0:
        var found = max_of(codes)
        if found.valid:
            top = Int(found.value)

    var remap = List[Int](capacity=top + 1)
    for _ in range(top + 1):
        remap.append(-1)

    var groups = 0
    for i in range(n):
        var raw = Int(at.unsafe_offset(i).unsafe_load())
        if remap[raw] < 0:
            remap[raw] = groups
            rows_at.append(i)
            groups += 1
        at.unsafe_offset(i).unsafe_store(UInt32(remap[raw]))
    return groups
