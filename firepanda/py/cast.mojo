"""The one rule a pandas facing cast follows that the kernel does not.

`firepanda/kernel/cast.mojo` is a straight conversion and says so: it does not
range check, a NaN converted to an integer is undefined the way it is in C, and
its docstring says the check belongs in the layer above. This is that layer, and
the rule is short enough that the whole of it fits in one function.

A float column holding a missing value, a NaN or an infinity cannot become a
plain integer column, because none of the three has an integer to be. pandas
raises `IntCastingNaNError` for all three, which is the error that exists to
explain why the nullable integer dtypes exist. firepanda handed back a null in
an integer column for the first two and the largest int64 there is for the
third, and the first of those is the worse answer, because nothing failed and
the caller ended up holding a column pandas could not have made.

The check lives here and not in the kernel because it is a pandas rule rather
than an Arrow one. An Arrow integer column holding a null is perfectly legal and
`firepanda/frame/series.mojo` keeps making them. Everything reaching this module
is on its way to Python through the pandas surface, so this is the boundary where
the pandas answer is the right one.

One consequence worth knowing about: firepanda reaches this from a direction
pandas cannot. `firepanda.Series([1, None, 3])` is an int64 column with a null
where the pandas one is a float64 column with a NaN, so a caller can ask to
convert an integer column that already holds a missing value. It is refused too,
since pandas would have refused the column it would have had.
"""

from firepanda.array.any import AnyArray
from firepanda.dtype.logical import LogicalType, TypeKind
from firepanda.kernel.cast import integer_ready
from firepanda.py.errors import NONFINITE, tagged


comptime NOT_FINITE = (
    "Cannot convert non-finite values (NA or inf) to integer. Replace or remove"
    " them first, or convert to a type that has somewhere to put them, which"
    " for a column of whole numbers means float64"
)
"""What a caller reads.

The first sentence is the one pandas says, so that a program matching on the
message keeps working. The rest is not: the pandas one is missing two spaces and
suggests the nullable `Int64` dtype, which firepanda does not have and is not
going to, since an Arrow column already holds missing values without a second
missing value model bolted on beside it."""


def checks_finite(wanted: LogicalType) -> Bool:
    """Reports whether a conversion to a type has to look at the values first.

    An integer column is the only one with nowhere to put a missing value, a NaN
    or an infinity, so it is the only target that needs the scan. Text, floats
    and booleans all have somewhere to put one and are never asked about.

    Callers that have to pay for the column before they can hand it over ask this
    first, so that the price is only paid when the answer can matter.

    Args:
        wanted: What a column is being converted to.

    Returns:
        True if `refuse_if_not_finite` will do any work.
    """
    return wanted.kind == TypeKind.INT


def refuse_if_not_finite(col: AnyArray, wanted: LogicalType) raises:
    """Refuses a conversion to an integer column that pandas would refuse.

    Does nothing at all unless `checks_finite` says the target is one that
    cares, which keeps the scan off every other cast in the library.

    Args:
        col: The column about to be converted.
        wanted: What it is being converted to.

    Raises:
        Error: Tagged `nonfinite` if the column holds a missing value, a NaN or
            an infinity and the target is an integer type.
    """
    if not checks_finite(wanted):
        return
    if integer_ready(col):
        return
    raise tagged(NONFINITE, NOT_FINITE)
