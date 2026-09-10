"""Reading a reduction's name, on the way in from Python.

`ops.mojo` does this for the twenty six operators and the argument is the same
one, so this file is short on purpose and exists only because the argument in
`ops.mojo` is about arithmetic and a reduction is not arithmetic. There is one
entry point per shape rather than one per reduction, the reduction crosses as a
word, and the word is the pandas method name so that `s.sum()` and a future
`df.agg("sum")` reach the same string from both directions.

### Why the parameter rides along beside the name

Four of the reductions take a number as well as a name. `var`, `std` and `sem`
take a delta degrees of freedom and `quantile` takes the quantile itself, which
is exactly the split `AggKind` already carries, so the boundary carries it too
rather than growing three more entry points. Everything that does not use it
passes zero and never reads it.

The parameter is a `Float64` even for the three that mean an integer. A delta
degrees of freedom is subtracted from a count and then divided by, so it is a
number in a formula rather than a length, and `AggKind` stores it as a float for
that reason. Reading it as an `Int` here and widening it there would be two
conversions to say the same thing.

### The twelve that are here and the five that are not

`AggKind` has seventeen reductions and twelve of them cross. `size`, `first` and
`last` are grouped shapes that a whole column reduction spells differently or
not at all, and `corr` and `cov` read a second column, which is a second entry
point and a question about aligning two indexes that this does not answer yet. A
name that is not on the list is a bug in the generated table rather than
something a user typed, because the Python layer never passes a word a user
wrote.
"""

from firepanda.kernel.group import AggKind
from firepanda.py.errors import VALUE, tagged


def reduction(name: String, param: Float64) raises -> AggKind:
    """Reads the name of a reduction.

    Args:
        name: The reduction, as pandas spells the method, such as `sum` or
            `std`.
        param: The delta degrees of freedom for `var`, `std` and `sem`, the
            quantile for `quantile`, and zero for the rest, which ignore it.

    Returns:
        The kind, carrying the parameter for the four that use one.

    Raises:
        Error: Tagged `value`, if the name is not one of the twelve.
    """
    if name == "sum":
        return AggKind.SUM
    if name == "mean":
        return AggKind.MEAN
    if name == "min":
        return AggKind.MIN
    if name == "max":
        return AggKind.MAX
    if name == "count":
        return AggKind.COUNT
    if name == "median":
        return AggKind.MEDIAN
    if name == "nunique":
        return AggKind.NUNIQUE
    if name == "skew":
        return AggKind.SKEW
    if name == "var":
        return AggKind(AggKind.VAR.code, param)
    if name == "std":
        return AggKind(AggKind.STD.code, param)
    if name == "sem":
        return AggKind(AggKind.SEM.code, param)
    if name == "quantile":
        return AggKind(AggKind.QUANTILE.code, param)
    raise tagged(VALUE, String("unknown reduction ", name))
