"""What to compute for each group, and what firepanda's group by promises.

`AggKind` in `firepanda.kernel.group` says which reduction. `AggSpec` here says
which column to run it on and what to call the result, which is a frame level
question rather than a kernel one. The reductions themselves are all in the
kernel; this file is the vocabulary `DataFrame.group_by` speaks.

## Where this differs from pandas, and why

**One call, not a chain.** pandas spells it `df.groupby("k").sum()`, where the
intermediate object holds a reference to the frame and defers everything. That
shape needs either a borrow that outlives the expression or a copy of the frame,
and firepanda has neither yet. `df.group_by(["k"], specs)` is one call that takes
what to group on and what to compute in the same place, so nothing is held
between statements. The chained spelling is a facade over this and it belongs
with the plan layer at M4, which is where deferral gets a lifetime story.

**Empty groups.** They cannot happen here, because a group only exists if a row
produced it. What can happen is a group in which every value of the aggregated
column is null, and those follow pandas: `sum` gives zero, `count` gives zero,
`size` counts the rows anyway, and `mean`, `min`, `max`, `first` and `last` are
null.

**Null keys.** `dropna` defaults to true, as pandas does, so rows whose key is
null form a group that is then dropped from the result. Passing false keeps it,
which is what Polars does and what you want when the null is a real category. The
group is formed either way: the flag only decides whether its row survives.

**Group order.** `sort` defaults to true and orders the output rows by the key
columns ascending, which is pandas' default. The sort runs on the result, which
is one row per group, not on the input. Passing false leaves the groups in the
order they were first seen, which is what pandas calls `sort=False` and Polars
does by default, and it is free.
"""

from firepanda.kernel.group import AggKind


struct AggSpec(Copyable, Movable, Writable):
    """One output column: a reduction, the column it reads, and its name."""

    var column: String
    """The column to aggregate. Must exist in the frame and is allowed to be a
    key column, because `group_by(["k"], [AggSpec("k", AggKind.COUNT)])` is a
    reasonable thing to ask for."""

    var kind: AggKind
    """Which reduction."""

    var other: String
    """The second column, for the reductions that read a pair of them. Empty for
    every other kind, and `AggKind.reads_two_columns` is what says which."""

    var name: String
    """The output column name. Empty means derive it, which gives `x_sum`."""

    def __init__(out self, column: String, kind: AggKind, name: String = ""):
        """Constructs a spec.

        Args:
            column: The column to reduce.
            kind: The reduction.
            name: The output name, or empty to derive one from the other two.
        """
        self.column = column
        self.other = ""
        self.kind = kind
        self.name = name

    def __init__(
        out self,
        column: String,
        other: String,
        kind: AggKind,
        name: String = "",
    ):
        """Constructs a spec for a reduction that reads two columns.

        `CORR` and `COV` are statements about a pair of columns rather than about
        one, which is the only reason `AggSpec` has a second column at all. They
        live here rather than in a method of their own on `DataFrame` so that a
        correlation composes with the other reductions in one call: asking for a
        correlation and three sums over the same keys should group the rows once,
        and it does.

        Args:
            column: The first column.
            other: The second.
            kind: The reduction, which must be one that reads two columns.
            name: The output name, or empty to derive one.
        """
        self.column = column
        self.other = other
        self.kind = kind
        self.name = name

    def output_name(self) -> String:
        """Returns the name this spec's column will have in the result.

        Deriving `x_sum` rather than reusing `x` is on purpose. Two reductions of
        the same column in one call is the normal case, and pandas handles the
        collision with a second level of column index, which firepanda does not
        have and does not intend to grow. A suffix is the flat way to say the same
        thing, and an explicit `name` overrides it.

        Returns:
            The explicit name, or the column and the reduction joined by an
            underscore.
        """
        if self.name != "":
            return self.name
        if self.kind.reads_two_columns():
            return String(self.column, "_", self.other, "_", self.kind)
        return String(self.column, "_", self.kind)

    def write_to(self, mut writer: Some[Writer]):
        """Writes the spec as it would be read back.

        Args:
            writer: The sink.
        """
        writer.write(self.kind, "(", self.column)
        if self.kind.reads_two_columns():
            writer.write(", ", self.other)
        writer.write(") as ", self.output_name())
