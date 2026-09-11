"""The binding table, and the generator that turns it into the three files it feeds.

Every public entry point crosses the Mojo to Python boundary exactly once and is
described here exactly once. Running this script writes:

    firepanda/py/_registration.mojo   the `def_function` and `def_method` calls
    python/firepanda/_frame.py        the Python classes users actually hold
    python/firepanda/_firepanda.pyi   stubs for the private extension module

Run `python tools/bindings.py` to write them and `python tools/bindings.py
--check` to fail if what is on disk is not what this file says, which is what CI
does. Nothing generated should ever be edited by hand.

### Why a generator rather than a table Mojo walks

Document 07 section 3 asks for one declarative table that the registration comes
out of, and the obvious reading of that is a Mojo value with a `@parameter for`
over it. That does not compile, and document 13 section 7 has the details.
`def_method` marks its function type parameter inferred only, so the concrete
function type has to be recoverable at the call site, and it does not survive
passing through anything generic. A `PyObjectFunction` built where the function
is named can be forwarded, but a collection of them cannot exist at all, because
every one has a different type and a variadic pack over them is rejected in the
parameter list before the body is looked at.

So the registration is a flat sequence of calls and cannot be anything else. The
property document 07 wanted survives anyway: an upstream change to the binding
API is still one file, and that file is this one.

### Why the Python class is generated too

`PythonTypeBuilder` can attach methods and nothing else, so `df["a"]`, `len(df)`
and `df.shape` are not expressible in Mojo, and 28 percent of the pandas surface
is properties and operators. That is measured in document 13. The pandas API
therefore lives in Python and the Mojo bindings are a private calling convention
underneath it, which means the two most easily divergent files in the project sit
on either side of the boundary. Generating both from one table is what keeps them
in agreement, and the parity tests in `python/tests/test_bindings.py` are what
prove it stayed true.

Every member in the table today is a plain delegation and the generated file is
entirely mechanical. The first member that needs real logic should go in a hand
written mixin that the generated class inherits from, rather than being smuggled
into the table as an expression, because the moment the table starts carrying
code it stops being reviewable as a table.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _docstring(text: str, indent: str) -> list[str]:
    """Wraps a docstring so the generated file stays under the line limit.

    A blank line in the text is a paragraph break and survives the wrapping. Most
    docstrings in the table are one sentence and never reach it, but a class that
    has to explain a decision needs a summary line and then the explanation, which
    is also what the docstring linter expects.

    Args:
        text: The docstring, with a blank line between paragraphs and each
            paragraph on one long line.
        indent: The indent to put in front of every line.

    Returns:
        The lines to emit.
    """
    width = 88 - len(indent)
    if len(text) <= width and "\n" not in text:
        return [f'{indent}"""{text}"""']
    lines: list[str] = []
    for at, paragraph in enumerate(text.split("\n\n")):
        if at:
            lines.append("")
        current = ""
        for word in paragraph.split():
            if current and len(current) + 1 + len(word) > width:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        lines.append(current)
    body = [indent + line if line else "" for line in lines[1:]]
    return [f'{indent}"""{lines[0]}', *body, f'{indent}"""']


def _guarded(statement: str, indent: str) -> list[str]:
    """Wraps one delegating statement in the error translation.

    Every generated member gets this and none of them gets to opt out, because
    the one that opts out is the one that hands a user a bare `Exception` with a
    `firepanda:column:` prefix still on the front of it.

    `from None` rather than `from error`. The error being suppressed is the
    binding layer's own untyped wrapper around a message this library wrote, so
    a chained traceback would show the same sentence twice and call the second
    one the direct cause of the first.

    Args:
        statement: The `return ...` line to guard.
        indent: The indent of the method body.

    Returns:
        The lines to emit.
    """
    return (
        [f"{indent}try:"]
        + _python_statement(statement, f"{indent}    ")
        + [
            f"{indent}except Exception as error:",
            f"{indent}    raise translate(error) from None",
        ]
    )


def _python_statement(statement: str, indent: str) -> list[str]:
    """Writes one delegating call the way `ruff format` would have written it.

    Same problem as `_python_def` and the same reason for solving it here, which
    is that the generated files are format checked and a statement that is merely
    valid is not enough. This one only knows the two layouts these statements
    ever need, all on one line and the arguments together on one continuation
    line, because every one of them is a single call with a single set of
    brackets. A statement that needs the third layout is a body that has grown
    past what the generator should be laying out, so it says so rather than
    writing something the formatter will then rewrite.

    Args:
        statement: The statement, which is one call.
        indent: The indent it sits at.

    Returns:
        The lines to emit.
    """
    one = f"{indent}{statement}"
    if len(one) <= PYTHON_COLUMNS:
        return [one]
    head, _, rest = statement.partition("(")
    close = rest.rfind(")")
    inside = f"{indent}    {rest[:close]}"
    if close < 0 or len(inside) > PYTHON_COLUMNS:
        raise SystemExit(
            f"the generated statement `{statement}` is too long for the"
            " generator to lay out the way ruff format wants, which needs its"
            f" arguments to fit on one line at an indent of {len(indent) + 4}."
            " Shorten the names it calls."
        )
    return [f"{indent}{head}(", inside, f"{indent}{rest[close:]}"]


@dataclass(frozen=True)
class Binding:
    """One callable on the extension side.

    This is the narrow convention, not the pandas API. Names here are chosen to
    be unambiguous rather than familiar, because nothing outside the generated
    Python layer ever calls them.
    """

    mojo: str
    """The Mojo callable, as it is spelled in a Mojo import, such as
    `PyDataFrame.length`."""

    name: str
    """The name it is registered under on the extension side."""

    doc: str
    """The docstring the extension carries. One line, ending in a full stop."""

    params: tuple[tuple[str, str], ...] = ()
    """Parameters after `py_self`, as name and annotation pairs, used for the
    stub signature. At most seven, which is the ceiling document 13 section 4
    measured."""

    returns: str = "object"
    """The stub return annotation. Everything really crosses as a
    `PythonObject`, so this is a claim about what the Mojo body puts in it
    rather than something the boundary enforces, and it is the claim
    `mypy --strict` then holds the Python layer to."""

    py_params: tuple[tuple[str, str], ...] = ()
    """The Python facing parameter list for a module level function, when it
    differs from the extension one. pandas calls the first argument of
    `read_csv` `filepath_or_buffer` and the parity test checks that we do too,
    while the extension side keeps a plainer name. Empty means the two agree."""


@dataclass(frozen=True)
class Member:
    """One member on the Python side, as a pandas user meets it."""

    name: str
    """The Python name, which may be a dunder."""

    kind: str
    """One of `method`, `property` or `dunder`."""

    body: str
    """The expression the member returns, written against `self._inner` and the
    parameter names below."""

    signature: str = ""
    """The parameter list after `self`, verbatim, including defaults and
    annotations. Empty for a property."""

    doc: str = ""
    """The docstring. Should say what pandas says, since this is the surface
    being copied."""

    returns: str = "object"
    """The return annotation. `mypy --strict` runs over the generated file, so
    every member needs one."""

    wraps: str = ""
    """The Python class to wrap the result in, when the result is another
    extension object. Empty means the result crosses as it is. It is a name
    rather than a flag because a frame method can hand back a series, so the
    class to wrap in is not always the class the method is on."""


@dataclass(frozen=True)
class Exposed:
    """One extension type, with both halves of it."""

    mojo: str
    """The Mojo struct name."""

    name: str
    """The name on the extension side."""

    py: str
    """The class name on the Python side, which is the pandas name."""

    doc: str
    """The Python class docstring."""

    init: str | None = None
    """The Mojo `py_init`, or None for a type Python cannot construct."""

    init_params: tuple[tuple[str, str], ...] = ()
    """The extension constructor's parameters. Narrower than the pandas one on
    purpose, because the Python layer turns the pandas call into this one."""

    constructed: bool = False
    """Whether the mixin writes `__init__`. A type that a user constructs has to
    expose the pandas constructor signature, and that cannot also be the internal
    hand off that puts a wrapper around an extension object, so the generator
    emits `_wrap` for the internal one and stays out of the way of the public
    one. Document 18 section 5."""

    bindings: tuple[Binding, ...] = ()
    """The methods on the extension side."""

    members: tuple[Member, ...] = ()
    """The members on the Python side."""

    module: str = "firepanda.py.frame"
    """The Mojo module the struct is defined in. The registration imports from
    here, and a type that lives in its own file rather than next to the frame
    says so instead of relying on the frame re-exporting it."""

    mixin: str = ""
    """A hand written base class in `python/firepanda/_pandas.py` for the members
    that are not a plain delegation. The note at the top of this file asks for
    exactly this rather than for expressions in the table growing logic, and
    `DataFrame.__getitem__` is the member that reached it: what `df[key]` does
    depends on what `key` is, and a conditional smuggled into a `body` string
    would be code in a table."""


ARITHMETIC: tuple[tuple[str, str], ...] = (
    ("add", "+"),
    ("sub", "-"),
    ("mul", "*"),
    ("truediv", "/"),
    ("floordiv", "//"),
    ("mod", "%"),
    ("pow", "**"),
)
"""The seven arithmetic operations, as pandas names them and as Python spells
them. The name is what crosses the boundary and the symbol is only ever used in a
docstring."""

COMPARISON: tuple[tuple[str, str], ...] = (
    ("eq", "=="),
    ("ne", "!="),
    ("lt", "<"),
    ("le", "<="),
    ("gt", ">"),
    ("ge", ">="),
)
"""The six comparisons, the same way."""

UNARY: tuple[tuple[str, str, str], ...] = (
    ("__neg__", "neg", "`-a`, which on a boolean column is the logical not."),
    ("__pos__", "pos", "`+a`, which copies and refuses a boolean column, as pandas does."),
    ("__abs__", "abs", "`abs(a)`, so the builtin works."),
    ("__invert__", "invert", "`~a`, the bitwise not, which needs an integer or a boolean."),
)
"""The four unary operations, as Python names them, as the boundary does and as
they read to somebody who has not read the kernel."""


def _operators(py: str) -> tuple[Member, ...]:
    """Writes the arithmetic and comparison members for one class.

    This is the one place in this file that builds rows in a loop rather than
    writing them out, and it is worth saying why, because the note at the top
    asks that the table stay a table. There are ninety four of these members
    between the two classes and they differ from each other in three letters. A
    literal table of ninety four rows is not more reviewable than fourteen names
    and a shape, it is less: nobody reads ninety four near identical rows closely
    enough to notice that one of them says `sub` where it means `rsub`, and a
    generator cannot make that mistake at all.

    What the loop is not allowed to do is decide behaviour. Every member here is
    still one expression against a mixin helper, the difference between a frame
    and a series is a parameter order rather than a branch, and the three
    behaviours `fill_value` has live in `_pandas.py` where they can be read.

    Args:
        py: The class name, `DataFrame` or `Series`, which is also what a member
            returns and how its signature is ordered.

    Returns:
        The members, in the order they should be written out.
    """
    thing = "frame" if py == "DataFrame" else "series"
    operands = "a frame, a series or a constant" if py == "DataFrame" else "a series or a constant"
    out: list[Member] = []

    for name, symbol in ARITHMETIC:
        out.append(
            Member(
                name=f"__{name}__",
                kind="dunder",
                signature="other: Any",
                body=f'self._operator(other, "{name}", False, False)',
                doc=f"`a {symbol} b`, against {operands}.",
                returns="Any",
            )
        )
        out.append(
            Member(
                name=f"__r{name}__",
                kind="dunder",
                signature="other: Any",
                body=f'self._operator(other, "{name}", True, False)',
                doc=f"`b {symbol} a`, which is what Python calls when the left side declines.",
                returns="Any",
            )
        )

    # No reflected forms here. Python has no `__req__`: a comparison reflects
    # onto its mirror image, so `a == b` falls back to `b == a` and `a < b` falls
    # back to `b > a`, and both of those are members this already writes.
    for name, symbol in COMPARISON:
        out.append(
            Member(
                name=f"__{name}__",
                kind="dunder",
                signature="other: Any",
                body=f'self._operator(other, "{name}", False, True)',
                doc=f"`a {symbol} b`, which refuses two {thing}s that are not labelled the same.",
                returns="Any",
            )
        )

    for name, _, doc in UNARY:
        out.append(
            Member(
                name=name,
                kind="dunder",
                body=f'self._unary("{name.strip("_")}")',
                doc=doc,
                returns="Any",
            )
        )
    out.append(
        Member(
            name="abs",
            kind="method",
            body='self._unary("abs")',
            doc="Every value with its sign removed.",
            returns="Any",
        )
    )

    for name, symbol in ARITHMETIC:
        for prefix, side in (("", "b"), ("r", "a")):
            flip = "True" if prefix else "False"
            out.append(
                Member(
                    name=f"{prefix}{name}",
                    kind="method",
                    signature=_named_signature(py, fill_value=True),
                    body=f'self._named(other, "{name}", axis, level, fill_value, {flip})',
                    doc=(
                        f"`{'a' if side == 'b' else 'b'} {symbol} {side}`, by"
                        " name, so it can take a fill value."
                    ),
                    returns="Any",
                )
            )

    for name, symbol in COMPARISON:
        out.append(
            Member(
                name=name,
                kind="method",
                signature=_named_signature(py, fill_value=py == "Series"),
                body=(
                    f'self._named(other, "{name}", axis, level, fill_value, False)'
                    if py == "Series"
                    else f'self._named(other, "{name}", axis, level, None, False)'
                ),
                doc=f"`a {symbol} b`, by name, which aligns where the operator refuses to.",
                returns="Any",
            )
        )

    if py == "Series":
        for prefix, flip in (("", "False"), ("r", "True")):
            out.append(
                Member(
                    name=f"__{prefix}divmod__",
                    kind="dunder",
                    signature="other: Any",
                    body=f"self._divmod(other, 0, None, None, {flip})",
                    doc="The floor division and the remainder, as a pair.",
                    returns="Any",
                )
            )
            out.append(
                Member(
                    name=f"{prefix}divmod",
                    kind="method",
                    signature=_named_signature(py, fill_value=True),
                    body=f"self._divmod(other, axis, level, fill_value, {flip})",
                    doc="The floor division and the remainder, as a pair, by name.",
                    returns="Any",
                )
            )

    return tuple(out)


def _named_signature(py: str, fill_value: bool) -> str:
    """Writes the parameter list of a named form, in the order pandas has it.

    The two classes order these differently and the difference is not cosmetic,
    because the signature parity test compares parameter names in order against
    a running pandas. A frame puts `axis` first and defaults it to the string
    `columns`, a series puts it last and defaults it to `0`.

    Args:
        py: The class name.
        fill_value: Whether the form takes one. Every arithmetic form does. A
            comparison does on a series and does not on a frame, which is pandas'
            own split rather than something chosen here.

    Returns:
        The parameter list after `self`.
    """
    if py == "DataFrame":
        parts = ["other: Any", 'axis: Any = "columns"', "level: Any = None"]
        if fill_value:
            parts.append("fill_value: Any = None")
        return ", ".join(parts)
    parts = ["other: Any", "level: Any = None"]
    if fill_value:
        parts.append("fill_value: Any = None")
    parts.append("axis: Any = 0")
    return ", ".join(parts)


PLAIN: tuple[tuple[str, str], ...] = (
    ("mean", "The average of the values."),
    ("min", "The smallest value."),
    ("max", "The largest value."),
    ("median", "The middle value."),
    ("skew", "The unbiased skew, normalised by N-1."),
)
"""The reductions whose only arguments are the four every reduction has."""

SPREAD: tuple[tuple[str, str], ...] = (
    ("std", "The sample standard deviation, normalised by N-1 by default."),
    ("var", "The unbiased variance, normalised by N-1 by default."),
    ("sem", "The unbiased standard error of the mean, normalised by N-1 by default."),
)
"""The three that also take a delta degrees of freedom."""


def _reductions(py: str) -> tuple[Member, ...]:
    """Writes the twelve reduction members for one class.

    A loop for the same reason `_operators` is one, and under the same
    restriction: what varies between these rows is a word and a parameter list,
    and nothing here decides what a reduction does. The word crosses the boundary
    and `firepanda/py/reduce.mojo` reads it, the arguments are checked in
    `_pandas.py`, and every body below is one call to a mixin helper.

    The signatures are pandas' own, measured rather than copied from the
    documentation, and the two classes differ in ways that are not cosmetic. A
    series returns a value and a frame returns a series of them. `quantile` and
    `nunique` and `count` take positional arguments while the other nine are
    keyword only. A frame's `quantile` takes two parameters a series' does not.
    The signature parity test compares the whole list in order, so each of those
    is written out rather than shared.

    Args:
        py: The class name, `DataFrame` or `Series`.

    Returns:
        The members, in the order they should be written out.
    """
    frame = py == "DataFrame"
    over = "column" if frame else "row"
    gives = "Series" if frame else "Any"
    plural = "One value per column." if frame else ""
    tail = ", **kwargs: Any"
    out: list[Member] = []

    for name, what in PLAIN + (("sum", "The sum of the values."),):
        start = "0" if frame or name != "sum" else "None"
        parts = [
            "*",
            f"axis: Any = {start}",
            "skipna: bool = True",
            "numeric_only: bool = False",
        ]
        if name == "sum":
            parts.append("min_count: int = 0")
        count = "min_count" if name == "sum" else "0"
        out.append(
            Member(
                name=name,
                kind="method",
                signature=", ".join(parts) + tail,
                body=f'self._reduce("{name}", 0.0, axis, skipna, numeric_only, {count})',
                doc=f"{what} Over the {over}s. {plural}".strip(),
                returns=gives,
            )
        )

    for name, what in SPREAD:
        out.append(
            Member(
                name=name,
                kind="method",
                signature=(
                    f"*, axis: Any = {'0' if frame else 'None'}, skipna: bool = True,"
                    " ddof: int = 1, numeric_only: bool = False" + tail
                ),
                body=f'self._reduce("{name}", float(ddof), axis, skipna, numeric_only, 0)',
                doc=f"{what} Over the {over}s. {plural}".strip(),
                returns=gives,
            )
        )

    quantile = (
        "q: Any = 0.5, axis: Any = 0, numeric_only: bool = False,"
        ' interpolation: str = "linear", method: str = "single"'
        if frame
        else 'q: Any = 0.5, interpolation: str = "linear"'
    )
    body = (
        "self._quantile(q, axis, numeric_only, interpolation, method)"
        if frame
        else "self._quantile(q, interpolation)"
    )
    out.append(
        Member(
            name="quantile",
            kind="method",
            signature=quantile,
            body=body,
            doc=f"The value at the given quantile. Over the {over}s. {plural}".strip(),
            returns=gives,
        )
    )

    out.append(
        Member(
            name="nunique",
            kind="method",
            signature="axis: Any = 0, dropna: bool = True" if frame else "dropna: bool = True",
            body="self._nunique(axis, dropna)" if frame else "self._nunique(0, dropna)",
            doc=f"How many distinct values there are. Over the {over}s. {plural}".strip(),
            returns=gives,
        )
    )

    if frame:
        out.append(
            Member(
                name="count",
                kind="method",
                signature="axis: Any = 0, numeric_only: bool = False",
                body='self._reduce("count", 0.0, axis, True, numeric_only, 0)',
                doc="How many values are not missing. Over the columns. One value per column.",
                returns="Series",
            )
        )

    return tuple(out)


CUMULATIVE: tuple[tuple[str, str], ...] = (
    ("cumsum", "The running total, where row i holds the sum of every row up to i."),
    ("cumprod", "The running product."),
    ("cummax", "The largest value seen so far."),
    ("cummin", "The smallest value seen so far."),
)
"""The four scans, which differ only in the operator they fold with."""


DT_PARTS: tuple[tuple[str, str, str], ...] = (
    ("year", "year", "The calendar year of every row."),
    ("month", "month", "The month of every row, 1 for January."),
    ("day", "day", "The day of the month of every row."),
    ("hour", "hour", "The hour on a twenty four hour clock."),
    ("minute", "minute", "The minute of the hour."),
    ("second", "second", "The second of the minute."),
    ("microsecond", "microsecond", "The microseconds past the second."),
    ("nanosecond", "nanosecond", "The nanoseconds past the microsecond."),
    ("dayofweek", "dayofweek", "The day of the week, 0 for Monday."),
    ("day_of_week", "dayofweek", "The day of the week, 0 for Monday. The same as dayofweek."),
    ("weekday", "dayofweek", "The day of the week, 0 for Monday. The same as dayofweek."),
    ("dayofyear", "dayofyear", "The day of the year, 1 for the first of January."),
    ("day_of_year", "dayofyear", "The day of the year. The same as dayofyear."),
    ("quarter", "quarter", "The quarter of the year, 1 to 4."),
    ("days_in_month", "days_in_month", "How many days the row's month has."),
    (
        "daysinmonth",
        "days_in_month",
        "How many days the row's month has. The same as days_in_month.",
    ),
    ("is_leap_year", "is_leap_year", "Whether the row's year has a twenty ninth of February."),
    ("is_month_start", "is_month_start", "Whether the row is the first day of its month."),
    ("is_month_end", "is_month_end", "Whether the row is the last day of its month."),
    ("is_quarter_start", "is_quarter_start", "Whether the row is the first day of its quarter."),
    ("is_quarter_end", "is_quarter_end", "Whether the row is the last day of its quarter."),
    ("is_year_start", "is_year_start", "Whether the row is the first day of its year."),
    ("is_year_end", "is_year_end", "Whether the row is the last day of its year."),
    ("date", "date", "The date part, with the clock dropped."),
    ("days", "days", "The whole days in each span, floored, for a duration column."),
)
"""The parts of `dt` that take nothing and answer a column, as pandas names them
and as the boundary does.

The two spellings are not always the same word and that is the whole reason for a
second column. pandas has `dayofweek` and `day_of_week` and `weekday` for one
field and `days_in_month` and `daysinmonth` for another, and the core has one
name for each because a kernel does not need three. Putting the aliases here
rather than in the Mojo keeps the pandas surface in the one file that is about
the pandas surface.

`normalize` is not in here even though it takes nothing, because it is a method
in pandas and everything in this table is a property, and `total_seconds` is not
in here for the same reason."""


def _datetime_members() -> tuple[Member, ...]:
    """Writes the members of the `dt` accessor.

    Nothing here decides what a part means. The word crosses the boundary,
    `firepanda/py/temporal.mojo` reads it, and every body is one call to a mixin
    helper, which is the same restriction `_reductions` and `_transformations`
    work under and for the same reason.

    The order is the parts first, then the two that answer a word, then the
    methods, which is roughly how the pandas documentation lists them and is the
    order somebody comparing the two would read them in.

    Returns:
        The members, in the order they should be written out.
    """
    out: list[Member] = []
    for name, crosses, what in DT_PARTS:
        out.append(
            Member(
                name=name,
                kind="property",
                body=f'self._part("{crosses}", "")',
                doc=what,
                returns="Series",
            )
        )

    out.append(
        Member(
            name="tz",
            kind="property",
            body="self._zone()",
            doc="The clock the column is read against, or None when it carries no zone.",
            returns="str | None",
        )
    )
    out.append(
        Member(
            name="unit",
            kind="property",
            body="self._resolution()",
            doc="The resolution the column is stored in, one of s, ms, us and ns.",
            returns="str",
        )
    )

    # Two names that take nothing and are still methods rather than properties,
    # which is why they are not in the table above. pandas spells both with an
    # empty parameter list, checked against a running one.
    out.append(
        Member(
            name="normalize",
            kind="method",
            signature="",
            body='self._part("normalize", "")',
            doc="Every clock moved back to midnight, keeping the timestamp type.",
            returns="Series",
        )
    )
    out.append(
        Member(
            name="total_seconds",
            kind="method",
            signature="",
            body='self._part("total_seconds", "")',
            doc="Each span as a number of seconds, for a duration column.",
            returns="Series",
        )
    )

    for name, what in (
        ("floor", "down to"),
        ("ceil", "up to"),
        ("round", "to the nearest"),
    ):
        out.append(
            Member(
                name=name,
                kind="method",
                signature='freq: Any, ambiguous: Any = "raise", nonexistent: Any = "raise"',
                body=f'self._rounded("{name}", freq, ambiguous, nonexistent)',
                doc=f"Every clock moved {what} the given frequency.",
                returns="Series",
            )
        )

    out.append(
        Member(
            name="as_unit",
            kind="method",
            signature="unit: str, round_ok: bool = True",
            body="self._as_unit(unit, round_ok)",
            doc="The column stored in another resolution.",
            returns="Series",
        )
    )

    for name, what in (
        ("day_name", "The name of the day of the week of every row."),
        ("month_name", "The name of the month of every row."),
    ):
        out.append(
            Member(
                name=name,
                kind="method",
                signature="locale: Any = None",
                body=f'self._named("{name}", locale)',
                doc=what,
                returns="Series",
            )
        )

    out.append(
        Member(
            name="strftime",
            kind="method",
            signature="date_format: str",
            body='self._part("strftime", date_format)',
            doc="Every row written out as text, in the given format.",
            returns="Series",
        )
    )
    out.append(
        Member(
            name="tz_convert",
            kind="method",
            signature="tz: Any",
            body="self._tz_convert(tz)",
            doc="The same instants read against another clock.",
            returns="Series",
        )
    )
    out.append(
        Member(
            name="tz_localize",
            kind="method",
            signature='tz: Any, ambiguous: Any = "raise", nonexistent: Any = "raise"',
            body="self._tz_localize(tz, ambiguous, nonexistent)",
            doc="The same readings put on a clock, or taken off one when tz is None.",
            returns="Series",
        )
    )
    out.append(
        Member(
            name="isocalendar",
            kind="method",
            signature="",
            body="self._isocalendar()",
            doc="The ISO 8601 year, week and day of every row, as a frame.",
            returns="DataFrame",
        )
    )
    return tuple(out)


def _string_members() -> tuple[Member, ...]:
    """Writes the members of the `str` accessor.

    Twenty one of pandas' fifty seven, in two groups. Twelve of them have one
    idea between them, which is that a position in a string is a character
    rather than a byte, and the other nine have a second, which is that the two
    ends of a row can be trimmed or padded without anything looking at the
    middle. The rest of the accessor is case conversion, the predicates,
    splitting and the regex methods, and each of those groups has an idea of its
    own that is worth landing on its own.

    `center`, `ljust` and `rjust` are `pad` with the side already chosen, and
    they are written out rather than being left to the caller because pandas
    writes them out. They cost a line each here and they are three names on the
    board.

    `index` and `rindex` are here without being in the extension, because they
    are `find` and `rfind` that raise rather than answering -1, and where that
    exception is thrown is a pandas question rather than a kernel one.

    Returns:
        The members, in the order they should be written out.
    """
    return (
        Member(
            name="len",
            kind="method",
            signature="",
            body='self._number("len")',
            doc="How many characters each row holds.",
            returns="Series",
        ),
        Member(
            name="slice",
            kind="method",
            signature="start: Any = None, stop: Any = None, step: Any = None",
            body="self._sliced(start, stop, step)",
            doc="A range of characters out of every row, under Python's slice rules.",
            returns="Series",
        ),
        Member(
            name="slice_replace",
            kind="method",
            signature="start: Any = None, stop: Any = None, repl: Any = None",
            body="self._replaced_slice(start, stop, repl)",
            doc="Every row with a range of characters swapped for a string.",
            returns="Series",
        ),
        Member(
            name="get",
            kind="method",
            signature="i: Any",
            body="self._at(i)",
            doc="One character out of every row, and nothing where the row is too short.",
            returns="Series",
        ),
        Member(
            name="find",
            kind="method",
            signature="sub: Any, start: Any = 0, end: Any = None",
            body='self._found("find", sub, start, end)',
            doc="Where a substring first sits in every row, or -1 where it is absent.",
            returns="Series",
        ),
        Member(
            name="rfind",
            kind="method",
            signature="sub: Any, start: Any = 0, end: Any = None",
            body='self._found("rfind", sub, start, end)',
            doc="Where a substring last sits in every row, or -1 where it is absent.",
            returns="Series",
        ),
        Member(
            name="index",
            kind="method",
            signature="sub: Any, start: Any = 0, end: Any = None",
            body='self._demanded("find", sub, start, end)',
            doc="The same as find, except that a row without the substring is an error.",
            returns="Series",
        ),
        Member(
            name="rindex",
            kind="method",
            signature="sub: Any, start: Any = 0, end: Any = None",
            body='self._demanded("rfind", sub, start, end)',
            doc="The same as rfind, except that a row without the substring is an error.",
            returns="Series",
        ),
        Member(
            name="startswith",
            kind="method",
            signature="pat: Any, na: Any = None",
            body='self._begins("startswith", pat, na)',
            doc="Whether every row begins with a string, or with any of several.",
            returns="Series",
        ),
        Member(
            name="endswith",
            kind="method",
            signature="pat: Any, na: Any = None",
            body='self._begins("endswith", pat, na)',
            doc="Whether every row ends with a string, or with any of several.",
            returns="Series",
        ),
        Member(
            name="removeprefix",
            kind="method",
            signature="prefix: Any",
            body='self._text("removeprefix", prefix)',
            doc="Every row with a leading string taken off, if it has one.",
            returns="Series",
        ),
        Member(
            name="removesuffix",
            kind="method",
            signature="suffix: Any",
            body='self._text("removesuffix", suffix)',
            doc="Every row with a trailing string taken off, if it has one.",
            returns="Series",
        ),
        Member(
            name="strip",
            kind="method",
            signature="to_strip: Any = None",
            body='self._trimmed("strip", to_strip)',
            doc="Every row with characters taken off both ends, whitespace by default.",
            returns="Series",
        ),
        Member(
            name="lstrip",
            kind="method",
            signature="to_strip: Any = None",
            body='self._trimmed("lstrip", to_strip)',
            doc="Every row with characters taken off the near end.",
            returns="Series",
        ),
        Member(
            name="rstrip",
            kind="method",
            signature="to_strip: Any = None",
            body='self._trimmed("rstrip", to_strip)',
            doc="Every row with characters taken off the far end.",
            returns="Series",
        ),
        Member(
            name="pad",
            kind="method",
            signature='width: Any, side: Any = "left", fillchar: Any = " "',
            body="self._padded(width, side, fillchar)",
            doc="Every row filled out to a width with a character, on the side named.",
            returns="Series",
        ),
        Member(
            name="center",
            kind="method",
            signature='width: Any, fillchar: Any = " "',
            body='self._padded(width, "both", fillchar)',
            doc="Every row filled out to a width from both ends at once.",
            returns="Series",
        ),
        Member(
            name="ljust",
            kind="method",
            signature='width: Any, fillchar: Any = " "',
            body='self._padded(width, "right", fillchar)',
            doc="Every row filled out to a width on the far end.",
            returns="Series",
        ),
        Member(
            name="rjust",
            kind="method",
            signature='width: Any, fillchar: Any = " "',
            body='self._padded(width, "left", fillchar)',
            doc="Every row filled out to a width on the near end.",
            returns="Series",
        ),
        Member(
            name="zfill",
            kind="method",
            signature="width: Any",
            body="self._filled(width)",
            doc="Every row filled out to a width with zeros, after any leading sign.",
            returns="Series",
        ),
        Member(
            name="repeat",
            kind="method",
            signature="repeats: Any",
            body="self._repeated(repeats)",
            doc="Every row written out several times, end to end.",
            returns="Series",
        ),
    )


def _categorical_members() -> tuple[Member, ...]:
    """Writes the members of the `cat` accessor.

    Eleven names, and three doors under them. The three are in the extension and
    the other eight are arithmetic over them, which is all in the mixin: what
    counts as adding a category, what counts as removing one, and which of the
    disagreements are a `ValueError` rather than a quiet answer are questions
    about the pandas surface and belong on the pandas side of the boundary.

    The order is the three that answer a value first and then the methods, which
    is how the pandas documentation lists them.

    Returns:
        The members, in the order they should be written out.
    """
    return (
        Member(
            name="categories",
            kind="property",
            body="self._levels()",
            doc="The categories, in the order the column holds them.",
            returns="Index",
        ),
        Member(
            name="ordered",
            kind="property",
            body="self._ordered()",
            doc="Whether comparing two of the categories means anything.",
            returns="bool",
        ),
        Member(
            name="codes",
            kind="property",
            body="self._codes()",
            doc="Which category each row holds, as positions into the categories.",
            returns="Series",
        ),
        Member(
            name="as_ordered",
            kind="method",
            signature="",
            body="self._with_order(True)",
            doc="The same column, with the category order made to mean something.",
            returns="Series",
        ),
        Member(
            name="as_unordered",
            kind="method",
            signature="",
            body="self._with_order(False)",
            doc="The same column, with the category order made to mean nothing.",
            returns="Series",
        ),
        Member(
            name="add_categories",
            kind="method",
            signature="new_categories: Any",
            body="self._added(new_categories)",
            doc="The same values, over more categories than before.",
            returns="Series",
        ),
        Member(
            name="remove_categories",
            kind="method",
            signature="removals: Any",
            body="self._removed(removals)",
            doc="The same values, with the named categories gone and their rows missing.",
            returns="Series",
        ),
        Member(
            name="remove_unused_categories",
            kind="method",
            signature="",
            body="self._thinned()",
            doc="The same values, over only the categories that appear in them.",
            returns="Series",
        ),
        Member(
            name="rename_categories",
            kind="method",
            signature="new_categories: Any",
            body="self._renamed(new_categories)",
            doc="The same values under new labels, matched by position.",
            returns="Series",
        ),
        Member(
            name="reorder_categories",
            kind="method",
            signature="new_categories: Any, ordered: Any = None",
            body="self._reordered(new_categories, ordered)",
            doc="The same categories in another order.",
            returns="Series",
        ),
        Member(
            name="set_categories",
            kind="method",
            signature="new_categories: Any, ordered: Any = None, rename: bool = False",
            body="self._set(new_categories, ordered, rename)",
            doc="A new list of categories, with the values matched against it.",
            returns="Series",
        ),
    )


GROUPED: tuple[tuple[str, str, str], ...] = (
    ("sum", "The sum of the values in each group.", "sum"),
    ("mean", "The mean of the values in each group.", "mean"),
    ("min", "The smallest value in each group.", "extreme"),
    ("max", "The largest value in each group.", "extreme"),
    ("count", "How many values in each group are not missing.", "bare"),
    ("size", "How many rows are in each group, missing values included.", "bare"),
    ("first", "The first value in each group, in the frame's own order.", "pick"),
    ("last", "The last value in each group, in the frame's own order.", "pick"),
    ("median", "The middle value in each group.", "plain"),
    ("nunique", "How many distinct values are in each group.", "nunique"),
    ("std", "The standard deviation within each group.", "spread"),
    ("var", "The variance within each group.", "spread"),
    ("sem", "The standard error of the mean within each group.", "sem"),
    ("skew", "The skewness within each group.", "skew"),
    ("quantile", "The value at one quantile within each group.", "quantile"),
)
"""The fifteen grouped reductions, with the shape of each one's parameter list.

Ten shapes rather than fifteen signatures written out, because pandas gives the
same list to several of them and the difference between the lists is what the
parity test compares. Every signature below was measured against a running
pandas rather than copied out of the documentation, and the shapes exist to make
a mismatch a one line change instead of a hunt through fifteen strings.

`min` and `max` are `extreme` and `first` and `last` are `pick`, which are the
same three parameters except that the first pair also takes an engine and the
second pair does not. That is not a rule with a reason behind it, it is what
pandas has, and the two shapes are separate because the board measures the
difference and reported it the first time they were written as one.

`size` and `count` take nothing at all, which is worth naming: they are the two
that cannot be asked to skip a missing value, since one counts rows and the
other counts the values that are there, and pandas leaves the arguments off
rather than declaring them and ignoring them.
"""


WINDOWED: tuple[tuple[str, str, str, str], ...] = (
    ("sum", "The total of the values in the window.", "", ""),
    ("mean", "The mean of the values in the window.", "", ""),
    ("count", "How many rows of the window hold a value.", "", ""),
    ("min", "The smallest value in the window.", "", ""),
    ("max", "The largest value in the window.", "", ""),
    (
        "var",
        "The variance of the values in the window.",
        "ddof: int = 1",
        "self._spread_settings(ddof)",
    ),
    (
        "std",
        "The standard deviation of the values in the window.",
        "ddof: int = 1",
        "self._spread_settings(ddof)",
    ),
    (
        "sem",
        "The standard error of the mean of the values in the window.",
        "ddof: int = 1",
        "self._spread_settings(ddof)",
    ),
    ("skew", "The skewness of the values in the window.", "", ""),
    ("kurt", "The excess kurtosis of the values in the window.", "", ""),
    ("median", "The middle value of the window.", "", ""),
    (
        "quantile",
        "The value a fraction of the way through the sorted window.",
        'q: float, interpolation: str = "linear"',
        "self._quantile_settings(q, interpolation)",
    ),
    (
        "rank",
        "Where the value in the window's last row sits among the window's values.",
        'method: str = "average", ascending: bool = True, pct: bool = False',
        "self._rank_settings(method, ascending, pct)",
    ),
)
"""The thirteen reductions a window can be run through, and what each answers.

Thirteen rather than pandas' twenty six. Ten of them are the ones that can be
carried from one window to the next as a number or a state rather than
recomputed, and the other three read a position in the sorted window, which is
carried as a count of the window's values by rank instead.
`firepanda/kernel/window.mojo` says which of the other thirteen are still
waiting and why each of them is its own piece of work.

The third and fourth columns are the parameters the reduction reads and the
window does not. Five of the thirteen have some and the other eight have none,
which is one of the two places this table is not uniform, and `_window_members`
reads the columns rather than declaring arguments the other eight would have to
ignore. The other is the engine arguments, which six of the thirteen do not
declare. Both columns are written in the order pandas declares them, because
that order is the surface being matched.
"""


WINDOW_STATE: tuple[tuple[str, str, str, str], ...] = (
    (
        "window",
        "self._window",
        "int | None",
        "How many rows wide, and None for an expanding window.",
    ),
    (
        "min_periods",
        "self._min_periods",
        "int | None",
        "How many values a window needs before it answers, and None for the default.",
    ),
    (
        "center",
        "self._center",
        "bool",
        "Whether the window sits around its row rather than behind it.",
    ),
    (
        "closed",
        "self._closed",
        "str | None",
        "Which of the two ends the window keeps, and None for the default.",
    ),
    (
        "step",
        "self._step",
        "int | None",
        "How many rows apart the answered rows are, and None for every row.",
    ),
    (
        "obj",
        "self._data",
        "Series | DataFrame",
        "The column or the frame the windows are read out of.",
    ),
    (
        "ndim",
        "2 if self._over_frame() else 1",
        "int",
        "The number of dimensions of what is being windowed.",
    ),
    (
        "method",
        '"single"',
        "str",
        "Whether the columns are reduced together, which here they are not.",
    ),
    (
        "win_type",
        "None",
        "str | None",
        "The weighting over the window, which here is always none.",
    ),
    (
        "on",
        "None",
        "str | None",
        "The column the window is ordered by, which here is always the rows.",
    ),
    (
        "exclusions",
        "frozenset()",
        "frozenset[str]",
        "The columns held out of the reduction, which here is none of them.",
    ),
)
"""The eleven things a window object reports about itself, and how each is read.

pandas puts these on the window object and code in the wild reads them, mostly
to find out what a window it was handed is going to do before asking it to do
it. Six of the eleven are the arguments back, which is why `_hold` keeps them as
they arrived rather than resolved: `df.rolling(2).closed` is None in pandas and
answering `right` here would be reporting a decision rather than an argument.
The other five are constant, because they describe the choices this library has
made once rather than per window.
"""


def _window_members(py: str) -> tuple[Member, ...]:
    """Writes the eleven properties and thirteen reductions for one window class.

    Same restriction as `_group_members`, which is that nothing here decides
    what a reduction does. The word crosses the boundary and
    `firepanda/py/window.mojo` reads it, and every body is one call to a mixin
    helper that already holds where the window sits.

    The two classes take the same arguments and answer the same thing, and the
    only difference between them is which parameters their constructor accepted,
    so this is one function rather than two tables.

    Args:
        py: The class name, `Rolling` or `Expanding`, used in the sentences.

    Returns:
        The members, in the order they should be written out.
    """
    engines = "engine: Any = None, engine_kwargs: Any = None"
    over = "rolling" if py == "Rolling" else "expanding"
    out: list[Member] = []
    for name, body, returns, what in WINDOW_STATE:
        out.append(
            Member(
                name=name,
                kind="property",
                body=body,
                doc=what,
                returns=returns,
            )
        )
    for name, what, own, settings in WINDOWED:
        # Six of the thirteen take no engine arguments in pandas and so take
        # none here. `count` never had a numba path to choose. `sem` is written
        # in pandas as a deviation over a root count rather than as a kernel, so
        # there was never a path there either to offer. `skew` and `kurt` have
        # kernels and still declare nothing, which is pandas' own inconsistency
        # and is copied because the signature is the surface being matched.
        # `quantile` and `rank` declare neither either, and `median` declares
        # both, which is why the three of them do not travel together here.
        engined = name not in ("count", "sem", "skew", "kurt", "quantile", "rank")
        signature = "numeric_only: bool = False"
        if own:
            signature = f"{own}, {signature}"
        if engined:
            signature = f"{signature}, {engines}"
        arguments = "numeric_only"
        if engined:
            arguments = f"{arguments}, engine, engine_kwargs"
            if settings:
                arguments = f"{arguments}, {settings}"
        elif settings:
            arguments = f"{arguments}, settings={settings}"
        out.append(
            Member(
                name=name,
                kind="method",
                signature=signature,
                body=f'self._reduce("{name}", {arguments})',
                doc=f"{what} Over every {over} window.",
                returns="Series | DataFrame",
            )
        )
    return tuple(out)


EWM_REDUCED: tuple[tuple[str, str, str, str], ...] = (
    ("sum", "The weighted total of every row up to and including this one.", "", ""),
    ("mean", "The weighted mean of every row up to and including this one.", "", ""),
    (
        "var",
        "The weighted variance of every row up to and including this one.",
        "bias: bool = False",
        "self._bias_settings(bias)",
    ),
    (
        "std",
        "The weighted deviation of every row up to and including this one.",
        "bias: bool = False",
        "self._bias_settings(bias)",
    ),
)
"""The four reductions an exponentially weighted window can be run through.

Four of pandas' nine. `corr` and `cov` need a second column and are the same
piece of work as `Rolling.corr` and `Rolling.cov`, so all four of those should
land together. `agg` and `aggregate` take a caller's own function and belong with
the rolling ones. `online` is a streaming object pandas implements only in numba.
`firepanda/kernel/ewm.mojo` says the rest.

The third and fourth columns are read the way `WINDOWED`'s are, and here the
split is two and two rather than five and eight: `var` and `std` read a `bias`
flag and `mean` and `sum` read nothing. The engine arguments split the same two
and two and the other way round, which is pandas' own arrangement and not a
tidier one this table imposed.
"""


EWM_STATE: tuple[tuple[str, str, str, str], ...] = (
    ("com", "self._com", "float | None", "The centre of mass, if that is how the decay arrived."),
    ("span", "self._span", "float | None", "The span, if that is how the decay arrived."),
    (
        "halflife",
        "self._halflife",
        "float | None",
        "The half life in rows, if that is how the decay arrived.",
    ),
    (
        "alpha",
        "self._alpha",
        "float | None",
        "The smoothing factor, if that is how the decay arrived.",
    ),
    (
        "min_periods",
        "self._min_periods",
        "int",
        "How many values a row needs before it is answered, which is never below one.",
    ),
    ("adjust", "self._adjust", "bool", "Whether every row weighs one rather than the factor."),
    (
        "ignore_na",
        "self._ignore_na",
        "bool",
        "Whether a missing row is skipped rather than taking up a slot in the decay.",
    ),
    ("obj", "self._data", "Series | DataFrame", "The column or the frame the decay runs down."),
    ("ndim", "2 if self._over_frame() else 1", "int", "The number of dimensions of what is decaying."),
    (
        "method",
        '"single"',
        "str",
        "Whether the columns decay together, which here they do not.",
    ),
    ("times", "None", "object", "The instants the decay is measured against, which here is none."),
    ("window", "None", "int | None", "The width, which an exponentially weighted window has none of."),
    ("center", "False", "bool", "Whether the window sits around its row, which here it cannot."),
    ("closed", "None", "str | None", "Which ends the window keeps, of which it has none."),
    ("step", "None", "int | None", "How many rows apart the answers are, which here is every row."),
    ("win_type", "None", "str | None", "The weighting over the window, which here is the decay."),
    ("on", "None", "str | None", "The column the window is ordered by, which here is the rows."),
    (
        "exclusions",
        "frozenset()",
        "frozenset[str]",
        "The columns held out of the reduction, which here is none of them.",
    ),
)
"""The eighteen things an exponentially weighted window reports about itself.

The first seven are the arguments back, and the four spellings of the decay are
reported the way they arrived rather than collapsed, because `ewm(span=5).com` is
None in pandas and answering 2.0 here would be reporting a conversion rather than
an argument. `min_periods` is the exception and is reported resolved, because
pandas resolves it in the constructor: `ewm(span=5, min_periods=0).min_periods`
is 1 there.

The last eleven are constant, and seven of those are the rolling window's
vocabulary answered with nothing. That is not padding. A caller handed a window
object reads `w.window` to find out how wide it is, and an exponentially weighted
window has no width, so None is the true answer and an absent attribute is not.
"""


def _ewm_members() -> tuple[Member, ...]:
    """Writes the eighteen properties and four reductions for the EWM class.

    One function rather than a parameterised pair, because there is one class
    here. `Rolling` and `Expanding` are two classes over one kernel and this is
    one class over another, which is the shape `firepanda/kernel/ewm.mojo`
    argues for.

    Returns:
        The members, in the order they should be written out.
    """
    engines = "engine: Any = None, engine_kwargs: Any = None"
    out: list[Member] = []
    for name, body, returns, what in EWM_STATE:
        out.append(
            Member(name=name, kind="property", body=body, doc=what, returns=returns)
        )
    for name, what, own, settings in EWM_REDUCED:
        # `mean` and `sum` have a numba path in pandas to choose between and so
        # declare the two engine arguments. `var` and `std` have none and
        # declare neither, which is pandas' own split and is copied because the
        # signature is the surface being matched.
        engined = name in ("mean", "sum")
        signature = "numeric_only: bool = False"
        if own:
            signature = f"{own}, {signature}"
        if engined:
            signature = f"{signature}, {engines}"
        arguments = "numeric_only"
        if engined:
            arguments = f"{arguments}, engine, engine_kwargs"
        elif settings:
            arguments = f"{arguments}, settings={settings}"
        out.append(
            Member(
                name=name,
                kind="method",
                signature=signature,
                body=f'self._reduce("{name}", {arguments})',
                doc=f"{what} Under an exponentially weighted window.",
                returns="Series | DataFrame",
            )
        )
    return tuple(out)


def _group_members(py: str) -> tuple[Member, ...]:
    """Writes the fifteen reduction members for one group by class.

    Same restriction as `_reductions`, which is that nothing here decides what a
    reduction does. The word crosses the boundary and
    `firepanda/py/reduce.mojo` reads it, the declared arguments are refused in
    `_pandas.py`, and every body is one call to a mixin helper.

    The two classes take the same arguments in the same order and differ only in
    what they answer, which is why this is one function with a return annotation
    in it rather than two tables. A frame's group by hands back a frame and a
    column's hands back a column, except for `size` on a frame, which is a
    column because it produces one number per group rather than one per column.

    Args:
        py: The class name, `DataFrameGroupBy` or `SeriesGroupBy`.

    Returns:
        The members, in the order they should be written out.
    """
    frame = py == "DataFrameGroupBy"
    gives = "DataFrame" if frame else "DataFrame | Series"
    over = "every column that is not a key" if frame else "the column"
    engines = "engine: Any = None, engine_kwargs: Any = None"
    signatures = {
        "sum": f"numeric_only: bool = False, min_count: int = 0, skipna: bool = True, {engines}",
        "mean": f"numeric_only: bool = False, skipna: bool = True, {engines}",
        "extreme": (
            f"numeric_only: bool = False, min_count: int = -1, skipna: bool = True, {engines}"
        ),
        "pick": "numeric_only: bool = False, min_count: int = -1, skipna: bool = True",
        "bare": "",
        "plain": "numeric_only: bool = False, skipna: bool = True",
        "nunique": "dropna: bool = True",
        "spread": f"ddof: int = 1, {engines}, numeric_only: bool = False, skipna: bool = True",
        "sem": "ddof: int = 1, numeric_only: bool = False, skipna: bool = True",
        "skew": "skipna: bool = True, numeric_only: bool = False, **kwargs: Any",
        "quantile": 'q: Any = 0.5, interpolation: str = "linear", numeric_only: bool = False',
    }
    bodies = {
        "sum": 'self._reduce("sum", 0.0, numeric_only, skipna, min_count, engine, engine_kwargs)',
        "mean": 'self._reduce("mean", 0.0, numeric_only, skipna, None, engine, engine_kwargs)',
        "bare": 'self._reduce("{name}")',
        "extreme": (
            'self._reduce("{name}", 0.0, numeric_only, skipna, min_count, engine,'
            " engine_kwargs)"
        ),
        "pick": 'self._reduce("{name}", 0.0, numeric_only, skipna, min_count)',
        "plain": 'self._reduce("{name}", 0.0, numeric_only, skipna)',
        "nunique": "self._nunique(dropna)",
        "spread": 'self._spread("{name}", ddof, numeric_only, skipna, engine, engine_kwargs)',
        "sem": 'self._reduce("sem", float(ddof), numeric_only, skipna)',
        "skew": 'self._reduce("skew", 0.0, numeric_only, skipna)',
        "quantile": "self._quantile(q, interpolation, numeric_only)",
    }
    out: list[Member] = []
    for name, what, shape in GROUPED:
        # `size` counts rows rather than reducing a column, so it is one number
        # per group either way and has its own door on the frame's group by. It
        # is also the one whose sentence does not name a column, for the same
        # reason, so it does not get the clause saying which ones it reads.
        sized = name == "size"
        out.append(
            Member(
                name=name,
                kind="method",
                signature=signatures[shape],
                body="self._size()" if frame and sized else bodies[shape].format(name=name),
                doc=what if sized else f"{what} Over {over}.",
                returns="DataFrame | Series" if sized else gives,
            )
        )
    return tuple(out)


def _transformations(py: str) -> tuple[Member, ...]:
    """Writes the transformation members for one class.

    Twelve on a series and twelve on a frame, and they are not the same twelve.
    `dropna` removes values from a column and removes rows from a frame, so the
    frame one goes through its own door and takes a `subset`, while the series
    one is a per column transformation like the rest. The frame gains
    `is_monotonic_increasing` from nowhere, because a frame has no such property
    in pandas, and the series gains it as a property rather than a method.

    Same restriction as `_reductions`. Nothing here decides what a
    transformation does. The word crosses the boundary,
    `firepanda/py/transform.mojo` reads it, and every body is one call to a
    mixin helper that checks the arguments.

    The signatures are pandas' own, measured. Two details in them are easy to
    get wrong and are load bearing for the parity test: the four scans take
    `*args, **kwargs` after their named parameters, and `pct_change` takes
    `**kwargs` without the `*args`.

    Args:
        py: The class name, `DataFrame` or `Series`.

    Returns:
        The members, in the order they should be written out.
    """
    frame = py == "DataFrame"
    this = "frame" if frame else "column"
    gives = py
    out: list[Member] = []

    if frame:
        out.append(
            Member(
                name="dropna",
                kind="method",
                signature=(
                    "*, axis: Any = 0, how: Any = NO_DEFAULT, thresh: Any = NO_DEFAULT,"
                    " subset: Any = None, inplace: bool = False, ignore_index: bool = False"
                ),
                body="self._dropna(axis, how, thresh, subset, inplace, ignore_index)",
                doc="The rows with no missing value in them.",
                returns="DataFrame",
            )
        )
    else:
        out.append(
            Member(
                name="dropna",
                kind="method",
                signature=(
                    "*, axis: Any = 0, inplace: bool = False, how: Any = None,"
                    " ignore_index: bool = False"
                ),
                body='self._transform("dropna", 0, axis, inplace, ignore_index)',
                doc="The values that are not missing.",
                returns="Series",
            )
        )

    out.append(
        Member(
            name="astype",
            kind="method",
            signature='dtype: Any, copy: Any = NO_DEFAULT, errors: Any = "raise"',
            body="self._astype(dtype, copy, errors)",
            doc=(
                "Every column converted to another type, or the ones a dict names."
                if frame
                else "The column converted to another type."
            ),
            returns=gives,
        )
    )

    for name, what in (
        ("isna", "True where a value is missing."),
        ("notna", "True where a value is present."),
    ):
        out.append(
            Member(
                name=name,
                kind="method",
                signature="",
                body=f'self._transform("{name}", 0, 0, False, False)',
                doc=what,
                returns=gives,
            )
        )

    for name, way in (("ffill", "before"), ("bfill", "after")):
        out.append(
            Member(
                name=name,
                kind="method",
                signature=(
                    "*, axis: Any = None, inplace: bool = False, limit: Any = None,"
                    " limit_area: Any = None"
                ),
                body=f'self._fill("{name}", axis, inplace, limit, limit_area)',
                doc=f"Each missing value taken from the nearest present one {way} it.",
                returns=gives,
            )
        )

    shift = (
        "periods: Any = 1, freq: Any = None, axis: Any = 0,"
        " fill_value: Any = NO_DEFAULT, suffix: Any = None"
    )
    out.append(
        Member(
            name="shift",
            kind="method",
            signature=shift,
            body="self._shift(periods, freq, axis, fill_value, suffix)",
            doc=f"The {this} with its rows moved along, leaving the gap missing.",
            returns=gives,
        )
    )

    out.append(
        Member(
            name="diff",
            kind="method",
            signature="periods: int = 1, axis: Any = 0" if frame else "periods: int = 1",
            body=(
                'self._transform("diff", periods, axis, False, False)'
                if frame
                else 'self._transform("diff", periods, 0, False, False)'
            ),
            doc="The difference between each row and the one that many rows before it.",
            returns=gives,
        )
    )

    out.append(
        Member(
            name="pct_change",
            kind="method",
            signature=(
                "periods: int = 1, fill_method: Any = None, freq: Any = None, **kwargs: Any"
            ),
            body="self._pct_change(periods, fill_method, freq)",
            doc="The fractional change between each row and the one that many rows before it.",
            returns=gives,
        )
    )

    for name, what in CUMULATIVE:
        parts = ["axis: Any = 0", "skipna: bool = True"]
        if frame:
            parts.append("numeric_only: bool = False")
        only = "numeric_only" if frame else "False"
        out.append(
            Member(
                name=name,
                kind="method",
                signature=", ".join(parts) + ", *args: Any, **kwargs: Any",
                body=f'self._scan("{name}", axis, skipna, {only})',
                doc=what,
                returns=gives,
            )
        )

    if not frame:
        for way in ("increasing", "decreasing"):
            out.append(
                Member(
                    name=f"is_monotonic_{way}",
                    kind="property",
                    body=f"self._inner.monotonic({way == 'increasing'})",
                    doc=(
                        f"Whether the values never go {'down' if way == 'increasing' else 'up'}."
                        " A missing value makes this False."
                    ),
                    returns="bool",
                )
            )

    return tuple(out)


FRAME = Exposed(
    mojo="PyDataFrame",
    name="DataFrame",
    py="DataFrame",
    doc=("A two dimensional labelled data structure with columns of potentially different types."),
    init="PyDataFrame.py_init",
    mixin="DataFrameMixin",
    constructed=True,
    init_params=(("data", "object"),),
    bindings=(
        Binding(
            mojo="PyDataFrame.length",
            name="length",
            doc="The number of rows.",
            returns="int",
        ),
        Binding(
            mojo="PyDataFrame.width",
            name="width",
            doc="The number of columns.",
            returns="int",
        ),
        Binding(
            mojo="PyDataFrame.names",
            name="names",
            doc="The column names, in order.",
            returns="list[str]",
        ),
        Binding(
            mojo="PyDataFrame.dtypes",
            name="dtypes",
            doc="The column types, in order, as dtype spells them.",
            returns="list[str]",
        ),
        Binding(
            mojo="PyDataFrame.set_index",
            name="set_index",
            doc="The frame with one column moved into the row labels.",
            params=(("name", "str"), ("drop", "bool")),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.reset_index",
            name="reset_index",
            doc="The frame with its row labels put back to a count from zero.",
            params=(("drop", "bool"),),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.sort_index",
            name="sort_index",
            doc="The frame with its rows in the order of their labels.",
            params=(("ascending", "bool"),),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.sort_values",
            name="sort_values",
            doc="The frame with its rows in the order of some of its columns.",
            params=(
                ("by", "list[str]"),
                ("descending", "list[bool]"),
                ("nulls_first", "list[bool]"),
            ),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.head",
            name="head",
            doc="The first n rows.",
            params=(("n", "int"),),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.tail",
            name="tail",
            doc="The last n rows.",
            params=(("n", "int"),),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.take",
            name="take",
            doc="Rows gathered by position, counting from the end when negative.",
            params=(("positions", "list[int]"),),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.slice_rows",
            name="slice_rows",
            doc="A half open range of rows.",
            params=(("start", "int"), ("end", "int")),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.filter_rows",
            name="filter_rows",
            doc="The rows a boolean column is true at.",
            params=(("mask", "Series"),),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.cell",
            name="cell",
            doc="One value, by row and by column position.",
            params=(("row", "int"), ("at", "int")),
            returns="object",
        ),
        Binding(
            mojo="PyDataFrame.column",
            name="column",
            doc="One column, as a series.",
            params=(("name", "str"),),
            returns="Series",
        ),
        Binding(
            mojo="PyDataFrame.select",
            name="select",
            doc="Several columns, as a frame.",
            params=(("names", "list[str]"),),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.renamed_columns",
            name="renamed_columns",
            doc="The same data under a different schema.",
            params=(("olds", "list[str]"), ("news", "list[str]")),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.renamed_axis",
            name="renamed_axis",
            doc="The frame with its row labels under a different level name.",
            params=(("name", "str | None"),),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.duplicated",
            name="duplicated",
            doc="Which rows repeat a key another row already carries.",
            params=(("subset", "list[str]"), ("keep", "str")),
            returns="Series",
        ),
        Binding(
            mojo="PyDataFrame.drop_duplicates",
            name="drop_duplicates",
            doc="The frame with the repeated rows removed, by a chosen rule.",
            params=(("subset", "list[str]"), ("keep", "str")),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.top_rows",
            name="top_rows",
            doc="The n best rows of the frame, by one column.",
            params=(
                ("column", "str"),
                ("n", "int"),
                ("largest", "bool"),
                ("keep", "str"),
            ),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.reindex",
            name="reindex",
            doc="The frame on a set of row labels, whether it has them or not.",
            params=(("labels", "object"), ("fill_value", "object")),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.reindex_columns",
            name="reindex_columns",
            doc="The frame under a set of column names, in that order.",
            params=(("names", "list[str]"), ("fill_value", "object")),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.reindex_like",
            name="reindex_like",
            doc="The frame on another frame's labels and column names.",
            params=(("other", "object"),),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.reduce",
            name="reduce",
            doc="Every column reduced to one value, as a series of them.",
            params=(("kind", "str"), ("param", "float")),
            returns="Series",
        ),
        Binding(
            mojo="PyDataFrame.transform",
            name="transform",
            doc="Every column put through one named transformation.",
            params=(("kind", "str"), ("periods", "int")),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.window_agg",
            name="window_agg",
            doc="One reduction over every window of every column.",
            params=(
                ("kind", "str"),
                ("window", "int | None"),
                ("min_periods", "int | None"),
                ("center", "bool"),
                ("closed", "str"),
                ("step", "int | None"),
                ("settings", "tuple[object, ...]"),
            ),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.ewm_agg",
            name="ewm_agg",
            doc="One exponentially weighted reduction down every column.",
            params=(
                ("kind", "str"),
                ("alpha", "float"),
                ("min_periods", "int"),
                ("adjust", "bool"),
                ("ignore_na", "bool"),
                ("settings", "tuple[object, ...]"),
            ),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.dropna",
            name="dropna",
            doc="The rows with no missing value in them.",
            params=(("subset", "list[str]"),),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.cast",
            name="cast",
            doc="Some columns converted to other types, as a new frame.",
            params=(
                ("names", "list[str]"),
                ("dtypes", "list[str]"),
                ("strict", "bool"),
            ),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.group_agg",
            name="group_agg",
            doc="One reduction applied to every column that is not a key.",
            params=(
                ("by", "list[str]"),
                ("kind", "str"),
                ("param", "float"),
                ("dropna", "bool"),
                ("sort", "bool"),
                ("as_index", "bool"),
            ),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.labels",
            name="labels",
            doc="The row labels, as an index.",
            returns="Index",
        ),
        Binding(
            mojo="PyDataFrame.binary_frame",
            name="binary_frame",
            doc="An operation between two frames, aligning on both axes.",
            params=(
                ("other", "DataFrame"),
                ("op", "str"),
                ("flip", "bool"),
                ("fill_value", "object | None"),
            ),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.binary_series",
            name="binary_series",
            doc="An operation between a frame and a series, along one axis.",
            params=(
                ("other", "Series"),
                ("op", "str"),
                ("axis", "int"),
                ("flip", "bool"),
            ),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.binary_value",
            name="binary_value",
            doc="An operation between every cell of a frame and one constant.",
            params=(("other", "object"), ("op", "str"), ("flip", "bool")),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.compare_frame",
            name="compare_frame",
            doc="A comparison between two frames labelled the same on both axes.",
            params=(("other", "DataFrame"), ("op", "str")),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.unary",
            name="unary",
            doc="One of the four unary operations, over every column.",
            params=(("op", "str"),),
            returns="DataFrame",
        ),
        Binding(
            mojo="PyDataFrame.arrow_c_schema",
            name="arrow_c_schema",
            doc="The frame's Arrow schema, in a capsule.",
            returns="object",
        ),
        Binding(
            mojo="PyDataFrame.arrow_c_array",
            name="arrow_c_array",
            doc="The frame's Arrow schema and data, in two capsules.",
            params=(("requested_schema", "object | None"),),
            returns="list[object]",
        ),
        Binding(
            mojo="PyDataFrame.arrow_c_stream",
            name="arrow_c_stream",
            doc="The frame as an Arrow stream, in a capsule.",
            params=(("requested_schema", "object | None"),),
            returns="object",
        ),
    ),
    members=(
        Member(
            name="__len__",
            kind="dunder",
            body="self._inner.length()",
            doc="The number of rows, so that len(df) works.",
            returns="int",
        ),
        Member(
            name="groupby",
            kind="method",
            signature=(
                "by: Any = None, level: Any = None, *, as_index: bool = True,"
                " sort: bool = True, group_keys: bool = True, observed: bool = True,"
                " dropna: bool = True"
            ),
            body="_grouped(self, by, level, as_index, sort, group_keys, observed, dropna)",
            doc="A grouping over the frame, which computes nothing until it is reduced.",
            returns="DataFrameGroupBy",
        ),
        Member(
            name="__repr__",
            kind="dunder",
            body="repr(self._inner)",
            doc="The frame, rendered.",
            returns="str",
        ),
        Member(
            name="__str__",
            kind="dunder",
            body="repr(self._inner)",
            doc="The frame, rendered. Same as repr, which is what pandas does.",
            returns="str",
        ),
        Member(
            name="columns",
            kind="property",
            body="self._inner.names()",
            doc="The column labels of the frame.",
            returns="list[str]",
        ),
        Member(
            name="shape",
            kind="property",
            body="(self._inner.length(), self._inner.width())",
            doc="A tuple of the number of rows and the number of columns.",
            returns="tuple[int, int]",
        ),
        Member(
            name="index",
            kind="property",
            body="self._inner.labels()",
            doc="The row labels of the frame.",
            returns="Index",
            wraps="Index",
        ),
        Member(
            name="head",
            kind="method",
            signature="n: int = 5",
            body="self._inner.head(n)",
            doc="The first n rows.",
            returns="DataFrame",
            wraps="DataFrame",
        ),
        Member(
            name="tail",
            kind="method",
            signature="n: int = 5",
            body="self._inner.tail(n)",
            doc="The last n rows.",
            returns="DataFrame",
            wraps="DataFrame",
        ),
        Member(
            name="iloc",
            kind="property",
            body="_Positional(self)",
            doc="Selection by position, where a slice excludes the row it stops at.",
            returns="Any",
        ),
        Member(
            name="loc",
            kind="property",
            body="_Labelled(self)",
            doc="Selection by label, where a slice includes the row it stops at.",
            returns="Any",
        ),
        Member(
            name="iat",
            kind="property",
            body="_Cell(self, False)",
            doc="One value, by row position and column position.",
            returns="Any",
        ),
        Member(
            name="at",
            kind="property",
            body="_Cell(self, True)",
            doc="One value, by row label and column name.",
            returns="Any",
        ),
        Member(
            name="take",
            kind="method",
            signature="indices: Any, axis: Any = 0, **kwargs: Any",
            body="self._take(indices, axis, kwargs)",
            doc="The rows or the columns at a set of positions, in the order given.",
            returns="DataFrame",
        ),
        Member(
            name="set_index",
            kind="method",
            signature=(
                "keys: Any, *, drop: bool = True, append: bool = False,"
                " inplace: bool = False, verify_integrity: Any = NO_DEFAULT"
            ),
            body="self._set_index(keys, drop, append, inplace, verify_integrity)",
            doc="The frame with one of its columns moved into the row labels.",
            returns="DataFrame",
        ),
        Member(
            name="reset_index",
            kind="method",
            signature=(
                "level: Any = None, *, drop: bool = False, inplace: bool = False,"
                " col_level: Any = 0, col_fill: Any = \"\","
                " allow_duplicates: Any = NO_DEFAULT, names: Any = None"
            ),
            body=(
                "self._reset_index(level, drop, inplace, col_level, col_fill,"
                " allow_duplicates, names)"
            ),
            doc="The frame with its row labels put back to a count from zero.",
            returns="DataFrame",
        ),
        Member(
            name="sort_index",
            kind="method",
            signature=(
                "*, axis: Any = 0, level: Any = None, ascending: Any = True,"
                " inplace: bool = False, kind: str = \"quicksort\","
                " na_position: str = \"last\", sort_remaining: bool = True,"
                " ignore_index: bool = False, key: Any = None"
            ),
            body=(
                "self._sort_index(axis, level, ascending, inplace, na_position,"
                " sort_remaining, ignore_index, key)"
            ),
            doc="The frame with its rows in the order of their labels.",
            returns="DataFrame",
        ),
        Member(
            name="sort_values",
            kind="method",
            signature=(
                "by: Any, *, axis: Any = 0, ascending: Any = True,"
                " inplace: bool = False, kind: str = \"quicksort\","
                " na_position: str = \"last\", ignore_index: bool = False,"
                " key: Any = None"
            ),
            body=(
                "self._sort_values(by, axis, ascending, inplace, na_position,"
                " ignore_index, key)"
            ),
            doc="The frame with its rows in the order of some of its columns.",
            returns="DataFrame",
        ),
        Member(
            name="filter",
            kind="method",
            signature=(
                "items: Any = None, like: str | None = None,"
                " regex: str | None = None, axis: Any = None"
            ),
            body="self._filter(items, like, regex, axis)",
            doc="The columns named by one of three rules, in the frame's own order.",
            returns="DataFrame",
        ),
        Member(
            name="get",
            kind="method",
            signature="key: Any, default: Any = None",
            body="self._get(key, default)",
            doc="One column, or the default when the frame has no such column.",
            returns="Any",
        ),
        Member(
            name="squeeze",
            kind="method",
            signature="axis: Any = None",
            body="self._squeeze(axis)",
            doc="The frame with an axis of length one dropped.",
            returns="Any",
        ),
        Member(
            name="select_dtypes",
            kind="method",
            signature="include: Any = None, exclude: Any = None",
            body="self._select_dtypes(include, exclude)",
            doc="The columns whose type is in one set of types and not in another.",
            returns="DataFrame",
        ),
        Member(
            name="truncate",
            kind="method",
            signature=(
                "before: Any = None, after: Any = None, axis: Any = None,"
                " copy: Any = NO_DEFAULT"
            ),
            body="self._truncate(before, after, axis, copy)",
            doc="The rows between two labels, with both of them kept.",
            returns="DataFrame",
        ),
        Member(
            name="duplicated",
            kind="method",
            signature="subset: Any = None, keep: Any = \"first\"",
            body="self._duplicated(subset, keep)",
            doc="Which rows repeat a key that another row already carries.",
            returns="Series",
        ),
        Member(
            name="drop_duplicates",
            kind="method",
            signature=(
                "subset: Any = None, *, keep: Any = \"first\","
                " inplace: bool = False, ignore_index: bool = False"
            ),
            body="self._drop_duplicates(subset, keep, inplace, ignore_index)",
            doc="The frame with the repeated rows removed, by a chosen rule.",
            returns="DataFrame",
        ),
        Member(
            name="reindex",
            kind="method",
            signature=(
                "labels: Any = None, *, index: Any = None, columns: Any = None,"
                " axis: Any = None, method: Any = None, copy: Any = NO_DEFAULT,"
                ' level: Any = None, fill_value: Any = float("nan"),'
                " limit: Any = None, tolerance: Any = None"
            ),
            body=(
                "self._reindex(labels, index, columns, axis, method, copy,"
                " level, fill_value, limit, tolerance)"
            ),
            doc="The frame on a set of row labels or column names, or both.",
            returns="DataFrame",
        ),
        Member(
            name="reindex_like",
            kind="method",
            signature=(
                "other: Any, method: Any = None, copy: Any = NO_DEFAULT,"
                " limit: Any = None, tolerance: Any = None"
            ),
            body="self._reindex_like(other, method, copy, limit, tolerance)",
            doc="The frame shaped the way another frame is shaped.",
            returns="DataFrame",
        ),
        Member(
            name="nlargest",
            kind="method",
            signature="n: int, columns: Any, keep: Any = \"first\"",
            body="self._top_rows(n, columns, keep, True)",
            doc="The n rows holding the largest values in one column.",
            returns="DataFrame",
        ),
        Member(
            name="nsmallest",
            kind="method",
            signature="n: int, columns: Any, keep: Any = \"first\"",
            body="self._top_rows(n, columns, keep, False)",
            doc="The n rows holding the smallest values in one column.",
            returns="DataFrame",
        ),
        Member(
            name="__arrow_c_schema__",
            kind="dunder",
            body="self._inner.arrow_c_schema()",
            doc="The frame's Arrow schema, as an arrow_schema PyCapsule.",
            returns="object",
        ),
        Member(
            name="__arrow_c_array__",
            kind="dunder",
            signature="requested_schema: object | None = None",
            body="tuple(self._inner.arrow_c_array(requested_schema))",
            doc="The frame's Arrow data, as an arrow_schema and an arrow_array PyCapsule.",
            returns="tuple[object, ...]",
        ),
        Member(
            name="__arrow_c_stream__",
            kind="dunder",
            signature="requested_schema: object | None = None",
            body="self._inner.arrow_c_stream(requested_schema)",
            doc="The frame as a stream of one batch, as an arrow_array_stream PyCapsule.",
            returns="object",
        ),
        Member(
            name="rolling",
            kind="method",
            signature=(
                "window: Any, min_periods: int | None = None, center: bool = False,"
                " win_type: str | None = None, on: str | None = None,"
                ' closed: str | None = None, step: int | None = None, method: str = "single"'
            ),
            body="_rolling(self, window, min_periods, center, win_type, on, closed, step, method)",
            doc="A window of a fixed width over every column, computing nothing until reduced.",
            returns="Rolling",
        ),
        Member(
            name="expanding",
            kind="method",
            signature='min_periods: int = 1, method: str = "single"',
            body="_expanding(self, min_periods, method)",
            doc="A window over every column that starts at the first row and grows.",
            returns="Expanding",
        ),
        Member(
            name="ewm",
            kind="method",
            signature=(
                "com: float | None = None, span: float | None = None,"
                " halflife: float | None = None, alpha: float | None = None,"
                " min_periods: int | None = 0, adjust: bool = True,"
                ' ignore_na: bool = False, times: Any = None, method: str = "single"'
            ),
            body=(
                "_ewm(self, com, span, halflife, alpha, min_periods, adjust,"
                " ignore_na, times, method)"
            ),
            doc="A decay over every row of every column, computing nothing until reduced.",
            returns="ExponentialMovingWindow",
        ),
        *_reductions("DataFrame"),
        *_transformations("DataFrame"),
        *_operators("DataFrame"),
    ),
)

SERIES = Exposed(
    mojo="PySeries",
    name="Series",
    py="Series",
    doc="A one dimensional labelled array holding data of a single type.",
    init="PySeries.py_init",
    module="firepanda.py.series",
    mixin="SeriesMixin",
    constructed=True,
    init_params=(("data", "object"), ("name", "str")),
    bindings=(
        Binding(
            mojo="PySeries.length",
            name="length",
            doc="The number of rows.",
            returns="int",
        ),
        Binding(
            mojo="PySeries.label",
            name="label",
            doc="The name of the column.",
            returns="str",
        ),
        Binding(
            mojo="PySeries.relabel",
            name="relabel",
            doc="A copy of the column under a different name.",
            params=(("name", "str"),),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.dtype",
            name="dtype",
            doc="The type, as firepanda spells it.",
            returns="str",
        ),
        Binding(
            mojo="PySeries.null_count",
            name="null_count",
            doc="How many rows are missing.",
            returns="int",
        ),
        Binding(
            mojo="PySeries.head",
            name="head",
            doc="The first n rows.",
            params=(("n", "int"),),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.tail",
            name="tail",
            doc="The last n rows.",
            params=(("n", "int"),),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.take",
            name="take",
            doc="Rows gathered by position, counting from the end when negative.",
            params=(("positions", "list[int]"),),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.sort_values",
            name="sort_values",
            doc="The column with its rows in the order of their own values.",
            params=(("descending", "bool"), ("nulls_first", "bool")),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.argsort",
            name="argsort",
            doc="The row order a sort would put the values in, as a column.",
            params=(("descending", "bool"), ("nulls_first", "bool")),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.slice_rows",
            name="slice_rows",
            doc="A half open range of rows.",
            params=(("start", "int"), ("end", "int")),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.filter_rows",
            name="filter_rows",
            doc="The rows a boolean column is true at.",
            params=(("mask", "Series"),),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.cell",
            name="cell",
            doc="One value, by position.",
            params=(("at", "int"),),
            returns="object",
        ),
        Binding(
            mojo="PySeries.renamed_axis",
            name="renamed_axis",
            doc="The series with its row labels under a different level name.",
            params=(("name", "str | None"),),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.reindex",
            name="reindex",
            doc="The series on a set of labels, whether it has them or not.",
            params=(("labels", "object"), ("fill_value", "object")),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.reindex_like",
            name="reindex_like",
            doc="The series on the labels an index carries, under its name.",
            params=(("index", "object"),),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.to_list",
            name="to_list",
            doc="Every value, copied into a Python list.",
            returns="list[object]",
        ),
        Binding(
            mojo="PySeries.labels",
            name="labels",
            doc="The row labels, as an index.",
            returns="Index",
        ),
        Binding(
            mojo="PySeries.to_index",
            name="to_index",
            doc="The values of the column, as the labels of an index.",
            params=(("name", "str | None"),),
            returns="Index",
        ),
        Binding(
            mojo="PySeries.reduce",
            name="reduce",
            doc="The whole column reduced to one Python value.",
            params=(("kind", "str"), ("param", "float")),
            returns="object",
        ),
        Binding(
            mojo="PySeries.transform",
            name="transform",
            doc="The column put through one named transformation.",
            params=(("kind", "str"), ("periods", "int")),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.cast",
            name="cast",
            doc="The column converted to another type.",
            params=(("dtype", "str"), ("strict", "bool")),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.categories",
            name="categories",
            doc="A category column's categories, as an index.",
            returns="Index",
        ),
        Binding(
            mojo="PySeries.codes",
            name="codes",
            doc="The codes of a category column, as positions.",
            returns="Series",
        ),
        Binding(
            mojo="PySeries.ordered",
            name="ordered",
            doc="Whether the categories have a meaningful order.",
            returns="bool",
        ),
        Binding(
            mojo="PySeries.set_ordered",
            name="set_ordered",
            doc="The same categories, ordered or not.",
            params=(("ordered", "bool"),),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.relabel_categories",
            name="relabel_categories",
            doc="New labels for the categories, matched by position.",
            params=(("names", "list[str]"), ("ordered", "bool")),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.recategorize",
            name="recategorize",
            doc="New categories, matched by value.",
            params=(("names", "list[str]"), ("ordered", "bool")),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.drop_unused_categories",
            name="drop_unused_categories",
            doc="Only the categories that appear in the values.",
            returns="Series",
        ),
        Binding(
            mojo="PySeries.monotonic",
            name="monotonic",
            doc="Whether the column is sorted, one way or the other.",
            params=(("increasing", "bool"),),
            returns="bool",
        ),
        Binding(
            mojo="PySeries.string_text",
            name="string_text",
            doc="One str accessor method that answers text, as a column.",
            params=(
                ("kind", "str"),
                ("arg", "str"),
                ("start", "int | None"),
                ("stop", "int | None"),
                ("step", "int"),
            ),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.window_agg",
            name="window_agg",
            doc="One reduction over every window of the column.",
            params=(
                ("kind", "str"),
                ("window", "int | None"),
                ("min_periods", "int | None"),
                ("center", "bool"),
                ("closed", "str"),
                ("step", "int | None"),
                ("settings", "tuple[object, ...]"),
            ),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.ewm_agg",
            name="ewm_agg",
            doc="One exponentially weighted reduction down the column.",
            params=(
                ("kind", "str"),
                ("alpha", "float"),
                ("min_periods", "int"),
                ("adjust", "bool"),
                ("ignore_na", "bool"),
                ("settings", "tuple[object, ...]"),
            ),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.string_flag",
            name="string_flag",
            doc="One str accessor method that answers a mask, as a column.",
            params=(("kind", "str"), ("arg", "str")),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.string_number",
            name="string_number",
            doc="One str accessor method that answers a number, as a column.",
            params=(
                ("kind", "str"),
                ("arg", "str"),
                ("start", "int | None"),
                ("stop", "int | None"),
            ),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.string_is_text",
            name="string_is_text",
            doc="Whether the column holds text at all.",
            params=(),
            returns="bool",
        ),
        Binding(
            mojo="PySeries.temporal_part",
            name="temporal_part",
            doc="One part of a temporal column, as a column.",
            params=(("kind", "str"), ("arg", "str")),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.temporal_word",
            name="temporal_word",
            doc="The clock or the resolution of a temporal column, as a string.",
            params=(("kind", "str"),),
            returns="str",
        ),
        Binding(
            mojo="PySeries.to_datetime",
            name="to_datetime",
            doc="A column of text or of whole numbers read as a column of instants.",
            params=(
                ("format", "str"),
                ("unit", "str"),
                ("coerce", "bool"),
                ("utc", "bool"),
            ),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.binary_series",
            name="binary_series",
            doc="An operation between two series, matching rows by label.",
            params=(
                ("other", "Series"),
                ("op", "str"),
                ("flip", "bool"),
                ("fill_value", "object | None"),
            ),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.binary_value",
            name="binary_value",
            doc="An operation between every row of a series and one constant.",
            params=(("other", "object"), ("op", "str"), ("flip", "bool")),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.compare_series",
            name="compare_series",
            doc="A comparison between two series labelled the same.",
            params=(("other", "Series"), ("op", "str")),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.unary",
            name="unary",
            doc="One of the four unary operations, over every row.",
            params=(("op", "str"),),
            returns="Series",
        ),
        Binding(
            mojo="PySeries.arrow_c_schema",
            name="arrow_c_schema",
            doc="The column's Arrow schema, in a capsule.",
            returns="object",
        ),
        Binding(
            mojo="PySeries.arrow_c_array",
            name="arrow_c_array",
            doc="The column's Arrow schema and data, in two capsules.",
            params=(("requested_schema", "object | None"),),
            returns="list[object]",
        ),
    ),
    members=(
        Member(
            name="__len__",
            kind="dunder",
            body="self._inner.length()",
            doc="The number of rows, so that len(s) works.",
            returns="int",
        ),
        Member(
            name="__repr__",
            kind="dunder",
            body="repr(self._inner)",
            doc="The series, rendered.",
            returns="str",
        ),
        Member(
            name="__str__",
            kind="dunder",
            body="repr(self._inner)",
            doc="The series, rendered. Same as repr, which is what pandas does.",
            returns="str",
        ),
        Member(
            name="name",
            kind="property",
            body="self._inner.label()",
            doc="The name of the series.",
            returns="str",
        ),
        Member(
            name="dtype",
            kind="property",
            body="self._inner.dtype()",
            doc="The type of the values, as a string rather than a numpy dtype.",
            returns="str",
        ),
        Member(
            name="size",
            kind="property",
            body="self._inner.length()",
            doc="The number of elements.",
            returns="int",
        ),
        Member(
            name="shape",
            kind="property",
            body="(self._inner.length(),)",
            doc="A tuple of the number of rows, which for a series is one long.",
            returns="tuple[int]",
        ),
        Member(
            name="index",
            kind="property",
            body="self._inner.labels()",
            doc="The row labels of the series.",
            returns="Index",
            wraps="Index",
        ),
        Member(
            name="head",
            kind="method",
            signature="n: int = 5",
            body="self._inner.head(n)",
            doc="The first n rows.",
            returns="Series",
            wraps="Series",
        ),
        Member(
            name="tail",
            kind="method",
            signature="n: int = 5",
            body="self._inner.tail(n)",
            doc="The last n rows.",
            returns="Series",
            wraps="Series",
        ),
        Member(
            name="iloc",
            kind="property",
            body="_Along(self, False)",
            doc="Selection by position, where a slice excludes the row it stops at.",
            returns="Any",
        ),
        Member(
            name="loc",
            kind="property",
            body="_Along(self, True)",
            doc="Selection by label, where a slice includes the row it stops at.",
            returns="Any",
        ),
        Member(
            name="iat",
            kind="property",
            body="_Point(self, False)",
            doc="One value, by row position.",
            returns="Any",
        ),
        Member(
            name="at",
            kind="property",
            body="_Point(self, True)",
            doc="One value, by row label.",
            returns="Any",
        ),
        Member(
            name="reindex",
            kind="method",
            signature=(
                "index: Any = None, *, axis: Any = None, method: Any = None,"
                " copy: Any = NO_DEFAULT, level: Any = None,"
                " fill_value: Any = None, limit: Any = None,"
                " tolerance: Any = None"
            ),
            body=(
                "self._reindex(index, axis, method, copy, level, fill_value,"
                " limit, tolerance)"
            ),
            doc="The series on a set of labels, whether it has them or not.",
            returns="Series",
        ),
        Member(
            name="reindex_like",
            kind="method",
            signature=(
                "other: Any, method: Any = None, copy: Any = NO_DEFAULT,"
                " limit: Any = None, tolerance: Any = None"
            ),
            body="self._reindex_like(other, method, copy, limit, tolerance)",
            doc="The series labelled the way another thing is labelled.",
            returns="Series",
        ),
        Member(
            name="get",
            kind="method",
            signature="key: Any, default: Any = None",
            body="self._get(key, default)",
            doc="The value at a label, or the default when there is no such label.",
            returns="Any",
        ),
        Member(
            name="squeeze",
            kind="method",
            signature="axis: Any = None",
            body="self._squeeze(axis)",
            doc="The one value of a series of one row, or the series itself.",
            returns="Any",
        ),
        Member(
            name="tolist",
            kind="method",
            body="list(self._inner.to_list())",
            doc="The values as a Python list, with None where a value is missing.",
            returns="list[object]",
        ),
        Member(
            name="count",
            kind="method",
            body="self._inner.length() - self._inner.null_count()",
            doc="The number of values that are not missing.",
            returns="int",
        ),
        Member(
            name="hasnans",
            kind="property",
            body="self._inner.null_count() > 0",
            doc="Whether any value is missing.",
            returns="bool",
        ),
        Member(
            name="__arrow_c_schema__",
            kind="dunder",
            body="self._inner.arrow_c_schema()",
            doc="The column's Arrow schema, as an arrow_schema PyCapsule.",
            returns="object",
        ),
        Member(
            name="__arrow_c_array__",
            kind="dunder",
            signature="requested_schema: object | None = None",
            body="tuple(self._inner.arrow_c_array(requested_schema))",
            doc="The column's Arrow data, as an arrow_schema and an arrow_array PyCapsule.",
            returns="tuple[object, ...]",
        ),
        *_reductions("Series"),
        *_transformations("Series"),
        Member(
            name="rolling",
            kind="method",
            signature=(
                "window: Any, min_periods: int | None = None, center: bool = False,"
                " win_type: str | None = None, on: str | None = None,"
                ' closed: str | None = None, step: int | None = None, method: str = "single"'
            ),
            body="_rolling(self, window, min_periods, center, win_type, on, closed, step, method)",
            doc="A window of a fixed width, which computes nothing until it is reduced.",
            returns="Rolling",
        ),
        Member(
            name="expanding",
            kind="method",
            signature='min_periods: int = 1, method: str = "single"',
            body="_expanding(self, min_periods, method)",
            doc="A window that starts at the first row and grows, reduced the same way.",
            returns="Expanding",
        ),
        Member(
            name="ewm",
            kind="method",
            signature=(
                "com: float | None = None, span: float | None = None,"
                " halflife: float | None = None, alpha: float | None = None,"
                " min_periods: int | None = 0, adjust: bool = True,"
                ' ignore_na: bool = False, times: Any = None, method: str = "single"'
            ),
            body=(
                "_ewm(self, com, span, halflife, alpha, min_periods, adjust,"
                " ignore_na, times, method)"
            ),
            doc="A decay over every row of the column, computing nothing until reduced.",
            returns="ExponentialMovingWindow",
        ),
        Member(
            name="dt",
            kind="accessor",
            body="DatetimeProperties",
            doc=(
                "The datetime accessor, which is where the calendar and clock parts of a"
                " temporal column live."
            ),
            returns="DatetimeProperties",
        ),
        Member(
            name="str",
            kind="accessor",
            body="StringAccessor",
            doc=(
                "The string accessor, which is where the methods that read a text"
                " column character by character live."
            ),
            returns="StringAccessor",
        ),
        Member(
            name="cat",
            kind="accessor",
            body="CategoricalAccessor",
            doc=(
                "The categorical accessor, which is where the categories of a category"
                " column live."
            ),
            returns="CategoricalAccessor",
        ),
        *_operators("Series"),
    ),
)


INDEX = Exposed(
    mojo="PyIndex",
    name="Index",
    py="Index",
    doc="The labels of the rows, which is what pandas addresses a row by.",
    init="PyIndex.py_init",
    module="firepanda.py.index",
    mixin="IndexMixin",
    constructed=True,
    init_params=(("data", "object"), ("name", "object")),
    bindings=(
        Binding(
            mojo="PyIndex.length",
            name="length",
            doc="The number of labels.",
            returns="int",
        ),
        Binding(
            mojo="PyIndex.label",
            name="label",
            doc="The level name, or None.",
            returns="str | None",
        ),
        Binding(
            mojo="PyIndex.dtype",
            name="dtype",
            doc="The type of the labels, as firepanda spells it.",
            returns="str",
        ),
        Binding(
            mojo="PyIndex.inferred_type",
            name="inferred_type",
            doc="What pandas calls the kind of the labels.",
            returns="str",
        ),
        Binding(
            mojo="PyIndex.is_range",
            name="is_range",
            doc="Whether the labels are still an arithmetic range.",
            returns="bool",
        ),
        Binding(
            mojo="PyIndex.start",
            name="start",
            doc="The first label of a range.",
            returns="int",
        ),
        Binding(
            mojo="PyIndex.nbytes",
            name="nbytes",
            doc="The bytes the labels occupy.",
            returns="int",
        ),
        Binding(
            mojo="PyIndex.null_count",
            name="null_count",
            doc="How many labels are missing.",
            returns="int",
        ),
        Binding(
            mojo="PyIndex.at",
            name="at",
            doc="One label, as a Python value.",
            params=(("i", "int"),),
            returns="object",
        ),
        Binding(
            mojo="PyIndex.to_list",
            name="to_list",
            doc="Every label, copied into a Python list.",
            returns="list[object]",
        ),
        Binding(
            mojo="PyIndex.slice_rows",
            name="slice_rows",
            doc="A half open range of rows.",
            params=(("start", "int"), ("end", "int")),
            returns="Index",
        ),
        Binding(
            mojo="PyIndex.take",
            name="take",
            doc="Labels gathered by position.",
            params=(("positions", "list[int]"),),
            returns="Index",
        ),
        Binding(
            mojo="PyIndex.is_unique",
            name="is_unique",
            doc="Whether every label appears once.",
            returns="bool",
        ),
        Binding(
            mojo="PyIndex.is_monotonic_increasing",
            name="is_monotonic_increasing",
            doc="Whether the labels never decrease.",
            returns="bool",
        ),
        Binding(
            mojo="PyIndex.is_monotonic_decreasing",
            name="is_monotonic_decreasing",
            doc="Whether the labels never increase.",
            returns="bool",
        ),
        Binding(
            mojo="PyIndex.get_loc",
            name="get_loc",
            doc="Every position one label sits at.",
            params=(("label", "object"),),
            returns="list[int]",
        ),
        Binding(
            mojo="PyIndex.searchsorted",
            name="searchsorted",
            doc="Where a label would go for the order to hold.",
            params=(("label", "object"), ("side", "str")),
            returns="int",
        ),
        Binding(
            mojo="PyIndex.isin",
            name="isin",
            doc="Which labels are among a set of values.",
            params=(("values", "object"),),
            returns="list[bool]",
        ),
        Binding(
            mojo="PyIndex.get_indexer",
            name="get_indexer",
            doc="Where each of a set of labels sits, with -1 for the missing.",
            params=(("target", "object"),),
            returns="list[int]",
        ),
        Binding(
            mojo="PyIndex.reindex",
            name="reindex",
            doc="The labels asked for and where each of them sits, as a pair.",
            params=(("target", "object"),),
            returns="list[object]",
        ),
        Binding(
            mojo="PyIndex.contains",
            name="contains",
            doc="Whether a label is in the index.",
            params=(("label", "object"),),
            returns="bool",
        ),
        Binding(
            mojo="PyIndex.equals",
            name="equals",
            doc="Whether two indexes hold the same labels.",
            params=(("other", "object"),),
            returns="bool",
        ),
        Binding(
            mojo="PyIndex.identical",
            name="identical",
            doc="Whether the labels and the name both match.",
            params=(("other", "object"),),
            returns="bool",
        ),
        Binding(
            mojo="PyIndex.same_as",
            name="same_as",
            doc="Whether two indexes are the same object underneath.",
            params=(("other", "object"),),
            returns="bool",
        ),
        Binding(
            mojo="PyIndex.unique",
            name="unique",
            doc="The first of each label.",
            returns="Index",
        ),
        Binding(
            mojo="PyIndex.renamed",
            name="renamed",
            doc="The index under a different level name.",
            params=(("name", "str | None"),),
            returns="Index",
        ),
        Binding(
            mojo="PyIndex.temporal_part",
            name="temporal_part",
            doc="One part of temporal labels, as another index.",
            params=(("kind", "str"), ("arg", "str")),
            returns="Index",
        ),
        Binding(
            mojo="PyIndex.temporal_word",
            name="temporal_word",
            doc="The one part of temporal labels that is a word.",
            params=(("kind", "str"),),
            returns="str",
        ),
        Binding(
            mojo="PyIndex.union",
            name="union",
            doc="Every label either side has.",
            params=(("other", "object"), ("sort", "bool")),
            returns="Index",
        ),
        Binding(
            mojo="PyIndex.intersection",
            name="intersection",
            doc="Every label both sides have.",
            params=(("other", "object"), ("sort", "bool")),
            returns="Index",
        ),
        Binding(
            mojo="PyIndex.difference",
            name="difference",
            doc="Every label this index has and the other does not.",
            params=(("other", "object"), ("sort", "bool")),
            returns="Index",
        ),
        Binding(
            mojo="PyIndex.symmetric_difference",
            name="symmetric_difference",
            doc="Every label exactly one side has.",
            params=(
                ("other", "object"),
                ("sort", "bool"),
                ("result_name", "str | None"),
            ),
            returns="Index",
        ),
        Binding(
            mojo="PyIndex.append",
            name="append",
            doc="One or several indexes put on the end of this one.",
            params=(("others", "list[object]"),),
            returns="Index",
        ),
        Binding(
            mojo="PyIndex.delete",
            name="delete",
            doc="The index without the labels at a set of positions.",
            params=(("positions", "list[int]"),),
            returns="Index",
        ),
        Binding(
            mojo="PyIndex.insert",
            name="insert",
            doc="The index with one label put in at a position.",
            params=(("position", "int"), ("label", "object")),
            returns="Index",
        ),
        Binding(
            mojo="PyIndex.drop",
            name="drop",
            doc="The index without every row carrying one of a set of labels.",
            params=(("labels", "object"), ("errors", "str")),
            returns="Index",
        ),
        Binding(
            mojo="PyIndex.putmask",
            name="putmask",
            doc="The index with the labels a mask picks out replaced.",
            params=(("mask", "list[bool]"), ("value", "object")),
            returns="Index",
        ),
        Binding(
            mojo="PyIndex.get_slice_bound",
            name="get_slice_bound",
            doc="Where a label sits when the index is read in order.",
            params=(("label", "object"), ("side", "str")),
            returns="int",
        ),
        Binding(
            mojo="PyIndex.slice_locs",
            name="slice_locs",
            doc="The half open row range a pair of labels describes.",
            params=(("start", "object"), ("end", "object")),
            returns="list[int]",
        ),
        Binding(
            mojo="PyIndex.slice_indexer",
            name="slice_indexer",
            doc="The same range with the step carried through.",
            params=(("start", "object"), ("end", "object"), ("step", "int")),
            returns="list[int]",
        ),
        Binding(
            mojo="PyIndex.arrow_c_schema",
            name="arrow_c_schema",
            doc="The labels' Arrow schema, in a capsule.",
            returns="object",
        ),
        Binding(
            mojo="PyIndex.arrow_c_array",
            name="arrow_c_array",
            doc="The labels' Arrow schema and data, in two capsules.",
            params=(("requested_schema", "object | None"),),
            returns="list[object]",
        ),
    ),
    members=(
        Member(
            name="__len__",
            kind="dunder",
            body="self._inner.length()",
            doc="The number of labels, so that len(index) works.",
            returns="int",
        ),
        Member(
            name="__repr__",
            kind="dunder",
            body="repr(self._inner)",
            doc="The index, rendered.",
            returns="str",
        ),
        Member(
            name="__str__",
            kind="dunder",
            body="repr(self._inner)",
            doc="The index, rendered. Same as repr, which is what pandas does.",
            returns="str",
        ),
        Member(
            name="name",
            kind="property",
            body="self._inner.label()",
            doc="The name of the level, or None when it does not have one.",
            returns="str | None",
        ),
        Member(
            name="dtype",
            kind="property",
            body="self._inner.dtype()",
            doc="The type of the labels, as a string rather than a numpy dtype.",
            returns="str",
        ),
        Member(
            name="inferred_type",
            kind="property",
            body="self._inner.inferred_type()",
            doc="What pandas calls the kind of the labels, such as integer or string.",
            returns="str",
        ),
        Member(
            name="size",
            kind="property",
            body="self._inner.length()",
            doc="The number of labels.",
            returns="int",
        ),
        Member(
            name="shape",
            kind="property",
            body="(self._inner.length(),)",
            doc="A tuple of the number of labels, which for a flat index is one long.",
            returns="tuple[int]",
        ),
        Member(
            name="ndim",
            kind="property",
            body="1",
            doc="The number of dimensions, which is one for every index that is not a MultiIndex.",
            returns="int",
        ),
        Member(
            name="nlevels",
            kind="property",
            body="1",
            doc="The number of levels, which is one until MultiIndex exists.",
            returns="int",
        ),
        Member(
            name="empty",
            kind="property",
            body="self._inner.length() == 0",
            doc="Whether the index has no labels at all.",
            returns="bool",
        ),
        Member(
            name="nbytes",
            kind="property",
            body="self._inner.nbytes()",
            doc="The bytes the labels occupy, which is zero for a range that stores none.",
            returns="int",
        ),
        Member(
            name="hasnans",
            kind="property",
            body="self._inner.null_count() > 0",
            doc="Whether any label is missing.",
            returns="bool",
        ),
        Member(
            name="values",
            kind="property",
            body="list(self._inner.to_list())",
            doc="The labels as a Python list, where pandas hands back a numpy array.",
            returns="list[object]",
        ),
        Member(
            name="is_unique",
            kind="property",
            body="self._inner.is_unique()",
            doc="Whether every label appears exactly once.",
            returns="bool",
        ),
        Member(
            name="has_duplicates",
            kind="property",
            body="not self._inner.is_unique()",
            doc="Whether any label appears more than once.",
            returns="bool",
        ),
        Member(
            name="is_monotonic_increasing",
            kind="property",
            body="self._inner.is_monotonic_increasing()",
            doc="Whether the labels never decrease, which is False if any is missing.",
            returns="bool",
        ),
        Member(
            name="is_monotonic_decreasing",
            kind="property",
            body="self._inner.is_monotonic_decreasing()",
            doc="Whether the labels never increase, which is False if any is missing.",
            returns="bool",
        ),
        Member(
            name="tolist",
            kind="method",
            body="list(self._inner.to_list())",
            doc="The labels as a Python list, with None where a label is missing.",
            returns="list[object]",
        ),
        Member(
            name="to_list",
            kind="method",
            body="list(self._inner.to_list())",
            doc="The labels as a Python list. The pandas spelling with an underscore.",
            returns="list[object]",
        ),
        Member(
            name="insert",
            kind="method",
            signature="loc: int, item: object",
            body="self._inner.insert(loc, item)",
            doc="The index with one label put in at a position.",
            returns="Index",
            wraps="Index",
        ),
        Member(
            name="get_slice_bound",
            kind="method",
            signature="label: object, side: str",
            body="self._inner.get_slice_bound(label, side)",
            doc="The position a label maps to when the index is read in order.",
            returns="int",
        ),
        Member(
            name="__arrow_c_schema__",
            kind="dunder",
            body="self._inner.arrow_c_schema()",
            doc="The labels' Arrow schema, as an arrow_schema PyCapsule.",
            returns="object",
        ),
        Member(
            name="__arrow_c_array__",
            kind="dunder",
            signature="requested_schema: object | None = None",
            body="tuple(self._inner.arrow_c_array(requested_schema))",
            doc="The labels' Arrow data, as an arrow_schema and an arrow_array PyCapsule.",
            returns="tuple[object, ...]",
        ),
    ),
)


FUNCTIONS = (
    Binding(
        mojo="open_csv",
        name="read_csv",
        doc="Reads a CSV file into a frame.",
        params=(("path", "str"),),
        returns="DataFrame",
        py_params=(("filepath_or_buffer", "str"),),
    ),
    Binding(
        mojo="open_arrow",
        name="from_arrow",
        doc="Builds a frame from a pyarrow, Polars or pandas frame.",
        params=(("source", "object"),),
        returns="DataFrame",
    ),
    # Not a user entry point either, and here for the reason the free function
    # in `firepanda/py/frame.mojo` gives: it reads a series and answers a frame,
    # so it belongs to neither bound type and the import graph will not let it
    # sit on the series. `DatetimeProperties.isocalendar` is what calls it.
    Binding(
        mojo="isocalendar",
        name="_isocalendar",
        doc="The ISO 8601 year, week and day of a temporal column, as a frame.",
        params=(("column", "object"),),
        returns="DataFrame",
    ),
    # Not a user entry point either, and here for the same reason the one above
    # is: it reads an index and answers a series, and the import graph will not
    # let a function of an index that gives a series live on the index.
    # `IndexMixin.to_series` is what calls it, and so does every index method
    # that is a column method underneath.
    Binding(
        mojo="index_to_series",
        name="_index_to_series",
        doc="The labels of an index, as a column that carries them twice.",
        params=(("index", "object"), ("labels", "object"), ("name", "object")),
        returns="Series",
    ),
    # The door going the other way, and here for the same reason: it reads a
    # series and answers a frame, and the import graph will not let a function
    # of a series that gives a frame live on the series. `SeriesMixin.to_frame`
    # is what calls it, and so does every series method that is a frame method
    # underneath.
    Binding(
        mojo="series_to_frame",
        name="_series_to_frame",
        doc="A column, as a frame of one column under a chosen name.",
        params=(("column", "object"), ("name", "object")),
        returns="DataFrame",
    ),
    # Not a user entry point. Every row of the error table in
    # `python/firepanda/errors.py` has to be exercised from Python, and five
    # bound methods cannot reach most of them, so the Mojo side offers a way to
    # raise one of each on request. It is registered under a leading underscore
    # and it is the only thing in the extension that is here for the tests.
    Binding(
        mojo="raise_for_test",
        name="_raise_for_test",
        doc="Raises one classified error of the given kind. For tests only.",
        params=(("kind", "str"),),
    ),
)

TYPES: tuple[Exposed, ...] = (FRAME, SERIES, INDEX)


@dataclass(frozen=True)
class Accessor:
    """One pandas namespace, which is a Python class with no extension type under it.

    `s.dt.year` is two attribute lookups and pandas answers the first one with an
    object that holds the series and does nothing else. There is no Mojo struct
    for that object and there should not be, because it holds a Python wrapper
    rather than a column and everything it does is one call on the series it was
    made from. So it is generated like the bound classes and emitted beside them,
    and it is a separate dataclass rather than an `Exposed` with the Mojo fields
    left empty, because a type with no extension behind it is a different thing
    and saying so in the schema is cheaper than a comment asking people to ignore
    four fields.
    """

    py: str
    """The class name, which is the pandas one."""

    owner: str
    """The class the accessor is reached from, and both of them where a namespace
    hangs off a column and a frame alike."""

    doc: str
    """The Python class docstring."""

    mixin: str
    """The hand written base class in `python/firepanda/_pandas.py`. Not optional
    the way it is on `Exposed`, because an accessor has no `_inner` of its own and
    something has to hold the series, so there is always logic to inherit."""

    members: tuple[Member, ...]
    """The members on the Python side."""


ACCESSORS: tuple[Accessor, ...] = (
    Accessor(
        py="DatetimeProperties",
        owner="Series",
        doc=(
            "The `dt` accessor, which is one class where pandas has two.\n\n"
            "pandas splits `DatetimeProperties` from `TimedeltaProperties` and puts a"
            " different set of names on each. This is one class carrying both sets,"
            " because the core has one `Series.dt(name)` door for both and the column's"
            " own type is what decides whether a name means anything, so splitting here"
            " would mean reading the dtype on every `.dt` just to pick which object to"
            " hand back. The visible difference is which error a caller sees: asking a"
            " timestamp column for `days` is a dtype error here and an AttributeError in"
            " pandas."
        ),
        mixin="DatetimeMixin",
        members=_datetime_members(),
    ),
    Accessor(
        py="StringAccessor",
        owner="Series",
        doc=(
            "The `str` accessor, which is the largest namespace pandas has.\n\n"
            "Reached from `s.str`, and only on a column of text, which pandas also"
            " refuses at the accessor rather than at the method: `s.str` on a column"
            " of numbers is an `AttributeError` there and here. The same reasoning"
            " the `cat` accessor gives applies, since a caller writing `s.str` has"
            " already decided what the column is.\n\n"
            "Twelve of the fifty seven names so far, and the twelve share one idea:"
            " a position in a string is a character and not a byte. Every other"
            " kernel in this library counts bytes, which is right for a `LIKE`"
            " pattern and for a sort order and is not what `s.str.len()` answers."
        ),
        mixin="StringMixin",
        members=_string_members(),
    ),
    Accessor(
        py="CategoricalAccessor",
        owner="Series",
        doc=(
            "The `cat` accessor, which is where a category column's categories"
            " live.\n\n"
            "Reached from `s.cat`, and only on a column that is a category, which is"
            " the one accessor pandas refuses to build at all rather than refusing"
            " each member: `s.cat` on a column of numbers is an `AttributeError`"
            " there and here. That is the opposite of what `dt` does, and it is not"
            " an inconsistency worth fixing, because a caller writing `s.cat` has"
            " already decided the column is a categorical and finding out at the"
            " accessor is finding out at the right place.\n\n"
            "Eleven names and three doors. A rename is decided by position, setting"
            " the categories is decided by value, and dropping the unused ones is"
            " decided by the codes. Everything else here is arithmetic over those"
            " three."
        ),
        mixin="CategoricalMixin",
        members=_categorical_members(),
    ),
    Accessor(
        py="Rolling",
        owner="Series and DataFrame",
        doc=(
            "A window of a fixed width, waiting for a reduction.\n\n"
            "Reached from `s.rolling(...)` and from `df.rolling(...)`, and it"
            " holds what it was given and the five numbers that say where each"
            " window sits rather than computing anything, which is what pandas"
            " does as well. The five are one question,"
            " `firepanda/kernel/window.mojo` states it as a pair of row numbers,"
            " and this class is where a caller's spelling of that question is"
            " checked.\n\n"
            "One class for both owners rather than pandas' two, because a window"
            " is a pair of row numbers and every column of a frame has the same"
            " rows, so a frame window is the columns windowed one at a time. The"
            " only place the difference is visible is `numeric_only`, which asks"
            " a question a frame can answer and a column cannot.\n\n"
            "Five of pandas' twenty six reductions so far, and they are the five"
            " a window can be carried through. A total can have the row that"
            " left subtracted from it and the row that arrived added to it, and"
            " a median cannot, which is the line between what is here and what"
            " is not."
        ),
        mixin="RollingMixin",
        members=_window_members("Rolling"),
    ),
    Accessor(
        py="Expanding",
        owner="Series and DataFrame",
        doc=(
            "A window that starts at the first row and grows, waiting for a"
            " reduction.\n\n"
            "Reached from `s.expanding(...)` and from `df.expanding(...)`. The"
            " same five reductions as `Rolling` over a window with no near end,"
            " which is why the two classes share everything below the"
            " constructor: an expanding window is a rolling one whose width is"
            " the height of what it reads. The one thing that is genuinely"
            " different is the default for `min_periods`, which is one here and"
            " the full width there, and pandas has the same split."
        ),
        mixin="ExpandingMixin",
        members=_window_members("Expanding"),
    ),
    Accessor(
        py="ExponentialMovingWindow",
        owner="Series and DataFrame",
        doc=(
            "A decay over every row so far, waiting for a reduction.\n\n"
            "Reached from `s.ewm(...)` and from `df.ewm(...)`. It is not a"
            " narrower `Rolling`, and the difference is worth stating before"
            " anybody goes looking for the shared base class: every other window"
            " in this library is a pair of row numbers, and this one has no"
            " edges at all. Every row is inside every window and what changes"
            " from row to row is how much each earlier row counts, which falls"
            " off geometrically with distance. So there is no width here, no"
            " centring, no closed rule and no step, and the seven properties"
            " that answer those questions answer them with nothing."
            " `firepanda/kernel/ewm.mojo` argues the whole of that.\n\n"
            "One class for both owners, for the reason `Rolling` gives. The"
            " decay runs down a column and every column of a frame has the same"
            " rows, so a frame decays a column at a time. That is pandas'"
            " `method=\"single\"`, which is its default.\n\n"
            "Four of pandas' nine reductions so far. The decay arrives as one of"
            " four numbers that mean the same thing and is collapsed to one"
            " before it crosses the boundary, which is checked here because this"
            " is where a caller's spelling of the question lives."
        ),
        mixin="EwmMixin",
        members=_ewm_members(),
    ),
    Accessor(
        py="DataFrameGroupBy",
        owner="DataFrame",
        doc=(
            "A frame with a grouping over it, waiting for a reduction.\n\n"
            "Reached from `df.groupby(...)` rather than from an attribute, which is"
            " the one way this differs from the accessor above and is why it holds"
            " the keys and the flags as well as the frame. Nothing is computed until"
            " a reduction is asked for, as in pandas, and nothing about the grouping"
            " is kept afterwards: `groups`, `indices` and `get_group` are absent"
            " rather than slow, because keeping an index per group whether or not"
            " anybody asks is the cost this library exists to not pay."
        ),
        mixin="DataFrameGroupByMixin",
        members=_group_members("DataFrameGroupBy"),
    ),
    Accessor(
        py="SeriesGroupBy",
        owner="DataFrameGroupBy",
        doc=(
            "One column of a grouped frame, waiting for a reduction.\n\n"
            "Reached from `df.groupby(...)[name]`. The same fifteen reductions over"
            " one column instead of all of them, answering a column rather than a"
            " frame, which is the whole difference between the two classes."
        ),
        mixin="SeriesGroupByMixin",
        members=_group_members("SeriesGroupBy"),
    ),
)

BANNER_MOJO = (
    "# Generated by tools/bindings.py. Do not edit.\n"
    "#\n"
    "# Run `python tools/bindings.py` after changing the table in that file.\n"
    "# CI runs it with --check and fails if this file is out of date.\n"
)

BANNER_PY = (
    "# Generated by tools/bindings.py. Do not edit.\n"
    "#\n"
    "# Run `python tools/bindings.py` after changing the table in that file.\n"
    "# CI runs it with --check and fails if this file is out of date.\n"
)


MOJO_COLUMNS = 80

PYTHON_COLUMNS = 100
"""What `ruff format` wraps at, which is the `line-length` in `pyproject.toml`.
Both numbers are here rather than read out of the config, because the layouts
below reproduce a formatter's output by hand and a config change should fail
loudly rather than quietly produce a file that no longer matches."""


def _python_def(indent: str, name: str, params: list[str], returns: str) -> list[str]:
    """Writes one `def` line the way `ruff format` would have written it.

    Same problem as `_register_call` on the Mojo side and the same reason for
    solving it here: `ruff format --check` runs over the generated files, so a
    signature that is merely valid is not enough, it has to be the text the
    formatter produces. ruff has three layouts and takes the first that fits: all
    on one line, then the parameters together on one continuation line, then one
    parameter per line with a trailing comma.

    Args:
        indent: The indent the `def` sits at.
        name: The function name.
        params: The parameters, `self` included, each already annotated.
        returns: The return annotation.

    Returns:
        The lines up to and including the colon.
    """
    joined = ", ".join(params)
    one = f"{indent}def {name}({joined}) -> {returns}:"
    if len(one) <= PYTHON_COLUMNS:
        return [one]
    together = f"{indent}    {joined}"
    if len(together) <= PYTHON_COLUMNS:
        return [f"{indent}def {name}(", together, f"{indent}) -> {returns}:"]
    return (
        [f"{indent}def {name}("]
        + [f"{indent}    {part}," for part in params]
        + [f"{indent}) -> {returns}:"]
    )


def _register_call(opener: str, name: str, doc: str) -> list[str]:
    """Writes one registration call the way `mojo format` would have written it.

    The generated file is checked by `tools/format_check.sh` like any other Mojo
    source, so emitting a call that is merely valid is not enough, it has to be
    the exact text the formatter produces, and getting that wrong shows up as a
    format failure on a generated file, which is a confusing thing to be handed.

    The formatter has three layouts and takes the first that fits in eighty
    columns: the name and the docstring on one line, then each on its own line,
    then the docstring in brackets on a line of its own at an indent of twelve.
    All three are reproduced here, because each of the three has turned up as
    soon as a docstring crossed the length that provokes it.

    Args:
        opener: The call up to and including the open bracket.
        name: The name to register under.
        doc: The docstring for it.

    Returns:
        The lines of the call, including the closing bracket.
    """
    whole = f'{opener}"{name}", docstring="{doc}")'
    if len(whole) <= MOJO_COLUMNS:
        return [whole]
    short = f'        "{name}", docstring="{doc}"'
    if len(short) <= MOJO_COLUMNS:
        return [opener, short, "    )"]
    own_line = f'        docstring="{doc}",'
    if len(own_line) <= MOJO_COLUMNS:
        return [opener, f'        "{name}",', own_line, "    )"]
    if len(doc) + 14 > MOJO_COLUMNS:
        raise SystemExit(
            f"the docstring for {name} is too long for the generator to lay out"
            " the way mojo format wants, which needs it to fit on one line at an"
            f" indent of twelve. Shorten it to {MOJO_COLUMNS - 14} characters or"
            f" fewer, it is currently {len(doc)}."
        )
    return [
        opener,
        f'        "{name}",',
        "        docstring=(",
        f'            "{doc}"',
        "        ),",
        "    )",
    ]


def _import(module: str, names: list[str]) -> list[str]:
    """Writes one import the way `mojo format` would have written it.

    Same problem as `_register_call` and the same reason for solving it here: the
    generated file is format checked like any other source, so an import that is
    merely valid is not enough. Over eighty columns the formatter puts the names
    in brackets, one per line, with a trailing comma.

    Args:
        module: The module to import from.
        names: The names to import, already sorted.

    Returns:
        The lines of the import.
    """
    one = f"from {module} import " + ", ".join(names)
    if len(one) <= MOJO_COLUMNS:
        return [one]
    return [f"from {module} import ("] + [f"    {name}," for name in names] + [")"]


def _import_order(name: str) -> tuple[int, str]:
    """Sorts imported names the way ruff's import rule wants them sorted.

    Three groups rather than one alphabetical run: a constant in capitals first,
    then the classes, then anything lowercase, and alphabetical inside each. That
    is `force-sort-within-sections` order and it is what the check compares
    against, so writing the names in plain sorted order fails the lint even
    though the import itself is correct.

    Inside a group the comparison ignores case, which is invisible until a name
    starting with an underscore and carrying a capital arrives. A hand written
    helper class is exactly that, and `_Cell` sorts before `_ewm` here and after
    it under a plain sort, because every capital letter is below every lowercase
    one in ASCII and ruff is not comparing them that way.

    Args:
        name: The imported name.

    Returns:
        The key to sort on.
    """
    if name.isupper():
        return (0, name.lower())
    if name[:1].isupper():
        return (1, name.lower())
    return (2, name.lower())


def _imported(names: list[str]) -> list[str]:
    """Writes the one Python import that can outgrow its line.

    The same job `_import` does for Mojo, against the Python line limit and with
    Python's own bracket style, which is one name per line with a trailing comma.
    Only `._pandas` needs it, because it is the only import whose contents grow
    every time a class is added to the table.

    Args:
        names: The names to import, already in the order ruff wants.

    Returns:
        The lines of the import.
    """
    one = "from ._pandas import " + ", ".join(names)
    if len(one) <= PYTHON_COLUMNS:
        return [one]
    return ["from ._pandas import ("] + [f"    {name}," for name in names] + [")"]


def _listed(name: str, items: list[str]) -> list[str]:
    """Writes a list literal the way `ruff format` would have written it.

    On one line while it fits and one item per line with a trailing comma once it
    does not, which is the formatter's rule for any collection. `__all__` is the
    only list here long enough to have crossed the limit.

    Args:
        name: What the list is being assigned to.
        items: The items, already written as source.

    Returns:
        The lines of the assignment.
    """
    one = f"{name} = [" + ", ".join(items) + "]"
    if len(one) <= PYTHON_COLUMNS:
        return [one]
    return [f"{name} = ["] + [f"    {item}," for item in items] + ["]"]


def registration() -> str:
    """Writes the Mojo file that registers everything.

    Returns:
        The file contents.
    """
    out = [BANNER_MOJO]
    out.append('"""The registration calls, one per binding.')
    out.append("")
    out.append("This is the flat sequence document 13 section 7 explains cannot be")
    out.append("a loop.")
    out.append('"""')
    out.append("")
    out.append("from std.python import PythonObject")
    out.append("from std.python.bindings import PythonModuleBuilder\n")
    wanted: dict[str, set[str]] = {}
    for fn in FUNCTIONS:
        wanted.setdefault("firepanda.py.frame", set()).add(fn.mojo.split(".")[0])
    for t in TYPES:
        wanted.setdefault(t.module, set()).add(t.mojo)
    for module in sorted(wanted):
        out.extend(_import(module, sorted(wanted[module])))
    out.append("")
    out.append("")
    out.append("def register(mut module: PythonModuleBuilder) raises:")
    out.append('    """Registers every binding on the module.')
    out.append("")
    out.append("    Args:")
    out.append("        module: The builder to register on.")
    out.append('    """')

    for fn in FUNCTIONS:
        out.extend(_register_call(f"    module.def_function[{fn.mojo}](", fn.name, fn.doc))

    for t in TYPES:
        out.append("")
        out.append(f'    ref {t.name.lower()} = module.add_type[{t.mojo}]("{t.name}")')
        if t.init:
            out.append(f"    _ = {t.name.lower()}.def_py_init[{t.init}]()")
        for b in t.bindings:
            out.extend(
                _register_call(f"    _ = {t.name.lower()}.def_method[{b.mojo}](", b.name, b.doc)
            )
    return "\n".join(out) + "\n"


def stubs() -> str:
    """Writes the type stubs for the private extension module.

    The extension is not the public API, so this file stays small on purpose. It
    describes the narrow convention so that the generated Python layer type
    checks against something, and it is not what a user's autocomplete reads.

    Returns:
        The file contents.
    """
    out = [BANNER_PY]
    out.append('"""Stubs for the compiled extension, which is private.')
    out.append("")
    out.append("The public API is `firepanda`, whose annotations are inline.")
    out.append('"""\n')
    for t in TYPES:
        out.append(f"class {t.name}:")
        if t.init_params:
            args = "".join(f", {name}: {kind}" for name, kind in t.init_params)
            out.append(f"    def __init__(self{args}) -> None:")
            out.append('        """Builds one. The Python layer owns the pandas signature."""')
            out.append("        ...")
            out.append("")
        for b in t.bindings:
            args = ["self"] + [f"{name}: {kind}" for name, kind in b.params]
            out.extend(_python_def("    ", b.name, args, b.returns))
            out.append(f'        """{b.doc}"""')
            out.append("        ...")
        out.append("")
    for at, fn in enumerate(FUNCTIONS):
        if at:
            out.append("")
        args = ", ".join(f"{name}: {kind}" for name, kind in fn.params)
        out.append(f"def {fn.name}({args}) -> {fn.returns}:")
        out.append(f'    """{fn.doc}"""')
        out.append("    ...")
    out.append("")
    out.append("def version() -> str:")
    out.append('    """The version the extension was built from."""')
    out.append("    ...")
    return "\n".join(out) + "\n"


def _members(members: tuple[Member, ...]) -> list[str]:
    """Writes the members of one Python class.

    Shared by the bound types and the accessors, which are different in what
    holds them up and identical in what a member looks like once it is written.

    Args:
        members: The members, in the order they should appear.

    Returns:
        The lines, each already indented for a class body.
    """
    out: list[str] = []
    for m in members:
        out.append("")
        # An accessor is a class attribute rather than anything callable, so it is
        # written and then documented, which is how a bare attribute carries a
        # docstring. It is not a property because a property read off the class
        # answers itself, and pandas answers the accessor class there.
        if m.kind == "accessor":
            out.append(f"    {m.name} = Namespace({m.body})")
            out.extend(_docstring(m.doc, "    "))
            continue
        if m.kind == "property":
            out.append("    @property")
            out.append(f"    def {m.name}(self) -> {m.returns}:")
        else:
            params = ["self"] + (m.signature.split(", ") if m.signature else [])
            out.extend(_python_def("    ", m.name, params, m.returns))
        out.extend(_docstring(m.doc, "        "))
        body = f"{m.wraps}._wrap({m.body})" if m.wraps else m.body
        out.extend(_guarded(f"return {body}", "        "))
    return out


def wrapper() -> str:
    """Writes the Python classes a user actually holds.

    Returns:
        The file contents.
    """
    out = [BANNER_PY]
    out.append('"""The pandas surface.')
    out.append("")
    out.append("Every class here holds an extension object and delegates to it. The reason")
    out.append("it is not the extension object itself is document 13: a bound Mojo type")
    out.append("cannot carry a property, an operator or a dunder, cannot be subclassed and")
    out.append("has no __dict__, so 28 percent of pandas is unreachable from there.")
    out.append("")
    out.append("Every delegation is wrapped, because a Mojo error arrives as a bare")
    out.append("Exception and `errors.translate` is what puts the class back. The try costs")
    out.append("nothing when nothing raises, which is measured in document 14.")
    out.append('"""\n')
    out.append("from __future__ import annotations\n")
    # Only emitted when something in the table actually asks for it, since an
    # import nothing uses is a lint failure rather than a harmless extra line.
    standard = []
    every = [m for t in TYPES for m in t.members] + [m for a in ACCESSORS for m in a.members]
    if any("Sequence[" in (m.signature or "") for m in every):
        standard.append("from collections.abc import Sequence")
    if any("Any" in (m.signature or "") or m.returns == "Any" for m in every):
        standard.append("from typing import Any")
    if standard:
        out.extend(standard)
        out.append("")
    out.append("from . import _firepanda")
    mixins = {t.mixin for t in TYPES if t.mixin} | {a.mixin for a in ACCESSORS}
    if any(m.kind == "accessor" for m in every):
        mixins.add("Namespace")
    if any("NO_DEFAULT" in (m.signature or "") for m in every):
        mixins.add("NO_DEFAULT")
    # `df.groupby(...)` is the one member whose body is a call to a hand written
    # function rather than to a method on something it already holds, because
    # what it builds is a different class from the one it is written on and the
    # arguments have to be read before there is an object to read them into.
    if any("_grouped(" in m.body for m in every):
        mixins.add("_grouped")
    # `s.rolling(...)` and `s.expanding(...)` are the same case as `groupby`
    # above, for the same reason, so they are hand written functions too.
    for builder in ("_rolling", "_expanding", "_ewm"):
        if any(f"{builder}(" in m.body for m in every):
            mixins.add(builder)
    # `df.iloc` and the three properties beside it are the same case a third
    # time. Each answers a small object that holds the frame and reads a key,
    # and the key is the whole of what they do, so the class is hand written
    # and the member is the one line that builds one.
    for accessor in ("_Positional", "_Labelled", "_Cell", "_Along", "_Point"):
        if any(f"{accessor}(" in m.body for m in every):
            mixins.add(accessor)
    if mixins:
        out.extend(_imported(sorted(mixins, key=_import_order)))
    out.append("from .errors import translate")

    out.append("")
    named = sorted([t.py for t in TYPES] + [a.py for a in ACCESSORS])
    out.extend(_listed("__all__", [f'"{n}"' for n in named]))

    # The accessors come first because a bound type reaches one through a class
    # body assignment, which runs while the class is being built and would find
    # nothing if the accessor class were still further down the file.
    for a in ACCESSORS:
        out.append("")
        out.append("")
        out.append(f"class {a.py}({a.mixin}):")
        out.extend(_docstring(a.doc, "    "))
        out.append("")
        out.append("    __slots__ = ()")
        out.extend(_members(a.members))

    for t in TYPES:
        out.append("")
        out.append("")
        base = f"({t.mixin})" if t.mixin else ""
        out.append(f"class {t.py}{base}:")
        out.extend(_docstring(t.doc, "    "))
        out.append("")
        out.append("    __slots__ = ()" if t.mixin else '    __slots__ = ("_inner",)')
        out.append("")
        out.append("    @classmethod")
        out.append(f"    def _wrap(cls, inner: _firepanda.{t.name}) -> {t.py}:")
        out.append('        """Puts the wrapper around an extension object.')
        out.append("")
        out.append("        Not a public entry point. It allocates without going through")
        out.append("        __init__ because __init__ is the pandas constructor, which takes")
        out.append("        data rather than an extension object.")
        out.append('        """')
        out.append("        self = object.__new__(cls)")
        out.append("        self._inner = inner")
        out.append("        return self")
        if not t.constructed:
            out.append("")
            out.append(f"    def __init__(self, inner: _firepanda.{t.name}) -> None:")
            out.append('        """Wraps an extension object. Not a public entry point."""')
            out.append("        self._inner = inner")

        out.extend(_members(t.members))

    for fn in FUNCTIONS:
        params = fn.py_params or fn.params
        args = ", ".join(f"{name}: {kind}" for name, kind in params)
        passed = ", ".join(name for name, _ in params)
        out.append("")
        out.append("")
        out.append(f"def {fn.name}({args}) -> {fn.returns}:")
        out.append(f'    """{fn.doc}"""')
        wrap = fn.returns if fn.returns in {t.py for t in TYPES} else ""
        call = f"_firepanda.{fn.name}({passed})"
        body = f"return {wrap}._wrap({call})" if wrap else f"return {call}"
        out.extend(_guarded(body, "    "))
    return "\n".join(out) + "\n"


OUTPUTS: tuple[tuple[str, str], ...] = (
    ("firepanda/py/_registration.mojo", "registration"),
    ("python/firepanda/_firepanda.pyi", "stubs"),
    ("python/firepanda/_frame.py", "wrapper"),
)


def main() -> int:
    """Writes the generated files, or checks them.

    Returns:
        A process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if a generated file on disk differs from what the table says",
    )
    args = parser.parse_args()

    stale = []
    for relative, name in OUTPUTS:
        path = REPO / relative
        wanted = globals()[name]()
        if args.check:
            found = path.read_text() if path.exists() else ""
            if found != wanted:
                stale.append(relative)
        else:
            path.write_text(wanted)
            print(f"wrote {relative}")

    if stale:
        print(
            "these files are not what tools/bindings.py says they should be:",
            file=sys.stderr,
        )
        for relative in stale:
            print(f"  {relative}", file=sys.stderr)
        print("run: python tools/bindings.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
