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

    Args:
        text: The docstring, as one long line.
        indent: The indent to put in front of every line.

    Returns:
        The lines to emit.
    """
    width = 88 - len(indent)
    if len(text) <= width:
        return [f'{indent}"""{text}"""']
    words, lines, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    lines.append(current)
    return [f'{indent}"""{lines[0]}'] + [indent + line for line in lines[1:]] + [f'{indent}"""']


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
    return [
        f"{indent}try:",
        f"{indent}    {statement}",
        f"{indent}except Exception as error:",
        f"{indent}    raise translate(error) from None",
    ]


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
            mojo="PyIndex.get_indexer",
            name="get_indexer",
            doc="Where each of a set of labels sits, with -1 for the missing.",
            params=(("target", "object"),),
            returns="list[int]",
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
            name="unique",
            kind="method",
            body="self._inner.unique()",
            doc="The index with each label kept once, in first seen order.",
            returns="Index",
            wraps="Index",
        ),
        Member(
            name="rename",
            kind="method",
            signature="name: str | None",
            body="self._inner.renamed(name)",
            doc="The index under a different level name.",
            returns="Index",
            wraps="Index",
        ),
        Member(
            name="take",
            kind="method",
            signature="indices: Sequence[int]",
            body="self._inner.take(list(indices))",
            doc="The labels at a set of positions, in the order given.",
            returns="Index",
            wraps="Index",
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
            name="slice_locs",
            kind="method",
            signature="start: object = None, end: object = None",
            body="tuple(self._inner.slice_locs(start, end))",
            doc="The half open row range a pair of labels describes, both ends inclusive.",
            returns="tuple[int, ...]",
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
    if any("Sequence[" in (m.signature or "") for t in TYPES for m in t.members):
        standard.append("from collections.abc import Sequence")
    if any("Any" in (m.signature or "") or m.returns == "Any" for t in TYPES for m in t.members):
        standard.append("from typing import Any")
    if standard:
        out.extend(standard)
        out.append("")
    out.append("from . import _firepanda")
    mixins = sorted({t.mixin for t in TYPES if t.mixin})
    if mixins:
        out.append("from ._pandas import " + ", ".join(mixins))
    out.append("from .errors import translate")

    out.append("")
    out.append("__all__ = [" + ", ".join(f'"{n}"' for n in sorted(t.py for t in TYPES)) + "]")

    for t in TYPES:
        out.append("")
        out.append("")
        base = f"({t.mixin})" if t.mixin else ""
        out.append(f"class {t.py}{base}:")
        out.extend(_docstring(t.doc, "    "))
        out.append("")
        out.append('    __slots__ = ()' if t.mixin else '    __slots__ = ("_inner",)')
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

        for m in t.members:
            out.append("")
            if m.kind == "property":
                out.append("    @property")
                out.append(f"    def {m.name}(self) -> {m.returns}:")
            else:
                params = ["self"] + (m.signature.split(", ") if m.signature else [])
                out.extend(_python_def("    ", m.name, params, m.returns))
            out.append(f'        """{m.doc}"""')
            body = f"{m.wraps}._wrap({m.body})" if m.wraps else m.body
            out.extend(_guarded(f"return {body}", "        "))

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
