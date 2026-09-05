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

    bindings: tuple[Binding, ...] = ()
    """The methods on the extension side."""

    members: tuple[Member, ...] = ()
    """The members on the Python side."""

    mixin: str = ""
    """A hand written base class in `python/firepanda/_pandas.py` for the members
    that are not a plain delegation. The note at the top of this file asks for
    exactly this rather than for expressions in the table growing logic, and
    `DataFrame.__getitem__` is the member that reached it: what `df[key]` does
    depends on what `key` is, and a conditional smuggled into a `body` string
    would be code in a table."""


FRAME = Exposed(
    mojo="PyDataFrame",
    name="DataFrame",
    py="DataFrame",
    doc=("A two dimensional labelled data structure with columns of potentially different types."),
    init="PyDataFrame.py_init",
    mixin="DataFrameMixin",
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
    ),
)

SERIES = Exposed(
    mojo="PySeries",
    name="Series",
    py="Series",
    doc="A one dimensional labelled array holding data of a single type.",
    init="PySeries.py_init",
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

TYPES: tuple[Exposed, ...] = (FRAME, SERIES)

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
    mojo_types = sorted({t.mojo for t in TYPES})
    imports = sorted({b.mojo.split(".")[0] for b in FUNCTIONS} | set(mojo_types))
    out.append("from firepanda.py.frame import " + ", ".join(imports) + "\n")
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
        for b in t.bindings:
            args = "".join(f", {name}: {kind}" for name, kind in b.params)
            out.append(f"    def {b.name}(self{args}) -> {b.returns}:")
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
    out.append("from . import _firepanda")
    mixins = sorted({t.mixin for t in TYPES if t.mixin})
    if mixins:
        out.append("from ._pandas import " + ", ".join(mixins))
    out.append("from .errors import translate")

    out.append("")
    out.append("__all__ = [" + ", ".join(f'"{t.py}"' for t in TYPES) + "]")

    for t in TYPES:
        out.append("")
        out.append("")
        base = f"({t.mixin})" if t.mixin else ""
        out.append(f"class {t.py}{base}:")
        out.extend(_docstring(t.doc, "    "))
        out.append("")
        out.append('    __slots__ = ("_inner",)')
        out.append("")
        if t.init:
            # `inner` defaults so that `firepanda.DataFrame()` reaches the Mojo
            # refusal, which says how to build one, rather than a missing
            # argument complaint about a parameter no user was ever meant to
            # pass. When construction from Python does get written, this is
            # where it arrives.
            out.append(f"    def __init__(self, inner: _firepanda.{t.name} | None = None) -> None:")
            out.append('        """Wraps an extension object. Not a public entry point."""')
            out.extend(
                _guarded(f"inner = _firepanda.{t.name}() if inner is None else inner", "        ")
            )
            out.append("        self._inner = inner")
        else:
            out.append(f"    def __init__(self, inner: _firepanda.{t.name}) -> None:")
            out.append('        """Wraps an extension object. Not a public entry point."""')
            out.append("        self._inner = inner")

        for m in t.members:
            out.append("")
            if m.kind == "property":
                out.append("    @property")
                out.append(f"    def {m.name}(self) -> {m.returns}:")
            else:
                sig = f", {m.signature}" if m.signature else ""
                out.append(f"    def {m.name}(self{sig}) -> {m.returns}:")
            out.append(f'        """{m.doc}"""')
            body = f"{m.wraps}({m.body})" if m.wraps else m.body
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
        body = f"return {wrap}({call})" if wrap else f"return {call}"
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
