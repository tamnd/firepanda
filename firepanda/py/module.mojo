"""The symbol CPython looks for.

Python loads a compiled extension by finding a shared library that exports
`PyInit_<name>()` and calling it. This file is that function, and almost nothing
else. The bindings themselves are in `_registration.mojo`, which is generated
from the table in `tools/bindings.py`, so this file holds only the two things
that cannot be generated: the exported symbol, and `version`, which exists to
check the build rather than to be used.

Document 13 section 7 says why the registration is a generated flat sequence of
calls rather than a table something iterates over: `def_method` will not accept a
function that has passed through anything generic, and a collection of bindings
cannot be given a type at all.

### The name is `_firepanda`, not `firepanda`

`docs/specs/07-python-bindings.md` says `PyInit_firepanda` in section 1 and
`_firepanda.so` in section 7, and those two cannot both be right. The extension
is the underscore one, because `python/firepanda/__init__.py` is the package a
user imports and an extension of the same name would shadow it. So the symbol is
`PyInit__firepanda` with two underscores, the first belonging to the protocol and
the second to the module name, and the module is reached as
`firepanda._firepanda`.

The module name in the symbol, the string given to `PythonModuleBuilder` and the
file name of the built library all have to agree, and nothing checks that for
you. A mismatch produces an `ImportError` about a missing init function, which
names the symbol it wanted and not the reason it is missing, so it is worth
reading those three spellings together whenever this file is touched.

### Three things about the declaration

`def`, not `fn`. `fn` was removed from Mojo before 1.0 and document 07's snippet
predates that.

`abi("C")` in the effects position, or the symbol is emitted with Mojo's own
mangling and CPython does not find it. The failure is the same `ImportError` as
a name mismatch, from a different cause.

`abort` is called for its effect rather than returned. There is no useful thing
to return from a failed module init: CPython is going to raise `SystemError` on a
null with no exception set, and a message on the way down is worth more than a
tidier signature.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from firepanda.py._registration import register
from firepanda.version import VERSION


@export
def PyInit__firepanda() abi("C") -> PythonObject:
    """Builds the `firepanda._firepanda` module. Called once, by CPython.

    Returns:
        The module object.
    """
    try:
        var module = PythonModuleBuilder("_firepanda")
        module.def_function[report_version](
            "version",
            docstring=(
                "The firepanda version the extension was built from, as a"
                " string."
            ),
        )
        register(module)

        return module.finalize()
    except e:
        abort(t"firepanda extension init failed: {e}")


def report_version() raises -> PythonObject:
    """Reports the version the extension was built from.

    This is a check rather than a feature, and it is the one binding that is
    written here by hand rather than generated, because it is about the artifact
    and not about the library. A wheel that imports and then reports a
    version from a different build than the Python package around it is a wheel
    that was assembled wrongly, and the two halves have no other way to notice.

    Returns:
        The version string.
    """
    return PythonObject(String(VERSION))
