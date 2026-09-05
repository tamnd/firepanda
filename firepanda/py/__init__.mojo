"""The Python front door.

Everything in this package exists to be called from CPython and nothing in it is
called from Mojo. That is the whole of the boundary rule: a file here may import
from anywhere in firepanda, and no file anywhere else in firepanda may import
from here, because the moment a kernel knows a `PythonObject` exists the engine
has a dependency on an interpreter it was built not to need.

`module.mojo` is the entry point CPython looks for and is the only file with an
`@export` in it. `docs/specs/07-python-bindings.md` describes what the rest of
this package becomes: a declarative binding table, generated registration and
stubs, the PyCapsule conversion layer, and the error mapping.

`docs/specs/12-the-python-front-door-measured.md` is document 07 checked against
a running toolchain and is the one to read first, because three of document 07's
claims moved when they were measured and two of the things it treats as details
turned out to be the parts that do not work yet.
"""
