"""Types for the compiled extension.

Mojo emits no type information that a Python type checker can read, so this file
is the only description of the extension's surface that mypy and editors get.
It is written by hand today and generated later: `docs/specs/07-python-bindings.md`
section 5 has the binding table producing both the registration code and these
stubs from one declaration, which is the point at which a stub can stop being a
thing somebody remembers to update.

Until then, treat a change to `firepanda/py/` that does not change this file as
unfinished.
"""

def version() -> str: ...
