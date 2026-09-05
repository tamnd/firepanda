# Errors across the boundary

Document 07 section 5 has a seven row table mapping firepanda's error taxonomy onto CPython exception classes, and it says the table is tested. Document 12 section 5 then found that a bound Mojo function can raise exactly one Python exception class, and that class is `Exception`. Both of those are true, and together they describe a hole rather than a design: there is a table saying what should happen and a measurement saying the obvious way of making it happen does not exist.

This document is how that hole gets filled. It is short because the answer is small, and it is written down because the answer is a wire format between two halves of one library, and a wire format nobody wrote down is a wire format that drifts.

## 1. Why this matters more than it looks like it does

An exception class is part of an API in a way that is easy to underrate, because it never shows up in a signature and never shows up in a conformance case. Every conformance case in `firepanda-compat` compares an answer. This is about what happens when there is no answer, and a library can pass every one of those cases while being unusable in the code people actually have.

The code people actually have looks like this, and it predates this project by fifteen years.

```python
try:
    frame = frame.groupby(key).sum()
except KeyError:
    return fallback_for_missing_column(key)
```

Against a firepanda that raises a bare `Exception`, that `except KeyError` never fires, the exception propagates past a handler written specifically for it, and the failure surfaces somewhere unrelated with a message that is correct and a class that is wrong. Nothing in the test suite of the calling project would catch it either, because their test for that path asserts a `KeyError` is caught and it was, back when the frame was a pandas frame.

So the table in document 07 section 5 is not a nicety. It is the difference between a drop in replacement and a rewrite.

## 2. The class cannot cross the boundary, so the name does

There is no API in the Mojo Python bindings for choosing which Python exception a raise turns into. `raise Error("...")` in a `def_function` or a `def_method` arrives as `Exception`, always, and the string survives intact. That is the entire mechanism available.

Since the string survives and nothing else does, the string is where the class goes. The Mojo side prefixes the message with `firepanda:<kind>: ` and the Python side reads the prefix off, looks the kind up, and raises the class the table promises with the rest of the message.

```
Mojo:    raise Error("firepanda:column: no such column 'regoin'")
wire:    Exception("firepanda:column: no such column 'regoin'")
Python:  ColumnNotFoundError("no such column 'regoin'")
```

The two halves are `firepanda/py/errors.mojo`, which holds the six prefixes and two helpers for applying them, and `python/firepanda/errors.py`, which holds the classes and the lookup. There is no third place, and in particular the generated code in `python/firepanda/_frame.py` knows nothing about kinds. It knows only to call `translate` on whatever comes out.

A prefix on a message is a blunt instrument and it is worth being honest about why it is acceptable here. It is unambiguous in practice, because the prefix is a fixed literal that nothing else in the codebase writes and it is only ever read off the front of the string. It costs nothing at runtime. It degrades safely, because a message that arrives without a recognised prefix still becomes something better than `Exception`. And the alternative, which would be to build the Python exception object in Mojo and raise it through the CPython C API by hand, buys a cleaner mechanism at the cost of reaching under the binding layer for every raise site in the library.

## 3. The binding tags, not the core

The obvious place to apply the tag is at the `raise`. That is wrong, and the reason is worth stating because it is the one design decision in this document that could reasonably have gone the other way.

There are 292 `raise Error(` sites in the Mojo core. Almost none of them know what they mean to a Python caller. A bounds check inside a take kernel is a `ValueError` when the user passed a bad `n` and it is a `RuntimeError` when the planner produced a bad index, and the kernel cannot tell those apart, because by the time control reaches it the reason it was called has been thrown away. Tagging at the raise site would mean either guessing, or threading a classification argument down through every kernel signature so that the answer arrives with the call.

The boundary does know. `open_csv` knows that everything which goes wrong reading a file is an `OSError` to a Python caller, whatever the reader itself said. `_int` knows that a bad `n` is a `TypeError` and it knows the argument is called `n`. So the tag is applied at the boundary, by the binding, on the way out, and the core keeps raising untagged errors with good messages in them.

That is what `retagged` is for. It takes an error that came up from the core and puts a classification on the front of it without touching what the core wrote, because the core's message says which file and what the operating system said about it, which is more than the binding knows.

```mojo
try:
    return PythonObject(alloc=PyDataFrame(read_csv(String(path))))
except cause:
    raise retagged(IO, cause)
```

The cost of this decision is that a path with no `try` on it leaks an untagged error, which becomes a `RuntimeError`. That is the last row of document 07's table and it is the honest answer for an error nobody classified, so the failure mode of forgetting is a slightly vague exception rather than a wrong one.

## 4. The classes are named, and they are still the builtins

Document 07's table gives builtins. Raising the builtins directly would satisfy it and would produce tracebacks ending in `TypeError: cannot add int64 and float64`, which tells a reader what went wrong and not where it came from.

So each kind gets a named class that subclasses the builtin its row promises.

| kind | class | catchable as |
|---|---|---|
| `column` | `ColumnNotFoundError` | `KeyError` |
| `dtype` | `DTypeError` | `TypeError` |
| `value` | `InvalidArgumentError` | `ValueError` |
| `io` | `ReaderError` | `OSError` |
| `unsupported` | `UnsupportedError` | `NotImplementedError` |
| `cancelled` | `CancelledError` | `KeyboardInterrupt` |
| anything else | `RuntimeError` | `RuntimeError` |

Every one of the first six also subclasses `FirepandaError`, which carries no behaviour and exists so that `except firepanda.errors.FirepandaError` can mean "anything this library considers its own fault or its user's". A traceback now reads `DTypeError` and an `except TypeError` written years ago still fires.

There is one deliberate exception to that, and it is a trap I walked into while writing this and want to leave a sign on. `CancelledError` does not subclass `FirepandaError`. `FirepandaError` is an `Exception`, `KeyboardInterrupt` is a `BaseException` and specifically not an `Exception`, and a class inheriting from both linearises with `Exception` in its ancestry. The result is a cancellation that any library with a bare `except Exception` in it will quietly swallow, which is the exact outcome `KeyboardInterrupt` sits outside `Exception` to prevent. So the marker is what gets dropped, the `BaseException` is what gets kept, and `test_a_cancellation_is_not_swallowed_by_except_exception` asserts it in those words.

`RuntimeError` gets no named class, because the whole point of that row is that it is the errors nobody classified, and a `FirepandaUnknownError` would suggest a category that had been thought about.

## 5. Two things that arrive already broken

Two errors reach the Python layer from outside firepanda's own raises, and both are handled in `translate` rather than in a binding, because neither one has a binding to handle it in.

The first is the arity error from document 13 section 5. Calling a bound method with the wrong number of arguments produces `Exception: TypeError: <mojo function>() takes 1 positional argument but 2 were given`, with the right words in the message and the wrong class on the object. Since the message is already formatted as `TypeError: ...`, `translate` reads that leading name and puts the class back, for the five builtins the binding layer is known to produce this way. Calling a method wrongly now raises a `TypeError`.

The second is the constructor path, also from document 13 section 5. A `raise Error(...)` inside a `def_py_init` arrives as `ValueError` rather than `Exception`, so `translate` treats an incoming `ValueError` the same way it treats an `Exception`. Anything more specific than those two is left alone, on the grounds that a `MemoryError` is Python's own and reclassifying it would be worse than leaving it.

## 6. Every generated method is wrapped, and it costs nothing

`translate` has to be called somewhere, and the somewhere is every method body in `python/firepanda/_frame.py`. Since that file is generated from the table in `tools/bindings.py`, the wrapping is generated too and there is no way to write a member that forgets it.

```python
def head(self, n: int = 5) -> DataFrame:
    """The first n rows."""
    try:
        return DataFrame(self._inner.head(n))
    except Exception as error:
        raise translate(error) from None
```

The `from None` is there so a traceback does not print the same sentence twice and call the second copy the direct cause of the first. What is being suppressed is the binding layer's untyped wrapper around a message this library wrote, and there is nothing in it a reader wants.

Wrapping unconditionally is only defensible if it is free, so it was measured rather than assumed. Python 3.11 moved to zero cost exception handling, in which a `try` block that does not raise costs nothing at runtime, and the question was whether that holds on the scale of a delegating method call. Two functions identical apart from the `try`, 200,000 calls each, interleaved over 25 rounds taking the minimum of each, because a naive back to back comparison on this machine drifts by up to 28 nanoseconds on the same function and would have been noise reported as a result.

```
unguarded  99.8 ns
guarded   100.1 ns
delta      +0.3 ns
```

Against a delegation that is already about 100 nanoseconds, and against a `head` call that does real work behind it, that is not a number worth designing around. So there is no opt out, and no member has to be reasoned about individually.

## 7. Holding the two halves to the same table

The failure this design is exposed to is drift. A kind added in Mojo and not in Python, or renamed on one side, produces a `RuntimeError` with a `firepanda:` prefix still visible in its message, at whatever moment a user first triggers that path.

So the tests do not check the Python half against strings written in the test file, which would test one half against itself. `firepanda/py/frame.mojo` exposes `raise_for_test`, registered as `_raise_for_test`, which raises one classified error of each kind on request. It is the one entry point in the extension that exists for the tests rather than for a user, and it is there because the bound surface is five methods and none of them can reach most of the rows.

`python/tests/test_errors.py` drives every row through it from Python and asserts the class rather than the message. On top of that it parses the prefixes back out of `firepanda/py/errors.mojo` and asserts that the set of kinds Mojo can emit is exactly the set of kinds Python knows, so a kind added on one side and forgotten on the other fails immediately and names itself.

There is a smaller trap that the same test file ran into and now documents. The repo root holds a directory called `firepanda`, the Mojo one, and it has no `__init__.py`. An `import firepanda` in a test that has not first asked for the staging fixture therefore succeeds, returns an empty namespace package pointing at the Mojo sources, and leaves it in `sys.modules` for every later import in the run. The whole file passed when run with the rest of the suite, because another file staged first, and failed when run alone. Every test in it now takes the fixture whether it calls into the extension or not.

## 8. What is not done

Three things are known to be missing and are better written down than rediscovered.

`_firepanda.DataFrame()` still raises `ValueError: firepanda:unsupported: ...` with the prefix showing. That is the raw extension type rather than the public `firepanda.DataFrame`, nothing wraps its constructor path, and a user reaching it has gone around the front door. It is acceptable and it is not correct.

`ReaderError` is an `OSError` and not a `FileNotFoundError`, so `except OSError` around a read works and `except FileNotFoundError` does not. Getting the second right means the reader distinguishing why the open failed and carrying that across as well, which is a finer taxonomy than one prefix per kind supports and is deferred until the reader itself has the information.

The message structure from document 04 section 8 is preserved by this layer but is not yet produced by the core. `translate` passes everything after the prefix through unedited, which is what makes the promise keepable, but the suggestion for a misspelled column and the plan position are still to be written where the errors are raised. Until then this document's example is what the mechanism carries rather than what a user currently sees.
