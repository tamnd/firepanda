"""The error table, tested from both sides of the boundary.

An exception class is part of an API. `except KeyError` around a column lookup
is in code that predates this project by fifteen years, and a firepanda that
raises a bare `Exception` for everything fails pandas compatibility in a way no
conformance case about values will ever catch, because every one of those is
comparing an answer and this is about what happens when there is no answer.

The mapping is carried across the boundary as a prefix on the message, which is
document 14, and a prefix is only as good as the agreement about it. So these
tests do not check the Python half against strings written in this file. They ask
the Mojo half to raise one of each kind and check what arrives, which is the only
arrangement in which a disagreement between the two shows up as a failure.

Every test here takes the `firepanda` fixture, including the few that only read
the Python half and never call into the extension. That is not decoration. The
repo root holds a directory called `firepanda` too, the Mojo one, and it has no
`__init__.py`, so an `import firepanda` that happens before the fixture has put
the staged package on the path succeeds and hands back an empty namespace package
pointing at the Mojo sources. It then sits in `sys.modules` and every later import
in the run gets it. Taking the fixture is what orders the staging first.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType

import pytest
from conftest import REPO

MOJO_SIDE = REPO / "firepanda" / "py" / "errors.mojo"

# What each kind promises to a caller who never heard of firepanda. This is the
# table from `docs/specs/07-python-bindings.md` section 5, written out a second
# time on purpose: `python/firepanda/errors.py` says which class, and this says
# which builtin that class has to remain catchable as. Deriving one from the
# other would test nothing.
CATCHABLE_AS: dict[str, type[BaseException]] = {
    "column": KeyError,
    "dtype": TypeError,
    "value": ValueError,
    "position": IndexError,
    "io": OSError,
    "unsupported": NotImplementedError,
    "cancelled": KeyboardInterrupt,
}


def test_the_two_halves_agree_on_which_kinds_exist(firepanda: ModuleType) -> None:
    """Every prefix Mojo can emit has a row in the Python table, and the reverse.

    This is the test that makes the wire format a wire format rather than two
    lists that happen to match today. A kind added on one side and forgotten on
    the other fails here rather than at whatever moment a user first triggers
    it, which would be a `RuntimeError` with a `firepanda:` prefix still showing.
    """
    from firepanda.errors import BY_KIND

    text = MOJO_SIDE.read_text()
    emitted = set(re.findall(r'comptime \w+ = "firepanda:(\w+): "', text))
    assert emitted, f"no prefixes found in {MOJO_SIDE}, has the format changed"
    assert emitted == set(BY_KIND)


def test_every_kind_is_catchable_as_the_builtin_it_promises(
    firepanda: ModuleType,
) -> None:
    """The classes are named for readability and are the builtins for compatibility."""
    from firepanda.errors import BY_KIND

    assert set(BY_KIND) == set(CATCHABLE_AS)
    for kind, builtin in CATCHABLE_AS.items():
        assert issubclass(BY_KIND[kind], builtin), (
            f"{kind} raises {BY_KIND[kind].__name__}, which is not a"
            f" {builtin.__name__}, so pandas code catching it will not fire"
        )


def test_a_cancellation_is_not_swallowed_by_except_exception(
    firepanda: ModuleType,
) -> None:
    """`KeyboardInterrupt` is a `BaseException` and must stay one.

    The marker class every other firepanda error carries is an `Exception`, and
    inheriting from both would quietly put `Exception` in this one's ancestry,
    after which any library with a bare `except Exception` in it would eat a
    Ctrl-C. That is a one word mistake to make and a miserable one to diagnose.
    """
    from firepanda.errors import CancelledError, FirepandaError

    assert not issubclass(CancelledError, Exception)
    assert not issubclass(CancelledError, FirepandaError)
    assert issubclass(FirepandaError, Exception)


@pytest.mark.parametrize("kind", sorted(CATCHABLE_AS))
def test_each_kind_arrives_from_mojo_as_the_right_class(firepanda: ModuleType, kind: str) -> None:
    """The whole path, from a `raise` in Mojo to a class a pandas user catches."""
    from firepanda._frame import _raise_for_test
    from firepanda.errors import BY_KIND

    with pytest.raises(CATCHABLE_AS[kind]) as caught:
        _raise_for_test(kind)
    assert type(caught.value) is BY_KIND[kind]


def test_the_prefix_is_really_on_the_wire(firepanda: ModuleType) -> None:
    """The untranslated error carries the prefix, which is what makes this work.

    Read from the extension directly rather than through the wrapper, because
    the point is what crosses the boundary rather than what the Python layer
    does with it afterwards. If this ever passes while the test above fails, the
    fault is on the Python side; if this fails, it is on the Mojo side.
    """
    with pytest.raises(Exception) as caught:
        firepanda._firepanda._raise_for_test("column")
    assert str(caught.value) == "firepanda:column: no such column 'regoin'"
    assert type(caught.value) is Exception


def test_the_message_survives_the_translation(firepanda: ModuleType) -> None:
    """Everything after the prefix reaches the user unedited.

    Document 04 section 8 promises the column, the dtypes and the suggestion, and
    a translation layer that reformatted the message would quietly break that
    promise for every error at once.
    """
    from firepanda._frame import _raise_for_test

    with pytest.raises(NotImplementedError) as caught:
        _raise_for_test("unsupported")
    assert str(caught.value) == "object dtype is not supported"


def test_a_key_error_stringifies_the_way_pandas_does(firepanda: ModuleType) -> None:
    """The quotes around a `KeyError` message are `KeyError`, not a bug here.

    `KeyError.__str__` is `repr` of its argument, so the message comes out
    quoted. pandas has exactly this behaviour for a missing column, so matching
    it is the point rather than an accident, and this test exists so that the
    next person to notice the quotes finds a decision instead of a mystery.
    """
    from firepanda._frame import _raise_for_test

    with pytest.raises(KeyError) as caught:
        _raise_for_test("column")
    assert str(caught.value) == "\"no such column 'regoin'\""
    assert caught.value.args == ("no such column 'regoin'",)


def test_an_untagged_error_becomes_a_runtime_error(firepanda: ModuleType) -> None:
    """The last row of the table, and a note that some binding wants a `try`.

    An error that reaches Python untagged came out of the core without a binding
    classifying it. `RuntimeError` is the honest answer, and it is much better
    than leaking a bare `Exception`, which is what a caller gets when nothing at
    all is done.
    """
    from firepanda._frame import _raise_for_test

    with pytest.raises(RuntimeError) as caught:
        _raise_for_test("untagged")
    assert type(caught.value) is RuntimeError
    assert str(caught.value) == "something went wrong a long way down"


def test_an_unknown_kind_keeps_its_prefix(firepanda: ModuleType) -> None:
    """A kind the Python side does not know about is version skew, not a mystery.

    It happens in a development tree with a stale extension in it, and the
    prefix is the evidence, so it is kept rather than stripped.
    """
    from firepanda.errors import translate

    result = translate(Exception("firepanda:invented: something"))
    assert type(result) is RuntimeError
    assert str(result) == "firepanda:invented: something"


def test_the_binding_layers_own_arity_error_gets_its_class_back(
    firepanda: ModuleType,
) -> None:
    """`Exception: TypeError: ...` becomes a `TypeError`.

    This is the commonest mistake a user can make and the binding layer answers
    it with the right words in the wrong place, which document 13 section 5
    measured. Calling a method with too many arguments should raise something
    `except TypeError` catches.
    """
    from firepanda.errors import translate

    raw = Exception("TypeError: <mojo function>() takes 1 positional argument but 2 were given")
    result = translate(raw)
    assert type(result) is TypeError
    assert str(result) == ("<mojo function>() takes 1 positional argument but 2 were given")


def test_translate_leaves_alone_what_is_not_ours(firepanda: ModuleType) -> None:
    """A `MemoryError` is Python's, and comes back untouched.

    `translate` is applied to everything a delegation can raise, so it has to be
    safe on errors that never crossed the boundary at all.
    """
    from firepanda.errors import ColumnNotFoundError, translate

    memory = MemoryError("out of memory")
    assert translate(memory) is memory

    ours = ColumnNotFoundError("already translated")
    assert translate(ours) is ours


def test_a_missing_file_is_an_os_error(firepanda: ModuleType, tmp_path: Path) -> None:
    """A real path, rather than the test raiser, ending where `open` would end.

    `except OSError` around a read is what people write, and `except
    FileNotFoundError` is what they write when they are being careful. The first
    works. The second does not yet, and that is recorded in document 14 rather
    than papered over.
    """
    missing = tmp_path / "nothing.csv"
    with pytest.raises(OSError) as caught:
        firepanda.read_csv(str(missing))
    assert str(missing) in str(caught.value)
    assert type(caught.value) is firepanda.errors.ReaderError


def test_a_bad_argument_names_the_argument(firepanda: ModuleType, tmp_path: Path) -> None:
    """The other real path, and the reason bindings convert arguments themselves.

    Mojo's own `Int(py=value)` says `invalid literal for int() with base 10:
    'x'`, which is true and names neither the argument nor the function. With
    three integer arguments a caller cannot tell from it which one was wrong.
    """
    csv = tmp_path / "one.csv"
    csv.write_text("a\n1\n")
    frame = firepanda.read_csv(str(csv))

    with pytest.raises(TypeError) as caught:
        frame.head("x")
    assert str(caught.value) == "n must be an integer, got str 'x'"

    with pytest.raises(TypeError) as caught:
        frame.tail(None)
    assert str(caught.value) == "n must be an integer, got NoneType None"


def test_nothing_is_chained_onto_the_translated_error(firepanda: ModuleType) -> None:
    """`raise ... from None`, so a traceback does not say the same thing twice.

    The error being suppressed is the binding layer's own untyped wrapper around
    a message this library wrote. Showing it would print the sentence again and
    call the second copy the direct cause of the first.
    """
    from firepanda._frame import _raise_for_test

    with pytest.raises(KeyError) as caught:
        _raise_for_test("column")
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
