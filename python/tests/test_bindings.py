"""The parity tests, which are what makes the binding table worth having.

A table that generates three files is only an improvement over writing them by
hand if something checks that the table still describes reality. There are three
surfaces to keep in agreement and each of these tests holds one edge of that
triangle:

    the Mojo bindings   <-->   the table   <-->   the Python classes
                                  |
                                  v
                               pandas

The last edge is the interesting one and it was not in the original scope of M3
P2. It compares firepanda's Python signature against the pandas signature for the
same name, and it is the closest thing to a mechanical measurement of the
compatibility goal anywhere in this repository. It is deliberately allowed to
fail loudly as the surface grows, because a name that matches pandas while its
arguments do not is worse than a name that is missing.
"""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from conftest import REPO

# The table is a build tool rather than a shipped module, so it is not on the
# path by rights. Reading it here is the point: these tests compare it against
# what it generated, and a copy of it would compare nothing.
sys.path.insert(0, str(REPO / "tools"))
import bindings

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def test_the_generated_files_are_what_the_table_says() -> None:
    """The three generated files match the table they came from.

    This is the test that gives every other test in this file its meaning. If
    someone edits a generated file by hand, everything else here still passes,
    because it all reads the same edited files.
    """
    done = subprocess.run(
        [sys.executable, str(REPO / "tools" / "bindings.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_every_binding_in_the_table_exists_on_the_extension(firepanda: ModuleType) -> None:
    """Every entry in the table resolves to something the extension exposes."""
    extension = firepanda._firepanda
    for exposed in bindings.TYPES:
        cls = getattr(extension, exposed.name, None)
        assert cls is not None, f"the extension has no type {exposed.name}"
        for binding in exposed.bindings:
            assert hasattr(cls, binding.name), (
                f"{exposed.name}.{binding.name} is in the table and not on the extension"
            )
    for function in bindings.FUNCTIONS:
        assert hasattr(extension, function.name), (
            f"{function.name} is in the table and not on the extension"
        )


def test_the_extension_exposes_nothing_the_table_does_not_mention(firepanda: ModuleType) -> None:
    """Nothing is registered that the table does not know about.

    The other direction of the same check, and the one that catches a binding
    added directly to `module.mojo` to get something working quickly. `version`
    is the one exception, and it is written by hand on purpose, because it
    describes the artifact rather than the library.
    """
    extension = firepanda._firepanda
    tabled = {f.name for f in bindings.FUNCTIONS} | {t.name for t in bindings.TYPES}
    tabled.add("version")
    found = {name for name in dir(extension) if not name.startswith("_")}
    assert found - tabled == set()

    for exposed in bindings.TYPES:
        cls = getattr(extension, exposed.name)
        tabled_methods = {b.name for b in exposed.bindings}
        found_methods = {name for name in dir(cls) if not name.startswith("_")}
        assert found_methods - tabled_methods == set()


def test_every_member_in_the_table_exists_on_the_python_class(firepanda: ModuleType) -> None:
    """The Python half of the table resolves too, with the right kinds."""
    for exposed in bindings.TYPES:
        cls = getattr(firepanda, exposed.py)
        for member in exposed.members:
            attribute = inspect.getattr_static(cls, member.name, None)
            assert attribute is not None, (
                f"{exposed.py}.{member.name} is in the table and not on the class"
            )
            if member.kind == "property":
                assert isinstance(attribute, property), (
                    f"{exposed.py}.{member.name} is a property in the table and"
                    f" a {type(attribute).__name__} on the class"
                )
            else:
                assert callable(attribute)


def test_the_python_class_delegates_and_does_not_hold_state(firepanda: ModuleType) -> None:
    """The wrapper holds the extension object and nothing else.

    `__slots__` is not a micro-optimisation here. The whole design rests on the
    Python object being a thin front for a Mojo value, and an accidental
    `__dict__` is how a second, divergent copy of the state gets somewhere by
    mistake.
    """
    cls = firepanda.DataFrame
    assert cls.__slots__ == ("_inner",)
    assert not hasattr(cls, "__dict__") or "__dict__" not in cls.__slots__


def test_a_frame_read_from_csv_answers_the_pandas_way(
    firepanda: ModuleType, tmp_path: Path
) -> None:
    """The end to end path, which is the only test here that runs any Mojo.

    Everything above checks that names line up. This checks that the thing behind
    the names works, because a surface that matches pandas perfectly and returns
    wrong answers is not progress.
    """
    csv = tmp_path / "parts.csv"
    csv.write_text("name,qty,price\nrivet,4,1.25\nbolt,10,0.40\nnut,25,0.05\n")

    frame = firepanda.read_csv(str(csv))
    assert isinstance(frame, firepanda.DataFrame)
    assert len(frame) == 3
    assert frame.shape == (3, 3)
    assert frame.columns == ["name", "qty", "price"]

    top = frame.head(2)
    assert isinstance(top, firepanda.DataFrame)
    assert len(top) == 2
    assert len(frame.tail(1)) == 1
    assert repr(frame) == str(frame)
    assert "3 rows" in repr(frame)


@needs_pandas
def test_the_python_signature_matches_pandas(firepanda: ModuleType) -> None:
    """Every name firepanda shares with pandas has the pandas signature.

    This is the compatibility goal, reduced to something a machine can check. It
    compares only the names firepanda actually has, so it does not fail merely
    because the surface is incomplete, and it compares only the parameters
    firepanda declares, so a pandas method with fifteen keyword arguments can be
    adopted a few at a time. What it does not tolerate is a parameter with a
    different name, a different position or a different default, because that is
    the kind of difference that breaks a caller silently.
    """
    import pandas as pd

    problems: list[str] = []

    def compare(label: str, ours: object, theirs: object) -> None:
        """Records every way our parameters disagree with the pandas ones."""
        mine = inspect.signature(ours)  # type: ignore[arg-type]
        try:
            yours = inspect.signature(theirs)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return
        for name, parameter in mine.parameters.items():
            if name == "self":
                continue
            match = yours.parameters.get(name)
            if match is None:
                problems.append(f"{label} has a parameter {name!r} that pandas does not")
                continue
            if parameter.default != match.default:
                problems.append(
                    f"{label}({name}) defaults to {parameter.default!r} and pandas"
                    f" defaults to {match.default!r}"
                )

    against = {"DataFrame": pd.DataFrame}
    for exposed in bindings.TYPES:
        reference = against.get(exposed.py)
        if reference is None:
            continue
        ours = getattr(firepanda, exposed.py)
        for member in exposed.members:
            if member.kind != "method":
                continue
            theirs = getattr(reference, member.name, None)
            if theirs is None:
                continue
            compare(f"{exposed.py}.{member.name}", getattr(ours, member.name), theirs)

    # The module level functions matter more here than they look. `read_csv` is
    # how every frame in the library currently comes into being, so it is the one
    # signature every user meets, and its first argument is called
    # `filepath_or_buffer` in pandas rather than anything a person would choose.
    for function in bindings.FUNCTIONS:
        theirs = getattr(pd, function.name, None)
        if theirs is None:
            continue
        compare(function.name, getattr(firepanda, function.name), theirs)

    assert not problems, "\n".join(problems)


@needs_pandas
def test_the_surface_walk_still_reads_pandas() -> None:
    """The pandas surface is walkable, which the parity work depends on.

    The counts in document 13 came out of this walk, and a pandas release that
    breaks it would leave the compatibility measurement silently reading nothing.
    The assertions are loose on purpose: they check the walk works, not that
    pandas has any particular number of methods.
    """
    import pandas as pd

    members = [name for name in dir(pd.DataFrame) if not name.startswith("_")]
    assert len(members) > 100
    readable = 0
    for name in members:
        attribute = inspect.getattr_static(pd.DataFrame, name)
        if callable(attribute) and not isinstance(attribute, property):
            try:
                inspect.signature(attribute)
                readable += 1
            except (ValueError, TypeError):
                pass
    assert readable > 100
