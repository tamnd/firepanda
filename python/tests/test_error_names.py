"""Every class in `pandas.errors` has a class of the same name in `firepanda.errors`.

Code written against pandas names these classes in `except` clauses and in
`warnings.filterwarnings`, so each one is here, catching as the pandas one
catches: on the same builtins and under the same parents.
"""

from __future__ import annotations

import builtins
import importlib.util
import inspect
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def pandas_classes() -> list[str]:
    """The names of the classes pandas has in `pandas.errors`."""
    import pandas.errors

    return sorted(
        name
        for name, value in vars(pandas.errors).items()
        if not name.startswith("_") and inspect.isclass(value)
    )


def test_every_pandas_class_is_here(firepanda: ModuleType) -> None:
    """No name is missing."""
    missing = [name for name in pandas_classes() if not hasattr(firepanda.errors, name)]
    assert missing == []


def test_each_class_is_in_all(firepanda: ModuleType) -> None:
    """A star import brings them all."""
    assert set(pandas_classes()) <= set(firepanda.errors.__all__)


@pytest.mark.parametrize("name", pandas_classes())
def test_each_class_catches_as_pandas_does(firepanda: ModuleType, name: str) -> None:
    """The builtins and the pandas classes above it are above this one too."""
    import pandas.errors

    theirs = getattr(pandas.errors, name)
    mine = getattr(firepanda.errors, name)
    for parent in theirs.__mro__:
        if parent is object:
            continue
        if parent.__module__ == "builtins":
            assert issubclass(mine, getattr(builtins, parent.__name__))
        else:
            assert parent.__name__ in {kind.__name__ for kind in mine.__mro__}


@pytest.mark.parametrize("name", pandas_classes())
def test_each_class_is_made_as_pandas_makes_it(firepanda: ModuleType, name: str) -> None:
    """The same parameters, where pandas writes its own constructor."""
    import pandas.errors

    theirs = getattr(pandas.errors, name)
    if "__init__" not in vars(theirs):
        return
    mine = getattr(firepanda.errors, name)
    assert list(inspect.signature(mine).parameters) == list(inspect.signature(theirs).parameters)


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [((1,), {}), ((int,), {"methodtype": "classmethod"}), (([],), {"methodtype": "property"})],
)
def test_an_abstract_method_error_names_the_class(
    firepanda: ModuleType, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    """Which class should have written the method, in pandas' words."""
    import pandas.errors

    mine = firepanda.errors.AbstractMethodError(*args, **kwargs)
    assert str(mine) == str(pandas.errors.AbstractMethodError(*args, **kwargs))


def test_an_abstract_method_error_refuses_an_unknown_kind(firepanda: ModuleType) -> None:
    """A kind that is not a method, a classmethod, a staticmethod or a property."""
    with pytest.raises(ValueError, match="methodtype must be one of"):
        firepanda.errors.AbstractMethodError(1, methodtype="field")
