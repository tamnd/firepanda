"""`DataFrame.query` and `DataFrame.eval`, checked against pandas.

Each expression is read with Python's own parser and worked out over whole
columns, so every test runs the same expression through both libraries and
compares the rows kept, or the value, or the mistake.
"""

from __future__ import annotations

import importlib.util
import inspect
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

NAN = float("nan")
FRAME = {
    "a": [1, 2, 3, 4, 5, 6],
    "b": [5.0, NAN, 1.0, 2.0, 3.0, 0.5],
    "s": ["x", "y", "z", "x", "w", "y"],
    "two words": [1, 0, 1, 0, 1, 1],
}
LIMIT = 2

QUERIES = [
    "a > 2",
    "a > 2 & b < 3",
    "a > 2 and b < 3",
    "a < 2 | b < 1",
    "a < 2 or b < 1",
    "1 < a < 4",
    "a in [1, 3]",
    "a not in [1, 3]",
    "s == ['x', 'z']",
    "s != ['x', 'z']",
    "s in ['y']",
    "a in b",
    "a > @LIMIT",
    "`two words` == 1",
    "index > 2",
    "b != b",
    "b == b",
    "abs(a - 3) < 2",
    "b.isna()",
    "s.str.startswith('x')",
    "not (a > 2)",
    "~(a > 2)",
    "a > 2 and s == 'x' or b < 2",
    "a + b > 5",
    "a * 2 == 6",
    "a % 2 == 0",
    "a ** 2 > 10",
    "-a < -3",
    "a == 0x2",
    "s == 'x'",
    's == "it\'s"',
    "a // 2 == 1",
    "b / 2 > 1",
]


@pytest.mark.parametrize("expr", QUERIES)
def test_a_query_keeps_pandas_rows(firepanda: ModuleType, expr: str) -> None:
    """Comparisons, chains, membership, names in backticks, variables and methods."""
    import pandas as pd

    got = firepanda.DataFrame(FRAME).query(expr)
    want = pd.DataFrame(FRAME).query(expr)
    assert list(got.index) == list(want.index)
    assert list(got.columns) == list(want.columns)
    assert got["a"].tolist() == want["a"].tolist()


def test_the_name_of_the_row_labels_is_read(firepanda: ModuleType) -> None:
    """A named index answers to its name as well as to `index`."""
    import pandas as pd

    for expr in ("idx > 13", "index > 13 and a < 6"):
        got = firepanda.DataFrame(FRAME).assign(idx=range(10, 16)).set_index("idx").query(expr)
        want = pd.DataFrame(FRAME).assign(idx=range(10, 16)).set_index("idx").query(expr)
        assert list(got.index) == list(want.index)


def test_a_variable_is_read_from_the_caller(firepanda: ModuleType) -> None:
    """A local of the calling function, and `local_dict` and `global_dict` over it."""
    low = 4  # noqa: F841, read by the query as @low
    assert list(firepanda.DataFrame(FRAME).query("a >= @low").index) == [3, 4, 5]
    got = firepanda.DataFrame(FRAME).query("a >= @low", local_dict={"low": 6})
    assert list(got.index) == [5]
    got = firepanda.DataFrame(FRAME).query("a < @high", global_dict={"high": 3})
    assert list(got.index) == [0, 1]


def test_a_resolver_is_read_before_the_columns(firepanda: ModuleType) -> None:
    """pandas reads the mappings in `resolvers` ahead of the frame."""
    import pandas as pd

    got = firepanda.DataFrame(FRAME).query("a > 2 and c", resolvers=[{"c": True}])
    want = pd.DataFrame(FRAME).query("a > 2 and c", resolvers=[{"c": True}])
    assert list(got.index) == list(want.index)


def test_the_python_parser_keeps_python_binding(firepanda: ModuleType) -> None:
    """With `parser="python"`, `&` binds tighter than a comparison, so parentheses are needed."""
    import pandas as pd

    expr = "(a > 2) & (b < 3)"
    got = firepanda.DataFrame(FRAME).query(expr, parser="python")
    want = pd.DataFrame(FRAME).query(expr, parser="python")
    assert list(got.index) == list(want.index)


def test_inplace_keeps_the_rows_in_the_frame(firepanda: ModuleType) -> None:
    """The frame holds the answer and None comes back."""
    frame = firepanda.DataFrame(FRAME)
    assert frame.query("a > 4", inplace=True) is None
    assert list(frame.index) == [4, 5]


EVALS = ["a + b", "a * 2", "a > 2 & b < 3", "s == 'x'", "abs(b - 3)", "`two words` + a"]


@pytest.mark.parametrize("expr", EVALS)
def test_eval_is_pandas_value(firepanda: ModuleType, expr: str) -> None:
    """The value of an expression, column for column."""
    import pandas as pd

    got = firepanda.DataFrame(FRAME).eval(expr)
    want = pd.DataFrame(FRAME).eval(expr)
    assert got.dtype == str(want.dtype)
    assert [x if x == x else None for x in got.tolist()] == [
        x if x == x else None for x in want.tolist()
    ]


def test_eval_assigns_a_column(firepanda: ModuleType) -> None:
    """`c = ...` answers the frame with the column, or keeps it in place."""
    import pandas as pd

    got = firepanda.DataFrame(FRAME).eval("c = a * 2")
    want = pd.DataFrame(FRAME).eval("c = a * 2")
    assert list(got.columns) == list(want.columns)
    assert got["c"].tolist() == want["c"].tolist()
    frame = firepanda.DataFrame(FRAME)
    assert frame.eval("a = a + 1", inplace=True) is None
    assert frame["a"].tolist() == [2, 3, 4, 5, 6, 7]


MISTAKES = [
    lambda f: f.query("nope > 1"),
    lambda f: f.query("a >"),
    lambda f: f.query("a is 1"),
    lambda f: f.query("a"),
    lambda f: f.query(""),
    lambda f: f.query(5),
    lambda f: f.query("a > 1", parser="sql"),
    lambda f: f.query("a > 1", engine="fast"),
    lambda f: f.query("a > @missing"),
    lambda f: f.query("(lambda x: x)(a) > 1"),
    lambda f: f.query("a if b else a"),
    lambda f: f.eval("a + 1", inplace=True),
    lambda f: f.eval("a + 1", nope=True),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Any) -> None:
    """The same class, or a subclass of it, by name, since each library has its own errors."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd.DataFrame(FRAME))
    with pytest.raises(Exception) as mine:
        build(firepanda.DataFrame(FRAME))
    names = [each.__name__ for each in type(mine.value).__mro__]
    assert type(theirs.value).__name__ in names, (mine.value, theirs.value)


def test_an_unknown_name_is_named(firepanda: ModuleType) -> None:
    """pandas' own message, and its own class."""
    import pandas as pd

    with pytest.raises(pd.errors.UndefinedVariableError) as theirs:
        pd.DataFrame(FRAME).query("nope > 1")
    with pytest.raises(firepanda.errors.UndefinedVariableError) as mine:
        firepanda.DataFrame(FRAME).query("nope > 1")
    assert str(mine.value) == str(theirs.value)


@pytest.mark.parametrize("name", ["query", "eval"])
def test_the_signature_is_pandas_signature(firepanda: ModuleType, name: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(getattr(firepanda.DataFrame, name)).parameters
    yours = inspect.signature(getattr(pd.DataFrame, name)).parameters
    assert [(p.name, p.kind) for p in ours.values()] == [(p.name, p.kind) for p in yours.values()]
    for each in ours:
        assert ours[each].default == yours[each].default, each


@pytest.mark.parametrize("is_local", [None, False, True])
def test_the_undefined_name_error_is_built_like_pandas(
    firepanda: ModuleType, is_local: bool | None
) -> None:
    """From the name and whether it was the caller's, with the same message."""
    import pandas as pd

    mine = firepanda.errors.UndefinedVariableError("x", is_local)
    theirs = pd.errors.UndefinedVariableError("x", is_local)
    assert str(mine) == str(theirs)
