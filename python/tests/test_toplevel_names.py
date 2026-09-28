"""`NA`, `IndexSlice`, `eval` and `col`, and functions as `loc` keys, compared with pandas."""

from __future__ import annotations

import copy
import datetime
import importlib.util
import operator
import pickle
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

OPERATORS = [
    "add",
    "sub",
    "mul",
    "truediv",
    "floordiv",
    "mod",
    "pow",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "and_",
    "or_",
    "xor",
    "matmul",
]

OTHERS: list[Callable[[ModuleType], Any]] = [
    lambda m: 1,
    lambda m: 0,
    lambda m: 1.5,
    lambda m: True,
    lambda m: False,
    lambda m: "s",
    lambda m: b"b",
    lambda m: m.NA,
    lambda m: None,
    lambda m: datetime.date(2020, 1, 1),
    lambda m: datetime.timedelta(1),
    lambda m: [1],
]


def outcome(call: Callable[[], Any]) -> str:
    """What a call answers, or the kind of error it raises."""
    try:
        return repr(call())
    except Exception as error:
        return type(error).__name__


@needs_pandas
@pytest.mark.parametrize("name", OPERATORS)
def test_na_answers_each_operator_as_pandas_does(firepanda: ModuleType, name: str) -> None:
    """Both ways round, against numbers, flags, text, bytes, dates, spans and the rest."""
    import pandas as pd

    def shown(m: ModuleType) -> list[str]:
        op = getattr(operator, name)
        answers = []
        for other in OTHERS:
            value = other(m)
            answers.append(outcome(lambda value=value: op(m.NA, value)))
            answers.append(outcome(lambda value=value: op(value, m.NA)))
        return answers

    assert shown(firepanda) == shown(pd)


@needs_pandas
def test_na_reads_as_pandas_na(firepanda: ModuleType) -> None:
    """The text, the hash, the unary operators, divmod and numpy's functions."""
    import numpy as np
    import pandas as pd

    def shown(m: ModuleType) -> Any:
        na = m.NA
        return (
            repr(na),
            str(na),
            format(na),
            format(na, ">6"),
            hash(na),
            type(na).__name__,
            [repr(f(na)) for f in (operator.neg, operator.pos, abs, operator.invert)],
            repr(divmod(na, 2)),
            repr(divmod(2, na)),
            outcome(lambda: bool(na)),
            m.isna(na),
            m.notna(na),
            repr(np.add(na, 1)),
            repr(np.array([1, 2]) + na),
            repr(np.log(na)),
            repr(np.divmod(na, 1)),
            repr(np.logical_and(na, False)),
            repr(np.bitwise_and(na, False)),
            repr(np.array([0, 2]) ** na),
            repr(na ** np.array([0, 2])),
        )

    assert shown(firepanda) == shown(pd)


def test_na_is_one_value(firepanda: ModuleType) -> None:
    """Pickling, copying and making another all give back the same object."""
    na = firepanda.NA
    assert pickle.loads(pickle.dumps(na)) is na
    assert copy.copy(na) is na
    assert copy.deepcopy(na) is na
    assert type(na)() is na
    with pytest.raises(TypeError, match="boolean value of NA is ambiguous"):
        bool(na)


@needs_pandas
def test_index_slice_hands_back_the_key(firepanda: ModuleType) -> None:
    """Whatever goes between the brackets comes out, slices and all."""
    import pandas as pd

    def shown(m: ModuleType) -> Any:
        s = m.IndexSlice
        return [s[1:3], s[:, "a"], s["x"], s[1, 2:, ::2], callable(s)]

    assert shown(firepanda) == shown(pd)


def frame(m: ModuleType) -> Any:
    return m.DataFrame({"a": [1, 2, 3], "b": [4.0, 5, 6], "s": ["x", "yy", "z"]})


def shown(answer: Any) -> Any:
    """An answer spelled the same in both, whatever kind of thing it is."""
    if hasattr(answer, "columns"):
        labels = answer.index.tolist()
        return "frame", list(answer.columns), labels, {c: answer[c].tolist() for c in answer}
    if hasattr(answer, "index") and hasattr(answer, "tolist"):
        return "column", answer.name, answer.index.tolist(), answer.tolist()
    if isinstance(answer, dict):
        return {key: shown(value) for key, value in answer.items()}
    if hasattr(answer, "item"):
        return answer.item()
    return answer


X = 10


def evaluated(expr: Any, **kwargs: Any) -> Callable[[ModuleType], Any]:
    """`eval` of `expr` in both, with `df` and `x` to read."""

    def answer(m: ModuleType) -> Any:
        df = frame(m)  # noqa: F841 - read by the expression
        x = X  # noqa: F841 - read by the expression
        given = dict(kwargs)
        if callable(given.get("target")):
            given["target"] = given["target"](m)
        return m.eval(expr, **given)

    return answer


EVALS: dict[str, Callable[[ModuleType], Any]] = {
    "numbers": evaluated("1 + 2 * 3"),
    "a variable": evaluated("x + 1"),
    "two columns": evaluated("df.a + df.b"),
    "an item": evaluated("df['a'] * 2"),
    "and binds loosely": evaluated("df.a > 1 & df.b < 6"),
    "the python parser": evaluated("(df.a > 1) & (df.b < 6)", parser="python"),
    "local_dict": evaluated("y * 2", local_dict={"y": 3}),
    "global_dict": evaluated("y * 2", global_dict={"y": 4}),
    "resolvers": evaluated("a + 1", resolvers=[{"a": 5}]),
    "a target": evaluated("c = df.a + 1", target=frame),
    "inplace": evaluated("c = df.a + 1", target=frame, inplace=True),
    "a dictionary target": evaluated("c = x + 1", target={"k": 1}),
    "a target not assigned": evaluated("df.a + 1", target=frame),
    "not text": evaluated(5),
    "a list": evaluated(["1+1", "2+2"]),
    "text": evaluated("'ab'"),
    "a flag": evaluated("True"),
    "a method": evaluated("df.a.sum()"),
    "sin": evaluated("sin(df.b)"),
    "arctan2": evaluated("arctan2(df.b, df.a)"),
    "sin of a frame": evaluated("sin(df[['a', 'b']])"),
    "in": evaluated("df.a in [1, 2]"),
    "power": evaluated("2 ** 3"),
    "abs": evaluated("abs(-df.a)"),
    "not": evaluated("~(df.a > 1)"),
    "a chain": evaluated("1 < x < 20"),
    "a frame": evaluated("df[['a', 'b']] + 1"),
    "division": evaluated("7 / 2"),
}

EVAL_MISTAKES: dict[str, Callable[[ModuleType], Any]] = {
    "@": evaluated("@x + 1"),
    "an assignment with no target": evaluated("c = 1 + 2"),
    "several lines with no target": evaluated("df.a + 1\ndf.a + 2"),
    "not defined": evaluated("zz + 1"),
    "empty": evaluated(""),
    "a parser": evaluated("1+1", parser="x"),
    "an engine": evaluated("1+1", engine="x"),
    "inplace with no target": evaluated("c = df.a", inplace=True),
    "inplace with no assignment": evaluated("df.a + 1", inplace=True),
    "a target that cannot be assigned": evaluated("c = 1", target=lambda m: [1]),
    "a column of a variable": evaluated("c = a + 1", target=frame),
}


@needs_pandas
@pytest.mark.parametrize("name", list(EVALS))
def test_eval_answers_what_pandas_answers(firepanda: ModuleType, name: str) -> None:
    import pandas as pd

    assert shown(EVALS[name](firepanda)) == shown(EVALS[name](pd))


@needs_pandas
@pytest.mark.parametrize("name", list(EVAL_MISTAKES))
def test_eval_refuses_what_pandas_refuses(firepanda: ModuleType, name: str) -> None:
    """The same words, and a ValueError where pandas raises one."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        EVAL_MISTAKES[name](pd)
    with pytest.raises(Exception) as mine:
        EVAL_MISTAKES[name](firepanda)
    for kind in (ValueError, KeyError, SyntaxError, NameError):
        assert isinstance(mine.value, kind) == isinstance(theirs.value, kind)
    assert str(mine.value) == str(theirs.value)


def test_eval_reads_the_callers_variables_by_level(firepanda: ModuleType) -> None:
    """`level` reaches past functions between the caller and `eval`."""
    x = 7  # noqa: F841 - read by the expression

    def inner() -> Any:
        return firepanda.eval("x", level=1)

    assert inner() == 7


def test_eval_reads_several_lines_into_a_target(firepanda: ModuleType) -> None:
    """Each line sees what the ones before it assigned."""
    df = frame(firepanda)
    answer = firepanda.eval("c = df.a + 1\nd = df.a + 2", target=df)
    assert answer["c"].tolist() == [2, 3, 4]
    assert answer["d"].tolist() == [3, 4, 5]
    assert "c" not in df.columns


def c(m: ModuleType, name: Any) -> Any:
    return m.col(name)


REPRS: dict[str, Callable[[ModuleType], Any]] = {
    "a column": lambda m: c(m, "a"),
    "operators": lambda m: (c(m, "a") + 1) * c(m, "b") - 2,
    "reflected": lambda m: 1 - c(m, "a"),
    "logic": lambda m: (c(m, "a") > 1) & ~(c(m, "b") == 5),
    "a method": lambda m: c(m, "a").sum(),
    "str": lambda m: c(m, "s").str.upper(),
    "dt": lambda m: c(m, "a").dt.year,
    "keywords": lambda m: c(m, "a").clip(lower=1, upper=2),
    "an item": lambda m: c(m, "a")[0],
    "signs": lambda m: (-c(m, "a")) ** 2 + abs(c(m, "b")),
    "a sign of an operator": lambda m: -(c(m, "a") + 1),
    "equal": lambda m: c(m, "a") == c(m, "a"),
    "a number name": lambda m: c(m, 0),
    "text": lambda m: c(m, "s") + "q",
    "a list": lambda m: c(m, "a").isin([1, 2]),
    "case_when": lambda m: c(m, "a").case_when([(c(m, "b") > 4, 0)]),
    "every operator": lambda m: [
        op(c(m, "a"), 2) for op in (operator.floordiv, operator.mod, operator.or_, operator.xor)
    ],
    "an attribute of an operator": lambda m: (c(m, "a") + 1).abs(),
}


@needs_pandas
@pytest.mark.parametrize("name", list(REPRS))
def test_an_expression_prints_as_pandas_does(firepanda: ModuleType, name: str) -> None:
    import pandas as pd

    assert repr(REPRS[name](firepanda)) == repr(REPRS[name](pd))


USES: dict[str, Callable[[ModuleType], Any]] = {
    "assign": lambda m: frame(m).assign(t=c(m, "a") + c(m, "b")),
    "assign str": lambda m: frame(m).assign(u=c(m, "s").str.upper()),
    "assign a sum": lambda m: frame(m).assign(t=c(m, "a").sum()),
    "assign case_when": lambda m: frame(m).assign(
        t=c(m, "a").case_when([(c(m, "b") > 4, c(m, "a") * 10)])
    ),
    "loc": lambda m: frame(m).loc[c(m, "a") > 1],
    "loc with columns": lambda m: frame(m).loc[c(m, "a") > 1, ["b"]],
    "[]": lambda m: frame(m)[c(m, "a") > 1],
    "where": lambda m: frame(m)[["b"]].where(c(m, "b") > 4, 0),
    "numpy": lambda m: frame(m).assign(t=__import__("numpy").sqrt(c(m, "b"))),
    "loc a function": lambda m: frame(m).loc[lambda d: d.a > 1],
    "loc a function of columns": lambda m: frame(m).loc[lambda d: d.a > 1, lambda d: ["b"]],
    "iloc a function": lambda m: frame(m).iloc[lambda d: [0, 2]],
    "a column's loc": lambda m: frame(m).a.loc[lambda s: s > 1],
    "a column's iloc": lambda m: frame(m).a.iloc[lambda s: [1]],
}


@needs_pandas
@pytest.mark.parametrize("name", list(USES))
def test_an_expression_is_read_where_pandas_reads_it(firepanda: ModuleType, name: str) -> None:
    import pandas as pd

    assert shown(USES[name](firepanda)) == shown(USES[name](pd))


def written(m: ModuleType) -> Any:
    df = frame(m)
    df.loc[lambda d: d.a > 1, "b"] = 0.0
    df.a.iloc[lambda s: [0]] = 9
    return df


COL_MISTAKES: dict[str, Callable[[ModuleType], Any]] = {
    "a column not there": lambda m: frame(m).assign(t=c(m, "zz") + 1),
    "truth": lambda m: bool(c(m, "a")),
    "iterating": lambda m: iter(c(m, "a")),
    "copying": lambda m: copy.copy(c(m, "a")),
    "deep copying": lambda m: copy.deepcopy(c(m, "a")),
    "a name that cannot be hashed": lambda m: c(m, ["a"]),
    "iloc a function answering a tuple": lambda m: frame(m).iloc[lambda d: (0, 1)],
}


@needs_pandas
@pytest.mark.parametrize("name", list(COL_MISTAKES))
def test_an_expression_refuses_what_pandas_refuses(firepanda: ModuleType, name: str) -> None:
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        COL_MISTAKES[name](pd)
    with pytest.raises(Exception) as mine:
        COL_MISTAKES[name](firepanda)
    for kind in (ValueError, TypeError):
        assert isinstance(mine.value, kind) == isinstance(theirs.value, kind)
    assert str(mine.value) == str(theirs.value)


@needs_pandas
def test_a_function_as_a_loc_key_writes_where_pandas_writes(firepanda: ModuleType) -> None:
    import pandas as pd

    assert shown(written(firepanda)) == shown(written(pd))
