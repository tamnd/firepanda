"""`col`, a column named now and read later from whatever frame it is used on.

`pd.col("a") + 1` is an `Expression`, which holds a function of a frame and
the text it prints as. Anywhere a method takes `lambda df: df["a"] + 1`, such
as `assign`, `loc`, `[]`, `where` and `case_when`, it takes the expression too.
An operator, an attribute, a call or an item on an expression makes another
one, so `pd.col("s").str.upper()` reads the column and upper cases it only when
a frame is given.

The printed form follows pandas 3.0: an operand made by an operator is put in
brackets when it sits inside another operator, as in `(col('a') + 1) * 2`.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Any, NoReturn

_SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "matmul": "@",
    "pow": "**",
    "truediv": "/",
    "floordiv": "//",
    "mod": "%",
    "and": "&",
    "or": "|",
    "xor": "^",
}
"""The operators with a reflected form, by the name of their method."""

_COMPARISONS = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<", "eq": "==", "ne": "!="}
"""The comparisons, which have no reflected form of their own."""


def _read(frame: Any, value: Any) -> Any:
    """An argument as it is on this frame: an expression is read, anything else is itself."""
    return value._eval_expression(frame) if isinstance(value, Expression) else value


def _arguments(*args: Any, **kwargs: Any) -> str:
    """The arguments of a call as they print, positional first."""
    shown = [repr(each) for each in args] + [f"{key}={value!r}" for key, value in kwargs.items()]
    return ", ".join(shown)


class Expression:
    """A column to be read from a frame later, made by `col`.

    It is not made directly. Its operators, attributes and calls make new
    expressions, and a method that takes a function of the frame reads it.
    """

    def __init__(
        self, func: Callable[[Any], Any], repr_str: str, needs_parenthese: bool = False
    ) -> None:
        self._func = func
        self._repr_str = repr_str
        self._needs_parentheses = needs_parenthese

    def _eval_expression(self, df: Any) -> Any:
        """The value of the expression on one frame."""
        return self._func(df)

    def _with_op(
        self, op: str, other: Any, repr_str: str, needs_parentheses: bool = True
    ) -> Expression:
        """A new expression that calls the method `op` of this one's value with `other`."""
        return Expression(
            lambda df: getattr(self._eval_expression(df), op)(_read(df, other)),
            repr_str,
            needs_parenthese=needs_parentheses,
        )

    def _maybe_wrap_parentheses(self, other: Any) -> tuple[str, str]:
        """Both sides as they print inside an operator, in brackets when they need them."""
        mine = f"({self!r})" if self._needs_parentheses else repr(self)
        wrapped = isinstance(other, Expression) and other._needs_parentheses
        theirs = f"({other!r})" if wrapped else repr(other)
        return mine, theirs

    def __invert__(self) -> Expression:
        return Expression(
            lambda df: ~self._eval_expression(df), f"~{self._repr_str}", needs_parenthese=True
        )

    def __neg__(self) -> Expression:
        return self._signed("-", lambda value: -value)

    def __pos__(self) -> Expression:
        return self._signed("+", lambda value: +value)

    def _signed(self, sign: str, how: Callable[[Any], Any]) -> Expression:
        shown = f"({self._repr_str})" if self._needs_parentheses else self._repr_str
        return Expression(
            lambda df: how(self._eval_expression(df)), sign + shown, needs_parenthese=True
        )

    def __abs__(self) -> Expression:
        return Expression(
            lambda df: abs(self._eval_expression(df)),
            f"abs({self._repr_str})",
            needs_parenthese=True,
        )

    def __array_ufunc__(
        self, ufunc: Callable[..., Any], method: str, *inputs: Any, **kwargs: Any
    ) -> Expression:
        def func(df: Any) -> Any:
            return ufunc(*(_read(df, each) for each in inputs), **kwargs)

        return Expression(func, f"{ufunc.__name__}({_arguments(*inputs, **kwargs)})")

    def __getitem__(self, item: Any) -> Expression:
        return self._with_op("__getitem__", item, f"{self!r}[{item!r}]")

    def __call__(self, *args: Any, **kwargs: Any) -> Expression:
        def func(df: Any) -> Any:
            arguments = [_read(df, each) for each in args]
            keywords = {key: _read(df, value) for key, value in kwargs.items()}
            return self._eval_expression(df)(*arguments, **keywords)

        return Expression(func, f"{self._repr_str}({_arguments(*args, **kwargs)})")

    def __getattr__(self, name: str, /) -> Any:
        shown = f"({self!r})" if self._needs_parentheses else repr(self)
        return Expression(lambda df: getattr(self._eval_expression(df), name), f"{shown}.{name}")

    def case_when(self, caselist: Sequence[tuple[Any, Any]]) -> Expression:
        """`Series.case_when` on the value, with expressions in `caselist` read on the frame."""

        def func(df: Any) -> Any:
            pairs = [(_read(df, when), _read(df, then)) for when, then in caselist]
            return self._eval_expression(df).case_when(pairs)

        return Expression(func, f"{self!r}.case_when(...)")

    def __repr__(self) -> str:
        return self._repr_str or "Expr(...)"

    def __bool__(self) -> NoReturn:
        raise TypeError("boolean value of an expression is ambiguous")

    def __iter__(self) -> NoReturn:
        raise TypeError("Expression objects are not iterable")

    def __copy__(self) -> NoReturn:
        raise TypeError("Expression objects are not copiable")

    def __deepcopy__(self, memo: dict[int, Any] | None) -> NoReturn:
        raise TypeError("Expression objects are not copiable")

    __hash__ = None  # type: ignore[assignment]


def _operator(name: str, symbol: str, reflected: bool) -> Callable[[Expression, Any], Expression]:
    """One operator method of `Expression`, which prints with `symbol`."""
    method = f"__r{name}__" if reflected else f"__{name}__"

    def op(self: Expression, other: Any) -> Expression:
        mine, theirs = self._maybe_wrap_parentheses(other)
        shown = f"{theirs} {symbol} {mine}" if reflected else f"{mine} {symbol} {theirs}"
        return self._with_op(method, other, shown)

    op.__name__ = method
    return op


for _name, _symbol in _SYMBOLS.items():
    setattr(Expression, f"__{_name}__", _operator(_name, _symbol, False))
    setattr(Expression, f"__r{_name}__", _operator(_name, _symbol, True))
for _name, _symbol in _COMPARISONS.items():
    setattr(Expression, f"__{_name}__", _operator(_name, _symbol, False))
del _name, _symbol


def col(col_name: Hashable) -> Expression:
    """A column of whatever frame the expression is used on.

    Anything that takes `lambda df: df[col_name]`, such as `DataFrame.assign`
    or `DataFrame.loc`, takes `col(col_name)` too.

    Args:
        col_name: The column's name.

    Returns:
        The expression.

    Raises:
        TypeError: When the name cannot be hashed.
    """
    if not isinstance(col_name, Hashable):
        raise TypeError(f"Expected Hashable, got: {type(col_name)}")

    def func(df: Any) -> Any:
        if col_name not in df.columns:
            names = str(list(df.columns))
            if len(names) > 90:
                names = names[:90] + "...]"
            raise ValueError(
                f"Column '{col_name}' not found in given DataFrame.\n\n"
                f"Hint: did you mean one of {names} instead?"
            )
        return df[col_name]

    return Expression(func, f"col({col_name!r})")


def applied(value: Any, obj: Any) -> Any:
    """`value` read on `obj`: an expression is read, a function is called, anything else is itself.

    An expression is callable, since calling one makes a call expression, so it
    is read before a function would be called.
    """
    if isinstance(value, Expression):
        return value._eval_expression(obj)
    return value(obj) if callable(value) else value
