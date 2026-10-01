"""`DataFrame.query` and `DataFrame.eval`, read with Python's own parser.

pandas reads a query as Python with three changes, and so does this module.
`&` and `|` bind looser than a comparison, so `a > 2 & b < 3` is two
comparisons joined rather than a comparison of `2 & b`. A name in backticks may
hold spaces or anything else a column name can hold. A name after `@` is a
variable of the caller rather than a column. Everything else is Python syntax,
and each node is worked out with the column arithmetic firepanda already has,
so `a + b > 5` is one sum and one comparison over whole columns rather than a
loop over rows.

The rules below were measured against pandas 3.0.

- A bare name is a column first, then the name of the row labels, then
  `index` for the row labels, and anything else is `UndefinedVariableError`.
- `==` and `!=` against a list, and `in` and `not in` against a list or a
  column, test membership, the way `isin` does.
- A chain such as `1 < a < 4` is each comparison joined with and.
- A query keeps the rows where the answer is True. An answer that is not a
  column of flags is handed to `loc`, which is where pandas' KeyError for
  `df.query("a")` comes from.
- `is`, `lambda`, comprehensions and the like raise NotImplementedError, as
  pandas does for the nodes it does not read.
"""

from __future__ import annotations

import ast
import operator
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .errors import InvalidArgumentError, UndefinedVariableError

if TYPE_CHECKING:
    from ._frame import DataFrame

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _math(name: str) -> Any:
    """numpy's function of this name over numbers, columns or frames, keeping the labels.

    A column's values go through numpy and come back as a column with the same
    labels, named as the columns given were when they all share a name. A frame
    goes through column by column.
    """

    def function(*args: Any) -> Any:
        import numpy

        from ._frame import DataFrame, Series

        how = getattr(numpy, name)
        frame = next((each for each in args if isinstance(each, DataFrame)), None)
        if frame is not None:
            columns = {
                column: function(*(each[column] if each is frame else each for each in args))
                for column in frame.columns
            }
            return DataFrame(columns, index=frame.index)
        columns = [each for each in args if isinstance(each, Series)]
        if not columns:
            return how(*args)
        values = [each.to_numpy() if isinstance(each, Series) else each for each in args]
        names = {each.name for each in columns}
        named = names.pop() if len(names) == 1 else None
        return Series(how(*values), index=columns[0].index, name=named)

    function.__name__ = name
    return function


_FUNCTIONS: dict[str, Any] = {
    name: _math(name)
    for name in (
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "expm1",
        "log1p",
        "sqrt",
        "sinh",
        "cosh",
        "tanh",
        "arcsin",
        "arccos",
        "arctan",
        "arccosh",
        "arcsinh",
        "arctanh",
        "log10",
        "floor",
        "ceil",
        "arctan2",
    )
}
_FUNCTIONS["abs"] = abs
"""The functions a query may call by name, which are pandas' list."""

_AT = "__firepanda_at_"
_QUOTED = "__firepanda_quoted_"


def _rewritten(expr: str, pandas_parser: bool) -> tuple[str, dict[str, str]]:
    """The expression as Python, and the column names that were in backticks.

    Text inside quotes is copied as it is. A name in backticks becomes an
    identifier that stands for it, `@name` becomes an identifier that says it
    is a variable, and under pandas' parser `&` and `|` become `and` and `or`,
    which is what gives them the looser binding.
    """
    out: list[str] = []
    quoted: dict[str, str] = {}
    at = 0
    while at < len(expr):
        char = expr[at]
        if char in "'\"":
            end = at + 1
            while end < len(expr) and expr[end] != char:
                end += 2 if expr[end] == "\\" else 1
            out.append(expr[at : end + 1])
            at = end + 1
        elif char == "`":
            end = expr.find("`", at + 1)
            if end < 0:
                raise SyntaxError("unterminated backtick in the query")
            stand_in = f"{_QUOTED}{len(quoted)}"
            quoted[stand_in] = expr[at + 1 : end]
            out.append(stand_in)
            at = end + 1
        elif char == "@":
            out.append(_AT)
            at += 1
        elif pandas_parser and char in "&|":
            out.append(" and " if char == "&" else " or ")
            at += 1
        else:
            out.append(char)
            at += 1
    return "".join(out), quoted


class _Reader:
    """Works out one parsed expression against one frame."""

    def __init__(
        self,
        frame: DataFrame,
        quoted: dict[str, str],
        variables: Mapping[str, Any],
        resolvers: list[Mapping[str, Any]],
    ) -> None:
        self._frame = frame
        self._quoted = quoted
        self._variables = variables
        self._resolvers = resolvers

    def read(self, node: ast.AST) -> Any:
        """The value of one node."""
        handler = getattr(self, f"_{type(node).__name__}", None)
        if handler is None:
            raise NotImplementedError(f"'{type(node).__name__}' nodes are not implemented")
        return handler(node)

    def _Expression(self, node: ast.Expression) -> Any:
        return self.read(node.body)

    def _Constant(self, node: ast.Constant) -> Any:
        return node.value

    def _List(self, node: ast.List) -> Any:
        return [self.read(each) for each in node.elts]

    def _Tuple(self, node: ast.Tuple) -> Any:
        return [self.read(each) for each in node.elts]

    def _Name(self, node: ast.Name) -> Any:
        name = node.id
        if name.startswith(_AT):
            wanted = name[len(_AT) :]
            if wanted not in self._variables:
                raise UndefinedVariableError(wanted, is_local=True)
            return self._variables[wanted]
        name = self._quoted.get(name, name)
        for resolver in self._resolvers:
            if name in resolver:
                return resolver[name]
        frame = self._frame
        if frame is None:
            if name in self._variables:
                return self._variables[name]
            if name in ("True", "False", "None"):
                return {"True": True, "False": False, "None": None}[name]
            raise UndefinedVariableError(name)
        if name in list(frame.columns):
            return frame[name]
        labels = frame.index
        if name == "index" or (labels.name is not None and name == labels.name):
            return labels.to_series().rename(labels.name)
        if name in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[name]
        raise UndefinedVariableError(name)

    def _BoolOp(self, node: ast.BoolOp) -> Any:
        values = [self.read(each) for each in node.values]
        join = operator.and_ if isinstance(node.op, ast.And) else operator.or_
        answer = values[0]
        for value in values[1:]:
            answer = join(answer, value)
        return answer

    def _UnaryOp(self, node: ast.UnaryOp) -> Any:
        value = self.read(node.operand)
        if isinstance(node.op, (ast.Not, ast.Invert)):
            return (not value) if isinstance(value, bool) else ~value
        if isinstance(node.op, ast.USub):
            return -value
        return +value

    def _BinOp(self, node: ast.BinOp) -> Any:
        how = _BINARY.get(type(node.op))
        if how is None:
            raise NotImplementedError(f"'{type(node.op).__name__}' nodes are not implemented")
        return how(self.read(node.left), self.read(node.right))

    def _Compare(self, node: ast.Compare) -> Any:
        left = self.read(node.left)
        answer = None
        for op, right_node in zip(node.ops, node.comparators, strict=True):
            right = self.read(right_node)
            step = self._compared(op, left, right)
            answer = step if answer is None else answer & step
            left = right
        return answer

    def _compared(self, op: ast.cmpop, left: Any, right: Any) -> Any:
        """One link of a comparison, with membership where pandas reads it."""
        if isinstance(op, (ast.In, ast.NotIn)) or (
            isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(right, list)
        ):
            inside = self._member(left, right)
            return ~inside if isinstance(op, (ast.NotIn, ast.NotEq)) else inside
        how = _COMPARE.get(type(op))
        if how is None:
            raise NotImplementedError(f"'{type(op).__name__}' nodes are not implemented")
        return how(left, right)

    def _member(self, left: Any, right: Any) -> Any:
        """Whether each value on the left is among the values on the right."""
        values = right.tolist() if hasattr(right, "tolist") else right
        if not isinstance(values, list):
            values = [values]
        if hasattr(left, "isin"):
            return left.isin(values)
        return left in values

    def _Attribute(self, node: ast.Attribute) -> Any:
        if node.attr.startswith("_"):
            raise NotImplementedError("a query reads no private attribute")
        return getattr(self.read(node.value), node.attr)

    def _Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name) and not node.func.id.startswith(_AT):
            function = _FUNCTIONS.get(node.func.id)
            if function is None:
                raise NotImplementedError(
                    f"the function {node.func.id!r} is not supported in a query yet"
                )
        elif isinstance(node.func, (ast.Name, ast.Attribute)):
            function = self.read(node.func)
        else:
            raise TypeError("Only named functions are supported")
        arguments = [self.read(each) for each in node.args]
        keywords = {each.arg: self.read(each.value) for each in node.keywords if each.arg}
        return function(*arguments, **keywords)

    def _Subscript(self, node: ast.Subscript) -> Any:
        return self.read(node.value)[self.read(node.slice)]


def _caller_variables(
    level: int, local_dict: Mapping[str, Any] | None, global_dict: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The variables `@name` can read: the caller's globals, then its locals."""
    found: dict[str, Any] = {}
    # Four frames up is whoever called `df.query`: this function, `_evaluate`,
    # the module's `query` or `evaluate`, and the method sit in between. The
    # wrappers that carry `attrs` add frames of their own, which are not counted.
    caller: Any = sys._getframe(0)
    steps = 4 + level
    while caller is not None and steps:
        caller = caller.f_back
        if caller is not None and not caller.f_code.co_filename.endswith("_attrs.py"):
            steps -= 1
    if caller is not None:
        found.update(caller.f_globals)
        found.update(caller.f_locals)
    if global_dict is not None:
        found.update(global_dict)
    if local_dict is not None:
        found.update(local_dict)
    return found


def _evaluate(
    frame: DataFrame,
    expr: Any,
    parser: str,
    engine: Any,
    local_dict: Mapping[str, Any] | None,
    global_dict: Mapping[str, Any] | None,
    resolvers: list[Mapping[str, Any]] | None,
    level: int,
) -> tuple[Any, str | None]:
    """The value of an expression, and the column it assigns to when it is `c = ...`."""
    if not isinstance(expr, str):
        raise InvalidArgumentError(f"expr must be a string to be evaluated, {type(expr)} given")
    if not expr.strip():
        raise InvalidArgumentError("expr cannot be an empty string")
    if parser not in ("pandas", "python"):
        raise KeyError(f"Invalid parser '{parser}' passed, valid parsers are ['pandas', 'python']")
    if engine not in (None, "python", "numexpr"):
        raise KeyError(f"Invalid engine '{engine}' passed, valid engines are ['numexpr', 'python']")
    variables = _caller_variables(level, local_dict, global_dict)
    text, quoted = _rewritten(expr.strip(), parser == "pandas")
    target = None
    tree = ast.parse(text, mode="exec")
    if len(tree.body) != 1:
        raise NotImplementedError("only one expression is read at a time for now")
    statement = tree.body[0]
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        name = statement.targets[0]
        if not isinstance(name, ast.Name):
            raise NotImplementedError("an assignment in eval names one column")
        target = quoted.get(name.id, name.id)
        body: ast.AST = statement.value
    elif isinstance(statement, ast.Expr):
        body = statement.value
    else:
        raise NotImplementedError(f"'{type(statement).__name__}' nodes are not implemented")
    reader = _Reader(frame, quoted, variables, list(resolvers or []))
    return reader.read(body), target


def query(
    frame: DataFrame,
    expr: Any,
    parser: str,
    engine: Any,
    local_dict: Mapping[str, Any] | None,
    global_dict: Mapping[str, Any] | None,
    resolvers: list[Mapping[str, Any]] | None,
    level: int,
) -> DataFrame:
    """The rows of `frame` where `expr` is True."""
    from ._frame import Series

    answer, target = _evaluate(
        frame, expr, parser, engine, local_dict, global_dict, resolvers, level
    )
    if target is not None:
        raise InvalidArgumentError("cannot assign without a target object")
    if isinstance(answer, Series):
        if answer.dtype == "bool":
            return frame[answer]
        answer = answer.tolist()
    return frame.loc[answer]


def evaluate(
    frame: DataFrame,
    expr: Any,
    parser: str,
    engine: Any,
    local_dict: Mapping[str, Any] | None,
    global_dict: Mapping[str, Any] | None,
    resolvers: list[Mapping[str, Any]] | None,
    level: int,
) -> Any:
    """The value of `expr`, or the frame with a column assigned for `c = ...`.

    Several lines are assignments read in turn, each seeing the columns the
    lines before it made, as pandas reads them.
    """
    lines = [line.strip() for line in expr.splitlines()] if isinstance(expr, str) else []
    lines = [line for line in lines if line]
    if len(lines) > 1:
        for line in lines:
            answer, target = _evaluate(
                frame, line, parser, engine, local_dict, global_dict, resolvers, level
            )
            if target is None:
                raise InvalidArgumentError(
                    "Multi-line expressions are only valid if all expressions contain an assignment"
                )
            frame = frame.assign(**{target: answer})
        return frame
    answer, target = _evaluate(
        frame, expr, parser, engine, local_dict, global_dict, resolvers, level
    )
    if target is None:
        return answer
    return frame.assign(**{target: answer})


def top_level(
    expr: Any,
    parser: str,
    engine: Any,
    local_dict: Mapping[str, Any] | None,
    global_dict: Mapping[str, Any] | None,
    resolvers: Any,
    level: int,
    target: Any,
    inplace: bool,
) -> Any:
    """`pandas.eval`: an expression over the caller's variables rather than a frame's columns.

    A bare name is a variable of the caller, so `@` is refused. `c = ...`
    assigns into `target`, a copy of it unless `inplace`, and several lines
    are read in turn, each seeing what the lines before it assigned.
    """
    if isinstance(expr, list | tuple):
        return [
            top_level(
                each, parser, engine, local_dict, global_dict, resolvers, level + 1, None, False
            )
            for each in expr
        ]
    if not isinstance(expr, str):
        return expr
    if parser not in ("pandas", "python"):
        raise KeyError(
            f"Invalid parser '{parser}' passed, valid parsers are dict_keys(['python', 'pandas'])"
        )
    if engine not in (None, "python", "numexpr"):
        raise KeyError(f"Invalid engine '{engine}' passed, valid engines are ['numexpr', 'python']")
    lines = [line.strip() for line in expr.strip().splitlines() if line.strip()]
    if not lines:
        raise InvalidArgumentError("expr cannot be an empty string")
    if len(lines) > 1 and target is None:
        raise InvalidArgumentError(
            "multi-line expressions are only valid in the context of data, use DataFrame.eval"
        )
    # Three frames up is whoever called `pandas.eval`: `_caller_variables`,
    # this function and the module's `eval` sit in between, one fewer than
    # the four `_caller_variables` counts for a frame's method.
    variables = _caller_variables(level - 1, local_dict, global_dict)
    written = target
    answer = None
    assigned = False
    for line in lines:
        text, quoted = _rewritten(line, parser == "pandas")
        if _AT in text:
            raise SyntaxError(
                "The '@' prefix is not allowed in top-level eval calls.\n"
                "please refer to your variables by name without the '@' prefix."
            )
        tree = ast.parse(text, mode="exec")
        statement = tree.body[0]
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            name = statement.targets[0]
            if not isinstance(name, ast.Name):
                raise NotImplementedError("an assignment in eval names one column")
            body: ast.AST = statement.value
            column: str | None = quoted.get(name.id, name.id)
        elif isinstance(statement, ast.Expr):
            body, column = statement.value, None
        else:
            raise NotImplementedError(f"'{type(statement).__name__}' nodes are not implemented")
        reader = _Reader(None, quoted, variables, list(resolvers or []))  # type: ignore[arg-type]
        answer = reader.read(body)
        if column is None:
            continue
        if written is None:
            raise InvalidArgumentError("cannot assign without a target object")
        if not assigned and not inplace:
            written = written.copy()
        assigned = True
        try:
            written[column] = answer
        except (TypeError, IndexError) as error:
            raise InvalidArgumentError("Cannot assign expression output to target") from error
    if inplace and not assigned:
        raise InvalidArgumentError("Cannot operate inplace if there is no assignment")
    if not assigned:
        return answer
    return None if inplace else written


def eval(
    expr: Any,
    parser: str = "pandas",
    engine: Any = None,
    local_dict: Mapping[str, Any] | None = None,
    global_dict: Mapping[str, Any] | None = None,
    resolvers: Any = (),
    level: int = 0,
    target: Any = None,
    inplace: bool = False,
) -> Any:
    """The value of a Python expression over the caller's variables, read as `DataFrame.eval` reads.

    `pd.eval("df.a + df.b")` adds two columns of the caller's `df`. Under
    pandas' parser `&` and `|` bind looser than a comparison. `c = ...`
    assigns a column of `target`, which is answered unless `inplace`.

    Args:
        expr: The expression, several lines of assignments, or a list of expressions.
        parser: `"pandas"` or `"python"`.
        engine: Accepted for pandas' sake, since every engine answers the same.
        local_dict: Variables read before the caller's own.
        global_dict: Variables read after `local_dict`.
        resolvers: Mappings read before any variable.
        level: How many more frames up the caller's variables are.
        target: What `c = ...` assigns into.
        inplace: Whether to assign into `target` itself rather than into a copy.

    Returns:
        The value, or the target with the assignment made, or None when `inplace`.

    Raises:
        ValueError: For an empty expression, an assignment with nothing to assign
            into, or several lines with no target.
        SyntaxError: For a name after `@`, which only a frame's `eval` reads.
    """
    return top_level(
        expr, parser, engine, local_dict, global_dict, resolvers, level, target, inplace
    )
