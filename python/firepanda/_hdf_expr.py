"""The where clauses of an HDF5 table, as pandas parses and splits them.

pandas reads a clause such as `index > 5 & columns = ['A', 'B']` with its own
expression machinery and splits it in two. The terms on the table's index and
data columns become a condition string that PyTables hands to numexpr, so only
the matching rows are read from disk, and the terms on the column labels, or
an equality on too many values, become a filter applied to the frame once it
is read. This module is that split, ported from pandas' `PyTablesExpr` with
the same rules for which term goes where, the same conversion of each value to
the type of its column, and the same messages. A name on the right of a
comparison is looked up among the caller's variables, and a name that is not
found stands for itself as text, as pandas treats it.
"""

from __future__ import annotations

import ast
import datetime
import io
import math
import os
import sys
import tokenize
from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

_CMP_OPS = (">", "<", ">=", "<=", "==", "!=", "in", "not in")
_BOOL_OPS = ("&", "|", "and", "or")
_ARITH_OPS = ("+", "-", "*", "/", "**", "//", "%")
_BINARY_OPS = _CMP_OPS + _BOOL_OPS + _ARITH_OPS
_UNARY_OPS = ("+", "-", "~", "not")

_AST_OPS: dict[type, str] = {
    ast.Gt: ">",
    ast.Lt: "<",
    ast.GtE: ">=",
    ast.LtE: "<=",
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.In: "in",
    ast.NotIn: "not in",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.And: "&",
    ast.Or: "|",
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Pow: "**",
    ast.FloorDiv: "//",
    ast.Mod: "%",
}

_LOCAL_TAG = "__pd_eval_local_"
_MAX_SELECTORS = 31
_PACKAGE = os.path.dirname(os.path.abspath(__file__))


def _default_globals() -> dict[str, Any]:
    from . import Timestamp

    return {
        "Timestamp": Timestamp,
        "datetime": datetime.datetime,
        "True": True,
        "False": False,
        "list": list,
        "tuple": tuple,
        "inf": math.inf,
        "Inf": math.inf,
    }


def caller_scope() -> dict[str, Any]:
    """The globals and locals of the first frame outside firepanda."""
    frame: Any = sys._getframe(1)
    while frame is not None and os.path.dirname(
        os.path.abspath(frame.f_code.co_filename)
    ).startswith(_PACKAGE):
        frame = frame.f_back
    found: dict[str, Any] = {}
    if frame is not None:
        found.update(frame.f_globals)
        found.update(frame.f_locals)
    return found


def is_list_like(value: Any) -> bool:
    return hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict))


def maybe_expression(s: Any) -> bool:
    """Whether a where entry is text holding an operator, as pandas decides it."""
    if not isinstance(s, str):
        return False
    return any(op in s for op in _BINARY_OPS + _UNARY_OPS + ("=",))


def _preparse(source: str) -> str:
    """pandas' rewrites before parsing: `@name`, `&` and `|` as words, `=` as `==`."""
    tokens = []
    previous_at = False
    for kind, text, *_ in tokenize.generate_tokens(io.StringIO(source).readline):
        if kind == tokenize.OP and text == "@":
            previous_at = True
            continue
        if previous_at and kind == tokenize.NAME:
            text = _LOCAL_TAG + text
        previous_at = False
        if kind == tokenize.OP and text == "&":
            kind, text = tokenize.NAME, "and"
        elif kind == tokenize.OP and text == "|":
            kind, text = tokenize.NAME, "or"
        elif kind == tokenize.OP and text == "=":
            text = "=="
        tokens.append((kind, text))
    return tokenize.untokenize(tokens)


def _text(value: Any) -> str:
    return str(value)


class _Term:
    """A name or a value in a clause, with the value pandas resolves it to."""

    is_term = True

    def __init__(self, value: Any, name: Any = None) -> None:
        self.value = value
        self.name = name

    def __repr__(self) -> str:
        return _text(self.name if self.name is not None else self.value)


class TermValue:
    """A value converted for its column, as numexpr is to see it."""

    def __init__(self, value: Any, converted: Any, kind: str) -> None:
        self.value = value
        self.converted = converted
        self.kind = kind

    def tostring(self, encoding: str | None) -> str:
        if self.kind == "string":
            if encoding is not None:
                return str(self.converted)
            return f'"{self.converted}"'
        if self.kind == "float":
            return repr(self.converted)
        return str(self.converted)


def _decoded(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if type(value).__module__ == "numpy" and hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return value


def _unit_value(value: Any, unit: str) -> int:
    """The count of `unit` in a moment or a span, which pandas reads as `_value`."""
    per = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}[unit]
    return int(value.value) // per


class BinOp:
    op: str

    def __init__(self, op: str, lhs: Any, rhs: Any, queryables: dict[str, Any], encoding: Any):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs
        self.queryables = queryables
        self.encoding = encoding
        self.condition: str | None = None

    is_term = False

    def __repr__(self) -> str:
        return f"({self.lhs!r} {self.op} {self.rhs!r})"

    def prune(self, klass: type) -> Any:
        def pr(left: Any, right: Any) -> Any:
            if left is None:
                return right
            if right is None:
                return left
            k = klass
            if isinstance(left, ConditionBinOp):
                if isinstance(right, ConditionBinOp):
                    k = JointConditionBinOp
                elif isinstance(left, k):
                    return left
                elif isinstance(right, k):
                    return right
            elif isinstance(left, FilterBinOp):
                if isinstance(right, FilterBinOp):
                    k = JointFilterBinOp
                elif isinstance(left, k):
                    return left
                elif isinstance(right, k):
                    return right
            return k(
                self.op, left, right, queryables=self.queryables, encoding=self.encoding
            ).evaluate()

        left, right = self.lhs, self.rhs
        if left.is_term and right.is_term:
            return pr(left.value, right.value)
        if not left.is_term and right.is_term:
            return pr(left.prune(klass), right.value)
        if left.is_term and not right.is_term:
            return pr(left.value, right.prune(klass))
        return pr(left.prune(klass), right.prune(klass))

    def conform(self, rhs: Any) -> Any:
        if not is_list_like(rhs):
            rhs = [rhs]
        if type(rhs).__module__ == "numpy":
            rhs = rhs.ravel()
        return rhs

    @property
    def is_valid(self) -> bool:
        try:
            return self.lhs in self.queryables
        except TypeError:
            return False

    @property
    def is_in_table(self) -> bool:
        return self.queryables.get(self.lhs) is not None

    @property
    def kind(self) -> Any:
        return getattr(self.queryables.get(self.lhs), "kind", None)

    @property
    def meta(self) -> Any:
        return getattr(self.queryables.get(self.lhs), "meta", None)

    @property
    def metadata(self) -> Any:
        return getattr(self.queryables.get(self.lhs), "metadata", None)

    def generate(self, v: TermValue) -> str:
        val = v.tostring(self.encoding)
        return f"({self.lhs} {self.op} {val})"

    def convert_value(self, conv_val: Any) -> TermValue:
        from . import Timedelta, Timestamp

        def stringify(value: Any) -> Any:
            if self.encoding is not None:
                return _text(value).encode(self.encoding, "replace")
            return _text(value)

        kind = _decoded(self.kind)
        meta = _decoded(self.meta)
        if kind == "datetime" or (kind and kind.startswith("datetime64")):
            if isinstance(conv_val, (int, float)):
                conv_val = stringify(conv_val)
            conv_val = _decoded(conv_val)
            unit = "ns"
            if "[" in kind:
                unit = kind.split("[")[-1][:-1]
            conv_val = Timestamp(conv_val).as_unit(unit)
            if conv_val.tz is not None:
                conv_val = conv_val.tz_convert("UTC")
            converted = _unit_value(conv_val, unit)
            return TermValue(conv_val, converted, kind)
        if kind.startswith("timedelta"):
            unit = "ns"
            if "[" in kind:
                unit = kind.split("[")[-1][:-1]
            if isinstance(conv_val, str):
                conv_val = Timedelta(conv_val)
            elif isinstance(conv_val, (int, float)) and not isinstance(conv_val, bool):
                conv_val = Timedelta(conv_val, unit="s")
            else:
                conv_val = Timedelta(conv_val)
            converted = _unit_value(conv_val.as_unit(unit), unit)
            return TermValue(converted, converted, kind)
        if meta == "category":
            metadata = list(self.metadata)
            result = metadata.index(conv_val) if conv_val in metadata else -1
            return TermValue(result, result, "integer")
        if kind == "integer":
            try:
                v_dec = Decimal(conv_val)
            except (InvalidOperation, TypeError, ValueError):
                float(conv_val)
            else:
                conv_val = int(v_dec.to_integral_exact(rounding="ROUND_HALF_EVEN"))
            return TermValue(conv_val, conv_val, kind)
        if kind == "float":
            conv_val = float(conv_val)
            return TermValue(conv_val, conv_val, kind)
        if kind == "bool":
            if isinstance(conv_val, str):
                conv_val = conv_val.strip().lower() not in [
                    "false",
                    "f",
                    "no",
                    "n",
                    "none",
                    "0",
                    "[]",
                    "{}",
                    "",
                ]
            else:
                conv_val = bool(conv_val)
            return TermValue(conv_val, conv_val, kind)
        if isinstance(conv_val, str):
            return TermValue(conv_val, stringify(conv_val), "string")
        raise TypeError(f"Cannot compare {conv_val} of type {type(conv_val)} to {kind} column")


class FilterBinOp(BinOp):
    filter: tuple[Any, Any, list[Any]] | None = None

    def __repr__(self) -> str:
        if self.filter is None:
            return "Filter: Not Initialized"
        return f"[Filter : [{self.filter[0]}] -> [{self.filter[1]}]"

    def invert(self) -> FilterBinOp:
        if self.filter is not None:
            self.filter = (self.filter[0], self.generate_filter_op(invert=True), self.filter[2])
        return self

    def format(self) -> list[Any]:
        return [self.filter]

    def evaluate(self) -> FilterBinOp | None:
        if not self.is_valid:
            raise ValueError(f"query term is not valid [{self}]")
        rhs = self.conform(self.rhs)
        values = list(rhs)
        if self.is_in_table:
            if self.op in ["==", "!="] and len(values) > _MAX_SELECTORS:
                self.filter = (self.lhs, self.generate_filter_op(), values)
                return self
            return None
        if self.op in ["==", "!="]:
            self.filter = (self.lhs, self.generate_filter_op(), values)
        else:
            raise TypeError(f"passing a filterable condition to a non-table indexer [{self}]")
        return self

    def generate_filter_op(
        self, invert: bool = False
    ) -> Callable[[list[Any], list[Any]], list[bool]]:
        if (self.op == "!=" and not invert) or (self.op == "==" and invert):
            return lambda axis, vals: [not hit for hit in _isin(axis, vals)]
        return _isin


def _isin(axis: Any, vals: Any) -> list[bool]:
    wanted = list(vals)
    hashed = set()
    loose = []
    for v in wanted:
        try:
            hashed.add(v)
        except TypeError:
            loose.append(v)
    nan = any(isinstance(v, float) and v != v for v in wanted)
    out = []
    for v in axis:
        try:
            hit = v in hashed
        except TypeError:
            hit = False
        if not hit and loose:
            hit = any(v == w for w in loose)
        if not hit and nan and isinstance(v, float) and v != v:
            hit = True
        out.append(bool(hit))
    return out


class JointFilterBinOp(FilterBinOp):
    def format(self) -> list[Any]:
        raise NotImplementedError("unable to collapse Joint Filters")

    def evaluate(self) -> JointFilterBinOp:
        return self


class ConditionBinOp(BinOp):
    def __repr__(self) -> str:
        return f"[Condition : [{self.condition}]]"

    def invert(self) -> Any:
        raise NotImplementedError("cannot use an invert condition when passing to numexpr")

    def format(self) -> str | None:
        return self.condition

    def evaluate(self) -> ConditionBinOp | None:
        if not self.is_valid:
            raise ValueError(f"query term is not valid [{self}]")
        if not self.is_in_table:
            return None
        rhs = self.conform(self.rhs)
        values = [self.convert_value(v) for v in rhs]
        if self.op in ["==", "!="]:
            if len(values) <= _MAX_SELECTORS:
                vs = [self.generate(v) for v in values]
                self.condition = f"({' | '.join(vs)})"
            else:
                return None
        else:
            self.condition = self.generate(values[0])
        return self


class JointConditionBinOp(ConditionBinOp):
    def evaluate(self) -> JointConditionBinOp:
        self.condition = f"({self.lhs.condition} {self.op} {self.rhs.condition})"
        return self


class UnaryOp:
    is_term = False

    def __init__(self, op: str, operand: Any) -> None:
        self.op = op
        self.operand = operand

    def __repr__(self) -> str:
        return f"{self.op}({self.operand!r})"

    def prune(self, klass: type) -> Any:
        if self.op != "~":
            raise NotImplementedError("UnaryOp only support invert type ops")
        operand = self.operand.prune(klass)
        if operand is not None and (
            (issubclass(klass, ConditionBinOp) and operand.condition is not None)
            or (
                not issubclass(klass, ConditionBinOp)
                and issubclass(klass, FilterBinOp)
                and operand.filter is not None
            )
        ):
            return operand.invert()
        return None


class _Visitor:
    """Reads a parsed clause into terms and operations, as pandas' visitor does."""

    def __init__(self, scope: Mapping[str, Any], queryables: dict[str, Any], encoding: Any):
        self.scope = scope
        self.queryables = queryables
        self.encoding = encoding

    def visit(self, node: Any, side: str | None = None) -> Any:
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise NotImplementedError(f"'{type(node).__name__}' nodes are not implemented")
        return method(node, side)

    def visit_Module(self, node: ast.Module, side: Any) -> Any:
        if len(node.body) != 1:
            raise SyntaxError("only a single expression is allowed")
        return self.visit(node.body[0])

    def visit_Expr(self, node: ast.Expr, side: Any) -> Any:
        return self.visit(node.value)

    def visit_Constant(self, node: ast.Constant, side: Any) -> _Term:
        return _Term(node.value)

    def visit_Name(self, node: ast.Name, side: Any) -> _Term:
        name = node.id
        if side == "left":
            if name not in self.queryables:
                raise NameError(f"name {name!r} is not defined")
            return _Term(name, name)
        if name.startswith(_LOCAL_TAG) or name not in self.scope:
            return _Term(name, name)
        return _Term(self.scope[name], name)

    def visit_List(self, node: ast.List, side: Any) -> _Term:
        return _Term([self.visit(e).value for e in node.elts])

    visit_Tuple = visit_List

    def visit_Index(self, node: Any, side: Any) -> Any:
        return self.visit(node.value).value

    def visit_Subscript(self, node: ast.Subscript, side: Any) -> _Term:
        value = self.visit(node.value)
        slobj = self.visit(node.slice)
        value = getattr(value, "value", value)
        if isinstance(slobj, _Term):
            slobj = slobj.value
        try:
            return _Term(value[slobj])
        except TypeError as err:
            raise ValueError(f"cannot subscript {value!r} with {slobj!r}") from err

    def visit_Slice(self, node: ast.Slice, side: Any) -> _Term:
        parts = [
            None if p is None else self.visit(p).value for p in (node.lower, node.upper, node.step)
        ]
        return _Term(slice(*parts))

    def visit_Attribute(self, node: ast.Attribute, side: Any) -> Any:
        attr = node.attr
        value = node.value
        if isinstance(node.ctx, ast.Load):
            resolved = self.visit(value)
            resolved = getattr(resolved, "value", resolved)
            try:
                return _Term(getattr(resolved, attr))
            except AttributeError:
                if isinstance(value, ast.Name) and value.id == attr:
                    return resolved
        raise ValueError(f"Invalid Attribute context {type(node.ctx).__name__}")

    def visit_Call(self, node: ast.Call, side: Any) -> _Term:
        if isinstance(node.func, ast.Attribute) and node.func.attr != "__call__":
            res = self.visit_Attribute(node.func, None)
        elif not isinstance(node.func, ast.Name):
            raise TypeError("Only named functions are supported")
        else:
            res = self.visit(node.func)
        res = getattr(res, "value", res)
        args = [self.visit(arg).value for arg in node.args]
        kwargs = {k.arg: self.visit(k.value).value for k in node.keywords if k.arg}
        return _Term(res(*args, **kwargs))

    def visit_UnaryOp(self, node: ast.UnaryOp, side: Any) -> Any:
        if isinstance(node.op, (ast.Not, ast.Invert)):
            return UnaryOp("~", self.visit(node.operand))
        if isinstance(node.op, ast.USub):
            return _Term(-self.visit(node.operand).value)
        if isinstance(node.op, ast.UAdd):
            raise NotImplementedError("Unary addition not supported")
        return None

    def _binop(self, op: str, left: Any, right: Any) -> BinOp:
        return BinOp(op, left, right, self.queryables, self.encoding)

    def visit_BinOp(self, node: ast.BinOp, side: Any) -> BinOp:
        op = _AST_OPS.get(type(node.op))
        if op is None:
            raise NotImplementedError(f"'{type(node.op).__name__}' nodes are not implemented")
        left = self.visit(node.left, side="left")
        right = self.visit(node.right, side="right")
        return self._binop(op, left, right)

    def visit_Compare(self, node: ast.Compare, side: Any) -> Any:
        ops = node.ops
        comps = node.comparators
        if len(comps) == 1:
            op = ops[0]
            if isinstance(op, ast.In):
                op = ast.Eq()
            return self.visit(ast.BinOp(op=op, left=node.left, right=comps[0]))
        left = node.left
        values = []
        for op, comp in zip(ops, comps, strict=True):
            values.append(ast.Compare(ops=[op], left=left, comparators=[comp]))
            left = comp
        return self.visit(ast.BoolOp(op=ast.And(), values=values))

    def visit_BoolOp(self, node: ast.BoolOp, side: Any) -> Any:
        op = _AST_OPS[type(node.op)]
        result = self.visit(node.values[0])
        for value in node.values[1:]:
            result = self._binop(op, result, self.visit(value))
        return result

    def visit_Assign(self, node: ast.Assign, side: Any) -> Any:
        cmpr = ast.Compare(ops=[ast.Eq()], left=node.targets[0], comparators=[node.value])
        return self.visit(cmpr)


def _validate_where(w: Any) -> Any:
    if not (isinstance(w, (PyTablesExpr, str)) or is_list_like(w)):
        raise TypeError(
            "where must be passed as a string, PyTablesExpr, or list-like of PyTablesExpr"
        )
    return w


def _flatten(items: Any) -> Any:
    for item in items:
        if is_list_like(item) and not isinstance(item, PyTablesExpr):
            yield from _flatten(item)
        else:
            yield item


class PyTablesExpr:
    """A where clause, with the caller's variables it may name."""

    def __init__(
        self,
        where: Any,
        queryables: dict[str, Any] | None = None,
        encoding: Any = None,
        scope_level: int = 0,
        scope: Mapping[str, Any] | None = None,
    ) -> None:
        where = _validate_where(where)
        self.encoding = encoding
        self.condition: Any = None
        self.filter: Any = None
        self.terms: Any = None
        found: dict[str, Any] | None = None
        if isinstance(where, PyTablesExpr):
            found = dict(where.scope)
            _where = where.expr
        elif is_list_like(where):
            where = list(where)
            for idx, w in enumerate(where):
                if isinstance(w, PyTablesExpr):
                    found = dict(w.scope)
                else:
                    where[idx] = _validate_where(w)
            _where = " & ".join([f"({w})" for w in _flatten(where)])
        else:
            _where = where
        self.expr = _where
        if found is None:
            found = dict(scope) if scope is not None else caller_scope()
        self.scope = {**_default_globals(), **found}
        if queryables is not None and isinstance(self.expr, str):
            self.terms = self.parse(queryables)

    def parse(self, queryables: dict[str, Any]) -> Any:
        visitor = _Visitor(self.scope, dict(queryables), self.encoding)
        tree = ast.fix_missing_locations(ast.parse(_preparse(self.expr)))
        return visitor.visit(tree)

    def __len__(self) -> int:
        return len(self.expr)

    def __repr__(self) -> str:
        if self.terms is not None:
            return repr(self.terms)
        return str(self.expr)

    def evaluate(self) -> tuple[Any, Any]:
        try:
            self.condition = self.terms.prune(ConditionBinOp)
        except AttributeError as err:
            raise ValueError(
                f"cannot process expression [{self.expr}], [{self}] is not a valid condition"
            ) from err
        try:
            self.filter = self.terms.prune(FilterBinOp)
        except AttributeError as err:
            raise ValueError(
                f"cannot process expression [{self.expr}], [{self}] is not a valid filter"
            ) from err
        return self.condition, self.filter


Term = PyTablesExpr


def ensure_term(where: Any) -> Any:
    """Each clause of `where` as a term holding the caller's variables."""
    if where is None:
        return None
    if isinstance(where, (list, tuple)):
        scope = caller_scope()
        where = [
            Term(term, scope=scope) if maybe_expression(term) else term
            for term in where
            if term is not None
        ]
    elif maybe_expression(where):
        where = Term(where, scope=caller_scope())
    return where if where is None or len(where) else None
