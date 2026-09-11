"""A plan as JSON, and the same JSON back as a plan.

`print.mojo` writes a plan for a person to read. This writes one for a program
to read, and reads it back, and the point of the pair is the round trip:
`from_json(to_json(p, r))` is the plan `p` again. That is what makes an
optimizer pass testable. A pass test is an input plan and an expected output
plan, and writing both as text rather than as thirty lines of builder calls is
the difference between a test somebody reads and a test somebody skips. It is
what docs/specs/sql/08-plan-and-optimizer.md asks for in its section 7, and the
reason it asks is section 9: every pass gets a unit test whose input and output
are written down.

## A tree, with the sharing written down

The natural JSON for a plan is a tree, because a plan is a tree and a tree in
JSON is a nested object that reads the way the query reads. The arena form,
which is what the plan actually is, would be two flat arrays of nodes holding
integer indices into each other, and nobody can read a diff of that.

A plan is not always a tree though. `cse.mojo` makes two equal expressions one
index, and `subplan.mojo` makes two equal nodes one node, and after either of
them the plan is a graph. A tree form would quietly copy the shared part, and
then the round trip of an optimized plan would not be the plan it started as.

So the sharing is written down. Anything reached more than once carries an `id`
where it is written out in full, and every later reach is `{"ref": id}` instead.
The ids count up in the order things are written rather than being the arena
index they already have, so that the document says nothing about how the arena
happened to get numbered and two plans of the same shape write the same bytes.
A plan with no sharing in it, which is every plan anybody writes by hand and most
plans a query lowers to, has no `id` and no `ref` anywhere in it and reads as a
plain tree.

## What is checked

Reading goes through the same builders a caller would use, so a plan that comes
out of here has had the same checks run over it as a plan built in code: inputs
that exist, names that cover the outputs, keys that are elementwise, arms that
are counted. A JSON file that describes an impossible plan is refused with the
error the builder gives, not turned into a plan that fails later.

What is not checked is the schema, exactly as in `node.mojo`, because that is
binding's job and binding is a pass that runs over the plan afterwards. A plan
that was bound before it was written is bound when it is read, because the
positions and the types are in the JSON, and a plan that was not is not.

## What is left out

A type that `LogicalType.write_to` cannot spell in full is refused by name
rather than written out lossily. A list and a struct are the two, since the part
that differs between two of them lives on the column and not on the type, and
writing `"list"` and reading it back would be inventing an element type.
"""

from std.collections.span import Span

from std.math import inf, isinf, isnan, nan

from firepanda.array.value import Value
from firepanda.dtype.lists import FLOAT, INTEGER, SIGNED, contains
from firepanda.dtype.logical import LogicalType, TypeKind, named_type
from firepanda.dtype.temporal import TimeUnit, TimeZone
from firepanda.io.jsonscan import (
    JSON_ARRAY,
    JSON_FALSE,
    JSON_NULL,
    JSON_NUMBER,
    JSON_OBJECT,
    JSON_STRING,
    JSON_TRUE,
    Member,
)
from firepanda.io.jsonscan import Value as JsonValue
from firepanda.io.jsonscan import (
    scan_array,
    scan_object,
    scan_value,
    skip_space,
)
from firepanda.io.jsonscan import text_of
from firepanda.io.parse import parse_float, parse_int
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.kernel.unary import UnaryOp
from firepanda.plan.expr import UNBOUND, ExprKind, Expressions
from firepanda.plan.node import (
    NO_LIMIT,
    SET_EXCEPT,
    SET_INTERSECT,
    SET_UNION,
    NodeKind,
    Plan,
)


struct Loaded(Movable):
    """A plan read out of JSON, and which of its nodes the document named.

    The root is carried rather than assumed, even though reading a tree always
    leaves it last, because a caller that has the pair does not have to know
    that and a later writer that emits something else does not have to keep it
    true.
    """

    var plan: Plan
    """The plan."""

    var root: Int
    """The node the document was about."""

    def __init__(out self, var plan: Plan, root: Int):
        """Holds the pair.

        Args:
            plan: The plan.
            root: The node the document was about.
        """
        self.plan = plan^
        self.root = root


def _quoted(text: StringSlice) -> String:
    """Writes text as a JSON string, with its quotes and its escapes.

    Args:
        text: The bytes.

    Returns:
        The quoted form.
    """
    var out = List[UInt8]()
    out.append(UInt8(34))
    var bytes = text.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c == UInt8(34) or c == UInt8(92):
            out.append(UInt8(92))
            out.append(c)
        elif c == UInt8(8):
            out.append(UInt8(92))
            out.append(UInt8(98))
        elif c == UInt8(9):
            out.append(UInt8(92))
            out.append(UInt8(116))
        elif c == UInt8(10):
            out.append(UInt8(92))
            out.append(UInt8(110))
        elif c == UInt8(12):
            out.append(UInt8(92))
            out.append(UInt8(102))
        elif c == UInt8(13):
            out.append(UInt8(92))
            out.append(UInt8(114))
        elif c < UInt8(32):
            # The rest of the control bytes have no short escape and have to go
            # out as a code point, which for anything under a space is always
            # four digits beginning with two zeroes.
            var digits = "0123456789abcdef".as_bytes()
            out.append(UInt8(92))
            out.append(UInt8(117))
            out.append(UInt8(48))
            out.append(UInt8(48))
            out.append(digits[Int(c) >> 4])
            out.append(digits[Int(c) & 15])
        else:
            out.append(c)
    out.append(UInt8(34))
    return String(StringSlice(unsafe_from_utf8=Span(out)))


def _type_text(type: LogicalType) raises -> String:
    """Writes a type as the name `_type_of` reads back.

    Args:
        type: The type.

    Returns:
        The name.

    Raises:
        Error: If the type is one whose name does not say all of it.
    """
    if type.kind == TypeKind.LIST or type.kind == TypeKind.STRUCT:
        raise Error(
            String(
                "a ",
                type,
                (
                    " does not say its element type, so a plan holding one does"
                    " not go to JSON and come back"
                ),
            )
        )
    return String(type)


def _type_of(name: String) raises -> LogicalType:
    """Reads a type name, including the ones `named_type` leaves out.

    `named_type` reads the spellings that are a single word, which is every type
    whose name says all of it except the timestamp and the duration. Those two
    carry a unit, and a timestamp may carry a zone as well, and both are written
    in the name here rather than in a field beside it so that the JSON says what
    an explain output says.

    Args:
        name: The name, spelled the way `LogicalType.write_to` writes it.

    Returns:
        The type.

    Raises:
        Error: If nothing is named that.
    """
    var width = name.byte_length()
    if name.startswith("datetime64[") and name.endswith("]"):
        var inside = name[byte = 11 : width - 1]
        var comma = inside.find(",")
        if comma == -1:
            return LogicalType.timestamp(_unit_of(inside))
        var zone = inside[byte = comma + 1 : inside.byte_length()].strip()
        return LogicalType.timestamp(
            _unit_of(inside[byte=0:comma]), TimeZone(String(zone))
        )
    if name.startswith("timedelta64[") and name.endswith("]"):
        return LogicalType.duration(_unit_of(name[byte = 12 : width - 1]))
    return named_type(name)


def _unit_of(name: StringSlice) raises -> TimeUnit:
    """Reads a time unit as `TimeUnit.write_to` writes it.

    Args:
        name: The unit.

    Returns:
        The unit.

    Raises:
        Error: If it is not one of the four.
    """
    if name == "s":
        return TimeUnit.SECOND
    if name == "ms":
        return TimeUnit.MILLI
    if name == "us":
        return TimeUnit.MICRO
    if name == "ns":
        return TimeUnit.NANO
    raise Error(String("no time unit is named ", name))


def _literal_text(value: Value) raises -> String:
    """Writes a constant as the JSON value that reads back as the same one.

    A null is `null` whatever type it has, since the type is beside it. A
    boolean is a boolean and text is a string. A number is a number, written at
    its own width and its own family, which is the part that has to be spelled
    out: `Value.write_to` prints every integer through int64, so a uint64 above
    the top of int64 prints as a negative number there, and that is fine for a
    frame nobody reads back and wrong for a document that has to.

    The three floats JSON has no syntax for go out as strings. A reader that saw
    a bare `NaN` would be a reader of something that is not JSON.

    Args:
        value: The constant.

    Returns:
        The JSON value.

    Raises:
        Error: If the type is one that has no written form.
    """
    if not value.present:
        return String("null")
    if value.type.kind == TypeKind.BOOL:
        return String("true") if value.bits != 0 else String("false")
    if value.type.is_variable_width():
        return _quoted(value.text.value())
    if value.type.is_float():
        # A float prints as `nan`, `inf` or `-inf` when it is one of the three,
        # and JSON has no syntax for any of them, so those three go out quoted
        # and everything else goes out as the number it is.
        if isnan(value.real) or isinf(value.real):
            return _quoted(String(value.real))
        return String(value.real)
    if contains[SIGNED](value.type.physical):
        return String(value.as_scalar[DType.int64]())
    return String(value.bits)


def _value_of(
    type: LogicalType, bytes: Span[UInt8, _], written: JsonValue
) raises -> Value:
    """Reads a constant back at the type the document gave it.

    The value is built at the physical width and then told what it is, which is
    what `Value.duration` and `Value.timestamp` do and for the same reason: the
    typed constructors take a dtype and a date is not a dtype.

    Args:
        type: The type the constant has.
        bytes: The document.
        written: Where the value is in it.

    Returns:
        The constant.

    Raises:
        Error: If what is written is not a value of that type.
    """
    if written.kind == JSON_NULL:
        return Value(null=type)
    if type.kind == TypeKind.BOOL:
        if written.kind != JSON_TRUE and written.kind != JSON_FALSE:
            raise Error("a bool is written true or false")
        var out = Value(written.kind == JSON_TRUE)
        out.type = type
        return out^
    if type.is_variable_width():
        if written.kind != JSON_STRING:
            raise Error(String("a ", type, " is written as a string"))
        var out = Value(text_of(bytes, written))
        out.type = type
        return out^
    if type.is_float():
        return _float_of(type, bytes, written)
    if written.kind != JSON_NUMBER:
        raise Error(String("a ", type, " is written as a number"))
    var field = bytes[written.start : written.end]
    comptime for dt in INTEGER:
        if type.physical == dt:
            var got = parse_int[dt](field)
            if not got.ok:
                raise Error(
                    String(
                        text_of(bytes, written),
                        " is not a ",
                        type,
                        ", and a plan will not hold a number it cannot hold",
                    )
                )
            var out = Value(got.value)
            out.type = type
            return out^
    raise Error(String("a plan does not carry a constant of type ", type))


def _float_of(
    type: LogicalType, bytes: Span[UInt8, _], written: JsonValue
) raises -> Value:
    """Reads a float back, including the three that are written as words.

    Args:
        type: The float type.
        bytes: The document.
        written: Where the value is in it.

    Returns:
        The constant.

    Raises:
        Error: If what is written is not a float.
    """
    var real: Float64
    if written.kind == JSON_STRING:
        var word = text_of(bytes, written)
        if word == "nan":
            real = nan[DType.float64]()
        elif word == "inf":
            real = inf[DType.float64]()
        elif word == "-inf":
            real = -inf[DType.float64]()
        else:
            raise Error(
                String(
                    word,
                    (
                        " is not a float, and the only ones written as words"
                        " are nan, inf and -inf"
                    ),
                )
            )
    elif written.kind == JSON_NUMBER:
        var got = parse_float[DType.float64](bytes[written.start : written.end])
        if not got.ok:
            raise Error(String(text_of(bytes, written), " is not a float"))
        real = got.value
    else:
        raise Error(String("a ", type, " is written as a number"))
    comptime for dt in FLOAT:
        if type.physical == dt:
            var out = Value(real.cast[dt]())
            out.type = type
            return out^
    raise Error(String("a plan does not carry a constant of type ", type))


def _agg_text(op: Int) raises -> String:
    """Writes an aggregate code as its word.

    Args:
        op: The code.

    Returns:
        The word.

    Raises:
        Error: If the code is not one of the folds.
    """
    if op < 0 or op > Int(AggKind.SKEW.code):
        raise Error(String("aggregate ", op, " is not one anybody can fold"))
    return String(AggKind(UInt8(op)))


def _agg_of(word: String) raises -> AggKind:
    """Reads an aggregate word back as its code.

    Written against `AggKind.write_to` rather than beside it, and
    `test_every_aggregate_is_named_and_reads_back` walks every code to keep the
    two honest, which is the same arrangement `named_type` has with
    `LogicalType.write_to`.

    Args:
        word: The word.

    Returns:
        The fold.

    Raises:
        Error: If nothing is called that.
    """
    for code in range(Int(AggKind.SKEW.code) + 1):
        var kind = AggKind(UInt8(code))
        if String(kind) == word:
            return kind
    raise Error(String("no aggregate is called ", word))


def _binary_of(word: String) raises -> BinaryOp:
    """Reads a binary operator back as its code.

    Args:
        word: The operator, as it is written.

    Returns:
        The operator.

    Raises:
        Error: If nothing is written that way.
    """
    for code in range(Int(BinaryOp.GE.code) + 1):
        var op = BinaryOp(UInt8(code))
        if String(op) == word:
            return op
    raise Error(String("no binary operator is written ", word))


def _unary_of(word: String) raises -> UnaryOp:
    """Reads a unary operator back as its code.

    Args:
        word: The operator, as it is written.

    Returns:
        The operator.

    Raises:
        Error: If nothing is written that way.
    """
    for code in range(UnaryOp.INVERT.code + 1):
        var op = UnaryOp(code)
        if String(op) == word:
            return op
    raise Error(String("no unary operator is written ", word))


def _join_of(word: String) raises -> JoinKind:
    """Reads a join kind back as its code.

    Args:
        word: The kind, as it is written.

    Returns:
        The kind.

    Raises:
        Error: If nothing is called that.
    """
    for code in range(Int(JoinKind.MARK.code) + 1):
        var kind = JoinKind(UInt8(code))
        if String(kind) == word:
            return kind
    raise Error(String("no join is called ", word))


struct _Naming(Movable):
    """How many times each thing is reached, and what the shared ones are called.

    One of these for the nodes and one for the expressions, since a `ref` is
    resolved against the list it was written in and the two lists are numbered
    apart.
    """

    var counts: List[Int]
    """How many times each arena slot is reached from the root."""

    var given: List[Int]
    """What each slot was called when it was written out, or minus one for a
    slot that has not been written yet."""

    var next: Int
    """The next name to hand out.

    Names are handed out in the order things are written rather than being the
    arena index they already have, and that is what makes the document
    canonical: two plans of the same shape write the same bytes however their
    arenas happened to get numbered, so a plan that went through here and came
    back writes what it wrote the first time.
    """

    def __init__(out self, size: Int):
        """Starts with nothing reached and nothing named.

        Args:
            size: How many slots the arena has.
        """
        self.counts = List[Int](length=size, fill=0)
        self.given = List[Int](length=size, fill=-1)
        self.next = 0


def _count_expr(tree: Expressions, at: Int, mut counts: List[Int]) raises:
    """Counts how many times each expression is reached from `at`.

    Stops at something already seen, because a second reach means the subtree
    under it is written once and reached through the reference after that, so
    counting it again would put ids on children that are not shared.

    Args:
        tree: The arena.
        at: Where to start.
        counts: One counter per arena slot, added to.
    """
    tree.check(at)
    counts[at] += 1
    if counts[at] > 1:
        return
    for i in range(len(tree.nodes[at].children)):
        _count_expr(tree, tree.nodes[at].children[i], counts)


def _count_node(
    plan: Plan, at: Int, mut counts: List[Int], mut on_exprs: List[Int]
) raises:
    """Counts how many times each node and each expression is reached.

    Args:
        plan: The plan.
        at: Where to start.
        counts: One counter per node, added to.
        on_exprs: One counter per expression, added to.
    """
    plan.check(at)
    counts[at] += 1
    if counts[at] > 1:
        return
    for i in range(len(plan.nodes[at].exprs)):
        _count_expr(plan.exprs, plan.nodes[at].exprs[i], on_exprs)
    for i in range(len(plan.nodes[at].inputs)):
        _count_node(plan, plan.nodes[at].inputs[i], counts, on_exprs)


def _expr_json(
    tree: Expressions, at: Int, mut naming: _Naming
) raises -> String:
    """Writes one expression out.

    Args:
        tree: The arena.
        at: The expression.
        naming: What is shared and what the shared parts are called.

    Returns:
        The JSON object.

    Raises:
        Error: If the expression holds something with no written form.
    """
    if naming.given[at] != -1:
        return String('{"ref": ', naming.given[at], "}")
    ref node = tree.nodes[at]
    var out = String("{")
    if naming.counts[at] > 1:
        naming.given[at] = naming.next
        naming.next += 1
        out += String('"id": ', naming.given[at], ", ")
    if node.kind == ExprKind.COLUMN:
        out += String('"kind": "column", "name": ', _quoted(node.name))
        if node.at != UNBOUND:
            out += String(', "at": ', node.at)
        if node.table != UNBOUND:
            out += String(', "table": ', node.table)
    elif node.kind == ExprKind.LITERAL:
        out += String(
            '"kind": "literal", "type": ',
            _quoted(_type_text(node.value.type)),
            ', "value": ',
            _literal_text(node.value),
        )
        if node.value.weak:
            out += ', "weak": true'
    elif node.kind == ExprKind.UNARY:
        out += String(
            '"kind": "unary", "op": ',
            _quoted(String(UnaryOp(node.op))),
            ', "over": ',
            _expr_json(tree, node.children[0], naming),
        )
    elif node.kind == ExprKind.BINARY:
        out += String(
            '"kind": "binary", "op": ',
            _quoted(String(BinaryOp(UInt8(node.op)))),
            ', "left": ',
            _expr_json(tree, node.children[0], naming),
            ', "right": ',
            _expr_json(tree, node.children[1], naming),
        )
    elif node.kind == ExprKind.CAST:
        out += String(
            '"kind": "cast", "to": ',
            _quoted(_type_text(node.type)),
            ', "over": ',
            _expr_json(tree, node.children[0], naming),
        )
    elif node.kind == ExprKind.CALL:
        out += String(
            '"kind": "call", "name": ',
            _quoted(node.name),
            ', "rowwise": ',
            "true" if node.rowwise else "false",
            ', "args": ',
            _exprs_json(tree, node.children, 0, len(node.children), naming),
        )
    elif node.kind == ExprKind.AGGREGATE:
        out += String(
            '"kind": "aggregate", "op": ',
            _quoted(_agg_text(node.op)),
            ', "over": ',
            _expr_json(tree, node.children[0], naming),
        )
    elif node.kind == ExprKind.CONDITIONAL:
        out += String(
            '"kind": "conditional", "when": ',
            _expr_json(tree, node.children[0], naming),
            ', "then": ',
            _expr_json(tree, node.children[1], naming),
            ', "otherwise": ',
            _expr_json(tree, node.children[2], naming),
        )
    else:
        out += String(
            '"kind": "window", "op": ',
            _quoted(_agg_text(node.op)),
            ', "over": ',
            _expr_json(tree, node.children[0], naming),
            ', "partition": ',
            _exprs_json(tree, node.children, 1, 1 + node.parts, naming),
            ', "order": ',
            _exprs_json(
                tree, node.children, 1 + node.parts, len(node.children), naming
            ),
        )
    # The type is written on everything that has one and left off where it is
    # still null, which is what an unbound expression holds and what a bound one
    # over a null holds too, so leaving it off loses nothing either way. A
    # literal and a cast say their type in a field of their own and do not want
    # it twice.
    if (
        node.type != LogicalType.NULL
        and node.kind != ExprKind.LITERAL
        and node.kind != ExprKind.CAST
    ):
        out += String(', "type": ', _quoted(_type_text(node.type)))
    return out + "}"


def _exprs_json(
    tree: Expressions,
    of: List[Int],
    start: Int,
    end: Int,
    mut naming: _Naming,
) raises -> String:
    """Writes a run of expressions out as an array.

    Args:
        tree: The arena.
        of: The indices.
        start: Where the run begins.
        end: One past where it ends.
        naming: What is shared and what the shared parts are called.

    Returns:
        The JSON array.

    Raises:
        Error: If one of them holds something with no written form.
    """
    var out = String("[")
    for i in range(start, end):
        if i != start:
            out += ", "
        out += _expr_json(tree, of[i], naming)
    return out + "]"


def _named_json(
    tree: Expressions,
    of: List[Int],
    names: List[String],
    start: Int,
    end: Int,
    mut naming: _Naming,
) raises -> String:
    """Writes a run of expressions out as an array of name and expression pairs.

    Args:
        tree: The arena.
        of: The indices.
        names: The names, indexed the same way.
        start: Where the run begins.
        end: One past where it ends.
        naming: What is shared and what the shared parts are called.

    Returns:
        The JSON array.

    Raises:
        Error: If one of them holds something with no written form.
    """
    var out = String("[")
    for i in range(start, end):
        if i != start:
            out += ", "
        out += String(
            '{"name": ',
            _quoted(names[i]),
            ', "expr": ',
            _expr_json(tree, of[i], naming),
            "}",
        )
    return out + "]"


def _strings_json(of: List[String]) -> String:
    """Writes a list of names out as an array of strings.

    Args:
        of: The names.

    Returns:
        The JSON array.
    """
    var out = String("[")
    for i in range(len(of)):
        if i != 0:
            out += ", "
        out += _quoted(of[i])
    return out + "]"


def _node_json(
    plan: Plan, at: Int, mut naming: _Naming, mut on_exprs: _Naming
) raises -> String:
    """Writes one node and everything under it out.

    Args:
        plan: The plan.
        at: The node.
        naming: What nodes are shared and what the shared ones are called.
        on_exprs: The same, for the expressions.

    Returns:
        The JSON object.

    Raises:
        Error: If the node holds something with no written form.
    """
    if naming.given[at] != -1:
        return String('{"ref": ', naming.given[at], "}")
    ref node = plan.nodes[at]
    var out = String("{")
    if naming.counts[at] > 1:
        naming.given[at] = naming.next
        naming.next += 1
        out += String('"id": ', naming.given[at], ", ")
    if node.kind == NodeKind.SCAN:
        return out + String(
            '"kind": "scan", "source": ',
            _quoted(node.source),
            ', "table": ',
            node.table,
            ', "columns": ',
            _strings_json(node.names),
            "}",
        )
    if node.kind == NodeKind.VALUES:
        out += String(
            '"kind": "values", "columns": ',
            _strings_json(node.names),
            ', "rows": [',
        )
        for i in range(0, len(node.exprs), node.parts):
            if i != 0:
                out += ", "
            out += _exprs_json(
                plan.exprs, node.exprs, i, i + node.parts, on_exprs
            )
        return out + "]}"
    if node.kind == NodeKind.TABLE_FUNCTION:
        return out + String(
            '"kind": "table_function", "function": ',
            _quoted(node.source),
            ', "columns": ',
            _strings_json(node.names),
            ', "args": ',
            _exprs_json(plan.exprs, node.exprs, 0, len(node.exprs), on_exprs),
            "}",
        )
    if node.kind == NodeKind.JOIN:
        var how = JoinKind(UInt8(node.op))
        out += String('"kind": "join", "how": ', _quoted(String(how)))
        if how == JoinKind.MARK:
            out += String(', "mark": ', _quoted(node.names[0]))
        out += ', "on": ['
        for i in range(node.parts):
            if i != 0:
                out += ", "
            out += String(
                '{"left": ',
                _expr_json(plan.exprs, node.exprs[i], on_exprs),
                ', "right": ',
                _expr_json(plan.exprs, node.exprs[node.parts + i], on_exprs),
                "}",
            )
        out += String(
            '], "left": ',
            _node_json(plan, node.inputs[0], naming, on_exprs),
            ', "right": ',
            _node_json(plan, node.inputs[1], naming, on_exprs),
        )
        return out + "}"
    if node.kind == NodeKind.UNION:
        var word = "union"
        if node.op == SET_EXCEPT:
            word = "except"
        elif node.op == SET_INTERSECT:
            word = "intersect"
        out += String(
            '"kind": "',
            word,
            '", "all": ',
            "true" if node.flags[0] else "false",
            ', "inputs": [',
        )
        for i in range(len(node.inputs)):
            if i != 0:
                out += ", "
            out += _node_json(plan, node.inputs[i], naming, on_exprs)
        return out + "]}"
    # Everything left has exactly one input, so the tail is the same on all of
    # them and only the middle differs.
    if node.kind == NodeKind.FILTER:
        out += String(
            '"kind": "filter", "predicate": ',
            _expr_json(plan.exprs, node.exprs[0], on_exprs),
        )
    elif node.kind == NodeKind.PROJECT:
        out += String(
            '"kind": "project", "columns": ',
            _named_json(
                plan.exprs, node.exprs, node.names, 0, len(node.exprs), on_exprs
            ),
        )
    elif node.kind == NodeKind.WINDOW:
        out += String(
            '"kind": "window", "columns": ',
            _named_json(
                plan.exprs, node.exprs, node.names, 0, len(node.exprs), on_exprs
            ),
        )
    elif node.kind == NodeKind.AGGREGATE:
        out += String(
            '"kind": "aggregate", "keys": ',
            _named_json(
                plan.exprs, node.exprs, node.names, 0, node.parts, on_exprs
            ),
            ', "aggregates": ',
            _named_json(
                plan.exprs,
                node.exprs,
                node.names,
                node.parts,
                len(node.exprs),
                on_exprs,
            ),
        )
    elif node.kind == NodeKind.SORT:
        out += '"kind": "sort", "keys": ['
        for i in range(len(node.exprs)):
            if i != 0:
                out += ", "
            out += String(
                '{"expr": ',
                _expr_json(plan.exprs, node.exprs[i], on_exprs),
                ', "descending": ',
                "true" if node.flags[i] else "false",
                ', "nulls_last": ',
                "true" if node.flags[len(node.exprs) + i] else "false",
                "}",
            )
        out += "]"
        # The bound a limit above put on the sort, which is a top n written
        # down. It is absent on a sort nobody has bounded, and the reader puts
        # it back the same way rather than treating it as a limit node.
        if node.length != NO_LIMIT:
            out += String(', "length": ', node.length)
    elif node.kind == NodeKind.LIMIT:
        out += String('"kind": "limit", "offset": ', node.offset)
        if node.length != NO_LIMIT:
            out += String(', "length": ', node.length)
    else:
        out += String(
            '"kind": "distinct", "keys": ',
            _exprs_json(plan.exprs, node.exprs, 0, len(node.exprs), on_exprs),
        )
    out += String(
        ', "input": ',
        _node_json(plan, node.inputs[0], naming, on_exprs),
    )
    return out + "}"


def to_json(plan: Plan, root: Int) raises -> String:
    """Writes the plan under a node out as JSON.

    Only what the root reaches is written, the same way `explain` prints only
    what the root reaches, so a plan arena holding the leftovers of a pass
    writes out as the plan and not as the leftovers.

    Args:
        plan: The plan.
        root: The node to write.

    Returns:
        The document, on one line.

    Raises:
        Error: If the root is not in the plan, or something under it has no
            written form.
    """
    plan.check(root)
    var naming = _Naming(len(plan.nodes))
    var on_exprs = _Naming(len(plan.exprs))
    _count_node(plan, root, naming.counts, on_exprs.counts)
    return _node_json(plan, root, naming, on_exprs)


def _at(
    bytes: Span[UInt8, _], members: List[Member], key: StringSlice
) raises -> Int:
    """Finds a member by name, or answers that it is not there.

    Args:
        bytes: The document.
        members: The object's members.
        key: The name.

    Returns:
        The member's position, or minus one.

    Raises:
        Error: If a key in the object is not readable text.
    """
    for i in range(len(members)):
        if text_of(bytes, members[i].key) == key:
            return i
    return -1


def _need(
    bytes: Span[UInt8, _],
    members: List[Member],
    key: StringSlice,
    what: StringSlice,
) raises -> Int:
    """Finds a member by name and refuses an object that does not have it.

    Args:
        bytes: The document.
        members: The object's members.
        key: The name.
        what: What the object is, for the message.

    Returns:
        The member's position.

    Raises:
        Error: If the object has no member of that name.
    """
    var found = _at(bytes, members, key)
    if found == -1:
        raise Error(String(what, " has no ", key))
    return found


def _flag(
    bytes: Span[UInt8, _], written: JsonValue, what: StringSlice
) raises -> Bool:
    """Reads a boolean.

    Args:
        bytes: The document.
        written: Where the value is.
        what: What it is, for the message.

    Returns:
        What it says.

    Raises:
        Error: If it is not a boolean.
    """
    if written.kind == JSON_TRUE:
        return True
    if written.kind == JSON_FALSE:
        return False
    raise Error(
        String(what, " is written true or false, not ", text_of(bytes, written))
    )


def _whole(
    bytes: Span[UInt8, _], written: JsonValue, what: StringSlice
) raises -> Int:
    """Reads a whole number.

    Args:
        bytes: The document.
        written: Where the value is.
        what: What it is, for the message.

    Returns:
        What it says.

    Raises:
        Error: If it is not a whole number.
    """
    if written.kind != JSON_NUMBER:
        raise Error(String(what, " is written as a whole number"))
    var got = parse_int[DType.int64](bytes[written.start : written.end])
    if not got.ok:
        raise Error(
            String(
                what,
                " is written as a whole number, not ",
                text_of(bytes, written),
            )
        )
    return Int(got.value)


def _names_of(
    bytes: Span[UInt8, _], written: JsonValue, what: StringSlice
) raises -> List[String]:
    """Reads an array of strings.

    Args:
        bytes: The document.
        written: Where the array is.
        what: What it is, for the message.

    Returns:
        The names.

    Raises:
        Error: If it is not an array of strings.
    """
    if written.kind != JSON_ARRAY:
        raise Error(String(what, " is written as an array"))
    var found = List[JsonValue]()
    _ = scan_array(bytes, written.start, found)
    var out = List[String]()
    for i in range(len(found)):
        if found[i].kind != JSON_STRING:
            raise Error(String(what, " is an array of names"))
        out.append(text_of(bytes, found[i]))
    return out^


def _elements(
    bytes: Span[UInt8, _], written: JsonValue, what: StringSlice
) raises -> List[JsonValue]:
    """Reads an array as its elements.

    Args:
        bytes: The document.
        written: Where the array is.
        what: What it is, for the message.

    Returns:
        The elements.

    Raises:
        Error: If it is not an array.
    """
    if written.kind != JSON_ARRAY:
        raise Error(String(what, " is written as an array"))
    var found = List[JsonValue]()
    _ = scan_array(bytes, written.start, found)
    return found^


def _expr_of(
    bytes: Span[UInt8, _],
    written: JsonValue,
    mut tree: Expressions,
    mut ids: Dict[Int, Int],
) raises -> Int:
    """Reads one expression back into the arena.

    Args:
        bytes: The document.
        written: Where the object is.
        tree: The arena being built.
        ids: Where each written id ended up, for the references to it.

    Returns:
        The index of the expression.

    Raises:
        Error: If the object does not describe an expression.
    """
    if written.kind != JSON_OBJECT:
        raise Error("an expression is written as an object")
    var members = List[Member]()
    _ = scan_object(bytes, written.start, members)
    var reference = _at(bytes, members, "ref")
    if reference != -1:
        var id = _whole(bytes, members[reference].value, "a ref")
        if id not in ids:
            raise Error(
                String(
                    "a ref names expression ",
                    id,
                    ", and nothing was written with that id before it",
                )
            )
        return ids[id]
    var kind = text_of(
        bytes, members[_need(bytes, members, "kind", "an expression")].value
    )
    var at: Int
    if kind == "column":
        at = tree.column(
            text_of(
                bytes, members[_need(bytes, members, "name", "a column")].value
            )
        )
        var position = _at(bytes, members, "at")
        if position != -1:
            tree.nodes[at].at = _whole(
                bytes, members[position].value, "a position"
            )
        var table = _at(bytes, members, "table")
        if table != -1:
            tree.nodes[at].table = _whole(
                bytes, members[table].value, "a table"
            )
    elif kind == "literal":
        var type = _type_of(
            text_of(
                bytes, members[_need(bytes, members, "type", "a literal")].value
            )
        )
        var value = _value_of(
            type,
            bytes,
            members[_need(bytes, members, "value", "a literal")].value,
        )
        var weak = _at(bytes, members, "weak")
        if weak != -1 and _flag(bytes, members[weak].value, "weak"):
            value = value.weakened()
        at = tree.literal(value^)
    elif kind == "unary":
        at = tree.unary(
            _unary_of(
                text_of(
                    bytes, members[_need(bytes, members, "op", "a unary")].value
                )
            ),
            _expr_of(
                bytes,
                members[_need(bytes, members, "over", "a unary")].value,
                tree,
                ids,
            ),
        )
    elif kind == "binary":
        var op = _binary_of(
            text_of(
                bytes, members[_need(bytes, members, "op", "a binary")].value
            )
        )
        var left = _expr_of(
            bytes,
            members[_need(bytes, members, "left", "a binary")].value,
            tree,
            ids,
        )
        var right = _expr_of(
            bytes,
            members[_need(bytes, members, "right", "a binary")].value,
            tree,
            ids,
        )
        at = tree.binary(op, left, right)
    elif kind == "cast":
        var to = _type_of(
            text_of(bytes, members[_need(bytes, members, "to", "a cast")].value)
        )
        at = tree.cast(
            to,
            _expr_of(
                bytes,
                members[_need(bytes, members, "over", "a cast")].value,
                tree,
                ids,
            ),
        )
    elif kind == "call":
        var name = text_of(
            bytes, members[_need(bytes, members, "name", "a call")].value
        )
        var rowwise = _flag(
            bytes,
            members[_need(bytes, members, "rowwise", "a call")].value,
            "whether a call is rowwise",
        )
        var args = _expr_list(
            bytes,
            members[_need(bytes, members, "args", "a call")].value,
            "the arguments of a call",
            tree,
            ids,
        )
        at = tree.call(name^, args^, rowwise)
    elif kind == "aggregate":
        at = tree.aggregate(
            _agg_of(
                text_of(
                    bytes,
                    members[_need(bytes, members, "op", "an aggregate")].value,
                )
            ),
            _expr_of(
                bytes,
                members[_need(bytes, members, "over", "an aggregate")].value,
                tree,
                ids,
            ),
        )
    elif kind == "conditional":
        var when = _expr_of(
            bytes,
            members[_need(bytes, members, "when", "a conditional")].value,
            tree,
            ids,
        )
        var then = _expr_of(
            bytes,
            members[_need(bytes, members, "then", "a conditional")].value,
            tree,
            ids,
        )
        var otherwise = _expr_of(
            bytes,
            members[_need(bytes, members, "otherwise", "a conditional")].value,
            tree,
            ids,
        )
        at = tree.conditional(when, then, otherwise)
    elif kind == "window":
        var op = _agg_of(
            text_of(
                bytes, members[_need(bytes, members, "op", "a window")].value
            )
        )
        var over = _expr_of(
            bytes,
            members[_need(bytes, members, "over", "a window")].value,
            tree,
            ids,
        )
        var partition = _expr_list(
            bytes,
            members[_need(bytes, members, "partition", "a window")].value,
            "the partition keys of a window",
            tree,
            ids,
        )
        var order = _expr_list(
            bytes,
            members[_need(bytes, members, "order", "a window")].value,
            "the order keys of a window",
            tree,
            ids,
        )
        at = tree.window(op, over, partition^, order^)
    else:
        raise Error(String("no expression is a ", kind))
    var type = _at(bytes, members, "type")
    if type != -1:
        tree.nodes[at].type = _type_of(text_of(bytes, members[type].value))
    var id = _at(bytes, members, "id")
    if id != -1:
        ids[_whole(bytes, members[id].value, "an id")] = at
    return at


def _expr_list(
    bytes: Span[UInt8, _],
    written: JsonValue,
    what: StringSlice,
    mut tree: Expressions,
    mut ids: Dict[Int, Int],
) raises -> List[Int]:
    """Reads an array of expressions back into the arena.

    Args:
        bytes: The document.
        written: Where the array is.
        what: What it is, for the message.
        tree: The arena being built.
        ids: Where each written id ended up.

    Returns:
        The indices, in order.

    Raises:
        Error: If it is not an array of expressions.
    """
    var found = _elements(bytes, written, what)
    var out = List[Int]()
    for i in range(len(found)):
        out.append(_expr_of(bytes, found[i], tree, ids))
    return out^


def _pairs_of(
    bytes: Span[UInt8, _],
    written: JsonValue,
    what: StringSlice,
    mut tree: Expressions,
    mut ids: Dict[Int, Int],
    mut names: List[String],
) raises -> List[Int]:
    """Reads an array of name and expression pairs.

    Args:
        bytes: The document.
        written: Where the array is.
        what: What it is, for the message.
        tree: The arena being built.
        ids: Where each written id ended up.
        names: Where the names go, appended to so that two runs of pairs can
            fill one name list in order.

    Returns:
        The expression indices, in order.

    Raises:
        Error: If it is not an array of pairs.
    """
    var found = _elements(bytes, written, what)
    var out = List[Int]()
    for i in range(len(found)):
        if found[i].kind != JSON_OBJECT:
            raise Error(
                String(what, " is an array of name and expression pairs")
            )
        var members = List[Member]()
        _ = scan_object(bytes, found[i].start, members)
        names.append(
            text_of(bytes, members[_need(bytes, members, "name", what)].value)
        )
        out.append(
            _expr_of(
                bytes,
                members[_need(bytes, members, "expr", what)].value,
                tree,
                ids,
            )
        )
    return out^


def _node_of(
    bytes: Span[UInt8, _],
    written: JsonValue,
    mut plan: Plan,
    mut ids: Dict[Int, Int],
    mut expr_ids: Dict[Int, Int],
) raises -> Int:
    """Reads one node and everything under it back into the plan.

    Args:
        bytes: The document.
        written: Where the object is.
        plan: The plan being built.
        ids: Where each written node id ended up.
        expr_ids: Where each written expression id ended up.

    Returns:
        The index of the node.

    Raises:
        Error: If the object does not describe a node, or describes one the
            builders refuse.
    """
    if written.kind != JSON_OBJECT:
        raise Error("a plan node is written as an object")
    var members = List[Member]()
    _ = scan_object(bytes, written.start, members)
    var reference = _at(bytes, members, "ref")
    if reference != -1:
        var id = _whole(bytes, members[reference].value, "a ref")
        if id not in ids:
            raise Error(
                String(
                    "a ref names node ",
                    id,
                    ", and nothing was written with that id before it",
                )
            )
        return ids[id]
    var kind = text_of(
        bytes, members[_need(bytes, members, "kind", "a plan node")].value
    )
    var at: Int
    if kind == "scan":
        at = plan.scan(
            text_of(
                bytes, members[_need(bytes, members, "source", "a scan")].value
            ),
            _names_of(
                bytes,
                members[_need(bytes, members, "columns", "a scan")].value,
                "the columns of a scan",
            ),
            _whole(
                bytes,
                members[_need(bytes, members, "table", "a scan")].value,
                "the relation a scan is",
            ),
        )
    elif kind == "values":
        at = _values_of(bytes, members, plan, expr_ids)
    elif kind == "table_function":
        at = plan.table_function(
            text_of(
                bytes,
                members[
                    _need(bytes, members, "function", "a table function")
                ].value,
            ),
            _expr_list(
                bytes,
                members[
                    _need(bytes, members, "args", "a table function")
                ].value,
                "the arguments of a table function",
                plan.exprs,
                expr_ids,
            ),
            _names_of(
                bytes,
                members[
                    _need(bytes, members, "columns", "a table function")
                ].value,
                "the columns of a table function",
            ),
        )
    elif kind == "join":
        at = _join_node_of(bytes, members, plan, ids, expr_ids)
    elif kind == "union" or kind == "except" or kind == "intersect":
        var op = SET_UNION
        if kind == "except":
            op = SET_EXCEPT
        elif kind == "intersect":
            op = SET_INTERSECT
        var all = _flag(
            bytes,
            members[_need(bytes, members, "all", "a set operation")].value,
            "whether duplicates survive a set operation",
        )
        var arms = _elements(
            bytes,
            members[_need(bytes, members, "inputs", "a set operation")].value,
            "the arms of a set operation",
        )
        var inputs = List[Int]()
        for i in range(len(arms)):
            inputs.append(_node_of(bytes, arms[i], plan, ids, expr_ids))
        at = plan.setop(inputs^, op, all)
    else:
        at = _over_one_of(bytes, members, kind, plan, ids, expr_ids)
    var id = _at(bytes, members, "id")
    if id != -1:
        ids[_whole(bytes, members[id].value, "an id")] = at
    return at


def _values_of(
    bytes: Span[UInt8, _],
    members: List[Member],
    mut plan: Plan,
    mut expr_ids: Dict[Int, Int],
) raises -> Int:
    """Reads a values node back.

    Args:
        bytes: The document.
        members: The object's members.
        plan: The plan being built.
        expr_ids: Where each written expression id ended up.

    Returns:
        The index of the node.

    Raises:
        Error: If the object does not describe a values.
    """
    var names = _names_of(
        bytes,
        members[_need(bytes, members, "columns", "a values")].value,
        "the columns of a values",
    )
    var written_rows = _elements(
        bytes,
        members[_need(bytes, members, "rows", "a values")].value,
        "the rows of a values",
    )
    var rows = List[Int]()
    for i in range(len(written_rows)):
        var row = _expr_list(
            bytes, written_rows[i], "a row of a values", plan.exprs, expr_ids
        )
        for j in range(len(row)):
            rows.append(row[j])
    return plan.values(rows^, names^)


def _join_node_of(
    bytes: Span[UInt8, _],
    members: List[Member],
    mut plan: Plan,
    mut ids: Dict[Int, Int],
    mut expr_ids: Dict[Int, Int],
) raises -> Int:
    """Reads a join back.

    The keys are read before the inputs, which is the one place the order of the
    document and the order of the builder call disagree. It does not matter:
    both arenas are append only and a key holds no node index.

    Args:
        bytes: The document.
        members: The object's members.
        plan: The plan being built.
        ids: Where each written node id ended up.
        expr_ids: Where each written expression id ended up.

    Returns:
        The index of the node.

    Raises:
        Error: If the object does not describe a join.
    """
    var how = _join_of(
        text_of(bytes, members[_need(bytes, members, "how", "a join")].value)
    )
    var pairs = _elements(
        bytes,
        members[_need(bytes, members, "on", "a join")].value,
        "the keys of a join",
    )
    var left_keys = List[Int]()
    var right_keys = List[Int]()
    for i in range(len(pairs)):
        if pairs[i].kind != JSON_OBJECT:
            raise Error("a join key is a left and a right")
        var pair = List[Member]()
        _ = scan_object(bytes, pairs[i].start, pair)
        left_keys.append(
            _expr_of(
                bytes,
                pair[_need(bytes, pair, "left", "a join key")].value,
                plan.exprs,
                expr_ids,
            )
        )
        right_keys.append(
            _expr_of(
                bytes,
                pair[_need(bytes, pair, "right", "a join key")].value,
                plan.exprs,
                expr_ids,
            )
        )
    var left = _node_of(
        bytes,
        members[_need(bytes, members, "left", "a join")].value,
        plan,
        ids,
        expr_ids,
    )
    var right = _node_of(
        bytes,
        members[_need(bytes, members, "right", "a join")].value,
        plan,
        ids,
        expr_ids,
    )
    var mark = String()
    if how == JoinKind.MARK:
        mark = text_of(
            bytes, members[_need(bytes, members, "mark", "a mark join")].value
        )
    return plan.join(left, right, left_keys^, right_keys^, how, mark^)


def _over_one_of(
    bytes: Span[UInt8, _],
    members: List[Member],
    kind: String,
    mut plan: Plan,
    mut ids: Dict[Int, Int],
    mut expr_ids: Dict[Int, Int],
) raises -> Int:
    """Reads back one of the seven kinds that sit over a single input.

    Args:
        bytes: The document.
        members: The object's members.
        kind: Which of the seven it says it is.
        plan: The plan being built.
        ids: Where each written node id ended up.
        expr_ids: Where each written expression id ended up.

    Returns:
        The index of the node.

    Raises:
        Error: If it is not one of the seven, or is not a well formed one.
    """
    # The node's own expressions are read before its input, which is the order
    # they were written in, and the order matters as soon as one of them carries
    # an id: a reference to it further down the document has to find it already
    # read. The builder call comes last because it wants both.
    var over = _need(bytes, members, "input", String("a ", kind))
    if kind == "filter":
        var predicate = _expr_of(
            bytes,
            members[_need(bytes, members, "predicate", "a filter")].value,
            plan.exprs,
            expr_ids,
        )
        var input = _node_of(bytes, members[over].value, plan, ids, expr_ids)
        return plan.filter(input, predicate)
    if kind == "project":
        var names = List[String]()
        var outputs = _pairs_of(
            bytes,
            members[_need(bytes, members, "columns", "a project")].value,
            "the columns of a project",
            plan.exprs,
            expr_ids,
            names,
        )
        var input = _node_of(bytes, members[over].value, plan, ids, expr_ids)
        return plan.project(input, outputs^, names^)
    if kind == "window":
        var names = List[String]()
        var outputs = _pairs_of(
            bytes,
            members[_need(bytes, members, "columns", "a window")].value,
            "the columns of a window",
            plan.exprs,
            expr_ids,
            names,
        )
        var input = _node_of(bytes, members[over].value, plan, ids, expr_ids)
        return plan.window(input, outputs^, names^)
    if kind == "aggregate":
        var names = List[String]()
        var keys = _pairs_of(
            bytes,
            members[_need(bytes, members, "keys", "an aggregate")].value,
            "the keys of an aggregate",
            plan.exprs,
            expr_ids,
            names,
        )
        var aggs = _pairs_of(
            bytes,
            members[_need(bytes, members, "aggregates", "an aggregate")].value,
            "the folds of an aggregate",
            plan.exprs,
            expr_ids,
            names,
        )
        var input = _node_of(bytes, members[over].value, plan, ids, expr_ids)
        return plan.aggregate(input, keys^, aggs^, names^)
    if kind == "sort":
        return _sort_of(bytes, members, over, plan, ids, expr_ids)
    if kind == "limit":
        var offset = _whole(
            bytes,
            members[_need(bytes, members, "offset", "a limit")].value,
            "the rows a limit skips",
        )
        var length = _at(bytes, members, "length")
        var keeps = NO_LIMIT if length == -1 else _whole(
            bytes, members[length].value, "the rows a limit keeps"
        )
        var input = _node_of(bytes, members[over].value, plan, ids, expr_ids)
        return plan.limit(input, offset, keeps)
    if kind == "distinct":
        var written_keys = _at(bytes, members, "keys")
        var keys = List[Int]() if written_keys == -1 else _expr_list(
            bytes,
            members[written_keys].value,
            "the keys of a distinct",
            plan.exprs,
            expr_ids,
        )
        var input = _node_of(bytes, members[over].value, plan, ids, expr_ids)
        return plan.distinct(input, keys^)
    raise Error(String("no plan node is a ", kind))


def _sort_of(
    bytes: Span[UInt8, _],
    members: List[Member],
    over: Int,
    mut plan: Plan,
    mut ids: Dict[Int, Int],
    mut expr_ids: Dict[Int, Int],
) raises -> Int:
    """Reads a sort back, including the bound a limit above it put on it.

    Args:
        bytes: The document.
        members: The object's members.
        over: Which member holds the node sorted.
        plan: The plan being built.
        ids: Where each written node id ended up.
        expr_ids: Where each written expression id ended up.

    Returns:
        The index of the node.

    Raises:
        Error: If the object does not describe a sort.
    """
    var written_keys = _elements(
        bytes,
        members[_need(bytes, members, "keys", "a sort")].value,
        "the keys of a sort",
    )
    var keys = List[Int]()
    var descending = List[Bool]()
    var nulls_last = List[Bool]()
    for i in range(len(written_keys)):
        if written_keys[i].kind != JSON_OBJECT:
            raise Error("a sort key is an expression and two directions")
        var key = List[Member]()
        _ = scan_object(bytes, written_keys[i].start, key)
        keys.append(
            _expr_of(
                bytes,
                key[_need(bytes, key, "expr", "a sort key")].value,
                plan.exprs,
                expr_ids,
            )
        )
        var down = _at(bytes, key, "descending")
        descending.append(
            False if down
            == -1 else _flag(
                bytes, key[down].value, "whether a sort key goes downwards"
            )
        )
        var last = _at(bytes, key, "nulls_last")
        nulls_last.append(
            False if last
            == -1 else _flag(
                bytes, key[last].value, "where the nulls of a sort key go"
            )
        )
    var input = _node_of(bytes, members[over].value, plan, ids, expr_ids)
    var at = plan.sort(input, keys^, descending^, nulls_last^)
    var length = _at(bytes, members, "length")
    if length != -1:
        plan.nodes[at].length = _whole(
            bytes, members[length].value, "the rows a sort has to get right"
        )
    return at


def from_json(text: StringSlice) raises -> Loaded:
    """Reads a document back as the plan it describes.

    Args:
        text: The document.

    Returns:
        The plan and its root.

    Raises:
        Error: If it is not JSON, or is not a plan, or is a plan the builders
            refuse.
    """
    var bytes = text.as_bytes()
    var start = skip_space(bytes, 0)
    var written = scan_value(bytes, start)
    var plan = Plan()
    var ids = Dict[Int, Int]()
    var expr_ids = Dict[Int, Int]()
    var root = _node_of(bytes, written, plan, ids, expr_ids)
    return Loaded(plan^, root)
