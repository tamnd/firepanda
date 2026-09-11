"""Which expressions are aggregates, which are windows, and where each may go.

A `SELECT` is one shape or two depending on what is written in it, and nothing
in the grammar says which. `SELECT a FROM t` reads a column per row. `SELECT
a, sum(a) FROM t` is refused, and the only difference is a name that happens
to be an aggregate. So the shape of a query is decided here, by walking the
expressions and seeing what is in them. See docs/specs/sql/05-ast-and-binder.md
section 10.

A query is an aggregate query when an aggregate appears in the select list,
`HAVING`, `QUALIFY` or `ORDER BY`, or when it has a `GROUP BY`, or when it has
a `HAVING` at all. In an aggregate query every column reference has to be a
group key or sit inside an aggregate, and `SELECT a, sum(a) FROM t` breaks
that with no keys at all.

The walk stops at a subquery. An aggregate written inside one belongs to that
query rather than to this one, which is why `SELECT a FROM t WHERE EXISTS
(SELECT sum(b) FROM t u WHERE u.a = t.a)` binds.

An aggregate exempts its arguments from the group key rule and a window does
not. `sum(sum(a)) OVER () FROM t GROUP BY g` binds because the inner `sum`
covers the `a`, while `max(a) OVER (PARTITION BY sum(b)) FROM t GROUP BY g` is
refused over the `a`, which is a plain reference sitting in a window's
argument. That asymmetry is the whole reason the two are counted separately
rather than as one "not a plain expression" flag.

Three of the refusals here are worded oddly and are DuckDB's. An aggregate in
a `JOIN ... ON` is reported against the `WHERE` clause, a clause the query may
not have. A `QUALIFY` with no window function anywhere asks for one in the
select list or in itself rather than saying `QUALIFY` needs a window. And the
group key message comes in two spellings, a long one with an `ANY_VALUE` hint
that nearly every clause gets, and a short one without it that only `HAVING`
produces.

Which names are aggregates is measured rather than declared: 88 of them on
DuckDB 1.5, and 13 of those refuse to be called without `OVER`, so
`row_number()` on its own is not an aggregate with no window but a scalar
function that does not exist. The full registry with signatures and overloads
is separate work. What is here is the name set, which is what classification
needs and all it needs.
"""

from .ast import (
    Ast,
    EXPR_BETWEEN,
    EXPR_BINARY,
    EXPR_CASE,
    EXPR_CAST,
    EXPR_COLLATE,
    EXPR_COLUMN,
    EXPR_EXISTS,
    EXPR_FRAME,
    EXPR_FUNCTION,
    EXPR_IN,
    EXPR_IN_SUBQUERY,
    EXPR_LIST,
    EXPR_LITERAL,
    EXPR_PARAMETER,
    EXPR_STAR,
    EXPR_STRUCT,
    EXPR_SUBQUERY,
    EXPR_UNARY,
    EXPR_WINDOW,
)
from .catalog import fold


comptime CLAUSE_WHERE: UInt8 = 1
"""`WHERE`, and a `JOIN ... ON` too, which DuckDB reports as `WHERE`."""

comptime CLAUSE_GROUP: UInt8 = 2
"""`GROUP BY`."""

comptime CLAUSE_HAVING: UInt8 = 3
"""`HAVING`."""

comptime CLAUSE_QUALIFY: UInt8 = 4
"""`QUALIFY`."""

comptime CLAUSE_SELECT: UInt8 = 5
"""The select list."""

comptime CLAUSE_ORDER: UInt8 = 6
"""`ORDER BY`."""


@fieldwise_init
struct Uses(Copyable, ImplicitlyCopyable, Movable):
    """What an expression tree has in it."""

    var aggregate: Bool
    """Whether an aggregate call appears anywhere below."""

    var window: Bool
    """Whether a call with `OVER` on it appears anywhere below."""

    def __init__(out self):
        """Nothing in it."""
        self.aggregate = False
        self.window = False

    def plain(self) -> Bool:
        """Whether the tree is neither an aggregate nor a window.

        Returns:
            True if it holds neither.
        """
        return not self.aggregate and not self.window


def inspect(ast: Ast, node: UInt32) raises -> Uses:
    """What an expression tree holds, without descending into a subquery.

    Args:
        ast: The arenas.
        node: The expression.

    Returns:
        What is in it.

    Raises:
        If the tree is deeper than the arena is long, which would mean a
        cycle.
    """
    var out = Uses()
    var stack = List[UInt32]()
    stack.append(node)
    var steps = 0
    while len(stack) != 0:
        steps += 1
        if steps > len(ast.exprs) + 1:
            raise Error("an expression that refers back to itself")
        var at = stack.pop()
        if at == 0:
            continue
        ref item = ast.exprs[Int(at)]
        if item.kind == EXPR_FUNCTION:
            if item.b != 0:
                out.window = True
            elif is_aggregate(ast.text(ast.at(item.payload, 0))):
                out.aggregate = True
        for child in children(ast, at):
            stack.append(child)
    return out


def children(ast: Ast, node: UInt32) raises -> List[UInt32]:
    """The expressions directly under one, subqueries left out.

    A subquery holds a statement index rather than an expression index, and a
    walk that followed it would be reading the wrong arena. It is also the
    right answer: an aggregate inside a subquery belongs to that query.

    Args:
        ast: The arenas.
        node: The expression.

    Returns:
        The child expressions, in no particular order.

    Raises:
        If the node index is out of the arena.
    """
    var out = List[UInt32]()
    if node == 0 or Int(node) >= len(ast.exprs):
        raise Error("an expression index that is not in the arena")
    ref item = ast.exprs[Int(node)]
    var kind = item.kind

    if kind == EXPR_LITERAL or kind == EXPR_PARAMETER or kind == EXPR_COLUMN:
        return out^
    if kind == EXPR_SUBQUERY or kind == EXPR_EXISTS:
        return out^
    if kind == EXPR_UNARY or kind == EXPR_CAST or kind == EXPR_COLLATE:
        out.append(item.a)
        return out^
    if kind == EXPR_BINARY or kind == EXPR_FRAME:
        out.append(item.a)
        out.append(item.b)
        return out^
    if kind == EXPR_IN_SUBQUERY:
        out.append(item.a)
        return out^
    if kind == EXPR_FUNCTION:
        for child in ast.items(item.children):
            out.append(child)
        out.append(item.b)
        return out^
    if kind == EXPR_CASE:
        out.append(item.a)
        out.append(item.b)
        for child in ast.items(item.children):
            out.append(child)
        return out^
    if kind == EXPR_BETWEEN or kind == EXPR_IN:
        out.append(item.a)
        for child in ast.items(item.children):
            out.append(child)
        return out^
    if kind == EXPR_LIST:
        for child in ast.items(item.children):
            out.append(child)
        return out^
    if kind == EXPR_STRUCT:
        # Alternating name and value, so every second slot is a name and not
        # an expression at all.
        var count = ast.length(item.children)
        for at in range(1, count, 2):
            out.append(ast.at(item.children, at))
        return out^
    if kind == EXPR_STAR:
        # Only the REPLACE run holds expressions, and it alternates too. The
        # other three runs are names.
        var count = ast.length(item.b)
        for at in range(1, count, 2):
            out.append(ast.at(item.b, at))
        return out^
    if kind == EXPR_WINDOW:
        for child in ast.items(item.children):
            out.append(child)
        out.append(item.b)
        return out^
    raise Error("an expression kind with no children rule")


def check_nesting(ast: Ast, node: UInt32) raises:
    """Refuses an aggregate or a window written inside the wrong thing.

    An aggregate may not hold another aggregate and may not hold a window. A
    window specification may not hold a window either, which DuckDB catches
    while parsing. A window's argument may hold an aggregate, which is how
    `sum(sum(a)) OVER ()` is written.

    What is caught here is the `PARTITION BY` and the frame. A window's own
    `ORDER BY` is a run of `STMT_ORDER` nodes in the statement arena rather
    than expressions, so a window written there is caught by whatever walks
    statements, not by this.

    Args:
        ast: The arenas.
        node: The expression to check, along with everything below it.

    Raises:
        If one is nested inside the other the wrong way round.
    """
    var stack = List[UInt32]()
    stack.append(node)
    var steps = 0
    while len(stack) != 0:
        steps += 1
        if steps > len(ast.exprs) + 1:
            raise Error("an expression that refers back to itself")
        var at = stack.pop()
        if at == 0:
            continue
        ref item = ast.exprs[Int(at)]
        if item.kind == EXPR_FUNCTION and item.b == 0:
            if is_aggregate(ast.text(ast.at(item.payload, 0))):
                for argument in ast.items(item.children):
                    var inner = inspect(ast, argument)
                    if inner.aggregate:
                        raise Error(nested_aggregate())
                    if inner.window:
                        raise Error(window_in_aggregate())
        if item.kind == EXPR_FUNCTION and item.b != 0:
            if inspect(ast, item.b).window:
                raise Error(window_in_window())
        for child in children(ast, at):
            stack.append(child)


def check_clause(ast: Ast, node: UInt32, clause: UInt8) raises:
    """Refuses an aggregate or a window in a clause that cannot hold one.

    Args:
        ast: The arenas.
        node: The expression written in the clause.
        clause: One of the `CLAUSE_` constants.

    Raises:
        If the clause refuses what the expression holds.
    """
    var uses = inspect(ast, node)
    if clause == CLAUSE_WHERE or clause == CLAUSE_GROUP:
        if uses.aggregate:
            raise Error(no_aggregates_here(clause))
        if uses.window:
            raise Error(no_windows_here(clause))
        return
    if clause == CLAUSE_HAVING and uses.window:
        raise Error(no_windows_here(clause))


def covered(
    ast: Ast, node: UInt32, keys: List[UInt32], aggregated: Bool
) raises -> UInt32:
    """The first column reference a query with grouping does not account for.

    A reference is accounted for when it is part of an expression equal to a
    group key, or when it sits inside an aggregate. An expression built only
    out of covered parts is covered, which is why `SELECT a + 1 FROM t GROUP
    BY a` binds without `a + 1` being a key.

    Args:
        ast: The arenas.
        node: The expression.
        keys: The group key expressions.
        aggregated: Whether the query aggregates at all. A query that does
            not may say anything it likes.

    Returns:
        The offending `EXPR_COLUMN` node, or 0 when every reference is
        accounted for.

    Raises:
        If the tree is deeper than the arena is long.
    """
    if not aggregated:
        return 0
    var stack = List[UInt32]()
    stack.append(node)
    var steps = 0
    while len(stack) != 0:
        steps += 1
        if steps > len(ast.exprs) + 1:
            raise Error("an expression that refers back to itself")
        var at = stack.pop()
        if at == 0:
            continue
        var matched = False
        for key in keys:
            if same(ast, at, key):
                matched = True
                break
        if matched:
            continue
        ref item = ast.exprs[Int(at)]
        if item.kind == EXPR_COLUMN:
            return at
        # An aggregate covers whatever is in its arguments. A window does not,
        # so its arguments go back on the stack like anything else.
        if item.kind == EXPR_FUNCTION and item.b == 0:
            if is_aggregate(ast.text(ast.at(item.payload, 0))):
                continue
        for child in children(ast, at):
            stack.append(child)
    return 0


def same(ast: Ast, left: UInt32, right: UInt32) raises -> Bool:
    """Whether two expressions are written the same way.

    This is what decides whether a select list entry matches a group key, so
    it compares the tree rather than the printed text. Two spellings of one
    name are the same expression and two spellings of one value are not, which
    is what a name being folded and a literal being interned already say.

    Args:
        ast: The arenas.
        left: One expression.
        right: The other.

    Returns:
        True if the trees have the same shape and the same tags in them.

    Raises:
        If either tree is deeper than the arena is long.
    """
    var stack = List[UInt32]()
    stack.append(left)
    stack.append(right)
    var steps = 0
    while len(stack) != 0:
        steps += 1
        if steps > 2 * len(ast.exprs) + 2:
            raise Error("an expression that refers back to itself")
        var b = stack.pop()
        var a = stack.pop()
        if a == b:
            continue
        if a == 0 or b == 0:
            return False
        if ast.exprs[Int(a)].kind != ast.exprs[Int(b)].kind:
            return False
        if _tags(ast, a) != _tags(ast, b):
            return False
        var mine = children(ast, a)
        var theirs = children(ast, b)
        if len(mine) != len(theirs):
            return False
        for at in range(len(mine)):
            stack.append(mine[at])
            stack.append(theirs[at])
    return True


comptime AGGREGATES: StringSlice[ImmStaticOrigin] = (
    ",any_value,approx_count_distinct,approx_quantile,approx_top_k,arbitrary,"
    "arg_max,arg_max_null,arg_max_nulls_last,arg_min,arg_min_null,"
    "arg_min_nulls_last,argmax,argmin,array_agg,avg,bit_and,bit_or,bit_xor,"
    "bitstring_agg,bool_and,bool_or,corr,count,count_if,count_star,countif,"
    "covar_pop,covar_samp,cume_dist,dense_rank,entropy,favg,fill,first,"
    "first_value,fsum,group_concat,histogram,histogram_exact,kahan_sum,"
    "kurtosis,kurtosis_pop,lag,last,last_value,lead,list,listagg,mad,max,"
    "max_by,mean,median,min,min_by,mode,nth_value,ntile,percent_rank,product,"
    "quantile,quantile_cont,quantile_disc,rank,rank_dense,regr_avgx,"
    "regr_avgy,regr_count,regr_intercept,regr_r2,regr_slope,regr_sxx,"
    "regr_sxy,regr_syy,reservoir_quantile,row_number,sem,skewness,stddev,"
    "stddev_pop,stddev_samp,string_agg,sum,sum_no_overflow,sumkahan,var_pop,"
    "var_samp,variance,"
)
"""Every aggregate DuckDB 1.5 has, folded, sorted and comma delimited.

Taken from `duckdb_functions()` rather than written by hand. The 88 names carry
1,177 overloads between them, which is thirteen apiece, because each is
instantiated across the numeric types and most exist in ordered, distinct and
windowed forms as well.

One string rather than a list because a module level list cannot be indexed at
runtime, and a function that rebuilt one would allocate 88 strings every time a
query asked whether a name was an aggregate. The leading and trailing commas
are what make a search for `,sum,` unable to match inside `,kahan_sum,`.
"""

comptime WINDOW_ONLY: StringSlice[ImmStaticOrigin] = (
    ",cume_dist,dense_rank,fill,first_value,lag,last_value,lead,nth_value,"
    "ntile,percent_rank,rank,rank_dense,row_number,"
)
"""The thirteen of those that refuse to be called without `OVER`.

Found by calling each of the 88 with no window on it and keeping the ones that
answered that no scalar function of that name exists.
"""


def is_aggregate(name: StringSlice) -> Bool:
    """Whether a name is one of DuckDB's aggregates.

    Args:
        name: The function name, in any case.

    Returns:
        True if it is.
    """
    return String(",", fold(name), ",") in AGGREGATES


def needs_over(name: StringSlice) -> Bool:
    """Whether a name can only be written with `OVER` on it.

    `row_number()` with no `OVER` is not an aggregate that forgot its window.
    DuckDB says there is no scalar function of that name, because there is
    not, and that message belongs with the rest of overload resolution rather
    than here.

    Args:
        name: The function name, in any case.

    Returns:
        True if it is one of the thirteen.
    """
    return String(",", fold(name), ",") in WINDOW_ONLY


def aggregate_names() -> List[String]:
    """The aggregate names, one per entry.

    Returns:
        The 88 names, sorted.
    """
    return _split(AGGREGATES)


def window_only_names() -> List[String]:
    """The window only names, one per entry.

    Returns:
        The thirteen, sorted.
    """
    return _split(WINDOW_ONLY)


def _split(table: StringSlice) -> List[String]:
    """Cuts a comma delimited table back into names.

    Args:
        table: The table, with a comma at each end.

    Returns:
        The names, in the order they are written.
    """
    var out = List[String]()
    for part in table.split(","):
        # The comma at each end gives an empty piece at each end, which is the
        # price of a search for `,sum,` not matching inside `,kahan_sum,`.
        if part.byte_length() != 0:
            out.append(String(part))
    return out^


def no_aggregates_here(clause: UInt8) -> String:
    """DuckDB's error for an aggregate in a clause that refuses one.

    Args:
        clause: One of the `CLAUSE_` constants.

    Returns:
        The message.
    """
    return String(
        "Binder Error: ",
        _clause_name(clause),
        " clause cannot contain",
        " aggregates!",
    )


def no_windows_here(clause: UInt8) -> String:
    """DuckDB's error for a window in a clause that refuses one.

    Args:
        clause: One of the `CLAUSE_` constants.

    Returns:
        The message.
    """
    return String(
        "Binder Error: ",
        _clause_name(clause),
        " clause cannot contain",
        " window functions!",
    )


def not_grouped(name: StringSlice) -> String:
    """DuckDB's error for a reference that is neither a key nor aggregated.

    Args:
        name: The column, as the query wrote it.

    Returns:
        The message, with the `ANY_VALUE` hint on the second line.
    """
    var written = String(name)
    return String(
        'Binder Error: column "',
        written,
        '" must appear in the GROUP BY clause or must be part of an aggregate',
        ' function.\nEither add it to the GROUP BY list, or use "ANY_VALUE(',
        written,
        ')" if the exact value of "',
        written,
        '" is not important.',
    )


def not_grouped_in_having(name: StringSlice) -> String:
    """The same error, in the shorter wording `HAVING` produces.

    Two spellings of one rule, both DuckDB's. This one has no quoting around
    the name and no hint after it.

    Args:
        name: The column, as the query wrote it.

    Returns:
        The message.
    """
    return String(
        "Binder Error: column ",
        name,
        (
            " must appear in the GROUP BY clause or be used in an aggregate"
            " function"
        ),
    )


def nested_aggregate() -> String:
    """DuckDB's error for an aggregate inside an aggregate.

    Returns:
        The message.
    """
    return String("Binder Error: aggregate function calls cannot be nested")


def window_in_aggregate() -> String:
    """DuckDB's error for a window inside an aggregate.

    Returns:
        The message.
    """
    return String(
        "Binder Error: aggregate function calls cannot contain window function"
        " calls"
    )


def window_in_window() -> String:
    """DuckDB's error for a window inside a window specification.

    Caught while parsing rather than while binding, so it carries the parser's
    prefix.

    Returns:
        The message.
    """
    return String(
        "Parser Error: window functions are not allowed in window definitions"
    )


def qualify_needs_a_window() -> String:
    """DuckDB's error for a `QUALIFY` with no window function anywhere.

    It asks for a window in the select list or in `QUALIFY` rather than saying
    that `QUALIFY` is what needs one, which is not what a reader would expect
    from the name of the clause.

    Returns:
        The message.
    """
    return String(
        "Binder Error: at least one window function must appear in the SELECT"
        " column or QUALIFY clause"
    )


def _clause_name(clause: UInt8) -> String:
    """What DuckDB calls a clause in a refusal.

    Args:
        clause: One of the `CLAUSE_` constants.

    Returns:
        The clause keyword.
    """
    if clause == CLAUSE_WHERE:
        return String("WHERE")
    if clause == CLAUSE_GROUP:
        return String("GROUP BY")
    if clause == CLAUSE_HAVING:
        return String("HAVING")
    if clause == CLAUSE_QUALIFY:
        return String("QUALIFY")
    if clause == CLAUSE_ORDER:
        return String("ORDER BY")
    return String("SELECT")


def _tags(ast: Ast, node: UInt32) raises -> String:
    """Everything about a node except the expressions under it.

    Written out as text rather than compared field by field, because which
    fields hold a child and which hold a tag is different for every kind, and
    a comparison that got one of those backwards would quietly make two
    different `CAST` targets look alike. Two group keys are a short list, so
    building a string to compare them costs nothing worth saving.

    Args:
        ast: The arenas.
        node: The expression.

    Returns:
        The tags, in a form that compares equal only for equal nodes.

    Raises:
        If a name run is empty where one is required.
    """
    ref item = ast.exprs[Int(node)]
    var kind = item.kind
    if kind == EXPR_COLUMN:
        return _folded_names(ast, item.children)
    if kind == EXPR_FUNCTION:
        return String(_folded_names(ast, item.payload), "/", item.a)
    if kind == EXPR_LITERAL:
        return String(item.b, "/", ast.text(item.payload))
    if kind == EXPR_UNARY or kind == EXPR_BINARY or kind == EXPR_COLLATE:
        return String(ast.text(item.payload))
    if kind == EXPR_CAST:
        return String(item.b, "/", ast.text(item.payload))
    if kind == EXPR_PARAMETER:
        return String(ast.text(item.b), "/", ast.text(item.payload))
    if kind == EXPR_BETWEEN or kind == EXPR_IN or kind == EXPR_IN_SUBQUERY:
        return String(item.payload)
    if kind == EXPR_STRUCT:
        var out = String()
        for at in range(0, ast.length(item.children), 2):
            out += ast.text(ast.at(item.children, at))
            out += "/"
        return out
    if kind == EXPR_FRAME:
        return String(item.payload)
    if kind == EXPR_WINDOW:
        return String(ast.text(item.payload))
    # A subquery and a star are each only equal to themselves. Two subqueries
    # that read the same could still be two different statements, and DuckDB
    # will not accept either as a group key anyway.
    if kind == EXPR_SUBQUERY or kind == EXPR_EXISTS or kind == EXPR_STAR:
        return String(node)
    if kind == EXPR_CASE or kind == EXPR_LIST:
        return String()
    raise Error("an expression kind with no tag rule")


def _folded_names(ast: Ast, run: UInt32) raises -> String:
    """A run of interned name parts, folded and joined.

    Args:
        ast: The arenas.
        run: The run.

    Returns:
        The parts, dot separated, so `T.A` and `t.a` come out the same.
    """
    var out = String()
    for at in range(ast.length(run)):
        if at != 0:
            out += "."
        out += fold(ast.text(ast.at(run, at)))
    return out^
