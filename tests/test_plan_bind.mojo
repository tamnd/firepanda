"""Tests for binding a plan: names to positions, and a type on every node.

Three kinds of test here and they are worth telling apart.

The first kind asserts a position. That is the whole point of binding and it is
the thing that lets execution stop looking columns up by name, so the tests
reach into `Expr.at` and say what number it should be rather than asserting
something prettier that would pass whatever the number was.

The second kind asserts a type, and most of those are really asserting that
binding asks the kernel rather than deciding for itself. A sum over an int32
column is an int64 because `accumulator` says so, and if that ever changes this
test should change with it rather than keep a second opinion alive.

The third kind asserts a refusal, and there is one for each thing that can be
wrong with a query before it runs: a name that is not there, a predicate that is
not a question, an operation with no answer for its operands, two join keys with
nothing in common, and two union arms of different widths.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_raises
from std.testing import assert_true

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.kernel.unary import UnaryOp
from firepanda.plan.bind import bind, bind_expr
from firepanda.plan.expr import UNBOUND, Expressions
from firepanda.plan.node import NO_LIMIT, SET_EXCEPT, SET_INTERSECT, Plan


def _customer() -> Schema:
    """Returns a schema shaped like the TPC-H customer table.

    Returns:
        Four columns, the key not nullable and the rest of them nullable.
    """
    var out = Schema()
    out.append(Field("c_custkey", LogicalType.INT64, False))
    out.append(Field("c_name", LogicalType.STRING, True))
    out.append(Field("c_mktsegment", LogicalType.STRING, True))
    out.append(Field("c_acctbal", LogicalType.FLOAT64, True))
    return out^


def _orders() -> Schema:
    """Returns a schema shaped like the TPC-H orders table.

    Returns:
        Three columns, the two keys not nullable.
    """
    var out = Schema()
    out.append(Field("o_orderkey", LogicalType.INT64, False))
    out.append(Field("o_custkey", LogicalType.INT64, False))
    out.append(Field("o_totalprice", LogicalType.FLOAT64, True))
    return out^


def test_a_scan_binds_its_columns_to_the_table_it_reads() raises:
    var plan = Plan()
    var scan = plan.scan("customer", ["c_name", "c_acctbal"], 0)
    var out = bind(plan, scan, [_customer()])
    assert_equal(len(out), 2, "two columns were asked for")
    assert_equal(out[0].name, "c_name", "in the order they were asked for")
    assert_equal(out[1].dtype, LogicalType.FLOAT64, "with the table's type")


def test_a_scan_with_no_column_list_reads_the_whole_table() raises:
    var plan = Plan()
    var scan = plan.scan("customer", List[String](), 0)
    var out = bind(plan, scan, [_customer()])
    assert_equal(len(out), 4, "every column of the table")
    assert_equal(out[3].name, "c_acctbal", "in the table's own order")


def test_a_scan_of_a_relation_with_no_schema_is_refused() raises:
    var plan = Plan()
    var scan = plan.scan("orders", ["o_orderkey"], 3)
    with assert_raises(contains="is relation 3 and 1 schemas were given"):
        _ = bind(plan, scan, [_customer()])


def test_a_name_that_is_close_to_a_column_gets_a_suggestion() raises:
    var plan = Plan()
    var scan = plan.scan("customer", ["c_acctbl"], 0)
    with assert_raises(contains="Did you mean 'c_acctbal'?"):
        _ = bind(plan, scan, [_customer()])


def test_a_name_that_is_close_to_nothing_gets_no_suggestion() raises:
    var plan = Plan()
    var scan = plan.scan("customer", ["l_shipdate"], 0)
    with assert_raises(contains="there is no column named 'l_shipdate'"):
        _ = bind(plan, scan, [_customer()])
    var caught = String()
    try:
        _ = bind(plan, scan, [_customer()])
    except e:
        caught = String(e)
    assert_false("Did you mean" in caught, "nothing here is close to it")


def test_a_filter_binds_its_predicate_to_positions() raises:
    var plan = Plan()
    var scan = plan.scan("customer", ["c_custkey", "c_mktsegment"], 0)
    var seg = plan.exprs.column("c_mktsegment")
    var want = plan.exprs.literal(Value(String("BUILDING")))
    var is_it = plan.exprs.binary(BinaryOp.EQ, seg, want)
    var kept = plan.filter(scan, is_it)
    _ = bind(plan, kept, [_customer()])
    assert_equal(plan.exprs.nodes[seg].at, 1, "the second column scanned")
    assert_equal(plan.exprs.nodes[seg].table, 0, "read from relation zero")
    assert_equal(
        plan.exprs.nodes[is_it].type,
        LogicalType.BOOL,
        "a comparison answers a yes or a no",
    )


def test_a_filter_keeps_the_schema_it_was_given() raises:
    var plan = Plan()
    var scan = plan.scan("customer", ["c_custkey", "c_acctbal"], 0)
    var bal = plan.exprs.column("c_acctbal")
    var zero = plan.exprs.literal(Value(Float64(0.0)))
    var rich = plan.exprs.binary(BinaryOp.GT, bal, zero)
    var kept = plan.filter(scan, rich)
    var out = bind(plan, kept, [_customer()])
    assert_equal(len(out), 2, "a filter drops rows and not columns")
    assert_equal(out[1].name, "c_acctbal", "unchanged")


def test_a_predicate_that_is_not_a_question_is_refused() raises:
    var plan = Plan()
    var scan = plan.scan("customer", ["c_acctbal"], 0)
    var bal = plan.exprs.column("c_acctbal")
    var kept = plan.filter(scan, bal)
    with assert_raises(contains="and this one asks float64"):
        _ = bind(plan, kept, [_customer()])


def test_arithmetic_takes_its_type_from_the_kernel() raises:
    var plan = Plan()
    var scan = plan.scan("customer", ["c_custkey", "c_acctbal"], 0)
    var key = plan.exprs.column("c_custkey")
    var bal = plan.exprs.column("c_acctbal")
    var both = plan.exprs.binary(BinaryOp.ADD, key, bal)
    var out = plan.project(scan, [both], ["mixed"])
    var schema = bind(plan, out, [_customer()])
    assert_equal(
        schema[0].dtype,
        LogicalType.FLOAT64,
        "int64 with float64 promotes the way binary_type says",
    )


def test_an_operation_with_no_answer_is_a_plan_error() raises:
    var plan = Plan()
    var scan = plan.scan("customer", ["c_name", "c_acctbal"], 0)
    var name = plan.exprs.column("c_name")
    var bal = plan.exprs.column("c_acctbal")
    var nonsense = plan.exprs.binary(BinaryOp.SUB, name, bal)
    var out = plan.project(scan, [nonsense], ["nonsense"])
    with assert_raises():
        _ = bind(plan, out, [_customer()])


def test_a_unary_operation_keeps_the_type_it_was_handed() raises:
    var tree = Expressions()
    var bal = tree.column("c_acctbal")
    var owed = tree.unary(UnaryOp.NEG, bal)
    bind_expr(tree, owed, _customer(), [0, 0, 0, 0])
    assert_equal(
        tree.nodes[owed].type, LogicalType.FLOAT64, "a negation of a float"
    )


def test_a_cast_answers_what_it_was_asked_for() raises:
    var tree = Expressions()
    var key = tree.column("c_custkey")
    var wider = tree.cast(LogicalType.FLOAT64, key)
    bind_expr(tree, wider, _customer(), [0, 0, 0, 0])
    assert_equal(tree.nodes[wider].type, LogicalType.FLOAT64, "as asked")
    assert_equal(
        tree.nodes[key].type, LogicalType.INT64, "under it, the column"
    )


def test_a_conditional_promotes_its_two_answers() raises:
    var tree = Expressions()
    var bal = tree.column("c_acctbal")
    var zero = tree.literal(Value(Float64(0.0)))
    var rich = tree.binary(BinaryOp.GT, bal, zero)
    var one = tree.literal(Value(Int64(1)))
    var picked = tree.conditional(rich, one, bal)
    bind_expr(tree, picked, _customer(), [0, 0, 0, 0])
    assert_equal(
        tree.nodes[picked].type,
        LogicalType.FLOAT64,
        "int64 and float64 both fit in float64",
    )


def test_a_conditional_that_asks_no_question_is_refused() raises:
    var tree = Expressions()
    var bal = tree.column("c_acctbal")
    var one = tree.literal(Value(Int64(1)))
    var picked = tree.conditional(bal, one, one)
    with assert_raises(contains="asks a yes or no question"):
        bind_expr(tree, picked, _customer(), [0, 0, 0, 0])


def test_the_connectives_are_calls_and_answer_a_yes_or_a_no() raises:
    var tree = Expressions()
    var bal = tree.column("c_acctbal")
    var zero = tree.literal(Value(Float64(0.0)))
    var rich = tree.binary(BinaryOp.GT, bal, zero)
    var name = tree.column("c_mktsegment")
    var want = tree.literal(Value(String("BUILDING")))
    var building = tree.binary(BinaryOp.EQ, name, want)
    var both = tree.call("and", [rich, building], rowwise=True)
    bind_expr(tree, both, _customer(), [0, 0, 0, 0])
    assert_equal(tree.nodes[both].type, LogicalType.BOOL, "a conjunction")


def test_a_conjunction_of_three_binds_because_the_pass_makes_them() raises:
    # The simplify pass flattens `a AND (b AND c)` into one call with three
    # arguments, and a plan that has been through a pass binds again after it,
    # so a fixed arity of two here would reject the pass's own output.
    var tree = Expressions()
    var bal = tree.column("c_acctbal")
    var zero = tree.literal(Value(Float64(0.0)))
    var rich = tree.binary(BinaryOp.GT, bal, zero)
    var richer = tree.binary(BinaryOp.GT, bal, zero)
    var richest = tree.binary(BinaryOp.GT, bal, zero)
    var all_of = tree.call("and", [rich, richer, richest], rowwise=True)
    bind_expr(tree, all_of, _customer(), [0, 0, 0, 0])
    assert_equal(tree.nodes[all_of].type, LogicalType.BOOL, "a conjunction")


def test_a_connective_with_one_argument_is_still_refused() raises:
    var tree = Expressions()
    var bal = tree.column("c_acctbal")
    var zero = tree.literal(Value(Float64(0.0)))
    var rich = tree.binary(BinaryOp.GT, bal, zero)
    var alone = tree.call("or", [rich], rowwise=True)
    with assert_raises(contains="takes two or more arguments"):
        bind_expr(tree, alone, _customer(), [0, 0, 0, 0])


def test_a_negation_still_takes_exactly_one() raises:
    var tree = Expressions()
    var bal = tree.column("c_acctbal")
    var zero = tree.literal(Value(Float64(0.0)))
    var rich = tree.binary(BinaryOp.GT, bal, zero)
    var twice = tree.call("not", [rich, rich], rowwise=True)
    with assert_raises(contains="'not' takes 1 argument"):
        bind_expr(tree, twice, _customer(), [0, 0, 0, 0])


def test_a_connective_over_something_that_is_not_a_question_is_refused() raises:
    var tree = Expressions()
    var bal = tree.column("c_acctbal")
    var name = tree.column("c_name")
    var both = tree.call("and", [bal, name], rowwise=True)
    with assert_raises(contains="'and' reads yes or no and argument 0 is"):
        bind_expr(tree, both, _customer(), [0, 0, 0, 0])


def test_a_function_nobody_has_written_yet_is_refused_by_name() raises:
    var tree = Expressions()
    var name = tree.column("c_name")
    var shouted = tree.call("upper", [name], rowwise=True)
    with assert_raises(contains="there is no function named 'upper' yet"):
        bind_expr(tree, shouted, _customer(), [0, 0, 0, 0])


def test_a_sum_widens_the_way_the_accumulator_does() raises:
    var narrow = Schema()
    narrow.append(Field("n", LogicalType.INT32, True))
    var plan = Plan()
    var scan = plan.scan("counts", ["n"], 0)
    var n = plan.exprs.column("n")
    var total = plan.exprs.aggregate(AggKind.SUM, n)
    var out = plan.aggregate(scan, List[Int](), [total], ["total"])
    var schema = bind(plan, out, [narrow^])
    assert_equal(
        schema[0].dtype,
        LogicalType.INT64,
        "a sum of int32 accumulates in int64",
    )


def test_a_group_key_keeps_its_own_type_and_the_aggregate_gets_its_own() raises:
    var plan = Plan()
    var scan = plan.scan("orders", ["o_custkey", "o_totalprice"], 0)
    var key = plan.exprs.column("o_custkey")
    var price = plan.exprs.column("o_totalprice")
    var spent = plan.exprs.aggregate(AggKind.SUM, price)
    var out = plan.aggregate(scan, [key], [spent], ["o_custkey", "spent"])
    var schema = bind(plan, out, [_orders()])
    assert_equal(len(schema), 2, "the key and then the aggregate")
    assert_equal(schema[0].dtype, LogicalType.INT64, "the key as it was")
    assert_equal(schema[1].dtype, LogicalType.FLOAT64, "a total of floats")


def test_a_count_is_a_number_whatever_it_counted() raises:
    var plan = Plan()
    var scan = plan.scan("customer", ["c_name"], 0)
    var name = plan.exprs.column("c_name")
    var how_many = plan.exprs.aggregate(AggKind.COUNT, name)
    var out = plan.aggregate(scan, List[Int](), [how_many], ["n"])
    var schema = bind(plan, out, [_customer()])
    assert_equal(schema[0].dtype, LogicalType.INT64, "a count of text")


def test_a_projected_literal_cannot_be_null() raises:
    var plan = Plan()
    var scan = plan.scan("customer", ["c_name"], 0)
    var one = plan.exprs.literal(Value(Int64(1)))
    var out = plan.project(scan, [one], ["one"])
    var schema = bind(plan, out, [_customer()])
    assert_false(schema[0].nullable, "a constant that is not null never is")
    assert_equal(schema[0].dtype, LogicalType.INT64, "and it knows its type")


def test_a_projected_column_keeps_the_nullability_it_had() raises:
    var plan = Plan()
    var scan = plan.scan("customer", ["c_custkey", "c_name"], 0)
    var key = plan.exprs.column("c_custkey")
    var name = plan.exprs.column("c_name")
    var out = plan.project(scan, [key, name], ["k", "n"])
    var schema = bind(plan, out, [_customer()])
    assert_false(schema[0].nullable, "the key was not nullable in the table")
    assert_true(schema[1].nullable, "and the name was")


def test_a_column_above_a_projection_binds_to_the_projection() raises:
    var plan = Plan()
    var scan = plan.scan("customer", ["c_name", "c_acctbal"], 0)
    var bal = plan.exprs.column("c_acctbal")
    var out = plan.project(scan, [bal], ["balance"])
    var again = plan.exprs.column("balance")
    var zero = plan.exprs.literal(Value(Float64(0.0)))
    var rich = plan.exprs.binary(BinaryOp.GT, again, zero)
    var kept = plan.filter(out, rich)
    _ = bind(plan, kept, [_customer()])
    assert_equal(
        plan.exprs.nodes[bal].at, 1, "under the projection, position 1"
    )
    assert_equal(plan.exprs.nodes[again].at, 0, "above it, position 0")


def test_a_computed_column_over_one_table_keeps_that_table() raises:
    var plan = Plan()
    var scan = plan.scan("customer", ["c_acctbal"], 2)
    var bal = plan.exprs.column("c_acctbal")
    var two = plan.exprs.literal(Value(Float64(2.0)))
    var doubled = plan.exprs.binary(BinaryOp.MUL, bal, two)
    var out = plan.project(scan, [doubled], ["doubled"])
    var again = plan.exprs.column("doubled")
    var big = plan.exprs.binary(BinaryOp.GT, again, two)
    var kept = plan.filter(out, big)
    _ = bind(plan, kept, [Schema(), Schema(), _customer()])
    assert_equal(plan.exprs.nodes[again].table, 2, "still relation two")
    assert_equal(plan.exprs.tables(big), UInt64(4), "and the mask says so")


def test_a_computed_column_over_no_table_has_none() raises:
    var plan = Plan()
    var scan = plan.scan("customer", ["c_name"], 0)
    var one = plan.exprs.literal(Value(Int64(1)))
    var out = plan.project(scan, [one], ["one"])
    var again = plan.exprs.column("one")
    var same = plan.exprs.binary(BinaryOp.EQ, again, one)
    var kept = plan.filter(out, same)
    _ = bind(plan, kept, [_customer()])
    assert_equal(
        plan.exprs.nodes[again].table,
        UNBOUND,
        "a constant column came from no relation",
    )
    with assert_raises(contains="has no table until binding has run"):
        _ = plan.exprs.tables(same)


def test_a_join_stacks_the_two_schemas_end_to_end() raises:
    var plan = Plan()
    var c = plan.scan("customer", ["c_custkey", "c_name"], 0)
    var o = plan.scan("orders", ["o_custkey", "o_totalprice"], 1)
    var ck = plan.exprs.column("c_custkey")
    var ok = plan.exprs.column("o_custkey")
    var joined = plan.join(c, o, [ck], [ok], JoinKind.INNER)
    var schema = bind(plan, joined, [_customer(), _orders()])
    assert_equal(len(schema), 4, "two from each side")
    assert_equal(schema[2].name, "o_custkey", "the right side after the left")
    assert_equal(plan.exprs.nodes[ok].at, 0, "a right key is bound on its own")
    assert_equal(plan.exprs.nodes[ok].table, 1, "to relation one")


def test_a_column_above_a_join_is_bound_across_both_sides() raises:
    var plan = Plan()
    var c = plan.scan("customer", ["c_custkey", "c_name"], 0)
    var o = plan.scan("orders", ["o_custkey", "o_totalprice"], 1)
    var ck = plan.exprs.column("c_custkey")
    var ok = plan.exprs.column("o_custkey")
    var joined = plan.join(c, o, [ck], [ok], JoinKind.INNER)
    var price = plan.exprs.column("o_totalprice")
    var floor = plan.exprs.literal(Value(Float64(100.0)))
    var big = plan.exprs.binary(BinaryOp.GT, price, floor)
    var kept = plan.filter(joined, big)
    _ = bind(plan, kept, [_customer(), _orders()])
    assert_equal(plan.exprs.nodes[price].at, 3, "the fourth column of the join")
    assert_equal(plan.exprs.nodes[price].table, 1, "still from orders")
    assert_equal(
        plan.exprs.tables(big), UInt64(2), "so the mask is orders only"
    )


def test_a_left_join_makes_the_right_side_able_to_go_missing() raises:
    var plan = Plan()
    var c = plan.scan("customer", ["c_custkey"], 0)
    var o = plan.scan("orders", ["o_custkey", "o_orderkey"], 1)
    var ck = plan.exprs.column("c_custkey")
    var ok = plan.exprs.column("o_custkey")
    var joined = plan.join(c, o, [ck], [ok], JoinKind.LEFT)
    var schema = bind(plan, joined, [_customer(), _orders()])
    assert_false(schema[0].nullable, "the left side is all still there")
    assert_true(schema[2].nullable, "a right key that matched nothing is null")


def test_a_semi_join_keeps_only_the_side_it_was_asking_about() raises:
    var plan = Plan()
    var c = plan.scan("customer", ["c_custkey", "c_name"], 0)
    var o = plan.scan("orders", ["o_custkey"], 1)
    var ck = plan.exprs.column("c_custkey")
    var ok = plan.exprs.column("o_custkey")
    var joined = plan.join(c, o, [ck], [ok], JoinKind.SEMI)
    var schema = bind(plan, joined, [_customer(), _orders()])
    assert_equal(len(schema), 2, "the left side and nothing else")
    assert_equal(schema[1].name, "c_name", "unchanged")


def test_a_join_key_pair_with_nothing_in_common_is_refused() raises:
    var plan = Plan()
    var c = plan.scan("customer", ["c_name"], 0)
    var o = plan.scan("orders", ["o_custkey"], 1)
    var name = plan.exprs.column("c_name")
    var key = plan.exprs.column("o_custkey")
    var joined = plan.join(c, o, [name], [key], JoinKind.INNER)
    with assert_raises(contains="there is no type that holds both"):
        _ = bind(plan, joined, [_customer(), _orders()])


def test_a_values_takes_the_type_that_holds_every_row() raises:
    var plan = Plan()
    var small = plan.exprs.literal(Value(Int32(1)))
    var large = plan.exprs.literal(Value(Float64(2.5)))
    var table = plan.values([small, large], ["n"])
    var schema = bind(plan, table, List[Schema]())
    assert_equal(len(schema), 1, "one column two rows tall")
    assert_equal(schema[0].dtype, LogicalType.FLOAT64, "the type that holds it")
    assert_false(schema[0].nullable, "two literals and neither is null")


def test_one_null_in_a_values_column_is_enough() raises:
    var plan = Plan()
    var one = plan.exprs.literal(Value(Int64(1)))
    var missing = plan.exprs.literal(Value(null=LogicalType.INT64))
    var table = plan.values([one, one, one, missing], ["a", "b"])
    var schema = bind(plan, table, List[Schema]())
    assert_false(schema[0].nullable, "no null in the first column")
    assert_true(schema[1].nullable, "and one in the second")


def test_a_values_column_of_two_types_that_do_not_meet_is_refused() raises:
    var plan = Plan()
    var one = plan.exprs.literal(Value(Int64(1)))
    var word = plan.exprs.literal(Value(String("a")))
    var table = plan.values([one, word], ["n"])
    with assert_raises(contains="column 0 of a VALUES puts"):
        _ = bind(plan, table, List[Schema]())


def test_a_values_belongs_to_no_relation() raises:
    # A qualified name over one has nothing to qualify, so the origin stays
    # unbound rather than pointing at whatever relation zero happens to be.
    var plan = Plan()
    var one = plan.exprs.literal(Value(Int64(1)))
    var table = plan.values([one], ["n"])
    var key = plan.exprs.column("n")
    var root = plan.project(table, [key], ["n"])
    _ = bind(plan, root, List[Schema]())
    assert_equal(plan.exprs.nodes[key].table, UNBOUND, "from nowhere")


def test_a_table_function_produces_one_int64_column() raises:
    var plan = Plan()
    var stop = plan.exprs.literal(Value(Int64(5)))
    var rows = plan.table_function("range", [stop], ["i"])
    var schema = bind(plan, rows, List[Schema]())
    assert_equal(len(schema), 1, "one column")
    assert_equal(schema[0].name, "i", "named the way the call named it")
    assert_equal(schema[0].dtype, LogicalType.INT64, "counting is in int64")
    assert_false(schema[0].nullable, "and a counted row is always there")


def test_a_table_function_types_the_arithmetic_in_its_arguments() raises:
    # The whole reason the arguments are bound at all. A literal knows its type
    # and `2 + 3` does not, and lowering reads the type off the expression.
    var plan = Plan()
    var two = plan.exprs.literal(Value(Int64(2)))
    var three = plan.exprs.literal(Value(Int64(3)))
    var sum = plan.exprs.binary(BinaryOp.ADD, two, three)
    var rows = plan.table_function("generate_series", [sum], ["i"])
    _ = bind(plan, rows, List[Schema]())
    assert_equal(plan.exprs.nodes[sum].type, LogicalType.INT64, "worked out")


def test_a_table_function_nobody_wrote_is_refused_by_name() raises:
    var plan = Plan()
    var stop = plan.exprs.literal(Value(Int64(5)))
    var rows = plan.table_function("read_parquet", [stop], ["i"])
    with assert_raises(
        contains="there is no table function called read_parquet"
    ):
        _ = bind(plan, rows, List[Schema]())


def test_a_table_function_takes_one_two_or_three_arguments() raises:
    var plan = Plan()
    var one = plan.exprs.literal(Value(Int64(1)))
    var four = plan.table_function("range", [one, one, one, one], ["i"])
    with assert_raises(contains="this call has 4 arguments"):
        _ = bind(plan, four, List[Schema]())
    var none = plan.table_function("range", List[Int](), ["i"])
    with assert_raises(contains="this call has 0 arguments"):
        _ = bind(plan, none, List[Schema]())


def test_a_table_function_counts_in_whole_numbers() raises:
    var plan = Plan()
    var half = plan.exprs.literal(Value(Float64(2.5)))
    var rows = plan.table_function("range", [half], ["i"])
    with assert_raises(contains="counts rows and is a float64"):
        _ = bind(plan, rows, List[Schema]())


def test_a_series_with_a_null_end_binds() raises:
    # It is a series of no rows rather than a query written wrong, which is
    # what DuckDB answers too.
    var plan = Plan()
    var missing = plan.exprs.literal(Value(null=LogicalType.NULL))
    var rows = plan.table_function("range", [missing], ["i"])
    var schema = bind(plan, rows, List[Schema]())
    assert_equal(schema[0].dtype, LogicalType.INT64, "still an int64 column")


def test_a_table_function_belongs_to_no_relation() raises:
    var plan = Plan()
    var stop = plan.exprs.literal(Value(Int64(5)))
    var rows = plan.table_function("range", [stop], ["i"])
    var i = plan.exprs.column("i")
    var root = plan.project(rows, [i], ["i"])
    _ = bind(plan, root, List[Schema]())
    assert_equal(plan.exprs.nodes[i].table, UNBOUND, "from nowhere")


def test_a_union_promotes_the_types_of_the_columns_it_stacks() raises:
    var narrow = Schema()
    narrow.append(Field("n", LogicalType.INT32, False))
    var wide = Schema()
    wide.append(Field("n", LogicalType.FLOAT64, False))
    var plan = Plan()
    var a = plan.scan("small", ["n"], 0)
    var b = plan.scan("large", ["n"], 1)
    var both = plan.union([a, b], all=True)
    var schema = bind(plan, both, [narrow^, wide^])
    assert_equal(
        schema[0].dtype, LogicalType.FLOAT64, "the type that holds both arms"
    )


def test_a_union_of_two_different_widths_is_refused() raises:
    var plan = Plan()
    var a = plan.scan("customer", ["c_custkey", "c_name"], 0)
    var b = plan.scan("orders", ["o_custkey"], 1)
    var both = plan.union([a, b], all=True)
    with assert_raises(contains="a union is between a 2 column input"):
        _ = bind(plan, both, [_customer(), _orders()])
    var apart = plan.setop([a, b], SET_EXCEPT, all=True)
    with assert_raises(contains="a difference is between a 2 column input"):
        _ = bind(plan, apart, [_customer(), _orders()])


def test_which_column_can_be_missing_is_a_question_per_operation() raises:
    # A union produces rows from either side, a difference only from the left,
    # and an intersection only rows that were on both, so the same pair of
    # inputs answers three different ways.
    var never = Schema()
    never.append(Field("n", LogicalType.INT64, False))
    var sometimes = Schema()
    sometimes.append(Field("n", LogicalType.INT64, True))

    var plan = Plan()
    var a = plan.scan("solid", ["n"], 0)
    var b = plan.scan("holey", ["n"], 1)
    var stacked = plan.union([a, b], all=True)
    var apart = plan.setop([a, b], SET_EXCEPT, all=True)
    var shared = plan.setop([a, b], SET_INTERSECT, all=True)

    # A fresh pair each time because binding takes them by value, and the point
    # is that the same two answer three different ways.
    assert_true(
        bind(plan, stacked, [Schema(copy=never), Schema(copy=sometimes)])[
            0
        ].nullable,
        "the right arm's missing values are in the answer",
    )
    assert_false(
        bind(plan, apart, [Schema(copy=never), Schema(copy=sometimes)])[
            0
        ].nullable,
        "every row of a difference came from the left",
    )
    assert_false(
        bind(plan, shared, [Schema(copy=never), Schema(copy=sometimes)])[
            0
        ].nullable,
        "and every row of an intersection was on both sides",
    )


def test_an_intersection_keeps_the_relation_the_left_arm_named() raises:
    # Unlike a union, where the column is two columns stacked and belongs to
    # neither table. A row in both is the left's row, so the left's origin is
    # true of it and a qualified name above still resolves.
    var plan = Plan()
    var a = plan.scan("customer", ["c_custkey"], 0)
    var b = plan.scan("orders", ["o_custkey"], 1)
    var shared = plan.setop([a, b], SET_INTERSECT, all=True)
    var key = plan.exprs.column("c_custkey")
    var floor = plan.exprs.literal(Value(Int64(0)))
    var big = plan.exprs.binary(BinaryOp.GT, key, floor)
    var kept = plan.filter(shared, big)
    _ = bind(plan, kept, [_customer(), _orders()])
    assert_equal(plan.exprs.nodes[key].table, 0, "the left arm's relation")


def test_a_union_of_two_relations_leaves_the_column_with_neither() raises:
    var plan = Plan()
    var a = plan.scan("customer", ["c_custkey"], 0)
    var b = plan.scan("orders", ["o_custkey"], 1)
    var both = plan.union([a, b], all=True)
    var key = plan.exprs.column("c_custkey")
    var floor = plan.exprs.literal(Value(Int64(0)))
    var big = plan.exprs.binary(BinaryOp.GT, key, floor)
    var kept = plan.filter(both, big)
    _ = bind(plan, kept, [_customer(), _orders()])
    assert_equal(
        plan.exprs.nodes[key].table,
        UNBOUND,
        "the column is two columns stacked, from two relations",
    )


def test_sort_limit_and_distinct_pass_the_schema_through() raises:
    var plan = Plan()
    var scan = plan.scan("customer", ["c_custkey", "c_acctbal"], 0)
    var bal = plan.exprs.column("c_acctbal")
    var ranked = plan.sort(scan, [bal], [True], [False])
    var top = plan.limit(ranked, 0, 10)
    var key = plan.exprs.column("c_custkey")
    var once = plan.distinct(top, [key])
    var schema = bind(plan, once, [_customer()])
    assert_equal(len(schema), 2, "none of the three touches a column")
    assert_equal(plan.exprs.nodes[bal].at, 1, "and the sort key still bound")
    assert_equal(plan.exprs.nodes[key].at, 0, "and so did the distinct key")


def test_a_node_that_is_not_under_the_root_is_left_alone() raises:
    var plan = Plan()
    var good = plan.scan("customer", ["c_custkey"], 0)
    # A scan a rewrite has already detached, naming a column that is not in the
    # schema. Binding the root should not look at it, and would refuse it if it
    # did.
    var stale = plan.scan("customer", ["gone"], 0)
    var key = plan.exprs.column("c_custkey")
    var floor = plan.exprs.literal(Value(Int64(0)))
    var big = plan.exprs.binary(BinaryOp.GT, key, floor)
    var kept = plan.filter(good, big)
    var schema = bind(plan, kept, [_customer()])
    assert_equal(len(schema), 1, "the live subtree bound")
    assert_true(stale > 0, "and the detached one was still in the plan")


def test_binding_a_node_that_is_not_in_the_plan_is_refused() raises:
    var plan = Plan()
    _ = plan.scan("customer", ["c_custkey"], 0)
    with assert_raises(contains="plan node 4 is not in a plan of 1"):
        _ = bind(plan, 4, [_customer()])


def test_the_whole_of_q3_binds() raises:
    var plan = Plan()
    var c = plan.scan("customer", ["c_custkey", "c_mktsegment"], 0)
    var seg = plan.exprs.column("c_mktsegment")
    var want = plan.exprs.literal(Value(String("BUILDING")))
    var building = plan.exprs.binary(BinaryOp.EQ, seg, want)
    var kept = plan.filter(c, building)

    var o = plan.scan("orders", ["o_orderkey", "o_custkey", "o_totalprice"], 1)
    var ck = plan.exprs.column("c_custkey")
    var ok = plan.exprs.column("o_custkey")
    var joined = plan.join(kept, o, [ck], [ok], JoinKind.INNER)

    var key = plan.exprs.column("o_orderkey")
    var price = plan.exprs.column("o_totalprice")
    var revenue = plan.exprs.aggregate(AggKind.SUM, price)
    var grouped = plan.aggregate(
        joined, [key], [revenue], ["o_orderkey", "revenue"]
    )
    var out = plan.exprs.column("revenue")
    var ranked = plan.sort(grouped, [out], [True], [False])
    var top = plan.limit(ranked, 0, 10)

    var schema = bind(plan, top, [_customer(), _orders()])
    assert_equal(len(schema), 2, "the group key and the total")
    assert_equal(schema[0].dtype, LogicalType.INT64, "the order key")
    assert_equal(schema[1].dtype, LogicalType.FLOAT64, "the revenue")
    assert_equal(plan.exprs.nodes[key].at, 2, "o_orderkey sits after customer")
    assert_equal(plan.exprs.nodes[key].table, 1, "and comes from orders")
    assert_equal(plan.exprs.nodes[out].at, 1, "revenue is the second output")


def _keyed(name: String) -> Schema:
    """Returns a two column schema whose key is called the same in both tables.

    Args:
        name: What to call the second column, so the two are otherwise apart.

    Returns:
        A `key` and one other column.
    """
    var out = Schema()
    out.append(Field("key", LogicalType.INT64, False))
    out.append(Field(name, LogicalType.STRING, True))
    return out^


def test_a_name_two_inputs_both_have_is_refused_rather_than_guessed() raises:
    # The reason this is an error and not a first match. A join puts its two
    # inputs end to end, so joining two tables that both have a `key` gives a
    # schema with two columns called `key`, and answering the left one is a
    # wrong answer that nothing downstream can notice.
    var plan = Plan()
    var left = plan.scan("l", List[String](), 0)
    var right = plan.scan("r", List[String](), 1)
    var lk = plan.exprs.column_of(0, "key")
    var rk = plan.exprs.column_of(1, "key")
    var joined = plan.join(left, right, [lk], [rk], JoinKind.INNER)
    var out = plan.exprs.column("key")
    var root = plan.project(joined, [out], ["key"])
    with assert_raises(contains="more than one column here, at 0 and at 2"):
        _ = bind(plan, root, [_keyed("a"), _keyed("b")])


def test_the_message_says_to_say_which_input_when_two_have_it() raises:
    var plan = Plan()
    var left = plan.scan("l", List[String](), 0)
    var right = plan.scan("r", List[String](), 1)
    var lk = plan.exprs.column_of(0, "key")
    var rk = plan.exprs.column_of(1, "key")
    var joined = plan.join(left, right, [lk], [rk], JoinKind.INNER)
    var out = plan.exprs.column("key")
    var root = plan.project(joined, [out], ["key"])
    with assert_raises(contains="say which input it is from"):
        _ = bind(plan, root, [_keyed("a"), _keyed("b")])


def test_a_name_one_input_has_twice_says_so_without_the_advice() raises:
    # Two outputs of one node called the same thing is ambiguous too, and the
    # advice that fits the join case does not fit this one, since there is only
    # one input and naming it would not narrow anything.
    var plan = Plan()
    var scan = plan.scan("orders", List[String](), 0)
    var key = plan.exprs.column("o_orderkey")
    var cust = plan.exprs.column("o_custkey")
    var twice = plan.project(scan, [key, cust], ["n", "n"])
    var out = plan.exprs.column("n")
    var root = plan.project(twice, [out], ["n"])
    with assert_raises(contains="more than one column here"):
        _ = bind(plan, root, [_orders()])
    try:
        _ = bind(plan, root, [_orders()])
    except e:
        assert_false(
            String(e).find("say which input") != -1,
            "naming the one input would not help",
        )


def test_a_column_that_says_which_input_it_is_from_binds_to_that_one() raises:
    var plan = Plan()
    var left = plan.scan("l", List[String](), 0)
    var right = plan.scan("r", List[String](), 1)
    var lk = plan.exprs.column_of(0, "key")
    var rk = plan.exprs.column_of(1, "key")
    var joined = plan.join(left, right, [lk], [rk], JoinKind.INNER)
    var theirs = plan.exprs.column_of(1, "key")
    var root = plan.project(joined, [theirs], ["key"])
    var schema = bind(plan, root, [_keyed("a"), _keyed("b")])
    assert_equal(len(schema), 1, "one column comes out")
    assert_equal(plan.exprs.nodes[theirs].at, 2, "the right arm's key")
    assert_equal(plan.exprs.nodes[theirs].table, 1, "from the right arm")


def test_saying_the_other_input_binds_to_the_other_one() raises:
    var plan = Plan()
    var left = plan.scan("l", List[String](), 0)
    var right = plan.scan("r", List[String](), 1)
    var lk = plan.exprs.column_of(0, "key")
    var rk = plan.exprs.column_of(1, "key")
    var joined = plan.join(left, right, [lk], [rk], JoinKind.INNER)
    var ours = plan.exprs.column_of(0, "key")
    var root = plan.project(joined, [ours], ["key"])
    _ = bind(plan, root, [_keyed("a"), _keyed("b")])
    assert_equal(plan.exprs.nodes[ours].at, 0, "the left arm's key")
    assert_equal(plan.exprs.nodes[ours].table, 0, "from the left arm")


def test_saying_an_input_that_does_not_have_it_says_another_one_does() raises:
    # The name is in the schema, on a column from the other arm, and a message
    # saying it is not here at all would send the reader looking for a typo
    # that is not there.
    var plan = Plan()
    var left = plan.scan("l", List[String](), 0)
    var right = plan.scan("r", List[String](), 1)
    var lk = plan.exprs.column_of(0, "key")
    var rk = plan.exprs.column_of(1, "key")
    var joined = plan.join(left, right, [lk], [rk], JoinKind.INNER)
    var wrong = plan.exprs.column_of(0, "b")
    var root = plan.project(joined, [wrong], ["b"])
    with assert_raises(contains="though another input has one"):
        _ = bind(plan, root, [_keyed("a"), _keyed("b")])


def test_saying_an_input_for_a_name_nobody_has_is_still_a_missing_name() raises:
    var plan = Plan()
    var scan = plan.scan("orders", List[String](), 0)
    var wrong = plan.exprs.column_of(0, "o_nothing")
    var root = plan.project(scan, [wrong], ["x"])
    with assert_raises(contains="there is no column named 'o_nothing'"):
        _ = bind(plan, root, [_orders()])


def test_saying_the_input_changes_nothing_when_the_name_is_unambiguous() raises:
    var plan = Plan()
    var scan = plan.scan("orders", List[String](), 0)
    var plain = plan.exprs.column("o_custkey")
    var said = plan.exprs.column_of(0, "o_custkey")
    var root = plan.project(scan, [plain, said], ["a", "b"])
    _ = bind(plan, root, [_orders()])
    assert_equal(plan.exprs.nodes[plain].at, 1, "the same position")
    assert_equal(plan.exprs.nodes[said].at, 1, "either way")


def test_a_column_says_nothing_about_an_input_until_it_is_asked_to() raises:
    var plan = Plan()
    var plain = plan.exprs.column("a")
    var said = plan.exprs.column_of(3, "a")
    assert_equal(plan.exprs.nodes[plain].table, UNBOUND, "nothing said")
    assert_equal(plan.exprs.nodes[said].table, 3, "and something said")
    assert_equal(
        plan.exprs.nodes[said].at, UNBOUND, "the position is binding's"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
