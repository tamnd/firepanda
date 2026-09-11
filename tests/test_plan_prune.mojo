"""Tests for projection pushdown.

Most of these are a plan in and a scan's column list out, because that list is
what the pass exists to shorten and it is the thing a reader can check by eye. A
few assert a whole printed plan instead, for the cases where the point is that a
project lost an output rather than that a scan lost a column.

The groups are: what a scan keeps, what it keeps when a node between it and the
root reads something nothing above wants, what a project and an aggregate drop,
the two places the pass declines to touch anything, and the shapes with more
than one input under them. The last test is a plan the shape of TPC-H q6 over a
sixteen column lineitem, which is where the pass is worth what the spec says it
is worth.
"""

from std.testing import TestSuite, assert_equal, assert_true

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.plan.node import SET_EXCEPT, Plan
from firepanda.plan.print import explain
from firepanda.plan.prune import prune


def _lineitem() -> Schema:
    """Returns a schema shaped like the TPC-H lineitem table.

    Returns:
        The sixteen columns, in the order the table has them.
    """
    var out = Schema()
    out.append(Field("l_orderkey", LogicalType.INT64, False))
    out.append(Field("l_partkey", LogicalType.INT64, False))
    out.append(Field("l_suppkey", LogicalType.INT64, False))
    out.append(Field("l_linenumber", LogicalType.INT32, False))
    out.append(Field("l_quantity", LogicalType.FLOAT64, True))
    out.append(Field("l_extendedprice", LogicalType.FLOAT64, True))
    out.append(Field("l_discount", LogicalType.FLOAT64, True))
    out.append(Field("l_tax", LogicalType.FLOAT64, True))
    out.append(Field("l_returnflag", LogicalType.STRING, True))
    out.append(Field("l_linestatus", LogicalType.STRING, True))
    out.append(Field("l_shipdate", LogicalType.DATE32, True))
    out.append(Field("l_commitdate", LogicalType.DATE32, True))
    out.append(Field("l_receiptdate", LogicalType.DATE32, True))
    out.append(Field("l_shipinstruct", LogicalType.STRING, True))
    out.append(Field("l_shipmode", LogicalType.STRING, True))
    out.append(Field("l_comment", LogicalType.STRING, True))
    return out^


def _orders() -> Schema:
    """Returns a schema shaped like a cut down TPC-H orders table.

    Returns:
        Four columns, the two keys not nullable.
    """
    var out = Schema()
    out.append(Field("o_orderkey", LogicalType.INT64, False))
    out.append(Field("o_custkey", LogicalType.INT64, False))
    out.append(Field("o_totalprice", LogicalType.FLOAT64, True))
    out.append(Field("o_comment", LogicalType.STRING, True))
    return out^


def _day(at: Int) -> Value:
    """Returns a date scalar, since a date column compares against a date."""
    var out = Value(Int32(at))
    out.type = LogicalType.DATE32
    return out^


def _read(plan: Plan, at: Int) -> String:
    """Returns a scan's column list as one comma separated string."""
    var out = String()
    for i in range(len(plan.nodes[at].names)):
        if i != 0:
            out += ", "
        out += plan.nodes[at].names[i]
    return out^


def test_a_scan_keeps_only_what_the_project_above_it_reads() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var qty = plan.exprs.column("l_quantity")
    var root = plan.project(scan, [qty], ["l_quantity"])
    _ = prune(plan, root, [_lineitem()])
    assert_equal(_read(plan, scan), "l_quantity", "one of sixteen")


def test_a_scan_with_a_column_list_already_on_it_is_narrowed_too() raises:
    # The list a caller wrote is a claim about what the query might read, not
    # about what it does read, so the pass narrows it the same way.
    var plan = Plan()
    var scan = plan.scan("lineitem", ["l_quantity", "l_discount"], 0)
    var qty = plan.exprs.column("l_quantity")
    var root = plan.project(scan, [qty], ["l_quantity"])
    _ = prune(plan, root, [_lineitem()])
    assert_equal(_read(plan, scan), "l_quantity", "the other one went")


def test_a_scan_keeps_the_columns_in_the_order_the_table_has_them() raises:
    # Not the order the expressions above ask for them in. A reader takes a
    # column list and the file's own order is the one it can read in one pass.
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var late = plan.exprs.column("l_shipdate")
    var early = plan.exprs.column("l_orderkey")
    var root = plan.project(scan, [late, early], ["a", "b"])
    _ = prune(plan, root, [_lineitem()])
    assert_equal(_read(plan, scan), "l_orderkey, l_shipdate", "table order")


def test_a_column_a_filter_reads_arrives_even_though_nothing_above_wants_it() raises:
    # The case the pass has to get right or it makes queries wrong rather than
    # slow. Nothing above the filter mentions `l_shipdate` and the filter
    # cannot run without it.
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var ship = plan.exprs.column("l_shipdate")
    var when = plan.exprs.literal(_day(9000))
    var kept = plan.filter(scan, plan.exprs.binary(BinaryOp.GE, ship, when))
    var qty = plan.exprs.column("l_quantity")
    var root = plan.project(kept, [qty], ["l_quantity"])
    _ = prune(plan, root, [_lineitem()])
    assert_equal(_read(plan, scan), "l_quantity, l_shipdate", "both")


def test_a_sort_key_arrives_even_though_nothing_above_wants_it() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var ship = plan.exprs.column("l_shipdate")
    var ordered = plan.sort(scan, [ship], [False], [False])
    var qty = plan.exprs.column("l_quantity")
    var root = plan.project(ordered, [qty], ["l_quantity"])
    _ = prune(plan, root, [_lineitem()])
    assert_equal(_read(plan, scan), "l_quantity, l_shipdate", "both")


def test_a_distinct_over_the_whole_row_reads_the_whole_row() raises:
    # A distinct with no keys dedups on every column, so narrowing the scan
    # under it would change which rows come out.
    var plan = Plan()
    var scan = plan.scan("orders", List[String](), 0)
    var whole = plan.distinct(scan, List[Int]())
    var key = plan.exprs.column("o_orderkey")
    var root = plan.project(whole, [key], ["o_orderkey"])
    _ = prune(plan, root, [_orders()])
    assert_equal(
        _read(plan, scan),
        "o_orderkey, o_custkey, o_totalprice, o_comment",
        "all four, written out",
    )


def test_a_distinct_with_keys_reads_its_keys_and_what_is_above() raises:
    var plan = Plan()
    var scan = plan.scan("orders", List[String](), 0)
    var cust = plan.exprs.column("o_custkey")
    var some = plan.distinct(scan, [cust])
    var price = plan.exprs.column("o_totalprice")
    var root = plan.project(some, [price], ["o_totalprice"])
    _ = prune(plan, root, [_orders()])
    assert_equal(_read(plan, scan), "o_custkey, o_totalprice", "two of four")


def test_a_limit_asks_for_nothing_of_its_own() raises:
    var plan = Plan()
    var scan = plan.scan("orders", List[String](), 0)
    var few = plan.limit(scan, 0, 10)
    var key = plan.exprs.column("o_orderkey")
    var root = plan.project(few, [key], ["o_orderkey"])
    _ = prune(plan, root, [_orders()])
    assert_equal(_read(plan, scan), "o_orderkey", "one of four")


def test_a_project_loses_an_output_nothing_above_it_reads() raises:
    # The other half of the pass. An unused output of a project is an
    # expression evaluated once per row for nobody, and the columns it reads
    # are columns nobody has to read either.
    var plan = Plan()
    var scan = plan.scan("orders", List[String](), 0)
    var key = plan.exprs.column("o_orderkey")
    var price = plan.exprs.column("o_totalprice")
    var both = plan.project(scan, [key, price], ["key", "price"])
    var again = plan.exprs.column("key")
    var root = plan.project(both, [again], ["key"])
    _ = prune(plan, root, [_orders()])
    assert_equal(len(plan.nodes[both].exprs), 1, "one output left")
    assert_equal(plan.nodes[both].names[0], "key", "and it is the used one")
    assert_equal(_read(plan, scan), "o_orderkey", "so the scan reads one")


def test_an_aggregate_loses_a_measure_nothing_above_it_reads() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var flag = plan.exprs.column("l_returnflag")
    var qty = plan.exprs.column("l_quantity")
    var tax = plan.exprs.column("l_tax")
    var summed = plan.aggregate(
        scan,
        [flag],
        [
            plan.exprs.aggregate(AggKind.SUM, qty),
            plan.exprs.aggregate(AggKind.SUM, tax),
        ],
        ["l_returnflag", "qty", "tax"],
    )
    var out = plan.exprs.column("qty")
    var root = plan.project(summed, [out], ["qty"])
    _ = prune(plan, root, [_lineitem()])
    assert_equal(len(plan.nodes[summed].exprs), 2, "the key and one measure")
    assert_equal(plan.nodes[summed].names[1], "qty", "the one that was read")
    assert_equal(_read(plan, scan), "l_quantity, l_returnflag", "and no l_tax")


def test_an_aggregate_keeps_a_group_key_nothing_above_it_reads() raises:
    # Dropping a key is not a projection, it is a different query: the rows
    # that came out one per flag would come out one for the lot.
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var flag = plan.exprs.column("l_returnflag")
    var qty = plan.exprs.column("l_quantity")
    var summed = plan.aggregate(
        scan,
        [flag],
        [plan.exprs.aggregate(AggKind.SUM, qty)],
        ["l_returnflag", "qty"],
    )
    var out = plan.exprs.column("qty")
    var root = plan.project(summed, [out], ["qty"])
    _ = prune(plan, root, [_lineitem()])
    assert_equal(len(plan.nodes[summed].exprs), 2, "both still there")
    assert_equal(
        _read(plan, scan), "l_quantity, l_returnflag", "so the key is read"
    )


def test_a_whole_frame_reduction_that_reads_nothing_still_reads_one_column() raises:
    # `count(*)` over a table demands no column of it, and a scan with no
    # column list means every column, so the narrowest list the pass can write
    # is one long rather than none.
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var one = plan.exprs.literal(Value(Int64(1)))
    var root = plan.aggregate(
        scan, List[Int](), [plan.exprs.aggregate(AggKind.COUNT, one)], ["n"]
    )
    _ = prune(plan, root, [_lineitem()])
    assert_equal(len(plan.nodes[scan].names), 1, "one column, not none")
    assert_equal(_read(plan, scan), "l_orderkey", "and the first will do")


def test_the_root_keeps_every_column_it_produces() raises:
    # The root's columns are the answer, so nothing above them is where the
    # demand starts rather than a reason to drop them.
    var plan = Plan()
    var scan = plan.scan("orders", List[String](), 0)
    var key = plan.exprs.column("o_orderkey")
    var price = plan.exprs.column("o_totalprice")
    var root = plan.project(scan, [key, price], ["key", "price"])
    var out = prune(plan, root, [_orders()])
    assert_equal(len(out), 2, "two columns out")
    assert_equal(_read(plan, scan), "o_orderkey, o_totalprice", "two in")


def test_a_scan_that_is_already_narrow_is_left_exactly_as_it_was() raises:
    # Nothing to drop, so the list the caller wrote comes back as it was. A
    # scan always comes out of the pass with an explicit list, so what this
    # checks is that rewriting it did not reorder or lose anything.
    var plan = Plan()
    var scan = plan.scan("lineitem", ["l_discount", "l_quantity"], 0)
    var disc = plan.exprs.column("l_discount")
    var qty = plan.exprs.column("l_quantity")
    var root = plan.project(scan, [disc, qty], ["a", "b"])
    _ = prune(plan, root, [_lineitem()])
    assert_equal(_read(plan, scan), "l_discount, l_quantity", "as written")


def test_a_project_with_two_outputs_of_one_name_is_left_alone() raises:
    # Dropping the first of two columns called the same thing would move the
    # second one into its position, and the pass rebinds by name afterwards, so
    # what an expression above resolved to would change under it. The plan is
    # ambiguous before the pass sees it and the pass declines to make it worse,
    # even though only the third output is wanted here.
    var plan = Plan()
    var scan = plan.scan("orders", List[String](), 0)
    var key = plan.exprs.column("o_orderkey")
    var cust = plan.exprs.column("o_custkey")
    var price = plan.exprs.column("o_totalprice")
    var twice = plan.project(scan, [key, cust, price], ["n", "n", "p"])
    var out = plan.exprs.column("p")
    var root = plan.project(twice, [out], ["p"])
    _ = prune(plan, root, [_orders()])
    assert_equal(len(plan.nodes[twice].exprs), 3, "all three outputs stay")


def test_a_join_narrows_each_side_to_what_that_side_provides() raises:
    var plan = Plan()
    var left = plan.scan("orders", List[String](), 0)
    var right = plan.scan("lineitem", List[String](), 1)
    var okey = plan.exprs.column("o_orderkey")
    var lkey = plan.exprs.column("l_orderkey")
    var joined = plan.join(left, right, [okey], [lkey], JoinKind.INNER)
    var price = plan.exprs.column("o_totalprice")
    var qty = plan.exprs.column("l_quantity")
    var root = plan.project(joined, [price, qty], ["price", "qty"])
    _ = prune(plan, root, [_orders(), _lineitem()])
    assert_equal(
        _read(plan, left), "o_orderkey, o_totalprice", "key and one column"
    )
    assert_equal(
        _read(plan, right), "l_orderkey, l_quantity", "key and one column"
    )


def test_a_join_key_arrives_even_though_nothing_above_wants_it() raises:
    var plan = Plan()
    var left = plan.scan("orders", List[String](), 0)
    var right = plan.scan("lineitem", List[String](), 1)
    var okey = plan.exprs.column("o_orderkey")
    var lkey = plan.exprs.column("l_orderkey")
    var joined = plan.join(left, right, [okey], [lkey], JoinKind.INNER)
    var qty = plan.exprs.column("l_quantity")
    var root = plan.project(joined, [qty], ["qty"])
    _ = prune(plan, root, [_orders(), _lineitem()])
    assert_equal(_read(plan, left), "o_orderkey", "the key alone")
    assert_equal(_read(plan, right), "l_orderkey, l_quantity", "key and one")


def test_both_arms_of_a_union_are_narrowed_the_same_way() raises:
    # Two arms of different widths are not a union, so asking both of them for
    # the same positions is what keeps the node legal after the rewrite.
    var plan = Plan()
    var first = plan.scan("orders", List[String](), 0)
    var second = plan.scan("orders", List[String](), 1)
    var stacked = plan.union([first, second], True)
    var key = plan.exprs.column("o_orderkey")
    var root = plan.project(stacked, [key], ["o_orderkey"])
    _ = prune(plan, root, [_orders(), _orders()])
    assert_equal(_read(plan, first), "o_orderkey", "one column")
    assert_equal(_read(plan, second), "o_orderkey", "and the same one")


def test_a_union_that_drops_duplicates_reads_the_whole_row() raises:
    # The same case as the distinct with no keys and it was not the same
    # answer. Two orders rows that agree on the key and differ on the price are
    # two rows, and narrowing both arms to the key alone makes them one, so the
    # pass would have deleted a row rather than a read.
    var plan = Plan()
    var first = plan.scan("orders", List[String](), 0)
    var second = plan.scan("orders", List[String](), 1)
    var stacked = plan.union([first, second], False)
    var key = plan.exprs.column("o_orderkey")
    var root = plan.project(stacked, [key], ["o_orderkey"])
    _ = prune(plan, root, [_orders(), _orders()])
    var all_four = "o_orderkey, o_custkey, o_totalprice, o_comment"
    assert_equal(_read(plan, first), all_four, "one arm whole")
    assert_equal(_read(plan, second), all_four, "and the other")


def test_a_values_is_left_alone_and_the_pass_walks_past_it() raises:
    # It has no input, so a pass that reached for one would be reaching into an
    # empty list. Leaving it whole is also the right answer: its rows are
    # already in the plan and reading them costs nothing.
    var plan = Plan()
    var one = plan.exprs.literal(Value(Int64(1)))
    var table = plan.values([one, one, one, one], ["a", "b"])
    var key = plan.exprs.column("a")
    var root = plan.project(table, [key], ["a"])
    _ = prune(plan, root, List[Schema]())
    assert_equal(
        explain(plan, root),
        "PROJECT [a]\n  VALUES [a, b] (1, 1), (1, 1)\n",
        "both columns still there",
    )


def test_a_difference_that_keeps_duplicates_reads_the_whole_row_too() raises:
    # The duplicate flag decides nothing here. Whether an orders row is in the
    # right arm is a question about the whole row whatever happens to the
    # copies of it afterwards, so both arms stay wide.
    var plan = Plan()
    var first = plan.scan("orders", List[String](), 0)
    var second = plan.scan("orders", List[String](), 1)
    var apart = plan.setop([first, second], SET_EXCEPT, all=True)
    var key = plan.exprs.column("o_orderkey")
    var root = plan.project(apart, [key], ["o_orderkey"])
    _ = prune(plan, root, [_orders(), _orders()])
    var all_four = "o_orderkey, o_custkey, o_totalprice, o_comment"
    assert_equal(_read(plan, first), all_four, "the left arm whole")
    assert_equal(_read(plan, second), all_four, "and the right one")


def test_a_union_that_keeps_duplicates_still_narrows() raises:
    # The one of the four that is a read and nothing else, so the pass is free
    # to take the other three columns away.
    var plan = Plan()
    var first = plan.scan("orders", List[String](), 0)
    var second = plan.scan("orders", List[String](), 1)
    var stacked = plan.union([first, second], True)
    var key = plan.exprs.column("o_orderkey")
    var root = plan.project(stacked, [key], ["o_orderkey"])
    _ = prune(plan, root, [_orders(), _orders()])
    assert_equal(_read(plan, first), "o_orderkey", "one column on the left")
    assert_equal(_read(plan, second), "o_orderkey", "and one on the right")


def test_a_q6_shaped_plan_reads_four_columns_of_sixteen() raises:
    # What the pass is worth, in the shape the spec measured it in. The
    # predicate reads three columns and the sum reads two, one of them shared,
    # so four columns of lineitem arrive and twelve never do.
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var ship = plan.exprs.column("l_shipdate")
    var disc = plan.exprs.column("l_discount")
    var qty = plan.exprs.column("l_quantity")
    var price = plan.exprs.column("l_extendedprice")
    var parts = List[Int]()
    parts.append(
        plan.exprs.binary(BinaryOp.GE, ship, plan.exprs.literal(_day(8766)))
    )
    parts.append(
        plan.exprs.binary(
            BinaryOp.LE, disc, plan.exprs.literal(Value(Float64(0.07)))
        )
    )
    parts.append(
        plan.exprs.binary(
            BinaryOp.LT, qty, plan.exprs.literal(Value(Float64(24.0)))
        )
    )
    # Nested rather than flat, because flattening is simplify's job and a plan
    # arrives here in whatever shape it was written in.
    var pair = List[Int]()
    pair.append(parts[0])
    pair.append(parts[1])
    var both = plan.exprs.call(String("and"), pair^, rowwise=True)
    var rest = List[Int]()
    rest.append(both)
    rest.append(parts[2])
    var kept = plan.filter(
        scan, plan.exprs.call(String("and"), rest^, rowwise=True)
    )
    var line = plan.exprs.binary(BinaryOp.MUL, price, disc)
    var root = plan.aggregate(
        kept,
        List[Int](),
        [plan.exprs.aggregate(AggKind.SUM, line)],
        ["revenue"],
    )
    _ = prune(plan, root, [_lineitem()])
    assert_equal(
        _read(plan, scan),
        "l_quantity, l_extendedprice, l_discount, l_shipdate",
        "four of sixteen, in table order",
    )
    assert_true(
        "SCAN lineitem [l_quantity" in explain(plan, root),
        "and the explain output says so",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
