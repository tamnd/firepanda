"""Tests for the JSON form of a plan.

Most of these are round trips and they all assert the same two things, which is
what `_trip` is for: the plan that comes back explains the way the plan that
went out explains, and it writes the same document a second time. The first is
what a reader of the test checks against, and the second is what catches a field
that was dropped on the way out and therefore never got a chance to come back
wrong.

The groups are: the eleven node kinds, the nine expression kinds, the constants,
what a bound plan keeps, the sharing that `cse` and `subplan` leave behind, JSON
written by hand rather than by the writer, and the things that are refused.
"""

from std.math import isinf, isnan
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.dtype.temporal import TimeUnit, TimeZone
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.kernel.unary import UnaryOp
from firepanda.plan.bind import bind
from firepanda.plan.expr import UNBOUND
from firepanda.plan.json import Loaded, from_json, to_json
from firepanda.plan.node import NO_LIMIT, SET_EXCEPT, SET_INTERSECT, Plan
from firepanda.plan.print import explain, render_expr


def _trip(plan: Plan, root: Int) raises -> Loaded:
    """Writes a plan out, reads it back, and checks that nothing moved.

    Args:
        plan: The plan.
        root: The node to write.

    Returns:
        What came back, for a test that wants to look closer.

    Raises:
        If the plan does not survive the trip.
    """
    var text = to_json(plan, root)
    var back = from_json(text)
    assert_equal(
        explain(back.plan, back.root),
        explain(plan, root),
        "the plan that came back explains the same",
    )
    assert_equal(
        to_json(back.plan, back.root), text, "and writes the same document"
    )
    return back^


def _table() -> Schema:
    """Returns a small schema to bind against.

    Returns:
        Three columns.
    """
    var out = Schema()
    out.append(Field("a", LogicalType.INT64, False))
    out.append(Field("b", LogicalType.FLOAT64, True))
    out.append(Field("c", LogicalType.STRING, True))
    return out^


def _constant(var value: Value) raises -> Value:
    """Puts one constant through a values node and brings it back.

    A literal on its own is not a plan, so the shortest plan that carries one is
    a one row one column table of it, and that is what every constant test here
    writes out and reads in.

    Args:
        value: The constant.

    Returns:
        The constant that came back.

    Raises:
        If it does not survive the trip.
    """
    var plan = Plan()
    var at = plan.values([plan.exprs.literal(value^)], ["x"])
    var back = from_json(to_json(plan, at))
    return back.plan.exprs.nodes[
        back.plan.nodes[back.root].exprs[0]
    ].value.copy()


def test_a_scan_goes_out_and_comes_back() raises:
    var plan = Plan()
    var at = plan.scan("t", ["a", "b"], 3)
    assert_equal(
        to_json(plan, at),
        '{"kind": "scan", "source": "t", "table": 3, "columns": ["a", "b"]}',
        "a scan is its source, its relation and its columns",
    )
    var back = _trip(plan, at)
    assert_equal(
        back.plan.nodes[back.root].table, 3, "and it is still relation three"
    )


def test_a_filter_goes_out_and_comes_back() raises:
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var a = plan.exprs.column("a")
    var one = plan.exprs.literal(Value(Int64(1)))
    var at = plan.filter(t, plan.exprs.binary(BinaryOp.GT, a, one))
    _ = _trip(plan, at)


def test_a_project_keeps_the_name_beside_the_expression() raises:
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var a = plan.exprs.column("a")
    var b = plan.exprs.column("b")
    var at = plan.project(
        t, [a, plan.exprs.binary(BinaryOp.ADD, a, b)], ["a", "sum"]
    )
    var text = to_json(plan, at)
    assert_true(
        text.find('{"name": "sum", "expr": {"kind": "binary"') != -1,
        "a column of a project is a name and an expression together",
    )
    _ = _trip(plan, at)


def test_an_aggregate_keeps_its_keys_and_its_folds_apart() raises:
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var a = plan.exprs.column("a")
    var b = plan.exprs.column("b")
    var at = plan.aggregate(
        t, [a], [plan.exprs.aggregate(AggKind.SUM, b)], ["a", "total"]
    )
    var back = _trip(plan, at)
    assert_equal(
        back.plan.nodes[back.root].parts, 1, "one of the two columns is a key"
    )


def test_an_aggregate_over_no_keys_is_a_whole_frame_fold() raises:
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var b = plan.exprs.column("b")
    var at = plan.aggregate(
        t, List[Int](), [plan.exprs.aggregate(AggKind.MEAN, b)], ["mean"]
    )
    var back = _trip(plan, at)
    assert_equal(
        back.plan.nodes[back.root].parts, 0, "and no keys is still no keys"
    )


def test_a_join_pairs_its_keys_up_across_the_two_arms() raises:
    var plan = Plan()
    var left = plan.scan("t", List[String](), 0)
    var right = plan.scan("u", List[String](), 1)
    var a = plan.exprs.column("a")
    var k = plan.exprs.column("k")
    var at = plan.join(left, right, [a], [k], JoinKind.LEFT)
    var text = to_json(plan, at)
    assert_true(
        text.find('"how": "left"') != -1, "the kind is written as its word"
    )
    _ = _trip(plan, at)


def test_a_sort_keeps_a_direction_and_a_null_placement_per_key() raises:
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var a = plan.exprs.column("a")
    var b = plan.exprs.column("b")
    var at = plan.sort(t, [a, b], [True, False], [False, True])
    _ = _trip(plan, at)


def test_a_bounded_sort_keeps_its_bound() raises:
    # What `limits` leaves behind when a limit sits over a sort, which is a top
    # n and is not a limit node.
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var at = plan.sort(t, [plan.exprs.column("a")], [False], [False])
    plan.nodes[at].length = 10
    assert_true(
        to_json(plan, at).find('"length": 10') != -1,
        "the bound is written down",
    )
    var back = _trip(plan, at)
    assert_equal(back.plan.nodes[back.root].length, 10, "and comes back")


def test_an_unbounded_sort_says_nothing_about_a_length() raises:
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var at = plan.sort(t, [plan.exprs.column("a")], [False], [False])
    assert_true(
        to_json(plan, at).find("length") == -1,
        "a sort nobody bounded has no bound in it",
    )
    var back = _trip(plan, at)
    assert_equal(
        back.plan.nodes[back.root].length,
        NO_LIMIT,
        "and comes back without one",
    )


def test_a_limit_that_only_skips_writes_no_length() raises:
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var at = plan.limit(t, 5, NO_LIMIT)
    assert_equal(
        to_json(plan, at).find("length"), -1, "there is no length to write"
    )
    var back = _trip(plan, at)
    assert_equal(
        back.plan.nodes[back.root].length, NO_LIMIT, "and none comes back"
    )


def test_a_limit_of_no_rows_is_not_a_limit_of_none() raises:
    # Zero rows and no bound are two different things and both are written, so
    # the one that is a real length has to survive as one.
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var at = plan.limit(t, 0, 0)
    var back = _trip(plan, at)
    assert_equal(back.plan.nodes[back.root].length, 0, "zero rows is a length")


def test_a_distinct_over_the_whole_row_has_no_keys() raises:
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var at = plan.distinct(t, List[Int]())
    _ = _trip(plan, at)


def test_a_distinct_over_some_columns_keeps_them() raises:
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var at = plan.distinct(t, [plan.exprs.column("a")])
    _ = _trip(plan, at)


def test_the_three_set_operations_write_their_own_word() raises:
    var plan = Plan()
    var left = plan.scan("t", List[String](), 0)
    var right = plan.scan("u", List[String](), 1)
    var stacked = plan.union([left, right], True)
    var minus = plan.setop([left, right], SET_EXCEPT, False)
    var both = plan.setop([left, right], SET_INTERSECT, False)
    assert_true(
        to_json(plan, stacked).find('"kind": "union", "all": true') != -1,
        "a union says so",
    )
    assert_true(
        to_json(plan, minus).find('"kind": "except"') != -1,
        "and so does a difference",
    )
    assert_true(
        to_json(plan, both).find('"kind": "intersect"') != -1,
        "and so does an intersection",
    )
    _ = _trip(plan, stacked)
    _ = _trip(plan, minus)
    _ = _trip(plan, both)


def test_a_union_of_more_than_two_arms_keeps_all_of_them() raises:
    var plan = Plan()
    var arms = List[Int]()
    for i in range(4):
        arms.append(plan.scan(String("t", i), List[String](), i))
    var at = plan.union(arms^, True)
    var back = _trip(plan, at)
    assert_equal(
        len(back.plan.nodes[back.root].inputs),
        4,
        "four arms went out and four came back",
    )


def test_a_values_goes_out_as_rows_rather_than_as_one_list() raises:
    var plan = Plan()
    var rows = List[Int]()
    for i in range(1, 5):
        rows.append(plan.exprs.literal(Value(Int64(i))))
    var at = plan.values(rows^, ["x", "y"])
    assert_true(
        to_json(plan, at).find('"rows": [[') != -1,
        "the rows are written as rows",
    )
    var back = _trip(plan, at)
    assert_equal(back.plan.nodes[back.root].parts, 2, "two columns wide")


def test_a_table_function_carries_its_name_and_its_arguments() raises:
    var plan = Plan()
    var start = plan.exprs.literal(Value(Int64(1)))
    var stop = plan.exprs.literal(Value(Int64(10)))
    var at = plan.table_function("range", [start, stop], ["i"])
    assert_true(
        to_json(plan, at).find('"function": "range"') != -1,
        "what it is called is in the document",
    )
    var back = _trip(plan, at)
    assert_equal(
        len(back.plan.nodes[back.root].exprs), 2, "both arguments came back"
    )


def test_a_table_function_with_no_arguments_comes_back_with_none() raises:
    var plan = Plan()
    var at = plan.table_function("now", List[Int](), ["t"])
    var back = _trip(plan, at)
    assert_equal(len(back.plan.nodes[back.root].exprs), 0, "and none came back")


def test_every_expression_kind_goes_out_and_comes_back() raises:
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var a = plan.exprs.column("a")
    var one = plan.exprs.literal(Value(Int64(1)))
    var negated = plan.exprs.unary(UnaryOp.NEG, a)
    var summed = plan.exprs.binary(BinaryOp.ADD, negated, one)
    var wide = plan.exprs.cast(LogicalType.FLOAT64, summed)
    var called = plan.exprs.call("round", [wide, one], True)
    var folded = plan.exprs.aggregate(AggKind.MAX, called)
    var chosen = plan.exprs.conditional(
        plan.exprs.binary(BinaryOp.GT, a, one), a, one
    )
    var running = plan.exprs.window(AggKind.SUM, a, [chosen], [one])
    var at = plan.project(t, [folded, running], ["most", "running"])
    var back = _trip(plan, at)
    assert_equal(
        render_expr(back.plan.exprs, back.plan.nodes[back.root].exprs[0]),
        render_expr(plan.exprs, folded),
        "the nested one reads the same",
    )


def test_a_call_says_whether_it_is_rowwise() raises:
    # The one thing about an expression that is a fact about the function rather
    # than about the tree, so it has to be written rather than worked out.
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var a = plan.exprs.column("a")
    var at = plan.project(
        t, [plan.exprs.call("cumsum", [a], False)], ["running"]
    )
    assert_true(
        to_json(plan, at).find('"rowwise": false') != -1, "and it says false"
    )
    var back = _trip(plan, at)
    assert_true(
        not back.plan.exprs.nodes[back.plan.nodes[back.root].exprs[0]].rowwise,
        "and it is still false",
    )


def test_a_window_keeps_its_partition_keys_apart_from_its_order_keys() raises:
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var a = plan.exprs.column("a")
    var b = plan.exprs.column("b")
    var c = plan.exprs.column("c")
    var at = plan.project(
        t, [plan.exprs.window(AggKind.SUM, a, [b], [c])], ["running"]
    )
    var back = _trip(plan, at)
    assert_equal(
        back.plan.exprs.nodes[back.plan.nodes[back.root].exprs[0]].parts,
        1,
        "one partition key and one order key",
    )


def test_a_cast_says_the_type_it_is_casting_to() raises:
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var at = plan.project(
        t, [plan.exprs.cast(LogicalType.INT32, plan.exprs.column("a"))], ["a"]
    )
    assert_true(
        to_json(plan, at).find('"to": "int32"') != -1,
        "the target is in the JSON",
    )
    _ = _trip(plan, at)


def test_a_bound_plan_comes_back_bound() raises:
    var plan = Plan()
    var t = plan.scan("t", ["a", "b"], 0)
    var a = plan.exprs.column("a")
    var one = plan.exprs.literal(Value(Int64(1)))
    var at = plan.filter(t, plan.exprs.binary(BinaryOp.GT, a, one))
    _ = bind(plan, at, [_table()])
    var text = to_json(plan, at)
    assert_true(text.find('"at": 0') != -1, "the position is written")
    assert_true(text.find('"type": "bool"') != -1, "and so is the type")
    var back = _trip(plan, at)
    var predicate = back.plan.nodes[back.root].exprs[0]
    assert_equal(
        back.plan.exprs.nodes[predicate].type,
        LogicalType.BOOL,
        "and the predicate is still a bool",
    )
    assert_equal(
        back.plan.exprs.nodes[back.plan.exprs.nodes[predicate].children[0]].at,
        0,
        "and the column is still bound to its position",
    )


def test_an_unbound_plan_comes_back_unbound() raises:
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var at = plan.project(t, [plan.exprs.column("a")], ["a"])
    var text = to_json(plan, at)
    assert_equal(
        text.find('"at"'), -1, "nothing is bound, so nothing says where"
    )
    var back = _trip(plan, at)
    assert_equal(
        back.plan.exprs.nodes[back.plan.nodes[back.root].exprs[0]].at,
        UNBOUND,
        "and it comes back unbound",
    )


def test_a_shared_expression_is_written_once_and_referred_to_after() raises:
    # What `cse` leaves behind. A tree form would copy it and the plan that came
    # back would compute it twice.
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var a = plan.exprs.column("a")
    var one = plan.exprs.literal(Value(Int64(1)))
    var shared = plan.exprs.binary(BinaryOp.ADD, a, one)
    var at = plan.project(
        t, [shared, plan.exprs.binary(BinaryOp.MUL, shared, shared)], ["x", "y"]
    )
    var text = to_json(plan, at)
    assert_true(text.find('"id": ') != -1, "the first write names it")
    assert_true(text.find('{"ref": ') != -1, "and the rest point at it")
    var back = _trip(plan, at)
    ref outputs = back.plan.nodes[back.root].exprs
    ref product = back.plan.exprs.nodes[outputs[1]]
    assert_equal(
        product.children[0],
        product.children[1],
        "both sides are the one expression",
    )
    assert_equal(
        outputs[0], product.children[0], "and so is the column beside it"
    )


def test_a_shared_node_is_written_once_and_referred_to_after() raises:
    # What `subplan` leaves behind, and the reason a plan is not always a tree.
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var a = plan.exprs.column("a")
    var one = plan.exprs.literal(Value(Int64(1)))
    var kept = plan.filter(t, plan.exprs.binary(BinaryOp.GT, a, one))
    var at = plan.union([kept, kept], True)
    var back = _trip(plan, at)
    assert_equal(
        back.plan.nodes[back.root].inputs[0],
        back.plan.nodes[back.root].inputs[1],
        "both arms are the one node",
    )


def test_a_plan_with_no_sharing_has_no_id_and_no_ref_in_it() raises:
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    var at = plan.project(t, [plan.exprs.column("a")], ["a"])
    var text = to_json(plan, at)
    assert_equal(text.find('"id"'), -1, "nothing needs naming")
    assert_equal(text.find('"ref"'), -1, "and nothing points anywhere")


def test_only_what_the_root_reaches_is_written() raises:
    # A pass leaves the node it replaced sitting in the arena, and a plan is
    # what its root reaches, exactly as `explain` has it.
    var plan = Plan()
    var t = plan.scan("t", List[String](), 0)
    _ = plan.limit(t, 0, 3)
    var at = plan.project(t, [plan.exprs.column("a")], ["a"])
    assert_equal(
        to_json(plan, at).find("limit"), -1, "the leftover is not the plan"
    )
    var back = _trip(plan, at)
    assert_equal(
        len(back.plan), 2, "two nodes were reachable and two came back"
    )


def test_a_null_constant_keeps_the_type_it_would_have_had() raises:
    var back = _constant(Value(null=LogicalType.INT64))
    assert_true(back.is_null(), "it is still absent")
    assert_equal(back.type, LogicalType.INT64, "and still an int64")


def test_a_bool_constant_goes_out_as_a_bool() raises:
    var back = _constant(Value(True))
    assert_equal(back.type, LogicalType.BOOL, "still a bool")
    assert_true(back.bits != 0, "and still true")


def test_a_text_constant_keeps_its_quotes_and_its_backslashes() raises:
    var back = _constant(Value(String('she said "a\\b" and a tab\there')))
    assert_equal(
        back.as_string(),
        'she said "a\\b" and a tab\there',
        "every byte comes back",
    )


def test_a_text_constant_keeps_what_is_not_ascii() raises:
    var back = _constant(Value(String("a piece of cake and a piece of pi")))
    assert_equal(
        back.as_string(), "a piece of cake and a piece of pi", "unchanged"
    )
    var other = _constant(Value(String("naïve, résumé, 東京")))
    assert_equal(
        other.as_string(), "naïve, résumé, 東京", "and so does the rest of it"
    )


def test_an_unsigned_constant_above_the_top_of_int64_comes_back_itself() raises:
    # `Value.write_to` prints every integer through int64 and this one prints as
    # minus one there, which is fine for a frame and wrong for a document.
    var back = _constant(Value(UInt64(18446744073709551615)))
    assert_equal(back.type, LogicalType.UINT64, "still a uint64")
    assert_equal(
        back.bits, UInt64(18446744073709551615), "and still the top of it"
    )


def test_a_negative_constant_of_a_narrow_type_comes_back_itself() raises:
    var back = _constant(Value(Int8(-5)))
    assert_equal(back.type, LogicalType.INT8, "still an int8")
    assert_equal(back.as_scalar[DType.int64](), -5, "and still minus five")


def test_a_float_constant_comes_back_to_the_last_bit() raises:
    var third = Float64(1) / Float64(3)
    var back = _constant(Value(third))
    assert_equal(back.real, third, "the same float, not one near it")
    var small = _constant(Value(Float32(0.1)))
    assert_equal(small.type, LogicalType.FLOAT32, "and a float32 stays narrow")


def test_the_three_floats_json_has_no_syntax_for_go_out_as_words() raises:
    var plan = Plan()
    var at = plan.values(
        [plan.exprs.literal(Value(Float64(1) / Float64(0)))], ["x"]
    )
    assert_true(to_json(plan, at).find('"inf"') != -1, "an infinity is a word")
    assert_true(
        isinf(_constant(Value(Float64(1) / Float64(0))).real),
        "and comes back one",
    )
    assert_true(
        isinf(_constant(Value(Float64(-1) / Float64(0))).real),
        "and so does the other one",
    )
    assert_true(
        isnan(_constant(Value(Float64(0) / Float64(0))).real),
        "and so does a nan",
    )


def test_a_date_constant_stays_a_date() raises:
    var value = Value(Int32(8766))
    value.type = LogicalType.DATE32
    var back = _constant(value^)
    assert_equal(back.type, LogicalType.DATE32, "still a date")
    assert_equal(back.as_scalar[DType.int64](), 8766, "and still that day")


def test_a_timestamp_constant_keeps_its_unit_and_its_zone() raises:
    var back = _constant(
        Value.timestamp(1700000000000000000, TimeUnit.NANO, TimeZone("UTC"))
    )
    assert_equal(
        back.type,
        LogicalType.timestamp(TimeUnit.NANO, TimeZone("UTC")),
        "the unit and the zone both came back",
    )
    assert_equal(
        back.as_scalar[DType.int64](),
        1700000000000000000,
        "and so did the count",
    )


def test_a_duration_constant_keeps_its_unit() raises:
    var back = _constant(Value.duration(-90, TimeUnit.SECOND))
    assert_equal(
        back.type,
        LogicalType.duration(TimeUnit.SECOND),
        "still a duration in seconds",
    )
    assert_equal(
        back.as_scalar[DType.int64](), -90, "and still ninety of them back"
    )


def test_a_weak_constant_is_still_weak_when_it_comes_back() raises:
    # A Python 2 has no width of its own and takes one from the column it meets,
    # which is a fact about where it came from and is lost if it is not written.
    var back = _constant(Value(Int64(2)).weakened())
    assert_true(back.weak, "it arrived without a dtype and still says so")


def test_every_aggregate_is_named_and_reads_back() raises:
    # The reader's table and `AggKind.write_to` are two lists that have to
    # agree, so the test walks every code rather than a sample of them.
    for code in range(Int(AggKind.SKEW.code) + 1):
        var plan = Plan()
        var t = plan.scan("t", List[String](), 0)
        var fold = plan.exprs.aggregate(
            AggKind(UInt8(code)), plan.exprs.column("a")
        )
        var at = plan.aggregate(t, List[Int](), [fold], ["x"])
        var back = _trip(plan, at)
        assert_equal(
            back.plan.exprs.nodes[back.plan.nodes[back.root].exprs[0]].op,
            code,
            String("aggregate ", code, " came back as itself"),
        )


def test_every_binary_operator_is_written_and_reads_back() raises:
    for code in range(Int(BinaryOp.GE.code) + 1):
        var plan = Plan()
        var t = plan.scan("t", List[String](), 0)
        var a = plan.exprs.column("a")
        var one = plan.exprs.literal(Value(Int64(1)))
        var at = plan.project(
            t, [plan.exprs.binary(BinaryOp(UInt8(code)), a, one)], ["x"]
        )
        var back = _trip(plan, at)
        assert_equal(
            back.plan.exprs.nodes[back.plan.nodes[back.root].exprs[0]].op,
            code,
            String("operator ", code, " came back as itself"),
        )


def test_every_unary_operator_is_written_and_reads_back() raises:
    for code in range(UnaryOp.INVERT.code + 1):
        var plan = Plan()
        var t = plan.scan("t", List[String](), 0)
        var at = plan.project(
            t, [plan.exprs.unary(UnaryOp(code), plan.exprs.column("a"))], ["x"]
        )
        var back = _trip(plan, at)
        assert_equal(
            back.plan.exprs.nodes[back.plan.nodes[back.root].exprs[0]].op,
            code,
            String("operator ", code, " came back as itself"),
        )


def test_every_join_kind_is_named_and_reads_back() raises:
    for code in range(Int(JoinKind.CROSS.code) + 1):
        var plan = Plan()
        var left = plan.scan("t", List[String](), 0)
        var right = plan.scan("u", List[String](), 1)
        var at = plan.join(
            left,
            right,
            [plan.exprs.column("a")],
            [plan.exprs.column("k")],
            JoinKind(UInt8(code)),
        )
        var back = _trip(plan, at)
        assert_equal(
            back.plan.nodes[back.root].op,
            code,
            String("join ", code, " came back as itself"),
        )


def test_a_plan_can_be_written_by_hand() raises:
    # The whole point of the form. Nothing here came out of the writer.
    var back = from_json(
        '{"kind": "filter",'
        ' "predicate": {"kind": "binary", "op": ">",'
        '   "left": {"kind": "column", "name": "a"},'
        '   "right": {"kind": "literal", "type": "int64", "value": 1}},'
        ' "input": {"kind": "scan", "source": "t", "table": 0,'
        '   "columns": ["a"]}}'
    )
    assert_equal(
        explain(back.plan, back.root),
        "FILTER a > 1\n  SCAN t [a]\n",
        "and it reads as the plan it says",
    )


def test_whitespace_and_member_order_do_not_matter() raises:
    var back = from_json(
        """
        {
            "input": { "kind": "scan", "columns": [], "table": 0,
                       "source": "t" },
            "kind": "limit",
            "length": 3,
            "offset": 1
        }
        """
    )
    assert_equal(
        explain(back.plan, back.root),
        "LIMIT 3 offset 1\n  SCAN t []\n",
        "a document is its members, not their order",
    )


def test_a_sort_key_written_with_no_direction_sorts_upwards() raises:
    # The two flags are what a person writing a plan by hand forgets first, and
    # ascending with the nulls first is what a bare key means everywhere else.
    var back = from_json(
        '{"kind": "sort", "keys": [{"expr": {"kind": "column", "name": "a"}}],'
        ' "input": {"kind": "scan", "source": "t", "table": 0, "columns": []}}'
    )
    assert_equal(
        explain(back.plan, back.root),
        "SORT [a asc]\n  SCAN t []\n",
        "up, with the nulls at the front",
    )


def test_a_distinct_written_with_no_keys_is_over_the_whole_row() raises:
    var back = from_json(
        '{"kind": "distinct",'
        ' "input": {"kind": "scan", "source": "t", "table": 0, "columns": []}}'
    )
    assert_equal(
        explain(back.plan, back.root),
        "DISTINCT [*]\n  SCAN t []\n",
        "no keys means all of them",
    )


def test_a_kind_nobody_has_written_is_refused_by_name() raises:
    with assert_raises(contains="no plan node is a pivot"):
        _ = from_json('{"kind": "pivot", "input": {"kind": "scan"}}')


def test_an_expression_kind_nobody_has_written_is_refused_by_name() raises:
    with assert_raises(contains="no expression is a lambda"):
        _ = from_json(
            '{"kind": "filter", "predicate": {"kind": "lambda"},'
            ' "input": {"kind": "scan", "source": "t", "table": 0,'
            '   "columns": []}}'
        )


def test_a_node_missing_something_it_needs_says_what_is_missing() raises:
    with assert_raises(contains="a scan has no source"):
        _ = from_json('{"kind": "scan", "table": 0, "columns": []}')


def test_a_document_that_is_not_a_plan_is_refused() raises:
    with assert_raises(contains="a plan node is written as an object"):
        _ = from_json("[1, 2, 3]")


def test_a_ref_to_nothing_is_refused() raises:
    with assert_raises(contains="nothing was written with that id before it"):
        _ = from_json(
            '{"kind": "union", "all": true, "inputs": [{"ref": 4},'
            ' {"kind": "scan", "source": "t", "table": 0, "columns": []}]}'
        )


def test_a_type_nobody_can_name_is_refused_by_name() raises:
    with assert_raises(contains="no type is named int65"):
        _ = from_json(
            '{"kind": "values", "columns": ["x"],'
            ' "rows": [[{"kind": "literal", "type": "int65", "value": 1}]]}'
        )


def test_a_constant_that_does_not_fit_its_type_is_refused() raises:
    with assert_raises(contains="a plan will not hold a number it cannot hold"):
        _ = from_json(
            '{"kind": "values", "columns": ["x"],'
            ' "rows": [[{"kind": "literal", "type": "int8", "value": 300}]]}'
        )


def test_a_constant_written_as_the_wrong_shape_is_refused() raises:
    with assert_raises(contains="is written as a number"):
        _ = from_json(
            '{"kind": "values", "columns": ["x"],'
            ' "rows": [[{"kind": "literal", "type": "int64", "value": "1"}]]}'
        )


def test_a_plan_the_builders_refuse_is_refused_here_too() raises:
    # Reading goes through the builders, so a document describing an impossible
    # plan fails where a caller building the same plan in code would fail, with
    # the same message and without a half built plan coming back.
    with assert_raises(contains="a difference is between two inputs"):
        _ = from_json(
            '{"kind": "except", "all": false, "inputs": ['
            ' {"kind": "scan", "source": "t", "table": 0, "columns": []},'
            ' {"kind": "scan", "source": "u", "table": 1, "columns": []},'
            ' {"kind": "scan", "source": "v", "table": 2, "columns": []}]}'
        )


def test_a_values_of_ragged_rows_is_refused_by_the_builder_too() raises:
    with assert_raises(contains="cannot be made of"):
        _ = from_json(
            '{"kind": "values", "columns": ["x", "y"], "rows": ['
            ' [{"kind": "literal", "type": "int64", "value": 1},'
            '  {"kind": "literal", "type": "int64", "value": 2}],'
            ' [{"kind": "literal", "type": "int64", "value": 3}]]}'
        )


def test_a_list_type_does_not_go_out_at_all() raises:
    # Its element type is on the column rather than on the type, so writing the
    # name and reading it back would be inventing one.
    var plan = Plan()
    var value = Value(Int64(1))
    value.type = LogicalType.list_of(DType.int32)
    var at = plan.values([plan.exprs.literal(value^)], ["x"])
    with assert_raises(contains="does not say its element type"):
        _ = to_json(plan, at)


def test_a_root_that_is_not_a_node_is_refused() raises:
    var plan = Plan()
    _ = plan.scan("t", List[String](), 0)
    with assert_raises(contains="plan node 7 is not in a plan of 1"):
        _ = to_json(plan, 7)


def test_the_names_a_scan_reads_survive_being_odd() raises:
    var plan = Plan()
    var at = plan.scan("a table with spaces", ['a "quoted" name', "a\nline"], 0)
    var back = _trip(plan, at)
    assert_equal(
        back.plan.nodes[back.root].names[1],
        "a\nline",
        "a newline in a name is a name",
    )
    assert_equal(
        back.plan.nodes[back.root].source,
        "a table with spaces",
        "and so is a source with spaces in it",
    )


def test_a_q19_shaped_plan_survives_the_trip() raises:
    # The one test here that is a plan somebody would actually run: a scan under
    # a filter under an aggregate, with a nested predicate over it.
    var plan = Plan()
    var t = plan.scan("lineitem", ["l_quantity", "l_shipmode"], 0)
    var quantity = plan.exprs.column("l_quantity")
    var mode = plan.exprs.column("l_shipmode")
    var low = plan.exprs.literal(Value(Float64(1)))
    var high = plan.exprs.literal(Value(Float64(11)))
    var air = plan.exprs.literal(Value(String("AIR")))
    var band = plan.exprs.binary(BinaryOp.GE, quantity, low)
    var under = plan.exprs.binary(BinaryOp.LE, quantity, high)
    var flown = plan.exprs.binary(BinaryOp.EQ, mode, air)
    var both = plan.exprs.call("and", [band, under, flown], True)
    var kept = plan.filter(t, both)
    var at = plan.aggregate(
        kept,
        List[Int](),
        [plan.exprs.aggregate(AggKind.SUM, quantity)],
        ["revenue"],
    )
    _ = _trip(plan, at)


def test_a_column_that_says_which_input_it_is_from_keeps_saying_it() raises:
    # A pinned column is neither bound nor entirely unbound, so it is the one
    # case where the position is missing and the input is not, and the writer
    # has to keep those two apart rather than writing both or neither.
    var plan = Plan()
    var t = plan.scan("t", ["a", "b"], 0)
    var said = plan.exprs.column_of(1, "a")
    var root = plan.project(t, [said], ["a"])
    var text = to_json(plan, root)
    assert_true(text.find('"table": 1') != -1, "the input is written")
    assert_true(text.find('"at":') == -1, "and the position is not")
    var back = _trip(plan, root)
    ref out = back.plan.nodes[back.root].exprs
    assert_equal(back.plan.exprs.nodes[out[0]].table, 1, "the input came back")
    assert_equal(
        back.plan.exprs.nodes[out[0]].at, UNBOUND, "and is still not bound"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
