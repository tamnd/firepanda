# 74. The mean of a set of points in time

## 1. One library, two answers

`df.group_by(["k"], [AggSpec("ts", AggKind.MEAN)])` answered a point in time. `SELECT avg(ts) FROM t GROUP BY k` answered a count of seconds. Same library, same column, same question, and which answer you got depended on which front door you walked through.

Neither half was wrong on its own terms and that is what kept it alive. The frame path runs the grouped kernel, which reads the pandas table in `temporal_agg_type` and puts the label back on the float the core computed. The SQL path runs the streaming group operator, which keeps a running sum and a running count and divides at the end, and a division of two numbers is a number. The operator's schema said float64 and the operator produced float64, so nothing inside it disagreed with anything else inside it. The disagreement was between two files that never call each other.

A divergence of this shape is the worst kind to find, because every test of either half passes. It was found by asking the question that neither half asks itself, which is whether the two agree.

## 2. Why the schema function was the hole

`agg_type` is the function that says what type a reduction produces, and it is read by the plan binder before any row moves and by both operators when they report their output schema. It had a paragraph admitting it did not handle a column of times, and the paragraph was honest about why: the two paths disagreed, and reconciling them was a change rather than a line.

So the hole was documented and then declared. The function returned float64 for a mean over instants, which is not a description of what either path produced, since one of them produced a timestamp. A schema that is not the type of the data under it is the one thing that function exists to prevent, and it was breaking its own rule for one family of input types.

It now hands a temporal input straight to `temporal_agg_type`, which is the table both kernels already read. That is the same argument document 54 section 5 makes for `truthy` being read from both places rather than copied into the second one: one function read from both sides cannot drift, and two tables that agree today are two tables that will disagree eventually, silently, with each half self consistent and each half's tests passing.

## 3. What comes with the table

The table is not only a mapping from reduction to type. It is also the list of reductions that have no answer, and reading it means inheriting the refusals.

A sum over instants is refused, because adding two points in time has no answer and a total is additions in a row. A variance is refused, because the answer would be in units of time squared and there is no dtype to hold one. Those refusals were already there in the kernels. What changed is that the operator now makes them at bind time rather than declaring a float64 for a call that raises once the first chunk arrives.

`SELECT sum(ts) FROM t` used to report the error about a sum, but it reported it from inside a mean that the user never wrote, because the operator split every mean into a sum and a count and the sum went past the same table. The error named the right problem for the wrong query. Now the sum refuses and says so about itself, and the mean does not go anywhere near a sum.

## 4. The mean that cannot fold

A reduction folds when the answer over two pieces can be recovered from the answers over each piece. A mean folds, but not as a mean: the operator keeps a sum and a count and divides once at the end. That is what lets a mean cross a chunk boundary without becoming a mean of means.

For a column of instants that state cannot exist. The numerator of the division is a sum of points in time, and that is a value this library refuses to produce by name. There is no labelling that fixes it either, because the refusal is about what the number means rather than about what it is stored in, and a running state that is only valid if nobody ever looks at it is not a design worth having.

So a mean over a column of times does not fold. The operator holds the column and calls the whole frame kernel once at the end, which is the route a median and a distinct count already take for their own reasons. The route was already there and the reduction simply joins it.

The result is stronger than making the two paths agree by writing the same rule twice. They agree because they are the same call. `Group` holds the values and hands them to `aggregate_group_any`, which is the exact function `DataFrame.group_by` calls, and `Reduce` hands them to `reduce_any`, which is the exact function `DataFrame.agg` calls. There is no second implementation left to drift.

What it costs is memory: a mean of times holds the column where a mean of numbers holds two numbers per group. That is what a median on the same column already costs, and a mean of times is a rare query next to a mean of numbers, so the cost lands where it is least often paid.

## 5. The label a fold loses

The reductions that do still fold over a column of times needed a second, smaller fix.

A running slot is an accumulator. It gets widened when the group count grows and settled back down at the end, and both of those are written against the physical dtype, because that is the only thing an accumulator is. So a total of lengths of time went in as a duration and came out of the fold as the int64 it is counted in. The numbers were right and the label was gone, and the schema the node had already reported said duration.

The fix is to put the declared type back on at the point the state becomes output, and only there. Nothing happens unless the declared type is temporal and the answer is not already carrying it, so a reduction over a column of numbers pays one comparison, and a minimum that kept its label the whole way through is left alone rather than relabelled to what it already says.

## 6. The one flag that is about the node rather than the expression

`temporal_agg_type` takes a `whole_column` argument, because pandas refuses a standard error over a whole column of times and answers one for a group, on the same column. firepanda copies both halves, for the reason document 09 section 1 gives: a user comparing the two libraries is comparing whichever spelling they wrote.

Which means `agg_type` needs the same argument, and the plan binder has to know something about the node rather than about the expression it is binding. It is the one place an aggregate's type depends on its surroundings. An aggregate node with no group key reduces the whole column, and the key count is already on the node, so the binder reads it and passes it down. A window is grouped by definition, since a partition is a group, so it passes false and does not have to think about it.
