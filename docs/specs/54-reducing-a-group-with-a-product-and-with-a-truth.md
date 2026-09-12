# Reducing a group with a product and with a truth

## 1. Three names the boundary already knew

Document 52 put `prod`, `any` and `all` on a column and on a frame. It did not put them on a group, and it said so: the grouped forms were named as left over in that document, in the progress note that followed it, and again in the slice after that. This is the slice that writes them.

The odd part of starting was how little there was to do at the boundary. `firepanda/py/reduce.mojo` already mapped all three words onto `AggKind.PROD`, `AggKind.ANY` and `AggKind.ALL`, and `grouped_reduction` already delegated to the same table the whole column path reads, so `df.groupby("k").prod()` crossed into the core with the right tag on it and got as far as the dispatch chain in `firepanda/kernel/group.mojo` before anything went wrong. What it hit there was the fall through at the end of `_dispatch_core`, whose message is exactly the situation it was written for: this reduction has a whole column implementation and no grouped one yet, so it reaches this chain only by being asked for per group. The type tables were ready too. `AggKind.result_dtype` already said that a product answers the accumulator dtype and a truth answers bool, and `agg_type` already said the same thing about the logical type, because both were written from the whole column slice and neither was told which paths would use them.

So the slice is three loops and the plumbing around them, and the interesting content is entirely in what the loops cannot copy from the loops beside them.

## 2. A null holds a zero, which a grouped sum spends and a grouped product cannot

Document 52 section 2 makes this argument for the whole column reductions and every word of it is true one group at a time. The invariant at the top of `firepanda/kernel/__init__.mojo` is that a null value is a zero in the values buffer, and `_sum_core` spends that invariant deliberately: it reads no bitmap at all, because a null adds nothing and a NaN is turned into a zero by `_addend` so that it adds nothing either. That is not a micro optimization, it is the reason a grouped sum is the fastest thing in the file, and the docstring says as much.

Zero is the identity for addition and it is the one number a product must never see. A grouped product written on the sum's loop would report zero for every group that happened to contain a missing value, which is a wrong answer that looks like a real one. So `_prod_core` reads the bitmap, through the same `_there` helper that `_extreme_core` and every other core in the file uses, and `_factor` is `_addend` with one for the identity instead of zero and a validity read in front of it. The two helpers sit next to each other on purpose, because the difference between them is the whole of what this section says.

`_prod_core` also leaves out the partitioned route that `_sum_core` has. That route exists because a grouped sum turned up in a profile with more groups than cache and somebody measured it, and until a grouped product does the same thing, writing a second copy of the partitioning would be adding a path nothing has exercised. It leaves out `as_float` for a different reason: the float accumulator exists so that `_mean_core` can build an average out of a sum without inheriting the wrap that pandas requires of the sum itself, and nothing is built out of a product the way a mean is built out of a sum. A grouped product over int64 accumulates in int64 and wraps, which is what the whole column product does and what pandas does.

## 3. A product over nothing is one, so there is no seen table

`_extreme_core` is the other core that reads validity, and at a glance the product should be a copy of it. It is not, and the difference is the most useful thing in this document.

A grouped minimum starts every accumulator at the identity for the comparison, which is the largest value of the dtype. That identity is a real value that a column could genuinely hold, so a group that no row ever reached is indistinguishable from a group whose only value happened to be that number, and the answer has to be overwritten at the end. That is what the `seen` table is: a byte per group per worker, merged alongside the values, purely so the final pass knows which slots to replace with a NaN or with a cleared validity bit. On a hundred thousand groups over thirty two workers it is three million bytes of bookkeeping that exist only to record absence.

A product carries none of that, because the identity of multiplication is the answer. A product over no values is one, in arithmetic and in pandas, so the slot a fill wrote is already correct for a group nothing reached. `_ones` writes the fill, every group comes out present, and there is no seen table, no final pass and no NaN rule. The same argument was made for the whole column product in document 52 section 2 and it is stronger here, because here it removes a per group per worker allocation rather than a single flag.

## 4. `any` and `all` are one body, and the store is conditional

`_truth_core` takes `want_all` as a parameter rather than an argument, the way `truth_over` does in `firepanda/kernel/agg.mojo` and the way `_extreme_core` takes `want_min`, so the branch is gone before anything is compiled. The identity is the parameter itself, which is the same coincidence the whole column version uses: an `any` over no values is False and an `all` over no values is True, so `_fill_truth` writing `want_all` into every slot both starts the fold and settles every group the fold never reaches. Like the product and unlike the extremes, every group comes out present and nothing has to be fixed up at the end.

What is different here from every other core in the file is the store. A sum reads its slot, adds, and writes it back, once per row, and the dependent store is most of what a scatter costs. A truth only writes when it would change the slot, so an `any` writes on the rows that are true and an `all` writes on the rows that are false. On the data either question is usually asked about, which is a column that is mostly one way, the loop degenerates to a read of the value and a branch that is almost never taken. The merge is the same operator a third time, an or or an and over two contiguous arrays, sixty four groups to an instruction because a bool is stored a byte wide here.

The whole column version can stop as soon as the answer cannot change and this one cannot, because the answer it is computing is one per group and settling group three says nothing about group four. That is the same asymmetry `_extreme_core` has against `extreme_over` and it is not worth a branch: checking whether every group has settled costs a pass over the table per row.

## 5. Where the truth rule lives, and why it lost its underscore

What counts as true is numpy's rule and therefore pandas': anything that is not zero, with booleans the same rule said in one fewer step. That rule was written for the whole column path as `_truthy` in `firepanda/kernel/agg.mojo` and the grouped core needs exactly it.

It is now `truthy`, with no underscore, and read from both places rather than copied into the second one. The reason is the one `temporal_agg_type` gives for being the whole table in one place: a whole column `any` and a grouped `any` disagreeing about what counts as true would be a difference nobody could explain and nobody would find, because each half would be self consistent and the tests for each half would pass. One function read from both sides cannot drift.

The rule about a NaN comes along with it, and document 52 section 4 explains why that rule is not the obvious one. A NaN is not equal to zero, so a truth loop written about zero alone reports that a column of nothing but gaps holds something true. Here the NaN never reaches `truthy` at all, because `_there` steps over it before the value is read, which is the same thing said in the place the grouped cores already say it.

## 6. Text has a truth value and no product

`any` and `all` are the only two of the eighteen grouped reductions that read a column of words and answer a number, so they are the only two that can go over a whole frame without the text columns being taken out of it first. That matters more than it sounds: a frame of a key, a number and a label is the ordinary shape of a frame, and a reduction that refuses it is a reduction nobody can call on their own data.

`aggregate_group_strings` gained a branch for the pair and `_text_truth_per_group` is the loop, which is `text_truth` one group at a time. A string is true when it is not empty, which is Python's rule and the one pandas keeps, and `text_truth` explains why the empty string is worth a function: a dataset that spells its missing text as an empty string gets a different answer from one that spells it as a null, and both answers are right. The loop is serial. The parallel route would work unchanged, since the accumulator is a byte per group either way, and it has not been written for the reason `_text_rows_per_group` gives for its own serial ceiling, which is that nothing has measured a query that wants it.

A product over text is refused and the message names the column type. pandas refuses it too, with a `TypeError` rather than with an answer, so the two libraries agree that this is not an operation and disagree only about which exception says so.

## 7. What pandas answers, measured rather than remembered

All of this is pandas 3.0.5 and all of it was run.

`DataFrameGroupBy.prod` is `prod(numeric_only=False, min_count=0, skipna=True)` and it takes no `engine` or `engine_kwargs`, which `DataFrameGroupBy.sum` does take. There is no reason for that difference, it is simply what the two signatures say, and writing them as one shape in `tools/bindings.py` is a mismatch the conformance board would report. `DataFrameGroupBy.any` and `DataFrameGroupBy.all` are `any(skipna=True)` and `all(skipna=True)`, with no `numeric_only` on either, which follows from the section above: a truth is a question a text column answers as readily as a number, so there is nothing for the flag to drop. `SeriesGroupBy` has the identical three signatures.

A group of nothing but missing values products to one, answers False to `any` and True to `all`, and every one of those rows is present rather than missing. `min_count=1` turns the product of that group into a missing value, which is the same rule the whole column product has and is the reason `min_count` defaults to zero for `prod` as it does for `sum` and to minus one for the four that pick a value out rather than combining values. `skipna=False` on a grouped product behaves as `min_count=1` does, which is not the behaviour the whole column version has and is not something firepanda implements either way.

A grouped product of a nullable integer column answers that nullable integer type, a float column answers float64, and a boolean column answers `Int64`, which is the same widening a grouped sum does. A grouped truth over a nullable integer column answers `BooleanDtype` and over a plain float column answers the numpy `bool`, which is the nullable dtype distinction that document 53 ran into from the other direction and is not a difference this slice creates or closes.

## 8. What is left

The eighteenth, nineteenth and twentieth rows of the grouped table are `corr` and `cov`, which read a pair of columns, and `size`, which reads none, so the grouped reduction list is now the whole of what `AggKind` has for a single column and the dispatch chain has no fall through case that a user can reach by spelling a pandas method name.

Three things were seen from here and not done here. `min_count` is still refused for anything but its default on every grouped reduction that has it, which is a check on the count after the reduction and the count is not kept. `skipna=False` is still refused everywhere, which is a second pass the kernels do not make. And the grouped string path answers a `ColumnNotFoundError` where pandas answers a `TypeError`, for a product over words and for every other reduction that text cannot take, which is a tagging question about the whole string path rather than anything this slice introduced.
