# Multiplying and asking whether a column is true

## 1. Four names that look like one name already here

`sum`, `min`, `max`, `mean` and `count` have been on both classes since the first reduction slice. `prod`, `product`, `any` and `all` are the four that were left, and at a glance they are the same shape: read a column, fold it to one value, answer that value. The dispatch table has a row per name, the kernel has a loop per name, and adding four rows and two loops should have been an afternoon.

It was not, and the reason is worth writing down because it is the kind of thing that only bites once and then never announces itself again. Two of the four cannot borrow the loop that `sum` uses, and the thing that stops them is an invariant this library depends on everywhere else.

## 2. A null holds a zero, which is why the sum is fast and the product is not

`firepanda/kernel/__init__.mojo` states the invariant at the top of the package: a null value is a zero in the values buffer. `Array.set_null` zeroes the slot it clears, `Array.slice` copies the zeros along, and a fresh allocation starts zeroed. That is what lets `sum_over` add straight down a column without ever reading the validity bitmap, and on a million rows it is the difference between 99 microseconds and 471.

Zero is the identity for addition. It is not the identity for multiplication. A product written the way the sum is written would multiply the zeros in too, so a column of prices with one gap in it would report a total of nothing. The bug would not be subtle in a test and it would be invisible in production, because zero is a number a product is allowed to be.

So `prod_over` takes the validity bitmap and is modelled on `extreme_over` rather than on `sum_over`: one path per validity word, a word of nothing but nulls skipped whole, a word of nothing but values pushed through the vector unit with no bit test, and a mixed word falling back to a test per value. It is slower than the sum and it has to be. `tests/test_reduce.mojo` has a test named for the mistake, `test_a_product_multiplies_only_the_values_that_are_there`, which multiplies a column holding two nulls and asserts 1296. A product that read the zeros would answer zero and the test exists to say so out loud.

The one thing `prod_over` does not need, which surprised the implementation, is the `seen` flag the extremes carry. `min_of` has to report that it found nothing, because there is no number that means "no minimum". A product over nothing is one, and one is also pandas' answer, so the identity carries the empty case for free.

## 3. `any` and `all` are one body with a compile time branch

They ask the same question and differ only in what they do with the answer per row and what they return when they run out of rows. `truth_over` takes `want_all` as a parameter rather than an argument, so the branch is gone by the time anything is compiled, and the final `return want_all` is both the empty case and the fallthrough: an `any` that found nothing is False, an `all` that found nothing is True.

The vector path builds a mask of which lanes are true and a mask of which lanes are real, ands them together, and then asks one question of the result. For `all` it asks whether any real lane is false and returns False if so. For `any` it asks whether any lane is true and returns True if so. Both are a `reduce_or` over a boolean vector, which is a movemask and a compare.

Booleans skip the vector unit entirely, the same way `_extreme_range` skips it for booleans. A one byte dtype gets a very wide vector and the loads dominate, and a column of bools short circuits on the first row that settles it far more often than a column of numbers does.

## 4. A NaN is missing rather than true

This is the other thing that does not fall out of the obvious loop. `x != 0` is True for a NaN, because a NaN is not equal to anything including zero. A truth loop that only asked about zero would report that a column of nothing but NaNs has something true in it, and pandas answers False.

pandas treats a NaN as missing here exactly as it does in `count` and `mean`, so `truth_over` folds an `isnan` into the lane mask on a float dtype and skips the value on the scalar paths. `test_a_nan_is_missing_rather_than_true` is the guard and it is written against an all NaN column so that the wrong answer is the one it asserts against. The product does the same thing for the same reason, except that a NaN there becomes the identity rather than being dropped from a count, which is the same skip spelled for a different operator.

`_is_there` in `firepanda/kernel/scalar.mojo` already had this rule, so both twins got it without being told. That is the twin arrangement working the way document 09 says it should: the slow loop is written from the rule and the fast one is checked against it.

## 5. What pandas answers, measured rather than remembered

All of this is pandas 3.0.5 and all of it was run rather than recalled.

A present value is true when it is not zero. Missing values are skipped by default, so `any` over an empty or all missing column is False and `all` is True. `skipna=False` does not make a missing value false, which would be the guessable behaviour. It makes it truthy, so an all missing column answers True to both questions.

A string is true when it is non empty. `["a", "b"].any()` is True, `["", "b"].all()` is False, and an all null string column answers False to `any` and True to `all` like every other dtype. A product over strings is refused with `Cannot perform reduction 'prod' with string dtype`, and `numeric_only=True` does not rescue it.

`bool_only=True` on a frame keeps only the boolean columns. On a column pandas takes the argument, does nothing at all with it, and answers: `pd.Series([1, 0, 3]).any(bool_only=True)` is True. The layer here accepts it and ignores it for exactly that reason, and the docstring says so, because a layer that refused it would be stricter than the thing it is copying.

The product wraps rather than raising. `2 ** 40` multiplied by itself in int64 is zero and pandas reports zero. int8 widens to int64, uint8 to uint64, bool to int64. An empty or all missing product is 1.

## 6. Time has a truth value only when it is a length

`pd.to_timedelta([0, 5], unit="D").any()` is True and `.all()` is False. A datetime column refuses both: `datetime64 type does not support operation 'any'`.

The division is not arbitrary and the refusal here says why rather than naming the dtype. A length of time has a zero and a zero length is false, so the question has an answer. A point in time has no zero to be measured against, because whatever epoch it is counted from is a storage detail and not a property of the instant, so asking whether a timestamp is true is asking a question with no content. `temporal_agg_type` refuses it in those words.

A product over either is refused, and the sentence there is about units rather than about support: a product of two durations would be in units of time multiplied by itself, and there is no dtype to put such an answer in.

## 7. The bug this slice found, which was not in this slice

`axis=None` on a frame folds the whole frame to one scalar in pandas. `df.sum(axis=None)` is one number, not a column of per column sums. Here, `_axis_number` mapped `None` to the default axis without comment, so every reduction on a frame had been quietly answering a Series where pandas answers a scalar, for every name, since the first reduction slice.

The fix is not a blanket one, because the reductions do not all behave the same way under folding. Six of them can be built out of their own per column answers, because each is the same question asked again: the sum of the column sums is the sum of the frame, and so are the product, the minimum, the maximum, and both truth values. Those six run the reduction a second time over the Series they produced. `mean`, `median`, `std`, `var`, `sem` and `skew` cannot be built that way and are refused with a `NotImplementedError` that says why, because a mean of means is not a mean unless the columns are the same length and the missing values line up.

`count`, `quantile` and `nunique` are the third case. pandas itself refuses `axis=None` for those three with `ValueError: No axis named None for object type DataFrame`, so the layer here raises the same thing with the same words rather than inventing a third behaviour.

The whole frame product is worth one line on its own. For a frame holding an integer column and a float column, pandas answers `-0.0` rather than `0.0`, because the per column products are widened to float64 and one of them carries a sign. The implementation here lands on the same value without being told to, which is the check that it is folding rather than special casing.

## 8. Why `product` is not a sixteenth name in the kernel

pandas has `prod` and `product` and they are the same method under two spellings. The binding generator writes both, both carry the same signature, and both send the string `prod` across the boundary. `AggKind` therefore has three new codes and not four, and `firepanda/py/reduce.mojo` has three new rows in its dispatch table. The alternative, a fourth code that is a synonym for the third, would put a decision about Python spelling inside the kernel, which is the one place in this library that does not know Python exists.

## 9. What is not here

The grouped forms. `GroupBy.any`, `GroupBy.all` and `GroupBy.prod` are not implemented, and the three new codes reach the grouped dispatch chain only to be refused by name with a sentence saying they have a whole column implementation and no grouped one yet. That sentence is asserted by `test_the_three_new_reductions_have_no_grouped_form_yet`, so adding the grouped branch later means deleting a test, which is the right amount of friction for a dividing line that is meant to move.

`min_count` is accepted and held at its default on the product, the same as it is on the sum, because the machinery for reporting a missing value when too few rows were present belongs with the rest of the argument holding rather than in a reduction slice.

The frame fold for the six reductions that currently refuse `axis=None` is a real gap and it is named in section 7 so that the next slice does not have to rediscover it.
