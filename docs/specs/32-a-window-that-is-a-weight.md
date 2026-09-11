# 32. A window that is a weight

Document 31 is about a pair of row numbers, because that is what `rolling` and `expanding` are for, and it says so in its first paragraph. `ExponentialMovingWindow` is pandas' third window type and it has no pair of row numbers. Every row before this one is in the window, and the ones further back weigh less. There is no width, no centring, no closed rule, no step and no clipping, which is four of the five parameters document 31 spends its first section on and the whole of `Shape.edges`.

That is why `firepanda/kernel/ewm.mojo` is a second kernel and not a fourth `WindowKind`. The two families share the word window and the four reduction names, and they share nothing else. A reader who goes looking for `ewm` inside the rolling kernel is looking for a `Shape` that cannot be written, and the honest thing is to make them not go looking.

## 1. The four spellings of the decay are one number

`ewm` takes the decay as exactly one of `com`, `span`, `halflife` and `alpha`, and all four describe the same smoothing factor:

`alpha = 1 / (1 + com)`, `alpha = 2 / (span + 1)`, `alpha = 1 - exp(-ln 2 / halflife)`, and `alpha` itself.

So `span=5` and `com=2` are the same window, both of them a factor of `0.3333333333333333`, and `halflife=3` is `0.2062994740159002`. Each spelling has its own range and its own sentence when it is outside it, and the four sentences are pandas' rather than mine: `span must satisfy: span >= 1`, `comass must satisfy: comass >= 0`, `halflife must satisfy: halflife > 0` and `alpha must satisfy: 0 < alpha <= 1`. Passing none of them is `Must pass one of comass, span, halflife, or alpha` and passing two is `comass, span, halflife, and alpha are mutually exclusive`, and the misspelling of comass in both is pandas' too.

The collapse happens in `python/firepanda/_pandas.py`, in `_smoothing`, and one number crosses the boundary. The alternative is sending four optional values across and deciding on the far side which to believe, which means the refusal sentences live in Mojo where pandas' wording has no business being, and it means three absent floats in the crossing's argument budget. Document 13 section 4 measured that budget at seven real arguments after the object, and the EWM crossing spends six of them as it is.

The factor is then checked a second time at the kernel's door, in `alpha_of` and again in `ewm_agg`. That is the same arrangement `closed` has on the rolling side and it is there for the same reason, which is that the Mojo API is also a caller and it does not come through Python.

## 2. The adjusted form is not a quadratic sum

Written as a definition, the adjusted mean at row `i` is the sum of `(1-alpha)^k` times row `i-k` over all `k`, divided by the sum of the weights. Read literally that is the height of the column times itself, and a column of ten thousand rows is a hundred million multiplications for an answer that needs ten thousand.

It is also unnecessary, because the numerator and the denominator each satisfy a one line recurrence: multiply what you had by `1-alpha` and add the new row. The kernel does not carry the numerator and the denominator separately either. It carries the mean itself and the weight behind it, and folds the new row in as a weighted average of the two, which keeps the carried number in the range of the data instead of growing without bound for a long column.

## 3. One recurrence covers both readings of `adjust`

`adjust=True` weighs every row it has ever seen at one relative to the others. `adjust=False` is the plain recursion, `alpha` times this row plus `1-alpha` times the answer before it, and the first row answers itself. Those read like two different computations and they are one computation with one number changed.

The number is what a newly arriving row weighs. Under `adjust=True` it weighs one and the weight behind it accumulates, so the two grow together and their ratio is the adjusted mean. Under `adjust=False` a new row weighs `alpha` and the weight behind it is reset to one after every row, so the fold reduces to the plain recursion exactly. `_weighted` in the kernel is that one loop, and `fresh = 1.0 if spec.adjust else spec.alpha` is the whole of the difference. The loop is also where `min_periods` is applied, by counting values and answering nothing until enough have arrived.

Two implementations that agree today is the arrangement document 31 section 2 argues against for the same reason: the way it fails is not that one of them is wrong when it is written, it is that the next parameter to arrive gets added to one of them.

## 4. A missing row is a decision and not a gap

`ignore_na` decides whether a missing row takes up a slot in the decay. With `ignore_na=False`, which is the default, the decay step still happens on a missing row, so the values either side of a gap are further apart than adjacent values. With `ignore_na=True` the row is not there at all and the values either side of it are adjacent.

Both readings are real and they give genuinely different numbers rather than two roundings of one. Crossed with `adjust`, that is four different answers to `ewm(alpha=0.3).mean()`, and on the column `1, None, 3, None, None, 7, 9` the last row is `7.162170410482522`, `6.203316225819187`, `5.643163966375967` and `4.954` respectively. `tests/test_ewm.mojo` pins all four, and the Python tests check all four against a running pandas, because a flag that is quietly dropped somewhere in the crossing still produces a plausible looking column.

A row before any value has arrived is neither of those cases. It is not a decayed nothing and it is not a zero, so the recurrence has not started and the row is missing. The first value to arrive becomes the carried mean directly rather than being averaged against a weight of nothing. A column that is entirely missing therefore answers a column that is entirely missing, at any `min_periods` including none.

`min_periods` counts values and not rows, which is the same reading the rolling side has, and pandas resolves a count of nought, a count of one and a negative count all to one. So `ewm(span=5).min_periods` is `1` and not `0`, and the resolution happens in the mixin's constructor where a caller can see it rather than inside the reduction.

## 5. The one guard that is copied rather than derived

pandas' fold has a comparison in it that does not belong to the arithmetic. It skips the whole update when the arriving row equals the carried mean already, and the comment beside it in the source says it is there to avoid numerical error on a constant series.

Without it, a column of one value repeated does not come back as that value repeated. The weighted average of a number with itself is that number in exact arithmetic and is that number plus or minus a bit in floating point, and after a few hundred rows a flat column has visibly drifted. So the guard is copied, comparison and all, and `test_one_value_repeated_comes_back_exactly` is what it is for.

This is one of the few places in the library where a line exists because pandas has it rather than because the mathematics asks for it. It earns that because the behaviour it produces is the correct one and the version without it is not.

## 6. Where the total differs from the mean, and the combination pandas refuses

`sum` is the numerator of the mean on its own, undivided, so it is the same loop with the division taken out and `total` deciding which. The carried value grows with the weights rather than staying in the range of the data, which is what a total is.

Under `adjust=False` there is no numerator to take, because the unadjusted form carries a mean directly and its weights are reset after every row. pandas does not invent an answer for that. It raises `NotImplementedError("sum is not implemented with adjust=False")`, and this raises the same class with the same sentence, from `EwmMixin._reduce` before anything is read. Choosing one of the two things such a total could mean would be the single place in this family where firepanda answers something pandas does not, and a caller relying on it would be relying on something that is not the pandas API.

## 7. The variance carries two weight totals and a correction

`var` and `std` are one reduction with two endings, which is the arrangement document 31 section 6 describes for the rolling side. The state is a weighted mean and a weighted second moment about that mean, folded together so that neither is computed from a sum of squares.

The correction needs two weight totals and not one. The biased variance is the second moment as carried. The corrected one multiplies it by the sum of the weights squared over the sum of the weights squared minus the sum of the squared weights, which is the weighted generalisation of the `n / (n - 1)` a plain sample variance uses, and it needs both totals to exist. So `weights` and `squares` are carried beside the moment, each with its own decay factor, `1-alpha` for the first and `(1-alpha)^2` for the second.

The denominator of that correction is zero while only one value has arrived, which is why the first row of a corrected variance is missing and the first row of a biased one is zero. The kernel tests the denominator rather than counting the rows, because that is the condition the arithmetic actually has.

`bias` is the one parameter one of these reductions reads that the decay does not, which is why it crosses in the settings tuple rather than as an argument of its own. `mean` and `sum` send an empty tuple. That is the same mechanism `window.mojo` uses and the same reason: the tuple's length is decided by the reduction, so the length is checked before anything is read out of it, and a mismatch means the two halves of the library disagree about a reduction rather than that a caller made a mistake.

## 8. The fused multiply-add in the pandas wheel

Two of the Mojo tests were written against pandas' answers and failed, both of them by one unit in the last place, on row five of `ewm(span=5).mean()` over the rows nought through nine and on row nine of the same column with `halflife=3`. The recurrence was identical on both sides.

Three algebraic orderings of the fold were written in Python and measured, and none of them reproduced pandas' number. The fourth attempt used `math.fma` and reproduced it exactly. pandas' `pandas/_libs/window/aggregations` ships compiled, and the wheel these were measured against was built with the multiply and the add of `old_weight * carried + new_weight * row` contracted into one fused instruction, which rounds once where two operations round twice.

I did not reproduce it. A kernel whose specification is another kernel's choice of instruction is a kernel nobody can maintain, and the contraction is a property of how that wheel was built rather than of pandas' definition, so a different wheel on a different platform would need a different kernel. The tests carry a relative tolerance of `1e-15` instead, which is about four units in the last place, and both the kernel and the test module say why in their docstrings. Document 31 section 3 declined to reverse engineer pandas' reset rule for the same reason and registered the differences instead, and this is the same call made a second time.

## 9. The flags that pandas reads for truth and this refuses

`ewm(adjust="no")`, `ewm(ignore_na="no")` and `var(bias="no")` are all accepted by pandas, and all three of those strings are truthy, so a caller who wrote any of them got the opposite of what they meant and no indication of it.

All three are refused here, with `adjust must be a boolean` and the two like it. That follows the `center must be a boolean` and `pct must be a boolean` refusals the rolling side already has, so it is the position this library has already taken rather than a new one. It is a deliberate strictness and it is a divergence from pandas, and it is a candidate for the registry in `firepanda-compat` the day a conformance case exercises it.

## 10. The Python surface, and the eighteen properties

`ewm` is one method on `Series` and one on `DataFrame`, with the full pandas signature, and it answers an `ExponentialMovingWindow` named exactly that on both owners. `times` and `method="table"` are declared so that the signature accepts what pandas accepts, and both are refused by name, because document 07 says a name must not resolve and then silently ignore what it was told.

The object reports eighteen properties. Four are the decay as it arrived, so `ewm(com=2).com` is `2` and `ewm(com=2).span` is `None`, which is what pandas does: the conversion is not reported back. Three more are `min_periods` as resolved, `adjust` and `ignore_na`. Then `obj`, `ndim` and `method`, and then the eight questions a rolling window answers that this one has no answer to at all, `times`, `window`, `center`, `closed`, `step`, `win_type`, `on` and `exclusions`, which are present and answer nothing. Those eight are worth checking against a running pandas rather than against a written down `None`, because an absent attribute is a different answer from `None` and only pandas can say which of the two it gives.

The frame form is the columns decayed one at a time and put back together, which is what `method="single"` means and is the only thing this library answers. `method="table"` decays across the columns of a row rather than down them, and it is a different computation and not a faster arrangement of this one. Every column is checked before any of them is read, so a frame with a text column in the middle of it raises rather than computing half an answer. It raises `DTypeError` where pandas raises `DataError`, which is the position the rolling path already took, and `numeric_only` is refused rather than answered, because which columns come back is a decision and this library does not make it.

The answer is always float64, for both owners and all four reductions, including from a column of whole numbers. A weighted mean of whole numbers is not a whole number.

## 11. What is here and what is not

Here: `Series.ewm` and `DataFrame.ewm`, all four spellings of the decay, `min_periods`, `adjust`, `ignore_na`, and `mean`, `sum`, `var` and `std` with `bias`.

Not here: `corr` and `cov`, which need a second column and are the same piece of work as `Rolling.corr` and `Rolling.cov`, so all four should land together and the crossing deliberately keeps its seventh argument slot free for them. `agg` and `aggregate`, which are the general aggregation door and belong with the rolling and grouped ones rather than being solved once here. `online`, which is a stateful object that resumes from where a previous pass stopped and is a different shape of thing entirely. A decay given as a real duration through `times`, which needs a datetime index to mean anything. And `method="table"`. None of those resolves, for the reason given above.
