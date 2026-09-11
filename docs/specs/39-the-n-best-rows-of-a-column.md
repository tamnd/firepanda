# 39. The n best rows of a column

Status: implemented. Two frame methods, no new kernel, one refusal written down, one trick that saves a second set of template instantiations, and a bug in the kernel's entry point that the new methods found.

## 1. The same kernel with one group in it

`group_nlargest` and `group_nsmallest` have been in the core since the top n kernel was written, and they answer the grouped question: the best few rows of every group. `nlargest` and `nsmallest` are the ungrouped question, which is the best few rows of the whole frame, and that is the grouped question asked about a frame that happens to have one group in it.

So there is no second kernel. `_top_rows` builds a codes array of zeros as tall as the frame, says there is one group, and hands both to `group_top_rows_any`, which is the same erased entry point the grouped methods use. What comes back is the rows it kept, best first, and `take` turns that into a frame.

The zeros are the one allocation the ungrouped form pays that the grouped form does not, four bytes a row. It could be avoided by teaching the kernel that a null codes array means one group, and that would put a branch on the inner loop of the scan, which is the loop that reads every row of the column. Paying four bytes a row once to keep a branch out of the hot loop is the right way round, and a column being ranked has already been read at least once by whatever produced it.

## 2. Ties are half of what this decides

A frame of distinct values gives the same answer under any tie rule at all, including no rule, so the tie is where the two libraries can differ and it is where this is worth being careful.

The kernel separates two equal values by row number and the earlier row wins. That is pandas' `keep="first"`, and it is also what makes the answer independent of how the scan was split across workers, which is why the kernel had it before anything asked for it.

`keep="last"` is the other way round, and it changes two things rather than one. The membership changes, because the last of a tie is kept where the first was. The ordering of the answer changes too: pandas hands the tied rows back in reverse order of appearance under this rule. Both fall out of the same thing, which is that pandas is reading the column from the other end.

## 3. Keeping the last is the column read backwards

The obvious implementation is a second tie rule in the kernel, which means a comptime parameter on `_beats`, and `_beats` is called by `_weakest` and `_offer`, which are called by `_rank_slots` and `_scan_into`, which are called by `_group_top_core`. A boolean parameter there doubles the instantiations of six functions for every dtype the kernel is used with, and what it buys is one comparison that goes the other way.

So the kernel is not told. Instead the ranked column is read back to front, by gathering it through the reversed positions, and each kept position is turned back into the row it came from on the way out. The last row of a tie is the first row the reversed column offers, so the kernel's existing rule keeps it, and the answer comes out in the order pandas gives because the reversal is doing to the output exactly what it did to the input.

This costs one gather of one column, which is the ranked column and nothing else, because the rest of the frame is never reversed and the row numbers are mapped rather than the data. It is paid only when `keep="last"` is asked for, which is the rare call.

The check that this is right is not a proof, it is the four answers pandas gives for a frame with three keys tied at the top and three tied at the bottom, written into the test file by hand. That is the kind of thing that is easy to get almost right, and almost right here means the membership correct and the order reversed.

## 4. The count is cut down before it is spent

The kernel holds a table of `n` slots per group, so `n` is an allocation as well as a question. A caller who writes `nlargest(1000000, "value")` on a frame of six rows means give me all six, and a kernel that took the number at its word would allocate a million slots to answer with six.

`_top_rows` takes the smaller of `n` and the number of rows before the table is built. A count of zero or less is an empty frame rather than a refusal, which is pandas' answer, and it is the right one: a count and a negative count are the same request and neither of them is a mistake worth reporting.

## 5. A missing value is ranked last, and it is still a row

The kernel never keeps a null or a NaN, because a NaN loses every comparison it is in and one sitting in a slot would hold a real value out. That is the right rule for the kernel and it is not pandas' answer for these two methods.

pandas ranks a missing value last and then pads with it. Asking for the four largest of a column with three present values and two missing ones gives four rows: the three, in order, and then the first missing one. It does this in row order under both tie rules, including the one that reverses everything else, so the padding does not reverse with the ranking.

`_top_rows` does the same, and it works out which rows to pad with by elimination rather than by asking each row whether its value is present. The kernel keeps every present value it can, so an answer shorter than what was asked for can only mean it ran out of them, which means every row missing from the answer is a row whose value is missing. Elimination needs no dtype dispatch and, more to the point, it does not write down a second copy of the kernel's rule for what counts as missing, which covers a NaN as well as a null and would be a rule in two places that could disagree.

This is the one place where the ungrouped methods differ from `group_nlargest` and `group_nsmallest`, which answer short. That is a difference worth having rather than a gap: the grouped pair are firepanda's own methods and the ungrouped pair carry pandas' names, and a method that carries pandas' name should carry its answer.

## 6. What is refused, and why each one is a refusal rather than a guess

`keep="all"` answers more than `n` rows when the `n`th value is tied, and pandas does answer it. A fixed table of `n` slots cannot hold an answer whose height depends on the values in the column, so this needs a second pass that finds the cut value and then takes every row equal to it. That is a different piece of work rather than a flag, and it is refused with a `NotImplementedError` rather than quietly answered as `keep="first"`, which would look correct on every frame until one had a tie in it.

More than one column is refused for the same kind of reason. pandas ranks by the first column and breaks its ties with the second, and the kernel holds one value per slot and has no second value to break anything with. A one item list is a single column and is accepted, and a name written twice is read once, because pandas accepts both.

An empty list of columns is neither refused nor ranked. pandas answers an empty frame with the columns kept, which is not obviously right, and this layer copies it, because a caller who hits this is reading pandas' documentation and not ours.

## 7. Which side each rule lives on

The refusals above are all in the pandas layer, and the core takes a column name, a count, a direction and a word. That split is the same one `duplicated` made in document 38 section 6: the boundary carries one kind of thing for each argument, and the questions about what a caller is allowed to write are asked where the caller is.

Two of them are worth naming. A count that is not a whole number is refused by asking Python to make an index out of it, which is what pandas does underneath and which gives back the same sentence for free. And `keep` is checked against the three words before the crossing, so the core only ever sees two of them and the error a caller reads is pandas' error rather than the core's.

## 8. Two kinds of wrong argument, told apart by the message

A name that is not a column is pandas' `KeyError`. A column that cannot be ranked is its `TypeError`. The core says both with a plain error, because the core has no opinion about Python exception classes, and the binding has to pick one tag for what comes back.

It picks by looking at the message, which is the escape hatch document 14 section 3 leaves open for exactly this: the binding tags and the core does not, so a binding facing two kinds of refusal from one call has to read them apart somehow. The rule is written down in the binding, and it is that the only refusal in this path that is about a type uses the word numeric. That is not lovely and it is the honest option available, and the alternative is tagging errors at the point they are raised in a kernel that has to compile with no Python anywhere near it.

## 9. The bug the string column found

The erased entry point decided whether a column could be ranked by walking the numeric dtypes and comparing each against the column's physical dtype, and raising if none of them matched. A string column is laid out as `uint8`, so it matched, and the kernel then ranked the first `rows` bytes of the values buffer as if they were the column. It answered rows rather than failing, and the rows it answered were about the spelling of the first few strings.

A dictionary column has the same shape of problem, since its physical dtype is whatever its codes are, and ranking a dictionary column by its codes ranks it by the order the values happened to be first seen in.

This was reachable through `group_nlargest` before these two methods existed and nothing had noticed, because nothing had asked for it. The fix is to ask the logical type rather than the physical one, and to accept the types whose values buffer holds one comparable number per row, which is the numeric ones and the temporal ones. A timestamp is an `int64` count and ranking it as one is right, which is why the check is a pair of questions rather than one.

## 10. The answer is sorted and the input is not

What comes back is a selection of rows and a sort of that selection, and not a sort of the frame. Every column and every dtype is carried along, the labels come with the rows they belong to, and those labels are the only thing in the answer that says which rows of the input these were, since the frame comes back reordered.

That makes `nlargest(n)` cheaper than sorting and slicing for a small `n`, which is the whole reason it exists as a method rather than as advice. The kernel is one pass over the column with a fixed table, so the work is the number of rows times the cost of missing the table, rather than the number of rows times its logarithm.
