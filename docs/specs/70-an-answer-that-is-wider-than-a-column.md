# 70. An answer that is wider than a column

## 1. The first two names that hand back a frame

Every `str` method written so far answers one column. `upper` answers text, `contains` answers a flag, `len` answers a number, and all three fit through one of the three doors document 07 describes. `partition` and `rpartition` do not. Each of them cuts every row at one occurrence of a separator and hands back what came before it, the separator itself, and what came after, which is three columns and therefore a frame.

They were picked as the next slice for exactly that reason. Of the fourteen `str` names still missing, these two are the only ones that force the question of what an accessor does when its answer is not a column, without also needing a regular expression engine, a list column type or the Unicode normalization tables. Everything else about them is a substring search, which this library has had since document 66.

## 2. Two names, and the difference that is not the one in the name

The `r` says search from the right instead of from the left, and that is half of the difference. `partition("a b c", " ")` cuts at the first space and `rpartition` cuts at the last.

The other half is what happens to a row the separator is not in at all, and it is the half an implementation written from the name alone gets wrong. The row is not dropped and it is not blanked. It survives whole, and `partition` puts it in the first column while `rpartition` puts it in the third. That is Python's rule for these two names and pandas inherits it directly, and an implementation that put the row first in both cases would pass every other check anyone would think to write.

That asymmetry is the one sentence worth carrying out of this document, so it is stated in the kernel docstring, in the Python test file and in the module docstring of `pattern.mojo`, and there is a Mojo test and a Python test that exist only to hold it in place.

## 3. pandas does not have a kernel for this either

The habit of the last four documents has been to work out which backend pandas is reading its answer out of, because pandas 3 holds text in Arrow and the Arrow answer and the Python answer have disagreed before. Here there is nothing to work out, and that is itself the finding.

`pyarrow.compute` has `partition_nth_indices`, which is about sorting and not about text, and it has `split_pattern`, `split_pattern_regex`, `ascii_split_whitespace` and `utf8_split_whitespace`. It has no partition kernel. So `ArrowExtensionArray._str_partition` calls `self._apply_elementwise(lambda val: val.partition(sep))`, which is a Python loop over the column calling CPython's own `str.partition` on every row, and then hands the tuples back to pyarrow to infer a type from. The object backend does the same thing through `_str_map`. Both backends are the same implementation wearing different coats.

This matters for two reasons. The first is that the oracle for this slice is Python and not Arrow, which is the opposite of the last five documents and is worth knowing before writing a test. The second is section 6.

## 4. One search, not three

The kernel could have been three kernels behind the existing text door: one that answers the part before the separator, one that answers the separator, one that answers the part after. That would have needed no new door and no new shape anywhere.

It would also have run the search three times and thrown away two thirds of each answer. The search is the entire cost of this operation and the slicing after it is free, so `text_partition` finds the separator once per row and fills three builders from that one offset. The signature says so: it takes a column and a separator and a direction, and it answers a list of three columns.

A list rather than a tuple, which reads oddly and is not a style choice. Taking a Mojo tuple apart copies what comes out of it, and a `StringArray` does not conform to `ImplicitlyCopyable`, so every caller of this would have had to be written around that. A list can have its elements moved out one at a time, which is what all three callers up the chain actually want, so the list goes all the way from the kernel to the Python layer.

## 5. A door of its own, which is the rule working rather than an exception to it

`firepanda/py/text.mojo` has three doors, and document 07 warns that adding a fourth every time a method is slightly unusual is how an accessor layer turns into fourteen near duplicates. The rule that stops that is: the doors are picked by the shape of the answer, never by the shape of the argument.

`translate` sits outside those three doors and document 68 section 9 defends that, because what makes `translate` different is its argument, which is column sized. That was an exception and it was argued for as one.

`partition` needs no such argument. Three columns is a shape, no door carries it, and so a function of its own is the rule doing what it was written to do rather than drift against it. The distinction is worth keeping sharp because it is the thing that decides the next few of these, and the module docstring now says it in place.

## 6. Where the two cannot agree, and both of them come from the same cause

There are exactly two divergences here and both of them come out of the fact that pandas does this row by row in Python while this library does it column by column.

The first is the labels. pandas labels the three columns with the integers 0, 1 and 2. A firepanda frame holds text column labels and nothing else, so the same three come back as `"0"`, `"1"` and `"2"`. Every value under them agrees. This is registered rather than worked around because what would close it is a column label type, which is a real piece of work with a long tail behind it (`split(expand=True)`, `extract`, and any frame built from a dictionary with integer keys, which today silently stringifies them), and not a different string written in this method.

The second is what happens when there is nothing to read. pandas raises its refusals inside CPython on the first row it reaches, and it works out how wide the answer is from the tuples that came back. So on a column with no readable rows there is no refusal and there is no width: `pd.Series([], dtype="str").str.partition("")` raises nothing and answers a frame with no columns at all, and a column holding only missing rows answers a frame with one. Neither of those is three columns and neither is what the pandas documentation for this name describes.

This library checks the separator once before it starts and always answers three columns. That is a deliberate disagreement rather than a gap: the width of the answer is a property of the method and an empty column is not an argument for a different answer shape. Both facts are measured in `python/tests/test_str_partition.py` so that the disagreement is recorded as a decision and not as an oversight.

## 7. The refusals, which are Python's sentences

`sep=""` raises `ValueError: empty separator` and a separator that is not a string raises `TypeError: must be str, not int`. Both of those sentences are CPython's, reached through pandas without being touched. This library raises `InvalidArgumentError` and `DTypeError`, which are a `ValueError` and a `TypeError`, with the same two sentences behind the `firepanda:` prefix that every message here carries.

`expand=False` is the third refusal and it is a different kind. It asks for one column holding three element tuples, and there is no column type here that holds a tuple. So it raises `UnsupportedError` and says in the message that only `expand=True` is written, which puts it on the board as a gap rather than as a disagreement. It is the same wall that `split`, `rsplit`, `findall`, `join` and `extractall` are behind, and it will come down for all six at once or not at all.

## 8. The twin

`text_partition_scalar` in `firepanda/kernel/scalar.mojo` is the naive version the fast one is checked against, following the habit documents 66 through 69 established. It has one loop that tries every byte offset in the row and compares the separator there, keeping the first hit it sees when searching forward and the last when searching backward.

That is the smallest difference the two names can be written with, and it is the point of writing a twin at all. The kernel reaches the same two answers through two different searches, one of which skips ahead, and the test that runs both over nine separators in both directions is checking that the fast path and the obvious path agree about which occurrence is the one.

## 9. What is not here

`expand=False` for both names, which needs a tuple column.

Integer column labels, which is the divergence in section 6 and is the same missing piece that `split(expand=True)` and `extract` will need when they land, and the same one behind a frame built from a dictionary with integer keys.

A `partition` on a frame rather than on a column, which pandas does not have either.

The twelve `str` names still missing after these two: `cat`, `decode`, `encode`, `extract`, `extractall`, `findall`, `get_dummies`, `join`, `normalize`, `rsplit`, `split` and `wrap`. Of those, `cat`, `wrap`, `normalize` and `get_dummies` need no new column type and are the next ones reachable.
