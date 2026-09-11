# 40. A label the frame does not have

Status: implemented. One frame method with ten parameters, of which two do the work, one new kernel function that three older private helpers should have been, and one rule about types that is pandas' rather than the gather's.

## 1. Two operations under one name

`reindex` is the method the index was built for, and it is two operations that share a name and share nothing else.

On the rows it is a lookup. The caller hands over a set of labels, `get_indexer` says where each of them sits in the frame, and the answer is a gather through those positions. A label the frame does not have comes back as a row of missing values, which costs no branch of its own because `get_indexer` answers a not found label with a negative position and `take_any` already reads a negative position as a null row. Document 19 is the lookup, and document 20 is where the same gather was used to write half of the index editing operations, so between the two of them the row half of this method is four lines.

On the columns it is a different lookup in a different place. The caller hands over a set of names, the schema says whether each one is there, and a name the frame does not have becomes a whole column of missing values as tall as the frame. There is no gather, no index, and no shared code with the row half beyond the fill value, which is why they are two methods in the core and one method at the boundary.

The boundary is where they are one thing, because that is where pandas put them. `reindex(index=..., columns=...)` does both, the columns first and then the rows, in that order because narrowing the frame before gathering it means the gather moves less.

## 2. The fill is a row, not a second pass

Without a `fill_value` a label the frame does not have gives a row of nulls. With one it gives a row of that value, and the obvious way to write the second is as the first followed by a fill over the answer.

That is wrong, and it is wrong in a way that is easy to ship. A fill over the answer covers every null in it, including the nulls the frame already had, and pandas does not do that: `fill_value` belongs to the rows the lookup failed to find and not to the rows it found. A frame with a hole in it, reindexed onto a label it does not have, comes back with the hole still in it and the new row filled.

So the fill is done on the way past rather than afterwards. One row holding the fill value is appended to each column, and every not found label is pointed at that row instead of at a negative position. The gather then does the filling itself, in the same loop it was going to run anyway, and a null that was already in the column is gathered as a null because nothing looked at it. There is no bitmap to construct, no second pass over the answer, and it works for a text column for free, since appending a row to a column is `concat_two_any` and that already knows about text.

The sentinel row costs one allocation per column and one copy of the column into it, which is the same cost a second pass would have paid and which buys the correct answer rather than the plausible one.

## 3. Widening is a pandas rule and it lives in the read path

An int64 column has no way to say missing. A frame that loses a row therefore cannot come back as int64 with a hole in it, and pandas' answer is that the column comes back as float64 with a NaN.

That rule is not in the gather. `take_any` on an int64 column with a negative position gives an int64 column with a null in it, which is correct for Arrow and correct for everything else that calls it. The widening happens one level up, in `DataFrame.reindex`, and only when the lookup actually failed, which is the same place and the same condition the rest of the library uses. `widen_for_missing` is the function, and it is the same one `shift` reaches for when its gap opens in an integer column, which is the other place this rule shows up.

Two consequences are worth writing down. A `fill_value` stops the widening, because with a fill there is no missing row to widen for and the column keeps the type it had. And `widen_for_missing` drops the validity bitmap and writes a NaN rather than keeping the bitmap and widening under it, so a test that asks a widened column whether row two is valid gets yes, and has to ask whether row two is a NaN instead. That caught two tests here before it caught anything else.

## 4. Where the block of one value went

A column of one value repeated is a thing this library needed in four places before anyone gave it a name. `shift` fills the gap it opens at one end of a column. `align` fills a column that one side of the pair does not have. The plan simplifier builds a one row block for a constant. And now `reindex` needs the sentinel row from section 2, which is that same block with one row in it.

Three of those four had already been written as private helpers in three different files, each slightly different and each unable to be the others. Rather than write a fourth, the one that could already do text was moved next to `all_null` in `binary.mojo` and given a public name, `filled_block`, because `all_null` is the same function with nothing to put in the rows. `shift` now calls it, `reindex` now calls it, and the module docstring in `shift.mojo` says where it went and why.

The other two were left alone on purpose. `align._constant` refuses text for a reason that belongs to alignment and not to block building, and folding that refusal into a shared function would have moved a domain rule into a place that has no domain. A consolidation that has to carry an argument about who is allowed to call it is not a consolidation.

`all_null` stays a function of its own for the same kind of reason, which is that it writes no values at all. The buffer arrives zeroed, a null holds a zero, and so it only has to install a bitmap. Making it a branch of `filled_block` would have made it slower to say the same thing.

## 5. Asking for nothing is asked and answered before the lookup

A reindex onto an empty set of labels is an empty frame, and it goes through a short circuit rather than through the lookup.

The reason is that a list with no values in it has no type in it either. `array_from` gives float64 for an empty list, since something has to be chosen, and `get_indexer` on an index that is not a range concatenates the wanted labels with the frame's labels so it can compare them, and a concat of float64 with int64 is a refusal. The lookup would therefore fail on a question nobody asked: an empty answer does not depend on what type the empty thing was.

So an empty target takes the rows the caller asked for, which is none of them, and puts the caller's labels on the result. One branch, and it also happens to be the fast path.

## 6. The eight parameters that do not do the work

pandas' `reindex` has ten parameters. `index`, `columns` and `fill_value` are the three that do something here, `labels` and `axis` are two spellings of the first two, and the rest are answered one at a time at the boundary rather than swept into a `**kwargs` nobody reads. A caller who passes one of them should get an answer about that parameter.

`method` is refused. It fills a label the frame does not have from the label beside it, which needs the labels in order to mean anything and is a different operation from putting a value in the row. It is a real gap and it is written down as one.

`limit` and `tolerance` belong to `method`, and passing either without it gives pandas' own sentence back, word for word, because the caller made pandas' mistake and deserves pandas' message. Document 22 is why the wording is copied rather than improved.

`copy` and `level` are accepted and ignored, which is what pandas does. `copy` is deprecated there and everything here is immutable anyway. `level` selects one level of a MultiIndex, and a flat index has exactly the one level, so ignoring it is not a shortcut, it is the answer.

`labels` with `axis` is the fifth and it has the one real surprise in it. The positional labels are not a second way of naming the axis `axis` names, they are the axis nobody named: `df.reindex(["word"], index=[30])` reindexes the rows onto 30 and the columns onto `word`, because the columns were the axis left over. Only when neither `index=` nor `columns=` is given does `axis` decide where the labels go, and the rows are its default. Naming both axes and passing labels as well is a `TypeError`, and so is writing `axis=` next to `index=`, which is naming one axis twice. All four of those were measured rather than read, because the rule is not what the signature suggests.

A `fill_value` of NaN is read as no fill value at all, for the same reason. pandas' own default for the parameter is NaN, so a caller who writes it out has asked for the rows to be missing, which is what happens when nothing is passed.

## 7. Two answers that are not pandas' answers

`reindex(columns=["a", "a"])` is legal in pandas and gives two columns under one name. A schema that is a list of names cannot hold that, and answering one column would be a quieter wrong answer than refusing, so it is refused as something the library does not do. This is the same disagreement document 38 section 7 records for a repeated column name in `drop_duplicates`, and it will stay until a frame can hold two columns under one name, which is not soon.

A label whose type is not the index's is the other one. `df.reindex(["10"])` on a frame labelled by integers gives pandas a frame of nothing but missing rows, on the reasonable grounds that no integer equals a string. The lookup here puts the two sets of labels in one column so it can compare them, and there is no column that holds both, so it refuses rather than inventing a rule about which types are comparable with which. That refusal lives in the lookup and not in this method, and lifting it means teaching the index about comparing across types, which is a piece of work with a wider blast radius than this one.

Both are in the divergence list rather than in the bug list. They are known, they are narrow, and each of them has a price written next to it.

## 8. What the boundary catches

The core raises for three things and pandas has three different classes for them, which is the shape document 14 section 3 exists for: the binding tags, not the core.

A frame whose own labels repeat is a `ValueError` in pandas, with the word unique in the sentence. A fill value that no column could hold is a `TypeError`. A label of the wrong type is a `TypeError` as well, by the argument in section 7. The first is told apart from the other two by looking for that word in the message, which is not a lovely rule and is the one the tagging design left available. It is written down here so that the next person to widen that path knows it is load bearing.

The fill value check itself only runs when there is a row for the fill to go in, which is pandas' rule and was measured rather than assumed. `df.reindex([10, 20], fill_value=0)` on a frame with a text column in it succeeds in pandas, and `df.reindex([10, 99], fill_value=0)` on the same frame raises. The check is therefore after the count of missing labels and not before it, which is one line and one of the more surprising things in this document.

Only one mismatch is checked, and it is text against everything else. A number read as a string raises on its own when it is asked for its bytes, and a string read as a number does not raise at all: it reads the store's integer field, which for a string is a zero. A silent zero is the failure worth spending a check on.
