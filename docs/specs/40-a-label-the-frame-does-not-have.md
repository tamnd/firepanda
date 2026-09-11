# 40. A label the frame does not have

Status: implemented. Three methods under one name, on the frame, on the series and on the index, of which the last is the lookup the other two are built on, handed back in the open, plus `reindex_like` on the first two. One new kernel function that three older private helpers should have been, and one rule about types that is pandas' rather than the gather's.

## 1. Two operations under one name

`reindex` is the method the index was built for, and it is two operations that share a name and share nothing else.

On the rows it is a lookup. The caller hands over a set of labels, `get_indexer` says where each of them sits in the frame, and the answer is a gather through those positions. A label the frame does not have comes back as a row of missing values, which costs no branch of its own because `get_indexer` answers a not found label with a negative position and `take_any` already reads a negative position as a null row. Document 19 is the lookup, and document 20 is where the same gather was used to write half of the index editing operations, so between the two of them the row half of this method is four lines.

On the columns it is a different lookup in a different place. The caller hands over a set of names, the schema says whether each one is there, and a name the frame does not have becomes a whole column of NaN as tall as the frame. There is no gather, no index, and no shared code with the row half beyond the fill value, which is why they are two methods in the core and one method at the boundary.

The boundary is where they are one thing, because that is where pandas put them. `reindex(index=..., columns=...)` does both, the columns first and then the rows, in that order because narrowing the frame before gathering it means the gather moves less.

## 2. The fill is a row, not a second pass

Without a `fill_value` a label the frame does not have gives a row of nulls. With one it gives a row of that value, and the obvious way to write the second is as the first followed by a fill over the answer.

That is wrong, and it is wrong in a way that is easy to ship. A fill over the answer covers every null in it, including the nulls the frame already had, and pandas does not do that: `fill_value` belongs to the rows the lookup failed to find and not to the rows it found. A frame with a hole in it, reindexed onto a label it does not have, comes back with the hole still in it and the new row filled.

So the fill is done on the way past rather than afterwards. One row holding the fill value is appended to each column, and every not found label is pointed at that row instead of at a negative position. The gather then does the filling itself, in the same loop it was going to run anyway, and a null that was already in the column is gathered as a null because nothing looked at it. There is no bitmap to construct, no second pass over the answer, and it works for a text column for free, since appending a row to a column is `concat_two_any` and that already knows about text.

The sentinel row costs one allocation per column and one copy of the column into it, which is the same cost a second pass would have paid and which buys the correct answer rather than the plausible one.

## 3. Widening is a pandas rule and it lives in the read path

An int64 column has no way to say missing. A frame that loses a row therefore cannot come back as int64 with a hole in it, and pandas' answer is that the column comes back as float64 with a NaN.

That rule is not in the gather. `take_any` on an int64 column with a negative position gives an int64 column with a null in it, which is correct for Arrow and correct for everything else that calls it. The widening happens one level up, in `DataFrame.reindex`, and only when the lookup actually failed, which is the same place and the same condition the rest of the library uses. `widen_for_missing` is the function, and it is the same one `shift` reaches for when its gap opens in an integer column, which is the other place this rule shows up.

A column made out of nothing by the other half of the method is spelled the same way, which is a NaN in the values rather than a cleared bit in a bitmap. That is not obvious from the code, since the natural thing to write there is `all_null`, and it is what the conformance board asked for: a null is not a NaN, the comparison keeps them apart, and pandas answers a column it had to invent with NaN.

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

A frame whose own labels repeat is a `ValueError` in pandas, and it is also the one refusal here whose message is replaced rather than passed along. The core's sentence is about `get_indexer` needing a unique index and names the function to call instead, which is the right thing to say to whoever called `get_indexer` and is not what happened here, so the binding says what pandas says: cannot reindex on an axis with duplicate labels. A fill value that no column could hold is a `TypeError`. A label of the wrong type is a `TypeError` as well, by the argument in section 7. The first is told apart from the other two by looking for the word unique in the core's message, which is not a lovely rule and is the one the tagging design left available. It is written down here so that the next person to widen that path knows it is load bearing.

The fill value check itself only runs when there is a row for the fill to go in, which is pandas' rule and was measured rather than assumed. `df.reindex([10, 20], fill_value=0)` on a frame with a text column in it succeeds in pandas, and `df.reindex([10, 99], fill_value=0)` on the same frame raises. The check is therefore after the count of missing labels and not before it, which is one line and one of the more surprising things in this document.

Only one mismatch is checked, and it is text against everything else. A number read as a string raises on its own when it is asked for its bytes, and a string read as a number does not raise at all: it reads the store's integer field, which for a string is a zero. A silent zero is the failure worth spending a check on.

## 9. The same thing on a series, written twice on purpose

A series is the frame's row half with one column under it, so `Series.reindex` is the same four lines with the loop over columns taken out. It was written out rather than routed through the frame, because routing it means building a frame, reindexing it, and taking the column back, and each of those three steps has a name to carry and lose. The series keeps its own name, the index keeps its own name, and the shortest way to be sure of that is for the method to hold both of them the whole time.

Writing it out means the two rules from sections 2 and 3 are now implemented twice, and that is the part that pays for itself. A fill value belonging to the rows the lookup did not find, and an integer column widening when a row goes missing, are the two places where the plausible implementation and the correct one differ. A second copy of a rule is a second place for it to be wrong, so the tests ask the series both questions again rather than trusting that the frame's answer carries across.

The parameter list is the one real difference and it is smaller in a way that is not obvious. There is no `columns` and no `labels`, so section 6's surprise about the axis nobody named does not arise, and `axis` goes from deciding something to deciding nothing: pandas takes it, does not look at it, and accepts `axis=1` on a thing with one axis. The other difference is the default. `DataFrame.reindex` defaults `fill_value` to NaN and `Series.reindex` defaults it to `None`, which are two spellings of leave the row missing, and both of them arrive at the boundary as no fill at all.

## 10. The index hands the lookup back

`Index.reindex` is not a smaller version of either of the above. It is the lookup itself, returned to the caller instead of consumed, and it answers a pair: the labels asked for, and where each of them sits. A caller who has their own rows to move is the audience, and pandas' own internals are the first of them.

The interesting part is when the second half of the pair is missing rather than empty. pandas leaves the lookup out entirely, as `None`, when the target is the index that was asked, because then nothing has to move and a caller who checks for that skips the gather. An empty target is a different answer, a lookup of no positions, which says move nothing rather than move everything to where it already is. Those two are easy to collapse into one and the distinction is the whole value of the return, so `Reindexed` carries an `Optional` and the three cases are three branches with nothing else in them.

The name rule was measured rather than reasoned about, and it is not what the rest of the library would suggest. A target handed in as an index keeps its own name, and a target handed in as a bare list of labels takes the name the source index had. So the same set of labels gives a differently named result depending on how it was spelled, which is pandas' rule and now ours.

This is the one method here that the binding generator could not write. It returns a pair, and the generator's return vocabulary has no way to say a tuple of an index and an optional list, so the extension hands back a Python list of two and the Python layer turns it into a tuple. That is the same shape `get_indexer` already has, where the extension answers a list and pandas answers a numpy array, and it keeps the awkwardness on the Python side of the boundary where it is cheap to read.

`limit` and `tolerance` are refused here rather than given pandas' sentence about pad and backfill, which is a deliberate difference from section 6. On the frame those two are pandas' own error for a real mistake the caller made. Here there is no filling written at all, so the honest answer is that the parameter is not implemented, and saying it is only valid with a method the library does not have would be answering a question about a feature that does not exist.

## 11. Reading the labels off something else

`reindex_like` is `reindex` with the labels taken from another object rather than written out, and the interesting question about it is why it is a method at all rather than a line at the call site.

The answer is the index name. `df.reindex(index=list(other.index))` gives the right rows in the right order and gives them back under the name `df`'s index already had, which is not what asking for another frame's shape means: the point is to come back labelled the way that frame is labelled so that the two can be compared, joined or stacked. A set of labels has nobody to name it and an index does, so the core's `reindex` grew a second overload that takes an `Index` and keeps its name, and the overload that takes a bare set of labels is now one line that builds an index out of them under this frame's own name and calls the other. That is the whole of the change in the core, and it is the same rule section 10 records for `Index.reindex`, which is that the name belongs to whoever was in a position to say what it was.

The frame's version does both axes and the series' version does one, which is where the two disagree about what they will take. A frame needs column names as well as labels, so it needs another frame, and pandas refuses a series there with a sentence about there being no axis named columns on one. A series needs only labels, so a frame is as good a source as a series, and both are accepted. That asymmetry looks like an oversight in pandas and is not: it falls out of how many axes each side has.

There is no `fill_value` here, because pandas does not offer one, so the widening rule from section 3 has nothing to stop it and a column that gains a row comes back as float64. `method`, `limit` and `tolerance` are answered exactly as they are on `reindex`, through the one helper all four paths now share, which exists because the same three refusals written out four times is three chances to reword one of them by accident.
