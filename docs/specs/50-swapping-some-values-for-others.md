# Swapping some values for others

## 1. The method that was supposed to need a kernel

`s.replace(to_replace, value)` answers the column with every row holding one value carrying another instead. The comment that closed the clip slice said this one was going to need the per value mapping kernel that the label half of `rename` has been waiting on. That turned out to be wrong, and it is worth saying plainly why rather than quietly shipping the thing that works.

The prediction came from reading `replace` as a lookup: build a hash table of what goes to what, walk the column once, and answer whatever the table says for each row. That is the right shape for a mapping of five hundred pairs and it is the wrong shape for what people actually call. The pairs that arrive here are almost always one or two, and every pair is a comparison against a value and a pick between two columns, which is exactly what document 48 built and document 49 used twice. So `replace` is `where` again, run once per pair, with the condition written for the caller instead of by them. No new kernel, no new binding, and the same widening and missing value rules fall out without anybody restating them.

The hash table is still the right answer at some size and section 7 says where the line is. It is not the right answer at the size that made this method worth having.

## 2. What is here

`DataFrame.replace` and `Series.replace` with the pandas 3.0 signature, which is `to_replace` and `value` positional followed by `inplace` and `regex` as keywords. There is no `limit` and no `method` here because pandas 3.0 does not have them either. `value` defaults to a private sentinel rather than to None, because None is a value somebody may want to put in a column and the method has to be able to tell the two apart, which is what pandas does with the same reasoning.

`to_replace` is taken in every shape pandas takes it on a column. A single value, a run of values, a mapping of pairs, and a column, which pandas reads as a mapping because a column carries a label against every row. On a frame there are three more shapes and they are section 5.

`inplace` is refused, which makes this the seventeenth and eighteenth name on that list. `regex` is refused for now, and section 8 says what it would take.

## 3. Every pair is judged against the column as it arrived

This is the rule the whole method turns on and it is one line of code. The condition for a pair is worked out against the original column, never against what the pair before it produced, and then the pick is applied to the running answer.

Two things people find surprising fall straight out of that. `s.replace([1, 2], [2, 3])` does not send the ones to two and then on to three, because the rows that hold a two are the rows that held a two when the call started. And `s.replace([1, 2], [2, 1])` swaps the ones and the twos for each other in a single call, which would be impossible if each pair could see the last one's work. pandas answers both the same way and for the same reason.

When two pairs both match a row, the last one wins, since its pick is applied last. `s.replace([1, 1], [2, 3])` sends the ones to three. That is not a rule anybody wrote either. It is what applying the picks in order does.

## 4. A value nothing holds, and a value nothing could hold

There are two different kinds of nothing here and they are handled in two different places.

A value the column could hold and does not hold is an ordinary comparison that answers no rows. The pick is never built, because a pair that moves nothing is dropped before anything is allocated, which is the same check `clip` makes per bound and `where` makes per column.

A value the column could never hold is the interesting one. `df.replace("zz", 9)` on a frame of integers is a comparison between a column of numbers and a word, which this library refuses. pandas does not refuse it: the word simply matches no rows and the frame comes back unchanged. So the comparison is made, the refusal about types is caught, and the pair is read as matching nothing. That is the one place in this method where a `DTypeError` is swallowed rather than raised, and it is swallowed only for the question of what matches, never for the question of what goes in.

A `to_replace` of nothing at all, which is a NaN or a None, is not a comparison. Nothing equals a missing value, including another missing value, so this is `isna` instead. That makes `s.replace(float("nan"), 0)` the same answer as `s.fillna(0)`, and pandas agrees, which is worth knowing mostly because it is the one route into this method that people reach for by accident.

## 5. The mapping is read two ways and people get it backwards

On a frame a mapping is read as a mapping of column names or as a mapping of values, and which one it is depends on what else arrived.

A mapping with a `value` beside it is a mapping of column names. `df.replace({"a": 2}, 5)` sends the twos in column `a` to five and leaves every other column alone.

A mapping with nothing beside it is a mapping of values, applied to every column. `df.replace({2: 7})` sends every two in the frame to seven. The consequence people trip on is that `df.replace({"a": 2})` on a frame with a column called `a` does nothing at all, because nothing in that call said `a` was a column name, so it is a value, and no column holds the word `a`.

The exception is the nested shape. A mapping whose values are all mappings is a mapping of column names to pairs, so `df.replace({"a": {2: 99}})` is per column even though no `value` arrived. That test is on the whole mapping rather than on each entry, which is why a mapping that is half nested is read as values and quietly does nothing useful. pandas behaves the same way and it is a wart in both.

`value` has a mapping shape of its own, read by column name, so `df.replace(2, {"a": 7, "b": 8})` is a different replacement per column for the same thing replaced. A name the frame does not carry is ignored on both sides rather than refused, which is the choice pandas makes for column labels and the opposite of the one it makes for row labels.

Four combinations are refused and the sentences are pandas' own, down to the trailing space on one of them. A mapping of pairs with a `value` beside it gets `Series.replace cannot specify both a dict-like 'to_replace' and a 'value'`, and that sentence names the column class even when the call was on a frame, because the frame hands each of its columns to the column method and that is where the two arguments are finally read. A single thing to replace with a mapping as the value gets `Series.replace cannot use dict-value and non-None to_replace`. A mapping of columns with a run as the value gets `value argument must be scalar, dict, or Series` as a `TypeError`. Two runs of different lengths get `Replacement lists must match in length. Expecting 2 got 1 `, with the space, because somebody out there is matching on the whole string. And a call that names nothing to replace and nothing to put there gets the sentence naming the class it was called on.

## 6. The type the answer has

This is document 48 section 5 again, unchanged. A column that a pair reaches keeps its type, and a replacement the column cannot hold is refused rather than widening the column to make room. `s.replace(2, 2.5)` on whole numbers is a column of floats in pandas and a `DTypeError` here.

Two more type facts are worth writing down because they read as differences and are not. Replacing with nothing gives a real null here and gives a column of objects holding None in pandas, because pandas has an object dtype to fall back on and this library does not. And a category keeps its type across a replacement into a category it already has, while a replacement into a category it does not have is refused with the sentence pandas raises, which the category fallback was already raising before this method existed.

## 7. What it costs, and where the hash table starts to win

One pair is one comparison and one pick over the column, so `n` pairs is `n` passes. Every comparison reads the original column, so the reads are all of the same memory and all of them are independent, and only the picks have to be applied in order.

The lookup version would be one pass over the column and one hash probe per row. Against `n` passes of a comparison, which is a vector instruction per lane and no memory beyond the flags, the crossover is not at two pairs and it is not at three. A comparison against a value is fast enough that the probe has to be amortised over a good many pairs before it pays, and the flags for a pass are a bit per row while the table is a bucket per pair. The honest reading is that the current shape is right up to somewhere in the tens of pairs and that nobody should guess more precisely than that without measuring.

What would actually change the answer is the shape of the call rather than the count. The label half of `rename` hands a mapping over a set of labels that are known to be unique and asks for exactly one lookup each, which is the hash table's own problem and not this one. That kernel is still worth building. It is just not what `replace` was waiting for.

## 8. What is not here yet

`regex=True` is refused, with a sentence saying why: a replacement read as a pattern is the text kernel's business and every condition in this method is a comparison. It is not hard to add once the pattern matching in document 43 can answer flags for a whole column, and it should be added there rather than bolted on here.

`inplace` is refused as everywhere, and the list of names waiting on it is now eighteen long. It has been the largest single thing standing between the conformance board and L3 for five slices running and it wants a design rather than another method.

`Index.replace` does not exist in pandas, so there is nothing to match there. `Series.replace` on a column of categories where the replacement would empty a category leaves the empty category in place, which is what pandas does, and nothing here removes unused categories.
