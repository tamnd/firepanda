# 72. An answer whose width is in the data

## 1. The third shape, and the first one nobody can predict

Document 70 was about a `str` method whose answer is wider than a column, and document 71 was about one whose answer is narrower. Both of those widths are constants. `partition` answers three columns whatever you give it and `cat` answers one string whatever you give it, so in both cases a reader can say how wide the answer is before seeing a single row.

`str.get_dummies` cannot be described that way. It splits every row at a separator, collects the distinct pieces across the whole column, and answers one column per piece. How wide the frame is and what its columns are called are both properties of the data, and there is no argument that constrains either. A column of two rows can answer four columns and a column of two thousand can answer one.

That is a third answer shape and it gets a function of its own in `firepanda/py/text.mojo` for the same reason the other two did. The rule in that file is that the shape of the answer picks the door, and a frame of unknown width is not a column, not three columns and not a string. This is the third consecutive slice where that rule has decided the layout question without an argument, which is the point of having written it down as a rule rather than as three decisions.

## 2. What a token is, which includes nothing

Splitting `"a|b"` at `|` gives `a` and `b`, which is the part nobody is confused about. The part worth writing down is what happens at the edges.

A row that is empty splits into one piece and that piece is the empty string. A row that starts with the separator has an empty piece before it. A row that ends with the separator has an empty piece after it. Two separators next to each other have an empty piece between them. All four of those produce the same token, which is the empty string, and pandas keeps it.

So the empty string is a column label that a dummy frame can genuinely have, and a column of `["", "|a", "a|", "a||b"]` answers three columns called `""`, `"a"` and `"b"`. This reads like an oversight and it is not obviously wrong either: the method is asking which pieces a row is made of, and a row made of nothing and something is made of nothing and something. Whatever the reasoning, it is measured rather than assumed, and the Python tests reach that label by all four routes so that an implementation skipping empty pieces fails on every one of them.

## 3. A set and not a count

A row holding the same token twice contributes one, not two. `"a|a|a"` answers a single column called `a` holding the value 1, and there is no value 2 anywhere in the output of this method.

The name says this and the return type hides it. The columns are int64 rather than bool, which is pandas' choice and is the first thing a reader guesses is a count. It is not. The only two values are 0 and 1 and the int64 is a convention rather than a widening of anything.

`dtype=bool` is written and gives the same frame with its columns read as flags, which is closer to what the data means. Anything else is refused by name, with a message saying that int64 and bool are the two written rather than saying the method is missing, because a caller asking for float here has a working method one argument away.

## 4. A missing row is zeros, which is the one place on this accessor it does not stay missing

Every other name on this accessor propagates. A missing row in, a missing row out, and where pandas holding its own string dtype disagrees that is registered as a divergence rather than followed.

This one does not. A missing row contributes no tokens, so it adds nothing to the column labels, and in the answer it is a row of zeros rather than a row of nulls. Both libraries agree and neither has a null anywhere in the output.

That is consistent once the answer is read as membership. The question each cell asks is whether this row holds this token, and a row that says nothing holds no tokens, so the answer is no rather than unknown. It is still worth flagging in the docstring because it is the single exception on a surface where the rule otherwise holds everywhere, and a reader carrying the rule across from the other twenty names will get it wrong.

## 5. Byte order, which is code point order

The columns come back sorted. The sort is a byte comparison, which for UTF-8 is the same ordering as comparing code points, because that property is the reason UTF-8 is encoded the way it is.

So a digit sorts before a capital before an underscore before a lowercase letter before anything outside ASCII, and `["b", "a", "C", "_", "1", "é", ""]` comes back as `"", "1", "C", "_", "a", "b", "é"`. The empty label sorts first because a prefix sorts before the thing it is a prefix of and the empty string is a prefix of everything.

No locale is consulted by either library and none should be, since a column label that changed with the machine's locale would make a frame that is not reproducible. This is one of the few places where doing the simple thing and doing the right thing coincide exactly.

## 6. Two passes, because the width has to be known before anything is written

The kernel reads the column twice and there is no way around it. The first pass collects the distinct tokens, because until every row has been read there is no way to know how many columns there are or what they are called. The second pass fills them.

The token set is held as a sorted list with a binary search on each insert. That is O(t log t) comparisons against O(t) distinct tokens, and t is the number of distinct tokens rather than the number of rows, which in practice is small: a column of a million rows tagged from a vocabulary of twenty answers twenty columns. A hash table would be the right structure if t were large, and if a caller ever turns up where it is, this is the place to change and the sorted list is the thing to replace. Until then the sorted list gives the ordering for free, and the ordering is required, so a hash table would need a sort after it anyway.

The second pass allocates one int64 column of zeros per token and writes a 1 at the found index. It splits each row a second time rather than remembering the first split, which trades the work of the split against holding every token of every row in memory at once. The split is a byte scan and the memory is the whole column again, so the trade is the right way round.

## 7. Where the split offsets are counted

The split is a byte search and the offsets are byte offsets. That is correct and it is also invisible on ASCII data, which is why the tests include rows where the tokens on both sides of the separator are wider than a byte.

An implementation that counted characters would still produce the right answer for `"a|b"`, and would produce a wrong answer or a broken string for `"é|ö"`. There is no way for an all ASCII test to catch it. This has now come up in enough slices that it is worth stating as a rule for this accessor: any test of an offset must include a row where a character is more than one byte, or it is not testing the offset.

## 8. Both refusals, one of which is pandas' own

An empty separator is refused. pandas refuses it too, from inside Arrow's split kernel, with `ArrowInvalid: Empty separator`. Since pandas raises here, this is a refusal that repeats a sentence rather than inventing one, and the test asserts that both libraries refuse rather than asserting the wording, which is the form that survives pandas changing its message.

A separator that is not a string is refused. pandas refuses this one too, with `TypeError: expected bytes, int found`, which is Arrow complaining about what it was handed. The message here names the argument instead, because Arrow's sentence describes a conversion the caller never asked for, but the type is the same and the test checks the type on both sides.

That is a pleasant change from the previous slice, where pandas checked none of `sep`, `na_rep` or `join` and both bad arguments fell into `str.join` to produce sentences that named pandas' implementation. Here the validation exists upstream and can be matched.

## 9. The disagreement, which is pandas failing rather than choosing

A column with no readable rows has no tokens, so the frame has no columns. Both libraries answer a frame of shape `(0, 0)` when the column has no rows at all.

For a column that has rows but where every one of them is missing, pandas raises:

```
ValueError: Empty data passed with indices specified.
```

That is pandas' own frame constructor refusing to build a frame with no columns and a non empty index. It is an internal sentence about assembling a frame, not a statement about this method, and there is no reading of `get_dummies` under which a column of missing rows is an error while a column of no rows is fine. The two questions have the same answer: no readable rows, no tokens, no columns.

This library answers the `(0, 0)` frame in both cases. The Python test asserts pandas' failure directly, so the difference is recorded as something measured and decided rather than something not noticed, and the test starts failing the day pandas fixes it, which is when this note should be revisited.

This is the second upstream observation in three slices where the interesting behaviour is pandas' own machinery surfacing rather than pandas choosing. `partition` deciding its refusals one row at a time was the first.

## 10. What this does not need, which is why it came now

Nine `str` names are left after this one and four of them are waiting on a regular expression engine. Of the ones that are not, `join` and the default form of `split` want a list column, and `encode` and `decode` want a binary column.

`get_dummies` wants neither. Its column labels are text and a firepanda frame holds text labels, so unlike `partition` it does not even carry the integer label divergence. It is the last name on this accessor that can be written with the types that already exist, which is the whole reason it came before the four the engine blocks.

What is left after it is genuinely blocked on missing machinery rather than on nobody having written it, and the list of that machinery is now short and specific: a regular expression engine, a list column, a binary column, a column label type and label alignment.
