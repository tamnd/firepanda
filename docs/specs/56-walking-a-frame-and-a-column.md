# 56. Walking a frame and a column

## 1. The methods that were missing were not methods

Every other document in this series is about a method. This one is about five Python protocols, `__iter__`, `__contains__`, `__bool__`, and the two ordinary members `keys` and `items` that pandas built on top of them, plus `itertuples`, which is the loop people actually write. The difference matters because a method that is missing is missing, and a protocol that is missing is not. Python has a fallback for almost every one of these, the fallback runs, and what it produces is not an error and is not the right answer either.

`__iter__` is the worst of them. A class that defines `__getitem__` and leaves `__iter__` out is still iterable, because Python falls back to the old sequence protocol: it calls `__getitem__` with `0`, then `1`, then `2`, and stops when it gets an `IndexError`. On a column here `__getitem__` reads a label, not a position, which is pandas' rule and document 36 section 11 is the argument for it. So the fallback asked for the label `0` and then the label `1`, and three things could happen. On a frame whose labels happen to be `0, 1, 2` in that order it worked, and gave the right answer for the wrong reason. On a frame whose labels were words it asked for a label that was not there, got a `KeyError` rather than an `IndexError`, and the loop did not stop, it raised. And on a frame whose labels were `0, 5, 9` it read the row labelled zero, failed to find one, and silently produced a one row answer.

Three different outcomes for the same line of code, decided by data the caller was not thinking about. That is worse than any one of them, and it is why this is a slice rather than a footnote.

`__bool__` has the same shape and a smaller blast radius. A class with `__len__` and no `__bool__` is true whenever it has a row in it, so `if df:` ran and meant something, and what it meant was not what the person writing it meant. pandas refuses that question outright, and the refusal is copied here word for word.

`__contains__` is the one with no fallback worth the name. Without it Python falls back to iterating and comparing, which once `__iter__` exists would have asked about values, which is the wrong question. It is here so that the fallback never runs.

## 2. A frame walks its names and a column walks its values

This is the rule that reads like a bug. `list(df)` gives the column names. `list(s)` gives the values. Two classes in the same library, one iterating keys and one iterating values, is exactly the kind of inconsistency a library gets criticised for, and it is right.

The argument for it is the loop it was chosen for:

```python
for name in df:
    print(name, df[name].sum())
```

A frame is a mapping from name to column, and a mapping iterates its keys, which is what every mapping in Python does. A column is a sequence of values, and a sequence iterates its values, which is what every sequence in Python does. The two classes iterate differently because they are two different kinds of thing, and each one follows the convention for its own kind. That it looks inconsistent from outside is a consequence of them sharing a library rather than of either one being wrong.

The same split runs through `in`, through `keys`, and through `items`. `"a" in df` asks about a name. `"p" in s` asks about a label. `df.keys()` is the names and `s.keys()` is the labels. `df.items()` pairs a name with a column and `s.items()` pairs a label with a value. Once the first decision is made the other four are not decisions at all.

## 3. `in` on a column never looks at a value

This is the rule that surprises everybody, and it is worth stating as flatly as possible: `2 in s` asks whether the column has a row labelled two. It does not look at what any row holds. So a column holding `[1, 2, 3]` with the labels `p`, `q`, `r` answers False to `1 in s` and True to `"p" in s`, which is the exact opposite of what the line reads like.

It follows from section 2 rather than being a separate choice, and it is pandas' rule. Both of those matter but neither is the real argument. The real argument is that a caller who wanted the other question and got this one has no way of noticing. There is no exception, no warning and no type error: there is a `True` or a `False`, and it is a plausible one. A library that answered the reasonable question here would silently disagree with pandas in code that has no test for it, which is the worst kind of divergence there is.

The caller who wanted the other question has `s.isin([2]).any()`, which document 55 built and which says what it does.

## 4. What a key of the wrong kind gets

`1 in df` is False. `None in df` is False. `1.5 in s` on a column whose labels are words is False. None of them raises.

`in` is a question, the answer to it is no, and a question that cannot be answered with a lookup has still been answered. `Index.__contains__` has said this since the index slice and the sentence in its docstring is the sentence here. It is also what pandas does, and the two agree for the same reason rather than by copying.

On a frame this is one line, because a column name here is always a string, so a key that is not a string is a key no column could have. On a column it is a question for the index, which already answers it, so `__contains__` is a call to `Index.contains` and nothing else.

## 5. `keys` gives back what the axis gives back

`df.keys()` is `df.columns` and `s.keys()` is `s.index`. Not a copy of them, not a new object, the same answer through a second name, because pandas defines it as exactly that and a caller who found the two disagreeing would be right to file it.

That means `df.keys()` gives back a list here where pandas gives back an `Index`, which is the divergence `columns` already carries rather than a new one. Document 21 section 7 has the general form of it, which is a list where pandas hands back something of its own. Fixing it means giving a frame an `Index` of column names, which is a change to what `columns` answers and a change to about forty call sites, and it is not this slice. What matters here is that `keys` and `columns` agree, and they do because one calls the other.

## 6. `items` is lazy on a frame and eager on a column

`df.items()` yields a name and a column at a time and never builds the list. `s.items()` reads both the labels and the values into lists and zips them.

The asymmetry is not an oversight. A frame's `items` produces one wrapper per column and there may be a thousand columns, and a caller who breaks out of the loop after the first should not have paid for the other nine hundred and ninety nine. A column's `items` produces a pair per row and there may be a million rows, and reading the values out one at a time across the boundary would cost a call per row where reading them once costs a call. So each one is lazy in the dimension that is cheap to be lazy in.

pandas makes the same split for the same reason, and the observable difference is what `type()` says: a generator on a frame and a `zip` on a column. Neither is asserted here, because a caller who depends on it is depending on something pandas has changed before.

## 7. `itertuples` is the loop, and `iterrows` is not in this slice

`itertuples` is how a person walks rows in pandas, and it is the fast one. `iterrows` builds a row as a series, which means an object and an index per row, and it is the reason people are told not to iterate a frame at all.

Neither of those costs is really about tuples. What `itertuples` does here is read every column into a Python list once and then cut the rows out of those lists, which is one crossing of the boundary per column. The alternative, reading each cell where it is needed, is one crossing per cell, and on a frame of ten columns and a hundred thousand rows that is ten crossings against a million. The whole method is that one decision plus the shape of the answer.

The shape is pandas' and is copied exactly. The tuple type is called `Pandas`, which reads strangely in a library that is not pandas and is kept because code matches on `type(row).__name__` and because a person reading a traceback should see the same word they would have seen. The first field is called `Index` and holds the row label, unless `index=False`. `name=None` gives a plain tuple instead.

The one rule nobody has to write is the renaming. A column called `a b` or `class` or `1x` cannot be a field name, and those become `_1`, `_2`, `_3` for their position in the tuple. That is `collections.namedtuple(..., rename=True)` doing its job, pandas hands the problem over the same way, and the two libraries agree on the answer without either one containing the rule. The same is true of a `name` that could not be a type name: it raises, with `namedtuple`'s own sentence, because pandas does not check the argument either.

`iterrows` is not here. A row across a mixed frame is a series whose values are of several types at once, and this library has no type that holds several types at once, which is the same gap that `xs`, `iloc[row]`, `loc[label]` and `squeeze(axis=0)` are all waiting on, and document 36 section 6 is where it is set out. Writing it for the uniform frame and refusing the mixed one would be a method that works on the frames in the tests and fails on the frames people have, so it waits for the row as a series to exist. Section 9 says so again.

## 8. What a missing row looks like coming out

`list(s)` on a float column with a gap in it gives `None` where pandas gives `nan`. On a text column it is the same, and pandas gives `nan` there too.

This is not a decision of this slice. Iteration hands back whatever `to_list` hands back, `to_list` hands back the one null this library has, and pandas hands back the one of its four missing values that belongs to that dtype. Document 55 section 5 is the long version and it is the same difference in a new place. It is called out here because it is visible in the most ordinary line of code there is, `for x in s`, and a caller writing `if x is None` against this library and `if pd.isna(x)` against pandas is writing two different programs.

A datetime column is worse and is also not this slice's doing: `list(s)` gives integers rather than timestamps, because that is what `to_list` gives, which is the same gap `DatetimeIndex.min` has. It is named here so that it is on the list twice rather than once.

## 9. What is refused and what is left

Nothing in this slice refuses anything. There is no parameter that is accepted and ignored, no sentence saying a shape is not written, and no divergence registered. That is unusual enough to be worth stating.

What is left over, named so it does not have to be found again:

`iterrows`, which needs the row as a series, for the reason section 7 gives. It is the same blocker as four other members and it should be one slice that unblocks all five rather than five slices that each work around it.

Iterating a group by. `for key, frame in df.groupby("k")` is ordinary pandas and it needs each group cut out as its own frame, which is a filter per group and is a real cost rather than a wrapper. A grouped `Series` iterates as columns and a grouped `DataFrame` as frames, which is two shapes rather than one. It belongs with the group by work rather than here.

`__reversed__` on a frame and a column. Neither library defines it, so both fall back to `__len__` and `__getitem__`, and the two agree on every frame including the ones where pandas raises. Defining it here would mean firepanda answering where pandas raises, which is a divergence created on purpose to fix somebody else's bug, and document 48 section 7 has the argument for not doing that. `Index` is the one that defines it in pandas, and the fallback here reads positions on an index, so the two agree there as well.

`empty`, which the refusal sentence in section 1 names and which does not exist yet on either class. The sentence is copied whole, pointing at three members that exist and one that does not, because a sentence people search for is not a sentence to edit.
