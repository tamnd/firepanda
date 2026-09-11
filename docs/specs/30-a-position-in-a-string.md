# 30. A position in a string

The `str` accessor is the largest namespace pandas has. Fifty seven names, all of them a column in and a column out, and the conformance board had every one of them at zero. This document is about the twelve that came first and about why those twelve are a group.

They are `len`, `slice`, `slice_replace`, `get`, `find`, `rfind`, `index`, `rindex`, `startswith`, `endswith`, `removeprefix` and `removesuffix`. What they share is not that they are easy. It is that every one of them either takes a position or answers one, and a position in a string is a character.

## The library already counted, and it counted bytes

There is a lot of string machinery here before any of this. `str_slice` is SQL `substring`, `like` walks a pattern, `find_bytes` filters sixteen candidates at a time with SIMD, sorting compares strings, and the hash used by a group by reads them. Every one of those counts bytes, and every one of them is right to.

A SQL `substring` is defined on bytes in some engines and on characters in others, and the one in this library is the byte form because that is what the planner needs and what the tests were written against. A `LIKE` pattern matches byte sequences because UTF-8 has the property that a valid encoding of one string is a substring of the encoding of another exactly when the first is a substring of the second, so the byte search gives the character answer for free. A sort order over UTF-8 bytes is code point order, which is not a collation but is a total order and is what everything downstream needs. A hash does not care.

So the byte orientation was never a shortcut. It is the correct implementation of a set of questions that are about bytes, or that are about characters and can be answered in bytes without loss.

`s.str.len()` is the first question that cannot. `"héllo"` is five characters in six bytes and `"日本語です"` is five characters in fifteen, and a caller asking how long a row is is asking the first number. So is `s.str[1]`, and so is the position `find` answers, because pandas defines that position as an argument you can hand back to `get`.

That is the whole of what these twelve have in common and the whole of why they landed together. There is one idea in the group, it fits in a sentence, and everything after it is the arithmetic.

## Counting is not free and the trick from the other kernel does not transfer

`substr.mojo` sizes its output in parallel. It can, because the length of the output element is derivable from the length of the input element, and the input length is in the offsets view where reading it costs nothing. Two passes over the offsets give you the total payload before any bytes move, and then the copying is a parallel loop into a buffer that is already the right size.

A character slice has no such property. How many bytes `[1:4]` takes out of a row depends on which characters are in that row, so there is no way to know the payload size without decoding, and decoding is the work. The new kernel builds its output one row at a time through the same builder every other string kernel uses, and takes the sequential append rather than a first pass that costs as much as the second.

Not every one of the twelve pays that. `len` does not build any strings, so it is a parallel pass over morsels that counts continuation bytes and writes an int64, and the count is a byte test rather than a decode: a byte begins a character when its top two bits are not `10`. `find` does not build strings either. It searches with the existing SIMD byte search, gets a byte offset, and converts that offset into a character position by counting starts before it, which is one linear pass over the prefix of a row that already matched. The rows where nothing matched pay nothing.

`startswith` and `endswith` are the two that do not belong to this document at all, in the sense that they were already implemented. A prefix test is a byte comparison and gives the character answer, which is the property above. They are here because pandas puts them on the same accessor and a caller reaching for `str.startswith` should find it, not because anything about them was character shaped. The Python layer sends them through a different door and the Mojo side wraps the mask kernel that already existed.

## Where the bounds are resolved

Python's slice rules are short to write and easy to write backwards. An absent bound means the far end, and which end is far depends on the sign of the step. Getting that wrong gives an empty string on every row instead of an error, which is the failure mode that survives a test suite, and it survived a first draft here until a test with `slice(None, None, 2)` on it caught it.

The rules cannot be applied on the Python side, because they resolve against the length of the string and every row has a different one. So a bound crosses the boundary as an absence and not as a number. `None` in `s.str.slice(None, None, -1)` is a value that means reverse the row, not a caller who left an argument out, and the reader that carries it is the only one in `args.mojo` that answers an `Optional`.

`get` takes its index in the `start` position, because an index is a position and that is where positions go. It is not a slice of one character, though, and the reason is worth stating: `""[1:2]` is the empty string and `s.str.get(1)` on an empty row is missing. pandas is the one being copied there, since Python would raise. That difference is a separate kernel and it is the smallest kernel in the file.

## Three doors and one accessor

Fifty seven names is not fifty seven bindings. The rule document 07 sets and `temporal.mojo` follows is that the shape of the answer picks the door, and on this accessor there are three shapes: a mask, a number, and text. So there are three bound methods, the name of the operation crosses as a word, and the Python layer is where each word has a name a caller was invited to type.

Argument shape deliberately does not pick a door, because it varies almost per method and grouping by it would give a door per method. The widest door carries what the others do not need, which is one string, two positions that are allowed to be absent, and a step.

The forty five names not spelled yet do not resolve. That is the same choice document 07 argues for: an absent name reads as unimplemented on the conformance board and a refusing one reads as a failure, and the second is a lie about a method nobody has written.

The accessor itself checks the column when it is built rather than when it is used, which is what pandas does and what `cat` does here. A caller who wrote `s.str` on a column of numbers asked for an attribute the object does not have, so `hasattr` should answer False rather than raise. The message is pandas' sentence with our vocabulary at the end: pandas names the type through `infer_dtype` and calls an int64 column `integer`, and inventing a second set of type names so that one message could read like pandas would be a bad trade.

## Two of the twelve are not in the extension

`index` and `rindex` are `find` and `rfind` that raise rather than answering -1. Where that exception is thrown is a pandas question and not a kernel one, so they are `find` and `rfind` followed by a scan of the answer, in the Python layer, and there is nothing in Mojo with those names.

This costs a full pass over the result before it is thrown away, and pandas pays the same cost for the same reason. The alternative is stopping at the first row without a match, which means having computed positions for the rows before it and having to decide what to do with them, and there is no answer to that a caller can use.

One detail there is not about strings at all. The generated method runs every error through `translate`, and an untagged `ValueError` is precisely what `translate` turns into a `RuntimeError`, on the argument in document 14 that an untagged error is a binding that forgot to classify something. So the class raised is `InvalidArgumentError`, which is one of ours and is also a `ValueError`, and a caller catching what pandas raises catches it.

## What the tuple and the na argument cost

`startswith` and `endswith` take a tuple of patterns, which is Python's own signature for them, and a fill value for the missing rows. Both are folds, and both are done over plain Python lists in the mixin rather than over columns, because the `Series` this library exposes has neither `|` nor `fillna` yet.

That is a real cost and it is written down in the mixin next to the code that pays it. Each of the two becomes one column operation the day those exist, and moving them down then is better than writing two kernels now that nothing else would call. A tuple of patterns is also not the common case, and the common case never reaches either fold.

## SQL asks the other question and gets the other answer

`.str.len()` counts characters and `STRLEN` counts bytes, and both of those are correct because they are two different questions from two different callers. DuckDB spells them apart as well: `length('café')` is 4 and `strlen('café')` is 5. So there are two kernels and not one with a flag, `text_character_length` in `chars.mojo` and `text_byte_length` in `substr.mojo`, and each lives with the file whose unit it shares.

They are a long way apart in cost, which is worth knowing before reaching for either. A byte length is four bytes out of every view and no indirection at all. A character length walks every byte of every element counting the ones that are not continuation bytes. On a million thirty two byte elements that is 254 microseconds against 3.3 milliseconds.

ClickBench q27 and q28 are what asked for the byte one. Both average `STRLEN` over a column of real web addresses, which have percent encoding and non ASCII in them, so the two counts differ on real rows rather than only in principle: the averages over the real `URL` column are 88.56 bytes and 86.57 characters. Answering the character question there would be a wrong answer to a benchmark query and not a defensible variation.

## What is left

Forty five names. They are not one more group, they are five or six, and each has an idea of its own that is worth landing on its own. Padding is a question about what a width means when a character is not a column. Case conversion is where Mojo's `String.lower()` on `İ` gives eight bytes and Python gives nine, which is a real divergence and needs a decision rather than a kernel. The ten `is` predicates need Unicode tables that the standard library does not expose. Splitting produces a column of lists, which is a type this library does not have. The regex methods are a dependency question before they are anything else.

None of those is blocked by this document. All of them are easier for it, because the thing they would each have had to decide first is decided: a position in a string is a character, the counting happens in one file, and the three doors are open.
