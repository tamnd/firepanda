# 82. An answer whose width is in the pattern

## 1. What this is

`str.extract` pulls the capture groups of the first match out of every row and answers one column per group. It is the first of the three methods document 81 left unwired, and it is the first name on the `str` accessor whose answer is wider than a column and whose width is neither a constant nor a property of the data. Issue #8 M6.

The engine it runs is the one document 81 built. Nothing in the machine changed for this slice. What is new is everything above it: a kernel that answers several columns, a door in `firepanda/py/text.mojo` that no existing door fitted, a label that travels on the compiled program, and a Python layer that has to decide between a frame and a column.

## 2. The fourth shape, and the first one settled before a row is read

Document 70 was about an answer wider than a column, document 71 about one narrower, and document 72 about one whose width is in the data. Those three plus the plain column are the shapes this accessor had. This is the fourth.

`partition` always answers three columns and `cat` always answers one string, so their widths are constants written into the method. `get_dummies` answers however many distinct tokens the column turned out to hold, so its width is a property of the data and cannot be known until the column has been read once, which is why that one takes two passes and two functions.

`extract` is neither. `s.str.extract(r"(a)(b)")` answers two columns and `s.str.extract(r"(a)")` answers one, for the same column, and the number is fixed the moment the pattern is compiled. That is a width that varies per call and is still known before the first row is touched, which is the useful half of both previous cases and the awkward half of neither. One pass, one function, and the builders are allocated to the right height and the right count before the loop starts.

This is the same rule the file has always had, working for the fourth time: the shape of the answer picks the door, and a frame whose width comes from the pattern is a shape no door carried. It is worth saying that the rule keeps deciding these questions without an argument, because the alternative each time was a flag on an existing door, and four flags would by now be a function with a branch at the end.

## 3. Which engine reads the pattern, which is not a choice this method makes

Document 76 built a router that reads a pattern and decides which engine pandas would send it to. `extract` never consults it.

That is not an omission. pandas does not consult it either: `contains`, `match`, `fullmatch`, `count` and `replace` ask whether the pattern holds something Arrow cannot do and fall back to Python when it does, and `extract` has no Arrow path at all to fall back from. It compiles with `re` and loops, for every pattern, always. So the compiled program here is built for Python's engine unconditionally, and the one line in `program_for` that says so sits above the router call rather than inside it.

Three consequences follow and all three are visible from Python. `\w` is 138558 code points here and 63 for `str.count` on the same accessor in the same session, which is the assertion at the bottom of both test files. A pattern holding syntax only RE2 refuses, a comment group or a `\u` escape, is compiled rather than refused, because refusing it would be refusing on behalf of an engine nobody asked. And a pattern holding a construct neither engine has yet, a lookaround or a backreference, is refused with the gap flag set and the sentence that says this engine has none yet, rather than with the sentence that says RE2 has none. That distinction was built in document 81 and this is the first method where only one side of it can ever fire.

`match` and `fullmatch` anchor their pattern before compiling and `extract` does not, for the same kind of reason: upstream runs `regex.search` here, so a pattern with no anchor finds its match in the middle of a row. The two behaviours sit on the same accessor with the same pattern and answer differently, which the Python test asserts side by side so that a reader meeting it for the first time sees both.

## 4. Three rules about missing, which look like one until a row disagrees with itself

A row with no match is missing in every column. A row that is itself missing is missing in every column. A group that took no part in a match that did happen is missing on its own.

The first two are the same answer reached by different routes, and only one of them runs the engine. The third is the one that makes the answer more than a match flag repeated across the width: `(a)(x)?` over a row holding `a` matched, so the first column holds `a`, and the optional group was never entered, so the second column holds nothing. That is the only case where the columns of one row disagree about whether there was a match, and a kernel that wrote the row's match state across the width would pass every test that did not include it.

There is a fourth state that reads like missing and is not. A group that matched the empty string took part in the match and holds the empty string, which is a value. `(a*)b` over a row holding `b` answers a column that is valid and empty rather than a column that is null, and the two are different cells in both libraries. The kernel tells them apart by asking whether the group's opening slot was ever written, not by asking whether the slice it names has any width in it.

## 5. The labels ride on the compiled program

A caller putting these columns into a frame needs to know what to call them, and what to call them is in the pattern rather than in the answer. `(?P<letter>[a-z])(\d)` names one of its two groups.

The parser already knew this. It keeps the names it saw and the group numbers they belong to as two lists as long as however many groups were named, which is what Python calls `groupindex` written the other way round. Two lists rather than one because a name on its own does not say which group it is: the pattern above opens two groups and names one, so the names list is one long and the number it holds is two.

What a caller labelling columns wants is different: one entry per group whether it was named or not, in the order the groups were opened. So the compiler turns the one into the other, once, at the end of `compile_program`, and the result rides on `Program.labels`. The alternative was to parse the pattern a second time in the Python layer, which is the work this whole package exists to do once per column rather than once per anything.

## 6. A name or a position, and a divergence that is already registered

pandas labels an unnamed group with its own position counted from zero and a named one with its name. `(?P<letter>[a-z])(\d)` gives `letter` and `1`, and the `1` is an integer while the `letter` is a string, in the same index.

A firepanda frame holds text column labels, so the integer comes back as `"1"`. That is the divergence `partition` already carries, where the three columns are labelled `"0"`, `"1"` and `"2"` against pandas' `0`, `1` and `2`, and it is not a new one. The fix is a column label type and not a string this method could write differently, and until that exists writing the position out as text is the closest thing to the answer that this library can hold.

One thing that could have been a collision is not one. A group name has to be a Python identifier and a position written out is digits, so a name can never be mistaken for a position or shadow one, whatever mixture of named and unnamed groups a pattern holds.

## 7. Where the column gets its name, which is the one place it is not the input's

With one group and `expand=False` the answer is a column rather than a frame, and what that column is called took a measurement rather than a guess.

If the group has a name, the column is named after the group. That is the only place on this accessor where the answer is not named after the column it was called on, and it is right: a caller who named a group asked for that name.

If the group has no name, the column keeps the name of the column that was read. It is not renamed to nothing. Upstream reaches that by handing `None` to its own wrapper, which reads an absent name as leave it alone, and the two cases therefore answer two different names for the same input column with two patterns that differ only in a name that neither answer contains. It is measured on both sides in the same test, because it is exactly the kind of rule that a reimplementation gets backwards by writing the obvious thing.

## 8. What is checked where, and in which order

`expand` is checked first, before the pattern is looked at, because pandas checks it first. A caller who wrote `expand=None` and a pattern with no groups is told about `expand`. The sentence is pandas' own.

The pattern is compiled next and a pattern with no capture groups is refused after that, again because that is upstream's order: `re.compile` runs and then `regex.groups` is read, so a pattern that is both ungrouped and malformed is a syntax error rather than a complaint about groups. Reproducing the order matters more than it looks, since it is the only thing that decides which of two errors a caller sees.

Both of those live on the Mojo side of the groups check and the Python side of the `expand` check, which is not an inconsistency: the groups check needs a parsed pattern and the parser is in Mojo, and `expand` is an argument that never crosses because it changes the shape of the answer rather than the answer.

`flags` is refused, the same refusal the five pattern methods above this one make and for the same reason: every flag is a statement about a regular expression, the scoped form of them is not carried on the node yet, and an ignored argument is the one failure mode a compatibility layer must not have.

## 9. Why this one is serial, and where the offsets are counted

The answers go into builders. A builder is one buffer with one cursor, and handing four threads a share of one is a different design rather than a flag, which is the same sentence `text_replace_regex` carries and the same reason. Document 80 has what closes it, which is a builder per morsel and a join, and this kernel joins the queue for that rather than adding a second reason.

Everything that can be paid once per column still is. The pattern is compiled before the first row. The machine, the decode buffer, the byte offsets and the capture slots are made once and handed to every row.

The engine walks code points and the answer is cut in bytes, so every row builds a table from code point position to byte offset before the match is asked for. That table is the thing a test of this kernel has to reach: an implementation that used the code point positions directly answers correctly for every all ASCII row and answers the wrong half of the row for anything else. Both test files therefore hold a row where every character is more than one byte wide, which document 72 section 7 already stated as a rule for this accessor and which this slice is the fourth to need.

## 10. What is not here yet

`findall` and `extractall` are the other two methods that never reach Arrow, and neither is wired. `findall` wants a list column, which does not exist, and its scan is `re.finditer`'s, which is neither the counting loop document 79 measured nor the replacing loop document 80 measured, so it is a third scan rather than a third caller of an existing one. `extractall` wants a frame with a `MultiIndex` whose names are the input's and `match`, and it drops the rows that did not match, so it needs both the list of matches `findall` needs and an index type this library does not have.

`expand=False` with more than one group answers a frame here and upstream, so there is no gap there. `expand=False` on an `Index` with more than one group is a refusal upstream and there is no `Index.str` here to refuse from.

The five constructs the router sends to Python are still refused by both engines: lookaround, backreference, conditional, atomic group and possessive quantifier. Case folding is two tables rather than one, because the two engines fold different alphabets, and it is the largest single piece of the differential's hold out. Scoped flags are not carried on the node. None of those five is a shortfall of this method and all five now report themselves in this engine's own voice, which is the part document 81 put in place.
