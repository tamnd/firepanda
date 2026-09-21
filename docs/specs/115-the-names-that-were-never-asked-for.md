# 115. The names that were never asked for

## 1. What this is

Document 103 wrote the table that `\p{...}` is read against, 196 names measured one at a time against the RE2 inside pyarrow, and it ended by saying that the corpus exercises two of the 163 script names in it. That sentence has been repeated at the end of every document since 110, and the text corpus of document 112 moved the text side to nine scripts and left the pattern side where it was.

This slice writes the patterns. It found that four of RE2's names are not in the table at all, and that this library has been telling callers RE2 has no such character class about classes RE2 has.

## 2. What was missing

The separators, which are `Zs` and `Zl` and `Zp` and the `Z` they make up.

```
    \p{Zs}    the space and the no break space and fifteen others
    \p{Zl}    U+2028, and nothing else
    \p{Zp}    U+2029, and nothing else
    \p{Z}     the three of them together
```

RE2 reads all four. pandas hands a pattern holding one to Arrow and Arrow answers a column. This library refused every one of them with `RE2 has no such character class`, which is a sentence about RE2 that was not true.

Measured rather than assumed, the same way everything else about this engine has been: `Series.str.contains(r"\p{Zs}")` over a row holding a space answers True in pandas 3.0.5 and raised here.

## 3. How it got missing

The generator asks RE2 about each name on a list, and the list of general categories was written out by hand in a dictionary mapping each one letter category to the two letter ones it is the union of. Letter, mark, number, punctuation, symbol, other. Somebody wrote the six they could think of and the seventh is the one nobody thinks of, because a separator is what is between the things a person is looking at rather than one of them.

Nothing downstream could catch it. The table is the only thing that knows which names exist, the parser asks the table, and a name the table has never heard of is a name that gets refused with a message the table is entitled to give. The differentials could have caught it, and did not, because the corpus never wrote a separator name: it writes `\p{L}` and `\p{Lu}` and `\p{Nd}` and `\p{Greek}` and `\p{Arabic}`, and `\p{Foo}` for a name that really does not exist, and that is the whole of it.

That is the failure worth naming. This was not a mistake in an algorithm, it was a list somebody wrote from memory in a generator, in a repository where nearly everything else is measured, and it survived because the thing that would have measured it was a corpus that had been asked to cover names and had never been widened to.

## 4. The list is gone

The one letter categories still name their parts, because that union is checked against RE2 rather than trusted and checking it is worth keeping. What is new is that the generator no longer believes the list is complete.

It now asks RE2 about every name of one letter and every name of a letter followed by a lower case letter, which is 702 patterns that compile and never run, and it refuses to write anything if RE2 takes a name it was never told about. Two letter script names are caught in the same net, which is why the comparison is against the scripts as well as the categories, and `Yi` is the one that makes that necessary.

The cost of that sweep is a second, against a table whose measurement is around two hundred calls over a column of 1112064 rows. It should have been there from the beginning and the reason it was not is that nobody had a reason to doubt the list until the list was wrong.

Longer names are still a written list. A script name cannot be swept for, because there is no shape to enumerate, and the generator already fails if a name it was told about stops being one. What it cannot do is notice a script that RE2 has gained, and RE2 here turns down every script Unicode 16 added, so there is nothing to gain yet.

## 5. The second reader that was wrong

`re2.mojo`, the grammar reader that answers whether RE2 would read a pattern, consults the same table and refuses a name that is not in it. So `re2_reads(r"\p{Zs}")` said no, and RE2 says yes, and that is a second defect from the same missing rows.

It was fixed by the same four rows, which is the argument for one table read by both. It is worth writing down that the reader differential did not catch it either, and for exactly the reason the other six did not: the reader is asked about the corpus, and the corpus had no separator name in it.

## 6. The pattern side of the script question

Thirty eight patterns are added to the measured list, which is the part of the corpus that is written rather than generated and always asked.

Thirteen are the separators, one per name and then the ones that go somewhere else in the compiler: inside a class, complemented two ways, repeated, under the ignore case flag, beside a literal, written without braces as `\pZ`.

The rest are the pairing document 103 asked for. Nine scripts that the text corpus now holds a character of, which are Latin and Han and Cyrillic and Hebrew and Thai and Hiragana and Hangul and Inherited and Common, then the categories behind the characters that were put in that corpus for this, which are `Nl` for the Roman numeral and `No` for the superscript two and `Mn` for the combining acute and `Lo` for the five scripts that have no case, then six more categories that nothing in the corpus matches and that are there because a name that matches nothing is still a name the compiler has to read, and then six pairings of a name with something around it. A pattern naming a script is now run against text in that script rather than against sixteen strings that were all Latin or Greek or Arabic.

## 7. What the differentials say

```
    routing         compared 29604, held out 831
    str.contains    compared 30201, held out 234
    str.match       compared 30239, held out 196
    str.fullmatch   compared 30239, held out 196
    str.count       compared 30187, held out 248
    str.replace     compared 30187, held out 248
    str.findall     compared 29961, held out 474
```

All seven at ten thousand agreements in ten thousand with no disagreements, and the grammar reader differential too.

Every held out tally is the number document 114 reported, to the pattern, and every compared count is 38 higher. That is what a corpus widening should look like: nothing moved except the amount of evidence.

Nine scripts and four separators and four more categories, run through the compiler and both engines and the counting loop and the replace loop and the grammar reader, and the only answer that changed is the four that used to be an exception.

## 8. What is left

The script names that no text exercises. There are 163 of them and the text corpus holds a character of nine, so the other 154 are now written in a pattern and run against text none of them can match, which tests that the name is read and the ranges are emitted and proves nothing about whether the ranges are right.

The ranges came from RE2 one code point at a time, so they are right unless the table has gone stale against a later pyarrow, and nothing checks that today. Regenerating the table and finding the file unchanged is the check, it takes a few minutes, and it is not wired to anything. A differential that walks the edges of all 5820 ranges and asks RE2 about each one is the smaller version of the same idea and would run in seconds.

`Cn` is still not a name RE2 has, and `Any` is still written by hand as the whole space, and both are checked before anything is written.
