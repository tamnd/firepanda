# 112. A corpus of texts

## 1. What this is

Every regular expression differential this library has ever run has compared two engines over thirty thousand patterns and sixteen pieces of text. Document 90 section 10 asked for that second number to be a real one and nothing was done about it for twenty two slices. Documents 110 and 111 both ended by asking again, and document 111 put it plainly: a reading that is wrong only on a text nobody picked is a reading this differential calls right.

This slice makes the texts a corpus. Ninety six of them, in one file, read by all four oracles and by all four of the differentials that run a pattern rather than only parse one.

## 2. Why sixteen was the wrong number

Not because sixteen is small. Because of what the sixteen were for.

Each of them was handpicked against a difference somebody already knew about. The Arabic Indic digits are there because `\d` reads them and RE2 does not, the vertical tab and the non breaking space are there because Python calls them whitespace and RE2 does not, and the three with newlines are there for the dollar sign. That is a list of answers to questions that had already been asked, which makes it a regression test and not a search.

The patterns are the other way round. Thirty thousand of them are generated from a grammar, nobody picked them one at a time, and so every disagreement the differentials have ever reported was found on the pattern side. That is not because the pattern side is where the mistakes are. It is because the pattern side is the only side that was looking.

## 3. What is in the ninety six

The sixteen, first and unchanged, and then eighty generated from forty five fragments.

Keeping the sixteen verbatim and keeping them at the front is the part worth stating. Every answer the differentials have agreed on for twenty two slices was agreed on over those texts in that order, and a corpus that replaced them would be a different measurement wearing the same name. Widening a measurement should only ever add rows.

The fragments are in four groups and each group is there for a reason that can be named.

```
    ASCII        a Z 1 _ space - . newline
    whitespace   tab, vertical tab, non breaking space, line separator
    outside      é Ω 中 𝔸 ٣ д א ก あ 한 Ⅵ ² combining acute ß K İ ı ς Σ
    punctuation  { } [ ] : ( ) * \ + | ^ $ ?
```

The first group is there so an ordinary pattern has somewhere ordinary to match and so a word boundary has two sides to sit between. The second is whitespace the two engines are known to disagree about. The fourth is punctuation that means something in a pattern, which matters because the pattern corpus writes a great many patterns that are read as literal text and a literal brace has to be found in a row before anybody can say it was read right.

The third group is the one the whole file exists for, and it is in three parts. The first is about bytes: `é` and `Ω` and `٣` are two bytes, the Han and Thai and Hiragana and Hangul are three, and `𝔸` is four, so any of them beside a letter puts a byte boundary inside a character. RE2 runs on bytes and Python runs on code points, and that difference is invisible until a text is built this way. The second is about names, because document 103 measured a table of 196 of them and said plainly that the corpus exercised two of the 163 scripts in it, and these are nine scripts with a Roman numeral and a superscript two beside them for the two number categories a reader is most likely to get wrong. The third is the folding corners, the sharp s and the Kelvin sign and the dotted and dotless i and the two sigmas.

## 4. Where the list lives

`tools/regex_texts.py`, and nowhere else.

It used to live in `tools/regex_match_oracle.py`, with the other three oracles importing it from there, which worked and was wrong in the way that only shows up later. The match oracle is not the owner of the texts, it is one of four callers, and a list that lives inside one of its readers is a list that acquires a reason to be edited for that reader's sake.

The Mojo side now reads the same module rather than reading one oracle. Each of the four differentials had an `ask_texts` that imported a different oracle and called `texts()` on it, which was four paths to one list and four chances for one of them to be repointed alone. There is one path now.

## 5. The generator

A linear congruential step and a base forty five spelling of the result, about a dozen lines, and deterministic.

Deterministic because a differential whose input moves is one where a fix cannot be told from a reshuffle. Its own arithmetic rather than the standard library's for the same reason the pattern corpus has its own, which is that a corpus regenerated three years from now has to be the same corpus.

Each text is between one and five fragments. Duplicates are dropped, including duplicates of the handpicked sixteen, so the ninety six are ninety six distinct strings.

## 6. Why ninety six and not a thousand

Because the cost is a product and the coverage is not.

Every one of the six comparisons runs every pattern against every text, so a text corpus ten times wider is a differential run ten times longer, and the difference between a run somebody waits for and a run somebody abandons is roughly where this number is. On the other side, a text is at most five fragments drawn from forty five, so the interesting pairings, which are a fragment of one kind next to a fragment of another, are mostly covered well before a hundred texts and are not covered much better by a thousand.

The measured cost is smaller than that argument allows for. A cached count differential run takes about three and a half seconds over ninety six texts, because the crossing into Python is what the run is actually paying for and the crossing happens once per pattern either way.

## 7. What it found

Nothing.

Zero disagreements, on all six comparisons, on the first run and on every run since. Nine scripts, four disputed whitespace characters, a combining mark, five folding corners and fourteen pieces of pattern punctuation, a six fold widening of one side of every comparison this library makes, and not one answer moved.

```
    routing         compared 29566, held out 831
    str.contains    compared 29825, held out 572
    str.match       compared 29759, held out 638
    str.fullmatch   compared 29759, held out 638
    str.count       compared 29811, held out 586
    str.replace     compared 30137, held out 260
    str.findall     compared 29923, held out 474
```

That is the result and it is not a disappointment. It is the first evidence anybody has that the agreement the differentials report is agreement about the engines rather than agreement about sixteen strings. Before this slice the honest statement was that firepanda and pandas agree on thirty thousand patterns over sixteen texts somebody chose. It is now thirty thousand patterns over ninety six texts, eighty of which nobody chose.

Every one of those seven numbers is the number document 111 section 9 reported, to the pattern. That is the expected thing rather than a finding, because a pattern is held out for what the pattern is, so widening the texts cannot change which patterns are compared. It is worth writing down anyway, since a tally that had moved would have meant the corpus change had reached somewhere it had no business reaching.

## 8. What this is for

The next slice.

Document 111 section 10 left two families and said both of them wanted this first. The larger is the hundred and twenty four patterns where RE2 puts a non boundary between two bytes of one character, and that family cannot be measured at all over texts that are mostly ASCII, because a byte boundary inside a character needs a character with two bytes in it. There are now forty five fragments of which nineteen are outside ASCII and eighty generated texts built from them.

It was measured directly while this slice was being written. RE2 counts `\B` as 1 in the empty string, 0 in `a`, 3 in `é`, 3 in `aé`, 2 in `éa` and 4 in `中`, and a straightforward reading of a non boundary at every byte position does not reproduce those numbers, since it predicts 2 for `aé` and measures 3. So the scan rule is involved in the count and that family needs a measurement of its own before it needs a fix. It now has an instrument to be measured with.

## 9. What is left

The pattern side of the script question. Document 103 counted 163 script names in the table and the corpus exercises two of them, and this slice moved the text side to nine and left the pattern side exactly where it was. A `\p{Thai}` written against a Thai text is the pairing that would close it, and the pattern corpus does not write one yet.

The four patterns document 110 ended on are still four, unchanged and for the reason document 111 section 10 gave.
