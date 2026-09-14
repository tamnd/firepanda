# 75. The one name answered by the standard library

## 1. A different kind of question

Every other name on the `str` accessor asks something about a character. Is it a digit, what is it in upper case, where does it sit in the row. `normalize` asks what a character is equivalent to, and equivalence between characters is not a property a character has on its own.

Unicode allows the same text to be spelled more than one way. An e with an acute accent is U+00E9, or it is a plain e followed by U+0301, and both are correct. They print the same, they mean the same, and they are not the same bytes, so they do not compare equal, they do not hash alike, and a group by on a column holding both puts them in different groups. Normalization picks one of the spellings. Two strings a reader would call the same string come out of it as the same bytes.

This is also the only name on the accessor whose rule is about sequences. Everything else can be answered a character at a time in any order. Here the answer for a character depends on which characters are next to it, which is why the kernel decodes a whole element before it writes anything and why the tests are about rows of three characters rather than about characters.

## 2. Which library is the authority, and why it is not Arrow

pandas holds text in Arrow, and most of this accessor is answered by an Arrow compute kernel. When Arrow and Python's standard library disagree, as they do about what a digit is and about what upper case means for thirty nine code points, Arrow is what pandas answers and Arrow is therefore what this library copies. Document 64 is that argument and the case kernels are built on it.

`normalize` is the other way round. It is defined once, in `ObjectStringArrayMixin`:

```python
def _str_normalize(self, form):
    f = lambda x: unicodedata.normalize(form, x)
    return self._str_map(f)
```

Nothing overrides it, `ArrowStringArray` does not have its own, and there is no Arrow normalization kernel for anything to override it with. On every backend pandas has, including the Arrow backed one that is the default in pandas 3, this name is CPython's answer applied a row at a time.

So the rule is not that Arrow is the authority for this accessor. The rule is that the authority for a name is whichever library pandas actually calls for it, found by reading, and it happens to be Arrow for most of them. Getting that backwards here would have produced a kernel built on the Unicode tables Arrow ships, which are a different version from the ones CPython ships, and the disagreements would have been in exactly the rare characters nobody writes a test for.

## 3. Four forms, which are two choices

There are four forms because there are two independent decisions, and naming them that way makes the four fall out rather than having to be memorised.

The first decision is which equivalence is meant. Canonical equivalence says two spellings are the same text, which is the accented e. Compatibility equivalence is wider and says two spellings are the same content with the formatting discarded, which turns the fi ligature into two letters, the circled one into a one, and the fullwidth Latin letters into ordinary ones. Compatibility is the K in the name and it loses information: a reader who cared that the one was in a circle does not get that back.

The second decision is whether to finish by taking characters apart or by putting them back together. D is apart and C is together.

So NFD is canonical and apart, NFC is canonical and together, NFKD and NFKC are the same two under the wider relation. The kernel takes the two decisions as two flags rather than taking a form name, and the name is read into them once in the Python layer.

NFC is the one worth knowing. It is what the web platform normalizes to, it is what most text is already in, and it is the form in which the accented e is one character.

## 4. Why the tables are expanded before they are written

A decomposition can be recursive. A character decomposes to two, one of which decomposes again, and a naive kernel walks that chain at run time with a depth nobody has bounded.

The generator applies every mapping until nothing maps any more, so one lookup answers the final sequence and the kernel has no recursion in it at all. It costs almost nothing: the canonical table is 2061 characters coming to 3406 code points and the compatibility table is 5857 coming to 9112. The longest canonical expansion is four characters and the longest compatibility one is eighteen.

That is a small enough difference in size that the argument for it is not space. The argument is that a depth which is shallow is still a depth, and the version of this kernel that walks the chain has a question in it about what to do if the chain does not end, which the version that does one lookup does not have.

## 5. Why the composition table is derived rather than read

Unicode does not put every decomposition back together. Some pairs are on the composition exclusion list, some characters have a singleton decomposition and composing would not give them back, and some decompositions begin with a character that is not a starter. Three rules, all of them fiddly, and reimplementing them out of the data files is a second chance to be wrong about the same thing.

The generator does not read the exclusion list. It takes every character whose canonical decomposition is exactly two, hands those two to CPython, and keeps the pair only when CPython gives the character back. What comes out is by construction the set of pairs CPython composes, which is the only set that matters, and all three rules are applied without this code knowing what any of them are.

Of 1026 candidate pairs, 941 survive. The 85 that do not are the three rules, and none of them had to be written down.

## 6. Hangul, which costs no table

A Korean syllable is a lead consonant, a vowel and an optional trailing consonant packed into one code point by a formula. Taking one apart is a division and putting one together is a multiplication, so the 11172 syllables are absent from CPython's decomposition table and are absent from the generated one for the same reason.

That is worth stating because it is the largest thing the file does not contain. A table holding them would be three times the size of everything else here.

## 7. The two elements that skip the work

An element that is entirely ASCII is already in all four forms and is copied straight through. No ASCII character has a decomposition of either kind, none has a combining class other than zero, and no pair of ASCII characters composes into anything. The check is one pass looking for a byte at or above 0x80, and it is checked in the generator rather than asserted here.

An element that is not valid UTF-8 is also copied through. There is nothing to normalize, and the alternative is a decode that trusts a lead byte and reads off the end of a truncated sequence into the next element, because a text column's payload is one buffer with the elements laid end to end. That is the same rule the case kernels follow for the same reason.

## 8. The rows that separate a right implementation from a plausible one

Six, and each of them fails differently.

Two combining marks in the wrong order catch a missing canonical sort. Two marks of the same class catch a sort that is not stable, which is a different bug with the same shape.

A letter, a mark of class 202 and a mark of class 230 catch a missing blocking rule in the composition. The letter and the first mark compose, the second mark cannot then reach the composite because the first sits between them with a class that is not lower, and without the rule the answer would depend on which mark was typed first, which is exactly what normalizing is meant to stop it depending on.

The angstrom sign is a singleton: it decomposes to an A with a ring and composes back to the ordinary letter, not to the sign it started as. Devanagari qa has a decomposition and is on the exclusion list, so it comes apart under all four forms and nothing puts it back. Both of those pass in an implementation that only asks whether composing after decomposing gives the input back.

And the long s with a dot above followed by a dot below, which is the example UAX 15 uses. Under NFC the long s keeps its own dot and the other stays loose. Under NFKC the long s has already become an ordinary s, so both dots land on it and the answer is one character the NFC answer has no route to. An implementation treating the K forms as a decomposition difference and nothing else is right everywhere except here.

## 9. Both refusals, and they are pandas' own

`unicodedata.normalize` refuses a form it does not know with `ValueError: invalid normalization form` and refuses anything that is not a string with a `TypeError`. Both are reproduced, with firepanda's own wording and its own error classes, which are subclasses of those two.

The lower case spelling is the one a caller actually reaches. `form="nfc"` is a `ValueError` and not an alias, in pandas and here.

The form is read twice, once in Python and once in the Mojo door. That is not belt and braces, it is that the Mojo API is a caller too, and an error raised from the kernel with no kind on it reaches Python as something other than the `ValueError` pandas raises.

## 10. The one difference, which is not about this method

A missing row stays missing under all four forms, and both libraries agree that it does. They write it differently when it is read out into Python: firepanda gives `None` and pandas gives a float nan. That is recorded in document 63, it is the same on every name in the accessor, and it is the only place the answers here differ from pandas at all.

It is worth contrasting with `get_dummies`, which is the one name on this accessor where a missing row does not stay missing. That follows from the answer being counts. Here the answer is text, and there is no normalization of an unknown string.

## 11. What this does not do yet

Nothing is measured about speed. The general path decodes an element into a list of code points, sorts a short run, walks it once more and encodes, which is four passes and two allocations that the ASCII path avoids entirely, and no benchmark says what that costs on a column that is mostly not ASCII. The comparison worth making is against `text_case`, which faces the same choice and answers it with a per element fast path measured in issue 756.

Nothing here is reused by anything else, and two things arguably should reuse it. Comparing two text columns for equality could normalize first, and grouping by a text column could too. Both would be wrong to do by default, since pandas does neither, but both are the kind of thing a caller asks for and neither has anywhere to be asked for yet.
