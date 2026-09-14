"""`str.normalize`, checked against pandas.

The one name on this accessor that asks what a character is equivalent to rather
than what it is or what it maps to, and the only one whose rule is about
sequences rather than about characters one at a time.

It is also the one name here whose answers do not come from Arrow. pandas defines
it once, in `ObjectStringArrayMixin`, as `unicodedata.normalize` applied a row at
a time, nothing overrides it, and Arrow has no normalization kernel for anything
to override it with. So on every backend pandas has, including the Arrow backed
one this file compares against, this name is CPython's answer. That is the
opposite of the case predicates next door, where Arrow is the authority and the
standard library is the one that is wrong, and it is worth knowing before reading
a test file that looks like the others.

Every row and every expected answer is written as code point escapes rather than
as characters. Half of them are a letter followed by two combining marks and are
unreadable either way, but that is not the reason. The reason is that a source
file is text and text can be normalized. A file saved by an editor that tidies on
the way out has no decomposed rows left in it, and every test here would go on
passing while testing nothing at all. Escapes cannot be tidied.

The refusals at the bottom are pandas' own, straight out of
`unicodedata.normalize`, and there are two kinds of them. A name that is not one
of the four is a `ValueError` and a form that is not a string at all is a
`TypeError`. The lower case spelling is a `ValueError` and not an alias, which is
the one a caller is most likely to hit.
"""

from __future__ import annotations

import importlib.util
import unicodedata
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

FORMS = ("NFC", "NFD", "NFKC", "NFKD")

ROWS = [
    "\u00e9",
    "e\u0301",
    "\ufb01",
    "\u2460",
    "\u01c5",
    "a\u0307\u0323",
    "a\u0328\u0301",
    "q\u0301\u0300",
    "\uac01",
    "\u1100\u1161\u11a8",
    "\u212b",
    "\u0958",
    "\u1e9b\u0323",
    "\u0130",
    "\uff21\uff22",
    None,
    "",
    "plain ascii",
    "\u65e5\u672c",
]
"""Nineteen rows, each of them a different way for this to be wrong.

The first two are an e with an acute written both ways, which is the smallest
thing that distinguishes the four forms from doing nothing. The fi ligature and
the circled one move only under the K forms. The title case digraph moves under
the K forms and does not move to the same place under both. The next two rows are
a letter with two marks, one pair needing a sort and one pair where the sort
leaves the second mark blocked from the letter, and the row after them has two
marks of the same class, which catches a sort that is not stable. The two Hangul
rows are the arithmetic in both directions. The angstrom sign is a singleton and
composes into a character other than the one it came from, and the Devanagari qa
is on the composition exclusion list, so it comes apart and stays apart. The long
s with two marks is the example UAX 15 uses to show the four forms are four. The
dotted capital I comes apart and goes back to itself. The fullwidth letters are a
compatibility difference with no marks in it at all. Then a missing row, an empty
row, a row of plain ASCII that has to come back untouched, and two characters
that are three bytes each with nothing to normalize.
"""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds it by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


def mine(column: Any) -> list[Any]:
    """A firepanda column read out row by row."""
    return [column[i] for i in range(len(column))]


def readable(values: list[Any]) -> list[Any]:
    """pandas' missing value written the way firepanda writes it.

    The one difference between the two answers anywhere on this page, and it is
    not about this method. pandas reads a missing text value out as a float nan
    and firepanda reads it out as None, which document 63 records and which is
    the same on every name in the accessor. Flattening it here keeps the
    comparison about normalization, and the test below keeps the difference
    itself in view rather than hiding it.
    """
    return [None if value is None or value != value else value for value in values]


@needs_pandas
def test_every_form_matches_pandas_on_every_row(firepanda: ModuleType) -> None:
    """The whole of it, four forms against nineteen rows."""
    for form in FORMS:
        assert mine(made(firepanda).str.normalize(form)) == readable(
            list(theirs().str.normalize(form))
        )


@needs_pandas
def test_the_missing_row_is_the_only_thing_written_differently(
    firepanda: ModuleType,
) -> None:
    """Stated once, so that `readable` above is not quietly hiding anything.

    Both libraries agree the row is missing. They disagree about what a missing
    text value looks like when it is read out into Python, which is a property
    of the library rather than of this method.
    """
    answer = made(firepanda).str.normalize("NFC")
    theirs_answer = theirs().str.normalize("NFC")
    assert answer[15] is None
    assert theirs_answer.iloc[15] != theirs_answer.iloc[15]
    assert answer.isna().sum() == 1
    assert theirs_answer.isna().sum() == 1


@needs_pandas
def test_every_form_matches_the_standard_library(firepanda: ModuleType) -> None:
    """Against `unicodedata` directly, which is where pandas gets it.

    Comparing against pandas alone would agree with pandas about any row pandas
    was itself wrong about. This asks the library underneath both of us.
    """
    for form in FORMS:
        answers = mine(made(firepanda).str.normalize(form))
        for row, answer in zip(ROWS, answers, strict=True):
            if row is None:
                assert answer is None
            else:
                assert answer == unicodedata.normalize(form, row)


@needs_pandas
def test_the_two_spellings_of_one_letter_become_one(firepanda: ModuleType) -> None:
    """Which is the whole point, stated as an assertion rather than as a table."""
    both = ["\u00e9", "e\u0301"]
    for form in FORMS:
        answers = mine(made(firepanda, both).str.normalize(form))
        assert answers[0] == answers[1]
    assert mine(made(firepanda, both).str.normalize("NFC")) == ["\u00e9", "\u00e9"]
    assert mine(made(firepanda, both).str.normalize("NFD")) == [
        "e\u0301",
        "e\u0301",
    ]


@needs_pandas
def test_the_k_forms_are_the_ones_that_throw_formatting_away(
    firepanda: ModuleType,
) -> None:
    """A ligature, a circled digit and a fullwidth letter, untouched by the other two."""
    rows = ["\ufb01", "\u2460", "\uff21"]
    assert mine(made(firepanda, rows).str.normalize("NFC")) == rows
    assert mine(made(firepanda, rows).str.normalize("NFD")) == rows
    assert mine(made(firepanda, rows).str.normalize("NFKC")) == ["fi", "1", "A"]
    assert mine(made(firepanda, rows).str.normalize("NFKD")) == ["fi", "1", "A"]
    assert list(theirs(rows).str.normalize("NFKC")) == ["fi", "1", "A"]


@needs_pandas
def test_the_title_case_digraph_takes_two_different_steps(
    firepanda: ModuleType,
) -> None:
    """Its NFKD is three characters and its NFKC is two, and neither is one.

    The compatibility decomposition is a D, a z and a caron. Composing that
    gives a D and a z with caron, because there is no single character for the
    pair, so the two K forms move it and they do not move it to the same place.
    """
    rows = ["\u01c5"]
    assert mine(made(firepanda, rows).str.normalize("NFKD")) == ["Dz\u030c"]
    assert mine(made(firepanda, rows).str.normalize("NFKC")) == ["D\u017e"]
    assert mine(made(firepanda, rows).str.normalize("NFC")) == rows


@needs_pandas
def test_marks_are_put_in_canonical_order(firepanda: ModuleType) -> None:
    """A dot above written before a dot below comes back the other way round.

    The dot below is combining class 220 and the dot above is 230, so the order
    is decided by the classes and not by which was typed. Only after the swap
    can the dot below reach the letter, which is why the composed answer is two
    characters rather than three.
    """
    rows = ["a\u0307\u0323"]
    assert mine(made(firepanda, rows).str.normalize("NFD")) == ["a\u0323\u0307"]
    assert mine(made(firepanda, rows).str.normalize("NFC")) == ["\u1ea1\u0307"]
    assert list(theirs(rows).str.normalize("NFD")) == ["a\u0323\u0307"]


@needs_pandas
def test_a_mark_can_be_blocked_from_the_letter_behind_it(
    firepanda: ModuleType,
) -> None:
    """The ogonek is class 202 and the acute is 230, so the acute stays loose.

    Without the blocking rule the acute would reach the a instead and the answer
    would depend on the order the two marks were written in, which is exactly
    what normalizing is meant to stop it depending on.
    """
    rows = ["a\u0328\u0301"]
    assert mine(made(firepanda, rows).str.normalize("NFC")) == ["\u0105\u0301"]
    assert list(theirs(rows).str.normalize("NFC")) == ["\u0105\u0301"]


@needs_pandas
def test_two_marks_of_the_same_class_keep_their_order(firepanda: ModuleType) -> None:
    """Which is what a stable sort means, and it is a different string if not."""
    rows = ["q\u0301\u0300"]
    assert mine(made(firepanda, rows).str.normalize("NFD")) == ["q\u0301\u0300"]
    assert list(theirs(rows).str.normalize("NFD")) == ["q\u0301\u0300"]


@needs_pandas
def test_a_hangul_syllable_comes_apart_and_goes_back(firepanda: ModuleType) -> None:
    """Both directions, and neither one costs a table."""
    rows = ["\uac01", "\uac00"]
    assert mine(made(firepanda, rows).str.normalize("NFD")) == [
        "\u1100\u1161\u11a8",
        "\u1100\u1161",
    ]
    assert mine(made(firepanda, rows).str.normalize("NFC")) == rows
    assert mine(made(firepanda, ["\u1100\u1161\u11a8"]).str.normalize("NFC")) == ["\uac01"]
    assert list(theirs(rows).str.normalize("NFD")) == [
        "\u1100\u1161\u11a8",
        "\u1100\u1161",
    ]


@needs_pandas
def test_an_excluded_pair_comes_apart_and_stays_apart(firepanda: ModuleType) -> None:
    """Devanagari qa, which has a decomposition and no way back.

    It comes apart under all four forms and nothing puts it back, because the
    pair is on the composition exclusion list. A composition table built by
    reading every two character decomposition without applying the exclusions
    would answer the input here instead.
    """
    rows = ["\u0958"]
    for form in FORMS:
        assert mine(made(firepanda, rows).str.normalize(form)) == ["\u0915\u093c"]
        assert list(theirs(rows).str.normalize(form)) == ["\u0915\u093c"]


@needs_pandas
def test_a_singleton_composes_into_a_different_character(
    firepanda: ModuleType,
) -> None:
    """The angstrom sign comes back as the ordinary letter it looks like."""
    rows = ["\u212b"]
    assert mine(made(firepanda, rows).str.normalize("NFC")) == ["\u00c5"]
    assert mine(made(firepanda, rows).str.normalize("NFD")) == ["A\u030a"]
    assert list(theirs(rows).str.normalize("NFC")) == ["\u00c5"]


@needs_pandas
def test_nfc_and_nfkc_can_be_different_characters(firepanda: ModuleType) -> None:
    """The long s with a dot above, followed by a dot below.

    Under NFC the long s keeps its own dot and the other stays loose. Under NFKC
    the long s has already become an ordinary s, so both dots land on it and the
    answer is a single character the NFC answer has no route to. This is the row
    that fails in an implementation which treats the K forms as a decomposition
    difference and nothing else.
    """
    rows = ["\u1e9b\u0323"]
    assert mine(made(firepanda, rows).str.normalize("NFC")) == ["\u1e9b\u0323"]
    assert mine(made(firepanda, rows).str.normalize("NFKC")) == ["\u1e69"]
    assert mine(made(firepanda, rows).str.normalize("NFD")) == ["\u017f\u0323\u0307"]
    assert list(theirs(rows).str.normalize("NFC")) == ["\u1e9b\u0323"]
    assert list(theirs(rows).str.normalize("NFKC")) == ["\u1e69"]


@needs_pandas
def test_a_missing_row_stays_missing(firepanda: ModuleType) -> None:
    """Unlike `get_dummies`, which is the one name here that fills one in."""
    for form in FORMS:
        assert made(firepanda).str.normalize(form).isna().sum() == 1
        assert theirs().str.normalize(form).isna().sum() == 1


@needs_pandas
def test_an_empty_row_stays_empty(firepanda: ModuleType) -> None:
    """Normalization never removes the last character and never invents one."""
    for form in FORMS:
        assert mine(made(firepanda, [""]).str.normalize(form)) == [""]
        assert list(theirs([""]).str.normalize(form)) == [""]


@needs_pandas
def test_ascii_comes_back_untouched(firepanda: ModuleType) -> None:
    """The fast path, which skips the work because no ASCII character moves."""
    rows = ["plain ascii", "A_z0", "\ttab\n", "~!@#$%^&*()"]
    for form in FORMS:
        assert mine(made(firepanda, rows).str.normalize(form)) == rows
        assert list(theirs(rows).str.normalize(form)) == rows


@needs_pandas
def test_every_form_is_idempotent(firepanda: ModuleType) -> None:
    """Normalizing an answer again gives the same answer.

    The property the whole thing exists for. Anything failing it would mean two
    strings a reader calls the same could still come out as different bytes.
    """
    for form in FORMS:
        once = made(firepanda).str.normalize(form)
        twice = once.str.normalize(form)
        assert mine(twice) == mine(once)


@needs_pandas
def test_a_form_that_is_not_one_of_the_four_is_refused(firepanda: ModuleType) -> None:
    """pandas refuses it too, from inside `unicodedata`."""
    import pandas as pd

    with pytest.raises(ValueError, match="normalization form"):
        made(firepanda).str.normalize("XYZ")
    with pytest.raises(ValueError, match="normalization form"):
        pd.Series(["a"], dtype="str").str.normalize("XYZ")


@needs_pandas
def test_the_lower_case_spelling_is_refused_and_not_an_alias(
    firepanda: ModuleType,
) -> None:
    """The refusal a caller is most likely to reach, and it is pandas' own."""
    import pandas as pd

    with pytest.raises(ValueError, match="normalization form"):
        made(firepanda).str.normalize("nfc")
    with pytest.raises(ValueError, match="normalization form"):
        pd.Series(["a"], dtype="str").str.normalize("nfc")


@needs_pandas
def test_a_form_that_is_not_a_string_is_refused_as_a_type_error(
    firepanda: ModuleType,
) -> None:
    """A different kind of refusal from the one above, and pandas agrees."""
    import pandas as pd

    for bad in (1, None, ["NFC"]):
        with pytest.raises(TypeError):
            made(firepanda).str.normalize(bad)
        with pytest.raises(TypeError):
            pd.Series(["a"], dtype="str").str.normalize(bad)


@needs_pandas
def test_normalize_on_a_column_that_is_not_text_is_refused(
    firepanda: ModuleType,
) -> None:
    """The accessor refuses before the method is reached, which is pandas."""
    with pytest.raises(AttributeError):
        firepanda.Series([1, 2, 3]).str.normalize("NFC")


@needs_pandas
def test_a_column_with_no_rows(firepanda: ModuleType) -> None:
    """A typed empty text column, reached by cutting one down."""
    for form in FORMS:
        assert len(made(firepanda, ["a"]).head(0).str.normalize(form)) == 0
        assert len(theirs(["a"]).head(0).str.normalize(form)) == 0
