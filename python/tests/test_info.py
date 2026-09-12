"""What a frame and a column say about themselves when asked to print it.

`info` is formatting. Every number in it already existed as a member, so what these tests are
about is the layout, and the layout is copied from pandas down to the trailing spaces. Most of
the tests below therefore run pandas beside firepanda and compare the report line for line,
with the two lines that are allowed to differ taken out: the class, which names this library,
and the memory, which counts Arrow buffers where pandas counts a numpy representation.

The third difference is the spelling of a type, which only shows on a frame that has text or
dates in it, so the comparisons run over numbers and the spelling gets tests of its own.
"""

from __future__ import annotations

import importlib.util
import io
import re
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

NUMBERS: dict[str, list[Any]] = {"v": [1, 2, 3], "f": [1.5, None, 3.5]}
"""Numbers only, because both libraries spell these two types the same way."""

MIXED: dict[str, list[Any]] = {"k": ["p", "qq", None], "v": [1, 2, 3], "f": [1.5, 2.5, 3.5]}
"""Text as well, which is where the spelling parts company."""


def report(frame: Any, **kwargs: Any) -> list[str]:
    """The report as lines, which is what every test here reads."""
    buf = io.StringIO()
    frame.info(buf=buf, **kwargs)
    return buf.getvalue().splitlines()


def comparable(lines: list[str]) -> list[str]:
    """The report with the two lines that are allowed to differ taken out."""
    return [
        line
        for line in lines
        if not line.startswith("<class ") and not line.startswith("memory usage:")
    ]


# ---------------------------------------------------------------------------
# The report as a whole
# ---------------------------------------------------------------------------


@needs_pandas
@pytest.mark.parametrize("kwargs", [{}, {"show_counts": False}, {"verbose": False}])
def test_the_report_is_pandas_report_line_for_line(
    firepanda: ModuleType, kwargs: dict[str, Any]
) -> None:
    """Trailing spaces and all, which is the whole point of writing this member.

    The rule under the header is as long as the header rather than as long as the column,
    which looks like an oversight and is pandas' output exactly. A caller who diffs the two
    reports should see three lines of difference and no fourth.
    """
    import pandas as pd

    mine = report(firepanda.DataFrame(NUMBERS), **kwargs)
    theirs = report(pd.DataFrame(NUMBERS), **kwargs)
    assert comparable(mine) == comparable(theirs)
    assert len(mine) == len(theirs)


def test_the_class_line_names_this_library(firepanda: ModuleType) -> None:
    """The first line, which is the one difference nobody could mistake for a bug."""
    assert report(firepanda.DataFrame(NUMBERS))[0] == "<class 'firepanda.DataFrame'>"
    assert report(firepanda.DataFrame(NUMBERS)["v"])[0] == "<class 'firepanda.Series'>"


def test_a_text_column_spells_its_type_the_way_this_library_spells_it(
    firepanda: ModuleType,
) -> None:
    """`string` where pandas 3 says `str`, which is the registered dtype divergence."""
    lines = report(firepanda.DataFrame(MIXED))
    assert [line.split()[-1] for line in lines[5:8]] == ["string", "int64", "float64"]
    assert lines[-2] == "dtypes: float64(1), int64(1), string(1)"


def test_the_memory_line_is_the_frames_own_memory_usage(firepanda: ModuleType) -> None:
    """Not a second measurement, the one `memory_usage` already answers, added up."""
    frame = firepanda.DataFrame(MIXED)
    written = report(frame)[-1]
    assert written == f"memory usage: {float(frame.memory_usage().sum()):3.1f} bytes"


def test_a_big_frame_reports_its_memory_in_larger_units(firepanda: ModuleType) -> None:
    """Powers of 1024 with one decimal, which is pandas' ladder and its rounding."""
    frame = firepanda.DataFrame({"v": list(range(4000))})
    assert report(frame)[-1] == "memory usage: 31.7 KB"


def test_there_is_never_a_plus_after_the_memory(firepanda: ModuleType) -> None:
    """pandas puts one there when it left something out and there is nothing left out here."""
    assert "+" not in report(firepanda.DataFrame(MIXED))[-1]


# ---------------------------------------------------------------------------
# The line about the labels
# ---------------------------------------------------------------------------


def test_a_range_of_labels_says_range_index_and_its_two_ends(firepanda: ModuleType) -> None:
    """The ends rather than the count, because the count is on the same line already."""
    assert report(firepanda.DataFrame(NUMBERS))[1] == "RangeIndex: 3 entries, 0 to 2"


def test_an_index_with_nothing_in_it_names_no_ends(firepanda: ModuleType) -> None:
    """There is no first label to print, so the line stops after the count."""
    assert report(firepanda.DataFrame({"a": []}))[1] == "RangeIndex: 0 entries"


@needs_pandas
def test_labels_that_were_declared_say_index_and_their_first_and_last(
    firepanda: ModuleType,
) -> None:
    """The class name is read off the index rather than written down."""
    import pandas as pd

    data = {"k": ["p", "qq", "rrr"], "v": [1, 2, 3]}
    mine = report(firepanda.DataFrame(data).set_index("k"))[1]
    assert mine == "Index: 3 entries, p to rrr"
    assert mine == report(pd.DataFrame(data).set_index("k"))[1]


def test_a_missing_label_prints_as_the_value_that_is_there(firepanda: ModuleType) -> None:
    """`None` here and `nan` in pandas, which is what each library actually holds."""
    frame = firepanda.DataFrame(MIXED).set_index("k")
    assert report(frame)[1] == "Index: 3 entries, p to None"


def test_a_frame_of_no_columns_says_so_and_stops(firepanda: ModuleType) -> None:
    """No table, no types and no memory, which is pandas' shape for this one."""
    lines = report(firepanda.DataFrame({}))
    assert lines == ["<class 'firepanda.DataFrame'>", "RangeIndex: 0 entries", "Empty DataFrame"]


# ---------------------------------------------------------------------------
# The table of columns
# ---------------------------------------------------------------------------


def test_there_is_one_line_per_column_numbered_from_zero(firepanda: ModuleType) -> None:
    """The position, the name, the count and the type, in that order."""
    lines = report(firepanda.DataFrame(MIXED))
    assert lines[2] == "Data columns (total 3 columns):"
    assert [line.split()[0] for line in lines[5:8]] == ["0", "1", "2"]
    assert [line.split()[1] for line in lines[5:8]] == ["k", "v", "f"]


def test_the_counts_are_the_rows_that_are_not_missing(firepanda: ModuleType) -> None:
    """A column with a gap in it says one fewer, which is what `count` says."""
    lines = report(firepanda.DataFrame(NUMBERS))
    assert re.findall(r"(\d+) non-null", "\n".join(lines)) == ["3", "2"]


@needs_pandas
def test_a_long_column_name_widens_the_table_the_way_pandas_widens_it(
    firepanda: ModuleType,
) -> None:
    """Every column is as wide as the widest thing in it, the header included."""
    import pandas as pd

    data: dict[str, list[Any]] = {"a": [1], "bbbbbbbbbbbbbbbbbbbb": [2]}
    assert comparable(report(firepanda.DataFrame(data))) == comparable(report(pd.DataFrame(data)))


def test_show_counts_false_drops_the_count_column_and_nothing_else(
    firepanda: ModuleType,
) -> None:
    """One column of the table goes and the rest of the report is untouched."""
    frame = firepanda.DataFrame(NUMBERS)
    with_counts = report(frame)
    without = report(frame, show_counts=False)
    assert "Non-Null Count" not in "\n".join(without)
    assert len(without) == len(with_counts)
    assert without[-1] == with_counts[-1]


# ---------------------------------------------------------------------------
# The summary form
# ---------------------------------------------------------------------------


def wide_frame(firepanda: ModuleType, count: int) -> Any:
    """A frame of `count` identical integer columns."""
    return firepanda.DataFrame({f"c{i:03d}": [1, 2, 3] for i in range(count)})


@needs_pandas
def test_a_frame_wider_than_the_limit_is_summarized_in_one_line(firepanda: ModuleType) -> None:
    """The first name and the last one, which is as much as a summary can say."""
    import pandas as pd

    mine = report(wide_frame(firepanda, 120))
    assert mine[2] == "Columns: 120 entries, c000 to c119"
    theirs = report(pd.DataFrame({f"c{i:03d}": [1, 2, 3] for i in range(120)}))
    assert comparable(mine) == comparable(theirs)


def test_the_limit_is_pandas_default_and_max_cols_moves_it(firepanda: ModuleType) -> None:
    """A hundred is the long form and a hundred and one is not, until `max_cols` says so."""
    assert firepanda._pandas.MAX_INFO_COLUMNS == 100
    assert report(wide_frame(firepanda, 100))[2].startswith("Data columns")
    assert report(wide_frame(firepanda, 101))[2].startswith("Columns:")
    assert report(wide_frame(firepanda, 101), max_cols=200)[2].startswith("Data columns")
    assert report(wide_frame(firepanda, 5), max_cols=2)[2].startswith("Columns:")


def test_verbose_beats_the_width(firepanda: ModuleType) -> None:
    """Asked for either form, the frame gives that one whatever its width is."""
    assert report(wide_frame(firepanda, 120), verbose=True)[2].startswith("Data columns")
    assert report(firepanda.DataFrame(NUMBERS), verbose=False)[2].startswith("Columns:")


# ---------------------------------------------------------------------------
# Where the report goes and what the call answers
# ---------------------------------------------------------------------------


def test_the_call_answers_nothing(firepanda: ModuleType) -> None:
    """It prints, which is why a caller who wants the text passes a buffer."""
    assert firepanda.DataFrame(NUMBERS).info(buf=io.StringIO()) is None


def test_with_no_buffer_it_goes_to_standard_output(
    firepanda: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """The default, and the only reason anybody types this at a prompt."""
    firepanda.DataFrame(NUMBERS).info()
    written = capsys.readouterr().out
    assert written.startswith("<class 'firepanda.DataFrame'>\n")
    assert written.endswith("bytes\n")


def test_the_buffer_gets_every_line_with_a_newline_after_the_last(
    firepanda: ModuleType,
) -> None:
    """A report that did not end in a newline would run into the next thing printed."""
    buf = io.StringIO()
    firepanda.DataFrame(NUMBERS).info(buf=buf)
    assert buf.getvalue().endswith("\n")


# ---------------------------------------------------------------------------
# The parameters that are about the memory line
# ---------------------------------------------------------------------------


def test_memory_usage_false_drops_the_last_line(firepanda: ModuleType) -> None:
    """And leaves the types line where it was, which is what is above it."""
    frame = firepanda.DataFrame(NUMBERS)
    without = report(frame, memory_usage=False)
    assert without == report(frame)[:-1]
    assert without[-1].startswith("dtypes:")


def test_deep_is_the_same_number(firepanda: ModuleType) -> None:
    """There are no object columns here, so every number is already the deep one."""
    frame = firepanda.DataFrame(MIXED)
    assert report(frame, memory_usage="deep") == report(frame)


@needs_pandas
@pytest.mark.parametrize("flag", [1, "bogus", 2.5])
def test_a_flag_that_is_not_a_boolean_is_not_refused(firepanda: ModuleType, flag: Any) -> None:
    """pandas does not check these either, and a library that refuses more breaks code.

    `memory_usage="bogus"` prints the number over there rather than raising, and the same
    goes for `verbose` and `show_counts`, so the same goes here.
    """
    import pandas as pd

    frame = firepanda.DataFrame(NUMBERS)
    theirs = pd.DataFrame(NUMBERS)
    for name in ("verbose", "memory_usage", "show_counts"):
        assert comparable(report(frame, **{name: flag})) == comparable(
            report(theirs, **{name: flag})
        )


# ---------------------------------------------------------------------------
# The column's own report
# ---------------------------------------------------------------------------


@needs_pandas
def test_a_column_reports_the_same_thing_one_column_narrower(firepanda: ModuleType) -> None:
    """No position and no name in the table, because there is one column to describe."""
    import pandas as pd

    mine = report(firepanda.DataFrame(NUMBERS)["v"])
    theirs = report(pd.DataFrame(NUMBERS)["v"])
    assert comparable(mine) == comparable(theirs)
    assert mine[3] == "Non-Null Count  Dtype"


def test_a_column_says_what_it_is_called(firepanda: ModuleType) -> None:
    """Including when it is called nothing, which prints as the word None."""
    assert report(firepanda.DataFrame(NUMBERS)["v"])[2] == "Series name: v"
    assert report(firepanda.Series([1, 2]))[2] == "Series name: None"


def test_a_column_ignores_the_two_parameters_about_width(firepanda: ModuleType) -> None:
    """There is one column to list and no width at which listing it is too much."""
    column = firepanda.DataFrame(NUMBERS)["v"]
    assert report(column, verbose=False) == report(column)
    assert report(column, max_cols=0) == report(column)


def test_a_column_can_drop_its_count_and_its_memory_too(firepanda: ModuleType) -> None:
    """The two parameters that do something here, doing it."""
    column = firepanda.DataFrame(NUMBERS)["f"]
    assert "non-null" not in "\n".join(report(column, show_counts=False))
    assert report(column, memory_usage=False) == report(column)[:-1]
