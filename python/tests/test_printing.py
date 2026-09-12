"""What a column prints when you look at it.

A column carrying labels that are not a range used to print positions down the side instead of
the labels, which is issue 719. The labels were there and every other way of reading them gave
the right answer, so it was display and nothing else, which is what made it easy to miss and bad
to leave: a person reading `df.set_index("k")["v"]` at a prompt was being told the wrong thing
about their own data by the one member whose entire job is to tell them.

Most of what is below compares the whole rendering against pandas line for line, because the
output is the whole contract here and an assertion loose enough to accept both spellings would
have accepted the bug too. The two places the two libraries genuinely differ, a missing label and
a frame, are asserted against firepanda on purpose and say why.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

DATA: dict[str, list[Any]] = {"k": ["p", "qq", "r"], "v": [1, 2, 3], "f": [1.5, 2.5, 3.5]}
"""Labels of three different widths, so the padding has something to do."""


def labelled(module: ModuleType) -> Any:
    """The frame from `DATA` with its text column moved into the labels."""
    return module.DataFrame(DATA).set_index("k")


# ---------------------------------------------------------------------------
# The labels themselves
# ---------------------------------------------------------------------------


@needs_pandas
def test_a_column_prints_its_labels_and_not_its_positions(firepanda: ModuleType) -> None:
    """The repro from issue 719, rendered by both libraries and compared whole."""
    import pandas as pd

    made = firepanda.DataFrame({"k": ["p", "q"], "v": [1, 2]}).set_index("k")["v"]
    want = pd.DataFrame({"k": ["p", "q"], "v": [1, 2]}).set_index("k")["v"]
    assert repr(made) == repr(want)


@needs_pandas
def test_labels_of_different_widths_line_up_the_way_pandas_lines_them_up(
    firepanda: ModuleType,
) -> None:
    """The labels are left aligned and the values are right aligned, which is pandas."""
    import pandas as pd

    assert repr(labelled(firepanda)["v"]) == repr(labelled(pd)["v"])


@needs_pandas
def test_numeric_labels_are_left_aligned_too(firepanda: ModuleType) -> None:
    """pandas does not line numeric labels up on their last digit, and neither does this."""
    import pandas as pd

    data: dict[str, list[Any]] = {"k": [10, 200, 3], "v": [1, 2, 3]}
    assert repr(firepanda.DataFrame(data).set_index("k")["v"]) == repr(
        pd.DataFrame(data).set_index("k")["v"]
    )


@needs_pandas
def test_a_range_still_prints_positions(firepanda: ModuleType) -> None:
    """Which is the case that was already right and had to stay right."""
    import pandas as pd

    assert repr(firepanda.Series([1, 2, 3], name="v")) == repr(pd.Series([1, 2, 3], name="v"))


@needs_pandas
def test_a_slice_prints_the_labels_it_kept(firepanda: ModuleType) -> None:
    """A range that no longer starts at zero is still a range and still carries its offset."""
    import pandas as pd

    made = firepanda.Series([0, 1, 2, 3, 4], name="v").iloc[2:]
    want = pd.Series([0, 1, 2, 3, 4], name="v").iloc[2:]
    assert repr(made) == repr(want)


def test_the_printed_labels_are_the_labels_the_index_holds(firepanda: ModuleType) -> None:
    """The property behind every comparison above, checked without reaching for pandas."""
    column = labelled(firepanda)["v"]
    printed = [line.split()[0] for line in repr(column).split("\n")[1:-1]]
    assert printed == column.index.tolist()


# ---------------------------------------------------------------------------
# The name of the level
# ---------------------------------------------------------------------------


@needs_pandas
def test_a_named_index_prints_its_name_on_a_line_of_its_own(firepanda: ModuleType) -> None:
    """Above the listing, and not padded out to the width of the labels under it."""
    import pandas as pd

    assert repr(labelled(firepanda)["v"]).split("\n")[0] == "k"
    assert repr(labelled(pd)["v"]).split("\n")[0] == "k"


@needs_pandas
def test_an_unnamed_index_prints_no_name_line(firepanda: ModuleType) -> None:
    """So the listing starts on the first line, as it does for a range."""
    import pandas as pd

    made = labelled(firepanda)["v"].rename_axis(None)
    want = labelled(pd)["v"].rename_axis(None)
    assert repr(made) == repr(want)
    assert repr(made).split("\n")[0].startswith("p")


@needs_pandas
def test_a_named_range_prints_its_name_as_well(firepanda: ModuleType) -> None:
    """The name is not a property of having real labels, and pandas prints it either way."""
    import pandas as pd

    made = firepanda.Series([1, 2], name="v").rename_axis("ix")
    want = pd.Series([1, 2], name="v").rename_axis("ix")
    assert repr(made) == repr(want)
    assert repr(made).split("\n")[0] == "ix"


@needs_pandas
def test_an_empty_column_says_nothing_about_its_labels(firepanda: ModuleType) -> None:
    """There is no listing for the name to sit above, so pandas leaves it out and so does this."""
    import pandas as pd

    made = firepanda.Series([], name="v").rename_axis("ix")
    assert repr(made) == "Series([], Name: v, dtype: float64)"
    assert repr(pd.Series([], dtype="float64", name="v").rename_axis("ix")).startswith(
        "Series([], Name: v, dtype: float"
    )


# ---------------------------------------------------------------------------
# Where the two libraries part
# ---------------------------------------------------------------------------


def test_a_missing_label_prints_the_way_a_missing_value_prints(firepanda: ModuleType) -> None:
    """`<NA>` rather than pandas' `NaN`, because that is the value this library holds there."""
    column = firepanda.DataFrame({"k": ["p", None], "v": [1, 2]}).set_index("k")["v"]
    assert repr(column).split("\n")[2].startswith("<NA>")


def test_a_frame_still_reports_its_schema_rather_than_its_rows(firepanda: ModuleType) -> None:
    """Printing a frame is a different decision, recorded in document 13, and is untouched."""
    printed = repr(labelled(firepanda))
    assert printed.startswith("DataFrame 3 rows x 2 columns")


# ---------------------------------------------------------------------------
# The parts that were already true
# ---------------------------------------------------------------------------


def test_str_and_repr_are_the_same_rendering(firepanda: ModuleType) -> None:
    """As they are in pandas, where one member does both."""
    column = labelled(firepanda)["v"]
    assert str(column) == repr(column)


def test_a_long_column_elides_its_middle_and_keeps_its_labels(firepanda: ModuleType) -> None:
    """The gap stands in for labels and values together, so the two columns stay in step."""
    names = [f"r{i}" for i in range(12)]
    column = firepanda.DataFrame({"k": names, "v": list(range(12))}).set_index("k")["v"]
    lines = repr(column).split("\n")
    assert lines[1].startswith("r0")
    assert lines[6].startswith("...")
    assert lines[7].startswith("r7")
    assert lines[11].startswith("r11")
