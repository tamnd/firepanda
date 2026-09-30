"""The name of the column axis, which pandas keeps on the columns' index.

A frame holds it beside its columns and hands it out on `df.columns`. It is
set by `rename_axis(columns=...)`, by naming the index `df.columns` answers, by
new columns handed in as a named index, and by the reshapes that make columns
out of a key: `pivot`, `pivot_table`, `crosstab` and `unstack`. pandas carries
it through nearly every method, prints it in the corner of the header, turns
it into the row name on a transpose, and names the labels of a reduction with
it. Each test here runs the same code on both libraries and compares.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def named(lib: Any) -> Any:
    """Two columns under the axis name `cn`, the rows unnamed."""
    frame = lib.DataFrame({"a": [1, 2], "bb": [3.5, 4.0]})
    frame.columns.name = "cn"
    return frame


def keyed(lib: Any) -> Any:
    """A frame with a row key, a column key and values, for the reshapes."""
    return lib.DataFrame({"k": ["x", "y", "x"], "c": ["u", "u", "v"], "v": [1, 2, 3]})


def both(firepanda: Any, make: Any) -> None:
    """Runs `make` on both libraries and compares what they print."""
    assert repr(make(firepanda)) == repr(make(pd))


SHAPES = {
    "printed": lambda lib: named(lib),
    "index": lambda lib: named(lib).columns,
    "rows named": lambda lib: named(lib).rename_axis("rows"),
    "rename_axis": lambda lib: named(lib).rename_axis(columns="longer_name"),
    "axis one": lambda lib: named(lib).rename_axis("z", axis=1),
    "function": lambda lib: named(lib).rename_axis(columns=str.upper).columns.name,
    "mapping": lambda lib: named(lib).rename_axis(columns={"cn": "z"}).columns.name,
    "cleared": lambda lib: named(lib).rename_axis(None, axis=1),
    "sum": lambda lib: named(lib).sum(),
    "row": lambda lib: named(lib).iloc[0],
    "transposed": lambda lib: named(lib).T,
    "transposed named": lambda lib: named(lib).rename_axis("r").T,
    "no index": lambda lib: named(lib).to_string(index=False),
    "no names": lambda lib: named(lib).to_string(index_names=False),
    "html": lambda lib: named(lib).to_html(),
    "head": lambda lib: named(lib).head(1),
    "arithmetic": lambda lib: named(lib) + 1,
    "assigned": lambda lib: named(lib).assign(d=1),
    "concat": lambda lib: lib.concat([named(lib), named(lib)]),
    "corr": lambda lib: named(lib).corr(),
    "melt": lambda lib: named(lib).melt(),
    "stack": lambda lib: named(lib).stack(),
    "tight": lambda lib: named(lib).to_dict("tight")["column_names"],
    "pivot": lambda lib: keyed(lib).pivot(index="k", columns="c", values="v"),
    "pivot_table": lambda lib: keyed(lib).pivot_table(
        index="k", columns="c", values="v", aggfunc="sum"
    ),
    "margins": lambda lib: keyed(lib).pivot_table(
        index="k", columns="c", values="v", aggfunc="sum", margins=True
    ),
    "crosstab": lambda lib: lib.crosstab(keyed(lib).k, keyed(lib).c),
    "unstack": lambda lib: keyed(lib).set_index(["k", "c"])["v"].unstack(),
    "frame unstack": lambda lib: keyed(lib).set_index(["k", "c"]).unstack(),
    "given": lambda lib: lib.DataFrame([[1, 2]], columns=lib.Index(["a", "b"], name="given")),
    "set_axis": lambda lib: named(lib).set_axis(lib.Index(["p", "q"], name="nm"), axis=1),
    "set_axis list": lambda lib: named(lib).set_axis(["p", "q"], axis=1),
    "grouped": lambda lib: named(lib).groupby("a").sum(),
    "rolling": lambda lib: named(lib).rolling(1).sum(),
}


@pytest.mark.parametrize("make", SHAPES.values(), ids=SHAPES.keys())
def test_the_column_axis_name_is_pandas(firepanda: Any, make: Any) -> None:
    both(firepanda, make)


def test_levels_of_columns_print_a_name_each(firepanda: Any) -> None:
    def make(lib: Any) -> Any:
        frame = lib.DataFrame({("a", "x"): [1, 2], ("a", "y"): [3, 4]})
        frame.columns.names = ["top", "low"]
        return frame.rename_axis("r")

    both(firepanda, make)


def test_assigning_plain_columns_drops_the_name(firepanda: Any) -> None:
    frame = named(firepanda)
    frame.columns = ["p", "q"]
    assert frame.columns.name is None


def test_a_name_set_in_place_stays(firepanda: Any) -> None:
    frame = named(firepanda)
    assert frame.rename_axis("z", axis=1, inplace=True) is None
    assert frame.columns.name == "z"
    frame["c"] = 1
    assert frame.columns.name == "z"


def test_a_frame_of_levels_wants_a_name_a_level(firepanda: Any) -> None:
    frame = firepanda.DataFrame({("a", "x"): [1], ("a", "y"): [2]})
    with pytest.raises(ValueError, match="Length of names must match"):
        frame.rename_axis(["only"], axis=1)
    with pytest.raises(ValueError, match="list-like for a MultiIndex"):
        frame.rename_axis("only", axis=1)


def test_duplicated_answers_an_unnamed_column(firepanda: Any) -> None:
    assert firepanda.DataFrame({"a": [1, 1]}).duplicated().name is None
