"""Replacing in a column of objects, axes named by an index key, and numpy grids.

`replace` on a column of python objects matches value by value, and a None put
in by a mapping turns the column into objects that hold the None, as pandas
does. `loc` and `reindex` name their labels after an index key. A frame meets a
two dimensional numpy array cell by cell, and numpy reads a frame as its grid.
"""

from __future__ import annotations

import numpy as np
import pytest

import firepanda as fp


def test_replace_in_a_column_of_objects_matches_each_value():
    column = fp.Series([10, "a", "a", "b", "a"])
    assert column.replace("a", "z").tolist() == [10, "z", "z", "b", "z"]
    assert column.replace({"a": None}).tolist() == [10, None, None, "b", None]


def test_a_none_from_a_mapping_turns_the_column_into_objects():
    made = fp.Series([1, 2]).replace({1: None})
    assert str(made.dtype) == "object"
    assert made.tolist() == [None, 2]
    assert str(fp.Series([1, 2]).replace(9, None).dtype) == "int64"
    assert str(fp.Series(["a", "b"]).replace("a", None).dtype) == "str"


def test_a_numpy_scalar_is_replaced_as_the_value_it_holds():
    column = fp.Series([1, 2])
    assert column.replace(column.iloc[0], 5).tolist() == [5, 2]


def test_an_index_key_names_the_labels_loc_answers():
    frame = fp.DataFrame({"a": [1, 4], "b": [2, 5]}, index=["cobra", "viper"]).rename_axis("k")
    assert frame.loc[fp.Index(["cobra"], name="foo")].index.name == "foo"
    assert frame.loc[fp.Index(["cobra"])].index.name is None
    assert frame.loc[["cobra"]].index.name == "k"
    assert frame["a"].loc[fp.Index(["viper"], name="z")].index.name == "z"
    assert frame.loc[:, fp.Index(["a"], name="q")].columns.name == "q"


def test_an_index_key_names_the_labels_reindex_answers():
    frame = fp.DataFrame({"a": [1, 4]}, index=["cobra", "viper"]).rename_axis("k")
    assert frame.reindex(fp.Index(["cobra"], name="foo")).index.name == "foo"
    assert frame["a"].reindex(fp.Index(["cobra"])).index.name is None
    assert frame.reindex(["cobra"]).index.name == "k"


def test_a_frame_meets_a_numpy_grid_cell_by_cell():
    frame = fp.DataFrame(np.arange(10).reshape(-1, 2), columns=["A", "B"])
    flags = frame % 3 == 0
    same = frame.where(flags, -frame) == np.where(flags, frame, -frame)
    assert same.all().all()
    assert (frame + np.ones((5, 2)))["A"].tolist() == [1.0, 3.0, 5.0, 7.0, 9.0]
    with pytest.raises(ValueError, match="shape must be"):
        frame + np.ones((3, 2))


def test_numpy_reads_a_frame_as_its_grid():
    frame = fp.DataFrame({"A": [0, 2], "B": [1, 3]})
    assert np.asarray(frame).tolist() == [[0, 1], [2, 3]]
