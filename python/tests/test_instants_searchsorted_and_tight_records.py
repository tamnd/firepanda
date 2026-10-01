"""A batch of small pandas agreements found by the doc sweep.

An index sorted with a NaN keeps the NaN rather than turning it into a null,
`searchsorted` with several values answers a numpy array, a list of numpy
instants reads as instants, dropping a row level keeps the name of the column
axis, a `tight` mapping names its axes and builds levels from tuples, and a numpy
record array names the columns of `from_records`. `isna` on a list or a numpy
array answers numpy flags, and the column axis of a frame is an index.
"""

from __future__ import annotations

import math

import numpy as np

import firepanda as fp


def test_an_index_sorted_with_a_nan_keeps_the_nan():
    index = fp.Index([2.0, math.nan, 1.0])
    made = index.sort_values()
    assert made.tolist()[:2] == [1.0, 2.0]
    assert math.isnan(made.tolist()[2])
    made, order = index.sort_values(return_indexer=True, ascending=False)
    assert math.isnan(made.tolist()[2])
    assert list(order) == [0, 2, 1]


def test_searchsorted_with_several_values_answers_a_numpy_array():
    answer = fp.Series([1, 3, 5]).searchsorted([2, 4])
    assert type(answer).__name__ == "ndarray"
    assert answer.tolist() == [1, 2]


def test_a_list_of_numpy_instants_reads_as_instants():
    values = [np.datetime64("2024-01-01"), None, np.datetime64("2024-01-03")]
    assert str(fp.Series(values).dtype).startswith("datetime64")
    assert str(fp.Index(values).dtype).startswith("datetime64")


def test_dropping_a_row_level_on_the_columns_keeps_their_name():
    frame = fp.DataFrame({"a": [1], "b": [2]})
    frame.columns = fp.MultiIndex.from_tuples([("x", "a"), ("x", "b")], names=["top", "low"])
    assert frame.droplevel(0, axis=1).columns.name == "low"


def test_a_tight_mapping_names_its_axes_and_builds_levels():
    data = {
        "index": [("a", "b"), ("a", "c")],
        "columns": [("x", 1), ("y", 2)],
        "data": [[1, 3], [2, 4]],
        "index_names": ["n1", "n2"],
        "column_names": ["z1", "z2"],
    }
    made = fp.DataFrame.from_dict(data, orient="tight")
    assert list(made.index.names) == ["n1", "n2"]
    assert list(made.columns.names) == ["z1", "z2"]
    plain = {"index": [1, 2], "columns": ["a"], "data": [[5], [6]], "index_names": ["q"]}
    assert fp.DataFrame.from_dict(plain, orient="tight").index.name == "q"


def test_a_record_array_names_the_columns_of_from_records():
    data = np.array([(3, "a"), (2, "b")], dtype=[("col_1", "i4"), ("col_2", "U1")])
    made = fp.DataFrame.from_records(data)
    assert list(made.columns) == ["col_1", "col_2"]
    assert made["col_1"].tolist() == [3, 2]
    assert len(fp.DataFrame.from_records(data, nrows=1)) == 2
    assert list(fp.DataFrame.from_records(data, index="col_2").index) == ["a", "b"]


def test_isna_on_a_list_or_an_array_answers_numpy_flags():
    array = np.array([[1, np.nan, 3], [4, 5, np.nan]])
    assert fp.isna(array).tolist() == [[False, True, False], [False, False, True]]
    assert fp.notna(array).tolist() == [[True, False, True], [True, True, False]]
    assert fp.isna(["a", None, math.nan]).tolist() == [False, True, True]
    assert fp.isna(np.array([1, 2])).tolist() == [False, False]


def test_the_column_axis_of_a_frame_is_an_index():
    axes = fp.DataFrame({"col1": [1, 2], "col2": [3, 4]}).axes
    assert type(axes[1]).__name__ == "Index"
    assert axes[1].tolist() == ["col1", "col2"]
