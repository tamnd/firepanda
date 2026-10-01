"""Index methods that differed from pandas: datetime searches and slices, joins on a level."""

import pytest

import firepanda as pd


def _dates():
    return pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"])


def test_datetime_searchsorted_takes_a_timestamp():
    dates = _dates()
    assert dates.searchsorted(pd.Timestamp("2024-01-02"), side="right") == 3
    assert dates.searchsorted("2024-01-02") == 2
    assert list(dates.searchsorted([pd.Timestamp("2024-01-02"), pd.Timestamp("2025-01-01")])) == [
        2,
        4,
    ]


def test_datetime_searchsorted_with_a_sorter():
    dates = pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-02"])
    assert dates.searchsorted(pd.Timestamp("2024-01-02"), sorter=dates.argsort()) == 1


def test_timedelta_searchsorted():
    spans = pd.TimedeltaIndex(["1D", "2D", "3D"])
    assert spans.searchsorted(pd.Timedelta("2D"), side="right") == 2


def test_datetime_slice_locs_reads_text_as_a_period():
    dates = _dates()
    assert dates.slice_locs(start="2024-01-01", end="2024-01-02") == (0, 3)
    assert dates.slice_locs("2024-01", "2024-01") == (0, 4)
    assert dates.slice_locs(pd.Timestamp("2024-01-02"), None) == (2, 4)


def test_datetime_std_and_mean_gap_is_nat():
    gapped = pd.DatetimeIndex(["2024-01-01", None, "2024-01-03"])
    assert gapped.std(skipna=False) is pd.NaT
    assert pd.DatetimeIndex(["2024-01-01"]).std() is pd.NaT
    assert gapped.mean(skipna=False) is pd.NaT


def test_datetime_to_numpy_objects_are_timestamps():
    pytest.importorskip("numpy")
    values = pd.Series(pd.to_datetime(["2024-01-01", None])).to_numpy(dtype=object)
    assert values[0] == pd.Timestamp("2024-01-01")
    assert type(values[0]).__name__ == "Timestamp"
    assert values[1] is pd.NaT
    spans = pd.Series(pd.to_timedelta(["1D", None])).to_numpy(dtype=object)
    assert type(spans[0]).__name__ == "Timedelta"


def test_multi_value_counts_orders_by_label():
    rows = [("b", 2), ("a", 1), ("b", 2), ("a", 2), ("c", 0), ("a", 1), ("b", 2)]
    multi = pd.MultiIndex.from_tuples(rows, names=["k", "n"])
    assert multi.value_counts(sort=False).index.tolist() == [("a", 1), ("a", 2), ("b", 2), ("c", 0)]
    assert multi.value_counts().tolist() == [3, 2, 1, 1]
    assert multi.value_counts().index.tolist()[2:] == [("a", 2), ("c", 0)]


def _multi():
    rows = [("a", 1), ("b", 2), ("a", 2), ("b", 1)]
    return pd.MultiIndex.from_tuples(rows, names=["k", "n"])


def _listed(answer):
    return [None if part is None else list(part) for part in answer]


def test_multi_join_return_indexers():
    other = pd.MultiIndex.from_tuples([("b", 2), ("c", 3), ("a", 1)], names=["k", "n"])
    joined = _multi().join(other, how="left", return_indexers=True)
    assert _listed(joined)[1:] == [None, [2, 0, -1, -1]]
    joined = _multi().join(other, how="outer", return_indexers=True)
    assert joined[0].tolist() == [("a", 1), ("a", 2), ("b", 1), ("b", 2), ("c", 3)]
    assert _listed(joined)[1:] == [[0, 2, 3, 1, -1], [2, -1, -1, 0, 1]]


def test_multi_join_on_a_level():
    flat = pd.Index([2, 1, 5], name="n")
    joined = _multi().join(flat, how="inner", level="n", return_indexers=True)
    assert _listed(joined)[1:] == [[0, 1, 2, 3], [1, 0, 0, 1]]
    joined = _multi().join(flat, how="left", level="n", return_indexers=True)
    assert joined[1] is None
    letters = pd.Index(["b", "z"], name="k")
    joined = _multi().join(letters, how="inner", level=0, return_indexers=True)
    assert joined[0].tolist() == [("b", 2), ("b", 1)]
    assert _listed(joined)[1:] == [[1, 3], [0, 0]]


def test_multi_join_on_the_shared_name():
    flat = pd.Index([2, 1, 5], name="n")
    joined = flat.join(_multi(), how="left", return_indexers=True)
    assert _listed(joined)[1:] == [[1, 0, 0, 1], [0, 1, 2, 3]]
    with pytest.raises(ValueError, match="no overlapping index names"):
        pd.Index([1, 2]).join(_multi(), how="inner")
    with pytest.raises(TypeError, match="ambiguous"):
        _multi().join(_multi(), level="n")


def test_multi_symmetric_difference_name_must_be_a_list():
    with pytest.raises(TypeError, match="list-like"):
        _multi().symmetric_difference(_multi()[:2], result_name="r")
