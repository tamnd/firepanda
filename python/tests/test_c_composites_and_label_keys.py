"""C library strftime composites, groupby fills on number names, loc by a column of labels."""

import firepanda as fp


def test_strftime_writes_the_c_composites_out():
    index = fp.date_range("2018-03-10 09:00", periods=2, freq="s")
    assert index.strftime("%B %d, %Y, %r").tolist()[0] == "March 10, 2018, 09:00:00 AM"
    assert index.strftime("%c").tolist()[0] == "Sat Mar 10 09:00:00 2018"
    assert index.strftime("%x %X").tolist()[1] == "03/10/18 09:00:01"
    assert fp.Series(index).dt.strftime("%%r %r").tolist()[0] == "%r 09:00:00 AM"


def test_groupby_fill_on_number_column_names():
    nan = float("nan")
    frame = fp.DataFrame({0: [nan, 2.0, nan, 3.0], 1: [3.0, nan, 5.0, nan]})
    filled = frame.groupby(fp.Series([0, 0, 1, 1])).ffill()
    assert filled[1].tolist() == [3.0, 3.0, 5.0, 5.0]
    assert filled.columns.tolist() == [0, 1]


def test_loc_reads_a_column_key_as_labels():
    frame = fp.DataFrame({"c": [40, 44, 50], "label": ["a", "b", "c"]})
    picked = frame.loc[(frame.c - 43).abs().argsort()]
    assert picked["label"].tolist() == ["b", "a", "c"]
    assert frame.c.loc[fp.Series([2, 0])].tolist() == [50, 40]
    assert frame.loc[fp.Series([1]), "c"].tolist() == [44]


def test_to_timedelta_reads_numbers_out_of_an_array():
    import numpy as np

    spans = fp.to_timedelta(np.arange(3), unit="D")
    assert spans.total_seconds().tolist() == [0.0, 86400.0, 172800.0]


def test_reindex_text_fill_beside_numbers_is_an_object_column():
    out = fp.Series([1, 2], index=["a", "b"]).reindex(["a", "c"], fill_value="missing")
    assert str(out.dtype) == "object"
    assert out.tolist() == [1, "missing"]
    floats = fp.Series([1.5, 2.0], index=["a", "b"]).reindex(["c", "a"], fill_value="z")
    assert floats.tolist() == ["z", 1.5]


def test_frame_reindex_text_fill_beside_numbers():
    frame = fp.DataFrame({"s": [200, 404], "t": [0.5, 1.0]}, index=["a", "b"])
    out = frame.reindex(["b", "z"], fill_value="missing")
    assert [str(kind) for kind in out.dtypes.tolist()] == ["object", "object"]
    assert out["s"].tolist() == [404, "missing"]
    assert out["t"].tolist() == [1.0, "missing"]


def _shapes():
    flat = fp.DataFrame(
        {"angles": [0, 3, 4], "degrees": [360, 180, 360]},
        index=["circle", "triangle", "rectangle"],
    )
    rows = [["A", "A", "A", "B", "B", "B"], ["circle", "triangle", "rectangle", "square", "a", "b"]]
    deep = fp.DataFrame(
        {"angles": [0, 3, 4, 4, 5, 6], "degrees": [360, 180, 360, 360, 540, 720]}, index=rows
    )
    return flat, deep


def test_a_flat_frame_spreads_onto_a_deep_one_by_level():
    flat, deep = _shapes()
    out = flat.div(deep, level=1, fill_value=0)
    assert out.index.tolist() == deep.index.tolist()
    assert out["degrees"].tolist()[:4] == [1.0, 1.0, 1.0, 0.0]
    assert flat.le(deep, level=1)["angles"].tolist() == [True, True, True, False, False, False]
