"""Dict operands, lists of arrays as a MultiIndex, compound steps, ufuncs and NA writes."""

import pytest
from firepanda.errors import InvalidArgumentError

import firepanda as fp


def test_a_dict_beside_a_frame_lines_up_with_the_columns_or_rows():
    frame = fp.DataFrame({"angles": [0, 3], "degrees": [360, 180]}, index=["c", "t"])
    out = frame.mul({"angles": 0, "degrees": 2})
    assert out["degrees"].tolist() == [720, 360]
    assert out["angles"].tolist() == [0, 0]
    rows = frame.mul({"c": 0, "t": 2}, axis="index")
    assert rows["degrees"].tolist() == [0, 360]
    swapped = frame.add({"degrees": 1, "angles": 2})
    assert swapped["angles"].tolist() == [2, 5]
    with pytest.raises(InvalidArgumentError, match="length must be 2: given 1"):
        frame.add({"angles": 1})


def test_a_list_of_arrays_is_a_multiindex():
    s = fp.Series([1, 2, 3, 4], index=[["a", "a", "b", "b"], [1, 2, 1, 2]])
    assert s.index.nlevels == 2
    assert s.index.tolist() == [("a", 1), ("a", 2), ("b", 1), ("b", 2)]
    frame = fp.DataFrame({"g": [1, 2]}, index=[["x", "y"], ["p", "q"]])
    assert frame.index.tolist() == [("x", "p"), ("y", "q")]
    wide = fp.DataFrame([[1, 2]], columns=[["a", "a"], ["x", "y"]])
    assert wide.columns.tolist() == [("a", "x"), ("a", "y")]
    named = fp.DataFrame({"A": [1], "B": [2], "C": [3]})
    named.columns = [list("ABC"), list("DEF")]
    assert named.columns.tolist() == [("A", "D"), ("B", "E"), ("C", "F")]


def test_a_step_in_several_units_is_their_sum():
    made = fp.date_range("2018-04-09", periods=3, freq="1D20min")
    assert [str(stamp) for stamp in made] == [
        "2018-04-09 00:00:00",
        "2018-04-10 00:20:00",
        "2018-04-11 00:40:00",
    ]
    assert made.freqstr == "1460min"


def test_a_short_spelling_of_the_instant_type():
    made = fp.DatetimeIndex(["2015-03-29 02:30:00"], dtype="M8[ns]")
    assert str(made.dtype) == "datetime64[ns]"


def test_a_ufunc_answers_a_column():
    import numpy

    s = fp.Series([1.0, 4.0], index=["p", "q"], name="v")
    root = numpy.sqrt(s)
    assert isinstance(root, fp.Series)
    assert root.tolist() == [1.0, 2.0]
    assert root.index.tolist() == ["p", "q"]
    assert root.name == "v"
    assert numpy.add(s, 1).tolist() == [2.0, 5.0]
    masked = numpy.sqrt(fp.Series([4, None], dtype="Int64"))
    assert str(masked.dtype) == "Float64"
    both = fp.Series(range(3)).transform([numpy.sqrt, numpy.exp])
    assert both.columns.tolist() == ["sqrt", "exp"]


def test_writing_na_is_writing_a_gap():
    frame = fp.DataFrame([[1.0, 2.0]])
    frame.iloc[0, 0] = fp.NA
    assert str(frame[0].tolist()[0]) == "nan"
    whole = fp.Series([1, 2])
    whole.iloc[0] = fp.NA
    assert str(whole.dtype) == "float64"
