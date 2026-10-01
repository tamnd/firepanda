"""A gap written into a float column is a NaN, as numpy holds it in pandas."""

import pyarrow as pa
import pytest

import firepanda as fp


@pytest.mark.parametrize("gap", [None, fp.NA, float("nan")])
def test_a_written_gap_is_a_nan_not_a_null(gap):
    floats = fp.Series([1.0, 2.0])
    floats.iloc[0] = gap
    whole = fp.Series([1, 2])
    whole.iloc[0] = gap
    frame = fp.DataFrame({"a": [1.5, 2.5]})
    frame.iloc[0, 0] = gap
    labelled = fp.DataFrame({"a": [1.5, 2.5]})
    labelled.loc[0, "a"] = gap
    for column in (floats, whole, frame["a"], labelled["a"]):
        assert str(column.dtype) == "float64"
        assert column.isna().tolist() == [True, False]
        assert pa.array(column, from_pandas=False).null_count == 0
