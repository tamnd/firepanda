"""abs, - and + over columns of objects, and a quantile over numbers beside moments."""

import math

import pytest

import firepanda as fp


def test_abs_of_complex_numbers_is_their_size():
    sizes = fp.Series([3 + 4j, 1j], name="z").abs()
    assert str(sizes.dtype) == "float64"
    assert sizes.tolist() == [5.0, 1.0]
    assert sizes.name == "z"
    gappy = fp.Series([1 + 1j, None]).abs().tolist()
    assert gappy[0] == pytest.approx(math.sqrt(2))
    assert math.isnan(gappy[1])


def test_objects_stay_objects_under_minus_and_abs():
    negated = -fp.Series([-1, 2], dtype="object")
    assert str(negated.dtype) == "object"
    assert negated.tolist() == [1, -2]
    with pytest.raises(TypeError, match="bad operand type for abs"):
        fp.Series([-1, "x"], dtype="object").abs()


def test_a_frame_with_complex_numbers_takes_abs_by_column():
    frame = abs(fp.DataFrame({"z": [3 + 4j, 1j], "n": [-1, 2]}))
    assert [str(dtype) for dtype in frame.dtypes] == ["float64", "int64"]
    assert frame.values.tolist() == [[5.0, 1], [1.0, 2]]


def test_quantile_of_numbers_beside_moments_is_objects():
    frame = fp.DataFrame(
        {"a": [1, 2, 3], "t": fp.to_datetime(["2020-01-01", "2020-01-03", "2020-01-05"])}
    )
    middle = frame.quantile(0.5, numeric_only=False)
    assert str(middle.dtype) == "object"
    assert middle.tolist() == [2.0, fp.Timestamp("2020-01-03")]
    assert middle.name == 0.5
    assert frame.quantile(0.5, numeric_only=True).tolist() == [2.0]
