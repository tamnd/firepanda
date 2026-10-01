"""pivot with several value columns, numbers beside text, read as pandas reads them."""

import math

import firepanda as fp


def made():
    return fp.DataFrame(
        {
            "foo": ["one", "one", "two", "two"],
            "bar": ["A", "B", "A", "B"],
            "baz": [1, 2, 3, 4],
            "zoo": ["x", "y", "z", "q"],
        }
    )


def test_numbers_beside_text_become_objects_and_text_stays_text():
    out = made().pivot(index="foo", columns="bar", values=["baz", "zoo"])
    assert [str(d) for d in out.dtypes.tolist()] == ["object", "object", "str", "str"]
    assert out[("baz", "A")].tolist() == [1, 3]
    assert out[("zoo", "B")].tolist() == ["y", "q"]


def test_a_gap_in_an_object_column_is_nan():
    frame = made().iloc[:3]
    out = frame.pivot(index="foo", columns="bar", values=["baz", "zoo"])
    assert math.isnan(out[("baz", "B")].tolist()[1])
    assert out[("baz", "A")].tolist() == [1, 3]


def test_ints_beside_floats_become_floats():
    frame = made().assign(zoo=[0.5, 1.5, 2.5, 3.5])
    out = frame.pivot(index="foo", columns="bar", values=["baz", "zoo"])
    assert {str(d) for d in out.dtypes.tolist()} == {"float64"}
    assert out[("baz", "B")].tolist() == [2.0, 4.0]
