"""A group reduction takes the cython engine, as pandas reads `engine` only for numba."""

import pytest

import firepanda as fp


def _grouped():
    return fp.DataFrame({"g": ["a", "b", "a"], "x": [1.0, 2.0, 4.0]}).groupby("g")


def test_cython_and_other_names_take_the_one_path():
    assert _grouped().sum(engine="cython", engine_kwargs={"a": 1})["x"].tolist() == [5.0, 2.0]
    assert _grouped().mean(engine=None, engine_kwargs={})["x"].tolist() == [2.5, 2.0]
    assert _grouped()["x"].max(engine="cython").tolist() == [4.0, 2.0]
    assert _grouped().transform("sum", engine="cython")["x"].tolist() == [5.0, 2.0, 5.0]
    assert _grouped().agg("min", engine="cython", engine_kwargs=None)["x"].tolist() == [1.0, 2.0]


def test_numba_is_refused():
    with pytest.raises(NotImplementedError, match="numba"):
        _grouped().std(engine="numba")
    with pytest.raises(NotImplementedError, match="numba"):
        _grouped().aggregate("sum", engine="numba")
