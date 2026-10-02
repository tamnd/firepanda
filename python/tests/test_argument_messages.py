"""Arguments of the wrong kind are refused in pandas' words."""

import pytest

import firepanda as pd


def _series():
    return pd.Series([2, 1, 3])


@pytest.mark.parametrize(
    "sort",
    [
        lambda s: s.sort_values(ascending="yes"),
        lambda s: s.sort_index(ascending="yes"),
        lambda s: s.to_frame("a").sort_values("a", ascending=["yes"]),
        lambda s: s.sort_values(ascending=None),
    ],
)
def test_ascending_must_be_a_flag(sort):
    with pytest.raises(ValueError, match=r'^For argument "ascending" expected type bool'):
        sort(_series())


def test_ascending_takes_whole_numbers():
    assert _series().sort_values(ascending=0).tolist() == [3, 2, 1]
    assert _series().sort_values(ascending=[1]).tolist() == [1, 2, 3]


def test_diff_periods():
    with pytest.raises(ValueError, match=r"^periods must be an integer$"):
        _series().diff(1.5)
    with pytest.raises(ValueError, match=r"^periods must be an integer$"):
        _series().to_frame().diff(1.5)
    assert _series().diff(2.0).tolist()[2] == 1.0


def test_keep_and_interpolation_wording():
    with pytest.raises(ValueError, match=r'^keep must be either "first", "last" or False$'):
        _series().duplicated(keep="m")
    with pytest.raises(ValueError, match=r"Use one of: dict_keys\(\['inverted_cdf'"):
        _series().quantile(0.5, interpolation="nope")


def test_regex_that_is_not_a_flag():
    with pytest.raises(ValueError, match=r"^'to_replace' must be 'None' if 'regex' is not a bool$"):
        _series().replace(1, 2, regex=5)
