"""Whole and decimal column names print flush left to one width, as pandas prints them."""

import firepanda as pd


def test_whole_names_share_a_width():
    text = repr(pd.DataFrame([[1, 2]], columns=[5, 100]))
    assert text == "   5    100\n0    1    2"


def test_decimal_names_share_a_precision():
    text = repr(pd.DataFrame([[1.5, 2]], columns=[1.5, 100.25]))
    assert text.splitlines()[0] == "   1.50    100.25"


def test_text_column_under_whole_names():
    assert repr(pd.DataFrame([["a", "b"]], columns=[1, 100])) == "  1   100\n0   a   b"


def test_cut_columns_widths_from_those_shown():
    text = repr(pd.DataFrame([list(range(30))]))
    assert text.splitlines()[0].startswith("   0   1   2")


def test_formatter_found_by_position_first():
    frame = pd.DataFrame([[1, 2], [3, 4]], columns=[1, 22])
    assert frame.to_string(formatters={1: str}).splitlines()[0] == "  1  22"
