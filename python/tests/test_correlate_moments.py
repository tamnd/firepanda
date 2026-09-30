"""`corr` and `cov` over instants, spans and categories.

pandas reads an instant or a span as a count of its unit and a column of
categories by its values. A frame reads NaT as missing, while a column read
against a column hands numpy the raw count, so a NaT with no zone counts as
the smallest int64 there. A frame's `cov` refuses instants and spans outright.
Each test runs the same code on both libraries and compares the answers.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

DAYS = ["2024-01-01", "2024-01-03", "NaT", "2024-01-08", "2024-01-02"]
VALUES = [1.0, 3.0, 2.0, 9.0, 4.0]


def stamps(lib: Any, days: list[str] = DAYS) -> Any:
    return lib.Series(lib.to_datetime(days))


def framed(lib: Any) -> Any:
    return lib.DataFrame({"d": stamps(lib), "v": VALUES})


SCALARS = {
    "instants": lambda lib: stamps(lib, [d for d in DAYS if d != "NaT"]).corr(
        lib.Series([1.0, 3.0, 9.0, 4.0])
    ),
    "instants with NaT": lambda lib: stamps(lib).corr(lib.Series(VALUES)),
    "numbers against instants": lambda lib: lib.Series(VALUES).corr(stamps(lib)),
    "spans with NaT": lambda lib: (stamps(lib) - stamps(lib)[0]).corr(lib.Series(VALUES)),
    "millis with NaT": lambda lib: stamps(lib).dt.as_unit("ms").corr(lib.Series(VALUES)),
    "zoned with NaT": lambda lib: stamps(lib).dt.tz_localize("UTC").corr(lib.Series(VALUES)),
    "cov with NaT": lambda lib: stamps(lib).cov(lib.Series(VALUES)),
    "autocorr": lambda lib: stamps(lib).autocorr(),
}

FRAMES = {
    "frame": lambda lib: framed(lib).corr(),
    "frame spearman": lambda lib: framed(lib).corr(method="spearman"),
    "frame numeric_only": lambda lib: framed(lib).corr(numeric_only=True),
    "frame zoned": lambda lib: lib.DataFrame(
        {"d": stamps(lib).dt.tz_localize("UTC"), "v": VALUES}
    ).corr(),
    "corrwith": lambda lib: framed(lib).corrwith(lib.Series(VALUES)),
    "number categories": lambda lib: lib.DataFrame(
        {"c": lib.Categorical([1, 5, 2, 2, 7]), "v": VALUES}
    ).corr(),
}


@pytest.mark.parametrize("make", SCALARS.values(), ids=SCALARS.keys())
def test_moment_correlation_is_pandas(firepanda: Any, make: Any) -> None:
    assert make(firepanda) == pytest.approx(make(pd), rel=1e-12)


@pytest.mark.parametrize("make", FRAMES.values(), ids=FRAMES.keys())
def test_moment_correlation_frame_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_text_categories_and_frame_cov_are_refused(firepanda: Any) -> None:
    text = firepanda.DataFrame({"c": firepanda.Categorical(["a", "b", "a"]), "v": [1.0, 2, 3]})
    with pytest.raises(ValueError, match="could not convert string to float: 'a'"):
        text.corr()
    with pytest.raises(TypeError, match="not supported for cov"):
        framed(firepanda).cov()
