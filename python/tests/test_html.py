"""`DataFrame.to_html` and `_repr_html_`, compared with pandas character for character.

Each case builds the same frame in both libraries and asks both for the markup with
the same arguments. A notebook shows `_repr_html_` as it is, so a difference of one
space is a difference someone sees.
"""

from __future__ import annotations

import datetime as dt
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def built(lib: ModuleType, name: str) -> Any:
    """One of the frames the cases print, made with either library."""
    if name == "mixed":
        return lib.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1.5, None, 2.25],
                "c": ["x<y", "p  q", None],
                "d": [True, False, True],
            }
        )
    if name == "named":
        return lib.DataFrame({"a": [1.0, 2.0]}, index=lib.Index(["r", "s"], name="k"))
    if name == "dates":
        return lib.DataFrame(
            {"t": [dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 3, 4)], "u": [1, 2]}
        )
    if name == "tall":
        return lib.DataFrame({"a": list(range(70)), "b": [x / 3 for x in range(70)]})
    if name == "wide":
        return lib.DataFrame({f"c{i}": [i, i + 1] for i in range(25)})
    if name == "empty":
        return lib.DataFrame({"a": []})
    if name == "nothing":
        return lib.DataFrame()
    if name == "links":
        return lib.DataFrame({"u": ["https://a.b/c?d=1&e=2", "plain"]})
    if name == "float-labels":
        return lib.DataFrame({"a": [1, 2]}, index=lib.Index([0.5, 1.5]))
    if name == "scientific":
        return lib.DataFrame({"a": [1e10, 2.5e-7]})
    if name == "category":
        return lib.DataFrame({"a": lib.Series(["x", "y", "x"], dtype="category"), "b": [1, 2, 3]})
    if name == "spans":
        return lib.DataFrame({"s": [dt.timedelta(days=1), dt.timedelta(hours=3), None]})
    if name == "zoned":
        moments = lib.to_datetime(["2024-01-02 03:00", "2024-02-03 00:00"])
        return lib.DataFrame({"t": moments.tz_localize("UTC")})
    if name == "long-text":
        return lib.DataFrame({"a": ["x" * 80, "short"]})
    if name == "date-labels":
        return lib.DataFrame({"a": [1, 2]}, index=lib.to_datetime(["2024-01-01", "2024-01-02"]))
    if name == "negative":
        return lib.DataFrame({"a": [-1.5, 2.0], "b": [-3, 4]})
    if name == "unicode":
        return lib.DataFrame({"é": ["ü", "\t"]})
    raise AssertionError(name)


FRAMES = [
    "mixed",
    "named",
    "dates",
    "tall",
    "wide",
    "empty",
    "nothing",
    "links",
    "float-labels",
    "scientific",
    "category",
    "spans",
    "zoned",
    "long-text",
    "date-labels",
    "negative",
    "unicode",
]

OPTIONS: list[dict[str, Any]] = [
    {},
    {"index": False},
    {"header": False},
    {"index": False, "header": False},
    {"na_rep": "-"},
    {"float_format": "{:.1f}".format},
    {"float_format": "%.3f"},
    {"float_format": "{:.2e}"},
    {"max_rows": 4},
    {"max_rows": 5},
    {"max_rows": 1},
    {"max_rows": 0},
    {"max_cols": 4},
    {"max_cols": 1},
    {"max_rows": 4, "max_cols": 3},
    {"index": False, "max_rows": 2, "max_cols": 3},
    {"show_dimensions": True},
    {"max_rows": 4, "show_dimensions": "truncate"},
    {"bold_rows": False},
    {"classes": ["x", "y"]},
    {"classes": "p q"},
    {"escape": False},
    {"notebook": True},
    {"border": 0},
    {"border": 3},
    {"border": False},
    {"table_id": "tid"},
    {"render_links": True},
    {"col_space": 40},
    {"col_space": "5em"},
    {"justify": "left"},
    {"index_names": False},
    {"decimal": ","},
    {"header": ["P", "Q"]},
]


def outcome(frame: Any, options: dict[str, Any]) -> str:
    """The markup, or the mistake as its class name and message."""
    try:
        return frame.to_html(**options)
    except Exception as error:
        return f"{type(error).__name__}: {error}"


@pytest.mark.parametrize("options", OPTIONS, ids=repr)
@pytest.mark.parametrize("name", FRAMES)
def test_the_markup_is_pandas_markup(name: str, options: dict[str, Any]) -> None:
    assert outcome(built(fp, name), options) == outcome(built(pd, name), options)


@pytest.mark.parametrize("name", FRAMES)
def test_a_notebook_shows_what_pandas_shows(name: str) -> None:
    assert built(fp, name)._repr_html_() == built(pd, name)._repr_html_()


MORE: list[tuple[str, dict[str, Any]]] = [
    ("mixed", {"columns": ["a", "c"]}),
    ("mixed", {"formatters": {"a": lambda v: f"<{v}>"}}),
    ("mixed", {"formatters": [str, str, str, str]}),
    ("mixed", {"formatters": {"__index__": lambda v: f"i{v}"}}),
    ("mixed", {"col_space": {"a": 10, "": 5}}),
    ("mixed", {"col_space": [1, 2, 3, 4]}),
    ("mixed", {"justify": "bogus"}),
    ("mixed", {"classes": 5}),
    ("mixed", {"col_space": {"zz": 3}}),
    ("mixed", {"col_space": [1]}),
    ("mixed", {"formatters": [str]}),
    ("named", {"escape": False, "notebook": True, "max_rows": 1}),
]


@pytest.mark.parametrize(("name", "options"), MORE, ids=repr)
def test_formatters_widths_and_mistakes(name: str, options: dict[str, Any]) -> None:
    assert outcome(built(fp, name), options) == outcome(built(pd, name), options)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("display.html.use_mathjax", False),
        ("display.html.border", 4),
        ("display.max_rows", 6),
        ("display.max_columns", 3),
        ("display.show_dimensions", True),
        ("display.max_colwidth", 10),
        ("display.float_format", "{:.1f}".format),
    ],
)
def test_the_display_options_reach_the_markup(option: str, value: Any) -> None:
    with pd.option_context(option, value), fp.option_context(option, value):
        for name in ("mixed", "tall", "wide", "long-text"):
            ours, theirs = built(fp, name), built(pd, name)
            assert ours._repr_html_() == theirs._repr_html_()
            assert ours.to_html() == theirs.to_html()


def test_a_notebook_falls_back_to_text_when_asked() -> None:
    with fp.option_context("display.notebook_repr_html", False):
        assert built(fp, "mixed")._repr_html_() is None


def test_the_markup_is_written_to_a_path_or_a_handle(tmp_path: Any) -> None:
    import io

    target = tmp_path / "frame.html"
    assert built(fp, "mixed").to_html(target, encoding="utf-8") is None
    assert target.read_text(encoding="utf-8") == built(pd, "mixed").to_html()
    handle = io.StringIO()
    built(fp, "unicode").to_html(handle)
    assert handle.getvalue() == built(pd, "unicode").to_html()
    with pytest.raises(ValueError, match="buf is not a file name and encoding is specified"):
        built(fp, "mixed").to_html(encoding="utf-8")


def test_the_signature_is_pandas_signature() -> None:
    import inspect

    ours = inspect.signature(fp.DataFrame.to_html)
    theirs = inspect.signature(pd.DataFrame.to_html)
    assert [(p.name, p.kind, p.default) for p in ours.parameters.values()] == [
        (p.name, p.kind, p.default) for p in theirs.parameters.values()
    ]
