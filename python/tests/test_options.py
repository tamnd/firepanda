"""pandas' options, checked against pandas.

Every option pandas registers is registered with the same default, a name is a
pattern searched for in every option's name, and a new value is checked the way
pandas checks it. Each test puts back what it changed, since the options are
shared by the whole process on both sides.
"""

from __future__ import annotations

import importlib.util
import inspect
import warnings
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

NAMES = ["get_option", "set_option", "reset_option", "describe_option", "option_context"]


@pytest.fixture(autouse=True)
def untouched(firepanda: ModuleType) -> Iterator[None]:
    """Every option on both sides back at its default after each test."""
    import pandas as pd

    yield
    firepanda.reset_option("all")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key in pd._config.config._registered_options:
            if key != "plotting.matplotlib.register_converters":
                pd.set_option(key, pd._config.config._registered_options[key].defval)


def same_error(mine: BaseException, theirs: BaseException) -> None:
    """The same message, and the same class or a class of the same name with the same bases."""
    assert str(mine) == str(theirs)
    if isinstance(mine, type(theirs)):
        return
    assert type(mine).__name__ == type(theirs).__name__
    assert [kind for kind in type(theirs).__mro__ if kind.__module__ == "builtins"] == [
        kind for kind in type(mine).__mro__ if kind.__module__ == "builtins"
    ]


def registered() -> dict[str, Any]:
    """pandas' options and their defaults."""
    import pandas as pd

    options = pd._config.config._registered_options
    return {key: options[key].defval for key in options}


def test_every_option_is_registered_with_pandas_default(firepanda: ModuleType) -> None:
    """The same 73 names and the same defaults."""
    wanted = registered()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = {key: firepanda.get_option(key) for key in wanted}
    assert got == wanted
    assert firepanda.describe_option("all", _print_desc=False).count("[default:") == len(wanted)


PATTERNS = [
    "display.max_rows",
    "max_colwidth",
    "DISPLAY.PRECISION",
    "max_rows",
    "display",
    "nope",
    "",
    "precision$",
    "styler.format.precision",
]


@pytest.mark.parametrize("pat", PATTERNS)
def test_a_pattern_finds_what_pandas_finds(firepanda: ModuleType, pat: str) -> None:
    """One match answers its value, and none or several raise pandas' error."""
    import pandas as pd

    try:
        want = pd.get_option(pat)
    except Exception as error:
        with pytest.raises(Exception) as mine:
            firepanda.get_option(pat)
        same_error(mine.value, error)
    else:
        assert firepanda.get_option(pat) == want


def test_a_value_set_is_read_back_every_way(firepanda: ModuleType) -> None:
    """By name, by pattern and by attribute, in pairs and one at a time."""
    firepanda.set_option("display.max_rows", 5, "display.precision", 3)
    assert firepanda.get_option("display.max_rows") == 5
    assert firepanda.options.display.max_rows == 5
    assert firepanda.get_option("display.precision") == 3
    firepanda.options.display.max_rows = None
    assert firepanda.get_option("display.max_rows") is None
    firepanda.reset_option("display")
    assert firepanda.options.display.max_rows == 60
    assert firepanda.options.display.precision == 6


def test_reset_takes_every_match(firepanda: ModuleType) -> None:
    """A pattern naming several options resets each of them."""
    firepanda.set_option("display.max_rows", 1, "styler.render.max_rows", 2)
    firepanda.reset_option("max_rows")
    assert firepanda.get_option("display.max_rows") == 60
    assert firepanda.get_option("styler.render.max_rows") is None


def test_a_context_puts_the_values_back(firepanda: ModuleType) -> None:
    """Inside the block the new values, after it the old ones, even after an error."""
    with firepanda.option_context("display.max_rows", 3, "display.width", None) as inside:
        assert inside is None
        assert firepanda.options.display.max_rows == 3
        assert firepanda.options.display.width is None
    assert firepanda.options.display.max_rows == 60
    with (
        pytest.raises(ZeroDivisionError),
        firepanda.option_context({"display.max_rows": 4}),
    ):
        assert firepanda.options.display.max_rows == 4
        1 / 0  # noqa: B018
    assert firepanda.options.display.max_rows == 60


def test_a_failed_pair_keeps_the_pairs_before_it(firepanda: ModuleType) -> None:
    """Pairs are set one after another, as pandas sets them."""
    with pytest.raises(ValueError):
        firepanda.set_option("display.max_rows", 7, "display.width", "x")
    assert firepanda.get_option("display.max_rows") == 7


def test_the_options_list_their_parts(firepanda: ModuleType) -> None:
    """`dir` of `options` and of a section, as tab completion shows them."""
    import pandas as pd

    assert sorted(dir(firepanda.options)) == sorted(dir(pd.options))
    assert sorted(dir(firepanda.options.display)) == sorted(dir(pd.options.display))
    assert sorted(dir(firepanda.options.io.excel)) == sorted(dir(pd.options.io.excel))


def test_a_deprecated_option_warns_as_pandas_warns(firepanda: ModuleType) -> None:
    """The same message each time the option is named."""
    import pandas as pd

    for key in ["mode.copy_on_write", "future.no_silent_downcasting"]:
        with pytest.warns(pd.errors.Pandas4Warning) as theirs:
            pd.get_option(key)
        with pytest.warns(firepanda.errors.Pandas4Warning) as mine:
            firepanda.get_option(key)
        assert [str(w.message) for w in mine] == [str(w.message) for w in theirs]


def test_a_description_ends_with_the_default_and_the_value(firepanda: ModuleType) -> None:
    """The last line of each option's description is pandas' last line."""
    import pandas as pd

    firepanda.set_option("display.max_rows", 9)
    pd.set_option("display.max_rows", 9)
    for pat in ["display.max_rows", "mode.copy_on_write", "styler.render"]:
        mine = firepanda.describe_option(pat, _print_desc=False)
        theirs = pd.describe_option(pat, _print_desc=False)
        pick = [line for line in mine.split("\n") if line.startswith("    [") or " : " in line]
        want = [line for line in theirs.split("\n") if line.startswith("    [")]
        assert [line for line in pick if line.startswith("    [")] == want
        assert [line.split(" ")[0] for line in pick if " : " in line] == [
            line.split(" ")[0] for line in theirs.split("\n") if line[:1].isalpha()
        ]


def test_describe_prints_when_asked(firepanda: ModuleType, capsys: Any) -> None:
    """With `_print_desc` left on it prints and answers None."""
    assert firepanda.describe_option("display.max_rows") is None
    assert "[default: 60] [currently: 60]" in capsys.readouterr().out


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.set_option("display.max_rows"),
    lambda m: m.set_option(),
    lambda m: m.set_option("display.max_rows", 1, "x"),
    lambda m: m.set_option("display.max_rows", "x"),
    lambda m: m.set_option("display.max_rows", -1),
    lambda m: m.set_option("display.width", "x"),
    lambda m: m.set_option("display.large_repr", "x"),
    lambda m: m.set_option("compute.use_numba", 1),
    lambda m: m.set_option("display.max_info_rows", True),
    lambda m: m.set_option("display.float_format", "x"),
    lambda m: m.set_option("display.memory_usage", "x"),
    lambda m: m.set_option("mode.string_storage", "x"),
    lambda m: m.set_option("styler.latex.multicol_align", "x"),
    lambda m: m.set_option("styler.format.formatter", 1),
    lambda m: m.set_option("display.colheader_justify", 1),
    lambda m: m.set_option("plotting.backend", "no_such_module_here"),
    lambda m: m.set_option("nope", 1),
    lambda m: m.set_option("display.max_rows", 1, foo=2),
    lambda m: m.get_option(1),
    lambda m: m.reset_option("nope"),
    lambda m: m.reset_option(""),
    lambda m: m.reset_option("max"),
    lambda m: m.describe_option("nope"),
    lambda m: m.options.nope,
    lambda m: m.options.display.nope,
    lambda m: setattr(m.options, "nope", 1),
    lambda m: setattr(m.options, "display", 1),
    lambda m: m.option_context("display.max_rows").__enter__(),
    lambda m: m.option_context().__enter__(),
    lambda m: m.option_context("nope", 1).__enter__(),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The same class or a subclass of it, and the same message."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    same_error(mine.value, theirs.value)


def test_info_reads_max_info_columns(firepanda: ModuleType, capsys: Any) -> None:
    """A frame wider than the option is summed up in one line, as pandas sums it up."""
    frame = firepanda.DataFrame({"a": [1], "b": [2], "c": [3]})
    with firepanda.option_context("display.max_info_columns", 2):
        frame.info()
    assert "Columns: 3 entries, a to c" in capsys.readouterr().out
    frame.info()
    assert "Data columns (total 3 columns):" in capsys.readouterr().out


@pytest.mark.parametrize("name", NAMES)
def test_the_signature_is_pandas_signature(firepanda: ModuleType, name: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(getattr(firepanda, name)).parameters
    yours = inspect.signature(getattr(pd, name)).parameters
    assert [(p.name, p.kind, p.default) for p in ours.values()] == [
        (p.name, p.kind, p.default) for p in yours.values()
    ]
