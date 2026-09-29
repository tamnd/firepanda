"""`HDFStore`, `read_hdf` and `to_hdf`, compared with pandas file for file.

Each test writes the same object with both libraries and compares the files
node by node, with every attribute, or reads one file with both and compares
the types, labels and values, or runs one call in both and compares the type
and words of the error. PyTables does the file work, so every test needs it,
and the comparison needs pandas.

firepanda does not keep the frequency of an index once the index labels a
Series or a frame, so no case here labels one with an index that has a
frequency. Its default labels are an `Index` of positions where pandas makes a
`RangeIndex`, and it holds complex numbers as objects, so the comparison reads
a `RangeIndex` as an `Index` and the complex case compares only the files.
"""

from __future__ import annotations

import datetime
import importlib
import importlib.util
import inspect
import shutil
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

needs_both = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None or importlib.util.find_spec("tables") is None,
    reason="pandas and PyTables are not both installed",
)
pytestmark = needs_both


def found(lib: ModuleType, path: str) -> Any:
    module, _, name = f"{lib.__name__}.{path}".rpartition(".")
    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize(
    "path",
    [
        "read_hdf",
        "HDFStore",
        "HDFStore.put",
        "HDFStore.append",
        "HDFStore.select",
        "HDFStore.remove",
        "HDFStore.append_to_multiple",
        "HDFStore.select_as_multiple",
        "HDFStore.copy",
        "DataFrame.to_hdf",
        "Series.to_hdf",
        "io.pytables.to_hdf",
        "io.pytables.read_hdf",
    ],
)
def test_the_signature_is_pandas(path: str) -> None:
    import pandas

    def named(lib: ModuleType) -> Any:
        head, _, member = path.partition(".")
        if head in ("HDFStore", "DataFrame", "Series") and member:
            return getattr(getattr(lib, head), member)
        return found(lib, path)

    ours = inspect.signature(named(fp))
    theirs = inspect.signature(named(pandas))
    assert list(ours.parameters) == list(theirs.parameters)
    assert [p.kind for p in ours.parameters.values()] == [
        p.kind for p in theirs.parameters.values()
    ]
    assert [p.default for p in ours.parameters.values()] == [
        p.default for p in theirs.parameters.values()
    ]


def test_the_names_are_where_pandas_keeps_them() -> None:
    import pandas

    assert fp.io.pytables.HDFStore is fp.HDFStore
    assert fp.io.api.read_hdf is fp.read_hdf
    assert repr(fp.HDFStore) == repr(pandas.HDFStore).replace("pandas", "firepanda")
    for name in fp.io.pytables.__all__:
        if name == "Term":
            # pandas keeps the where expression in `pandas.core.computation`.
            continue
        ours = getattr(fp.io.pytables, name)
        theirs = getattr(pandas.io.pytables, name)
        assert ours.__module__ == theirs.__module__.replace("pandas", "firepanda"), name


def dump(path: Path) -> list[str]:
    """Every node of the file with its type, shape, first values and attributes."""
    import tables

    out = []
    with tables.open_file(str(path)) as handle:
        for node in handle.walk_nodes("/"):
            line = [node._v_pathname, type(node).__name__]
            if isinstance(node, tables.Leaf):
                kind = node.dtype if isinstance(node, tables.Table) else node.atom.type
                line += [str(kind), str(node.shape), repr(node.read()[:6])]
            out.append(" ".join(line))
            for key in sorted(node._v_attrs._v_attrnamesuser):
                out.append(f"    {key} = {node._v_attrs[key]!r}")
    return out


def gap(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, float) and value != value)
        or str(value)
        in (
            "NaT",
            "<NA>",
        )
    )


def shown(obj: Any) -> Any:
    """The types, labels and values of a frame, a Series or an index."""
    if isinstance(obj, list):
        return [shown(part) for part in obj]
    if hasattr(obj, "columns"):
        cols = [shown(obj.iloc[:, i]) for i in range(obj.shape[1])]
        return [str(c) for c in obj.columns], shown(obj.index), cols
    if type(obj).__name__ == "Series":
        dtype = str(obj.dtype).replace("string", "str")
        cells = ["nan" if gap(v) else repr(v) for v in obj.tolist()]
        return dtype, str(obj.name), shown(obj.index), cells
    if type(obj).__name__.endswith("Index"):
        names = [str(n) for n in obj.names]
        kind = type(obj).__name__.replace("RangeIndex", "Index")
        return kind, names, [str(v) for v in obj]
    return repr(obj).replace("firepanda", "pandas")


def outcome(call: Callable[[], Any]) -> Any:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return shown(call())
    except Exception as err:
        return type(err).__name__, str(err)


def base(m: ModuleType) -> Any:
    return m.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.5, None, 3.0],
            "s": ["x", "y", None],
            "t": ["p", "q", "r"],
            "n": [4, 5, 6],
        }
    )


def zoned(m: ModuleType) -> Any:
    return m.DataFrame(
        {
            "d": m.date_range("2020", periods=2, tz="US/Eastern"),
            "e": m.date_range("2020", periods=2, tz="US/Eastern"),
            "u": m.date_range("2020", periods=2, tz="UTC"),
        }
    )


def pairs(m: ModuleType) -> Any:
    return m.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["k", None])


TABLE = {"format": "table"}
WRITES: dict[str, tuple[Callable[[ModuleType], Any], dict[str, Any]]] = {
    "fixed": (base, {}),
    "table": (base, TABLE),
    "data-columns": (base, {**TABLE, "data_columns": ["n", "s"]}),
    "all-data-columns": (base, {**TABLE, "data_columns": True}),
    "min-itemsize": (base, {**TABLE, "min_itemsize": {"s": 20}}),
    "min-itemsize-values": (base, {**TABLE, "min_itemsize": 30}),
    "compressed-fixed": (base, {"complevel": 5, "complib": "zlib"}),
    "compressed-table": (base, {**TABLE, "complevel": 5, "complib": "blosc"}),
    "nan-rep": (base, {**TABLE, "nan_rep": "NA"}),
    "no-index": (base, {**TABLE, "index": False}),
    "appended": (base, {"append": True}),
    "numbers-fixed": (
        lambda m: m.DataFrame(
            {
                "u": m.Series([1, 2], dtype="uint8"),
                "f": m.Series([1, 2], dtype="float32"),
                "b": [True, False],
                "i": [1, 2],
                "j": [3, 4],
            }
        ),
        {},
    ),
    "numbers-table": (
        lambda m: m.DataFrame(
            {"u": m.Series([1, 2], dtype="uint8"), "f": m.Series([1, 2], dtype="float32")}
        ),
        TABLE,
    ),
    "periods-fixed": (
        lambda m: m.DataFrame({"x": [1, 2, 3]}, index=m.period_range("2020", periods=3, freq="M")),
        {},
    ),
    "periods-table": (
        lambda m: m.DataFrame({"x": [1, 2, 3]}, index=m.period_range("2020", periods=3, freq="M")),
        TABLE,
    ),
    "timedeltas": (lambda m: m.DataFrame({"x": m.to_timedelta([1, 2, 3], unit="s")}), TABLE),
    "categories": (
        lambda m: m.DataFrame({"c": m.Categorical(["a", None, "b"]), "x": [1, 2, 3]}),
        {**TABLE, "data_columns": ["c"]},
    ),
    "multi-fixed": (lambda m: m.DataFrame({"x": [1, 2]}, index=pairs(m)), {}),
    "multi-table": (lambda m: m.DataFrame({"x": [1, 2]}, index=pairs(m)), TABLE),
    "multi-series": (lambda m: m.Series([1, 2], index=pairs(m), name="v"), TABLE),
    "series-fixed": (lambda m: m.Series([1.0, 2.0], index=["p", "q"], name="v"), {}),
    "series-table": (lambda m: m.Series([1.0, 2.0], index=["p", "q"]), TABLE),
    "text-series": (lambda m: m.Series(["a", None]), {}),
    "zoned-fixed": (zoned, {}),
    "zoned-table": (lambda m: zoned(m)[["d", "u"]], TABLE),
    "datetimes": (
        lambda m: m.DataFrame({"d": m.date_range("2020", periods=2), "x": [1.0, 2.0]}),
        {**TABLE, "data_columns": ["d"]},
    ),
    "text-index": (lambda m: m.DataFrame({"x": [1, 2]}, index=["a", None]), TABLE),
    "number-labels-fixed": (lambda m: m.DataFrame({0: [1, 2], 1: [3, 4]}), {}),
    "number-labels-table": (lambda m: m.DataFrame({0: [1, 2], 1: [3, 4]}), TABLE),
    "complex": (lambda m: m.DataFrame({"c": [1 + 2j, 3j]}), {}),
}
FILE_ONLY = {"complex"}


@pytest.mark.parametrize("name", list(WRITES))
def test_a_file_is_the_file_pandas_writes(name: str, tmp_path: Path) -> None:
    """Both write the same nodes and attributes, and the file reads the same either way."""
    import pandas

    files = {}
    for m in (pandas, fp):
        make, options = WRITES[name]
        path = tmp_path / f"{m.__name__}.h5"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            make(m).to_hdf(path, key="k", **options)
            if options.get("append"):
                make(m).to_hdf(path, key="k", **options)
        files[m] = path
    assert dump(files[fp]) == dump(files[pandas])
    if name in FILE_ONLY:
        return
    want = outcome(lambda: pandas.read_hdf(files[pandas], "k"))
    assert outcome(lambda: fp.read_hdf(files[pandas], "k")) == want
    assert outcome(lambda: pandas.read_hdf(files[fp], "k")) == want


@pytest.fixture(scope="module")
def stored(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A file pandas wrote with frames and Series in both formats."""
    import numpy as np
    import pandas

    path = tmp_path_factory.mktemp("hdf") / "stored.h5"
    df = pandas.DataFrame(
        {
            "a": np.arange(10),
            "b": np.arange(10) * 1.5,
            "s": list("abcdefghij"),
            "t": list("pqrstuvwxy"),
        },
        index=pandas.DatetimeIndex([f"2020-01-{day:02}" for day in range(1, 11)]),
    )
    df.to_hdf(path, key="df", format="table", data_columns=["a", "s"])
    df.to_hdf(path, key="fixed", format="fixed")
    df[["a", "b"]].to_hdf(path, key="g/one", format="table", data_columns=True)
    df[["s", "t"]].to_hdf(path, key="g/two", format="table")
    pandas.Series(np.arange(5), name="v").to_hdf(path, key="ser", format="table")
    cats = pandas.DataFrame({"c": pandas.Categorical(list("abcab")), "x": range(5)})
    cats.to_hdf(path, key="cat", format="table", data_columns=["c"])
    return path


def reading(m: ModuleType, key: str, options: dict[str, Any]) -> Callable[[Path], Any]:
    return lambda path: m.read_hdf(path, key, **options)


READS: dict[str, tuple[str, dict[str, Any]]] = {
    "number": ("df", {"where": "a > 5"}),
    "or": ("df", {"where": "s == 'c' | s == 'e'"}),
    "in": ("df", {"where": "s in ['c', 'e', 'z']"}),
    "index": ("df", {"where": "index >= '2020-01-05'"}),
    "timestamp": ("df", {"where": "index < Timestamp('2020-01-03')"}),
    "columns-term": ("df", {"where": "columns = ['a', 't']"}),
    "terms": ("df", {"where": ["a > 2", "a < 6"]}),
    "not-a-data-column": ("df", {"where": "b > 2"}),
    "columns": ("df", {"columns": ["b", "s"]}),
    "range": ("df", {"start": 2, "stop": 5}),
    "fixed": ("fixed", {}),
    "fixed-where": ("fixed", {"where": "a > 5"}),
    "fixed-columns": ("fixed", {"columns": ["a"]}),
    "category": ("cat", {"where": "c == 'b'"}),
    "series": ("ser", {"where": "index > 2"}),
    "missing-key": ("nope", {}),
    "bad-mode": ("df", {"mode": "w"}),
}


@pytest.mark.parametrize("name", list(READS))
def test_a_read_is_pandas_read(name: str, stored: Path) -> None:
    import pandas

    key, options = READS[name]
    assert outcome(lambda: fp.read_hdf(stored, key, **options)) == outcome(
        lambda: pandas.read_hdf(stored, key, **options)
    )


def test_a_local_name_in_a_where_is_pandas(stored: Path) -> None:
    import pandas

    def read(m: ModuleType) -> Any:
        lim = 7  # noqa: F841
        return outcome(lambda: m.read_hdf(stored, "df", where="a > lim"))

    assert read(fp) == read(pandas)


def test_chunks_and_a_file_of_one_object_read_as_pandas(stored: Path, tmp_path: Path) -> None:
    import pandas

    single = tmp_path / "single.h5"
    pandas.DataFrame({"a": [1, 2]}).to_hdf(single, key="only", format="table")
    calls: list[Callable[[ModuleType], Any]] = [
        lambda m: list(m.read_hdf(stored, "df", chunksize=4)),
        lambda m: list(m.read_hdf(stored, "df", chunksize=3, where="a > 3")),
        lambda m: m.read_hdf(stored),
        lambda m: m.read_hdf(single),
        lambda m: m.read_hdf(tmp_path / "missing.h5", "df"),
    ]
    for call in calls:
        assert outcome(lambda c=call: c(fp)) == outcome(lambda c=call: c(pandas))


def within(m: ModuleType, path: Path, call: Callable[[Any], Any]) -> Any:
    with m.HDFStore(path, mode="r") as store:
        return call(store)


STORE: dict[str, Callable[[Any], Any]] = {
    "keys": lambda st: st.keys(),
    "native-keys": lambda st: st.keys(include="native"),
    "len": len,
    "contains": lambda st: ["df" in st, "/g/one" in st, "g" in st, "zz" in st],
    "storers": lambda st: [repr(st.get_storer(k)) for k in st],
    "column": lambda st: st.select_column("df", "s"),
    "index-column": lambda st: st.select_column("df", "index", start=1, stop=3),
    "not-a-column": lambda st: st.select_column("df", "b"),
    "coordinates": lambda st: st.select_as_coordinates("df", "a > 6"),
    "by-coordinates": lambda st: st.select("df", where=st.select_as_coordinates("df", "a > 6")),
    "multiple": lambda st: st.select_as_multiple(
        ["g/one", "g/two"], where="a > 6", selector="g/one"
    ),
    "walk": lambda st: list(st.walk()),
    "items": lambda st: [k for k, _ in st.items()],
    "attribute": lambda st: st.ser,
    "item": lambda st: st["g/two"],
}


@pytest.mark.parametrize("name", list(STORE))
def test_the_store_reads_as_pandas(name: str, stored: Path) -> None:
    import pandas

    call = STORE[name]
    assert outcome(lambda: within(fp, stored, call)) == outcome(
        lambda: within(pandas, stored, call)
    )


def test_the_store_prints_as_pandas(stored: Path) -> None:
    import pandas

    def printed(m: ModuleType) -> Any:
        store = m.HDFStore(stored, mode="r")
        text = [repr(store), store.info()]
        store.close()
        failed = outcome(lambda: store["df"])
        return [*text, store.is_open, failed, repr(store), store.info()]

    def named(part: Any) -> Any:
        return (
            part.replace("pandas.HDFStore", "firepanda.HDFStore") if isinstance(part, str) else part
        )

    assert printed(fp) == [named(part) for part in printed(pandas)]


CHANGES: dict[str, Callable[[Any], Any]] = {
    "remove-where": lambda st: [st.remove("df", where="a > 6"), st["df"]],
    "remove-key": lambda st: [st.remove("g/one"), st.keys()],
    "delete-item": lambda st: [st.__delitem__("ser"), st.keys()],
    "remove-range": lambda st: [st.remove("df", start=2, stop=4), st["df"]],
    "append": lambda st: [st.append("df", st["df"].iloc[:2]), st.select("df", where="a < 2")],
    "append-to-fixed": lambda st: st.append("fixed", st["fixed"]),
    "append-other-columns": lambda st: st.append("df", st["df"][["a", "b"]]),
    "put": lambda st: [
        st.put("new", st["fixed"], format="table", data_columns=["a"]),
        st.select("new", "a < 3"),
    ],
    "set-item": lambda st: [st.__setitem__("new", st["ser"]), st["new"]],
    "append-to-multiple": lambda st: [
        st.append_to_multiple({"x": ["a", "b"], "y": None}, st["df"], selector="x"),
        st["x"],
        st["y"],
    ],
    "table-index": lambda st: [
        st.create_table_index("df", columns=["a"], optlevel=9, kind="full"),
        st.get_storer("df").table.cols.a.index.optlevel,
    ],
    "bad-format": lambda st: st.put("q", st["ser"], format="zz"),
}


@pytest.mark.parametrize("name", list(CHANGES))
def test_a_change_is_pandas_change(name: str, stored: Path, tmp_path: Path) -> None:
    import pandas

    def changed(m: ModuleType) -> Any:
        path = tmp_path / f"{m.__name__}.h5"
        shutil.copy(stored, path)
        with m.HDFStore(path) as store:
            return outcome(lambda: CHANGES[name](store))

    assert changed(fp) == changed(pandas)


def test_a_copy_is_pandas_copy(stored: Path, tmp_path: Path) -> None:
    import pandas

    def copied(m: ModuleType) -> Any:
        with m.HDFStore(stored, mode="r") as store:
            new = store.copy(str(tmp_path / f"{m.__name__}.h5"))
            try:
                return [new.keys(), shown(new["df"]), repr(new.get_storer("df"))]
            finally:
                new.close()

    assert copied(fp) == copied(pandas)
    assert dump(tmp_path / "firepanda.h5") == dump(tmp_path / "pandas.h5")


def attempt(m: ModuleType, path: Path, call: Callable[[ModuleType, Path], Any]) -> Any:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            call(m, path)
            got: Any = "ok"
        except Exception as err:
            got = (type(err).__name__, str(err).replace(str(path), "P"))
    return got, [(w.category.__name__, str(w.message)) for w in caught]


def one(m: ModuleType, path: Path, call: Callable[[Any], Any]) -> Any:
    with m.HDFStore(path) as store:
        return call(store)


def date_column(m: ModuleType) -> Any:
    return m.DataFrame({"d": m.Series([datetime.date(2020, 1, 1)], dtype=object)})


ERRORS: dict[str, Callable[[ModuleType, Path], Any]] = {
    "category-fixed": lambda m, p: m.DataFrame({"c": m.Categorical(["a", "b"])}).to_hdf(p, key="k"),
    "nullable-table": lambda m, p: m.DataFrame({"c": m.array([1, None], dtype="Int64")}).to_hdf(
        p, key="k", format="table"
    ),
    "nullable-fixed": lambda m, p: m.DataFrame({"c": m.array([1, None], dtype="Int64")}).to_hdf(
        p, key="k"
    ),
    "nullable-compressed": lambda m, p: m.DataFrame(
        {"c": m.array([1, None], dtype="Int64")}
    ).to_hdf(p, key="k", complevel=3),
    "boolean": lambda m, p: m.DataFrame({"c": m.array([True, None], dtype="boolean")}).to_hdf(
        p, key="k", format="table"
    ),
    "periods-table": lambda m, p: m.DataFrame(
        {"c": m.period_range("2020", periods=2, freq="M")}
    ).to_hdf(p, key="k", format="table"),
    "periods-fixed": lambda m, p: m.DataFrame(
        {"c": m.period_range("2020", periods=2, freq="M")}
    ).to_hdf(p, key="k"),
    "mixed-fixed": lambda m, p: m.DataFrame({"c": m.Series([1, "a"], dtype=object)}).to_hdf(
        p, key="k"
    ),
    "mixed-table": lambda m, p: m.DataFrame({"c": m.Series([1, "a"], dtype=object)}).to_hdf(
        p, key="k", format="table"
    ),
    "compressed-put": lambda m, p: one(m, p, lambda st: st.put("k", base(m), complib="zlib")),
    "bad-complib": lambda m, p: m.HDFStore(p, complib="nope"),
    "format-keyword": lambda m, p: m.HDFStore(p, format="table"),
    "not-a-frame": lambda m, p: one(m, p, lambda st: st.put("k", [1, 2])),
    "columns-keyword": lambda m, p: one(m, p, lambda st: st.append("k", base(m), columns=["a"])),
    "longer-text": lambda m, p: [
        base(m).to_hdf(p, key="k", format="table"),
        m.DataFrame({"a": [3], "b": [1.0], "s": ["longer"], "t": ["x"], "n": [1]}).to_hdf(
            p, key="k", append=True
        ),
    ],
    "bad-min-itemsize": lambda m, p: base(m).to_hdf(
        p, key="k", format="table", min_itemsize={"zz": 5}
    ),
    "zone-change": lambda m, p: [
        m.DataFrame({"x": [1]}, index=m.date_range("2020", periods=1, tz="UTC")).to_hdf(
            p, key="k", format="table"
        ),
        m.DataFrame({"x": [1]}, index=m.date_range("2021", periods=1, tz="Asia/Tokyo")).to_hdf(
            p, key="k", append=True
        ),
    ],
    "type-change": lambda m, p: [
        base(m).to_hdf(p, key="k", format="table"),
        base(m).astype({"a": "float64"}).to_hdf(p, key="k", append=True),
    ],
    "number-data-column": lambda m, p: m.DataFrame({0: [1], 1: [2]}).to_hdf(
        p, key="k", format="table", data_columns=[0]
    ),
    "date-column": lambda m, p: date_column(m).to_hdf(p, key="k", format="table"),
    "complex-data-column": lambda m, p: m.DataFrame({"c": [1 + 2j, 3j]}).to_hdf(
        p, key="k", format="table", data_columns=["c"]
    ),
    "chunks-of-fixed": lambda m, p: [
        base(m).to_hdf(p, key="k"),
        m.read_hdf(p, "k", chunksize=1),
    ],
    "multi-columns-data-columns": lambda m, p: m.DataFrame(
        [[1, 2]], columns=m.MultiIndex.from_tuples([("a", "b"), ("a", "c")])
    ).to_hdf(p, key="k", format="table", data_columns=True),
    "multi-name-repeats": lambda m, p: m.DataFrame(
        {"k": [1]}, index=m.MultiIndex.from_tuples([("a", 1)], names=["k", "n"])
    ).to_hdf(p, key="k", format="table"),
    "reopen-to-write": lambda m, p: one(m, p, lambda st: [st.put("k", base(m)), st.open("w")]),
}


@pytest.mark.parametrize("name", list(ERRORS))
def test_an_error_is_pandas_error(name: str, tmp_path: Path) -> None:
    import pandas

    got = {}
    for m in (pandas, fp):
        got[m] = attempt(m, tmp_path / f"{m.__name__}.h5", ERRORS[name])
    assert got[fp] == got[pandas]
