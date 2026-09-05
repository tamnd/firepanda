"""The Arrow PyCapsule protocol, checked against the libraries it exists for.

Document 07 section 4 says data crosses as Arrow and never as objects, and that
implementing `__arrow_c_schema__` and `__arrow_c_array__` once buys zero copy
interchange with every library that speaks the protocol without a line of library
specific code. That is a claim about other people's code, so it is tested against
other people's code rather than against a reading of the specification.

The zero copy claim needs a word about how it is tested from here, because the
obvious test cannot be written. Nothing in Python can see a firepanda buffer's
address directly, so there is no address on our side to compare pyarrow's against.
What can be seen is that exporting the same frame twice hands out the same
address, which a producer that copied could not do, since the first copy is still
alive and holding its allocation while the second is made. The Mojo suite asserts
the pointer identity directly, in `tests/test_arrow_export.mojo`, and this is the
half of it that can be asserted from where the consumer stands.
"""

from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

needs = {
    name: pytest.mark.skipif(
        importlib.util.find_spec(name) is None, reason=f"{name} is not installed"
    )
    for name in ("pyarrow", "polars", "pandas")
}

MIXED = (
    "name,qty,price,ok\n"
    "rivet,4,1.25,true\n"
    "bolt,10,0.40,false\n"
    "a much longer part name than twelve bytes,25,0.05,true\n"
)


def frame_of(firepanda: ModuleType, tmp_path: Path, text: str = MIXED) -> object:
    """Reads a CSV into a frame, since that is the only way to make one today."""
    csv = tmp_path / "parts.csv"
    csv.write_text(text)
    return firepanda.read_csv(str(csv))


def test_the_frame_answers_both_halves_of_the_protocol(
    firepanda: ModuleType, tmp_path: Path
) -> None:
    """A consumer decides what a frame is by looking for these two names."""
    frame = frame_of(firepanda, tmp_path)
    assert hasattr(frame, "__arrow_c_schema__")
    assert hasattr(frame, "__arrow_c_array__")


def test_the_capsules_are_named_what_the_protocol_reserves(
    firepanda: ModuleType, tmp_path: Path
) -> None:
    """The names are the whole of the handshake.

    A capsule carries a name and a consumer checks it before touching the pointer,
    so a capsule called anything else is not an Arrow capsule to anybody, however
    correct the struct inside it is.
    """
    import ctypes

    is_valid = ctypes.pythonapi.PyCapsule_IsValid
    is_valid.argtypes = [ctypes.py_object, ctypes.c_char_p]
    is_valid.restype = ctypes.c_int

    frame = frame_of(firepanda, tmp_path)
    schema, array = frame.__arrow_c_array__()
    assert is_valid(schema, b"arrow_schema") == 1
    assert is_valid(array, b"arrow_array") == 1
    assert is_valid(frame.__arrow_c_schema__(), b"arrow_schema") == 1


def test_the_array_half_hands_back_a_pair(firepanda: ModuleType, tmp_path: Path) -> None:
    """`__arrow_c_array__` returns a tuple, which is what consumers unpack."""
    got = frame_of(firepanda, tmp_path).__arrow_c_array__()
    assert isinstance(got, tuple)
    assert len(got) == 2


@needs["pyarrow"]
def test_pyarrow_reads_the_schema(firepanda: ModuleType, tmp_path: Path) -> None:
    """The column names and types arrive, which is what a frame is a struct for."""
    import pyarrow as pa

    schema = pa.schema(frame_of(firepanda, tmp_path))
    assert schema.names == ["name", "qty", "price", "ok"]
    assert schema.types == [pa.string_view(), pa.int64(), pa.float64(), pa.bool_()]


@needs["pyarrow"]
def test_pyarrow_reads_the_values(firepanda: ModuleType, tmp_path: Path) -> None:
    """Every value survives the crossing, for every type a CSV can produce."""
    import pyarrow as pa

    batch = pa.record_batch(frame_of(firepanda, tmp_path))
    assert batch.num_rows == 3
    assert batch.column("qty").to_pylist() == [4, 10, 25]
    assert batch.column("price").to_pylist() == [1.25, 0.40, 0.05]
    assert batch.column("ok").to_pylist() == [True, False, True]
    assert batch.column("name").to_pylist()[0] == "rivet"
    assert batch.column("name").to_pylist()[2].startswith("a much longer")


@needs["pyarrow"]
def test_a_string_longer_than_twelve_bytes_survives(firepanda: ModuleType, tmp_path: Path) -> None:
    """The one string case that reads the payload buffer rather than the view.

    A view holds up to twelve bytes inline and anything longer as a pointer into a
    payload block. A test with only short strings passes without the payload
    buffer being correct, or being present at all.
    """
    import pyarrow as pa

    batch = pa.record_batch(frame_of(firepanda, tmp_path))
    assert batch.column("name").to_pylist()[2] == ("a much longer part name than twelve bytes")


@needs["pyarrow"]
def test_nulls_arrive_as_nulls(firepanda: ModuleType, tmp_path: Path) -> None:
    """An empty CSV field is a null, and Arrow's validity says so."""
    import pyarrow as pa

    frame = frame_of(firepanda, tmp_path, "name,qty,price\nrivet,,1.25\n,10,\nnut,25,0.05\n")
    batch = pa.record_batch(frame)
    assert batch.column("qty").to_pylist() == [None, 10, 25]
    assert batch.column("price").to_pylist() == [1.25, None, 0.05]
    assert batch.column("name").to_pylist() == ["rivet", None, "nut"]


@needs["pyarrow"]
def test_the_same_frame_exported_twice_hands_out_the_same_memory(
    firepanda: ModuleType, tmp_path: Path
) -> None:
    """The zero copy assertion, in the only form Python can make it.

    Both batches are alive at once, so a producer that copied would have had to
    allocate the second copy somewhere the first was not, and the addresses could
    not match. See the note at the top of this file.
    """
    import pyarrow as pa

    frame = frame_of(firepanda, tmp_path)
    first = pa.record_batch(frame)
    second = pa.record_batch(frame)
    for name in ("qty", "price"):
        assert first.column(name).buffers()[1].address == second.column(name).buffers()[1].address


@needs["pyarrow"]
def test_the_frame_can_be_dropped_while_a_consumer_still_holds_the_data(
    firepanda: ModuleType, tmp_path: Path
) -> None:
    """The ownership question, which is what zero copy costs.

    The consumer has pointers into memory firepanda allocated and no idea that it
    did. Dropping the last Python reference to the frame has to leave the data
    alive, because the exported array holds a share of the frame and not a borrow
    of it, and the reading below is reading freed memory if that is wrong.
    """
    import pyarrow as pa

    frame = frame_of(firepanda, tmp_path)
    batch = pa.record_batch(frame)
    del frame
    gc.collect()
    assert batch.column("qty").to_pylist() == [4, 10, 25]
    assert batch.column("name").to_pylist()[0] == "rivet"


@needs["pyarrow"]
def test_a_child_outliving_its_parent_is_still_readable(
    firepanda: ModuleType, tmp_path: Path
) -> None:
    """One column kept, the batch and the frame both dropped.

    A consumer is allowed to take a child out and let the parent go, so every
    child holds its own share of the frame rather than relying on the parent's.
    """
    import pyarrow as pa

    frame = frame_of(firepanda, tmp_path)
    batch = pa.record_batch(frame)
    column = batch.column("qty")
    del frame, batch
    gc.collect()
    assert column.to_pylist() == [4, 10, 25]


def test_a_capsule_nobody_consumes_is_freed_rather_than_leaked(
    firepanda: ModuleType, tmp_path: Path
) -> None:
    """The destructor has to cope with a capsule whose struct is untouched.

    A consumer that takes the data moves the struct out and blanks its release
    callback. A capsule that is simply dropped never does, so the destructor is
    the only thing that will ever release it. There is nothing to assert here
    beyond the process still being alive afterwards, which is what a double free
    would take away.
    """
    for _ in range(64):
        frame = frame_of(firepanda, tmp_path)
        _ = frame.__arrow_c_array__()
        _ = frame.__arrow_c_schema__()
    gc.collect()


def test_a_requested_schema_is_refused_rather_than_ignored(
    firepanda: ModuleType, tmp_path: Path
) -> None:
    """Converting on the way out is not written, so asking for it says so.

    The protocol allows a producer to refuse, and the alternative to refusing is
    handing back the data in a different schema from the one the consumer asked
    for and letting it find out later.
    """
    import pyarrow as pa

    frame = frame_of(firepanda, tmp_path)
    wanted = pa.schema([pa.field("qty", pa.int32())])
    with pytest.raises(NotImplementedError):
        frame.__arrow_c_array__(wanted.__arrow_c_schema__())


@needs["pandas"]
def test_pandas_reads_the_frame(firepanda: ModuleType, tmp_path: Path) -> None:
    """The conversion the compatibility goal is measured through."""
    import pyarrow as pa

    frame = pa.record_batch(frame_of(firepanda, tmp_path)).to_pandas()
    assert list(frame.columns) == ["name", "qty", "price", "ok"]
    assert frame["qty"].tolist() == [4, 10, 25]


@needs["polars"]
def test_polars_reads_the_frame(firepanda: ModuleType, tmp_path: Path) -> None:
    """No firepanda specific code on their side, and none on ours."""
    import polars as pl

    frame = pl.DataFrame(frame_of(firepanda, tmp_path))
    assert frame.columns == ["name", "qty", "price", "ok"]
    assert frame["qty"].to_list() == [4, 10, 25]


@needs["pyarrow"]
def test_an_empty_frame_exports(firepanda: ModuleType, tmp_path: Path) -> None:
    """A header and no rows is a frame with a schema and nothing in it."""
    import pyarrow as pa

    batch = pa.record_batch(frame_of(firepanda, tmp_path, "name,qty\n"))
    assert batch.num_rows == 0
    assert batch.schema.names == ["name", "qty"]
