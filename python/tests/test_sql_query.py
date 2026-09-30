"""`firepanda.sql`, the front door a query string comes in by.

Every expected answer is DuckDB 1.5's to the same statement over the same rows.
"""

from __future__ import annotations

from types import ModuleType

import pytest

ORDERS = {"a": [1, 2, 1, 3, 2], "b": [10, 20, 30, 40, 50]}


def test_a_local_frame_is_a_table(firepanda: ModuleType) -> None:
    """A frame bound to a local variable is read under the variable's name."""
    orders = firepanda.DataFrame(ORDERS)
    got = firepanda.sql("SELECT a, sum(b) AS s FROM orders GROUP BY a ORDER BY a")
    assert list(got.columns) == ["a", "s"]
    assert got["a"].tolist() == [1, 2, 3]
    assert got["s"].tolist() == [40, 70, 40]
    assert len(orders) == 5


def test_a_name_is_folded_as_sql_folds_it(firepanda: ModuleType) -> None:
    """`Orders` and `"ORDERS"` both reach a variable called `orders`."""
    orders = firepanda.DataFrame(ORDERS)
    got = firepanda.sql("SELECT count(*) AS n FROM Orders")
    assert got["n"].tolist() == [5]
    got = firepanda.sql('SELECT max(b) AS m FROM "ORDERS"')
    assert got["m"].tolist() == [50]
    del orders


def test_two_frames_join(firepanda: ModuleType) -> None:
    """Every frame the query names is in scope at once."""
    orders = firepanda.DataFrame(ORDERS)
    names = firepanda.DataFrame({"a": [1, 2], "name": ["one", "two"]})
    got = firepanda.sql(
        "SELECT name, sum(b) AS s FROM orders JOIN names USING (a) GROUP BY name ORDER BY name"
    )
    assert got["name"].tolist() == ["one", "two"]
    assert got["s"].tolist() == [40, 70]
    del orders, names


def test_a_series_is_a_frame_of_its_column(firepanda: ModuleType) -> None:
    """A column captured by name reads as a table with that one column."""
    prices = firepanda.Series([3, 1, 2], name="p")
    got = firepanda.sql("SELECT p FROM prices ORDER BY p")
    assert got["p"].tolist() == [1, 2, 3]
    del prices


def test_a_registration_comes_before_a_variable(firepanda: ModuleType) -> None:
    """A registered frame wins, and a variable that is not a frame hides nothing."""
    firepanda.register("stock", firepanda.DataFrame({"x": [7]}))
    try:
        stock = firepanda.DataFrame({"x": [1, 2]})
        assert firepanda.sql("SELECT x FROM stock")["x"].tolist() == [7]
        del stock
        stock_list = [1, 2, 3]
        assert firepanda.sql("SELECT x FROM Stock")["x"].tolist() == [7]
        del stock_list
    finally:
        firepanda.unregister("STOCK")
    with pytest.raises(ValueError, match="stock"):
        firepanda.sql("SELECT x FROM stock")


def test_a_variable_that_is_not_a_frame_is_passed_over(firepanda: ModuleType) -> None:
    """A list under the name is not a table, so the name is not found."""
    orders = [1, 2, 3]
    with pytest.raises(ValueError, match="orders"):
        firepanda.sql("SELECT * FROM orders")
    del orders


def test_two_variables_one_name_is_ambiguous(firepanda: ModuleType) -> None:
    """Names that differ only in case are one name to SQL, so neither is guessed."""
    t = firepanda.DataFrame({"x": [1]})
    T = firepanda.DataFrame({"x": [2]})
    with pytest.raises(ValueError, match="ambiguous"):
        firepanda.sql("SELECT x FROM t")
    del t, T


def test_capture_can_be_turned_off(firepanda: ModuleType) -> None:
    """With capture off only a registration is in scope."""
    orders = firepanda.DataFrame(ORDERS)
    with pytest.raises(ValueError, match="orders"):
        firepanda.sql("SELECT * FROM orders", capture=False)
    assert firepanda.sql("SELECT 1 + 1 AS two", capture=False)["two"].tolist() == [2]
    del orders


def test_a_statement_changes_nothing_the_next_one_sees(firepanda: ModuleType) -> None:
    """Each call has a catalog of its own, gone when it returns."""
    got = firepanda.sql("CREATE TABLE made(x INTEGER)")
    assert got.shape[1] == 0
    with pytest.raises(ValueError, match="made"):
        firepanda.sql("SELECT * FROM made")


def test_the_mistakes(firepanda: ModuleType) -> None:
    """A wrong statement is a ValueError and a missing feature says so."""
    with pytest.raises(ValueError):
        firepanda.sql("SELEC 1")
    with pytest.raises(NotImplementedError):
        firepanda.sql("SET TimeZone = 'UTC'")
    with pytest.raises(TypeError):
        firepanda.sql(5)


def test_a_frame_runs_sql_over_itself_as_self(firepanda: ModuleType) -> None:
    """`df.sql` reads the frame as `self`, beside the caller's frames."""
    orders = firepanda.DataFrame(ORDERS)
    got = orders.sql("SELECT a, sum(b) AS s FROM self GROUP BY a ORDER BY a")
    assert got["a"].tolist() == [1, 2, 3]
    assert got["s"].tolist() == [40, 70, 40]
    names = firepanda.DataFrame({"a": [1, 3], "name": ["one", "three"]})
    got = orders.sql("SELECT name, b FROM self JOIN names USING (a) ORDER BY b")
    assert got["name"].tolist() == ["one", "one", "three"]
    assert got["b"].tolist() == [10, 30, 40]
    with pytest.raises(ValueError, match="names"):
        orders.sql("SELECT * FROM names", capture=False)
    del names


def test_self_comes_before_a_registration(firepanda: ModuleType) -> None:
    """A registered frame called `self` does not hide the frame `sql` was called on."""
    firepanda.register("self", firepanda.DataFrame({"x": [7]}))
    try:
        frame = firepanda.DataFrame({"x": [1, 2]})
        assert frame.sql("SELECT sum(x) AS s FROM self")["s"].tolist() == [3]
        assert firepanda.sql("SELECT x FROM self")["x"].tolist() == [7]
    finally:
        firepanda.unregister("self")
