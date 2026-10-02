"""A SQL session: one catalog that lasts from one statement to the next.

`firepanda.sql` builds a catalog for each call out of the frames in scope and
throws it away when the call returns, which is right for a library call and
wrong for a shell. A shell is a sequence of statements where `CREATE TABLE` in
one is read by the next, a `SET` holds until a `RESET`, and a `PREPARE` is
there to `EXECUTE`. This is the object that holds all of that between calls.

It holds the dialect too, because the grammar and the function catalog are two
milliseconds to read and a shell runs many statements against the same ones.
"""

from std.os import abort
from std.memory import ArcPointer, Pointer
from std.python import Python, PythonObject
from std.python.bindings import check_arguments_arity

from firepanda.frame import DataFrame
from firepanda.py.args import words
from firepanda.py.errors import UNSUPPORTED, VALUE, retagged, tagged
from firepanda.py.frame import PyDataFrame
from firepanda.sql.cache import PlanCache
from firepanda.sql.catalog import Catalog
from firepanda.sql.run import Dialect


struct PySqlSession(Movable, Writable):
    """A catalog and a dialect, with a CPython object wrapped around them."""

    var catalog: Catalog
    """Every name a statement has made or been handed, and the settings."""

    var dialect: Dialect
    """The grammar, the jump table and the function catalog, read once."""

    var cache: PlanCache
    """The plans of the queries this session has run, by text and generation."""

    var shared: PlanCache
    """The plans of the queries `firepanda.sql` has run through this session,
    by text and the shape of the catalog each call built."""

    def __init__(out self) raises:
        """An empty catalog and the dialect."""
        self.catalog = Catalog()
        self.dialect = Dialect()
        self.cache = PlanCache()
        self.shared = PlanCache()

    @staticmethod
    def py_init(
        out self: Self, args: PythonObject, kwargs: PythonObject
    ) raises:
        """Starts a session with nothing in it.

        Args:
            args: Nothing.
            kwargs: Nothing.
        """
        check_arguments_arity(0, args, "SqlSession")
        self = Self()

    @staticmethod
    def _held(py_self: PythonObject) -> Pointer[Self, MutAnyOrigin]:
        """Recovers the Mojo value out of the Python object holding it.

        Args:
            py_self: The Python object.

        Returns:
            A pointer to the wrapped value.
        """
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("not a firepanda SqlSession: ", e))

    @staticmethod
    def execute(
        py_self: PythonObject, query: PythonObject
    ) raises -> PythonObject:
        """Runs one statement against the session's catalog.

        Args:
            py_self: The session.
            query: The statement.

        Returns:
            What the statement answers, as a new frame.

        Raises:
            Error: Tagged `unsupported` for a statement firepanda refuses by
                name, and `value` for one that is wrong, as `run_sql` tags them.
        """
        var text = words(query, "query")
        var held = Self._held(py_self)
        var answer: DataFrame
        try:
            answer = held[].cache.answer(
                held[].dialect,
                text,
                held[].catalog,
                String(held[].catalog.generation()),
            )
        except cause:
            var message = String(cause)
            if message.startswith("firepanda ") or message.startswith(
                "Not Implemented Error"
            ):
                raise tagged(UNSUPPORTED, message)
            raise tagged(VALUE, message)
        return PythonObject(alloc=PyDataFrame(ArcPointer(answer^)))

    @staticmethod
    def register(
        py_self: PythonObject, name: PythonObject, frame: PythonObject
    ) raises -> PythonObject:
        """Puts a frame under a name, replacing whatever the name held.

        Args:
            py_self: The session.
            name: The name statements will say.
            frame: The frame.

        Returns:
            None.
        """
        var called = words(name, "name")
        var given = PyDataFrame._other(frame, called)
        try:
            Self._held(py_self)[].catalog.register(
                called, DataFrame(copy=given[])
            )
        except cause:
            raise retagged(VALUE, cause)
        return Python.none()

    @staticmethod
    def names(py_self: PythonObject) raises -> PythonObject:
        """Every table and view the session holds, as they were written.

        Args:
            py_self: The session.

        Returns:
            A list of strings.
        """
        var found = Python.list()
        for name in Self._held(py_self)[].catalog.names():
            found.append(PythonObject(name))
        return found

    @staticmethod
    def functions(py_self: PythonObject) raises -> PythonObject:
        """Every function name the dialect knows, aliases included, sorted.

        Args:
            py_self: The session.

        Returns:
            A list of strings, operators among them, since the catalog holds
            `+` by the same rule it holds `abs`.
        """
        var found = Python.list()
        for name in Self._held(py_self)[].dialect.registry.names:
            found.append(PythonObject(name))
        return found

    def write_to(self, mut writer: Some[Writer]):
        """Writes how many names the session holds.

        Args:
            writer: Where to write.
        """
        writer.write("SqlSession(", len(self.catalog.names()), " names)")

    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the same as `write_to`, since a dialect has no repr to derive.

        Args:
            writer: Where to write.
        """
        self.write_to(writer)
