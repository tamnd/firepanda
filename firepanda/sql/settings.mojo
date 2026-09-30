"""What `SET`, `RESET` and `PRAGMA` can change, and what each change does.

A DuckDB setting falls in one of three places here.

**It changes how a query runs and not what it answers.** `threads`,
`memory_limit`, the profiler, the progress bar, the checkpoint knobs and the
rest of storage. firepanda has no storage and chooses its own parallelism, so
there is nothing for these to reach, and the answer they leave is the answer
DuckDB gives under any value of them. They are accepted and remembered, so a
script that sets one runs as it does in DuckDB, and `RESET` puts one back.

**It changes what a query answers.** `default_order` and `default_null_order`
are the two held here. Each decides what an `ORDER BY` that did not say means,
and each is applied to the parse before it is lowered: an entry with no
direction or no null placement written gets the one the setting names, and the
lowering never learns there was a setting. A window's `ORDER BY` and an
ordered aggregate's are the same entries and follow it too, as they do in
DuckDB.

**It changes which rewrites the optimizer makes.** `disabled_optimizers`
names passes to leave out, and `PRAGMA disable_optimizer` leaves out all of
them. Every pass answers the same thing with it or without it, so these change
the plan and never the answer, but unlike the first group they reach something
here, which is `firepanda/plan/optimize.mojo`.

**It takes a capability away.** `enable_external_access` set false turns off
every way a statement could reach the file system, and it is one way, as it is
in DuckDB: once off it cannot be turned back on or reset, and setting it true
is refused even while it is still on, so a query cannot give itself back what
the program took from it. firepanda's SQL reads no file yet, so for now the
setting is held and locked and there is nothing further for it to refuse.

**Anything else** is refused by name. A setting that changes answers and is not
held, such as `TimeZone`, would give a wrong answer if it were accepted and
dropped, and one DuckDB does not have is refused rather than guessed at.
"""

from firepanda.plan.optimize import passes

from .ast import (
    NULLS_DEFAULT,
    NULLS_FIRST,
    NULLS_LAST,
    SORT_DEFAULT,
    SORT_DESCENDING,
    STMT_ORDER,
    Ast,
)


comptime NULLS_ALWAYS_FIRST: UInt8 = 0
"""`nulls_first`: a null sorts before every value, whichever way the sort
goes."""

comptime NULLS_ALWAYS_LAST: UInt8 = 1
"""`nulls_last`, DuckDB's default: a null sorts after every value."""

comptime NULLS_SMALLEST: UInt8 = 2
"""`nulls_first_on_asc_last_on_desc`, SQLite's and MySQL's rule, where a null
is smaller than every value."""

comptime NULLS_LARGEST: UInt8 = 3
"""`nulls_last_on_asc_first_on_desc`, Postgres's rule, where a null is larger
than every value."""


def _inert() -> List[StaticString]:
    """The settings that change how a query runs and never what it answers,
    folded, since DuckDB reads a setting name without regard to case."""
    return [
        "allocator_background_threads",
        "allocator_flush_threshold",
        "checkpoint_threshold",
        "custom_profiling_settings",
        "debug_checkpoint_abort",
        "debug_force_external",
        "debug_force_no_cross_product",
        "debug_window_mode",
        "disable_checkpoint_on_shutdown",
        "disable_print_progress_bar",
        "disable_profile",
        "disable_profiling",
        "disable_progress_bar",
        "disable_verification",
        "disable_verify_external",
        "disable_verify_fetch_row",
        "disable_verify_parallelism",
        "disable_verify_serializer",
        "enable_checkpoint_on_shutdown",
        "enable_fsst_vectors",
        "enable_http_metadata_cache",
        "enable_object_cache",
        "enable_print_progress_bar",
        "enable_profile",
        "enable_profiling",
        "enable_progress_bar",
        "enable_progress_bar_print",
        "enable_verification",
        "explain_output",
        "external_threads",
        "force_checkpoint",
        "force_compression",
        "force_parallelism",
        "immediate_transaction_mode",
        "index_scan_max_count",
        "index_scan_percentage",
        "max_memory",
        "max_temp_directory_size",
        "max_vacuum_tasks",
        "memory_limit",
        "merge_join_threshold",
        "nested_loop_join_threshold",
        "partitioned_write_flush_threshold",
        "partitioned_write_max_open_files",
        "perfect_ht_threshold",
        "prefer_range_joins",
        "preserve_insertion_order",
        "profile_output",
        "profiling_mode",
        "profiling_output",
        "progress_bar_time",
        "streaming_buffer_size",
        "temp_directory",
        "threads",
        "tracked_metrics",
        "verify_external",
        "verify_fetch_row",
        "verify_parallelism",
        "verify_serializer",
        "wal_autocheckpoint",
        "worker_threads",
    ]


def _fold(text: StringSlice) -> String:
    """`text` in lower case, which is how a setting name is compared."""
    return String(text).lower()


def _unquoted(text: StringSlice) -> String:
    """A setting's value as the words it says: a string literal's contents, or
    an identifier as written, since DuckDB reads `SET x = nulls_first` and
    `SET x = 'nulls_first'` the same."""
    var bytes = text.as_bytes()
    var n = len(bytes)
    if n >= 2 and bytes[0] == Byte(ord("'")) and bytes[n - 1] == Byte(ord("'")):
        return String(StringSlice(unsafe_from_utf8=bytes[1 : n - 1])).replace(
            "''", "'"
        )
    return String(text)


def _boolean(value: StringSlice) raises -> Bool:
    """A setting's value read as DuckDB casts it to `BOOLEAN`: a number is
    true when it is not zero, and a word is one of the spellings DuckDB reads
    without regard to case."""
    var bytes = value.as_bytes()
    if len(bytes) > 0 and bytes[0] != Byte(ord("'")):
        try:
            return Int(String(value)) != 0
        except:
            pass
    var said = _unquoted(value)
    var folded = _fold(said)
    var yes: List[String] = ["true", "t", "y", "yes", "1"]
    var no: List[String] = ["false", "f", "n", "no", "0"]
    if folded in yes:
        return True
    if folded in no:
        return False
    raise Error(
        String(
            "Invalid Input Error: Failed to cast value: Could not convert"
            " string '",
            said,
            "' to BOOL",
        )
    )


comptime _NO_WAY_BACK = (
    "Invalid Input Error: Cannot enable external access while database is"
    " running"
)
"""DuckDB's answer to any `SET` that would turn external access on and to any
`RESET` of it."""


def _optimizers() -> List[StaticString]:
    """Every name DuckDB takes in `disabled_optimizers`, and the one firepanda
    adds for the pass DuckDB has no name for.

    Turning off one that is not a firepanda pass turns nothing off, since there
    is nothing by that name to skip.
    """
    return [
        "expression_rewriter",
        "filter_pullup",
        "filter_pushdown",
        "empty_result_pullup",
        "cte_filter_pusher",
        "regex_range",
        "in_clause",
        "join_order",
        "deliminator",
        "unnest_rewriter",
        "unused_columns",
        "statistics_propagation",
        "common_subexpressions",
        "common_aggregate",
        "column_lifetime",
        "limit_pushdown",
        "row_group_pruner",
        "top_n",
        "top_n_window_elimination",
        "build_side_probe_side",
        "compressed_materialization",
        "duplicate_groups",
        "reorder_filter",
        "sampling_pushdown",
        "join_filter_pushdown",
        "extension",
        "materialized_cte",
        "sum_rewriter",
        "late_materialization",
        "cte_inlining",
        "common_subplan",
        "join_elimination",
        "window_self_join",
        "projection_merge",
    ]


def _disabled(value: StringSlice) raises -> List[String]:
    """The pass names a `disabled_optimizers` value lists, comma separated.

    Raises:
        If one of them is not a name, with DuckDB's message.
    """
    var out = List[String]()
    for piece in _unquoted(value).split(","):
        var name = _fold(piece.strip())
        if name == "":
            continue
        var known = False
        for entry in _optimizers():
            if name == entry:
                known = True
                break
        if not known:
            raise Error(
                String(
                    'Parser Error: Optimizer type "', name, '" not recognized'
                )
            )
        out.append(name^)
    return out^


def is_inert(name: StringSlice) -> Bool:
    """Whether a setting changes how a query runs and never what it answers.

    Args:
        name: The setting, as written.

    Returns:
        True for a setting that is accepted and has nothing to act on here.
    """
    var folded = _fold(name)
    for entry in _inert():
        if folded == entry:
            return True
    return False


struct Settings(Copyable, Movable):
    """The settings a session has changed from DuckDB's defaults."""

    var descending: Bool
    """`default_order`: whether an `ORDER BY` entry with no direction written
    sorts descending."""

    var nulls: UInt8
    """`default_null_order`, one of the `NULLS_` rules above."""

    var disabled: List[String]
    """`disabled_optimizers`: the passes to leave out, folded."""

    var optimizer: Bool
    """False after `PRAGMA disable_optimizer`, which leaves out every pass."""

    var external: Bool
    """`enable_external_access`: whether a statement may reach the file
    system. Only ever turned off."""

    var names: List[String]
    """Every setting a statement has set and not reset, folded."""

    var values: List[String]
    """The value each of `names` was given, as it was written."""

    def __init__(out self):
        """DuckDB's defaults: ascending, and nulls last either way."""
        self.descending = False
        self.nulls = NULLS_ALWAYS_LAST
        self.disabled = List[String]()
        self.optimizer = True
        self.external = True
        self.names = List[String]()
        self.values = List[String]()

    def is_default(self) -> Bool:
        """Whether nothing held here would change an answer."""
        return not self.descending and self.nulls == NULLS_ALWAYS_LAST

    def set(mut self, name: StringSlice, value: StringSlice) raises:
        """Changes one setting, as `SET name = value` does.

        Args:
            name: The setting, as written.
            value: The value, as written: a literal or a bare word.

        Raises:
            If the value is not one the setting takes, with DuckDB's message,
            or if the setting is not one firepanda holds or can accept.
        """
        var folded = _fold(name)
        if folded == "default_order":
            var said = _fold(_unquoted(value))
            if said == "asc" or said == "ascending":
                self.descending = False
            elif said == "desc" or said == "descending":
                self.descending = True
            else:
                raise Error(
                    String(
                        (
                            "Invalid Input Error: Unrecognized parameter for"
                            " option"
                        ),
                        ' DEFAULT_ORDER "',
                        _unquoted(value),
                        '". Expected ASC or DESC.',
                    )
                )
        elif folded == "default_null_order" or folded == "null_order":
            var said = _fold(_unquoted(value)).replace("_", " ")
            if said == "nulls first":
                self.nulls = NULLS_ALWAYS_FIRST
            elif said == "nulls last":
                self.nulls = NULLS_ALWAYS_LAST
            elif (
                said == "sqlite"
                or said == "mysql"
                or said == "nulls first on asc last on desc"
            ):
                self.nulls = NULLS_SMALLEST
            elif (
                said == "postgres" or said == "nulls last on asc first on desc"
            ):
                self.nulls = NULLS_LARGEST
            else:
                raise Error(
                    String(
                        "Parser Error: Unrecognized parameter for option",
                        ' NULL_ORDER "',
                        _unquoted(value),
                        '", expected either NULLS FIRST, NULLS LAST, SQLite,',
                        " MySQL or Postgres",
                    )
                )
        elif folded == "disabled_optimizers":
            self.disabled = _disabled(value)
        elif folded == "enable_external_access":
            if _boolean(value):
                raise Error(_NO_WAY_BACK)
            self.external = False
        elif not is_inert(folded):
            raise Error(
                String(
                    "firepanda does not support the setting ",
                    name,
                    (
                        ", because it changes what a query answers and nothing"
                        " here acts on it yet"
                    ),
                )
            )
        self._remember(folded, value)

    def reset(mut self, name: StringSlice) raises:
        """Puts one setting back to DuckDB's default, as `RESET name` does.

        Args:
            name: The setting, as written.

        Raises:
            If the setting is not one `set` would take.
        """
        var folded = _fold(name)
        if folded == "default_order":
            self.descending = False
        elif folded == "default_null_order" or folded == "null_order":
            self.nulls = NULLS_ALWAYS_LAST
        elif folded == "disabled_optimizers":
            self.disabled = List[String]()
        elif folded == "enable_external_access":
            raise Error(_NO_WAY_BACK)
        elif not is_inert(folded):
            raise Error(
                String(
                    "firepanda does not support the setting ",
                    name,
                    (
                        ", because it changes what a query answers and nothing"
                        " here acts on it yet"
                    ),
                )
            )
        for i in range(len(self.names)):
            if self.names[i] == folded:
                _ = self.names.pop(i)
                _ = self.values.pop(i)
                return

    def pragma(mut self, name: StringSlice) -> Bool:
        """Runs a `PRAGMA` that takes no arguments, which switches something
        on or off by its name.

        Args:
            name: The pragma, as written.

        Returns:
            Whether it is one this knows. One it does not is left for the
            caller to refuse.
        """
        var folded = _fold(name)
        if folded == "disable_optimizer":
            self.optimizer = False
            return True
        if folded == "enable_optimizer":
            self.optimizer = True
            return True
        return is_inert(folded)

    def disabled_passes(self) -> List[String]:
        """The optimizer passes to leave out, every one of them after `PRAGMA
        disable_optimizer`."""
        if not self.optimizer:
            var out = List[String]()
            for name in passes():
                out.append(String(name))
            return out^
        return self.disabled.copy()

    def _remember(mut self, var folded: String, value: StringSlice):
        for i in range(len(self.names)):
            if self.names[i] == folded:
                self.values[i] = String(value)
                return
        self.names.append(folded^)
        self.values.append(String(value))

    def settle(self, mut ast: Ast):
        """Writes the direction and the null placement the settings name into
        every `ORDER BY` entry of a parse that did not write its own.

        After this an entry with no null placement written means nulls last,
        which is what the lowering reads it as, so the lowering needs no
        settings of its own.

        Args:
            ast: The parse, changed in place.
        """
        if self.is_default():
            return
        for i in range(len(ast.stmts)):
            if ast.stmts[i].kind != STMT_ORDER:
                continue
            if ast.stmts[i].b == SORT_DEFAULT and self.descending:
                ast.stmts[i].b = SORT_DESCENDING
            if ast.stmts[i].payload != NULLS_DEFAULT:
                continue
            var down = ast.stmts[i].b == SORT_DESCENDING
            var first: Bool
            if self.nulls == NULLS_ALWAYS_FIRST:
                first = True
            elif self.nulls == NULLS_SMALLEST:
                first = not down
            elif self.nulls == NULLS_LARGEST:
                first = down
            else:
                first = False
            ast.stmts[i].payload = NULLS_FIRST if first else NULLS_LAST
