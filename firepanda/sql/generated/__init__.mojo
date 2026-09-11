"""Generated tables: the grammar from tools/gen_grammar.py, and the tier 1
function catalog from tools/gen_functions.py."""

from .functions import (
    DUCKDB_VERSION,
    KIND_AGGREGATE,
    KIND_MACRO,
    KIND_SCALAR,
    NAME_COUNT,
    NO_TYPE,
    OVERLOAD_COUNT,
    TYPE_COUNT,
)
from .functions import TABLE as FUNCTION_TABLE
from .keywords import (
    KEYWORD_CLASS_COUNT,
    KEYWORD_COLUMN_NAME,
    KEYWORD_COUNT,
    KEYWORD_FUNC_NAME,
    KEYWORD_MAX_LENGTH,
    KEYWORD_RESERVED,
    KEYWORD_TYPE_NAME,
    KEYWORD_UNRESERVED,
    KEYWORDS,
)
from .rules import (
    FILTER_BITS,
    FILTER_COUNT,
    MATCHER_COUNT,
    MEMOIZED_COUNT,
    OVERRIDDEN_COUNT,
    NODE_COUNT,
    RULE_COUNT,
    RULE_END_OF_INPUT,
    RULE_PROGRAM,
    RULE_WHITESPACE,
    STRING_COUNT,
    SUGGESTION_COUNT,
    TABLE,
)
